"""str 通道（ProstT5 逐残基 1024d）提取 runner：断点续跑 + OOM 自动降批 + 本地缓存离线加载。

为什么单独写一个 runner
----------------------
features/extract_seq_str.py 的 extract_prostt5 是一次性整批跑完的，缺少：
  * 断点续跑（中断后从头再来）
  * 显存不足时的自动降批
  * 已有散文件的跳过与校验
本脚本补齐这三点，并复用同一套编码/对齐逻辑（ProstT5 的 <aa2fold> 前缀）。

前置
----
  pip install transformers sentencepiece safetensors      （+ CUDA 版 torch）
  权重已在 HF 缓存：Rostlab/ProstT5（config.FeatureModels.prostt5 已修正）
  8 GB 显存：默认 --bs 1；12 GB+ 可试 2

用法
----
  python prep/run_str_extract.py --bs 1 --workers 1
  python prep/run_str_extract.py --limit 20 --bs 1        # 先试 20 条
  python prep/run_str_extract.py --force                  # 忽略已有文件

产物
----
  workdir/features/str_{pid}.npy   (L,1024) float16
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LEN, FeatureModels, Paths   # noqa: E402


def read_fasta(path, want):
    seqs, name, buf = {}, None, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name in want:
                    seqs[name] = "".join(buf)
                tok = line[1:].split()
                name, buf = (tok[0] if tok else None), []
            elif line:
                buf.append(line)
    if name in want:
        seqs[name] = "".join(buf)
    return seqs


def is_done(path, expect_len):
    if not os.path.exists(path) or os.path.getsize(path) < 256:
        return False
    try:
        z = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return z.ndim == 2 and z.shape[1] == 1024 and z.shape[0] == expect_len and z.dtype == np.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default=None, help="默认 <feat-dir>/pids.txt")
    ap.add_argument("--fasta", default="./workdir/data/sequences_primary.fasta")
    ap.add_argument("--feat-dir", default=None)
    ap.add_argument("--bs", type=int, default=1, help="8GB 显存用 1；12GB+ 可试 2")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑用）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    feat = a.feat_dir or Paths().feat_dir
    os.makedirs(feat, exist_ok=True)
    pids_csv = a.pids or os.path.join(feat, "pids.txt")
    fm = FeatureModels()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        print("[str] WARNING: 没有可用 CUDA —— ProstT5 约 2.7B 参数，CPU 上不可行。", flush=True)

    from transformers import T5EncoderModel, T5Tokenizer

    print(f"[str] model={fm.prostt5} device={dev} bs={a.bs}", flush=True)
    tok = T5Tokenizer.from_pretrained(fm.prostt5, legacy=True)
    enc = T5EncoderModel.from_pretrained(fm.prostt5).eval().to(dev)
    if dev == "cuda":
        enc = enc.bfloat16()      # fp16 在 T5 上易溢出；Ada 架构原生支持 bf16
    off = len(tok("<aa2fold>", add_special_tokens=False)["input_ids"])

    pids = [l.strip() for l in open(pids_csv, encoding="utf-8") if l.strip()]
    if a.limit:
        pids = pids[:a.limit]
    seqs = read_fasta(a.fasta, set(pids))
    missing = [p for p in pids if p not in seqs]
    if missing:
        print(f"[str] fasta 缺 {len(missing)} 条，e.g. {missing[:5]}", flush=True)

    todo = []
    for p in pids:
        if p not in seqs:
            continue
        L = min(len(seqs[p]), MAX_LEN)
        out = os.path.join(feat, f"str_{p}.npy")
        if not a.force and is_done(out, L):
            continue
        todo.append((p, seqs[p][:MAX_LEN], L))
    print(f"[str] pids={len(pids)} todo={len(todo)}", flush=True)

    t0 = time.time()
    done = fail = 0
    fails = []
    bs = a.bs
    i = 0
    while i < len(todo):
        chunk = todo[i:i + bs]
        try:
            enc_in = tok([f"<aa2fold> {' '.join(s)}" for _, s, _ in chunk],
                         return_tensors="pt", padding=True)
            with torch.no_grad():
                h = enc(input_ids=enc_in["input_ids"].to(dev),
                        attention_mask=enc_in["attention_mask"].to(dev)).last_hidden_state.float()
            for j, (p, _s, L) in enumerate(chunk):
                total = int(enc_in["attention_mask"][j].sum())
                if total != off + L + 1:
                    raise AssertionError(f"对齐失败 {p}: {total} != {off + L + 1}")
                emb = h[j, off:off + L].cpu().numpy()
                np.save(os.path.join(feat, f"str_{p}.npy"), emb.astype(np.float16))
            done += len(chunk)
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = 1
                print(f"[str] OOM -> 降批到 bs=1", flush=True)
                continue
            fail += 1
            fails.append(f"{chunk[0][0]}\tOOM at bs=1")
            i += 1
        except Exception as e:
            fail += 1
            fails.append(f"{chunk[0][0]}\t{type(e).__name__}: {e}")
            i += 1
        if done and done % 200 < bs:
            el = time.time() - t0
            print(f"[str] {i}/{len(todo)} ok={done} fail={fail} "
                  f"{done/max(el,1e-9):.2f}/s elapsed={el/60:.1f}m "
                  f"eta={(len(todo)-i)/max(done/max(el,1e-9),1e-9)/60:.1f}m", flush=True)

    try:
        with open(os.path.join(feat, "_str_failed.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(fails))
    except OSError:
        pass
    for line in fails[:10]:
        print(f"[str] FAIL {line}", flush=True)
    print(f"[str] DONE ok={done} fail={fail} total={time.time()-t0:.0f}s", flush=True)
    print(f"[str] 失败清单 -> {os.path.join(feat, '_str_failed.txt')}", flush=True)


if __name__ == "__main__":
    main()
