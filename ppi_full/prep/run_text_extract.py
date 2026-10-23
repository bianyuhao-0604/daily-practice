"""text 通道（128d）：UniProt 文本 → 句向量 → PCA(128)。

P8：PCA **只在 train 蛋白上 fit**，避免 transductive 泄漏。文本抽取复用
`features/extract_text_genome.py:uniprot_text()`，与论文口径一致（CC 段 + GO + KW，截断 6000 字符）。

用法：
    python prep/run_text_extract.py                     # 全量 11,020 个蛋白
    python prep/run_text_extract.py --limit 50          # 冒烟
    python prep/run_text_extract.py --bs 64 --device cuda
编码器：$FeatureModels.st_local = sentence-transformers/all-mpnet-base-v2（768d）。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import FeatureModels  # noqa: E402

FEAT = os.path.join(ROOT, "workdir", "features")
UNIPROT = os.path.join(ROOT, "workdir", "uniprot")
META = os.path.join(FEAT, "meta.json")
MIN_CHARS = 20          # 少于这个字符数视为无文本 → 不写文件（has_text=0）


def all_pids():
    if os.path.exists(META):
        return list(json.load(open(META, encoding="utf-8"))["pid2idx"].keys())
    out = []
    for sp in ("train", "val", "test"):
        p = os.path.join(ROOT, "workdir", "data", f"{sp}.csv")
        with open(p, encoding="utf-8") as f:
            next(f)
            for line in f:
                a, b, _ = line.split(",")
                out += [a.strip(), b.strip()]
    return sorted(set(out))


def train_pids():
    p = os.path.join(ROOT, "workdir", "data", "train.csv")
    s = set()
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, _ = line.split(",")
            s.add(a.strip())
            s.add(b.strip())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feat-dir", default=FEAT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    feat = a.feat_dir
    os.makedirs(feat, exist_ok=True)
    pids = all_pids()
    tr = train_pids()
    if a.limit:  # 冒烟时优先取 train 蛋白，保证 PCA 可 fit
        pids = sorted(p for p in pids if p in tr)[:a.limit]

    # 1) 文本
    from features.extract_text_genome import uniprot_text
    texts, skipped = {}, []
    t0 = time.time()
    for i, p in enumerate(pids, 1):
        try:
            t = uniprot_text(p, UNIPROT)
        except Exception:
            t = ""
        if len(t) < MIN_CHARS:
            skipped.append(p)
        else:
            texts[p] = t
        if i % 2000 == 0:
            print(f"[text] 读取 {i}/{len(pids)} 空={len(skipped)} {time.time()-t0:.0f}s", flush=True)
    print(f"[text] 可编码 {len(texts)} / 跳过空文本 {len(skipped)} ({time.time()-t0:.0f}s)", flush=True)

    # 2) 句向量
    import torch
    from sentence_transformers import SentenceTransformer
    fm = FeatureModels()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[enc] model={fm.st_local} device={dev} bs={a.bs} maxlen={a.maxlen}", flush=True)
    model = SentenceTransformer(fm.st_local, device=dev)
    model.max_seq_length = a.maxlen
    items = list(texts.items())
    emb = {}
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(items), a.bs):
            batch = [t for _, t in items[k:k + a.bs]]
            v = model.encode(batch, convert_to_numpy=True, show_progress_bar=False,
                             batch_size=len(batch))
            for (p, _), e in zip(items[k:k + a.bs], v):
                emb[p] = e.astype(np.float32)
            if (k // a.bs) % 20 == 0 or k + a.bs >= len(items):
                done = min(k + a.bs, len(items))
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(items) - done) / max(rate, 1e-9)
                print(f"[enc] {done}/{len(items)} {rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    # 3) P8：PCA 只在 train 上 fit
    from sklearn.decomposition import PCA
    import joblib
    fit = [p for p in texts if p in tr]
    assert fit, "train.csv 与可编码蛋白无交集"
    mat = np.stack([emb[p] for p in fit])
    n_comp = min(128, mat.shape[0], mat.shape[1])
    if n_comp < 128:
        print(f"[pca] WARN n_components 降级为 {n_comp}（样本不足 128，仅冒烟可用）", flush=True)
    pca = PCA(n_components=n_comp, random_state=0).fit(mat)
    joblib.dump(pca, os.path.join(feat, "text_pca.joblib"))
    print(f"[pca] fit on {len(fit)} train pids dim={n_comp} "
          f"evr={pca.explained_variance_ratio_.sum():.3f}", flush=True)

    # 4) 落盘 text_{pid}.npy (128,) float16
    n_new = n_skip = 0
    for p, e in emb.items():
        dst = os.path.join(feat, f"text_{p}.npy")
        if os.path.exists(dst) and not a.force:
            n_skip += 1
            continue
        np.save(dst, pca.transform(e[None])[0].astype(np.float16))
        n_new += 1
    print(f"[save] written={n_new} existed={n_skip} skipped_empty={len(skipped)}", flush=True)
    print("[done] 接下来: python prep/collect_channels.py --channels text", flush=True)


if __name__ == "__main__":
    main()
