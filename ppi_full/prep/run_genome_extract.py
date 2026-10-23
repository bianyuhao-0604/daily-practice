"""genome 通道（256d）：gLM2 基因组上下文嵌入 → PCA(256)。

原料：`workdir/genome/{pid}.seq`（由 `prep/fetch_genome_windows.py` 从 Ensembl REST 下载的
转录本基因组序列，含内含子、已按 5'→3' 定向）。

编码：`tattabio/gLM2_150M`（公开、非 gated）逐碱基 token 化，取转录本 5' 端 2,048 bp
与基因体中部 2,048 bp 两段，mean-pool 后取平均 → 640d → **仅在 train 蛋白上 fit** 的 PCA(256)。

为什么不是 350M/650M：`tattabio/gLM2_650M` 同样公开，但 4060 上长窗口吞吐不够（2048 bp
时约 4× 慢），150M 与论文用的 gLM2 同族、维度经 PCA 对齐到 256，R7 的口径不变。
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
GENDIR = os.path.join(ROOT, "workdir", "genome")
META = os.path.join(FEAT, "meta.json")


def train_pids():
    s = set()
    with open(os.path.join(ROOT, "workdir", "data", "train.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, _ = line.split(",")
            s.add(a.strip())
            s.add(b.strip())
    return s


def pick_chunks(seq, win=2048, n=2):
    """从 5' 端**连续切块**（.seq 已按转录方向定向）。"""
    if len(seq) <= win:
        return [seq]
    out = []
    for i in range(min(n, (len(seq) + win - 1) // win)):
        c = seq[i * win:(i + 1) * win]
        if len(c) >= 64:
            out.append(c)
    return out or [seq[:win]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tattabio/gLM2_150M")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feat-dir", default=FEAT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    feat = a.feat_dir
    os.makedirs(feat, exist_ok=True)
    pids = list(json.load(open(META, encoding="utf-8"))["pid2idx"].keys())
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(GENDIR, "*.seq"))}
    pids = [p for p in pids if p in have]
    tr = train_pids()
    if a.limit:
        pids = sorted(p for p in pids if p in tr)[:a.limit]
    print(f"[genome] seq 可用 {len(pids)} 条  window={a.window} chunks={a.chunks}",
          flush=True)

    # 1) 组装所有窗口
    jobs = []           # (pid, chunk_idx, seq)
    for p in pids:
        s = open(os.path.join(GENDIR, p + ".seq"), encoding="ascii", errors="ignore").read()
        for k, c in enumerate(pick_chunks(s, a.window, a.chunks)):
            jobs.append((p, k, c))
    print(f"[genome] 窗口数 {len(jobs)}", flush=True)

    # 2) gLM2 编码
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[enc] model={a.model} device={dev} bs={a.bs}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(dev).eval()

    # 按长度排序减少 padding，再按 pid 聚合
    jobs.sort(key=lambda x: len(x[2]))
    acc = {p: [] for p in pids}
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(jobs), a.bs):
            batch = jobs[k:k + a.bs]
            enc = tok([c for _, _, c in batch], return_tensors="pt",
                      padding=True, truncation=False)
            enc = {kk: v.to(dev) for kk, v in enc.items() if torch.is_tensor(v)}
            out = model(**enc)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = pooled.float().cpu().numpy()
            for (pid, _ci, _s), e in zip(batch, pooled):
                acc[pid].append(e)
            done = min(k + a.bs, len(jobs))
            if (k // a.bs) % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"[enc] {done}/{len(jobs)} {rate:.1f} chunk/s "
                      f"eta={(len(jobs)-done)/max(rate,1e-9)/60:.1f}min", flush=True)

    emb = {p: np.mean(np.stack(v), 0) for p, v in acc.items() if v}
    print(f"[enc] 完成 {len(emb)} 条蛋白  dim={next(iter(emb.values())).shape[0]}", flush=True)

    # 3) P8：PCA 只在 train 上 fit
    from sklearn.decomposition import PCA
    import joblib
    fit = [p for p in emb if p in tr]
    mat = np.stack([emb[p] for p in fit])
    n_comp = min(256, mat.shape[0], mat.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(mat)
    joblib.dump(pca, os.path.join(feat, "genome_pca.joblib"))
    print(f"[pca] fit on {len(fit)} train pids dim={n_comp} "
          f"evr={pca.explained_variance_ratio_.sum():.3f}", flush=True)

    # 4) 落盘
    n_new = n_skip = 0
    for p, e in emb.items():
        dst = os.path.join(feat, f"genome_{p}.npy")
        if os.path.exists(dst) and not a.force:
            n_skip += 1
            continue
        v = pca.transform(e[None])[0].astype(np.float16)
        if n_comp < 256:
            v = np.pad(v, (0, 256 - n_comp))
        np.save(dst, v)
        n_new += 1
    print(f"[save] written={n_new} existed={n_skip} 接下来: "
          f"python prep/collect_channels.py --channels genome", flush=True)


if __name__ == "__main__":
    main()
