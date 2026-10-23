"""按通道增量入库：只打开目标通道，绝不触碰其它通道（防止把 text/genome 清零）。

背景
----
features/store.py 的 StoreWriter 用 mode="w+" 一次性重开全部通道，而
text_*.npy / genome_*.npy 散文件并不存在（只有库里的 text.npy / genome.npy）。
因此直接跑 `store.py --collect` 会把这两个通道写成全零、has_* 变 False —— 静默毁数据。

本脚本为每个通道单独 open_memmap，只重写 --channels 指定的通道，
并原样保留 meta.json 的 pid2idx / has_text / has_genome。

安全阀
------
1. 若某通道在 --src-dir 下找不到任何散文件，直接拒绝执行（否则等于清零该通道）。
2. 若散文件覆盖率 < 50%，打印醒目警告并要求 --allow-low-coverage。
3. 不写 lengths.npy / plddt.npy（那两个已经是对的）。
4. 每通道写完立即回读抽样校验（非零、无 NaN/Inf、padding 为 0）。

用法
----
  python prep/collect_channels.py --channels surf iface --src-dir workdir/features/surf_fix
  python prep/collect_channels.py --channels str --src-dir workdir/features
  python prep/collect_channels.py --channels surf iface --dry-run
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LEN, Paths                       # noqa: E402
from features.store import RES_LAYOUT, PROT_LAYOUT      # noqa: E402

DTYPE = np.float16


def channel_shape(n, ch):
    """返回 (shape, is_protein_level)。未知通道返回 (None, False)。"""
    if ch in PROT_LAYOUT:
        return (n, *PROT_LAYOUT[ch]), True
    if ch in RES_LAYOUT:
        tail = RES_LAYOUT[ch]
        return ((n, MAX_LEN, *tail) if tail else (n, MAX_LEN)), False
    return None, False


def load_primaries(feat_dir):
    """与 build_store_meta.py 一致的行顺序：features_node.tsv 的 primary 列。"""
    p = os.path.join(feat_dir, "features_node.tsv")
    if not os.path.exists(p):
        raise SystemExit(f"缺少 {p}，无法确定行顺序")
    rows = [l.split("\t")[0] for l in open(p, encoding="utf-8").readlines()[1:]]
    return [r.strip() for r in rows if r.strip()]


def count_src(src_dir, ch):
    return sum(1 for f in os.listdir(src_dir) if f.startswith(ch + "_") and f.endswith(".npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=["surf", "iface"],
                    help="要入库的通道，如 surf iface 或 str")
    ap.add_argument("--src-dir", default=None, help="散文件所在目录，默认 = feat-dir")
    ap.add_argument("--feat-dir", default=None, help="默认 Paths().feat_dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-low-coverage", action="store_true")
    a = ap.parse_args()

    feat = a.feat_dir or Paths().feat_dir
    src = a.src_dir or feat
    prim = load_primaries(feat)
    n = len(prim)

    meta = json.load(open(os.path.join(feat, "meta.json"), encoding="utf-8"))
    if int(meta["n"]) != n:
        raise SystemExit(f"meta.n={meta['n']} 与 features_node.tsv 行数 {n} 不一致，先修 meta")
    bad = [p for p in prim if p not in meta["pid2idx"]]
    if bad:
        raise SystemExit(f"{len(bad)} 个 primary 不在 meta.pid2idx 中，e.g. {bad[:3]}")

    lengths = np.load(os.path.join(feat, "lengths.npy"))

    print(f"[collect] feat_dir={os.path.abspath(feat)}  src_dir={os.path.abspath(src)}  n={n}")
    for ch in a.channels:
        shape, is_prot = channel_shape(n, ch)
        if shape is None:
            raise SystemExit(f"未知通道 '{ch}'，已知: {list(RES_LAYOUT) + list(PROT_LAYOUT)}")
        ns = count_src(src, ch)
        print(f"[collect] channel={ch:<6} shape={shape} src_files={ns}", flush=True)
        if ns == 0:
            raise SystemExit(f"通道 '{ch}' 在 {src} 下没有任何散文件 —— 拒绝执行（否则会把该通道清零）")
        if ns < 0.5 * n and not a.allow_low_coverage:
            raise SystemExit(f"通道 '{ch}' 覆盖率仅 {ns}/{n} (<50%)；确认无误请加 --allow-low-coverage")
        if a.dry_run:
            continue

        arr = np.lib.format.open_memmap(os.path.join(feat, ch + ".npy"),
                                        mode="w+", dtype=DTYPE, shape=shape)
        present = np.zeros(n, bool)
        miss, mismatch = 0, 0
        for i, pid in enumerate(prim):
            f = os.path.join(src, f"{ch}_{pid}.npy")
            if not os.path.exists(f):
                miss += 1
                continue
            v = np.load(f)
            if is_prot:
                d = PROT_LAYOUT[ch][0]
                if v.shape != (d,):
                    mismatch += 1
                    continue
                arr[i] = v.astype(DTYPE)
                present[i] = bool(np.abs(v).max() > 1e-6)
            else:
                L = int(min(int(lengths[i]), MAX_LEN))
                c = int(min(len(v), L))
                if c <= 0:
                    miss += 1
                    continue
                if v.ndim == 1:
                    arr[i, :c] = v[:c].astype(DTYPE)
                else:
                    arr[i, :c, :] = v[:c, :].astype(DTYPE)
                present[i] = bool(np.abs(np.asarray(v[:c])).max() > 1e-6)
        arr.flush()
        del arr

        # 回读抽样校验
        back = np.load(os.path.join(feat, ch + ".npy"), mmap_mode="r")
        idx = np.linspace(0, n - 1, num=min(256, n)).astype(int)
        sub = np.asarray(back[idx], np.float32)
        nan = int(np.isnan(sub).sum())
        inf = int(np.isinf(sub).sum())
        mx = float(np.abs(sub).max())
        print(f"[collect]   wrote {ch}: present={int(present.sum())}/{n} missing={miss} "
              f"shape_mismatch={mismatch} | 抽样 NaN={nan} Inf={inf} |max|={mx:.4f}", flush=True)
        if nan or inf:
            raise SystemExit(f"通道 {ch} 写入后含 NaN/Inf，已中止")
        if mx <= 1e-6:
            raise SystemExit(f"通道 {ch} 写入后全零，已中止")

        meta[f"has_{ch}"] = present.tolist()
        meta.setdefault("channels_present", [])
        if ch not in meta["channels_present"]:
            meta["channels_present"].append(ch)

    if not a.dry_run:
        json.dump(meta, open(os.path.join(feat, "meta.json"), "w", encoding="utf-8"))
        print("[collect] meta.json updated (pid2idx/has_text/has_genome 保持不变)", flush=True)
    print("[collect] DONE", flush=True)


if __name__ == "__main__":
    main()
