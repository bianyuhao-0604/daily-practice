"""从 struct_npz + fasta 重算 surf / iface / pLDDT（修正 F1 残基编码 bug）。

背景
----
早期 features/extract_surf.py::parse_cif 把 Bio.PDB 的三字母残基名（如 "ALA"）
直接传给 descriptors()，而后者用一字母集合判断残基类别（HYDRO/ARO/POS/NEG），
三字母永不命中 → grp 恒为 2 → 80 维中 19–27 / 28–32 / 63–64 共 16 维恒为常数
（20% 特征死亡）。实测 2026-10-23 生成的 10,840 个 surf_*.npy 均带此指纹。

本脚本改走 struct_npz（含 CA 坐标 / pLDDT / label_seq_id）+ sequences_primary.fasta，
按 seqid 取一字母残基名，因此
  * 不需要 CIF、不需要 biopython、不需要网络；
  * 覆盖 struct_npz 的全部 10,915 个蛋白（比现有 10,840 个 surf 散文件更全）。

输出
----
默认写到独立目录 workdir/features/surf_fix/（不覆盖旧文件，便于 A/B 对照）：
  surf_{pid}.npy   (L,80) float16
  iface_{pid}.npy  (L,)   float16
  plddt_{pid}.npy  (L,)   float16

用法
----
  python prep/surf_from_npz.py --workers 14
  python prep/surf_from_npz.py --workers 14 --force      # 忽略已有产物，全部重算
  python prep/surf_from_npz.py --out-dir workdir/features   # 直接写回主目录（覆盖旧文件）
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LEN, Paths                              # noqa: E402
from features.extract_surf import descriptors, iface_heuristic  # noqa: E402

_FASTA_CACHE = {}


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                tok = line[1:].split()
                name, buf = (tok[0] if tok else None), []
            elif line:
                buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def get_fasta(path):
    """每个 worker 进程内只读一次 fasta（约 5 MB）。"""
    if path not in _FASTA_CACHE:
        _FASTA_CACHE[path] = read_fasta(path)
    return _FASTA_CACHE[path]


def work(pid, npz_dir, fasta_path, out_dir):
    """单个蛋白：npz → 描述子 → 三个 .npy。返回 (pid, L)。"""
    z = np.load(os.path.join(npz_dir, pid + ".npz"))
    sid, plddt, ca = z["seq"], z["plddt"], z["ca"]
    L = int(min(len(sid), len(plddt), len(ca), MAX_LEN))
    if L <= 0:
        raise ValueError("empty structure")
    sid, plddt, ca = sid[:L], plddt[:L].astype(np.float32), ca[:L].astype(np.float32)
    if not np.isfinite(ca).all():
        raise ValueError("non-finite CA coordinates")

    fa = get_fasta(fasta_path).get(pid, "")
    res = []
    for s in sid:
        i = int(s) - 1
        res.append(fa[i] if 0 <= i < len(fa) else "X")

    D = descriptors(ca, res, plddt)
    iface = iface_heuristic(ca, D, plddt)

    np.save(os.path.join(out_dir, f"surf_{pid}.npy"), D.astype(np.float16))
    np.save(os.path.join(out_dir, f"iface_{pid}.npy"), np.asarray(iface, np.float32).astype(np.float16))
    np.save(os.path.join(out_dir, f"plddt_{pid}.npy"), plddt.astype(np.float16))
    return pid, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default=None, help="默认 <feat-dir>/pids.txt")
    ap.add_argument("--npz-dir", default=None, help="默认 <feat-dir>/struct_npz")
    ap.add_argument("--fasta", default="./workdir/data/sequences_primary.fasta")
    ap.add_argument("--out-dir", default=None, help="默认 <feat-dir>/surf_fix")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--force", action="store_true", help="忽略已有产物，全部重算")
    a = ap.parse_args()

    feat = Paths().feat_dir
    pids_csv = a.pids or os.path.join(feat, "pids.txt")
    npz_dir = a.npz_dir or os.path.join(feat, "struct_npz")
    out_dir = a.out_dir or os.path.join(feat, "surf_fix")
    os.makedirs(out_dir, exist_ok=True)

    pids = [l.strip() for l in open(pids_csv, encoding="utf-8") if l.strip()]
    have_npz = [p for p in pids if os.path.exists(os.path.join(npz_dir, p + ".npz"))]
    no_npz = [p for p in pids if p not in set(have_npz)]

    if a.force:
        todo = have_npz
    else:
        todo = [p for p in have_npz
                if not all(os.path.exists(os.path.join(out_dir, f"{k}_{p}.npy"))
                           for k in ("surf", "iface", "plddt"))]

    print(f"[surf_from_npz] pids={len(pids)} with_npz={len(have_npz)} no_npz={len(no_npz)} "
          f"todo={len(todo)} workers={a.workers}", flush=True)
    print(f"[surf_from_npz] out_dir={os.path.abspath(out_dir)}", flush=True)
    if no_npz:
        print(f"[surf_from_npz] 无 struct_npz 的前10个: {no_npz[:10]}", flush=True)

    t0 = time.time()
    done = fail = 0
    fails = []
    if todo:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(work, p, npz_dir, a.fasta, out_dir): p for p in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                pid = futs[fut]
                try:
                    fut.result()
                    done += 1
                except Exception as e:                        # N5：单条失败记账，不中断
                    fail += 1
                    fails.append(f"{pid}\t{type(e).__name__}: {e}")
                if i % 250 == 0 or i == len(todo):
                    el = time.time() - t0
                    rate = i / max(el, 1e-9)
                    eta = (len(todo) - i) / max(rate, 1e-9)
                    print(f"[surf_from_npz] {i}/{len(todo)} ok={done} fail={fail} "
                          f"{rate:.1f}/s elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

    with open(os.path.join(out_dir, "_surf_fix_failed.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(fails))
    print(f"[surf_from_npz] DONE ok={done} fail={fail} skipped={len(have_npz) - len(todo)} "
          f"total={time.time() - t0:.0f}s", flush=True)
    print("[surf_from_npz] 失败清单 ->", os.path.join(out_dir, "_surf_fix_failed.txt"), flush=True)


if __name__ == "__main__":
    main()
