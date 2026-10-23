"""补齐缺失的 struct_npz：对没有结构的蛋白按 v6→v5→v4 下载 AFDB CIF → 解析 → 存 npz。

背景：workdir/afdb 里 10,915 个蛋白有 CIF，剩余 ~105 个连 CIF 都没有。AFDB 的
"{pid}.cif"（本地命名）与 "AF-{pid}-F1-model_v4.cif"（官方命名）混用，且 v4 已于
2025 年下线（实测 404），所以这里统一试 v6→v5→v4。

用法：
    python prep/fetch_missing_struct.py                # 自动找 meta 里有、npz 里没有的
    python prep/fetch_missing_struct.py --limit 5      # 先试 5 条
    python prep/fetch_missing_struct.py --workers 8
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CIF_DIR = os.path.join(ROOT, "workdir", "afdb")
NPZ_DIR = os.path.join(ROOT, "workdir", "features", "struct_npz")
META = os.path.join(ROOT, "workdir", "features", "meta.json")
URLS = ("https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v6.cif",
        "https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v5.cif",
        "https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v4.cif")
UA = {"User-Agent": "ppi_full/2.3 (+struct repair)"}


def _load_parse_cif():
    """复用 prep/parse_structures.py 的解析器，避免两份实现漂移。"""
    spec = importlib.util.spec_from_file_location(
        "ps", os.path.join(ROOT, "prep", "parse_structures.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_cif


def missing_pids():
    meta = json.load(open(META, encoding="utf-8"))
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(NPZ_DIR, "*.npz"))}
    return [p for p in meta["pid2idx"] if p not in have]


def fetch_one(pid):
    """返回 (pid, version|None, err)。已存在的 CIF 直接复用。"""
    dst = os.path.join(CIF_DIR, pid + ".cif")
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return pid, "cached", ""
    for url in URLS:
        v = url.rsplit("_", 1)[1][:-4]
        try:
            req = urllib.request.Request(url.format(pid=pid), headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 1024:
                continue
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return pid, v, ""
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return pid, None, f"HTTP {e.code}"
        except Exception as e:
            return pid, None, f"{type(e).__name__}: {str(e)[:60]}"
    return pid, None, "404 (v6/v5/v4 均无)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(CIF_DIR, exist_ok=True)
    os.makedirs(NPZ_DIR, exist_ok=True)
    pids = missing_pids()
    if a.limit:
        pids = pids[:a.limit]
    print(f"[fetch] missing={len(pids)} workers={a.workers}", flush=True)

    t0 = time.time()
    got, failed = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (pid, v, err) in enumerate(ex.map(fetch_one, pids), 1):
            if v:
                got.append((pid, v))
            else:
                failed.append((pid, err))
            if i % 20 == 0 or i == len(pids):
                print(f"[fetch] {i}/{len(pids)} ok={len(got)} fail={len(failed)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # 下载成功的立刻解析成 npz（与 parse_structures.py 完全同格式）
    parse_cif = _load_parse_cif()
    n_ok = n_empty = 0
    for pid, _v in got:
        src = os.path.join(CIF_DIR, pid + ".cif")
        dst = os.path.join(NPZ_DIR, pid + ".npz")
        if os.path.exists(dst) and not a.force:
            n_ok += 1
            continue
        try:
            seqid, pl, xyz = parse_cif(src)
            if not pl:
                n_empty += 1
                failed.append((pid, "cif 无 CA 原子"))
                continue
            import numpy as np
            np.savez_compressed(dst, seq=np.asarray(seqid, dtype="int32"),
                                plddt=np.asarray(pl, dtype="float32"),
                                ca=np.asarray(xyz, dtype="float32"))
            n_ok += 1
        except Exception as e:
            failed.append((pid, f"parse: {type(e).__name__}: {str(e)[:50]}"))

    print(f"[fetch] DONE cif_ok={len(got)} npz_ok={n_ok} empty={n_empty} "
          f"still_failed={len(failed)} total={time.time()-t0:.0f}s", flush=True)
    if failed:
        out = os.path.join(ROOT, "workdir", "features", "_struct_failed.txt")
        try:
            with open(out, "w", encoding="utf-8") as f:
                for pid, err in failed:
                    f.write(f"{pid}\t{err}\n")
            print(f"[fetch] 失败清单 -> {out}", flush=True)
        except Exception as e:
            print(f"[fetch] 写失败清单失败: {e}", flush=True)
        for pid, err in failed[:10]:
            print(f"  FAIL {pid}: {err}", flush=True)


if __name__ == "__main__":
    main()
