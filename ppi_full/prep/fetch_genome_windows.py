"""抓取每个蛋白的「基因组上下文」原始 DNA（Ensembl REST，基因体 + 内含子）。

`genome` 通道需要的是基因座位的基因组序列（论文口径：含邻基因上下文的 interleave 窗口）。
本机全盘没有任何基因组文件，但 UniProt 注释里带 Ensembl 转录本 ID（`DR   Ensembl; ENST…`），
可直接用 Ensembl REST 拿到该转录本覆盖的基因组序列（minus 链会被反向互补，5'→3'）。

输出：`workdir/genome/{pid}.seq`（纯 ACGT 字符串）+ `_enst_map.json`
用法：
    python prep/fetch_genome_windows.py --limit 5
    python prep/fetch_genome_windows.py --workers 8
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

UNIPROT = os.path.join(ROOT, "workdir", "uniprot")
OUT = os.path.join(ROOT, "workdir", "genome")
META = os.path.join(ROOT, "workdir", "features", "meta.json")
API = "https://rest.ensembl.org/sequence/id/{id}?type=genomic;content-type=text/plain"
HDRS = {"Content-Type": "text/plain",
        "User-Agent": "ppi_full/2.3 (genome context fetch)"}


def find_transcript(pid):
    """从本地 UniProt txt 里抠第一个 Ensembl 转录本 ID（没有则返回 None）。"""
    p = os.path.join(UNIPROT, pid + ".txt")
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r"^DR   Ensembl; (ENST\d+)", txt, re.M)
    return m.group(1) if m else None


def fetch_seq(enst, tries=3, timeout=90):
    for t in range(tries):
        try:
            req = urllib.request.Request(API.format(id=enst), headers=HDRS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                s = r.read().decode("ascii", "ignore").strip().upper()
            s = "".join(c for c in s if c in "ACGTN")
            if len(s) < 200:
                return None, f"too_short:{len(s)}"
            return s, ""
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (t + 1))
                continue
            if e.code in (400, 404):
                return None, f"HTTP{e.code}"
            time.sleep(2 * (t + 1))
        except Exception as e:
            time.sleep(2 * (t + 1))
            last = f"{type(e).__name__}"
    return None, locals().get("last", "retries_exhausted")


def one(pid):
    dst = os.path.join(OUT, pid + ".seq")
    if os.path.exists(dst) and os.path.getsize(dst) > 200:
        return pid, "skip", os.path.getsize(dst), ""
    enst = find_transcript(pid)
    if not enst:
        return pid, "no_ensembl", 0, ""
    s, err = fetch_seq(enst)
    if s is None:
        return pid, "fail", 0, err
    tmp = dst + ".part"
    with open(tmp, "w", encoding="ascii") as f:
        f.write(s)
    os.replace(tmp, dst)
    return pid, "ok", len(s), enst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pids = list(json.load(open(META, encoding="utf-8"))["pid2idx"].keys())
    if a.limit:
        pids = pids[:a.limit]
    if a.force:
        for p in pids:
            f = os.path.join(OUT, p + ".seq")
            if os.path.exists(f):
                os.remove(f)
    print(f"[genome] pids={len(pids)} workers={a.workers} out={OUT}", flush=True)

    t0 = time.time()
    stat = {"ok": 0, "skip": 0, "fail": 0, "no_ensembl": 0}
    fails, enst_map = [], {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (pid, st, n, info) in enumerate(ex.map(one, pids), 1):
            stat[st] = stat.get(st, 0) + 1
            if st == "ok" and isinstance(info, str) and info.startswith("ENST"):
                enst_map[pid] = info
            if st == "fail":
                fails.append(f"{pid}\t{info}")
            if i % 200 == 0 or i == len(pids):
                el = time.time() - t0
                print(f"[genome] {i}/{len(pids)} ok={stat['ok']} skip={stat['skip']} "
                      f"fail={stat['fail']} no_ensembl={stat['no_ensembl']} "
                      f"{i/max(el,1e-9):.1f}/s {el:.0f}s", flush=True)

    with open(os.path.join(OUT, "_genome_failed.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(fails))
    old = {}
    mp = os.path.join(OUT, "_enst_map.json")
    if os.path.exists(mp):
        try:
            old = json.load(open(mp, encoding="utf-8"))
        except Exception:
            old = {}
    old.update(enst_map)
    json.dump(old, open(mp, "w", encoding="utf-8"), indent=0)
    print(f"[genome] DONE {stat} total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
