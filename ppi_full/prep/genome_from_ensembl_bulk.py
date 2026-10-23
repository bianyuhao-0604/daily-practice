"""从 Ensembl 全量 FTP 直接切出每个蛋白的基因组窗口（不走 REST，稳定且快）。

流程：
  1. 下载 `Homo_sapiens.GRCh38.112.gtf.gz`（56 MB）与
     `Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz`（882 MB）到 `workdir/genome/raw/`
     （支持断点续传：已有 .part / 完整文件会跳过）。
  2. 解析 GTF 的 `transcript` 行 → `ENST → (chr, start, end, strand)`。
  3. 对每个 pid：从 `workdir/uniprot/{pid}.txt` 取第一个 ENST，按链方向切出**转录本 5' 端
     4,096 bp**（minus 链取反向互补），写成 `workdir/genome/{pid}.seq`（纯 ACGT）。
  4. FASTA 按染色体流式读取，单条染色体载入内存后切片即释放（峰值 ~250 MB）。

用法：
    python prep/genome_from_ensembl_bulk.py                # 下载 + 切窗口
    python prep/genome_from_ensembl_bulk.py --skip-download
"""
import argparse
import gzip
import io
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW = os.path.join(ROOT, "workdir", "genome", "raw")
OUT = os.path.join(ROOT, "workdir", "genome")
UNIPROT = os.path.join(ROOT, "workdir", "uniprot")
META = os.path.join(ROOT, "workdir", "features", "meta.json")
REL = "112"
FASTA_URL = (f"https://ftp.ensembl.org/pub/release-{REL}/fasta/homo_sapiens/dna/"
             f"Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz")
GTF_URL = (f"https://ftp.ensembl.org/pub/release-{REL}/gtf/homo_sapiens/"
           f"Homo_sapiens.GRCh38.{REL}.gtf.gz")
WIN = 4096
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def download(url, dst, tries=3):
    name = os.path.basename(dst)
    if os.path.exists(dst) and os.path.getsize(dst) > 1 << 20:
        print(f"[dl] 已存在 {name} {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
        return dst
    part = dst + ".part"
    for t in range(tries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={
                "User-Agent": "ppi_full/2.3", **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                done = have
                t0 = time.time()
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    f.write(blk)
                    done += len(blk)
                    el = time.time() - t0
                    if el > 5:
                        rate = (done - have) / el / 1e6
                        pct = f"{100*done/total:.0f}%" if total else "?"
                        print(f"[dl] {name} {done/1e6:.0f} MB {pct} {rate:.1f} MB/s", flush=True)
                        t0 = time.time()
            os.replace(part, dst)
            print(f"[dl] {name} done {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
            return dst
        except Exception as e:
            print(f"[dl] {name} 第 {t+1} 次失败 {type(e).__name__} {str(e)[:60]}", flush=True)
            time.sleep(5 * (t + 1))
    raise SystemExit(f"下载失败: {url}")


def parse_gtf(path, wanted_enst):
    """只留需要的转录本坐标。返回 {enst: (chr, start, end, strand)}"""
    out = {}
    pat = re.compile(r'transcript_id "([^"]+)"')
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.rstrip("\n").split("\t")
            if len(t) < 9 or t[2] != "transcript":
                continue
            m = pat.search(t[8])
            if not m or m.group(1) not in wanted_enst:
                continue
            out[m.group(1)] = (t[0], int(t[3]), int(t[4]), t[6])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--local-fasta", default="", help="用本地已下载的 .fa.gz 代替流式读取")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    # pid → ENST
    pids = list(__import__("json").load(open(META, encoding="utf-8"))["pid2idx"].keys())
    if a.limit:
        pids = pids[:a.limit]
    pid_enst = {}
    for p in pids:
        f = os.path.join(UNIPROT, p + ".txt")
        if not os.path.exists(f):
            continue
        m = re.search(r"^DR   Ensembl; (ENST\d+)", open(f, encoding="utf-8", errors="ignore").read(), re.M)
        if m:
            pid_enst[p] = m.group(1)
    print(f"[gtf] pid={len(pids)} 有 Ensembl 转录本={len(pid_enst)}", flush=True)

    gtf = os.path.join(RAW, os.path.basename(GTF_URL))
    fa = os.path.join(RAW, os.path.basename(FASTA_URL))
    if not a.skip_download:
        download(GTF_URL, gtf)          # 只落盘 GTF（56 MB）；FASTA 884 MB 走流式，不落盘
    if not os.path.exists(gtf):
        raise SystemExit("缺 GTF，去掉 --skip-download 再跑")
    if a.local_fasta:
        fa = a.local_fasta
        if not os.path.exists(fa):
            raise SystemExit(f"--local-fasta 不存在: {fa}")
        print(f"[fa] 使用本地 FASTA {fa}", flush=True)

    coords = parse_gtf(gtf, set(pid_enst.values()))
    print(f"[gtf] 命中转录本坐标 {len(coords)}/{len(set(pid_enst.values()))}", flush=True)

    # 组装每条染色体上要切的窗口
    need = {}   # chr -> list[(pid, start0, end0, strand)]  start0/end0 为 0-based 半开
    for p, e in pid_enst.items():
        c = coords.get(e)
        if not c:
            continue
        chrom, s, en, strand = c
        if strand == "+":
            a0, b0 = s - 1, s - 1 + WIN
        else:
            a0, b0 = max(0, en - WIN), en
        need.setdefault(chrom, []).append((p, a0, b0, strand))
    print(f"[cut] 待切 {sum(len(v) for v in need.values())} 条，跨 {len(need)} 条染色体", flush=True)

    cur_chr, buf, done, t0 = None, bytearray(), 0, time.time()

    def flush_chrom():
        nonlocal done
        if cur_chr is None or cur_chr not in need:
            return
        seqs = bytes(buf)
        for pid, a0, b0, strand in need[cur_chr]:
            seg = seqs[max(0, a0):min(len(seqs), b0)].decode("ascii", "ignore").upper()
            if strand == "-":
                seg = seg.translate(COMP)[::-1]
            if len(seg) < 64:
                continue
            with open(os.path.join(OUT, pid + ".seq"), "w", encoding="ascii") as f:
                f.write(seg)
            done += 1

    # 直接流式读取远端 FASTA（不落盘）：单条染色体载入后切片即释放
    # 若给了 --local-fasta，则改为读本地 gz 文件
    MAXTRY = 3
    for attempt in range(MAXTRY):
        try:
            if os.path.exists(fa):
                print(f"[fa] 读本地 {fa}", flush=True)
                fh = gzip.open(fa, "rt", encoding="ascii", errors="ignore")
            else:
                print("[fa] 流式读取远端 FASTA（882 MB gz，1 MB/s 量级 → 10+ 分钟）", flush=True)
                req = urllib.request.Request(FASTA_URL, headers={"User-Agent": "ppi_full/2.3"})
                r = urllib.request.urlopen(req, timeout=90)
                fh = io.TextIOWrapper(gzip.GzipFile(fileobj=r), encoding="ascii",
                                      errors="ignore", newline="")
            with fh:
                for line in fh:
                    if line.startswith(">"):
                        flush_chrom()
                        cur_chr, buf = line[1:].split()[0], bytearray()
                        if done:
                            print(f"[cut] {done} 条 elapsed={time.time()-t0:.0f}s "
                                  f"chr={cur_chr}", flush=True)
                        continue
                    buf += line.strip().encode("ascii", "ignore")
            flush_chrom()
            break
        except Exception as e:
            print(f"[fa] 第 {attempt+1} 次流式读取中断: {type(e).__name__} {str(e)[:60]}", flush=True)
            if attempt == MAXTRY - 1:
                raise
    print(f"[cut] DONE 写出 {done} 个 .seq  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
