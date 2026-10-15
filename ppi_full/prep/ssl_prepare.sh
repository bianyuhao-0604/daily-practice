#!/usr/bin/env bash
# SSL 语料制备（32GB RAM 友好路线）：等步长采样 → 30% 聚类去冗余 → 对 Bernett <30% 过滤。
# 前置：conda install -c bioconda mmseqs2；workdir 空闲 ≥30GB；OOM 则 STRIDE 8→16 重跑。
set -e
STRIDE=8
python - <<'PY'
import gzip
SRC, DST, STRIDE = "uniref50.fasta.gz", "ssl_sub8m.fasta", 8
n = kept = 0
out = open(DST, "w")
name, buf = None, []
def flush():
    global kept
    if name is not None and kept % STRIDE == 0:
        out.write(name + "\n" + "\n".join("".join(buf)[i:i+80] for i in range(0, len("".join(buf)), 80)) + "\n")
    kept += 1
with gzip.open(SRC, "rt") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            flush(); name, buf = line, []
        else:
            buf.append(line)
flush(); out.close()
print("kept ~", (kept + STRIDE - 1) // STRIDE)
PY
mmseqs easy-linclust ssl_sub8m.fasta ssl_dedup tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0 --threads 16
mmseqs easy-search ssl_dedup_rep_seq.fasta workdir/data/sequences.fasta hits.m8 tmp2 --max-seqs 1 --threads 16
awk -F'\t' '$3 < 0.3 {print $1}' hits.m8 > keep_ids.txt
python - <<'PY'
keep = set(l.strip() for l in open("keep_ids.txt") if l.strip())
out = open("workdir/ssl/uniref50_lt30.fasta", "w")
name, buf, n = None, [], 0
for line in open("ssl_dedup_rep_seq.fasta"):
    line = line.strip()
    if line.startswith(">"):
        if name is not None and name.split()[0][1:] in keep:
            out.write(name + "\n" + "\n".join("".join(buf)[i:i+80] for i in range(0, len("".join(buf)), 80)) + "\n"); n += 1
        name, buf = line, []
    else:
        buf.append(line)
if name is not None and name.split()[0][1:] in keep:
    out.write(name + "\n" + "\n".join("".join(buf)[i:i+80] for i in range(0, len("".join(buf)), 80)) + "\n"); n += 1
out.close(); print("final SSL seqs:", n)
PY
#rm -rf tmp tmp2 ssl_sub8m.fasta ssl_dedup* hits.m8 keep_ids.txt   # 清理 15–30GB 中间文件
