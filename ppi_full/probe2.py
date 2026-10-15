# 全文件逐 member 校验（修复 v1 只验第一段的盲区）
import zlib, os, sys
P = "uniref50.fasta.gz"
SLICE = 1 << 16                       # 64KB 喂入粒度 -> 坏点定位 ±64KB
size, members, total_out = os.path.getsize(P), 0, 0
member_start = member_fed = 0
d, pending, fpos = None, b"", 0
def fail(kind, lo, hi, msg):
    print(f"BAD | {kind} | 已过 member={members} | 坏点区间 [{lo},{hi}) = {lo/2**30:.2f} GiB | {msg}")
    sys.exit(1)
with open(P, "rb") as f:
    while True:
        chunk = f.read(1 << 20)
        if not chunk: break
        fpos += len(chunk); pending += chunk
        while pending:
            if d is None:
                if len(pending) < 10: break
                if pending[:2] != b"\x1f\x8b":
                    fail("FRAMING", fpos-len(pending), fpos-len(pending), "非gzip魔数: "+pending[:8].hex())
                member_start, member_fed = fpos-len(pending), 0
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            i = 0
            while i < len(pending) and not d.eof:
                s = pending[i:i+SLICE]
                try:
                    total_out += len(d.decompress(s))
                except zlib.error as e:
                    fail("STREAM", member_start+member_fed, member_start+member_fed+len(s),
                         f"member {members+1}(起点{member_start}): {e}")
                member_fed += len(s); i += len(s)
            if d.eof:
                members += 1            # 该段含 CRC/长度校验全部通过
                pending = d.unused_data + pending[i:]; d = None
            else:
                pending = b""
        if fpos % (1 << 30) < (1 << 20):
            print(f"  ... {fpos/2**30:.1f} GiB, members={members}", flush=True)
if d is not None: fail("TRUNCATED", member_start, member_start, "文件结束而member未闭合")
if pending:
    fail("TRUNCATED" if pending[:2]==b"\x1f\x8b" else "TAIL", fpos-len(pending), fpos, f"尾{len(pending)}B异常")
print(f"CLEAN | members={members} | 解压总量 {total_out/2**30:.1f} GiB | 文件 {size} B")
