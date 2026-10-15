import zlib, os
P = "uniref50.fasta.gz"
size = os.path.getsize(P)
d = zlib.decompressobj(16 + zlib.MAX_WBITS)
pos = 0
with open(P, "rb") as f:
    while True:
        c = f.read(1 << 20)
        if not c:
            print(f"CLEAN | size={size}"); break
        try:
            d.decompress(c)
        except zlib.error:
            print(f"BAD | first_bad~{pos} bytes ({pos/2**30:.2f} GiB) | size={size}"); break
        pos += len(c)
