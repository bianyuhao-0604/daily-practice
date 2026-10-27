"""按 MANIFEST.sha256 校验包内文件完整性（跨平台）。

用法：
    python verify_bundle.py              # 校验全部
    python verify_bundle.py --quick      # 只校验字节数（快，不读 50 GB）
    python verify_bundle.py --only code  # 只校验路径含 "code" 的条目
"""
import argparse
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "MANIFEST.sha256")


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            digest, size, rel = ln.split("  ", 2)
            rows.append((digest, int(size), rel))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只比对字节数")
    ap.add_argument("--only", default=None, help="只校验路径包含该子串的条目")
    a = ap.parse_args()

    if not os.path.exists(MANIFEST):
        sys.exit(f"找不到清单：{MANIFEST}")

    rows = load_manifest(MANIFEST)
    if a.only:
        rows = [r for r in rows if a.only in r[2]]
    print(f"清单 {len(rows)} 个文件，模式={'仅大小' if a.quick else 'SHA-256'}")

    bad, missing, done = [], [], 0
    t0 = time.time()
    for digest, size, rel in rows:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            missing.append(rel)
            print(f"MISSING  {rel}")
            continue
        got = os.path.getsize(p)
        if got != size:
            bad.append((rel, f"大小 {got} != {size}"))
            print(f"BAD      {rel}  (大小 {got} != {size})")
            continue
        if not a.quick:
            h = sha256_file(p)
            if h != digest:
                bad.append((rel, "sha256 不匹配"))
                print(f"BAD      {rel}  (sha256)")
                continue
        done += 1
        print(f"OK       {rel}  ({size / 2**20:.1f} MiB)")

    dt = time.time() - t0
    print("-" * 60)
    print(f"OK={done}  BAD={len(bad)}  MISSING={len(missing)}  用时 {dt:.1f}s")
    if bad or missing:
        sys.exit(1)
    print(f"全部 {done} 个文件校验通过。")


if __name__ == "__main__":
    main()
