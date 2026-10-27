"""不落盘校验 zip：流式重算每个条目的 SHA-256，与包内 MANIFEST.sha256 比对。

等价于“解包后跑 verify_bundle.py”，但不需要先写 50 GB 到磁盘。
另外检查：Zip64 条目可读、Unix 权限位、CRLF、包内清单自洽。

用法：
    python _bundle/verify_zip.py [zip路径]
"""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\pythonprojects\practice-github\ppi_full\_bundle\ppi_full_bundle_ubuntu_20261027.zip")

EXPECT_EXEC = {"ppi_full/setup_ubuntu.sh", "ppi_full/RUN_ALL_EXPERIMENTS.sh", "ppi_full/mmseqs/bin/mmseqs"}


def main():
    t0 = time.time()
    zf = zipfile.ZipFile(ZIP)
    infos = zf.infolist()
    names = [i.filename for i in infos]
    print(f"zip: {ZIP.name}  {ZIP.stat().st_size / 2**30:.2f} GiB  条目 {len(infos)}")

    dup = {n for n in names if names.count(n) > 1}
    assert not dup, f"重名条目: {dup}"

    man_raw = zf.read("ppi_full/MANIFEST.sha256").decode("utf-8")
    rows = [ln.split("  ", 2) for ln in man_raw.splitlines() if ln.strip()]
    print(f"MANIFEST 行数 {len(rows)}")
    expect = {r[2]: (r[0], int(r[1])) for r in rows}
    print(f"manifest.json: {json.loads(zf.read('ppi_full/manifest.json'))['total_files']} 个文件")

    bad, missing, total = [], [], 0
    for i, attr in enumerate(infos, 1):
        if attr.filename in ("ppi_full/MANIFEST.sha256", "ppi_full/manifest.json"):
            continue
        rel = attr.filename[len("ppi_full/"):]
        if rel not in expect:
            bad.append((rel, "不在 MANIFEST 里"))
            continue
        want_sha, want_size = expect[rel]
        h = hashlib.sha256()
        got = 0
        with zf.open(attr) as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                got += len(b)
                h.update(b)
        if got != want_size:
            bad.append((rel, f"大小 {got} != {want_size}"))
        elif h.hexdigest() != want_sha:
            bad.append((rel, "sha256 不匹配"))
        else:
            total += got
        if i % 20 == 0 or got > (1 << 30):
            print(f"  [{i:3d}/{len(infos)}] {rel:<40} {got/2**20:9.1f} MiB  "
                  f"{total/2**30:6.2f} GiB 累计  {time.time()-t0:5.0f}s", flush=True)

    missing = sorted(set(expect) - {i.filename[len("ppi_full/"):] for i in infos})
    bad_names = {r for r, _ in bad}
    ok_count = len([r for r in expect if r not in bad_names and r not in set(missing)])
    print(f"\nOK={ok_count}  BAD={len(bad)}  MISSING={len(missing)}  用时 {time.time()-t0:.0f}s")
    for r, why in bad:
        print(f"  BAD {r}: {why}")
    for r in missing:
        print(f"  MISSING {r}")

    # ---- 权限位 / 换行 / 编码 ----
    problems = []
    for attr in infos:
        mode = (attr.external_attr >> 16) & 0o777
        if attr.create_system != 3:
            problems.append(f"{attr.filename}: create_system={attr.create_system}（应为 3=Unix）")
        want = 0o755 if attr.filename in EXPECT_EXEC else 0o644
        if mode != want:
            problems.append(f"{attr.filename}: mode={oct(mode)} 期望 {oct(want)}")
        if not attr.filename.isascii():
            problems.append(f"{attr.filename}: 文件名非 ASCII")
    for name in ("ppi_full/setup_ubuntu.sh", "ppi_full/RUN_ALL_EXPERIMENTS.sh"):
        data = zf.read(name)
        if b"\r\n" in data:
            problems.append(f"{name}: 含 CRLF")
        if not data.startswith(b"#!/usr/bin/env bash"):
            problems.append(f"{name}: shebang 不对")

    print(f"\n权限/换行/编码检查：{'全部通过' if not problems else f'{len(problems)} 个问题'}")
    for p in problems:
        print("  " + p)

    ok = not bad and not missing and not problems
    print("\n结论：" + ("通过 ✅" if ok else "有问题 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
