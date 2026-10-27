"""构建 Ubuntu 版可分发实验包（单文件 ZIP，ZIP_STORED，Zip64）。

与 make_bundle.py 的唯一差别：
  1. 舞台目录为 _bundle_linux/，随包文件换成 Linux 脚本与文档；
  2. 显式写入 Unix 权限位（create_system=3），保证 .sh 与 mmseqs/bin/mmseqs 解包后即可执行；
  3. 输出 ppi_full_bundle_ubuntu_<date>.zip。

只读取原文件，不修改/不移动任何原始文件；构建前后对源文件做 (size, mtime) 快照对比。

用法：
    python _bundle/make_bundle_linux.py
"""
import hashlib
import json
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(r"D:\pythonprojects\practice-github\ppi_full")
STAGE = ROOT / "_bundle_linux"
OUTDIR = ROOT / "_bundle"
ZIP_NAME = "ppi_full_bundle_ubuntu_20261027.zip"
PREFIX = "ppi_full/"

ROOT_FILES = [
    "config.py", "data.py", "model.py", "engine.py", "stats.py", "run_all.py",
    "check_data_all.py", "smoke_all.py", "build_str_channels.py", "download_afdb_cif.py",
    "ssl_pretrain.py", "patch_pretrain.py", "probe2.py", "probe_gz.py", "probe_konwledge_lr.py",
    "requirements.txt", "ARCHITECTURE.md", ".gitignore",
    "pids.txt", "train_pids.txt", "missing_struct_103.txt", "missing_genome_186.txt",
    "data_check_report.md", "data_check_report.json", "smoke_report.md",
]
DATA_FILES = [
    "workdir/data/train.csv", "workdir/data/val.csv", "workdir/data/test.csv",
    "workdir/data/sequences.fasta", "workdir/data/sequences_primary.fasta",
    "workdir/data/truncation_report.json",
    "workdir/features/seq.npy", "workdir/features/str.npy", "workdir/features/surf.npy",
    "workdir/features/iface.npy", "workdir/features/text.npy", "workdir/features/genome.npy",
    "workdir/features/lengths.npy", "workdir/features/plddt.npy", "workdir/features/meta.json",
    "workdir/features/uniprot_primary_map.txt", "workdir/features/pids.txt",
    "workdir/features/_struct_failed.txt", "workdir/features/_str_failed.txt",
    "workdir/features/_salvaged_from_unisave.txt", "workdir/features/_uniprot_dead_accessions.txt",
    "workdir/features/features_annot.tsv", "workdir/features/features_node.tsv",
    "workdir/features/features_struct.tsv",
    "workdir/ssl/adapter.pt", "workdir/ssl/adapter.pt.meta.json",
]
STAGE_FILES = [
    "README_FIRST_LINUX.md", "USAGE_LINUX.md",
    "setup_ubuntu.sh", "RUN_ALL_EXPERIMENTS.sh", "verify_bundle.py",
]
DIR_TREES = ["features", "prep", "mmseqs"]

EXEC_PATHS = {"setup_ubuntu.sh", "RUN_ALL_EXPERIMENTS.sh", "mmseqs/bin/mmseqs"}


def walk_tree(rel_dir, skip=("__pycache__",)):
    base = ROOT / rel_dir
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if any(part in skip for part in p.relative_to(ROOT).parts):
            continue
        out.append(p.relative_to(ROOT).as_posix())
    return out


def entry_list():
    entries, missing = [], []
    for r in ROOT_FILES + DATA_FILES:
        p = ROOT / r
        (entries.append((p, PREFIX + r)) if p.is_file() else missing.append(r))
    for d in DIR_TREES:
        for r in walk_tree(d):
            entries.append((ROOT / r, PREFIX + r))
    for r in STAGE_FILES:
        p = STAGE / r
        (entries.append((p, PREFIX + r)) if p.is_file() else missing.append(f"_bundle_linux/{r}"))
    return entries, missing


def snapshot(entries):
    return {str(src): (src.stat().st_size, int(src.stat().st_mtime)) for src, _ in entries}


def npy_header(path):
    import numpy as np
    a = np.load(path, mmap_mode="r")
    return {"shape": list(a.shape), "dtype": str(a.dtype)}


def main():
    t_all = time.time()
    entries, missing = entry_list()
    if missing:
        sys.exit(f"缺少必需文件：{missing}")

    total = sum(src.stat().st_size for src, _ in entries)
    free = shutil.disk_usage(OUTDIR.parent).free
    print(f"待打包 {len(entries)} 个文件 / {total / 2**30:.2f} GiB")
    print(f"输出目录剩余 {free / 2**30:.1f} GiB")
    if free < total * 1.05:
        sys.exit("磁盘剩余空间不足")

    OUTDIR.mkdir(exist_ok=True)
    zip_path = OUTDIR / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    snap_before = snapshot(entries)
    manifest, written = [], 0
    t0 = time.time()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True, strict_timestamps=False) as zf:
        for i, (src, arc) in enumerate(entries, 1):
            rel = arc[len(PREFIX):]
            size = src.stat().st_size
            zi = zipfile.ZipInfo.from_file(src, arc, strict_timestamps=False)
            zi.compress_type = zipfile.ZIP_STORED
            zi.file_size = size
            zi.create_system = 3                       # Unix：让 unzip 认得权限位
            zi.external_attr = (0o100755 if rel in EXEC_PATHS else 0o100644) << 16
            h = hashlib.sha256()
            with open(src, "rb") as fsrc, zf.open(zi, "w", force_zip64=True) as fdst:
                while True:
                    chunk = fsrc.read(8 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    fdst.write(chunk)
            manifest.append({"path": rel, "bytes": size, "sha256": h.hexdigest()})
            written += size
            el = time.time() - t0
            rate = written / el / 2**20 if el > 0 else 0
            print(f"[{i:3d}/{len(entries)}] {rel:<42} {size/2**20:9.1f} MiB  "
                  f"{rate:6.1f} MiB/s  ({written/total*100:5.1f}%)", flush=True)

        doc = {
            "package": ZIP_NAME,
            "root_dir": "ppi_full",
            "variant": "ubuntu-22.04/24.04 (x86-64), bash",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_files": len(manifest),
            "total_bytes": written,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "build_host": platform.node(),
            "data": {"n_proteins": None, "pid2idx_entries": None, "channels": {}, "splits": {}},
        }
        meta = json.loads((ROOT / "workdir/features/meta.json").read_text())
        doc["data"]["n_proteins"] = meta.get("n")
        doc["data"]["pid2idx_entries"] = len(meta.get("pid2idx", {}))
        for ch in ("seq", "str", "surf", "iface", "text", "genome", "lengths", "plddt"):
            doc["data"]["channels"][ch] = npy_header(ROOT / f"workdir/features/{ch}.npy")
        import pandas as pd
        for sp in ("train", "val", "test"):
            df = pd.read_csv(ROOT / f"workdir/data/{sp}.csv")
            doc["data"]["splits"][sp] = {
                "pairs": int(len(df)),
                "proteins": int(len(set(df.pid_a) | set(df.pid_b))),
                "positives": int(df.label.sum()),
            }

        for name, text in (
            ("MANIFEST.sha256", "".join(f"{m['sha256']}  {m['bytes']}  {m['path']}\n" for m in manifest)),
            ("manifest.json", json.dumps(doc, ensure_ascii=False, indent=1)),
        ):
            data = text.encode("utf-8")
            zi = zipfile.ZipInfo(PREFIX + name, time.localtime()[:6])
            zi.compress_type = zipfile.ZIP_STORED
            zi.file_size = len(data)
            zi.create_system = 3
            zi.external_attr = 0o100644 << 16
            with zf.open(zi, "w", force_zip64=True) as fdst:
                fdst.write(data)
            print(f"[add] {name}  {len(data)} bytes")

    print(f"\nZIP 完成：{zip_path}  {zip_path.stat().st_size / 2**30:.2f} GiB  "
          f"用时 {(time.time() - t0)/60:.1f} 分钟")

    h = hashlib.sha256()
    t1 = time.time()
    with open(zip_path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    sidecar = zip_path.with_suffix(".zip.sha256")
    sidecar.write_text(f"{h.hexdigest()}  {ZIP_NAME}\n", encoding="utf-8")
    print(f"归档 SHA256：{h.hexdigest()}  ({time.time()-t1:.0f}s) -> {sidecar.name}")

    snap_after = snapshot(entries)
    changed = [k for k in snap_before if snap_before[k] != snap_after.get(k)]
    print(f"源文件快照比对：{len(snap_before)} 个，变更 {len(changed)} 个（原始文件未被改动）"
          + (f" -> {changed}" if changed else ""))
    print(f"总耗时 {(time.time() - t_all)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
