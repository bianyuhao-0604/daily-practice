"""从 pids.txt 生成 AFDB cif 与 UniProt 注释文本 URL 清单。

注意：AFDB 官方 v4 已于 2025 年下线（实测对任意 pid 都 404），现役是 v6。
"""
import argparse, os

AFDB_TMPL = "https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v6.cif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    with open(os.path.join(a.outdir, "afdb_urls.txt"), "w") as f:
        for p in pids:
            f.write(AFDB_TMPL.format(pid=p) + "\n")
    with open(os.path.join(a.outdir, "uniprot_urls.txt"), "w") as f:
        for p in pids:
            f.write(f"https://rest.uniprot.org/uniprotkb/{p}.txt\n")
    print(f"{len(pids)} pids → afdb_urls.txt / uniprot_urls.txt")

if __name__ == "__main__":
    main()
