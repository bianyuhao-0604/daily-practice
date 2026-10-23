"""AFDB 下载 + pLDDT(B-factor) + 80d 确定性几何描述子 + iface。
iface providers: masif_npy(论文级必用) | heuristic(仅管线打通)。并行版见 prep/geo_worker.py。"""
import argparse, os, sys, urllib.request
import warnings
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from config import MAX_LEN, Paths, FeatureModels

# F1：Bio.PDB 只在解析 CIF 时用，改为延迟导入（parse_cif 内部），
# 使 descriptors()/iface_heuristic() 可在无 biopython 的环境下被复用（如 prep/surf_from_npz.py）。
HYDRO = set("AVILMFWC")
ARO = set("FWY")
POS = set("KRH")
NEG = set("DE")

# F1：三字母 → 一字母。Bio.PDB 的 get_resname() 返回 "ALA" 这类三字母码，
# 而上面的类别集合是一字母码；早期直接透传导致 grp 恒为 2，
# 使 80 维中 19–27 / 28–32 / 63–64 共 16 维恒为常数。
THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def to_one_letter(resname):
    """三字母残基名 → 一字母；未知返回 X。"""
    return THREE2ONE.get(str(resname).upper(), "X")

def download_afdb(pid, out_dir, url_tpl):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"AF-{pid}-F1-model_v4.cif")
    if not os.path.exists(p):
        urllib.request.urlretrieve(url_tpl.format(pid=pid), p)
    return p

def parse_cif(path, max_res=MAX_LEN):
    from Bio.PDB import MMCIFParser                    # F1：延迟导入，避免硬依赖 biopython
    s = MMCIFParser(QUIET=True).get_structure("s", path)[0]
    chain = next(iter(s))
    coords, res, plddt = [], [], []
    for r in chain:
        if "CA" not in r:
            continue
        coords.append(r["CA"].coord.tolist())
        res.append(to_one_letter(r.get_resname()))    # F1：三字母 → 一字母（原先直接透传，特征恒常数）
        plddt.append(r["CA"].bfactor)
        if len(coords) >= max_res:          # titin(34350aa) 提前截断：下游本来就只用前 1022
            break
    return np.array(coords), res, np.array(plddt, np.float32)


def descriptors(coords, res, plddt):
    L = len(res)
    D = np.zeros((L, 80), np.float32)
    grp = np.zeros(L, np.int64)
    for i, r in enumerate(res):
        grp[i] = 1 if r in ARO else 0 if r in HYDRO else 3 if r in POS else 4 if r in NEG else 2
    radii = [4, 6, 8, 10, 12, 14, 16, 18]
    for i in range(L):
        d = np.linalg.norm(coords - coords[i], axis=1)
        D[i, 0:8] = [(d <= r).sum() - 1 for r in radii]
        nb = coords[d <= 12]
        if len(nb) >= 3:
            c = nb - nb.mean(0)
            cov = c.T @ c / max(1, len(nb) - 1)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            D[i, 8] = ev[0] / (ev.sum() + 1e-6)
            D[i, 9] = ev[1] / (ev.sum() + 1e-6)
            D[i, 10] = ev[2] / (ev.sum() + 1e-6)
            D[i, 11] = 1 - ev[2] / (ev[0] + 1e-6)
            D[i, 12] = ev[1] / (ev[0] + 1e-6)
            n1 = np.roll(c, -1, axis=0)
            n2 = np.roll(c, 1, axis=0)
            D[i, 13] = np.mean(np.abs(np.einsum("ij,ij->i", n1, n2)))
        if 0 < i < L - 1:
            v1, v2 = coords[i] - coords[i - 1], coords[i + 1] - coords[i]
            ang = np.arccos(np.clip(np.dot(v1, v2) /
                                    (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
            D[i, 14], D[i, 15] = np.sin(ang), np.cos(ang)
            if 1 < i < L - 2:
                v3 = coords[i + 2] - coords[i + 1]
                n = np.cross(v1, v2)
                m = np.cross(v2, v3)
                dih = np.arccos(np.clip(np.dot(n, m) /
                                  (np.linalg.norm(n) * np.linalg.norm(m) + 1e-6), -1, 1))
                D[i, 16], D[i, 17] = np.sin(dih), np.cos(dih)
        D[i, 18] = plddt[i] / 100.0
        for rr, base in ((8, 19), (12, 22), (16, 25)):
            sel = d <= rr
            if sel.sum() > 1:
                g = grp[sel]
                D[i, base] = (g == 0).mean()
                D[i, base + 1] = (g == 2).mean()
                D[i, base + 2] = ((g == 3) | (g == 4)).mean()
        D[i, 28:33] = np.eye(5)[grp[i]]
        for f in range(8):
            D[i, 33 + 2 * f] = np.sin((f + 1) * i / 20.0)
            D[i, 34 + 2 * f] = np.cos((f + 1) * i / 20.0)
        nb8 = d[(d > 0) & (d <= 8)]
        if len(nb8):
            D[i, 49:53] = np.percentile(nb8, [25, 50, 75, 100])
        if len(nb) >= 3:
            dp = np.abs((nb - nb.mean(0)) @ np.linalg.svd(nb - nb.mean(0))[2][2])
            D[i, 53], D[i, 54] = dp.mean(), dp.std()
        for rr, base in ((8, 56), (12, 58)):
            idxs = np.where((d > 0) & (d <= rr))[0]
            if len(idxs):
                D[i, base] = np.abs(idxs - i).mean() / rr
                D[i, base + 1] = np.abs(idxs - i).std() / rr
        sel = d <= 8
        D[i, 60] = plddt[sel].mean() if sel.sum() else plddt[i] / 100.0
        D[i, 61] = plddt[sel].std() if sel.sum() > 1 else 0.0
        if len(nb) >= 3:
            rg = np.sqrt(((nb - nb.mean(0)) ** 2).sum(1)).mean()
            D[i, 62] = rg / 12.0
        if (d <= 8).sum() > 1:
            D[i, 63] = (grp[d <= 8] == 0).mean()
            D[i, 64] = (grp[d <= 8] == 4).mean()
        if (d > 0).sum() > 0:
            D[i, 65] = d[d > 0].mean()
            if (d > 0).sum() > 1:
                D[i, 66] = d[d > 0].std()
        D[i, 67] = (d <= 8).sum() / L
        D[i, 68] = i / max(1, L - 1)
        D[i, 79] = 1.0                                    # 69–78 预留 = 0
    return D

def iface_heuristic(coords, D, plddt):
    z = 2.0 * (1.0 - np.clip(D[:, 2] / 10.0, 0, 1)) + 0.5 * (plddt / 100.0 - 0.5)
    return 1.0 / (1.0 + np.exp(-z))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--provider", choices=["heuristic", "masif_npy"], default="heuristic")
    ap.add_argument("--masif-dir", default="")
    ap.add_argument("--afdb-dir", default="./workdir/afdb")
    ap.add_argument("--feat-dir", default=None)
    a = ap.parse_args()
    fm = FeatureModels()
    feat_dir = a.feat_dir or Paths().feat_dir
    os.makedirs(feat_dir, exist_ok=True)
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    for n, pid in enumerate(pids):
        cif = download_afdb(pid, a.afdb_dir, fm.afdb_url)
        coords, res, plddt = parse_cif(cif)
        L = min(len(res), MAX_LEN)
        D = descriptors(coords[:L], res[:L], plddt[:L])
        if a.provider == "masif_npy":
            iface = np.load(os.path.join(a.masif_dir, pid + ".iface.npy"))[:L]
        else:
            iface = iface_heuristic(coords[:L], D, plddt[:L])
        np.save(os.path.join(feat_dir, f"surf_{pid}.npy"), D.astype(np.float16))
        np.save(os.path.join(feat_dir, f"iface_{pid}.npy"), iface.astype(np.float16))
        np.save(os.path.join(feat_dir, f"plddt_{pid}.npy"), plddt[:L].astype(np.float16))
        if n % 200 == 0:
            print(f"surf {n}/{len(pids)}")

if __name__ == "__main__":
    main()
