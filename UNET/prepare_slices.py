"""prepare_slices.py — 一次性把 MSD 3D 体离线切成 2D 切片（快速模式的前置步骤）

慢的根源：msd_2d_dataset.MSD2DDataset 每个样本都要解压整个 3D 体（几十 MB gzip）
只为取 1 张切片，每 epoch 约 1620 次全量解压（≈130GB），数据管线把 GPU 饿死。
本脚本把切片一次算好存成 .npz，之后训练/评估只读 ~0.3MB 的小文件，epoch 提速一个量级。

用法：
  python prepare_slices.py --data_root data/msd_task06/Task06_Lung --out data/slices --size 256

产出（out 目录）：
  train/xxx_z0001.npz ...   img float32（肺窗 [0,1]，已裁剪/补零到 size²）、msk uint8 {0,1}
  val/xxx_z0001.npz ...
  meta.json                 各 split 的例数/切片数统计 + 泄漏自检结果

划分与 MSD2DDataset 完全一致：末尾 --val_cases 例做 val，其余 train（无泄漏）。
train：每例全部正切片 + neg_ratio×正切片数的负切片；无肿瘤例取 30 张均匀负切片。
val  ：每例全部正切片 + 等量负切片（无肿瘤例取 15 张均匀负切片）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def hu_window(vol, lo=-1000, hi=400):
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


def fit_slice(img, msk, size):
    """中心裁剪/补零到 size×size（与 MSD2DDataset.__getitem__ 的几何处理一致）"""
    H, W = img.shape
    if H > size or W > size:
        y0 = max(0, (H - size) // 2); x0 = max(0, (W - size) // 2)
        img = img[y0:y0 + size, x0:x0 + size]
        msk = msk[y0:y0 + size, x0:x0 + size]
    img = np.pad(img, [(0, size - img.shape[0]), (0, size - img.shape[1])])
    msk = np.pad(msk, [(0, size - msk.shape[0]), (0, size - msk.shape[1])])
    return img, msk


def resize_pair(img, msk, size):
    """全视野缩放到 size×size：图双线性（抗锯齿）、掩码最近邻后二值化。
    解决中心裁剪切掉贴边/胸膜下肿瘤的问题（512→256 中心窗曾丢 ~27% 正切片）。"""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(img))[None, None]
    m = torch.from_numpy(np.ascontiguousarray(msk))[None, None].float()
    img2 = F.interpolate(t, size=(size, size), mode="bilinear",
                         align_corners=False, antialias=True)[0, 0].numpy()
    msk2 = F.interpolate(m, size=(size, size), mode="nearest")[0, 0].numpy()
    return img2.astype(np.float32), (msk2 > 0.5).astype(np.uint8)


def case_stem(p):
    return Path(p).name.replace(".nii.gz", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/msd_task06/Task06_Lung")
    ap.add_argument("--out",       default="data/slices")
    ap.add_argument("--size",      type=int, default=256)
    ap.add_argument("--val_cases", type=int, default=10)      # 与训练配置保持一致
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--neg_ratio", type=float, default=1.0)   # train 每例 负:正
    ap.add_argument("--resize", action="store_true",
                    help="全视野缩放到 size×size（推荐）：替代中心裁剪，肿瘤不再被切掉")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    df = json.load(open(root / "dataset.json"))
    n_total = len(df["training"])
    vc = args.val_cases
    if not 1 <= vc <= n_total - 1:
        raise ValueError(f"val_cases 需在 1..{n_total - 1} 之间，当前 {vc}")

    splits = {"train": df["training"][:n_total - vc],
              "val":   df["training"][n_total - vc:]}
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    meta = {}
    for split, cases in splits.items():
        n_pos_sum = n_neg_sum = 0
        for ci, case in enumerate(cases):
            stem = case_stem(case["image"])
            img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(root / case["image"])))
            lbl3d = (sitk.GetArrayFromImage(sitk.ReadImage(str(root / case["label"]))) > 0)
            n_z = lbl3d.shape[0]
            pos_z = np.where(lbl3d.sum(axis=(1, 2)) > 0)[0]
            neg_z = np.where(lbl3d.sum(axis=(1, 2)) == 0)[0]

            rng = np.random.default_rng([args.seed, ci])
            if len(pos_z) == 0:                       # 无肿瘤例：只取均匀负切片
                k = 30 if split == "train" else 15
                zs = [int(z) for z in
                      (neg_z[np.linspace(0, len(neg_z) - 1, min(k, len(neg_z))).astype(int)]
                       if len(neg_z) else [])]
            else:
                n_neg = (min(len(neg_z), int(round(len(pos_z) * args.neg_ratio)))
                         if split == "train" else min(len(neg_z), len(pos_z)))
                neg_sel = rng.choice(neg_z, size=n_neg, replace=False) if n_neg else []
                zs = sorted(pos_z.tolist()) + [int(z) for z in neg_sel]

            n_pos = 0
            for z in zs:
                img_w, msk_w = hu_window(img3d[z]), lbl3d[z].astype(np.uint8)
                if args.resize:
                    img, msk = resize_pair(img_w, msk_w, args.size)
                else:
                    img, msk = fit_slice(img_w, msk_w, args.size)
                np.savez_compressed(out / split / f"{stem}_z{z:04d}.npz", img=img, msk=msk)
                n_pos += int(lbl3d[z].any())
            n_pos_sum += n_pos; n_neg_sum += len(zs) - n_pos
            print(f"[{split}] {stem}: shape={img3d.shape} pos={n_pos} neg={len(zs) - n_pos}")
            del img3d, lbl3d                          # 及时释放，防止内存峰值
        meta[split] = {"cases": len(cases), "pos": n_pos_sum,
                       "neg": n_neg_sum, "total": n_pos_sum + n_neg_sum}
        print(f"== {split}: {meta[split]}")

    tr = {case_stem(c["image"]) for c in splits["train"]}
    va = {case_stem(c["image"]) for c in splits["val"]}
    meta["leakage_check"] = "pass" if not (tr & va) else "FAIL"
    assert meta["leakage_check"] == "pass"
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"done. meta -> {out / 'meta.json'}")


if __name__ == "__main__":
    main()
