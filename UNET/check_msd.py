"""check_msd.py — MSD Task06 数据体检
用法：python check_msd.py --root data/msd_task06/Task06_Lung
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/msd_task06/Task06_Lung")
    args = p.parse_args()
    root = Path(args.root)

    df = json.load(open(root / "dataset.json"))
    print(f"Modality: {df['modality']}, Train cases: {len(df['training'])}")

    for i, sample in enumerate(df["training"][:5]):
        img = sitk.ReadImage(str(root / sample["image"]))
        lbl = sitk.ReadImage(str(root / sample["label"]))
        v, m = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(lbl)
        vox = np.prod(img.GetSpacing()) / 1000  # 单体素 mm³→cm³
        tumor_vol = (m > 0).sum() * vox
        print(f"[{i}] {Path(sample['image']).stem[:35]}")
        print(f"    shape={v.shape}  spacing={np.round(img.GetSpacing(), 2).tolist()}")
        print(f"    HU range=[{v.min()}, {v.max()}]  tumor voxels={int((m > 0).sum())}  "
              f"volume≈{tumor_vol:.1f} cm³")


if __name__ == "__main__":
    main()
