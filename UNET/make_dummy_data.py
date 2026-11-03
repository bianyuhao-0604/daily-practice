"""make_dummy_data.py — 生成假 MSD Task06 结构数据（冒烟测试用，不下载真实数据即可跑通全流程）
用法：python make_dummy_data.py --out data/smoke --cases 14
说明：第 0 例标签值用 2（测标签二值化）、中间一例无肿瘤（测空 mask 分支）、
      部分 case 平面尺寸更小（测补零分支），其余走中心裁剪分支。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SPACING = (0.8, 0.8, 2.5)   # sitk 顺序 (x, y, z)


def make_case(rng, shape, label_value=1):
    z, y, x = shape
    vol = np.full(shape, -900, dtype=np.int16)                 # 空气基底
    yy, xx = np.mgrid[0:y, 0:x]
    body = ((yy - y / 2) ** 2 / (0.45 * y) ** 2 + (xx - x / 2) ** 2 / (0.45 * x) ** 2) < 1
    lung = ((yy - y / 2) ** 2 / (0.32 * y) ** 2 + (xx - x / 2) ** 2 / (0.22 * x) ** 2) < 1
    vol[:, ~body] = -1000
    vol[:, body & ~lung] = 30                                  # 软组织壳
    vol[:, lung] = -750                                        # 肺实质
    msk = np.zeros(shape, dtype=np.uint8)
    for _ in range(int(rng.integers(1, 3))):                   # 1~2 个球状肿瘤
        cz = int(rng.integers(z // 4, 3 * z // 4))
        cy = int(rng.integers(y // 3, 2 * y // 3))
        cx = int(rng.integers(x // 3, 2 * x // 3))
        r = int(rng.integers(3, 6))
        zz, yy2, xx2 = np.ogrid[:z, :y, :x]
        blob = (zz - cz) ** 2 + (yy2 - cy) ** 2 + (xx2 - cx) ** 2 <= r * r
        msk[blob] = label_value
        vol[blob] = 60
    vol += rng.integers(-20, 20, size=shape).astype(np.int16)  # 噪声
    return vol, msk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/smoke")
    p.add_argument("--cases", type=int, default=14)
    args = p.parse_args()

    out = Path(args.out)
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    training = []
    for i in range(args.cases):
        shape = (24, 48, 48) if i % 5 == 4 else (24, 96, 96)    # 部分小尺寸测补零
        lv = 2 if i == 0 else 1                                # 第 0 例标签值 2，测二值化
        vol, msk = make_case(rng, shape, label_value=lv)
        if i == args.cases // 2:
            msk[:] = 0                                         # 一例无肿瘤，测空分支
        name = f"lung_{i + 1:03d}"
        img = sitk.GetImageFromArray(vol); img.SetSpacing(SPACING)
        lbl = sitk.GetImageFromArray(msk); lbl.SetSpacing(SPACING)
        sitk.WriteImage(img, str(out / "imagesTr" / f"{name}.nii.gz"))
        sitk.WriteImage(lbl, str(out / "labelsTr" / f"{name}.nii.gz"))
        training.append({"image": f"imagesTr/{name}.nii.gz",
                         "label": f"labelsTr/{name}.nii.gz"})

    with open(out / "dataset.json", "w") as f:
        json.dump({"name": "DummyLung", "modality": {"0": "CT"},
                   "training": training, "test": []}, f, indent=1)
    print(f"written {args.cases} dummy cases -> {out}")


if __name__ == "__main__":
    main()
