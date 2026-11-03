"""verify_extra.py — 附加验证（针对 make_dummy_data.py 生成的假数据）
用法：python verify_extra.py --root data/smoke
验证点：val 确定性 / seed 敏感性 / train-val 无泄漏 / 张量契约 / 标签二值化
"""
import argparse

import torch

from msd_2d_dataset import MSD2DDataset

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/smoke")
p.add_argument("--size", type=int, default=64)
p.add_argument("--prepared", default=None,
               help="prepare_slices.py 输出目录（可选，给出则加测快速模式契约）")
args = p.parse_args()

# 1) val 确定性：两个独立实例逐样本一致
a = MSD2DDataset(args.root, split="val", size=args.size, seed=42)
b = MSD2DDataset(args.root, split="val", size=args.size, seed=42)
assert len(a) == len(b) and len(a) > 0, (len(a), len(b))
same = all(torch.equal(a[i][0], b[i][0]) and torch.equal(a[i][1], b[i][1])
           for i in range(len(a)))
print(f"[1] val deterministic across instances: {same}  (n={len(a)})")
assert same

# 2) 换 seed 后 val 负切片选择应不同（负切片 mask 恒为零，须比图像）
c = MSD2DDataset(args.root, split="val", size=args.size, seed=7)
diff = sum(not torch.equal(a[i][0], c[i][0]) for i in range(min(len(a), len(c))))
print(f"[2] seed=7 vs seed=42: {diff}/{min(len(a), len(c))} 个样本图像不同 (应>0)")
assert diff > 0

# 3) train/val 按病例无重叠
t = MSD2DDataset(args.root, split="train", size=args.size, seed=42)
tr_files = set(x["image"] for x in t.cases)
va_files = set(x["image"] for x in a.cases)
print(f"[3] train cases={len(tr_files)}, val cases={len(va_files)}, "
      f"overlap={len(tr_files & va_files)}")
assert not (tr_files & va_files)

# 4) 张量契约：形状/dtype/值域
img, msk = t[0]
assert img.shape == (1, args.size, args.size) and msk.shape == (1, args.size, args.size)
assert img.dtype == torch.float32 and msk.dtype == torch.float32
assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
assert set(msk.unique().tolist()) <= {0.0, 1.0}
print(f"[4] tensor contract OK: img{tuple(img.shape)}[{float(img.min()):.2f},"
      f"{float(img.max()):.2f}], msk values={sorted(set(msk.unique().tolist()))}")

# 5) 标签值=2 的 case（假数据第 0 例）二值化后 mask ∈ {0,1}
found_pos = False
for i in range(180):           # 前 30 个 idx 属于 case 0，随机采样多次
    im, mk = t[i]
    if mk.max() > 0:
        assert mk.max() == 1.0
        found_pos = True
        break
print(f"[5] 非 0/1 标签二值化后 max=1.0: {found_pos}")
assert found_pos

# 6) val 固定列表：正/负切片分布
n_pos_items = sum(1 for ci, z in a.items if a._read_label(ci)[z].any())
print(f"[6] val 固定列表: {len(a.items)} 项, 其中正切片 {n_pos_items} "
      f"(每 case 上限 val_slices_per_case=30)")
print("ALL EXTRA CHECKS PASSED")

# 7) 可选：快速模式（PreparedSliceDataset）契约
if args.prepared:
    from pathlib import Path
    from msd_2d_dataset import PreparedSliceDataset
    pt = PreparedSliceDataset(Path(args.prepared) / "train", augment=True)
    pv = PreparedSliceDataset(Path(args.prepared) / "val")
    pv2 = PreparedSliceDataset(Path(args.prepared) / "val")
    img, msk = pv[0]
    assert img.shape == msk.shape == (1, img.shape[-1], img.shape[-1])
    assert img.dtype == msk.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert set(msk.unique().tolist()) <= {0.0, 1.0}
    assert all(torch.equal(pv[i][0], pv2[i][0]) and torch.equal(pv[i][1], pv2[i][1])
               for i in range(len(pv)))                      # 无增强路径确定性
    assert not ({f.name for f in pt.files} & {f.name for f in pv.files})  # 病例零重叠
    print(f"[7] prepared 契约 OK: train={len(pt)} val={len(pv)}, "
          f"确定性/值域/二值/无泄漏 全部通过")
    print("ALL EXTRA CHECKS (incl. prepared) PASSED")
