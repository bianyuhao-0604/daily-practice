"""
infer_eval.py — 在验证集上逐切片推理：Dice 汇总 + 原图/GT/预测三联图
用法：
  python infer_eval.py --data_root data/msd_task06/Task06_Lung \
                       --ckpt data/checkpoints/best.pth --out data/preds
说明：--val_cases / --size / --seed 需与训练时一致，保证评估的就是训练时的验证集。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from msd_2d_dataset import MSD2DDataset, PreparedSliceDataset
from unet import UNet


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/msd_task06/Task06_Lung")
    p.add_argument("--ckpt",    default="data/checkpoints/best.pth")
    p.add_argument("--out",     default="data/preds")
    p.add_argument("--thr",     type=float, default=0.5)
    p.add_argument("--val_cases", type=int, default=10)   # 与训练一致（非 prepared 模式）
    p.add_argument("--size",    type=int, default=256)    # 与训练一致（非 prepared 模式）
    p.add_argument("--seed",    type=int, default=42)     # 与训练一致（非 prepared 模式）
    p.add_argument("--prepared", default=None,            # 快速模式：prepare_slices.py 的输出目录
                   help="如 data/slices；与训练用同一目录，保证评估的就是训练时的验证集")
    p.add_argument("--max_plots", type=int, default=100,  # 最多出多少张三联图（0=全部）
                   help="验证集大时出图很耗时；优先画有肿瘤的切片")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=1, num_classes=1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt} (epoch={ckpt['epoch']}, dice={ckpt['dice']:.4f})")

    if args.prepared:
        ds = PreparedSliceDataset(Path(args.prepared) / "val")
    else:
        ds = MSD2DDataset(args.data_root, split="val", size=args.size,
                          seed=args.seed, val_cases=args.val_cases)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    mode = f"prepared: {args.prepared}" if args.prepared else f"on-the-fly: {args.data_root}"
    print(f"val slices: {len(ds)}  ({mode})  -> {out_dir}")

    dices, gt_pos, names = [], [], []
    n_plots = 0
    for i in range(len(ds)):
        img, msk = ds[i]
        img_b = img[None].to(device)
        logits = model(img_b)
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > args.thr).astype(np.uint8)
        gt = msk[0].numpy().astype(np.uint8)
        gt_pos.append(gt.sum() > 0)
        names.append(ds.files[i].stem if hasattr(ds, "files") else f"{i:04d}")

        denom = pred.sum() + gt.sum()
        inter = (pred * gt).sum()
        dice = 2 * inter / (denom + 1e-6) if denom > 0 else 1.0
        dices.append(dice)

        # 可视化：原图 / GT / 预测 三联图（优先画有肿瘤的切片，限 --max_plots 张）
        if gt.any() and n_plots < args.max_plots:
            n_plots += 1
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img[0], cmap="gray"); axes[0].set_title("Image")
            axes[1].imshow(img[0], cmap="gray")
            if gt.any(): axes[1].contour(gt, levels=[0.5], colors="lime")
            axes[1].set_title("GT")
            axes[2].imshow(img[0], cmap="gray")
            if pred.any(): axes[2].contour(pred, levels=[0.5], colors="red")
            axes[2].set_title(f"Pred  dice={dice:.3f}")
            for a in axes: a.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{i:04d}_{names[-1]}.png", dpi=100)
            plt.close(fig)

    dices = np.array(dices); gt_pos = np.array(gt_pos)
    print(f"\n=== Summary ({len(dices)} slices) ===")
    print(f"mean Dice          : {dices.mean():.4f}")
    print(f"median Dice        : {np.median(dices):.4f}")
    if gt_pos.any():
        print(f"mean Dice (有肿瘤切片): {dices[gt_pos].mean():.4f}  [n={int(gt_pos.sum())}]")
    if (~gt_pos).any():
        print(f"mean Dice (空切片)    : {dices[~gt_pos].mean():.4f}  [n={int((~gt_pos).sum())}]")
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"n_slices={len(dices)}\nn_plots={n_plots}\n")
        f.write(f"mean_dice={dices.mean():.4f}\nmedian_dice={np.median(dices):.4f}\n")
        if gt_pos.any():
            f.write(f"mean_dice_pos_slices={dices[gt_pos].mean():.4f}\n")
    with open(out_dir / "dice_per_slice.csv", "w") as f:
        f.write("index,name,gt_pos,dice\n")
        for i, (nm, gp, dc) in enumerate(zip(names, gt_pos, dices)):
            f.write(f"{i:04d},{nm},{int(gp)},{dc:.4f}\n")


if __name__ == "__main__":
    main()
