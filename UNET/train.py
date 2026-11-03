"""
train.py — 2D U-Net 训练（MSD Task06 肺部肿瘤逐层分割）
用法：
  python train.py --data_root data/msd_task06/Task06_Lung --epochs 100 --batch_size 8 --lr 1e-4
说明：
  - 训练/验证按病例切分：训练集 = 除末尾 --val_cases 例之外的全部病例（无泄漏）；
  - 验证集为固定切片列表（初始化时扫描 label 生成，一次性开销几秒到十几秒）；
  - CPU 环境或 --no_amp 时自动关闭混合精度。
"""
import argparse
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from msd_2d_dataset import MSD2DDataset, PreparedSliceDataset
from unet import UNet


# ===== Loss =====
class DiceLoss(nn.Module):
    def forward(self, logits, target, eps=1e-6):
        pred = torch.sigmoid(logits)
        num = 2 * (pred * target).sum(dim=(2, 3)) + eps
        den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        return 1 - (num / den).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice(logits, target)


# ===== 评估 =====
@torch.no_grad()
def dice_score(logits, target, thr=0.5, eps=1e-6):
    """逐样本 Dice 的批内平均（宏平均），与 infer_eval.py 完全同口径：
    每张切片一票；空标签+空预测记 1.0。
    （旧版把整个 batch 的像素池在一起算微平均，失败切片按面积重罚，
    导致训练日志的 val_dice 比 infer_eval 低 ~0.33、best 模型选择失真。）"""
    pred = (torch.sigmoid(logits) > thr).float()
    p, t = pred.flatten(1), target.flatten(1)
    num = 2 * (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    d = torch.where(den > 0, num / (den + eps), torch.ones_like(den))
    return d.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/msd_task06/Task06_Lung")
    p.add_argument("--ckpt_dir",  default="data/checkpoints")
    p.add_argument("--epochs",    type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)      # 8GB 显存 256×256 + AMP 用 8 稳
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=None,
                   help="None=自动（Windows 0 / 其他系统 4）；prepared 数据单进程毫无压力")
    p.add_argument("--val_cases", type=int, default=10)      # 末尾 N 例做验证，其余训练
    p.add_argument("--size",      type=int, default=256)
    p.add_argument("--slices_per_case", type=int, default=30)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--no_amp",    action="store_true")       # 关闭混合精度（CPU 上自动关）
    p.add_argument("--prepared",  default=None,             # 快速模式：prepare_slices.py 的输出目录
                   help="如 data/slices；设置后忽略 --data_root/--size，直接读预切片")
    args = p.parse_args()

    # Windows 上 DataLoader 多进程是已知的偶发挂起源；prepared 数据 IO 极小，单进程足矣
    if args.num_workers is None:
        args.num_workers = 0 if platform.system() == "Windows" else 4

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = (not args.no_amp) and device.type == "cuda"

    if args.prepared:
        train_ds = PreparedSliceDataset(Path(args.prepared) / "train", augment=True)
        val_ds = PreparedSliceDataset(Path(args.prepared) / "val")
    else:
        train_ds = MSD2DDataset(args.data_root, split="train", size=args.size, seed=args.seed,
                                slices_per_case=args.slices_per_case, val_cases=args.val_cases)
        val_ds = MSD2DDataset(args.data_root, split="val", size=args.size, seed=args.seed,
                              val_cases=args.val_cases)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin)
    if args.prepared:
        print(f"device={device.type}  [prepared mode: {args.prepared}]  "
              f"train={len(train_ds)}  val={len(val_ds)}  num_workers={args.num_workers}")
    else:
        print(f"device={device.type}  train={len(train_ds)} slices ({len(train_ds.cases)} cases)  "
              f"val={len(val_ds)} fixed slices ({len(val_ds.cases)} cases)  num_workers={args.num_workers}")

    model = UNet(in_ch=1, num_classes=1).to(device)
    print(f"UNet params: {sum(p_.numel() for p_ in model.parameters()) / 1e6:.2f}M")
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if hasattr(torch.amp, "GradScaler"):          # torch >= 2.3
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    else:                                         # 旧版本兼容
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_kwargs = dict(device_type=device.type,
                      dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                      enabled=use_amp)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    log_path = ckpt_dir / "train_log.txt"
    with open(log_path, "a") as f:
        f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(args)}\n")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        t0 = time.time()
        tr_loss, n_seen = 0.0, 0
        for img, msk in train_dl:
            img, msk = img.to(device, non_blocking=True), msk.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**amp_kwargs):
                logits = model(img)
                loss = criterion(logits, msk)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * img.size(0); n_seen += img.size(0)
        tr_loss /= max(n_seen, 1)
        scheduler.step()

        # ---- val ----
        model.eval()
        va_dice, va_loss, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for img, msk in val_dl:
                img, msk = img.to(device), msk.to(device)
                with torch.autocast(**amp_kwargs):
                    logits = model(img)
                    loss = criterion(logits, msk)
                va_loss += loss.item() * img.size(0)
                va_dice += dice_score(logits.float(), msk) * img.size(0)
                n_val += img.size(0)
        va_loss /= max(n_val, 1); va_dice /= max(n_val, 1)
        dt = time.time() - t0

        line = (f"[{epoch:3d}/{args.epochs}] "
                f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                f"val_dice={va_dice:.4f}  time={dt:.1f}s")
        print(line)
        with open(log_path, "a") as f: f.write(line + "\n")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "dice": va_dice},
                       ckpt_dir / "best.pth")
            print(f"  ↳ saved new best (dice={va_dice:.4f})")

        # last.pth：始终记录最新一轮（best.pth 受验证噪声影响时可用于对比）
        torch.save({"model": model.state_dict(),
                    "epoch": epoch, "dice": va_dice},
                   ckpt_dir / "last.pth")

    print(f"Done. Best val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
