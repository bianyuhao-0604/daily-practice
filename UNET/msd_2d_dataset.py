"""msd_2d_dataset.py — 把 MSD 3D 体切成 2D 切片

train：随机抽 (case, slice)，正/负切片比例约 1:neg_ratio（默认 1:1）；
val  ：初始化时扫描各验证 case 的 label，预生成固定切片列表（正切片均匀抽样 +
       负切片补足名额），保证每个 epoch 验证集完全一致、可复现。
训练/验证按病例切分（训练集 = 除末尾 val_cases 例），无数据泄漏。
"""
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def hu_window(vol, lo=-1000, hi=400):
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


class MSD2DDataset(Dataset):
    def __init__(self, root="data/msd_task06/Task06_Lung", split="train",
                 neg_ratio=1.0, size=256, seed=42,
                 slices_per_case=30, val_cases=10, val_slices_per_case=30):
        assert split in ("train", "val")
        self.root = Path(root)
        self.size = size
        self.split = split
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.slices_per_case = slices_per_case
        self.val_slices_per_case = val_slices_per_case

        df = json.load(open(self.root / "dataset.json"))
        n_total = len(df["training"])
        if not 1 <= val_cases <= n_total - 1:
            raise ValueError(f"val_cases 需在 1..{n_total - 1} 之间，当前 {val_cases}")
        self.cases = (df["training"][:n_total - val_cases] if split == "train"
                      else df["training"][n_total - val_cases:])

        self.items = None
        if split == "val":
            self._build_val_list()

    def __len__(self):
        if self.split == "val":
            return len(self.items)
        return len(self.cases) * self.slices_per_case

    # ---------- IO ----------
    def _read_label(self, ci):
        case = self.cases[ci]
        lbl3d = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root / case["label"])))
        return lbl3d > 0                       # 统一二值化，防 1..N 多值标签直接进 BCE

    def _read_case(self, ci):
        case = self.cases[ci]
        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root / case["image"])))
        return img3d, self._read_label(ci)

    # ---------- val：固定切片列表 ----------
    def _build_val_list(self):
        """每个 case：正切片均匀抽最多 val_slices_per_case//2 张，负切片补足剩余名额。"""
        rng = np.random.default_rng(self.seed)
        items = []
        for ci in range(len(self.cases)):
            lbl3d = self._read_label(ci)
            slc = lbl3d.sum(axis=(1, 2))
            pos_z = np.where(slc > 0)[0]
            neg_z = np.where(slc == 0)[0]
            n_pos = min(len(pos_z), self.val_slices_per_case // 2)
            pos = (pos_z[np.linspace(0, len(pos_z) - 1, n_pos).astype(int)].tolist()
                   if n_pos > 0 else [])
            n_neg = min(len(neg_z), self.val_slices_per_case - n_pos)
            neg = rng.choice(neg_z, size=n_neg, replace=False).tolist() if n_neg > 0 else []
            items += [(ci, int(z)) for z in pos + neg]
        self.items = items

    # ---------- train：按需随机采样 ----------
    def _sample_z(self, lbl3d, n_slices):
        """切片采样用 python random：DataLoader 多 worker 时 PyTorch 会按
        worker+epoch 自动重播 python random 的种子；若把 numpy rng 存在 self
        里，各 worker 会复制同一份状态，导致采样大量重复。"""
        pos_z = np.where(lbl3d.sum(axis=(1, 2)) > 0)[0]
        p_neg = self.neg_ratio / (1.0 + self.neg_ratio)
        if len(pos_z) == 0:                       # 无肿瘤 case：任意层
            return random.randrange(n_slices)
        if random.random() >= p_neg:              # 正样本层
            return int(random.choice(pos_z.tolist()))
        neg_pool = np.setdiff1d(np.arange(n_slices), pos_z)
        return int(random.choice(neg_pool.tolist())) if len(neg_pool) else int(pos_z[0])

    def __getitem__(self, idx):
        ci = idx // self.slices_per_case if self.split == "train" else self.items[idx][0]
        img3d, lbl3d = self._read_case(ci)
        z = self._sample_z(lbl3d, img3d.shape[0]) if self.split == "train" \
            else self.items[idx][1]

        img = hu_window(img3d[z])
        msk = lbl3d[z].astype(np.float32)

        # 中心裁剪或补零到 size×size（肺窗区域通常居中）
        H, W = img.shape
        if H > self.size or W > self.size:
            y0 = max(0, (H - self.size) // 2); x0 = max(0, (W - self.size) // 2)
            img = img[y0:y0 + self.size, x0:x0 + self.size]
            msk = msk[y0:y0 + self.size, x0:x0 + self.size]
        img = np.pad(img, [(0, self.size - img.shape[0]), (0, self.size - img.shape[1])])
        msk = np.pad(msk, [(0, self.size - msk.shape[0]), (0, self.size - msk.shape[1])])

        # 轻量增强（仅 train）：翻转 + 90° 旋转
        if self.split == "train":
            if random.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5: img, msk = img[::-1, :], msk[::-1, :]
            k = random.randint(0, 3)
            if k: img, msk = np.rot90(img, k), np.rot90(msk, k)

        return (torch.from_numpy(np.ascontiguousarray(img)[None]),
                torch.from_numpy(np.ascontiguousarray(msk)[None]).float())


class PreparedSliceDataset(Dataset):
    """快速模式：读取 prepare_slices.py 预切好的 2D 切片（<root>/<split>/*_z*.npz）。

    与 MSD2DDataset 的输出契约完全一致：(1,size,size) float32，img∈[0,1]，msk∈{0,1}。
    区别只在 IO：每样本读 ~0.3MB 的 npz，而不是解压几十 MB 的 3D 体。
    augment=True 时做与 MSD2DDataset 相同的翻转+90° 旋转（仅训练用）。
    """
    def __init__(self, split_dir, augment=False):
        self.split_dir = Path(split_dir)
        self.files = sorted(self.split_dir.glob("*_z*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"{self.split_dir} 下没有 *_z*.npz —— 先运行 prepare_slices.py 生成")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        img = d["img"].astype(np.float32)
        msk = d["msk"].astype(np.float32)

        if self.augment:                          # 轻量增强（仅 train）
            if random.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5: img, msk = img[::-1, :], msk[::-1, :]
            k = random.randint(0, 3)
            if k: img, msk = np.rot90(img, k), np.rot90(msk, k)

        return (torch.from_numpy(np.ascontiguousarray(img)[None]),
                torch.from_numpy(np.ascontiguousarray(msk)[None]).float())
