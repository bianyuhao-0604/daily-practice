# 2D U-Net 肺部肿瘤逐层分割（MSD Task06_Lung）

## 文件清单
| 文件 | 作用 |
|---|---|
| `check_msd.py` | 数据体检：模态/例数/shape/spacing/HU 范围/肿瘤体积 |
| `msd_2d_dataset.py` | 3D 体 → 2D 切片数据集（train 随机采样；val 固定切片列表，可复现）+ PreparedSliceDataset（读预切片，快速模式） |
| `prepare_slices.py` | 一次性预处理：3D 体 → 2D npz 切片库（快速模式前置，几分钟） |
| `unet.py` | 标准 U-Net，base=32，约 7.2M/7.8M 参数（双线性/转置卷积）；`python unet.py` 可自检双上采样路径 |
| `train.py` | BCE+Dice 训练；AMP、余弦退火；按 val Dice 保存 `best.pth` |
| `infer_eval.py` | 验证集逐切片推理：Dice 汇总 + 原图/GT/预测三联图 |
| `make_dummy_data.py` | 生成假 MSD 结构数据（冒烟测试用，可选） |

## 环境
```bash
pip install torch numpy SimpleITK matplotlib
```

## 数据准备（真实实验）
从 MSD 官网（medicaldecathlon.com）下载 Task06_Lung，解压到 `data/msd_task06/`，最终结构：
```
data/msd_task06/Task06_Lung/{dataset.json, imagesTr/*.nii.gz, labelsTr/*.nii.gz}
```

## 运行命令

### 快速模式（推荐：先预切片，epoch 提速约一个量级）
慢的根源：默认模式每个样本都要解压整个 3D 体（每 epoch ≈1620 次、≈130GB 解压量），
GPU 全程等数据（显存占用低、epoch 数百秒）。先跑一次性预处理（几分钟）：
```bash
# --resize：全视野缩放 512→256（推荐）。默认的中心裁剪会切掉 ~27% 贴边/胸膜下肿瘤
python prepare_slices.py --data_root data/msd_task06/Task06_Lung --out data/slices_v4 --size 256 --resize
```
之后训练/评估都走 `--prepared`，只读 ~0.3MB 的小切片文件：
```bash
# 训练（显存富余可加 --batch_size 16；本机实测 bs=8 仅占 ~2G 显存）
# val_dice 为逐切片宏平均（与 infer_eval.py 同口径，空标签+空预测记 1.0）
python train.py --prepared data/slices_v4 --epochs 100 --batch_size 16 --lr 1e-4
# 评估（--max_plots 控制三联图数量，默认 100 张、优先有肿瘤切片；0=全画）
python infer_eval.py --prepared data/slices_v4 --ckpt data/checkpoints/best.pth --out data/preds_v4
```
产物 ckpt：`best.pth`（按宏平均 val_dice 选优）+ `last.pth`（最新一轮，用于对比）。

### 慢速模式（不推荐，仅作对照）
```bash
# 1) 数据体检
python check_msd.py --root data/msd_task06/Task06_Lung

# 2) 训练（默认：末尾 10 例做验证、其余 54 例训练；256×256、bs=8、AMP）
python train.py --data_root data/msd_task06/Task06_Lung --epochs 100 --batch_size 8 --lr 1e-4

# 3) 推理 + 评估 + 三联图（--val_cases/--size/--seed 与训练保持一致）
python infer_eval.py --data_root data/msd_task06/Task06_Lung \
                     --ckpt data/checkpoints/best.pth --out data/preds
```

产物：`data/slices/{train,val}/*.npz + meta.json`（快速模式）、`data/checkpoints/best.pth` + `train_log.txt`、
`data/preds/编号_切片名.png + summary.txt + dice_per_slice.csv`

## 冒烟测试（不下载真实数据，先验证全流程）
```bash
python make_dummy_data.py --out data/smoke --cases 14
python check_msd.py --root data/smoke
# 快速模式（与真实数据流程完全一致）
python prepare_slices.py --data_root data/smoke --out data/slices_smoke --size 64
python train.py --prepared data/slices_smoke --epochs 3 --batch_size 4 --num_workers 2
python infer_eval.py --prepared data/slices_smoke --ckpt data/checkpoints/best.pth --out data/preds_smoke
# 慢速模式（对照，可不跑）
python train.py --data_root data/smoke --size 64 --epochs 3 --batch_size 4 --num_workers 2
python infer_eval.py --data_root data/smoke --size 64 --ckpt data/checkpoints/best.pth --out data/preds_smoke
```

## 注意事项
- **数据划分**：train/val 按病例切分（训练 = 除末尾 `--val_cases` 例），无泄漏；两种模式同一划分，val 固定可复现。
- **快速模式**下每 epoch 训练集 = 全部正切片 + 等量负切片（约 3000~8000 张，与慢速模式的 1620 张/epoch 不是同一集合，但覆盖更全）。
- **CPU / `--no_amp`** 自动关闭混合精度；Windows 下多进程报错可加 `--num_workers 0`。
- 慢速模式每张切片都按需读取整个 3D 体，IO 密集；**推荐一律走快速模式**。
- 非快速模式下 `--val_cases`、`--size`、`--seed` 在 train 与 infer_eval 间需保持一致；快速模式下共用 `--prepared` 目录即可。
