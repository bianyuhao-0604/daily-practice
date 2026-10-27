# 先看这里（60 秒）

这是 **PPI 全通道模型** 的完整实验包：代码 + 运行实验所需的全部数据（50.1 GiB）+ 使用指南。
解包后不需要联网、不需要重新抽特征，装好依赖即可训练。

## 三步开始

```powershell
tar -xf ppi_full_bundle_20261025.zip -C D:\      # 1) 解包 → D:\ppi_full（5–15 分钟）
cd D:\ppi_full
python verify_bundle.py                          # 2) 校验（3–6 分钟，应全部 OK）

py -3.11 -m venv .venv                           # 3) 装环境
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

python smoke_all.py                              # 自检：33 项应全绿（2–5 分钟）
```

## 然后跑实验

```powershell
# 先测速：单级单 seed
python run_all.py ladder --levels R7 --seeds 42 --bs 64 --lr 1e-4

# 全套（约 2.5–5 天，单卡；详细说明见《使用指南.md》）
.\RUN_ALL_EXPERIMENTS.ps1
```

## 文件导航

| 文件 | 内容 |
|---|---|
| `使用指南.md` | **主文档**：包内容、硬件说明、环境、逐条命令、耗时、口径、排错 |
| `RUN_ALL_EXPERIMENTS.ps1` | 一键顺序跑全套实验并落日志 |
| `verify_bundle.py` | 按 `MANIFEST.sha256` 校验每个文件 |
| `manifest.json` | 结构化清单（大小、SHA-256、数据形状、样本数） |
| `smoke_report.md` | 打包时的冒烟结果（33/33，CPU） |
| `ARCHITECTURE.md` | 模型与数据管线设计（注意：其中文件数/通道数等统计已过时，以本包实际为准） |
| `workdir/runs/` | 打包时的旧 CPU 产物（R0/R1/R2 部分 seed），可删 |

## 两句话记住关键约束

1. 所有命令都在 `D:\ppi_full` 下执行（`config.Paths` 是相对路径）。
2. `workdir/features/` 里 8 个 `.npy` 是运行必需，**不要只拷代码**。
