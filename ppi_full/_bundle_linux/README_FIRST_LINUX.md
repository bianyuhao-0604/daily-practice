# 先看这里（Ubuntu 版，60 秒）

这是 **PPI 全通道模型** 的 Ubuntu 实验包：代码 + 全部运行数据（50.1 GiB）+ Linux 脚本与指南。
解包后**不需要联网、不需要重新抽特征**，装好依赖即可训练。

> 目标系统：**Ubuntu 22.04 / 24.04 LTS（x86-64）**，NVIDIA RTX 4090 48 GB。
> 与 Windows 版的唯一差别是**启动脚本与文档**；`config.py`/`data.py`/`model.py`/`engine.py`/`stats.py`/`run_all.py` 等 Python 源码**逐字节未改**。

## 三步开始

```bash
sudo apt install -y unzip                                # 若未安装
unzip -q ppi_full_bundle_ubuntu_20261027.zip -d "$HOME"  # 解到 ~/ppi_full
cd ~/ppi_full
python3 verify_bundle.py                                 # 校验：OK=149 BAD=0 MISSING=0（3-6 分钟）

bash setup_ubuntu.sh --smoke                             # 装环境 + 自检 + 33 项冒烟
```

跑完 `setup_ubuntu.sh` 就该看到 **33/33 通过**。然后先测速：

```bash
source .venv/bin/activate
python run_all.py ladder --levels R7 --seeds 42 --bs 64 --lr 1e-4
```

确认单个 epoch 的耗时后，再决定跑全套：

```bash
tmux new -s ppi 'bash RUN_ALL_EXPERIMENTS.sh'   # 3 seed / 40 trial，约 4-7 天
```

## 文件导航

| 文件 | 内容 |
|---|---|
| `USAGE_LINUX.md` | **主文档**：包内容、环境、逐条命令、长跑/后台、耗时、口径、Linux 排错 |
| `setup_ubuntu.sh` | 一键建 `.venv`、装 CUDA torch 与依赖、文件自检、可选冒烟 |
| `RUN_ALL_EXPERIMENTS.sh` | 一键顺序跑全套（支持 `--quick`、`--dry-run`、`--skip-*`） |
| `verify_bundle.py` | 按 `MANIFEST.sha256` 逐文件校验（跨平台，Windows 版同一个文件） |
| `manifest.json` | 结构化清单：大小、SHA-256、通道形状、样本数 |
| `ARCHITECTURE.md` | 模型与数据管线设计（其中文件数/通道数等统计已过时，以本包实际为准） |
| `workdir/` | 数据目录：`features/`（8 个 .npy，50 GB，**必需**）、`data/`（三份划分 CSV）、`ssl/adapter.pt`；跑完后 `runs/` 里出现产物 |

## 三条硬约束

1. **所有命令都在 `~/ppi_full` 下执行**（`config.Paths()` 全是 `./workdir/...` 相对路径）。
2. `workdir/features/` 下 8 个 `.npy` 是运行必需（共 50 GB），**不要只拷代码**。
3. 文件系统必须是 **ext4 / xfs**（本地盘或 NVMe），**不要放在 NTFS/exFAT/SMB 上**——`mmap` 会退化成龟速甚至报 `Bus error`。
