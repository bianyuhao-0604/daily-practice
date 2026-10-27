# PPI 全通道模型 · Ubuntu 实验包使用指南

> **包体**：`ppi_full_bundle_ubuntu_20261027.zip`（≈ 50.1 GiB，149 个文件）
> **打包时间**：2026-10-27
> **目标系统**：Ubuntu 22.04 / 24.04 LTS（x86-64）+ NVIDIA RTX 4090 48 GB
> **代码版本**：与 `ppi_full_bundle_20261025.zip`（Windows 版）**同一份 Python 源码，逐字节未改**；
> 本版只替换了启动脚本（`.ps1` → `.sh`）与文档，并修正了 Windows 专属的路径/权限说明。
> **已验证**：包内容完整性（`MANIFEST.sha256` 逐文件）、脚本语法（`bash -n`）、参数解析（`--dry-run`）；
> **未验证**：真实 Ubuntu 上的 GPU 训练。**换平台后必须在目标机跑一次 `smoke_all.py`**（见 §5.2）。

---

## 0. 五分钟上手

```bash
# 1) 解包（50 GB，约 4-10 分钟）
sudo apt install -y unzip            # 若未安装
unzip -q ppi_full_bundle_ubuntu_20261027.zip -d "$HOME"
cd ~/ppi_full

# 2) 校验（3-6 分钟；应输出 OK=149 BAD=0 MISSING=0）
python3 verify_bundle.py

# 3) 装环境 + 自检 + 冒烟（10-20 分钟，取决于网速）
bash setup_ubuntu.sh --smoke

# 4) 测速：单级单 seed
source .venv/bin/activate
python run_all.py ladder --levels R7 --seeds 42 --bs 64 --lr 1e-4
```

全套一键跑（先看 §7 的长跑建议，别直接 SSH 前台跑 5 天）：`bash RUN_ALL_EXPERIMENTS.sh`。

---

## 1. 与 Windows 版的差异（就这些）

| 项 | Windows 版 | Ubuntu 版 |
|---|---|---|
| 启动脚本 | `RUN_ALL_EXPERIMENTS.ps1` | `RUN_ALL_EXPERIMENTS.sh`（`bash`，LF 换行） |
| 环境脚本 | 无（手册里手敲 4 条） | `setup_ubuntu.sh`（建 venv + 装依赖 + 自检 + 可选冒烟） |
| 激活虚拟环境 | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| 解释器 | `py -3.11` | `python3`（3.10-3.12 均可，24.04 自带 3.12） |
| 解包 | `tar -xf ... -C D:\`（bsdtar 支持 zip） | `unzip -q ... -d ~`（GNU tar **不认** zip） |
| 路径 | `D:\ppi_full` | `~/ppi_full`（或 `/opt/ppi_full`，**必须在 ext4/xfs 上**） |
| 可执行位 | 不适用 | `setup_ubuntu.sh`、`RUN_ALL_EXPERIMENTS.sh`、`mmseqs/bin/mmseqs` 需 `+x`（脚本已 `chmod`） |
| 长跑守护 | 不适用 | `tmux` / `nohup` / `systemd-run`（见 §7） |
| 文件数 | 148 | 149（`.ps1` 换成两个 `.sh`，`使用指南.md` 换成 `USAGE_LINUX.md`/`README_FIRST_LINUX.md`，`MANIFEST`/`manifest.json` 仍在） |

包内**非 ASCII 文件名已全部改为 ASCII**（Windows 版的 `使用指南.md` 在某些 `unzip` 版本 + `LANG=C` 下会解成乱码名），中文只在文件**内容**里，不涉及文件名编码。

---

## 2. 包内包含什么

### 2.1 代码（56 个 `.py`：根 15 + `features/` 4 + `prep/` 37）

| 位置 | 内容 | 作用 |
|---|---|---|
| 根目录 | `config.py` `data.py` `model.py` `engine.py` `stats.py` `run_all.py` | 训练/评测/统计主链路 |
| 根目录 | `check_data_all.py` | 数据完整性检查器 |
| 根目录 | `smoke_all.py` | 端到端冒烟（8 组 33 项） |
| 根目录 | `build_str_channels.py` `download_afdb_cif.py` `ssl_pretrain.py` `patch_pretrain.py` | 特征重建 / SSL 预训练（本包不需要跑） |
| 根目录 | `probe2.py` `probe_gz.py` `probe_konwledge_lr.py` | 探针脚本（可选） |
| `features/` | `store.py` `extract_seq_str.py` `extract_surf.py` `extract_text_genome.py` | 特征提取器（重建时才用） |
| `prep/` | 39 个文件（37 个 `.py`） | 全流程数据准备 |
| `mmseqs/` | MMseqs2 工具目录（约 40 MB） | 仅重建划分/聚类时用；**`bin/mmseqs` 是 Linux x86-64 ELF**，可直接用 |

### 2.2 数据（训练真正需要的全部数据）

| 文件 | 大小 | 说明 |
|---|---|---|
| `workdir/features/seq.npy` | 26.85 GiB | ESM-2 t33 650M 序列嵌入 `(11018,1022,1280)` float16 |
| `workdir/features/str.npy` | 21.48 GiB | ProstT5 结构字母表嵌入 `(11018,1022,1024)` float16 |
| `workdir/features/surf.npy` | 1.68 GiB | 表面几何 `(11018,1022,80)` float16 |
| `workdir/features/iface.npy` | 21.5 MiB | 界面倾向 `(11018,1022)` float16 |
| `workdir/features/plddt.npy` | 21.5 MiB | 每残基 pLDDT（**仅用于可用性掩码**，不进模型） |
| `workdir/features/text.npy` | 2.7 MiB | UniProt 文本 PCA `(11018,128)` |
| `workdir/features/genome.npy` | 5.4 MiB | 基因组窗口嵌入 `(11018,256)` |
| `workdir/features/lengths.npy` | 43 KiB | 真实长度 `(11018,)` int32 |
| `workdir/features/meta.json` | 496 KiB | `pid2idx` + 四类可用性掩码 + 通道清单 |
| `workdir/features/{uniprot_primary_map.txt,pids.txt,_*.txt,*.tsv}` | 2.2 MiB | 别名背书（`check_data_all.py` 要用）、失败清单、注释表 |
| `workdir/data/{train,val,test}.csv` | 4.4 MiB | 163,192 / 59,260 / 52,048 对 |
| `workdir/data/sequences*.fasta` | 17.9 MiB | 序列原文（重建时才用） |
| `workdir/ssl/adapter.pt` | 1.3 MiB | **R2–R7 必需**：SeqAdapter 预训练权重（与 Windows 版同一文件） |

合计 **50.05 GiB**，解包即「可直接训练」，**不需要联网、不需要重抽特征**。

### 2.3 未包含（及补法）

| 未包含 | 大小 | 为什么 | 怎么补 |
|---|---|---|---|
| `workdir/afdb/` CIF | 10.97 GB | 结构已抽成 `str/surf/iface/plddt.npy` | `python download_afdb_cif.py` |
| `workdir/uniprot/` | 0.28 GB | 文本已抽成 `text.npy` | `prep/fetch_files.py` + `prep/parse_uniprot.py` |
| `workdir/genome/` | 0.48 GB | 基因组已抽成 `genome.npy` | `prep/genome_from_ensembl_bulk.py`（需 GRCh38） |
| `features/` 下 76k 个散文件 | ≈22 GB | 构建残留，代码不读 | 无需 |
| UniRef50 / `uniref50_lt30.fasta` | 8.8 / 1.37 GB | 只用于重训 adapter，`adapter.pt` 已在包内 | `ssl_pretrain.py`（需 `fair-esm` + GPU） |
| `workdir/runs/` 旧产物 | 0.19 GB | 旧 CPU 结果 | 无 |

---

## 3. 解包

```bash
sudo apt install -y unzip            # Ubuntu Server 最小安装可能没有

# 解到主目录 → ~/ppi_full
unzip -q ppi_full_bundle_ubuntu_20261027.zip -d "$HOME"
cd ~/ppi_full
```

- 想放到独立 NVMe（推荐，避免 home 分区写满）：
  ```bash
  sudo mkdir -p /data && sudo chown "$USER":"$USER" /data
  unzip -q ppi_full_bundle_ubuntu_20261027.zip -d /data && cd /data/ppi_full
  ```
  **务必确认 `/data` 是 ext4/xfs**：`df -T /data`（`ntfs`/`exfat`/`cifs` 都不行，`mmap` 会坏）。
- 备选解包器（若 `unzip` 报 `bad zipfile offset` 之类）：
  ```bash
  sudo apt install -y libarchive-tools && bsdtar -xf ppi_full_bundle_ubuntu_20261027.zip -C "$HOME"
  # 或纯 Python：python3 -m zipfile -e ppi_full_bundle_ubuntu_20261027.zip "$HOME"
  ```
- 解完立刻校验（逐个比对 SHA-256，50 GB 约 3-6 分钟）：
  ```bash
  python3 verify_bundle.py                  # OK=149 BAD=0 MISSING=0
  python3 verify_bundle.py --quick           # 只比字节数，10 秒
  ```
- `unzip` 有概率不还原可执行位（取决于版本/umask），脚本里已兜底：
  ```bash
  chmod +x setup_ubuntu.sh RUN_ALL_EXPERIMENTS.sh mmseqs/bin/mmseqs
  ```

---

## 4. 环境安装

### 4.1 一条命令

```bash
bash setup_ubuntu.sh              # 建 .venv + torch(cu124) + requirements + 自检
bash setup_ubuntu.sh --cpu        # 没有 NVIDIA 卡时（只能跑 R0/R1 与检查）
bash setup_ubuntu.sh --smoke      # 装完直接跑 33 项冒烟（推荐）
bash setup_ubuntu.sh --dry-run    # 只打印命令，不改环境
```

### 4.2 手动等价步骤（想自己控制时）

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential unzip tmux

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 关键：CUDA 版 torch 单独装；不要和 requirements 混在一个 pip 命令里
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

> Ubuntu 24.04 的系统 Python 受 PEP 668 保护（`externally-managed-environment`），
> **必须用 venv**；`sudo pip install` 会直接报错，这是预期行为，不要用 `--break-system-packages` 绕过。

### 4.3 驱动（torch 轮子自带 CUDA runtime，不需要装 CUDA Toolkit）

```bash
nvidia-smi                                  # 能出表说明驱动就绪
sudo ubuntu-drivers autoinstall             # 或手动：sudo apt install nvidia-driver-550
sudo reboot
nvidia-smi | head -12                       # 应看到 RTX 4090 / 49140 MiB 左右
```

- 轮子 `cu124` 要求驱动 **≥ 550**（`nvidia-smi` 右上角 `CUDA Version: 12.4+`）。
- 改装/鼓风 4090 48 GB 若是 PCIe 版，`nvidia-smi -q | grep -i "bus id"` 正常即可；Secure Boot 开启时驱动签名需走 MOK 流程（`ubuntu-drivers` 会提示）。
- 不需要 `cuda-toolkit`、不需要 `nvidia-cuda-toolkit`、不需要 cuDNN 单独装。

### 4.4 验证

```bash
source .venv/bin/activate
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30,1))"
# 期望：2.x.x  12.4  True  NVIDIA GeForce RTX 4090  47.5 左右
python -c "import torch;print(torch.cuda.is_bf16_supported())"   # True（sm_89 支持 bf16）
```

---

## 5. 数据自检与冒烟测试

### 5.1 数据完整性

```bash
python check_data_all.py --root . --sample 512 --out-dir .
```

**预期：`PASS=49 WARN=5 FAIL=0`**（解包即为该数字）。5 条 WARN 都已知且无害：

1. `env | CUDA 不可用` —— 没装驱动的机器上必然出现（装了卡就消失）；
2. `meta.n | n=11018, pid2idx=11020` —— 2 组构建期别名，有 `uniprot_primary_map.txt` 背书；
3. `未被引用的键 | ['Q86VQ6']` —— 别名键，未被任何划分引用；
4/5. `text.npy / genome.npy dtype float16` —— 建议项，非错误。

**出现 `FAIL>0` 才需要处理**（最常见原因：`uniprot_primary_map.txt` 没解出来）。

### 5.2 冒烟测试：有必要做，而且是换平台后的必做项

**判定：有必要。** 但要点在于「在哪做」：

| 检查项 | 在 Windows 上能验证吗 | 结论 |
|---|---|---|
| Python 语法、模块导入、CLI 参数 | ✅ 已做 | 平台无关，无需重做 |
| 模型前向/反向、损失、评测、落盘 | ✅ 已做（33/33 通过） | 纯 PyTorch，平台无关 |
| **`DataLoader(num_workers>0)` 多进程 + `mmap`** | ❌ **从未真正跑过** | Windows 侧被沙箱拒（命名管道），一直降级到 `num_workers=0`。Linux 用 `fork`，是**唯一会真正执行这条路径**的地方 → 必须验 |
| **CUDA + `autocast(bfloat16)` 前向** | ❌ 本机是 CPU-only | GPU 代码路径一次都没执行过 → 必须验 |
| 换行/可执行位/路径分隔符 | ❌ 只能在 Linux 显现 | 已在打包侧静态修复，需实机确认 |
| ext4 上 26.85 GiB 文件的 `mmap` | ❌ | 必须验 |

所以：**包已完整 = 静态可验的都验了；功能正确性必须在目标机跑一次**，这也是 `setup_ubuntu.sh` 默认提示、`RUN_ALL_EXPERIMENTS.sh` 第一步自动执行的那一项。

```bash
source .venv/bin/activate
python smoke_all.py                 # 8 组 33 项：R0-R7 / 18 消融 / 搜索 / 基线 / stats / MC / 真实 DataLoader / 7 个 CLI
python smoke_all.py --fresh         # 先清 _smoke_runs 再跑
python smoke_all.py --only ladder   # 只跑某一组
```

**预期：33/33 ✅**，CPU 约 2-6 分钟，GPU 更快。报告写在 `smoke_report.md`，明细在 `_smoke_runs/smoke_summary.json`。

> 冒烟通过 ≠ 实验有效：它只证明「能建图、能前向反传、能评测、能落盘」，指标数值无科学含义（`bs=4`、1 epoch × 4 step）。
> 在 Ubuntu 上若 `DATAPATH` 组出现「num_workers=2 被拒→降级 0」，说明是权限/文件系统问题，不是代码问题，按 §9 处理。

---

## 6. 运行全部实验（逐条命令）

> 以下都在 `~/ppi_full` 下、`.venv` 已激活的状态执行。`export PYTHONUNBUFFERED=1` 可让日志实时落盘。

### 6.0 先测速（强烈建议）

```bash
python run_all.py ladder --levels R7 --seeds 42 --bs 64 --lr 1e-4
```

日志里每个 epoch 末尾有 `(NNNs)`，那就是单 epoch 墙钟时间。以它为锚算总时长：
`总小时 ≈ 118 × 平均epoch数 × T_epoch / 60`（`--seeds 42` 时约 66 个 run）。

- `T_epoch` 2.5-4 min → 全套约 4-7 天
- `T_epoch` > 6 min → 数据没进 page cache 或 `num_workers` 太小（见 §9）
- `--trials 10` + `--seeds 42`（≈`--quick`）→ 约 1.5-3 天

### 6.1 R0–R7 阶梯（核心表）

```bash
python run_all.py ladder --levels R0 R1 R2 R3 R4 R5 R6 R7 --seeds 42 1234 2024 --bs 64 --lr 1e-4
# 单 seed 快查：
python run_all.py ladder --levels R0 R1 R2 R3 R4 R5 R6 R7 --seeds 42 --bs 64 --lr 1e-4
# 复用搜索到的最优超参：
python run_all.py ladder --levels R0 R1 R2 R3 R4 R5 R6 R7 --seeds 42 1234 2024 \
       --hyperparams workdir/runs/best_hyperparams.json
```

### 6.2 18 项消融（基座固定 R7）

```bash
python run_all.py ablation --seeds 42 1234 2024 --bs 64 --lr 1e-4 \
       --hyperparams workdir/runs/best_hyperparams.json

# 只跑几项
python run_all.py ablation --names L4-unidirectional L5-mixup No-SSL --seeds 42
```

可用名字（18 个，必须完全一致，拼错会直接报错并列出可选项）：
`L4-unidirectional` `L4-no-iface-bias` `L4-no-siface` `L4-no-surface-gate` `L4-no-selfattn-ctx`
`L4-alpha-to-add` `L4-global-concat-nosym` `L4-no-global-path` `L4-proj-seq-early`
`L5-no-asym` `L5-no-siface` `L5-plain-mlp` `L5-mixup` `L5-no-ls` `L5-no-rdrop` `L5-concat-asym`
`No-Seq` `No-SSL`

### 6.3 超参随机搜索（R7）

```bash
python run_all.py search --trials 40
```

产物 `workdir/runs/best_hyperparams.json`（`best_overrides` 可直接喂给 `--hyperparams`）。
注意：`search` 内部固定 `bs=16`、`lr` 从搜索空间里选，**不接受 `--bs/--lr`**。

### 6.4 基线

```bash
python run_all.py baseline            # RFC-mean / RFC-40，约 10-40 分钟
```

Linux 上 `n_jobs=-1` 会占满所有核（joblib/loky，163k×1280 float32 约 0.8 GB/进程），128 GB 内存无压力。

### 6.5 统计汇总（bootstrap CI + Wilcoxon + 配对 Δ）

```bash
python run_all.py stats --tags R0 R1 R2 R3 R4 R5 R6 R7 --baseline R0
cp workdir/runs/summary.csv workdir/runs/summary_ladder.csv

python run_all.py stats --tags L4-unidirectional L4-no-iface-bias L4-no-siface L4-no-surface-gate \
  L4-no-selfattn-ctx L4-alpha-to-add L4-global-concat-nosym L4-no-global-path L4-proj-seq-early \
  L5-no-asym L5-no-siface L5-plain-mlp L5-mixup L5-no-ls L5-no-rdrop L5-concat-asym No-Seq No-SSL \
  --baseline R7
cp workdir/runs/summary.csv workdir/runs/summary_ablation.csv
```

`summary.csv` 每次被覆盖，所以必须 `cp` 另存。统计只汇总**实际存在**的 seed。

### 6.6 MC-Dropout 不确定性

```bash
python run_all.py mc --tags R7 --seeds 42 1234 2024 --T 10
```

产物 `workdir/runs/R7/seed42/{mc_dropout.npz,mc_metrics.json}`（含 `spearman_var_abserr`）。

### 6.7 一键全跑

```bash
bash RUN_ALL_EXPERIMENTS.sh                      # search → ladder → ablation → baseline → stats → mc
bash RUN_ALL_EXPERIMENTS.sh --quick              # 1 seed + 10 trial，约 1.5-3 天
bash RUN_ALL_EXPERIMENTS.sh --skip-search        # 跳过搜索
bash RUN_ALL_EXPERIMENTS.sh --dry-run            # 只看命令，不执行
```

脚本把所有 stdout 同时写进 `logs/bundle_run_<时间戳>.log`；任一步失败即中止并打印退出码
（`mc` 是派生量，失败只警告不中止）。

---

## 7. 长跑：别让 SSH 断线杀掉 5 天的任务

**必须**放到 `tmux`（或 `nohup`）里。`run_all.py` 没有断点续跑，进程一死，当前 run 白跑。

```bash
sudo apt install -y tmux

tmux new -s ppi
# 在 tmux 里：
cd ~/ppi_full && source .venv/bin/activate
bash RUN_ALL_EXPERIMENTS.sh
# 按 Ctrl-b 再按 d 脱离；断线不受影响
```

回来看进度：

```bash
tmux attach -t ppi            # 重新进入
tail -f logs/bundle_run_*.log # 或直接看日志
watch -n 5 nvidia-smi         # 另一个窗口看 GPU
```

不想用 tmux 的话：

```bash
nohup bash RUN_ALL_EXPERIMENTS.sh > logs/nohup_$(date +%F).log 2>&1 &
echo $! > logs/ppi.pid
```

另外两个容易忘的：

```bash
# 1) 防止系统休眠（桌面版尤其重要）
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
# 2) 4090 降功耗换稳定性（可选，性能损失约 5%）
sudo nvidia-smi -pl 450
```

---

## 8. 产物在哪、怎么看

```
workdir/runs/
├── R0/seed42/{cfg.json, metrics.json, ckpt.pt, best_ckpt.pt, predictions.npz}
├── R7/seed42/{..., mc_dropout.npz, mc_metrics.json}
├── L4-unidirectional/seed42/...
├── RFC-mean/seed42/...
├── best_hyperparams.json
├── summary.csv
└── summary_ladder.csv / summary_ablation.csv
```

- `metrics.json`：`test` 与 `test_calibrated` 两套九指标（auprc/auroc/acc/precision/recall/f1/mcc/brier/ece）+ `T` + `best_epoch` + `val_auprc`。
- `predictions.npz`：`probs, probs_cal, logits, labels, T, pid_a, pid_b`（test 顺序固定，可跨 run 逐位配对）。读取时**必须** `np.load(..., allow_pickle=True)`。
- `summary.csv`：每 tag 的 seed 均值±std、bootstrap 95% CI、与 baseline 的配对 Δ 与 Wilcoxon p。
- 参数量：R0 1,045,569；R1 3,823,874；R2–R7 4,155,522。

---

## 9. Ubuntu 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `bash: $'\r': command not found` | 脚本被 CRLF 污染（用 Windows 记事本存过） | `sed -i 's/\r$//' *.sh`；本包内脚本已是 LF |
| `Permission denied`（`./RUN_ALL_EXPERIMENTS.sh`） | 无可执行位 | `bash RUN_ALL_EXPERIMENTS.sh` 或 `chmod +x *.sh` |
| `externally-managed-environment` | Ubuntu 23.04+ 的 PEP 668 | 用 `.venv`，别 `sudo pip` |
| `No module named venv` | 缺 `python3-venv` | `sudo apt install -y python3-venv` |
| `torch.cuda.is_available()` 为 False | 驱动太旧 / 装了 CPU 轮子 | `nvidia-smi` 看 `CUDA Version` ≥12.4；`pip show torch` 里应是 `+cu124` |
| `CUDA error: no kernel image is available` | 轮子不支持 sm_89 | 装 `cu124`（或更新）轮子 |
| `Bus error` / `Segmentation fault` 读 `.npy` | 包放在 NTFS/exFAT/SMB/NFS 上 | 换到本地 ext4/xfs，`df -T` 确认 |
| `OSError: [Errno 12] Cannot allocate memory` | `num_workers` 过大 + 内存压力 | `config.py:82` 改 `num_workers: int = 0` 或减到 2 |
| `DataLoader worker killed by signal: Bus error` | worker 里 `mmap` 失败 | 同上，先把 `num_workers=0` 跑通 |
| 单 epoch > 6 min | 数据没进 page cache；`num_workers=2` 太小 | 预热：`python -c "import numpy as np;a=np.load('workdir/features/seq.npy',mmap_mode='r');a[0,0].sum()"`；`num_workers` 调到 4-6 |
| `CUDA out of memory` | `--bs` 太大 | 降到 32；48 GB 上 bs=64 正常应只占 8-10 GB |
| 日志/输出中文乱码 | `LANG` 是 POSIX | `export LANG=C.UTF-8` |
| `stats` 报 `FileNotFoundError ... predictions.npz` | 该 run 没跑完（只有 `cfg.json`） | 重跑该 tag/seed，或从 `--tags` 去掉 |
| `RuntimeError: ... 特征 'struct' 缺失或全零` | 特征库没解包全 | 确认 `workdir/features/str.npy` 存在且 >20 GiB |
| `ValueError: Object arrays cannot be loaded...` | 读 `predictions.npz` 少了 `allow_pickle` | 加 `allow_pickle=True` |
| 中断后想续跑 | `run_all.py` 会**覆盖**同 tag/seed | 先 `cp -r workdir/runs workdir/runs_bak` |

---

## 10. 完整性校验

```bash
python3 verify_bundle.py                     # 逐文件 SHA-256
sha256sum ppi_full_bundle_ubuntu_20261027.zip   # 与 .sha256 边文件比对
```

`MANIFEST.sha256` 每行：`<sha256>  <bytes>  <相对路径>`；`manifest.json` 另存结构化清单（通道形状、样本数、生成时间）。

---

## 11. 读结果前必须知道的口径

1. **数据缺口已掩码**：103 个蛋白无 AFDB 结构（pLDDT 写 100.0 哨兵，surf/iface 全零 + `has_surf/has_iface=0` 掩码），186 个无基因组窗口（`has_genome=0`）。受影响 pair 占 2.38%。
2. **`L4-no-iface-bias` 训练早期不可辨识**：`beta_init=0.0` 使 `bias = beta * iface_t` 近乎恒 0，要归因必须真实规模训练（≥10 epoch）。
3. **`L4-no-global-path` 与 R5 同口径**：R5 本就没有 text/genome 全局通路，两者指标接近属预期。
4. **`L5-mixup` 是「开 mixup」正向消融**（`mixup_prob=0.25`），基座 R7 默认 mixup=0。
5. **`struct` 开关的实际通道是 `str`**：`run_all.py` 里 `_CH_ALIAS={"struct":"str"}`。
6. **`num_workers`**：Windows 侧沙箱会拒（命名管道）；Linux 正常，建议 4-6。

---

## 12. 复现核对清单

- [ ] `python3 verify_bundle.py` → `OK=149 BAD=0 MISSING=0`
- [ ] `bash setup_ubuntu.sh` 最后打印 CUDA 可用、GPU 47.5 GiB
- [ ] `python smoke_all.py` → **33/33**
- [ ] `python check_data_all.py --root .` → `FAIL=0`
- [ ] `python run_all.py ladder --levels R0 --seeds 42` → test AUPRC 落在 0.60-0.66（旧 CPU 实测 0.6175/0.6512/0.6330）
- [ ] `workdir/runs/R7/seed42/metrics.json` 存在且 `test.auprc` 有限
- [ ] `python run_all.py stats --tags R0 R1 --baseline R0` 能出 `summary.csv`
