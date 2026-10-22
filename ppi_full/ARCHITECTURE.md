# ARCHITECTURE.md — ppi_full 架构与代码地图

> 代码版本：v2.3 ｜ 文档生成：2026-10-22 ｜ 适用根目录：`ppi_full/`（含 `workdir/`）
> 本文档描述"代码是什么、怎么串起来、哪里会踩坑"。所有路径均相对仓库根目录。

---

## 0. TL;DR

| 项 | 内容 |
|---|---|
| 任务 | 人类蛋白–蛋白互作（PPI）二分类，输出 0–1 概率 |
| 数据 | Bernett 基准：11,018 个蛋白、274,500 个配对；train/val/test 蛋白级不相交 |
| 主指标 | **AUPRC**（测试集，负样本经度平衡，随机基线 0.5） |
| 骨架 | `L0 数据 → L2 质量门 → L3 SSL 适配器 → L4 双通路融合 → L5 对称头`，外加 `R0 复现锚点` |
| 实验设计 | **R0–R7 阶梯**（逐级多开一个特征开关）+ **18 项消融** + 超参搜索 + 统计检验 |
| 模型规模 | 全量 4,155,522 参数（L3 331,648 + L4/L5 3,823,874）；R0 锚点 1,045,569 |
| 特征库 | 5 通道：seq(1280) / str(1024) / surf(80) / text(128) / genome(256) + iface/pLDDT（逐残基） |

一句话：**这是一套"证明每一层、每个特征都有增量"的消融实验框架**，而不是一个追求 SOTA 的单体模型。因此代码里大量篇幅在**防作弊**（泄漏断言、非平凡性检查、知识基线探针、打乱对照）。

---

## 1. 文件总表

共 **46 个 `.py`**（外加 shell 脚本 [`prep/ssl_prepare.sh`](prep/ssl_prepare.sh)）。
按职责分四组：核心、特征工程、验收/探针、一次性补丁。

### 1.1 核心（6 个文件，读懂这 6 个就读懂了项目）

| 文件 | 层 | 一句话职责 |
|---|---|---|
| [`config.py`](config.py) | 全局 | 全部 `dataclass` 配置 + `R_LEVELS` 阶梯 + `ABLATIONS` 消融表；`train__` 前缀区分模型超参与训练超参 |
| [`data.py`](data.py) | L0/L2 | csv→配对表、泄漏断言、`FeatureStore`（mmap 只读）、`PairDataset`、`make_collate`（动态 padding + 质量门）、`BucketSampler` |
| [`model.py`](model.py) | L3/L4/L5 | `SeqAdapter`、`L4Fusion`、`L5Head`、`Reim2D`（R0 锚点）、`PPIModel` 总装 |
| [`engine.py`](engine.py) | 训练/评测 | 训练协议、损失栈（LS/mixup/R-Drop）、早停、温度标定、9 项指标、MC-Dropout |
| [`stats.py`](stats.py) | 统计 | Wilcoxon / Cohen's dz / McNemar / Stouffer / bootstrap CI |
| [`run_all.py`](run_all.py) | 编排 | 6 个子命令：`ladder` `ablation` `search` `baseline` `stats` `mc` |

### 1.2 特征工程

| 文件 | 职责 | 产物 |
|---|---|---|
| [`features/store.py`](features/store.py) | **特征库写入端**（memmap 建库）+ `--init/--collect` CLI | `{seq,str,surf,iface,text,genome}.npy`、`lengths.npy`、`plddt.npy`、`meta.json` |
| [`features/extract_seq_str.py`](features/extract_seq_str.py) | ESM-2 t33 (1280d) / ProstT5 (1024d) 逐残基嵌入 | `seq_{pid}.npy` / `str_{pid}.npy` |
| [`features/extract_surf.py`](features/extract_surf.py) | AFDB CIF → pLDDT + 80 维几何描述子 + iface | `surf_/iface_/plddt_{pid}.npy` |
| [`features/extract_text_genome.py`](features/extract_text_genome.py) | UniProt 文本 → 嵌入 → **PCA 128d**；genome 256d | `text_{pid}.npy` / `genome_{pid}.npy` |
| [`ssl_pretrain.py`](ssl_pretrain.py) | 冻结 ESM-2，只训 adapter 的 MLM 预训练 | `workdir/ssl/adapter.pt` |

### 1.3 `prep/` — 数据准备流水线（28 个文件，按用途分）

**下载**

| 文件 | 职责 |
|---|---|
| [`prep/make_url_lists.py`](prep/make_url_lists.py) | `pids.txt` → AFDB / UniProt 两组 URL 清单 |
| [`download_afdb_cif.py`](download_afdb_cif.py) | AFDB 结构下载：**v6→v5→v4 逐版本回退** + 429 退避；落盘恒为 v4 文件名 |
| [`prep/parallel_fetch.py`](prep/parallel_fetch.py) | 通用断点续跑下载器（`.part` 原子替换、指数退避、失败清单） |
| [`prep/fetch_files.py`](prep/fetch_files.py) | AFDB 专用旧版下载器（**已被上面两者取代**） |
| [`prep/fetch_fasta.py`](prep/fetch_fasta.py) | 人参考蛋白组 gz → 按 pid 过滤 → 同工型映射 → REST 兜底 |

**解析 / 修数据**

| 文件 | 职责 |
|---|---|
| [`prep/parse_uniprot.py`](prep/parse_uniprot.py) | 8 进程解析 `uniprot/*.txt` → `features_annot.tsv` + `uniprot_primary_map.txt`（次级→主登录号） |
| [`prep/parse_structures.py`](prep/parse_structures.py) | 手写 CIF 解析（不依赖 Bio.PDB）→ `struct_npz/{pid}.npz`（`seq`/`plddt`/`ca`）+ `features_struct.tsv` |
| [`prep/fix_len_zero.py`](prep/fix_len_zero.py) | `length=0` 条目重抓重解析 |
| [`prep/fix_deleted.py`](prep/fix_deleted.py) | 已删除条目走 UniSave 捞历史版本，仍失败记 `_uniprot_dead_accessions.txt` |
| [`prep/join_features.py`](prep/join_features.py) | annot + struct 按主登录号 join → **`features_node.tsv`（全库主键表）** |

**建库 / 提取**

| 文件 | 职责 |
|---|---|
| [`prep/build_store_meta.py`](prep/build_store_meta.py) | 骨架库：`meta.json` + `lengths.npy` + `plddt.npy` |
| [`prep/preflight_seq.py`](prep/preflight_seq.py) | 序列预检：fair-esm 可用性、header 格式（pipe/plain）、补漏、校正 `lengths.npy` |
| [`prep/run_seq_extract.py`](prep/run_seq_extract.py) | 断点续跑地调 ESM-2 提取（可跳过已完成的散文件） |
| [`prep/build_seq_store.py`](prep/build_seq_store.py) | **只建 seq 通道**的增量建库 + 自带两阶段验收（Phase1 collect / Phase2 verify） |
| [`prep/geo_worker.py`](prep/geo_worker.py) | 几何描述子的多进程并行版（单条异常记账不炸池） |
| [`prep/ssl_prepare.sh`](prep/ssl_prepare.sh) | SSL 语料制备：UniRef50 1/8 抽样 → MMseqs2 30% 聚类 → 剔除与 Bernett ≥30% 相似 |

**验收 / 探针**

| 文件 | 职责 |
|---|---|
| [`check_data_all.py`](check_data_all.py) | **数据资产总检**（5 层、PASS/WARN/FAIL、退出码即门禁、输出 `data_check_report.{md,json}`） |
| [`prep/verify_store2.py`](prep/verify_store2.py) | **当前应使用**的库验收：fp16 存储合同 + 三 split + collate 冒烟 |
| [`prep/verify_store.py`](prep/verify_store.py) | 旧验收：**硬断言 str/surf/iface 存在**（缺通道时会直接崩） |
| [`prep/verify_store_orig.py`](prep/verify_store_orig.py) | 与上者内容完全相同的备份 |
| [`prep/qa_splits.py`](prep/qa_splits.py) | 三 split 行数/自环/重复对/泄漏/配对重叠报告 |
| [`prep/smoke_r0.py`](prep/smoke_r0.py) | CPU 秒级 R0 全链路契约测试（store→collate→Reim2D→loss→backward→_predict） |
| [`prep/forensics_nan.py`](prep/forensics_nan.py) | NaN 取证：散文件 / 库 / npy 头字节三路对照 |
| [`prep/probe_store_read.py`](prep/probe_store_read.py) | "读取层定罪"：确认 memmap 没有把 npy 头当数据读 |
| [`prep/run_knowledge_probe.py`](prep/run_knowledge_probe.py) | 知识基线 A：只用注释/结构**标量**训 LR |
| [`probe_konwledge_lr.py`](probe_konwledge_lr.py) | 知识基线 B：只用 store 的 text/genome 拼配对特征训 SGD + 打乱对照 |
| [`probe_gz.py`](probe_gz.py) / [`probe2.py`](probe2.py) | UniRef50 gz 完整性校验（probe2 修掉了 v1 只验第一段的盲区） |

### 1.4 一次性补丁（**不要执行**，均已生效）

这 8 个文件（`prep/` 下 7 个 + 根目录 1 个）是对源码做字节级替换的历史热修，自带"锚点唯一"保护：

[`prep/patch_store.py`](prep/patch_store.py)、[`prep/patch_data_mm.py`](prep/patch_data_mm.py)、[`prep/patch_collate_pickle.py`](prep/patch_collate_pickle.py)、[`prep/patch_collate_v2.py`](prep/patch_collate_v2.py)、[`prep/patch_collate_channels.py`](prep/patch_collate_channels.py)、[`prep/patch_engine_makedirs.py`](prep/patch_engine_makedirs.py)、[`prep/patch_logits_shape.py`](prep/patch_logits_shape.py)、[`patch_pretrain.py`](patch_pretrain.py)。

它们对应的**代码注释编号**（`P0-x / P1-6 / N1-N7 / D1-D5 / B3 / C2`）就是缺陷登记号，读源码时遇到别当成变量名。

### 1.5 数据目录

```
workdir/
├── data/        train.csv val.csv test.csv sequences.fasta sequences_primary.fasta truncation_report.json
├── raw/         Intra*_pos_rr.txt / Intra*_neg_rr.txt（Bernett 原始互作文件）
├── urls/        afdb_urls.txt uniprot_urls.txt
├── afdb/        {PID}.cif 或 AF-{pid}-F1-model_v4.cif   ← 两种命名并存，见 §7.6
├── uniprot/     {PID}.txt（UniProt 平文本，文本通道与注释的来源）
├── features/    特征库（§3）：*.npy + meta.json + 中间散文件 seq_*.npy 等
├── ssl/         uniref50_lt30.fasta + adapter.pt
└── runs/        {tag}/seed{42,1234,2024}/{cfg.json, ckpt.pt, metrics.json, predictions.npz}
```

---

## 2. 分层架构

### 2.1 总览

```
                    ┌──────────────── R 阶梯（消融主线）────────────────┐
                    │ R0 锚点  R1 主体  R2 +SSL  R3 +struct  R4 +iface  │
                    │ R5 +surf          R6 +text          R7 +genome    │
                    └───────────────────────┬──────────────────────────┘
                                            │
  csv ──► L0 配对表 ──► FeatureStore ──► collate ──► L3 adapter ──► L4 融合 ──► L5 头 ──► logit
          (P7 泄漏断言)   (mmap 41GB)   (padding+质量门)  (1280→128→1280)  (双通路+交叉注意力) (对称+不对称)
```

### 2.2 L0 数据层 + L2 质量门 — [`data.py`](data.py)

- `load_pairs()`：列名别名归一（`protein1/uniprot_a/…` → `pid_a`），可选时间截断（B3）。
- `validate_splits()`：**P7 断言**——任意两个 split 的蛋白集合必须零交集，否则抛 `LEAKAGE`。
- `FeatureStore`：全部通道用 `np.load(mmap_mode="r")` 打开（41 GiB 不占内存），并**校验形状**。
- `PairDataset`：把 pid 映射成行号；若有 pid 不在库里，断言失败并提示重跑提取。
- `make_collate()`：一个 batch 内按 `max(La,Lb)` 动态定长（向上取 8 的倍数），缺通道零填充；
  逐残基的 `len`/`pad`/`iface`/`plddt` 同步裁剪。
- **L2 质量门（无参数）**：`pLDDT < 70` 的残基，其 `surf` 与 `iface` 直接置零——
  即"低置信结构不贡献表面/界面证据"。缺结构时 pLDDT 被填 100，等于门控关闭。
- `BucketSampler`：按 `max(La,Lb)` 稳定排序后切桶，桶内成批、批序每 epoch 打乱（确定性可复现）。

### 2.3 L3 SSL 适配器 — [`model.SeqAdapter`](model.py) + [`ssl_pretrain.py`](ssl_pretrain.py)

瓶颈残差：`1280 → 128 → 1280`（GELU + LayerNorm），331,648 参数。

预训练：ESM-2 t33 主体**冻结**，只在 UniRef50（已剔除与 Bernett ≥30% 相似）上做 MLM（80/10/10 mask），
反向只流经 adapter + MLM 头。产出 `workdir/ssl/adapter.pt`，训练时**双侧共享**同一个 adapter（保证对称性，P1-6）。

### 2.4 L4 双通路融合 — [`model.L4Fusion`](model.py)

```
_embed(单个蛋白):
  seq ──Linear──┐
               ├─(α 加权融合)─► u  [B,L,256]
  str ──Linear──┘   α=sigmoid(可学习向量)
  surf ─(tanh·sigmoid 门控)─► 加在 seq 上（surface_gate）

_interact(A→B):
  交叉注意力(cross-attn)：Q=A, KV=B，可加 iface 偏置 β·iface[τ 阈值化]
  从注意力权重抠出界面证据 s = Σ_k w·iface  （siface_out）
  再接 self-attn（selfattn_ctx）→ masked mean → pair_mlp([p_s, p_t, p_s-p_t, p_s*p_t])
  → z_ab [B,256]，s_ab [B,1]

双向：z_ba 同理（bidirectional 开关 = L4-unidirectional 消融）

全局通路（text/genome）：
  sym  : [ta+tb, ta*tb, |ta−tb|, ga+gb, ga*gb, |ga−gb|] → [B,6d] → MLP → g，两侧共享
  asym : [tp, gp, tq, gq]                                → [B,4d] → MLP
```

### 2.5 L5 对称头 — [`model.L5Head`](model.py)

```
z_sym = (z_ab+z_ba)/2 ;  z_asym = |z_ab−z_ba| ;  s = (s_ab+s_ba)/2
z_in  = [z_sym, z_asym, s]  = [B, 513]
 → Linear(513,256) → 2×PreNormBlock → LayerNorm → Linear(256,1) → /T
```
`T` 是 buffer，训练期恒 1；**校准一律 post-hoc**（验证集拟合温度，不参与训练）。

### 2.6 R0 复现锚点 — [`model.Reim2D`](model.py)

共享编码器（1280→640→320→64）+ 分块外积（chunk=8，控显存）+ 1×1 conv + **masked max**。
它是"不用预训练、不用结构、不用文本"的复现基线，用来回答"后面的增量是不是真的来自我们的设计"。

### 2.7 R 阶梯与消融矩阵 — [`config.py`](config.py)

| 级别 | kind | SSL | struct | iface | surf | text | genome | 含义 |
|---|---|---|---|---|---|---|---|---|
| R0 | reim | – | – | – | – | – | – | 复现锚点 |
| R1 | l4 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 纯序列主体 |
| R2 | l4 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | + SSL 适配器 |
| R3 | l4 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | + 结构 |
| R4 | l4 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | + 界面 |
| R5 | l4 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | + 表面 |
| R6 | l4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | + 文本 |
| R7 | l4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 全量 |

消融 18 项，覆盖：L4 结构（单向/去 bias/去 siface/去表面门控/去 self-attn/α 改加法/去对称/去全局通路/早投影）、
L5 结构（去不对称/去 siface/plain MLP/去 mixup/去 LS/去 R-Drop/改 concat）、
以及反向消融 `No-Seq`（把序列置零）与 `No-SSL`。

> **参数量守恒**是硬约束：`l5_plain_mlp` 也用常驻构造，切换时参数量不变（N4）。

### 2.8 张量形状速查

| 张量 | 形状 | 说明 |
|---|---|---|
| `batch["A"]["seq"/"str"/"surf"]` | `[B,Lmax,1280/1024/80]` | Lmax ≤ 1022，8 的倍数 |
| `batch["A"]["iface"/"plddt"/"pad"]` | `[B,Lmax]` | pad 为 True 表示 padding |
| `batch["A"]["text"/"genome"]` | `[B,128]` / `[B,256]` | 蛋白级 |
| `batch["A"]["text_mask"/"genome_mask"]` | `[B]` | 0/1，缺失即 0 |
| `u`（L4 嵌入） | `[B,L,256]` | d_model=256 |
| `z_ab`,`z_ba` / `s_ab`,`s_ba` | `[B,256]` / `[B,1]` | |
| `logits` | `[B,1]` | 下游统一 `squeeze(-1)`（N1） |

---

## 3. 特征库契约

### 3.1 数组布局（`workdir/features/`）

| 文件 | 形状 | dtype | 语义 |
|---|---|---|---|
| `seq.npy` | `(N, 1022, 1280)` | float16 | ESM-2 逐残基嵌入 |
| `str.npy` | `(N, 1022, 1024)` | float16 | ProstT5 逐残基嵌入 |
| `surf.npy` | `(N, 1022, 80)` | float16 | 几何描述子 |
| `iface.npy` | `(N, 1022)` | float16 | 界面概率（0–1 或 logit 型分数） |
| `plddt.npy` | `(N, 1022)` | float16 | 逐残基 pLDDT（0–100） |
| `text.npy` | `(N, 128)` | float16 | UniProt 文本嵌入（PCA 到 128） |
| `genome.npy` | `(N, 256)` | float16 | 基因组上下文嵌入 |
| `lengths.npy` | `(N,)` | int | 每蛋白真实残基数（≤1022，已截断） |
| `meta.json` | – | – | `{pid2idx, n, has_text[], has_genome[]}` |

**硬约定**

1. 第 `i` 行的 `[:lengths[i]]` 是有效数据，其余必须为 0（`check_data_all.py` 会采样检查）。
2. `has_text[i] == 1` ⟺ `text[i]` 非全零；`has_genome` 同理。
3. `pid2idx` 允许**别名**（次级登录号指向同一主登录号的行），但被划分引用的重复索引必须能由
   `uniprot_primary_map.txt` 背书，否则视为特征错位。
4. 超过 1022 的蛋白**截断**（如 titin），`truncation_report.json` 记录明细。

### 3.2 80 维描述子的字段分配（[`features/extract_surf.py`](features/extract_surf.py)）

| 下标 | 含义 |
|---|---|
| 0–7 | 8/10/12/…/18 Å 邻域残基计数 |
| 8–13 | 12 Å 邻域惯量张量的主轴分布、planarity、形状因子 |
| 14–17 | 主链键角、二面角的 sin/cos |
| 18 | pLDDT/100 |
| 19–27 | 8/12/16 Å 内疏水/其他/带电占比 |
| 28–32 | 残基类别 one-hot（芳香/疏水/其他/正电/负电） |
| 33–48 | 位置周期性编码（sin/cos） |
| 49–52 | 8 Å 邻域距离的 25/50/75/100 分位 |
| 53–54 | 12 Å 邻域沿主轴投影的均值/标准差 |
| 56–59 | 邻近残基的序列间隔统计 |
| 60–61 | 8 Å 邻域 pLDDT 均值/标准差 |
| 62 | 回转半径（归一化） |
| 63–64 | 8 Å 内芳香/负电占比 |
| 65–66 | 全蛋白距离均值/标准差 |
| 67 | 8 Å 邻域密度（归一化） |
| 68 | 相对序列位置 i/(L−1) |
| 69–78 | 预留（恒 0） |
| 79 | 常数 1（偏置位） |

---

## 4. 端到端流水线（runbook）

> 全部命令在**仓库根目录**执行（所有脚本用相对路径 `workdir/...`）。

| # | 命令 | 产物 | 验收 |
|---|---|---|---|
| 0 | 准备 Bernett 原始文件 → `data.py --in --out` 规范化 | `workdir/data/{train,val,test}.csv` | `python prep/qa_splits.py` |
| 1 | `python prep/make_url_lists.py --pids pids.txt --outdir workdir/urls` | 两组 URL 清单 | – |
| 2 | `python download_afdb_cif.py --pids workdir/features/pids.txt` ／ `python prep/parallel_fetch.py --urls workdir/urls/uniprot_urls.txt --outdir workdir/uniprot --workers 6` | `afdb/*.cif`、`uniprot/*.txt` | 看 `_dl_failed.txt` |
| 3 | `python prep/parse_uniprot.py` → `python prep/parse_structures.py` → `python prep/join_features.py` | `features_annot.tsv`、`struct_npz/`、`features_node.tsv` | `_uniprot_dead_accessions.txt` 应为空 |
| 4 | `python prep/build_store_meta.py` | 骨架 `meta.json`/`lengths.npy`/`plddt.npy` | 打印的 `missing` 应为 0 |
| 5 | `python prep/preflight_seq.py` | `sequences_primary.fasta` | 以 `READY` 结尾 |
| 6 | `python prep/run_seq_extract.py 3` | `seq_{pid}.npy` | `EXTRACT_SEQ_DONE files_ok=11018/11018` |
| 7 | `python features/extract_seq_str.py --kind str --pids ... --fasta ...` | `str_{pid}.npy` | 每文件形状 `(L,1024)` |
| 8 | `python prep/geo_worker.py --pids ... --cif-dir workdir/afdb --workers 12` | `surf_/iface_/plddt_{pid}.npy` | `_geo_failed.txt` 应为空 |
| 9 | `python features/extract_text_genome.py --kind text --pids ... --train-pids train_pids.txt` | `text_{pid}.npy` | P8：PCA 只在 train 蛋白上 fit |
| 10 | `python prep/build_seq_store.py`（仅 seq）或 `python features/store.py --collect`（全通道） | 各通道 `.npy` | ⚠️ 见 §7.5 |
| 11 | `python prep/verify_store2.py` | 验收输出 | 以 `STORE VERIFIED` 结尾 |
| 12 | `bash prep/ssl_prepare.sh` → `python ssl_pretrain.py` | `workdir/ssl/adapter.pt` | 参数量 331,648 |
| 13 | `python check_data_all.py --runs` | `data_check_report.{md,json}` | 退出码 0 |
| 14 | `python run_all.py ladder --levels R0 R1 … R7` | `workdir/runs/R*/seed*` | `metrics.json` |
| 15 | `python run_all.py stats --tags R1 R7 --baseline R1` | `runs/summary.csv` | – |

---

## 5. 训练与评测协议

### 5.1 训练（[`engine.train_one_run`](engine.py)）

| 组件 | 设置 |
|---|---|
| 优化器 | AdamW，`lr=5e-5`（8 GB 默认；24 GB+ 用 `1e-4`），`wd=1e-6` |
| 调度 | 5% 线性 warmup + 余弦退火 |
| 精度 | bf16 autocast（CUDA）+ 梯度裁剪 1.0 |
| 批大小 | 16（8 GB）；分桶采样控制显存峰值 |
| 早停 | 监控 val AUPRC，`patience=8`，保存最优 `best_ckpt.pt` |

### 5.2 损失栈

```
L = BCEWithLogits(logit, smooth(y; ε=0.05))                    # 标签平滑
  [+ mixup]  以 p 概率做 mixup，perm/λ 每 step 采样一次、双前向共享(D1)
  [+ R-Drop] L → ½(BCE₁+BCE₂) + α·½(KL(p₁‖p₂)+KL(p₂‖p₁))       # α=1.0
```
`R0(Reim2D)` 自动关闭 label smoothing 与 R-Drop（保持"朴素复现"语义）。

### 5.3 评测

- **指标**：AUPRC、AUROC、ACC、Precision、Recall、F1、MCC、Brier、ECE(15 bins)。
- **温度标定**：`fit_temperature()` 在 **val** 上用 LBFGS 拟合单参数 logT，再作用到 test；
  `metrics.json` 里同时给 `test` 与 `test_calibrated`。AUPRC 对校准不敏感，ECE/Brier 敏感。
- **MC-Dropout**：`mc_dropout_eval()` 只把 `nn.Dropout` 切回 train，做 T=10 次前向，
  报告均值指标与 `spearman(var, |err|)`（不确定性能否预测错误）。

### 5.4 统计检验（[`stats.py`](stats.py)）

| 检验 | 用途 |
|---|---|
| Wilcoxon signed-rank (exact) | 跨 seed 配对比较两个 tag |
| Cohen's dz | 效应量（配对） |
| McNemar (exact) + Stouffer | 同 seed 的逐样本对错差异，跨 seed 合并 |
| Bootstrap 1000× | AUPRC 的 95% CI |

> 前提：`predictions.npz` 的 `labels` 顺序在 seed 间一致（`load_runs` 会断言）。

---

## 6. 实验编排 — [`run_all.py`](run_all.py)

| 子命令 | 作用 | 关键参数 |
|---|---|---|
| `ladder` | 跑 R 阶梯 | `--levels --seeds --lr --bs --hyperparams` |
| `ablation` | 跑 18 项消融（基座固定 R7） | `--names` |
| `search` | R7 上随机搜索 6 维超参 | `--trials`（8 GB 建议 10–15） |
| `baseline` | RFC-mean / RFC-40（sklearn） | – |
| `stats` | 汇总 + 检验 + `summary.csv` | `--tags --baseline` |
| `mc` | MC-Dropout 评测 | `--tags --T` |

`run_tagged()` 里有两个**发车前置检查**：
- **P10 非平凡性**：`switch_x=1` 但对应特征缺失或全零 → 直接抛错（防止 zero-provider 假实验）。
- **D4 权重存在性**：`use_ssl=True` 但 `adapter.pt` 不存在 → 抛错。

---

## 7. 设计决策与已知陷阱

### 7.1 为什么要"补丁编号"系统
项目在推进中不断发现"实验会给出假结论"的缺陷，每条都给了编号并写进注释与 docstring。
读代码时看到 `P0-4`、`N1`、`D1` 不是变量，是缺陷登记号。**改动这些位置前先读对应注释**。

### 7.2 Windows 多进程：collate 为什么传字符串路径
`make_collate` 返回 `functools.partial(_collate_dispatch, store.dir)`，只传**目录字符串**、
在 worker 内自己 `FeatureStore(feat_dir)`。原因：① `DataLoader(num_workers>0)` 要求 collate 可 pickle，
闭包不可 pickle；② 直接传 `FeatureStore` 会把整个对象（含 mmap 句柄）序列化。
对应补丁 `patch_collate_pickle` → `patch_collate_v2`。

### 7.3 memmap 的坑
早期用 `np.memmap(path, shape=...)` 导致 **npy 文件头被当成数据**（前 128 字节解析成极小值/NaN）。
现在统一 `np.load(path, mmap_mode="r")` 并断言形状。`probe_store_read.py` 就是这次事故的取证脚本。

### 7.4 float16 存储、float32 计算
所有特征库按 float16 存（41 GiB → 可控），collate 后转 float32 送入模型。
`check_data_all.py` 对 `text/genome` 期望 float32，与实现（float16）不一致，会**恒报 WARN**——功能等价，属报告噪音。

### 7.5 ⚠️ 全通道 `--collect` 会清零 text/genome
`StoreWriter.__init__` 用 `mode="w+"` **重新打开所有通道**，因此：
- `features/store.py --collect` 会按 `{kind}_{pid}.npy` 散文件**重写全部通道**；
- 若 `text_*.npy` / `genome_*.npy` 散文件不存在（只有库里的 `text.npy`），
  则 text/genome 会被写成全零、`has_text/has_genome` 变 False——**已有数据被静默清空**。

**安全做法**：增量建库（像 [`prep/build_seq_store.py`](prep/build_seq_store.py) 那样只打开目标通道），
并保留原 `meta.json` 的 `pid2idx` / `has_*`。补某个通道前务必先确认散文件齐全。

### 7.6 CIF 命名不一致（当前阻塞点之一）
`download_afdb_cif.py` 落盘为 `AF-{pid}-F1-model_v4.cif`，但现有 `workdir/afdb/` 里同时存在
`{pid}.cif`（10,915 个）与 `AF-{pid}-F1-model_v4.cif`（2,637 个，多为同一 pid 的重复下载）。
而 [`prep/geo_worker.py`](prep/geo_worker.py)、[`features/extract_surf.py`](features/extract_surf.py)
只认后一种 → 按前者命名的 10,915 个蛋白会被判为"缺结构"。
**修法**：解析时依次尝试两种命名（或优先用已解析好的 `struct_npz/{pid}.npz`，其中已含 `ca` 坐标与 `plddt`）。

### 7.7 `descriptors()` 的残基编码 bug（**未触发，但是真的**）
`parse_cif` 把 Bio.PDB 的 `get_resname()`（三字母，如 `ALA`）直接传给 `descriptors()`，
而后者用一字母集合判断：`HYDRO=set("AVILMFWC")`、`ARO=set("FWY")`、`POS/NEG`。
三字母永不命中 → `grp` 恒为 2，导致 80 维里 **19–27、28–32、63–64 等位恒为常数**（氨基酸组成类特征全废）。
因为 surf 通道从未真正提取过，此 bug 尚未污染任何实验数据。**重建 surf 时必须做三字母→一字母映射**。

### 7.8 其它
| 位置 | 问题 |
|---|---|
| [`config.FeatureModels`](config.py) | `prostt5` 写的是 `athleted/prostt5`，实际 HF 仓库是 `Rostlab/ProstT5`（源码注释已标"用前核对"） |
| [`features/extract_text_genome.py`](features/extract_text_genome.py) | `proteinclip` / `glm2` 均为 `NotImplementedError` 接入点 |
| [`prep/verify_store.py`](prep/verify_store.py) | 硬断言三通道存在，缺通道时不能用；应改用 `verify_store2.py` |
| [`prep/run_knowledge_probe.py`](prep/run_knowledge_probe.py) 与 [`probe_konwledge_lr.py`](probe_konwledge_lr.py) | 写同一个 `runs/probe_knowledge_lr.json`，会互相覆盖（且测量对象不同） |
| [`prep/build_seq_store.py`](prep/build_seq_store.py) 等 | 硬编码 `n == 11018`；换数据集需改 |
| `prep/*.py` | 全部依赖"从仓库根目录执行" |

---

## 8. 当前状态（2026-10-22 实测）

| 资产 | 状态 | 证据 |
|---|---|---|
| 数据划分 | ✅ 完成 | train 163,192 / val 59,260 / test 52,048；正负各半；三集蛋白零交集 |
| `seq.npy` | ✅ 完成 | `(11018,1022,1280)` float16，26.85 GiB；非平凡性 `|max|=9.44` |
| `text.npy` / `genome.npy` | ✅ 完成 | `(11018,128)` / `(11018,256)`，`has_*=1` 全命中 |
| `lengths.npy` / `plddt.npy` | ✅ 完成 | lengths 51–1022（触顶 1592 个）；pLDDT 0–100 |
| `str.npy` | ❌ **缺失** | 无散文件 `str_*.npy`，库中无 `str.npy` |
| `surf.npy` / `iface.npy` | ❌ **缺失** | 同上；但 `struct_npz/` 覆盖 10,915/11,018，原料齐备 |
| `adapter.pt` | ✅ 完成 | 331,648 参数，键/形状校验通过 |
| R0 | ✅ 3 seeds | AUPRC 0.6175 / 0.6512 / 0.6330（seed 42/1234/2024） |
| R1 | ✅ 3 seeds | AUPRC 0.6927 / 0.6904 / 0.6902 |
| R2 | ⚠️ 部分 | seed42 = 0.6888；seed1234 只有 `cfg.json`，无 `metrics.json` |
| R3–R7 | ⛔ 阻塞 | 需要 `str`（R3 起 struct=1），随后 R4 需 iface、R5 需 surf |

**恢复 R3+ 的最小路径**

1. 用 `struct_npz/{pid}.npz`（含 `ca`、`plddt`、`seq`）＋ `sequences_primary.fasta` 生成 `surf_/iface_/plddt_{pid}.npy`
   （顺带修 §7.7 的残基编码 bug；不依赖 Bio.PDB）。
2. 用 ProstT5（`Rostlab/ProstT5`）生成 `str_{pid}.npy`；注意 8 GB 显存需 `bs=1~2`。
3. 写一个**只打开目标通道**的增量 collect（勿用全量 `--collect`，见 §7.5），保留原 `meta.json`。
4. `python prep/verify_store2.py` 验收 → `python check_data_all.py` 退出码 0 → `run_all.py ladder --levels R3 R4 R5`。

---

## 9. 扩展指南：新增一个特征通道

1. **定维度**：在 [`features/store.py`](features/store.py) 的 `RES_LAYOUT`（逐残基）或 `PROT_LAYOUT`（蛋白级）加一项。
2. **写提取器**：输出 `{kind}_{pid}.npy`，形状严格对齐（逐残基 `(L,D)`，蛋白级 `(D,)`）。
3. **加开关**：在 [`config.Switches`](config.py) 加字段，并在 `R_LEVELS` 的对应级别打开。
4. **接模型**：在 [`model.L4Fusion`](model.py) 里消费该通道；若参与全局通路，注意 `global_mlp` 的输入维度。
5. **接 collate**：[`data.make_collate`](data.py) 里补齐零填充与 mask；再更新 `check_data_all.py` 的 `EXP_DTYPE`/`RES_CH`。

---

## 10. 术语表

| 术语 | 含义 |
|---|---|
| R0–R7 | 特征阶梯，R0 是复现锚点，R1 起是本文主体 |
| tag / seed | `runs/{tag}/seed{n}/` 是一次独立训练；tag 可以是 `R3`、`L4-no-siface`、`SEARCH07` |
| iface / siface | 逐残基界面概率 / 从注意力抠出的界面证据（标量） |
| pLDDT | AlphaFold 逐残基置信度（0–100），本项目用它做**质量门** |
| P7 / P8 / P10 / N1 / D1 … | 缺陷登记号，见 §7.1 |
| 度平衡负采样 | Bernett 原生负样本构造方式，使随机基线 AUPRC≈0.5 |
