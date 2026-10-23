"""全局配置（v2.3）。默认超参按 8GB 显存笔电校准；24GB+ 卡用 --bs 64 --lr 1e-4 提回。"""
from dataclasses import dataclass, field, replace, asdict
import json, os

MAX_LEN = 1022  # 每蛋白残基数（不含 BOS/EOS）

@dataclass
class Paths:
    data_dir: str = "./workdir/data"        # train.csv / val.csv / test.csv
    feat_dir: str = "./workdir/features"
    out_dir: str = "./workdir/runs"
    ssl_fasta: str = "./workdir/ssl/uniref50_lt30.fasta"
    adapter_ckpt: str = "./workdir/ssl/adapter.pt"

@dataclass
class FeatureModels:
    esm2: str = "esm2_t33_650M_UR50D"          # fair-esm
    prostt5: str = "Rostlab/ProstT5"           # F2：实际 HF 仓库（原 "athleted/prostt5" 不存在）；本机缓存已有 10.75 GB 权重
    # gLM2 官方权重在 tattabio 组织下（EvolutionaryScale/gLM2-350M 不存在，会 401）
    glm2: str = "tattabio/gLM2_150M"  # ← 用前核对
    proteinclip_ckpt: str = ""                 # 可得时填本地路径
    openai_model: str = "text-embedding-3-large"
    st_local: str = "sentence-transformers/all-mpnet-base-v2"
    # AFDB v4 已于 2025 年下线（实测 404），v6 在役；保留 v4 仅作最后回退
    afdb_url: str = "https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v6.cif"
    afdb_url_fallback: str = "https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v4.cif"

@dataclass
class Switches:
    struct: int = 0
    iface: int = 0
    surf: int = 0
    text: int = 0
    genome: int = 0

@dataclass
class ModelCfg:
    kind: str = "l4"                  # "reim" | "l4"
    use_ssl: bool = False
    sw: Switches = field(default_factory=Switches)
    dims: dict = field(default_factory=lambda: dict(seq=1280, str=1024, surf=80, text=128, genome=256))
    d_model: int = 256
    n_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.2
    beta_init: float = 0.0
    iface_tau: float = 0.5
    # ---- L4 组件消融 ----
    bidirectional: bool = True        # False → L4-unidirectional
    iface_bias: bool = True           # False → L4-no-iface-bias
    siface_out: bool = True           # False → L4-no-siface
    surface_gate: bool = True         # False → L4-no-surface-gate
    selfattn_ctx: bool = True         # False → L4-no-selfattn-ctx
    alpha_mode: str = "weighted"      # "add" → L4-alpha-to-add
    global_sym: bool = True           # False → L4-global-concat-nosym
    global_path: bool = True          # False → L4-no-global-path
    seq_proj_pos: str = "late"        # "early" → L4-proj-seq-early
    zero_seq: bool = False            # True → No-Seq 反向消融
    # ---- L5 消融 ----
    l5_no_asym: bool = False
    l5_siface: bool = True
    l5_plain_mlp: bool = False        # plain/blocks 双模块常驻构造（N4）：切换时参数量守恒
    l5_concat_asym: bool = False

@dataclass
class TrainCfg:
    # 8GB 笔电默认；A100/4090 建议 bs=64, lr=1e-4, num_workers=4
    lr: float = 5e-5
    wd: float = 1e-6
    bs: int = 16
    max_epochs: int = 50
    patience: int = 8
    warmup_frac: float = 0.05
    clip: float = 1.0
    label_smooth: float = 0.05
    mixup_prob: float = 0.0           # 默认关；搜索时调
    mixup_alpha: float = 0.2
    rdrop_alpha: float = 1.0
    rdrop_sym: bool = True
    seed: int = 42
    bf16: bool = True
    num_workers: int = 2              # Windows memmap 冲突时改 0
    train_subset_frac: float = 1.0
    date_col: str = ""                # 时间截断（仅作用于 train）
    cutoff: str = ""

def _sw(**kw):
    base = dict(struct=0, iface=0, surf=0, text=0, genome=0)
    base.update(kw)
    return Switches(**base)

# R 阶梯：逐级只多开一个开关；负采样=Bernett 原生（度平衡）；ILP hard negatives 为二期扩展
R_LEVELS = {
    "R0": dict(kind="reim", use_ssl=False, sw=_sw()),
    "R1": dict(kind="l4",   use_ssl=False, sw=_sw()),
    "R2": dict(kind="l4",   use_ssl=True,  sw=_sw()),
    "R3": dict(kind="l4",   use_ssl=True,  sw=_sw(struct=1)),
    "R4": dict(kind="l4",   use_ssl=True,  sw=_sw(struct=1, iface=1)),
    "R5": dict(kind="l4",   use_ssl=True,  sw=_sw(struct=1, iface=1, surf=1)),
    "R6": dict(kind="l4",   use_ssl=True,  sw=_sw(struct=1, iface=1, surf=1, text=1)),
    "R7": dict(kind="l4",   use_ssl=True,  sw=_sw(struct=1, iface=1, surf=1, text=1, genome=1)),
}

ABLATIONS = {
    "L4-unidirectional":      dict(bidirectional=False),
    "L4-no-iface-bias":       dict(iface_bias=False),
    "L4-no-siface":           dict(siface_out=False),
    "L4-no-surface-gate":     dict(surface_gate=False),
    "L4-no-selfattn-ctx":     dict(selfattn_ctx=False),
    "L4-alpha-to-add":        dict(alpha_mode="add"),
    "L4-global-concat-nosym": dict(global_sym=False),
    "L4-no-global-path":      dict(global_path=False),
    "L4-proj-seq-early":      dict(seq_proj_pos="early"),
    "L5-no-asym":             dict(l5_no_asym=True),
    "L5-no-siface":           dict(l5_siface=False),
    "L5-plain-mlp":           dict(l5_plain_mlp=True),
    "L5-no-mixup":            dict(train__mixup_prob=0.0),
    "L5-no-ls":               dict(train__label_smooth=0.0),
    "L5-no-rdrop":            dict(train__rdrop_alpha=0.0),
    "L5-concat-asym":         dict(l5_concat_asym=True),
    "No-Seq":                 dict(zero_seq=True),
    "No-SSL":                 dict(use_ssl=False),
}

def build_model_cfg(level, overrides=None):
    spec = R_LEVELS[level]
    cfg = ModelCfg(kind=spec["kind"], use_ssl=spec["use_ssl"], sw=spec["sw"])
    for k, v in (overrides or {}).items():
        if not k.startswith("train__"):
            setattr(cfg, k, v)
    return cfg

def build_train_cfg(overrides=None, base=None):
    cfg = replace(base) if base is not None else TrainCfg()
    for k, v in (overrides or {}).items():
        if k.startswith("train__"):
            setattr(cfg, k[len("train__"):], v)
    return cfg

def model_cfg_from_dict(d):
    sw = Switches(**d["sw"])
    rest = {k: v for k, v in d.items() if k != "sw"}
    return ModelCfg(sw=sw, **rest)

def train_cfg_from_dict(d):
    return TrainCfg(**d)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path):
    with open(path) as f:
        return json.load(f)
