#!/usr/bin/env bash
# =============================================================================
# PPI 全通道模型 · Ubuntu 环境安装与自检（一条命令把环境弄好）
#
# 用法：
#   bash setup_ubuntu.sh              # 建 .venv + 装 CUDA 版 torch(cu124) + 依赖 + 自检
#   bash setup_ubuntu.sh --cpu        # 不装 CUDA 版 torch（只能在 CPU 上跑 R0/R1 与检查）
#   bash setup_ubuntu.sh --skip-deps  # 跳过装包，只做文件/环境自检（已装好时用）
#   bash setup_ubuntu.sh --smoke      # 装完立刻跑 33 项冒烟（约 2-6 分钟）
#   bash setup_ubuntu.sh --dry-run    # 只打印将要执行的命令，什么都不改
#
# 说明：
#   * 只读写本目录（`$(dirname $0)`），不碰系统 Python，不 sudo。
#   * apt 部分只做“检测 + 提示”，需要你自己执行（脚本不静默 sudo）。
# =============================================================================
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SELF")"
cd "$HERE"

DRY=0
CPU=0
SKIP_DEPS=0
DO_SMOKE=0
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

for arg in "$@"; do
    case "$arg" in
        --cpu)       CPU=1 ;;
        --skip-deps) SKIP_DEPS=1 ;;
        --smoke)     DO_SMOKE=1 ;;
        --dry-run)   DRY=1 ;;
        -h|--help)   sed -n '2,/^[^#]/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           echo "未知参数: $arg（用 --help 看用法）" >&2; exit 2 ;;
    esac
done

if [[ "$CPU" == "1" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
    if [[ "$DRY" == "1" ]]; then
        printf '  \033[90m[dry-run] %s\033[0m\n' "$*"
        return 0
    fi
    printf '  \033[90m$ %s\033[0m\n' "$*"
    "$@"
}

# ---------------------------------------------------------------- 1. 文件自检
say "1/6 包完整性（必需文件与体积）"
need() {  # need <路径> <最小字节> <说明>
    local p="$1" min="$2" label="$3"
    [[ -f "$p" ]] || die "缺少 $p（$label）—— 请确认解包完整，或先跑 python verify_bundle.py"
    local sz
    sz=$(stat -c %s "$p" 2>/dev/null || stat -f %z "$p")
    (( sz >= min )) || die "$p 只有 $sz 字节（应 >= $min）—— 解包不完整，重跑 unzip"
    local human
    if (( sz >= 1048576 )); then human="$(( sz / 1048576 )) MiB"; else human="$(( sz / 1024 )) KiB"; fi
    ok "$p  $human  ($label)"
}

need config.py               1000      "主配置"
need run_all.py             2000      "实验入口"
need smoke_all.py           5000      "冒烟脚本"
need requirements.txt       1000      "依赖清单"
need workdir/features/meta.json         100000  "特征库索引"
need workdir/features/lengths.npy       40000   "真实长度"
need workdir/features/seq.npy           25000000000  "ESM-2 序列嵌入"
need workdir/features/str.npy           20000000000  "ProstT5 结构嵌入"
need workdir/features/surf.npy          1600000000   "表面几何"
need workdir/features/iface.npy         20000000     "界面倾向"
need workdir/features/text.npy          2500000      "文本 PCA"
need workdir/features/genome.npy        5000000      "基因组窗口"
need workdir/features/plddt.npy         20000000     "pLDDT"
need workdir/data/train.csv             2000000      "训练划分 163,192 对"
need workdir/data/val.csv               900000       "验证划分 59,260 对"
need workdir/data/test.csv              800000       "测试划分 52,048 对"
need workdir/ssl/adapter.pt             1000000      "SeqAdapter 权重（R2-R7 必需）"

chmod +x RUN_ALL_EXPERIMENTS.sh setup_ubuntu.sh 2>/dev/null || true
if [[ -f mmseqs/bin/mmseqs ]]; then
    chmod +x mmseqs/bin/mmseqs 2>/dev/null || true
    if head -c 4 mmseqs/bin/mmseqs | grep -q $'\x7fELF'; then
        ok "mmseqs/bin/mmseqs 是 Linux ELF 可执行（已 +x）"
    else
        warn "mmseqs/bin/mmseqs 不是 ELF（重建划分才需要，不影响训练）"
    fi
fi

free_kb=$(df -Pk . | awk 'NR==2 {print $4}')
if (( free_kb < 60000000 )); then
    warn "当前分区剩余 $(( free_kb / 1048576 )) GiB，低于建议的 60 GiB（每套实验产物约 7 GiB）"
else
    ok "分区剩余 $(( free_kb / 1048576 )) GiB"
fi

# ---------------------------------------------------------------- 2. Python
say "2/6 Python 解释器"
PY=""
if [[ "$SKIP_DEPS" == "0" ]]; then
    for cand in python3.11 python3.12 python3.10 python3 python; do
        command -v "$cand" >/dev/null 2>&1 || continue
        # 某些系统上 python 是个坏壳（比如 Windows 的 py 存根），探测失败就跳过，不能让它带崩 set -e
        v=$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
        case "$v" in
            3.10|3.11|3.12) PY="$cand"; break ;;
        esac
    done
    [[ -n "$PY" ]] || die "找不到 Python 3.10-3.12。Ubuntu 22.04/24.04 自带 python3；若缺请：
      sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    ok "使用 $PY ($($PY -V 2>&1))"

    if ! "$PY" -c 'import venv' >/dev/null 2>&1; then
        die "$PY 缺少 venv 模块：sudo apt install -y python3-venv"
    fi
else
    PY="python3"
    command -v "$PY" >/dev/null 2>&1 || die "找不到 python3"
    ok "跳过选型，直接使用 $(command -v "$PY")"
fi

# ---------------------------------------------------------------- 3. venv
say "3/6 虚拟环境 .venv"
if [[ -x .venv/bin/python ]]; then
    ok ".venv 已存在（$(.venv/bin/python -V 2>&1)）"
else
    if [[ "$DRY" == "1" ]]; then
        echo "  [dry-run] 不实际建 venv"
    else
        run "$PY" -m venv .venv
        ok "已创建 .venv"
    fi
fi
VPY=".venv/bin/python"
[[ "$DRY" == "1" ]] || [[ -x "$VPY" ]] || die ".venv/bin/python 不存在，venv 创建失败"

# ---------------------------------------------------------------- 4. 依赖
say "4/6 依赖（torch 必须单独装，不能和 requirements 混在一个命令里）"
if [[ "$SKIP_DEPS" == "0" ]]; then
    run "$VPY" -m pip install --upgrade pip -q
    run "$VPY" -m pip install torch --index-url "$TORCH_INDEX"
    run "$VPY" -m pip install -r requirements.txt
    ok "依赖安装完成"
else
    warn "已按 --skip-deps 跳过"
fi

# ---------------------------------------------------------------- 5. 运行时自检
say "5/6 运行时自检"
if [[ "$DRY" == "1" ]]; then
    echo "  [dry-run] 跳过实际 import 检查"
else
    "$VPY" - <<'PYEOF'
import sys, os
import numpy, pandas, scipy, sklearn
import torch
print(f"  python  {sys.version.split()[0]}")
print(f"  numpy {numpy.__version__}  pandas {pandas.__version__}  "
      f"scipy {scipy.__version__}  sklearn {sklearn.__version__}")
print(f"  torch   {torch.__version__}   (cuda build: {torch.version.cuda})")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
    print(f"  bf16    {torch.cuda.is_bf16_supported()}")
else:
    print("  GPU     不可用 -> 只能跑 R0/R1 与数据检查（训练请装 NVIDIA 驱动后重跑本脚本）")
PYEOF
    if .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        ok "CUDA 可用"
        if [[ "$CPU" == "1" ]]; then
            warn "你装的是 CPU 版 torch（--cpu），CUDA 仍不可用是预期的"
        fi
    else
        warn "CUDA 不可用：确认已装 NVIDIA 驱动（nvidia-smi 能出表），且 torch 是 cu12x 轮子"
    fi
fi

# ---------------------------------------------------------------- 6. 冒烟
say "6/6 端到端冒烟"
if [[ "$DO_SMOKE" == "1" ]]; then
    run "$VPY" smoke_all.py
else
    echo "  未指定 --smoke。强烈建议现在就跑一次（这是换平台后唯一能证明 DataLoader/CUDA 通路正常的检查）："
    echo "      .venv/bin/python smoke_all.py"
    echo "  或：bash setup_ubuntu.sh --smoke"
fi

say "完成"
cat <<'TIP'
下一步（先测速，再决定要不要跑全套 5 天）：
    source .venv/bin/activate
    python run_all.py ladder --levels R7 --seeds 42 --bs 64 --lr 1e-4

一键全套（后台跑，见 USAGE_LINUX.md §7）：
    bash RUN_ALL_EXPERIMENTS.sh --quick          # 1 seed + 10 trial，约 1.5-3 天
    tmux new -s ppi 'bash RUN_ALL_EXPERIMENTS.sh'
TIP
