#!/usr/bin/env bash
# =============================================================================
# 一键跑完全部实验（Ubuntu / bash）：smoke -> search -> ladder -> ablation
#                                        -> baseline -> stats -> mc
#
# 用法：
#   bash RUN_ALL_EXPERIMENTS.sh                       # 完整流程（3 seed，40 trial）
#   bash RUN_ALL_EXPERIMENTS.sh --quick               # 快速（1 seed，10 trial）
#   bash RUN_ALL_EXPERIMENTS.sh --seeds 42 --trials 10
#   bash RUN_ALL_EXPERIMENTS.sh --skip-search         # 用默认超参，不做搜索
#   bash RUN_ALL_EXPERIMENTS.sh --dry-run             # 只打印将要执行的命令
#
# 参数：
#   --seeds a,b,c      参考 run_all.py --seeds（默认 42,1234,2024）
#   --bs N             批大小（默认 64；48 GB 显存可用 64~128）
#   --lr X             学习率（默认 1e-4）
#   --trials N         超参随机搜索 trial 数（默认 40）
#   --t N              MC-Dropout 采样次数（默认 10）
#   --quick            等价于 --seeds 42 --trials 10
#   --skip-smoke      跳过冒烟
#   --skip-search     跳过搜索（若已有 best_hyperparams.json 会自动复用）
#   --skip-baseline   跳过 RFC 基线
#   --dry-run          只打印，不执行
#
# 注意：run_all.py 没有断点续跑，重复执行会覆盖同 tag/seed 的产物。
#       长跑请放到 tmux 里：tmux new -s ppi 'bash RUN_ALL_EXPERIMENTS.sh'
# =============================================================================
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SELF")"
cd "$HERE"

SEEDS="42,1234,2024"
BS=64
LR=1e-4
TRIALS=40
T=10
QUICK=0
SKIP_SMOKE=0
SKIP_SEARCH=0
SKIP_BASELINE=0
DRY=0

usage() { sed -n '2,/^[^#]/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)         SEEDS="$2"; shift 2 ;;
        --bs)            BS="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --trials)        TRIALS="$2"; shift 2 ;;
        --t)             T="$2"; shift 2 ;;
        --quick)         QUICK=1; shift ;;
        --skip-smoke)    SKIP_SMOKE=1; shift ;;
        --skip-search)   SKIP_SEARCH=1; shift ;;
        --skip-baseline) SKIP_BASELINE=1; shift ;;
        --dry-run)       DRY=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "未知参数: $1（用 --help 看用法）" >&2; exit 2 ;;
    esac
done

if [[ "$QUICK" == "1" ]]; then
    SEEDS="42"
    TRIALS=10
fi

# ---- 解释器：优先 .venv ----
PY="python3"
if [[ -x .venv/bin/python ]]; then
    PY=".venv/bin/python"
elif [[ "$DRY" == "0" ]]; then
    echo "提示：没找到 .venv/bin/python，将使用系统 python3。建议先跑 bash setup_ubuntu.sh" >&2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

LADDER_TAGS=(R0 R1 R2 R3 R4 R5 R6 R7)
ABL_TAGS=(
    L4-unidirectional L4-no-iface-bias L4-no-siface L4-no-surface-gate
    L4-no-selfattn-ctx L4-alpha-to-add L4-global-concat-nosym L4-no-global-path
    L4-proj-seq-early L5-no-asym L5-no-siface L5-plain-mlp L5-mixup
    L5-no-ls L5-no-rdrop L5-concat-asym No-Seq No-SSL
)

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
SEED_ARR=("${SEED_ARR[@]// /}")

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="logs/bundle_run_${STAMP}.log"
: > "$MASTER"

hr() { printf '%s\n' "$(printf '=%.0s' {1..78})"; }

run_step() {   # run_step <标题> <命令...>
    local title="$1"; shift
    local t0=$SECONDS code=0
    {
        echo
        hr
        echo "[$(date +%H:%M:%S)] $title"
        echo "[$(date +%H:%M:%S)] \$ $*"
    } | tee -a "$MASTER"

    if [[ "$DRY" == "1" ]]; then
        return 0
    fi

    set +e
    "$@" 2>&1 | tee -a "$MASTER"
    code=${PIPESTATUS[0]}
    set -e

    local dt=$(( SECONDS - t0 ))
    local msg="[$(date +%H:%M:%S)] $title 结束 exit=$code 用时 $(( dt / 60 )) 分 $(( dt % 60 )) 秒"
    if [[ "$code" -eq 0 ]]; then
        printf '\033[32m%s\033[0m\n' "$msg" | tee -a "$MASTER"
    else
        printf '\033[31m%s\033[0m\n' "$msg" | tee -a "$MASTER"
    fi
    return "$code"
}

{
    echo "python  = $PY"
    echo "seeds   = ${SEED_ARR[*]}   bs=$BS  lr=$LR  trials=$TRIALS  T=$T"
    echo "日志     = $MASTER"
    echo "开始     = $(date -Is)"
} | tee -a "$MASTER"

# ---- 1) 冒烟 ----
if [[ "$SKIP_SMOKE" == "0" ]]; then
    run_step "smoke（33 项自检）" "$PY" smoke_all.py --fresh
else
    echo "（跳过 smoke）" | tee -a "$MASTER"
fi

# ---- 2) 超参搜索（先做，后面复用 best_hyperparams.json）----
hp=()
HP_FILE="workdir/runs/best_hyperparams.json"
if [[ "$SKIP_SEARCH" == "0" ]]; then
    run_step "search（${TRIALS} trial）" "$PY" run_all.py search --trials "$TRIALS"
fi
if [[ -f "$HP_FILE" ]]; then
    hp=(--hyperparams "$HP_FILE")
    echo "复用超参：$HP_FILE" | tee -a "$MASTER"
else
    echo "无 $HP_FILE，使用 config.py 默认超参" | tee -a "$MASTER"
fi

# ---- 3) 阶梯 ----
run_step "ladder R0-R7" \
    "$PY" run_all.py ladder --levels "${LADDER_TAGS[@]}" \
    --seeds "${SEED_ARR[@]}" --bs "$BS" --lr "$LR" ${hp[@]+"${hp[@]}"}

# ---- 4) 消融 ----
run_step "ablation 18 项" \
    "$PY" run_all.py ablation --names "${ABL_TAGS[@]}" \
    --seeds "${SEED_ARR[@]}" --bs "$BS" --lr "$LR" ${hp[@]+"${hp[@]}"}

# ---- 5) 基线 ----
if [[ "$SKIP_BASELINE" == "0" ]]; then
    run_step "baseline RFC-mean / RFC-40" "$PY" run_all.py baseline
fi

# ---- 6) 统计 ----
run_step "stats ladder vs R0" \
    "$PY" run_all.py stats --tags "${LADDER_TAGS[@]}" --baseline R0
[[ "$DRY" == "1" ]] || cp -f workdir/runs/summary.csv workdir/runs/summary_ladder.csv

run_step "stats ablation vs R7" \
    "$PY" run_all.py stats --tags "${ABL_TAGS[@]}" --baseline R7
[[ "$DRY" == "1" ]] || cp -f workdir/runs/summary.csv workdir/runs/summary_ablation.csv

# ---- 7) MC-Dropout（派生量，失败不终止整个流程）----
if ! run_step "mc R7（T=$T）" \
        "$PY" run_all.py mc --tags R7 --seeds "${SEED_ARR[@]}" --T "$T"; then
    echo "警告：MC-Dropout 失败（多半是某个 seed 缺 ckpt.pt），已跳过。" | tee -a "$MASTER"
fi

{
    echo
    hr
    echo "全部完成：$(date -Is)"
    echo "汇总（workdir/runs/*.csv）："
    ls -lh workdir/runs/*.csv 2>/dev/null || echo "  （无 csv）"
    echo "日志：$MASTER"
} | tee -a "$MASTER"
