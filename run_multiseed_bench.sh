#!/usr/bin/env bash
# run_multiseed_bench.sh — SAC / TQC / PPO × 5 seeds, ms=3, 3M steps
#
# 목적: ms=3 frontier에서 알고리즘 간 공정 비교 (Phase 2 baseline).
# 조건: pp/st 없음, sp=0.1, tp=1.0 (simple baseline reward)
# Steps: 3M (기존 1M의 3배 — ms=3이 더 sparse하므로 더 많은 학습 필요)
#
# Usage:
#   source .venv/bin/activate
#   bash run_multiseed_bench.sh           # 포그라운드 (로그 확인용)
#   nohup bash run_multiseed_bench.sh > logs/multiseed_bench.log 2>&1 &
#                                         # 백그라운드 실행

set -e

STEPS=3000000
N_BALLS=3
MAX_STEPS=3
STEP_PENALTY=0.1
TRUNC_PENALTY=1.0
SEEDS=(0 1 2 3 42)
ALGOS=(SAC TQC PPO)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "  Multi-seed benchmark: ms=3, 3M steps"
echo "  Algos: ${ALGOS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Total runs: $((${#ALGOS[@]} * ${#SEEDS[@]}))"
echo "  Started: $(date)"
echo "========================================================"

total_runs=$(( ${#ALGOS[@]} * ${#SEEDS[@]} ))
current=0

for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        current=$(( current + 1 ))
        echo ""
        echo "----------------------------------------------------"
        echo "  [$current/$total_runs] $ALGO seed=$SEED"
        echo "  $(date)"
        echo "----------------------------------------------------"

        python "$SCRIPT_DIR/train.py" \
            --algo "$ALGO" \
            --steps "$STEPS" \
            --n-balls "$N_BALLS" \
            --max-steps "$MAX_STEPS" \
            --step-penalty "$STEP_PENALTY" \
            --trunc-penalty "$TRUNC_PENALTY" \
            --seed "$SEED"
    done
done

echo ""
echo "========================================================"
echo "  All $total_runs runs complete."
echo "  Finished: $(date)"
echo "========================================================"
