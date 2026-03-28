#!/usr/bin/env bash
set +e  # 한 실험 실패해도 계속 진행

# ── Phase 0 (n_balls=1) 비교 실험 ────────────────────────────────────────────
# SB3 SAC vs VanillaSAC (vs WM SAC — 준비되면 주석 해제)
# n_balls=1, 1-step episode → step/trunc penalty 불필요 (0.0)
#
# eval_freq: exp16 --eval-freq는 total_steps 기준 (+=n_envs per step)
#            root  는 n_calls 기준 (10_000 total steps = 1_000 n_calls with n_envs=10)
#            → 둘 다 10,000 total steps마다 eval
# ─────────────────────────────────────────────────────────────────────────────

WANDB_PROJECT="billiards-rl-exp16-phase0"
COMMON_EXP16="--n-envs 10 --total-steps 2_000_000 --eval-freq 10000 --eval-episodes 50 --learning-starts 5000 --n-balls 1 --max-steps 1 --step-penalty 0.0 --trunc-penalty 0.0 --wandb-project $WANDB_PROJECT"
COMMON_ROOT="--algo SAC --steps 2_000_000 --n-balls 1 --max-steps 1 --step-penalty 0.0 --trunc-penalty 0.0 --wandb-project $WANDB_PROJECT"

FAILED=()

run() {
    echo ""
    echo "▶ $*"
    "$@"
    local code=$?
    if [ $code -ne 0 ]; then
        echo "✗ FAILED (exit $code): $*"
        FAILED+=("$*")
    else
        echo "✓ done: $*"
    fi
}

run python train.py             $COMMON_ROOT  --seed 0
run python -m exp16_wm.train --agent vanilla --seed 0 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 1
run python -m exp16_wm.train --agent vanilla --seed 1 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 2
run python -m exp16_wm.train --agent vanilla --seed 2 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 3
run python -m exp16_wm.train --agent vanilla --seed 3 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 42
run python -m exp16_wm.train --agent vanilla --seed 42 $COMMON_EXP16

# ── WM agent (준비되면 주석 해제) ─────────────────────────────────────────────
# run python -m exp16_wm.train --agent wm --seed 0 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 1 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 2 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 3 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 42 $COMMON_EXP16

echo ""
echo "══════════════════════════════════════════"
echo "  All experiments finished."
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Result: all passed ✓"
else
    echo "  Result: ${#FAILED[@]} failed ✗"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "══════════════════════════════════════════"
