#!/usr/bin/env bash
set +e  # 한 실험 실패해도 계속 진행

# ── Exp-06 재현 파라미터 ──────────────────────────────────────────────────────
# root/train.py  : SB3 SAC baseline  (n_envs=10, eval_freq=10k total steps, eval_eps=50 하드코딩)
# exp16/train.py : VanillaSAC custom (동일 하이퍼파라미터)
# pp 없음, 두 코드 동등 비교용
#
# NOTE: exp16의 --eval-freq는 total_steps 기준 (+=n_envs per step)
#       root는 n_calls 기준 (eval_freq=10_000//N_ENVS=1000 n_calls = 10,000 total steps)
#       → 둘 다 10,000 total steps마다 eval하려면 --eval-freq 10000
# ─────────────────────────────────────────────────────────────────────────────

WANDB_PROJECT="billiards-rl-exp16"
COMMON_EXP16="--n-envs 10 --total-steps 2_000_000 --eval-freq 10000 --eval-episodes 50 --learning-starts 5000 --n-balls 3 --max-steps 5 --step-penalty 0.1 --trunc-penalty 1.0 --wandb-project $WANDB_PROJECT"
COMMON_ROOT="--algo SAC --steps 2_000_000 --n-balls 3 --max-steps 5 --step-penalty 0.1 --trunc-penalty 1.0 --wandb-project $WANDB_PROJECT"

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

run python train.py             $COMMON_ROOT  --seed 1
run python -m exp16_wm.train --agent vanilla --seed 1 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 2
run python -m exp16_wm.train --agent vanilla --seed 2 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 3
run python -m exp16_wm.train --agent vanilla --seed 3 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 42
run python -m exp16_wm.train --agent vanilla --seed 42 $COMMON_EXP16

run python train.py             $COMMON_ROOT  --seed 0
run python -m exp16_wm.train --agent vanilla --seed 0 $COMMON_EXP16

# ── WM agent (준비되면 주석 해제) ─────────────────────────────────────────────
# run python -m exp16_wm.train --agent wm --seed 1 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 2 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 3 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 4 $COMMON_EXP16
# run python -m exp16_wm.train --agent wm --seed 5 $COMMON_EXP16

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
