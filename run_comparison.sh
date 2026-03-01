#!/usr/bin/env bash
# run_comparison.sh — Train SAC, PPO, TQC with the same seed, then compare.
#
# Usage:
#   bash run_comparison.sh                    # SAC + PPO + TQC, 1M steps, seed 42
#   ALGOS="SAC TQC" bash run_comparison.sh   # only SAC and TQC
#   SEED=0 STEPS=500000 bash run_comparison.sh

set -e   # exit on any error

SEED=${SEED:-42}
STEPS=${STEPS:-1000000}
ALGOS=${ALGOS:-"SAC PPO TQC"}
VENV=".venv/bin/python"

echo "========================================================"
echo "  billiards-rl comparison run"
echo "  algos: ${ALGOS}"
echo "  steps: ${STEPS}"
echo "  seed : ${SEED}"
echo "========================================================"

i=1
total=$(echo $ALGOS | wc -w | tr -d ' ')

for ALGO in $ALGOS; do
    echo ""
    echo ">>> [${i}/$((total+1))] Training ${ALGO} ..."
    $VENV train.py --algo $ALGO --steps $STEPS --seed $SEED
    i=$((i+1))
done

# ── Compare ───────────────────────────────────────────────────────────────────
echo ""
echo ">>> [$((total+1))/$((total+1))] Comparing results ..."
mkdir -p outputs
$VENV compare.py --out outputs/comparison_s${SEED}.png

echo ""
echo "========================================================"
echo "  Done. Open outputs/comparison_s${SEED}.png"
echo "  TensorBoard: tensorboard --logdir logs/tensorboard"
echo "========================================================"
