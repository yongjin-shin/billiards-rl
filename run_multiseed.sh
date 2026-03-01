#!/usr/bin/env bash
# run_multiseed.sh — Multi-seed benchmark: SAC / TQC / PPO × N seeds
#
# Runs every (algo, seed) combination sequentially and saves results
# in logs/experiments/. Use compare.py afterward to plot all curves.
#
# Usage:
#   bash run_multiseed.sh                          # SAC+TQC+PPO, seeds 0 1 2, 1M steps
#   ALGOS="SAC TQC" bash run_multiseed.sh          # only SAC and TQC
#   SEEDS="42 0 1" bash run_multiseed.sh           # custom seeds
#   STEPS=500000 bash run_multiseed.sh             # fewer steps (faster)

set -e

ALGOS=${ALGOS:-"SAC TQC PPO"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-1000000}
VENV=".venv/bin/python"

# ── Count total runs ──────────────────────────────────────────────────────────
n_algos=$(echo $ALGOS | wc -w | tr -d ' ')
n_seeds=$(echo $SEEDS | wc -w | tr -d ' ')
total=$((n_algos * n_seeds))
est_min=$((total * 27))   # rough estimate: ~27 min/run (TQC pace)
est_h=$((est_min / 60))
est_m=$((est_min % 60))

echo "========================================================"
echo "  billiards-rl — multi-seed benchmark"
echo "  algos : ${ALGOS}"
echo "  seeds : ${SEEDS}"
echo "  steps : ${STEPS}"
echo "  total : ${total} runs  (~${est_h}h ${est_m}m estimated)"
echo "========================================================"

run=0
start_wall=$(date +%s)

for SEED in $SEEDS; do
    for ALGO in $ALGOS; do
        run=$((run + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [${run}/${total}]  algo=${ALGO}  seed=${SEED}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        $VENV train.py --algo $ALGO --steps $STEPS --seed $SEED

        # ── Wall-clock ETA ────────────────────────────────────────────
        now=$(date +%s)
        elapsed=$(( now - start_wall ))
        avg_per_run=$(( elapsed / run ))
        remaining=$(( (total - run) * avg_per_run ))
        eta_h=$(( remaining / 3600 ))
        eta_m=$(( (remaining % 3600) / 60 ))
        echo "  ✓ done  |  wall elapsed: $((elapsed/60))m  |  ETA remaining: ${eta_h}h ${eta_m}m"
    done
done

# ── Summary table + plot ──────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All ${total} runs complete. Generating comparison ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p outputs
OUTFILE="outputs/multiseed_s${SEEDS// /-}_${STEPS%000000}M.png"
$VENV compare.py --out "$OUTFILE"

echo ""
echo "========================================================"
echo "  Done."
echo "  Plot  → ${OUTFILE}"
echo "  Table → python compare.py"
echo "========================================================"
