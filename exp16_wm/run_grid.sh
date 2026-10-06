#!/usr/bin/env bash
# exp16_wm/run_grid.sh — run Exp-16 arms × seeds with the shared baseline config.
#
# Usage:
#   ARMS="vanilla rssm geom hmix" SEEDS="0 1 2 3" PARALLEL=2 bash exp16_wm/run_grid.sh
#
# Arms: vanilla | traj | rssm | geom (Exp-16 G) | hmix (Exp-16 M: rssm + real-h mixing)
# PARALLEL: concurrent runs. Each run uses --n-envs 10 worker processes + 1 trainer,
#           so keep PARALLEL × 11 ≤ CPU cores. Do NOT change --n-envs (it sets the
#           update-to-data ratio and breaks comparability across arms).
# Per-run stdout goes to logs/runs/<arm>_s<seed>.log; a summary line per run to logs/runs/grid.log.
set -u
cd "$(dirname "$0")/.."

ARMS=${ARMS:-"vanilla rssm geom hmix"}
SEEDS=${SEEDS:-"0 1 2 3"}
PARALLEL=${PARALLEL:-1}
PY=${PY:-python}
COMMON="--n-balls 3 --max-steps 5 --step-penalty 0.1 --trunc-penalty 1.0 --total-steps 2000000 \
--n-envs 10 --learning-starts 5000 --eval-freq 1000 --eval-episodes 50 --best-ckpt-window 5"

arm_args() {
  case "$1" in
    vanilla) echo "--agent vanilla" ;;
    traj)    echo "--agent wm --wm-target traj" ;;
    rssm)    echo "--agent wm --wm-target rssm" ;;
    geom)    echo "--agent geom" ;;
    hmix)    echo "--agent wm --wm-target rssm --h-mix-start 0.5 --h-mix-steps 1000000" ;;
    *)       echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local arm=$1 seed=$2 args
  args=$(arm_args "$arm") || exit 1
  echo "start $arm s$seed $(date '+%F %T')" >> logs/runs/grid.log
  if $PY -u -m exp16_wm.train $args --seed "$seed" $COMMON > "logs/runs/${arm}_s${seed}.log" 2>&1; then
    echo "done  $arm s$seed $(date '+%F %T')" >> logs/runs/grid.log
  else
    echo "FAIL  $arm s$seed $(date '+%F %T')" >> logs/runs/grid.log
  fi
}
export -f run_one arm_args
export PY COMMON

mkdir -p logs/runs
for s in $SEEDS; do for a in $ARMS; do echo "$a $s"; done; done \
  | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$0" "$1"'
echo "all done $(date '+%F %T')" >> logs/runs/grid.log
