#!/usr/bin/env bash
# exp16: 공 2개로 시작하는 state 생성 + robust label 파이프라인 (experiments.md 2026-10-11)
#   A = env 기본 배치, B = 테이블 전체 균일 배치. 각 7,500 states, 5 shards 병렬.
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
N=7500; SH=5
make() {   # $1 = A|B, $2 = placement, $3 = seed
  local D=logs/teacher_data/two_ball_$1 L=logs/run_logs/two_ball_$1
  mkdir -p $D logs/run_logs
  $PY -m exp16_wm.two_ball_states --placement $2 --n $N --seed $3 --out $D/states.npz > ${L}_states.log 2>&1
  for i in $(seq 0 $((SH-1))); do
    $PY -m exp16_wm.relabel --data $D/states.npz --shard $i --n-shards $SH --out-dir $D > ${L}_relabel$i.log 2>&1 &
  done
  wait
  $PY -m exp16_wm.pair_labels --dir $D/robust > ${L}_pairs.log 2>&1
  $PY -m exp16_wm.outcomes --dir $D/robust > ${L}_outcomes.log 2>&1
  for i in $(seq 0 $((SH-1))); do
    $PY -m exp16_wm.pair_stats --root $D --dirs robust --shard $i --n-shards $SH --base states > ${L}_pairstats$i.log 2>&1 &
  done
  wait
  echo "two_ball_$1 done" >> ${L}_states.log
}
make A env 2000 &
make B uniform 3000 &
wait
