#!/usr/bin/env bash
# exp16: factorized 실험의 변경점 ①~④를 baseline A에서 하나씩 켜기 (seed 0, experiments.md 2026-10-11)
cd "$(dirname "$0")/.."
TS=$(date +%m%d_%H%M)
COMMON=(--data "logs/teacher_data/relabel_s1000/robust/*.npz" --loss nearest_good --epochs 500 --lr 1e-3
        --lr-schedule cosine --patience 500 --model gnn --obs-transform relative --sel-weight 10
        --select within --seed 0 --threads 2)
run() { local tag=$1; shift; nohup .venv/bin/python -u -m exp16_wm.bc "${COMMON[@]}" "$@" > "logs/run_logs/ablate_${tag}_${TS}.log" 2>&1 & }
if [ "$1" = "seeds12" ]; then            # ①만, ②만, ①②④ — seed 1/2 (seed 0은 기존 run)
  for sd in 1 2; do
    run c1_struct_s$sd    --selection factorized --sel-target label --reg-target angle --seed $sd
    run c2_seltarget_s$sd --selection global     --sel-target made  --reg-target angle --seed $sd
    run c124_s$sd         --selection factorized --sel-target made  --reg-target angle --scratch-head --seed $sd
  done
  wait; exit
fi
if [ "$1" = "c4seeds" ]; then            # ④만, seed 1/2 (seed 0은 기존 run)
  for sd in 1 2; do run c4_scratch_s$sd --selection global --sel-target label --reg-target angle --scratch-head --seed $sd; done
  wait; exit
fi
if [ "$1" = "c124" ]; then               # ①②④ 켜고 ③ 끔
  run c124 --selection factorized --sel-target made --reg-target angle --scratch-head
  wait; exit
fi
if [ "$1" = "made_same" ]; then          # ③ 원인 검증 (experiments.md 2026-10-11)
  run c3_madesame --selection global --sel-target label --reg-target made_same
  wait; exit
fi
run control      --selection global     --sel-target label --reg-target angle
run c1_struct    --selection factorized --sel-target label --reg-target angle
run c2_seltarget --selection global     --sel-target made  --reg-target angle
run c3_regtarget --selection global     --sel-target label --reg-target made
run c4_scratch   --selection global     --sel-target label --reg-target angle --scratch-head
run all          --selection factorized --sel-target made  --reg-target made  --scratch-head
wait
