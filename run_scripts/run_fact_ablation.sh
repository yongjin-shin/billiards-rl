#!/usr/bin/env bash
# exp16: factorized 실험의 변경점 ①~④를 baseline A에서 하나씩 켜기 (seed 0, experiments.md 2026-10-11)
cd "$(dirname "$0")/.."
TS=$(date +%m%d_%H%M)
COMMON=(--data "logs/teacher_data/relabel_s1000/robust/*.npz" --loss nearest_good --epochs 500 --lr 1e-3
        --lr-schedule cosine --patience 500 --model gnn --obs-transform relative --sel-weight 10
        --select within --seed 0 --threads 2)
run() { local tag=$1; shift; nohup .venv/bin/python -u -m exp16_wm.bc "${COMMON[@]}" "$@" > "logs/run_logs/ablate_${tag}_${TS}.log" 2>&1 & }
if [ "$1" = "geoang" ]; then             # 각도 head에서 이웃 message 빼기(--geom-angle) vs control, seed 0/1/2
  for sd in 0 1 2; do
    run geoang_s$sd         --pair-supervision --no-wta-fallback --geom-angle --seed $sd
    run geoang_control_s$sd --pair-supervision --no-wta-fallback --seed $sd
  done
  wait; exit
fi
if [ "$1" = "capacity" ]; then           # encoder 용량: 폭 256 / 깊이 4, made2 head 포함, seed 0/1/2
  for sd in 0 1 2; do
    run cap_H256_s$sd --pair-supervision --no-wta-fallback --made2-head --gnn-hidden 256 --seed $sd
    run cap_L4_s$sd   --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "made2geo" ]; then           # made head 진단: 기하 조준각 차이를 입력으로(--made2-geom), seed 0/1/2
  for sd in 0 1 2; do run made2geo_s$sd --pair-supervision --no-wta-fallback --made2-head --made2-geom --seed $sd; done
  wait; exit
fi
if [ "$1" = "made2" ]; then              # look-ahead 2단계: sims identity 복원 후 made2 head 학습, seed 0/1/2
  for f in logs/teacher_data/relabel_s1000/sims/ghost_s1000_shard*.npz; do
    .venv/bin/python -u -m exp16_wm.sim_identity --file "$f" > "logs/run_logs/sim_identity_$(basename "$f" .npz).log" 2>&1 &
  done
  wait
  for sd in 0 1 2; do run made2_s$sd --pair-supervision --no-wta-fallback --made2-head --seed $sd; done
  wait; exit
fi
if [ "$1" = "twoball" ]; then            # 공 2개 데이터 train-only 추가 (+A / +B), 새 baseline 설정, seed 0/1/2
  for sd in 0 1 2; do
    for X in A B; do
      run twoball_${X}_s$sd --pair-supervision --no-wta-fallback --seed $sd \
          --train-only-data "logs/teacher_data/two_ball_$X/robust/*.npz"
    done
  done
  wait; exit
fi
if [ "$1" = "wtafb" ]; then              # WTA fallback 검증: old code(--pair-supervision) 그대로 vs fallback 끔, seed 0/1/2
  for sd in 0 1 2; do
    run oldA_s$sd    --pair-supervision --seed $sd
    run oldA_nofb_s$sd --pair-supervision --no-wta-fallback --seed $sd
  done
  wait; exit
fi
if [ "$1" = "control12" ]; then          # new-code control, seed 1/2 (baseline A는 이전 코드)
  for sd in 1 2; do run control_s$sd --selection global --sel-target label --reg-target angle --seed $sd; done
  wait; exit
fi
if [ "$1" = "madehead" ]; then           # look-ahead 1단계: ⑤ made head만, ④+⑤ — seed 0/1/2
  for sd in 0 1 2; do
    run c5_made_s$sd  --selection global --sel-target label --reg-target angle --made-head --seed $sd
    run c45_made_s$sd --selection global --sel-target label --reg-target angle --scratch-head --made-head --seed $sd
  done
  wait; exit
fi
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
