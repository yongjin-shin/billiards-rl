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
if [ "$1" = "depth" ]; then              # encoder 깊이 스케일링: 6층 / 8층, made2 head 포함, seed 0/1/2
  for sd in 0 1 2; do
    run depth_L6_s$sd --pair-supervision --no-wta-fallback --made2-head --gnn-layers 6 --seed $sd
    run depth_L8_s$sd --pair-supervision --no-wta-fallback --made2-head --gnn-layers 8 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "long" ]; then               # 학습 길이 3배(1500 epoch, cosine도 1500): 깊이 4 vs 깊이 6, made2 head 포함, seed 0/1/2 (experiments.md 2026-10-12)
  for sd in 0 1 2; do                    # 출력 이름에 epoch 수가 없어서 기존 500 epoch run과 glob가 겹치지 않도록 --out-root를 따로 둔다
    run long_L4_s$sd --epochs 1500 --patience 1500 --out-root logs/experiments_long --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --seed $sd
    run long_L6_s$sd --epochs 1500 --patience 1500 --out-root logs/experiments_long --pair-supervision --no-wta-fallback --made2-head --gnn-layers 6 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "speed" ]; then              # 속도 분포 head: 같은 코드 control / L1 / 구간별 확률(21개), 깊이 4, made2 head 포함, seed 0/1/2 (experiments.md 2026-10-12)
  for sd in 0 1 2; do                    # 출력 이름이 기존 깊이 4 run과 겹치지 않도록 --out-root를 따로 둔다
    run speed_ctrl_s$sd --out-root logs/experiments_speed --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --seed $sd
    run speed_l1_s$sd   --out-root logs/experiments_speed --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --speed-loss l1 --seed $sd
    run speed_bin_s$sd  --out-root logs/experiments_speed --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --speed-bins 21 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "loss" ]; then               # 각도 loss 가중치 수동 4배(--angle-scale-deg 0.1), 깊이 4, made2 head 포함, seed 0/1/2 (experiments.md 2026-10-12)
  for sd in 0 1 2; do
    run loss_a01_s$sd --out-root logs/experiments_loss --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --angle-scale-deg 0.1 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "kendall" ]; then            # 각도·속도·선택 항의 가중치를 학습(--uncertainty-weights), 깊이 4, made2 head 포함, seed 0/1/2 (experiments.md 2026-10-12)
  for sd in 0 1 2; do
    run loss_kw_s$sd --out-root logs/experiments_loss --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --uncertainty-weights --seed $sd
  done
  wait; exit
fi
if [ "$1" = "geom" ]; then               # 기하 look-ahead 보조 목표(--geom-aux-weight 5), 깊이 4, made2 head 포함, seed 0/1/2 (experiments.md 2026-10-12)
  for sd in 0 1 2; do
    run geom_ga5_s$sd --out-root logs/experiments_geom --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --geom-aux-weight 5 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "geomlog" ]; then            # 보조 loss 곡선 확인: 이전 geom seed 0 run과 같은 인자, epoch별 보조 MSE 기록(experiments.md 2026-10-13)
  run geomlog_ga5_s0 --out-root logs/experiments_geomlog --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --geom-aux-weight 5 --seed 0
  wait; exit
fi
if [ "$1" = "geomw20" ]; then            # 보조 가중치 5 → 20, seed 0 (experiments.md 2026-10-13)
  run geomw20_ga20_s0 --out-root logs/experiments_geomw20 --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --geom-aux-weight 20 --seed 0
  wait; exit
fi
if [ "$1" = "auxonly" ]; then            # 목표만 학습하는 encoder(BC loss 없음), seed 0 (experiments.md 2026-10-13)
  run auxonly_s0 --out-root logs/experiments_auxonly --pair-supervision --no-wta-fallback --gnn-layers 4 --geom-aux-weight 1 --aux-only --seed 0
  wait; exit
fi
if [ "$1" = "pretrain" ]; then           # aux-only encoder로 초기화한 뒤 joint w=5, seed 0 (experiments.md 2026-10-13)
  INIT=$(ls -d logs/experiments_auxonly/*_s0_* | head -1)/bc_model.pt
  run pretrain_ga5_s0 --out-root logs/experiments_pretrain --pair-supervision --no-wta-fallback --made2-head --gnn-layers 4 --geom-aux-weight 5 --init-from "$INIT" --seed 0
  wait; exit
fi
if [ "$1" = "geomL4" ]; then             # E2: 깊이 4 + --made2-geom(진단), seed 0/1/2 (experiments.md 2026-10-13 다음 실험 순서)
  for sd in 0 1 2; do
    run geomL4_s$sd --out-root logs/experiments_made2geomL4 --pair-supervision --no-wta-fallback --made2-head --made2-geom --gnn-layers 4 --seed $sd
  done
  wait; exit
fi
if [ "$1" = "aimenc" ]; then             # E3: 조준각 전용 encoder(aux-only, 목표 = 18 pair의 정확한 조준각), seed 0 (experiments.md 2026-10-13 다음 실험 순서)
  run aimenc_s0 --out-root logs/experiments_aimenc --pair-supervision --no-wta-fallback --gnn-layers 4 --geom-aux-weight 1 --geom-aux-set aim --aux-only --seed 0
  wait; exit
fi
if [ "$1" = "aimenc2" ]; then            # E3 재측정: 같은 인자 + --save-last(마지막 epoch 가중치도 저장), seed 0
  run aimenc2_s0 --out-root logs/experiments_aimenc2 --pair-supervision --no-wta-fallback --gnn-layers 4 --geom-aux-weight 1 --geom-aux-set aim --aux-only --save-last --seed 0
  wait; exit
fi
if [ "$1" = "aimenc12" ]; then           # E4 준비: seed 1·2용 조준각 encoder(각 seed의 train split만), 마지막 epoch 저장
  for sd in 1 2; do
    run aimenc2_s$sd --out-root logs/experiments_aimenc2 --pair-supervision --no-wta-fallback --gnn-layers 4 --geom-aux-weight 1 --geom-aux-set aim --aux-only --save-last --seed $sd
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
