#!/bin/bash
# World Model Curriculum Learning Pipeline
# Usage: bash run_scripts/run_curriculum.sh [--max-stage N]
set -e
cd "$(dirname "$0")/.."

# sp0.0_tp0.0 모델만 사용 — 현재 환경(obs_dim=16)과 obs space 일치
SAC_5M=$(ls logs/experiments/SAC_5000k_s42_sp0.0_tp0.0_*/best_model/best_model.zip 2>/dev/null | head -1)
SAC_2M_S42=$(ls logs/experiments/SAC_2000k_s42_sp0.0_tp0.0_*/best_model/best_model.zip 2>/dev/null | head -1)
SAC_2M_S1=$(ls logs/experiments/SAC_2000k_s1_sp0.0_tp0.0_*/best_model/best_model.zip 2>/dev/null | head -1)

SAC_ARGS=""
for m in "$SAC_5M" "$SAC_2M_S42" "$SAC_2M_S1"; do
    [ -n "$m" ] && SAC_ARGS="$SAC_ARGS $m"
done

.venv/bin/python world_model/curriculum_loop.py \
    --data-dir world_model/data_v3 \
    --new-data-dir world_model/data_v3_cur \
    --out-dir "world_model/results/curriculum_$(date +%Y%m%d_%H%M%S)" \
    --advance-threshold 0.72 \
    --new-episodes 15000 \
    --batch-size 512 \
    --lr 3e-4 \
    ${SAC_ARGS:+--sac-models $SAC_ARGS} \
    "$@"
