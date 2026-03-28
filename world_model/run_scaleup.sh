#!/usr/bin/env bash
set +e

# ── Scale-up 실험: epochs & lstm_hidden 키우기 ────────────────────────────────
#
# 목표: pos_mse 0.03~0.01 달성
#   Step 1) epochs 300, h=256        (저비용, 현재 모델 충분히 수렴)
#   Step 2) epochs 300, h=512        (medium 급)
#   Step 3) epochs 300, h=1024       (large 급, 필요시)
#
# bash world_model/run_scaleup.sh [step]   → step=1,2,3 (기본: 1)
# bash world_model/run_scaleup.sh 2 --no-wandb
# ─────────────────────────────────────────────────────────────────────────────

STEP=${1:-1}
[[ "$2" == "--no-wandb" || "$1" == "--no-wandb" ]] && NO_WANDB="--no-wandb" || NO_WANDB=""

WANDB_PROJECT="billiards-wm-predictor"
COMMON="--model lstm \
        --data world_model/data_abs --tags sac_abs random_abs \
        --wandb-project $WANDB_PROJECT \
        --epochs 300 --batch-size 256 --lr 3e-4 --pos-weight 1.0 --device cpu \
        --strategy tf_ratio --ss-epochs 50 --ss-min-ratio 0.0 \
        --ctx-hidden 128 --lstm-layers 1 \
        $NO_WANDB"

FAILED=()
run() {
    echo ""; echo "▶ $*"
    "$@"
    local code=$?
    [[ $code -ne 0 ]] && { echo "✗ FAILED: $*"; FAILED+=("$*"); } || echo "✓ done"
}

if [[ "$STEP" == "1" ]]; then
    echo "══ Step 1: h=256, epochs=300 (수렴 확인) ══"
    run python -m world_model.train_predictor $COMMON --lstm-hidden 256 --seed 0
    run python -m world_model.train_predictor $COMMON --lstm-hidden 256 --seed 1
    run python -m world_model.train_predictor $COMMON --lstm-hidden 256 --seed 2

elif [[ "$STEP" == "2" ]]; then
    echo "══ Step 2: h=512, epochs=300 (medium 급) ══"
    run python -m world_model.train_predictor $COMMON --lstm-hidden 512 --seed 0
    run python -m world_model.train_predictor $COMMON --lstm-hidden 512 --seed 1
    run python -m world_model.train_predictor $COMMON --lstm-hidden 512 --seed 2

elif [[ "$STEP" == "3" ]]; then
    echo "══ Step 3: h=1024, epochs=300 (large 급) ══"
    run python -m world_model.train_predictor $COMMON --lstm-hidden 1024 --seed 0
    run python -m world_model.train_predictor $COMMON --lstm-hidden 1024 --seed 1
    run python -m world_model.train_predictor $COMMON --lstm-hidden 1024 --seed 2

else
    echo "Usage: bash world_model/run_scaleup.sh [1|2|3] [--no-wandb]"
    exit 1
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Step $STEP done. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -gt 0 ]] && for f in "${FAILED[@]}"; do echo "    - $f"; done
echo "══════════════════════════════════════════"
