#!/usr/bin/env bash
set +e

# ── Scale-up 실험: epochs=300, h=256 → 512 → 1024 전부 순차 실행 ─────────────
#
# 목표: pos_mse 0.03~0.01 달성
#   Step 1) h=256   (저비용, 수렴 확인)
#   Step 2) h=512   (medium 급)
#   Step 3) h=1024  (large 급)
#
# bash run_scaleup.sh            → 1,2,3 전부
# bash run_scaleup.sh --no-wandb
# ─────────────────────────────────────────────────────────────────────────────

[[ "$1" == "--no-wandb" ]] && NO_WANDB="--no-wandb" || NO_WANDB=""

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

echo "══ Step 1: h=256, epochs=300 ══"
run python -m world_model.train_predictor $COMMON --lstm-hidden 256 --seed 0
run python -m world_model.train_predictor $COMMON --lstm-hidden 256 --seed 1
run python -m world_model.train_predictor $COMMON --lstm-hidden 256 --seed 2

echo "══ Step 2: h=512, epochs=300 ══"
run python -m world_model.train_predictor $COMMON --lstm-hidden 512 --seed 0
run python -m world_model.train_predictor $COMMON --lstm-hidden 512 --seed 1
run python -m world_model.train_predictor $COMMON --lstm-hidden 512 --seed 2

echo "══ Step 3: h=1024, epochs=300 ══"
run python -m world_model.train_predictor $COMMON --lstm-hidden 1024 --seed 0
run python -m world_model.train_predictor $COMMON --lstm-hidden 1024 --seed 1
run python -m world_model.train_predictor $COMMON --lstm-hidden 1024 --seed 2

echo ""
echo "══════════════════════════════════════════"
echo "  All done. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -gt 0 ]] && for f in "${FAILED[@]}"; do echo "    - $f"; done
echo "══════════════════════════════════════════"
