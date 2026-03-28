#!/usr/bin/env bash
set +e

# ── 2-layer LSTM vs 1-layer LSTM (tf_ratio, 동일 파라미터) ──────────────────
#   1-layer h=256          : 348,056 params
#   2-layer h=148 ctx=120  : 348,032 params  (recurrent 용량 최대화)
#
# bash world_model/run_2layer.sh
# bash world_model/run_2layer.sh --no-wandb
# ─────────────────────────────────────────────────────────────────────────────

WANDB_PROJECT="billiards-wm-predictor"
COMMON="--data world_model/data_abs --tags sac_abs random_abs \
        --wandb-project $WANDB_PROJECT \
        --epochs 100 --batch-size 256 --lr 3e-4 --pos-weight 1.0 --device cpu \
        --strategy tf_ratio --ss-epochs 30 --ss-min-ratio 0.0"

[[ "$1" == "--no-wandb" ]] && COMMON="$COMMON --no-wandb"

FAILED=()
run() {
    echo ""; echo "▶ $*"
    "$@"
    local code=$?
    [[ $code -ne 0 ]] && { echo "✗ FAILED: $*"; FAILED+=("$*"); } || echo "✓ done"
}

# ── 1-layer h=256 (baseline, 이미 돌렸지만 비교 기준) ──
ONE="--ctx-hidden 128 --lstm-hidden 256 --lstm-layers 1"
run python -m world_model.train_predictor --model lstm $ONE $COMMON --seed 0
run python -m world_model.train_predictor --model lstm $ONE $COMMON --seed 1
run python -m world_model.train_predictor --model lstm $ONE $COMMON --seed 2

# ── 2-layer h=148 ctx=120 (동일 파라미터, 계층 구조) ──
TWO="--ctx-hidden 120 --lstm-hidden 148 --lstm-layers 2"
run python -m world_model.train_predictor --model lstm $TWO $COMMON --seed 0
run python -m world_model.train_predictor --model lstm $TWO $COMMON --seed 1
run python -m world_model.train_predictor --model lstm $TWO $COMMON --seed 2

echo ""
echo "══════════════════════════════════════════"
echo "  Result: ${#FAILED[@]} failed"
[[ ${#FAILED[@]} -gt 0 ]] && for f in "${FAILED[@]}"; do echo "    - $f"; done
echo "══════════════════════════════════════════"
