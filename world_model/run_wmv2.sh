#!/usr/bin/env bash
set +e

# ── WMPredictor v2 비교 실험 ──────────────────────────────────────────────────
#
# 비교축 1) Data augmentation : aug vs no-aug
# 비교축 2) Model capacity    : h=256 vs h=512
#
# 실행:
#   bash world_model/run_wmv2.sh
#   bash world_model/run_wmv2.sh --no-wandb   (로컬 디버그)
# ─────────────────────────────────────────────────────────────────────────────

[[ "$1" == "--no-wandb" ]] && NO_WANDB="--no-wandb" || NO_WANDB=""

WANDB_PROJECT="billiards-wm-v2"
COMMON="--data world_model/data_v2 --tags sac_abs random_abs \
        --wandb-project $WANDB_PROJECT \
        --epochs 150 --batch-size 256 --lr 3e-4 \
        --lambda-event 1.0 --lambda-pos 1.0 \
        --ss-epochs 60 --ss-min-ratio 0.0 \
        --enc-hidden 128 128 --lstm-layers 1 --event-embed-dim 32 \
        --device cpu \
        $NO_WANDB"

FAILED=()
run() {
    echo ""; echo "▶ $*"
    "$@"
    local code=$?
    [[ $code -ne 0 ]] && { echo "✗ FAILED: $*"; FAILED+=("$*"); } || echo "✓ done"
}

# ── h=256: aug vs no-aug ──────────────────────────────────────────────────────
echo "══ h=256 + augmentation ══"
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --seed 0
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --seed 1
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --seed 2

echo "══ h=256 + no-augmentation ══"
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --no-augment --seed 0
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --no-augment --seed 1
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --no-augment --seed 2

# ── h=512: aug vs no-aug ──────────────────────────────────────────────────────
echo "══ h=512 + augmentation ══"
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --seed 0
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --seed 1
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --seed 2

echo "══ h=512 + no-augmentation ══"
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --no-augment --seed 0
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --no-augment --seed 1
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --no-augment --seed 2

echo ""
echo "══════════════════════════════════════════"
echo "  All done. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -gt 0 ]] && for f in "${FAILED[@]}"; do echo "    - $f"; done
echo "══════════════════════════════════════════"
