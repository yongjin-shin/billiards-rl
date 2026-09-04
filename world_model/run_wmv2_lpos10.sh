#!/usr/bin/env bash
set +e

# ── WMPredictor v2 : lambda_pos=10 실험 ──────────────────────────────────────
#
# 이전 실험 결론:
#   - h=512 > h=256
#   - no-aug > aug  (TB flip의 angle 처리 이슈)
#
# 이번 실험:
#   - h=512 + no-aug (현재 best 설정)
#   - lambda_pos: 1.0 → 10.0  (position MSE 가중치 강화)
#
# 실행:
#   bash world_model/run_wmv2_lpos10.sh
#   bash world_model/run_wmv2_lpos10.sh --no-wandb
# ─────────────────────────────────────────────────────────────────────────────

[[ "$1" == "--no-wandb" ]] && NO_WANDB="--no-wandb" || NO_WANDB=""

WANDB_PROJECT="billiards-wm-v2"
COMMON="--data world_model/data_v2 --tags sac_abs random_abs \
        --wandb-project $WANDB_PROJECT \
        --epochs 150 --batch-size 256 --lr 3e-4 \
        --lambda-event 1.0 --lambda-pos 10.0 \
        --ss-epochs 60 --ss-min-ratio 0.0 \
        --enc-hidden 128 128 --lstm-hidden 256 --lstm-layers 1 --event-embed-dim 32 \
        --no-augment \
        --device cpu \
        --lr 1e-3 --pct-start 0.1 --final-div-factor 1e3 \
        $NO_WANDB"

FAILED=()
run() {
    echo ""; echo "▶ $*"
    "$@"
    local code=$?
    [[ $code -ne 0 ]] && { echo "✗ FAILED: $*"; FAILED+=("$*"); } || echo "✓ done"
}

echo "══ h=512 + no-aug + lambda_pos=10 ══"
run python -m world_model.train_wm_predictor $COMMON --seed 0
run python -m world_model.train_wm_predictor $COMMON --seed 1
run python -m world_model.train_wm_predictor $COMMON --seed 2

echo ""
echo "══════════════════════════════════════════"
echo "  All done. ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -gt 0 ]] && for f in "${FAILED[@]}"; do echo "    - $f"; done
echo "══════════════════════════════════════════"
