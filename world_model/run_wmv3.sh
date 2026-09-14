#!/usr/bin/env bash
set +e

# ── WMPredictor v3 실험 ───────────────────────────────────────────────────────
#
# 아키텍처 변경사항 (v2 대비):
#   [1] Decoder 입력에 Δpos 추가: [E[type] | cue | tgt | Δcue | Δtgt]  (d+4 → d+8)
#   [2] Encoder MLP 확장:  128→128  →  128→256  (h0/c0 병목 완화)
#   [3] LSTM 2-layer + dropout=0.1  (계층적 transition 학습)
#   [4] pos → event 순서 변경:
#         hidden → pos_head(H→4) → pos
#         [hidden | pos] → event_head(H+4→K) → type
#
# 비교축:
#   lstm_hidden : 256 vs 512
#   lr          : 3e-4 vs 1e-3
#
# 고정:
#   no-aug, λ_pos=10, label_smoothing=0.1
#   enc_hidden=128 256, lstm_layers=2, lstm_dropout=0.1
#   OneCycleLR (pct_start=0.1, final_div_factor=1e3)
#
# 실행:
#   bash world_model/run_wmv3.sh
#   bash world_model/run_wmv3.sh --no-wandb
# ─────────────────────────────────────────────────────────────────────────────

[[ "$1" == "--no-wandb" ]] && NO_WANDB="--no-wandb" || NO_WANDB=""

WANDB_PROJECT="billiards-wm-v2"
COMMON="--data world_model/data_v2 --tags sac_abs random_abs \
        --wandb-project $WANDB_PROJECT \
        --epochs 150 --batch-size 256 \
        --lambda-event 1.0 --lambda-pos 1.0 --label-smoothing 0.1 \
        --ss-epochs 60 --ss-min-ratio 0.0 \
        --enc-hidden 128 256 --lstm-layers 2 --lstm-dropout 0.1 \
        --event-embed-dim 32 \
        --pct-start 0.1 --final-div-factor 1e3 \
        --no-augment \
        --device cpu \
        $NO_WANDB"

FAILED=()
run() {
    echo ""; echo "▶ $*"
    "$@"
    local code=$?
    [[ $code -ne 0 ]] && { echo "✗ FAILED: $*"; FAILED+=("$*"); } || echo "✓ done"
}

# ── h=256, lr=3e-4 ────────────────────────────────────────────────────────────
echo "══ h=256  lr=3e-4 ══"
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --lr 3e-4 --seed 0
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --lr 3e-4 --seed 1
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --lr 3e-4 --seed 2

# ── h=256, lr=1e-3 ────────────────────────────────────────────────────────────
echo "══ h=256  lr=1e-3 ══"
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --lr 1e-3 --seed 0
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --lr 1e-3 --seed 1
run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 256 --lr 1e-3 --seed 2

# ── h=512, lr=3e-4 ────────────────────────────────────────────────────────────
# echo "══ h=512  lr=3e-4 ══"
# run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --lr 3e-4 --seed 0
# run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --lr 3e-4 --seed 1
# run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --lr 3e-4 --seed 2

# ── h=512, lr=1e-3 ────────────────────────────────────────────────────────────
# echo "══ h=512  lr=1e-3 ══"
# run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --lr 1e-3 --seed 0
# run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --lr 1e-3 --seed 1
# run python -m world_model.train_wm_predictor $COMMON --lstm-hidden 512 --lr 1e-3 --seed 2

echo ""
echo "══════════════════════════════════════════════════"
echo "  All done.  ${#FAILED[@]} failed."
[[ ${#FAILED[@]} -gt 0 ]] && for f in "${FAILED[@]}"; do echo "    - $f"; done
echo "══════════════════════════════════════════════════"
