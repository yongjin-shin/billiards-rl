#!/usr/bin/env bash
set +e  # 한 실험 실패해도 계속 진행

# ── World Model Predictor 비교 실험 ──────────────────────────────────────────
#
# 비교축 1) 모델 아키텍처 : MLP  vs  LSTM
# 비교축 2) LSTM 학습 전략: curriculum  vs  tf_ratio
#
# 실행:
#   bash world_model/run_predictor.sh
#   bash world_model/run_predictor.sh --no-wandb   (로컬 디버그)
# ─────────────────────────────────────────────────────────────────────────────

WANDB_PROJECT="billiards-wm-predictor"
DATA="world_model/data_abs"
TAGS="sac_abs random_abs"

COMMON="--data $DATA --tags $TAGS --wandb-project $WANDB_PROJECT --epochs 100 --batch-size 256 --lr 3e-4 --pos-weight 1.0 --device cpu"

# wandb 끄기 옵션
if [[ "$1" == "--no-wandb" ]]; then
    COMMON="$COMMON --no-wandb"
fi

FAILED=()

run() {
    echo ""
    echo "▶ $*"
    "$@"
    local code=$?
    if [ $code -ne 0 ]; then
        echo "✗ FAILED (exit $code): $*"
        FAILED+=("$*")
    else
        echo "✓ done: $*"
    fi
}

# ── MLP ───────────────────────────────────────────────────────────────────────
MLP_ARCH="--hidden 256 512 256"

run python -m world_model.train_predictor --model mlp $MLP_ARCH $COMMON --seed 0
run python -m world_model.train_predictor --model mlp $MLP_ARCH $COMMON --seed 1
run python -m world_model.train_predictor --model mlp $MLP_ARCH $COMMON --seed 2

# ── LSTM + curriculum (기본 전략) ─────────────────────────────────────────────
LSTM_ARCH="--ctx-hidden 128 --lstm-hidden 256 --lstm-layers 1"
CURRICULUM="--strategy curriculum --curriculum-epochs 80"

run python -m world_model.train_predictor --model lstm $LSTM_ARCH $CURRICULUM $COMMON --seed 0
run python -m world_model.train_predictor --model lstm $LSTM_ARCH $CURRICULUM $COMMON --seed 1
run python -m world_model.train_predictor --model lstm $LSTM_ARCH $CURRICULUM $COMMON --seed 2

# ── LSTM + tf_ratio (scheduled sampling, 비교용) ──────────────────────────────
TF_RATIO="--strategy tf_ratio --ss-epochs 30 --ss-min-ratio 0.0"

run python -m world_model.train_predictor --model lstm $LSTM_ARCH $TF_RATIO $COMMON --seed 0
run python -m world_model.train_predictor --model lstm $LSTM_ARCH $TF_RATIO $COMMON --seed 1
run python -m world_model.train_predictor --model lstm $LSTM_ARCH $TF_RATIO $COMMON --seed 2

# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════"
echo "  All experiments finished."
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Result: all passed ✓"
else
    echo "  Result: ${#FAILED[@]} failed ✗"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "══════════════════════════════════════════════════════"
