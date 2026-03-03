"""
run_exp11_ms2_extend.py -- Exp-11 extension: Stage 4 ms=2

현재 curriculum (ms=5→4→3) 결과에서 stage3 best_model을 로드해
ms=2 (500k steps)로 추가 학습.

Purpose: ms=2 학습 가능성 탐색.
  clear = 2샷 3포켓 → 적어도 1번 더블 포켓 필요
  → 학습 신호 거의 없을 것으로 예상 (검증 목적)
"""

import glob
import os
import sys
import time

from train import _Tee
from train_curriculum import train_stage

STEPS4 = 500_000
SEED   = 42


def find_stage3_best():
    """가장 최근 ms5-4-3 curriculum 실험에서 stage3 best_model 경로 반환."""
    pattern = "logs/experiments/SAC_curriculum_ms5-4-3_*"
    dirs = sorted(glob.glob(pattern), reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No curriculum dir found matching: {pattern}")
    parent_dir = dirs[0]
    # stage dir name: stage3_ms3_{steps//1000}k (steps=500k → 500k)
    best_path = os.path.join(parent_dir, "stage3_ms3_500k", "best_model", "best_model")
    if not os.path.exists(best_path + ".zip"):
        raise FileNotFoundError(f"stage3 best_model not found: {best_path}.zip")
    return parent_dir, best_path


def main():
    parent_dir, s3_best = find_stage3_best()

    print("=" * 55)
    print(f"  Exp-11 extension: Stage 4  ms=2")
    print(f"  parent_dir : {parent_dir}")
    print(f"  pretrained : {s3_best}")
    print(f"  steps      : {STEPS4:,}    seed: {SEED}")
    print(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    train_stage(
        stage          = 4,
        max_steps      = 2,
        steps          = STEPS4,
        seed           = SEED,
        parent_dir     = parent_dir,
        step_penalty   = 0.1,
        trunc_penalty  = 1.0,
        pretrained_path= s3_best,
    )

    print(f"\n  Done: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/exp11_ms2_extend_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _tee = _Tee(sys.stdout, log_file)
    sys.stdout = sys.stderr = _tee
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _tee.close()
