"""
run_phase0.py — Phase 0 재훈련: SAC, n_balls=1, 1M steps

Exp-13a/b/c의 선행 조건.
Exp-01 weight 유실로 재훈련 필요. obs/action 설계는 그대로.

  obs  : 16-dim  [cue_x, cue_y, ball_x, ball_y, p0x,p0y, …, p5x,p5y]
  act  : delta_angle [-π, π], speed [0.5, 8.0]
  ep   : horizon=1 (공 1개, 샷 1번으로 종료)

완료 후 best_model 경로를 콘솔과 phase0_result.json에 출력.
"""

import os
import sys
import time

from train import train, _Tee

STEPS = 1_000_000
SEED  = 42
ALGO  = "SAC"


def main():
    print("=" * 55)
    print(f"  Phase 0 재훈련: {ALGO}, n_balls=1, {STEPS//1000}k steps, seed={SEED}")
    print(f"  obs: 16-dim | action: delta_angle + speed")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    exp_dir = train(
        algo          = ALGO,
        steps         = STEPS,
        seed          = SEED,
        n_balls       = 1,
        max_steps     = 1,       # ignored for n_balls=1 (horizon=1)
        step_penalty  = 0.0,     # no penalty (single shot, no urgency needed)
        trunc_penalty = 0.0,
        abs_angle     = False,   # delta_angle 유지
    )

    best_model = os.path.join(exp_dir, "best_model", "best_model")
    print(f"\n  Phase 0 완료!")
    print(f"  best_model → {best_model}")
    print(f"\n  Exp-13a 실행 시 이 경로를 Phase 0 weight로 사용:")
    print(f"    PHASE0_PATH = \"{best_model}\"")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/phase0_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _tee = _Tee(sys.stdout, log_file)
    sys.stdout = sys.stderr = _tee
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _tee.close()
