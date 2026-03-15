"""
run_exp12_aa_5M.py -- SAC × 3 seeds, abs_angle=True, 5M steps  (Exp-12 benchmark)

Exp-10 대비 비교:
  Exp-10 SAC: delta_angle, 2M steps  → pocket 41.7% / clear  8.4%
  Exp-12 SAC: abs_angle,   5M steps  → ?

설정: n_balls=3, ms=3, sp=0.1, tp=1.0
"""

import os
import sys
import time
import traceback
from train import train, _Tee

STEPS     = 5_000_000
SEEDS     = [0, 1, 42]
ALGO      = "SAC"
SP        = 0.1
TP        = 1.0
ABS_ANGLE = True


def main():
    total = len(SEEDS)

    print("=" * 56)
    print(f"  Exp-12 aa benchmark: {ALGO} × {total} seeds, 5M steps")
    print(f"  abs_angle=True  (baseline: Exp-10 delta_angle 2M)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 56)

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'--'*28}")
        print(f"  [{i}/{total}]  {ALGO}  seed={seed}")
        print(f"  {time.strftime('%H:%M:%S')}")
        print(f"{'--'*28}")
        try:
            exp_dir = train(
                algo          = ALGO,
                steps         = STEPS,
                seed          = seed,
                n_balls       = 3,
                max_steps     = 3,
                step_penalty  = SP,
                trunc_penalty = TP,
                abs_angle     = ABS_ANGLE,
            )
            print(f"  -> saved: {exp_dir}")
        except Exception as e:
            print(
                f"  ERROR in {ALGO} seed={seed}: {type(e).__name__}: {e}\n"
                + traceback.format_exc(),
                file=sys.stderr,
            )

    print("\n" + "=" * 56)
    print(f"  All {total} seeds complete.")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 56)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    bench_log = f"logs/exp12_aa_5M_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _tee = _Tee(sys.stdout, bench_log)
    sys.stdout = sys.stderr = _tee
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _tee.close()
