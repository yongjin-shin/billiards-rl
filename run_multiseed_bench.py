"""
run_multiseed_bench.py -- SAC / TQC / PPO x 3 seeds, ms=3, 2M steps

Phase 2 baseline benchmark.
Reward: sp=0.1, tp=1.0 (simple, no pp/st)
Total: 3 algos x 3 seeds = 9 sequential runs.

Logs are written automatically:
  logs/bench_YYYYMMDD_HHMMSS.log  ← benchmark-level (this script)
  logs/experiments/{run}/train.log ← per-experiment (inside train())

Usage:
    python run_multiseed_bench.py          # foreground (logs saved automatically)
    python run_multiseed_bench.py &        # background (no redirection needed)
"""

import os
import sys
import time
from train import train, _Tee

STEPS       = 2_000_000
MAX_STEPS   = 3
SP          = 0.1
TP          = 1.0
SEEDS       = [0, 1, 42]
ALGOS       = ["SAC", "TQC", "PPO"]
ABS_ANGLE   = False     # Exp-12: set True to use absolute angle [0, 2π]

def main():
    runs = [(algo, seed) for algo in ALGOS for seed in SEEDS]
    total = len(runs)

    print("=" * 56)
    print("  Multi-seed benchmark: ms=3, 2M steps")
    print(f"  Algos: {ALGOS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 56)

    for i, (algo, seed) in enumerate(runs, 1):
        print(f"\n{'--'*28}")
        print(f"  [{i}/{total}]  {algo}  seed={seed}")
        print(f"  {time.strftime('%H:%M:%S')}")
        print(f"{'--'*28}")
        
        try:
            exp_dir = train(
                algo          = algo,
                steps         = STEPS,
                seed          = seed,
                n_balls       = 3,
                max_steps     = MAX_STEPS,
                step_penalty  = SP,
                trunc_penalty = TP,
                abs_angle     = ABS_ANGLE,
            )
            print(f"  -> saved: {exp_dir}")
        except Exception as e:
            print(f"  ERROR in {algo} seed={seed}: {e}", file=sys.stderr)

    print("\n" + "=" * 56)
    print(f"  All {total} runs complete.")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 56)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    bench_log = f"logs/bench_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _tee = _Tee(sys.stdout, bench_log)
    sys.stdout = sys.stderr = _tee
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _tee.close()
