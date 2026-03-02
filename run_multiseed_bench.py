"""
run_multiseed_bench.py -- SAC / TQC / PPO x 5 seeds, ms=3, 3M steps

Phase 2 baseline benchmark.
Reward: sp=0.1, tp=1.0 (simple, no pp/st)
Total: 3 algos x 5 seeds = 15 sequential runs.

Usage:
    source .venv/bin/activate
    python run_multiseed_bench.py                   # foreground
    nohup python run_multiseed_bench.py > logs/multiseed_bench.log 2>&1 &
"""

import sys
import time
from train import train

STEPS       = 3_000_000
MAX_STEPS   = 3
SP          = 0.1
TP          = 1.0
SEEDS       = [0, 1, 2, 3, 42]
ALGOS       = ["SAC", "TQC", "PPO"]

def main():
    runs = [(algo, seed) for algo in ALGOS for seed in SEEDS]
    total = len(runs)

    print("=" * 56)
    print("  Multi-seed benchmark: ms=3, 3M steps")
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
            )
            print(f"  -> saved: {exp_dir}")
        except Exception as e:
            print(f"  ERROR in {algo} seed={seed}: {e}", file=sys.stderr)

    print("\n" + "=" * 56)
    print(f"  All {total} runs complete.")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 56)

if __name__ == "__main__":
    main()
