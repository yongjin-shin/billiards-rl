"""
run_placement_ablation.py — Ball placement 범위가 Phase 0 성능에 미치는 영향 검증.

가설: 89431bd 커밋에서 n_balls=1 배치 범위가 바뀌면서 77.6%→42.4% 하락.

비교 대상:
  A) legacy_placement=True  : cue y∈[0.2,0.4], target y∈[0.6,0.9]  (원본 Exp-01 조건)
  B) legacy_placement=False : cue y∈[0.15,0.40], target y∈[0.30,0.85] (현재 기본)

두 조건 모두 동일 설정:
  SAC, n_balls=1, 1M steps, seed=42, step_penalty=0.0, trunc_penalty=0.0

결과는 콘솔 + logs/placement_ablation_<ts>.log + 각 exp_dir/results.json 에 저장.
"""

import json
import os
import sys
import time

from train import train, _Tee

STEPS = 1_000_000
SEED  = 42
ALGO  = "SAC"


def main():
    results = {}

    for label, legacy in [("legacy (원본)", True), ("current (현재)", False)]:
        print("\n" + "=" * 55)
        print(f"  Placement: {label}")
        print(f"  legacy_placement={legacy}")
        print(f"  {ALGO}, n_balls=1, {STEPS//1000}k steps, seed={SEED}")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

        exp_dir = train(
            algo             = ALGO,
            steps            = STEPS,
            seed             = SEED,
            n_balls          = 1,
            max_steps        = 1,
            step_penalty     = 0.0,
            trunc_penalty    = 0.0,
            abs_angle        = False,
            legacy_placement = legacy,
        )

        res_path = os.path.join(exp_dir, "results.json")
        with open(res_path) as f:
            res = json.load(f)

        key = "legacy" if legacy else "current"
        results[key] = {
            "pocket_rate" : res["trained_pocket_rate"],
            "random_rate" : res["random_pocket_rate"],
            "exp_dir"     : exp_dir,
        }
        print(f"\n  [{label}] pocket={res['trained_pocket_rate']:.1f}%  "
              f"random={res['random_pocket_rate']:.1f}%")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PLACEMENT ABLATION — SUMMARY")
    print("=" * 55)
    print(f"  Legacy (원본 Exp-01 조건): {results['legacy']['pocket_rate']:.1f}%"
          f"  (random {results['legacy']['random_rate']:.1f}%)")
    print(f"  Current (현재 기본):       {results['current']['pocket_rate']:.1f}%"
          f"  (random {results['current']['random_rate']:.1f}%)")
    delta = results["legacy"]["pocket_rate"] - results["current"]["pocket_rate"]
    print(f"  Δ = {delta:+.1f}pp  ({'legacy 우세' if delta > 0 else 'current 우세'})")
    print("=" * 55)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/placement_ablation_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _tee = _Tee(sys.stdout, log_file)
    sys.stdout = sys.stderr = _tee
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _tee.close()
