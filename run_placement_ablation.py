"""
run_placement_ablation.py — Phase 0 성능 하락 원인 격리 실험.

발견된 원인:
  1. Scratch penalty (-0.5) — 원본에는 없었음. random action의 ~26%가 scratch
     → expected reward가 +0.026 → -0.099로 역전. SAC가 scratch 회피를 학습.
  2. Ball placement 범위 — 89431bd 커밋에서 target y: [0.6,0.9] → [0.30,0.85]로 확대

수정:
  - n_balls=1에서는 scratch penalty 적용 안 함 (원본 동작 복원)
  - legacy_placement=True 로 배치 범위도 원본으로 복원

비교 조건 (모두 SAC, 1M steps, seed=42, sp=0.0):
  A) legacy  placement  (cue y∈[0.2,0.4], target y∈[0.6,0.9])  ← 원본 완전 재현
  B) current placement  (cue y∈[0.15,0.4], target y∈[0.30,0.85])  ← 현재 기본
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

    for label, legacy in [("legacy (원본 배치)", True), ("current (현재 배치)", False)]:
        print("\n" + "=" * 60)
        print(f"  Condition: {label}")
        print(f"  legacy_placement={legacy}  |  scratch_penalty=disabled(n_balls=1)")
        print(f"  {ALGO}, n_balls=1, {STEPS//1000}k steps, seed={SEED}")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

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
    print("\n" + "=" * 60)
    print("  PLACEMENT ABLATION — SUMMARY (scratch penalty 비활성화 후)")
    print("=" * 60)
    print(f"  [원본 배치] legacy : {results['legacy']['pocket_rate']:.1f}%"
          f"  (random {results['legacy']['random_rate']:.1f}%)")
    print(f"  [현재 배치] current: {results['current']['pocket_rate']:.1f}%"
          f"  (random {results['current']['random_rate']:.1f}%)")
    delta = results["legacy"]["pocket_rate"] - results["current"]["pocket_rate"]
    print(f"  Δ = {delta:+.1f}pp")
    print()
    print("  비교 기준 (이전 결과):")
    print("  - 원본 Exp-01 (legacy + no scratch): 77.6%  ← 목표")
    print("  - run_phase0 (current + scratch):    42.4%  ← 기존 낮은 결과")
    print("=" * 60)


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
