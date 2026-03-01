"""
benchmark.py — Find the fastest training configuration for this machine.

Tests all combinations of:
  - vec_env:  DummyVecEnv vs SubprocVecEnv
  - device:   mps vs cpu
  - n_envs:   4, 8, 10

Each combo runs 5,000 timesteps and measures steps/sec.
Prints a ranked table at the end.

Usage:
    source .venv/bin/activate
    python benchmark.py
"""

import time
import sys
sys.path.insert(0, ".")

from simulator import BilliardsEnv


TIMESTEPS   = 5_000
N_ENVS_LIST = [4, 8, 10]
DEVICES     = ["cpu", "mps"]
VEC_TYPES   = ["dummy", "subproc"]


def run_one(vec_type, device, n_envs):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if device == "mps" and not torch.backends.mps.is_available():
        return None

    cls = SubprocVecEnv if vec_type == "subproc" else None
    try:
        vec_env = make_vec_env(
            BilliardsEnv,
            n_envs=n_envs,
            vec_env_cls=cls,
        )
        model = SAC(
            "MlpPolicy",
            vec_env,
            device=device,
            verbose=0,
            learning_starts=500,
            batch_size=256,
            buffer_size=10_000,
        )
        t0 = time.time()
        model.learn(total_timesteps=TIMESTEPS)
        elapsed = time.time() - t0
        vec_env.close()
        return TIMESTEPS / elapsed
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print(f"=== benchmark — {TIMESTEPS:,} steps per config ===\n")
    print("Warming up Numba JIT (first simulate is slow)...")
    warmup_env = BilliardsEnv()
    warmup_env.reset()
    warmup_env.step(warmup_env.action_space.sample())
    print("Warmup done.\n")

    results = []
    total = len(VEC_TYPES) * len(DEVICES) * len(N_ENVS_LIST)
    i = 0

    for vec_type in VEC_TYPES:
        for device in DEVICES:
            for n_envs in N_ENVS_LIST:
                i += 1
                label = f"{vec_type:7s} | {device:3s} | n_envs={n_envs}"
                print(f"[{i:2d}/{total}] {label} ... ", end="", flush=True)
                sps = run_one(vec_type, device, n_envs)
                if sps is None:
                    print("skipped (MPS unavailable)")
                elif isinstance(sps, str):
                    print(sps)
                    results.append((label, -1))
                else:
                    print(f"{sps:,.0f} steps/sec")
                    results.append((label, sps))

    # --- Ranked summary -------------------------------------------------------
    valid = [(l, s) for l, s in results if isinstance(s, float) and s > 0]
    valid.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 52)
    print("  RESULTS  (ranked fastest → slowest)")
    print("=" * 52)
    best = valid[0][1] if valid else 1
    for rank, (label, sps) in enumerate(valid, 1):
        bar = "█" * int(sps / best * 20)
        print(f"  #{rank}  {label}  {sps:>7,.0f} sps  {bar}")
    print("=" * 52)
    if valid:
        print(f"\n  Best config: {valid[0][0]}")
        print(f"  Worst config: {valid[-1][0]}")
        print(f"  Speedup (best/worst): {valid[0][1]/valid[-1][1]:.1f}x")


if __name__ == "__main__":
    main()
