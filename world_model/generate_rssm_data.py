"""
world_model/generate_rssm_data.py

RSSM 전용 데이터 생성기.
충돌 이벤트만 저장 → 파일 크기 소형 유지.
ShotData 리스트를 pickle로 청크 저장.

Usage:
    # SAC policy (권장)
    python world_model/generate_rssm_data.py \
        --policy sac \
        --sac-model logs/experiments/<run>/best_model/best_model.zip \
        --n-episodes 50000 \
        --out-dir world_model/data_rssm

    # Random policy
    python world_model/generate_rssm_data.py \
        --policy random \
        --n-episodes 10000 \
        --out-dir world_model/data_rssm
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.rssm_dataset import generate_shot_data, ShotData


def _make_policy(args, env):
    if args.policy == "random":
        def policy(obs):
            return env.action_space.sample()
        return policy, "random"

    # SAC
    from stable_baselines3 import SAC
    model_path = args.sac_model
    if model_path is None:
        import glob
        paths = sorted(glob.glob(
            "logs/experiments/SAC_5000k_*/best_model/best_model.zip"
        ))
        assert paths, "SAC model not found — pass --sac-model"
        model_path = paths[-1]
    print(f"SAC: {model_path}")
    sac = SAC.load(model_path)

    def policy(obs):
        action, _ = sac.predict(obs, deterministic=False)
        return action

    return policy, "sac"


def generate(
    n_episodes  : int,
    policy_fn,
    n_balls     : int = 1,
    seed_start  : int = 0,
    report_every: int = 1000,
) -> list[ShotData]:
    from simulator import BilliardsEnv

    ball_ids = ["cue"] + [str(i) for i in range(1, n_balls + 1)]
    shots: list[ShotData] = []
    n_pocketed = 0
    total_events = 0
    t0 = time.time()

    env = BilliardsEnv(n_balls=n_balls)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_start + ep)
            action  = policy_fn(obs)
            env.step(action)

            shot = generate_shot_data(env.system, ball_ids)
            if not shot.event_steps:
                continue

            shots.append(shot)

            # pocket 여부: type 3 (pocket) 이벤트 존재 시
            has_pocket = any(
                s.event_type == 3 for s in shot.event_steps
            )
            if has_pocket:
                n_pocketed += 1
            total_events += len(shot.event_steps)

            if (ep + 1) % report_every == 0:
                elapsed = time.time() - t0
                pct     = 100 * n_pocketed / len(shots)
                avg_ev  = total_events / len(shots)
                eps_s   = (ep + 1) / elapsed
                remain  = (n_episodes - ep - 1) / eps_s
                ts      = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] {ep+1:>6}/{n_episodes}"
                    f"  shots={len(shots)}"
                    f"  pocketed={pct:.1f}%"
                    f"  avg_events={avg_ev:.1f}"
                    f"  {eps_s:.1f}ep/s"
                    f"  ETA={remain/60:.1f}min"
                )
    finally:
        env.close()

    return shots


def generate_balanced(
    quotas      : dict[int, int],
    policy_fn,
    n_balls     : int = 2,
    seed_start  : int = 0,
    max_attempts: int = 2_000_000,
    report_every: int = 5_000,
) -> list[ShotData]:
    """Rejection-sample episodes into buckets keyed by n_pocketed_targets().

    Keeps generating episodes until every bucket in `quotas` reaches its
    target count (or `max_attempts` episodes have been tried). Shots whose
    bucket is already full are discarded so rare buckets (e.g. 2-pocket)
    don't get drowned out by the dominant 0-pocket bucket.
    """
    from simulator import BilliardsEnv

    ball_ids = ["cue"] + [str(i) for i in range(1, n_balls + 1)]
    buckets: dict[int, list[ShotData]] = {k: [] for k in quotas}
    t0 = time.time()

    env = BilliardsEnv(n_balls=n_balls)
    try:
        ep = 0
        while ep < max_attempts:
            if all(len(buckets[k]) >= quotas[k] for k in quotas):
                break

            obs, _ = env.reset(seed=seed_start + ep)
            action  = policy_fn(obs)
            env.step(action)
            ep += 1

            shot = generate_shot_data(env.system, ball_ids)
            if not shot.event_steps:
                continue

            n_pocketed = shot.n_pocketed_targets()
            bucket = buckets.get(n_pocketed)
            if bucket is None or len(bucket) >= quotas[n_pocketed]:
                continue
            bucket.append(shot)

            if ep % report_every == 0:
                elapsed = time.time() - t0
                eps_s   = ep / elapsed
                ts      = datetime.now().strftime("%H:%M:%S")
                counts  = "  ".join(f"n={k}:{len(buckets[k])}/{quotas[k]}" for k in sorted(quotas))
                print(f"[{ts}] attempts={ep:>8}  {counts}  {eps_s:.1f}ep/s")
    finally:
        env.close()

    total_attempts = ep
    elapsed = time.time() - t0
    ts = datetime.now().strftime("%H:%M:%S")
    counts = "  ".join(f"n={k}:{len(buckets[k])}/{quotas[k]}" for k in sorted(quotas))
    print(f"[{ts}] done: attempts={total_attempts}  {counts}  elapsed={elapsed/60:.1f}min")

    shots: list[ShotData] = []
    for k in sorted(buckets):
        shots.extend(buckets[k])
    return shots


def save_chunks(shots: list[ShotData], out_dir: Path, tag: str, chunk_size: int):
    """Split shots into chunks and save as pickle files."""
    chunks = [shots[i:i+chunk_size] for i in range(0, len(shots), chunk_size)]
    saved = []
    for ci, chunk in enumerate(chunks):
        fname = f"{tag}_chunk{ci:03d}.pkl"
        path  = out_dir / fname
        with open(path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved.append(fname)
        print(f"  Saved chunk {ci}: {len(chunk)} shots → {path}")
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",      default="world_model/data_rssm")
    p.add_argument("--policy",       choices=["sac", "random"], default="sac")
    p.add_argument("--sac-model",    default=None)
    p.add_argument("--n-episodes",   type=int, default=50_000)
    p.add_argument("--n-balls",      type=int, default=1)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--chunk-size",   type=int, default=5_000)
    p.add_argument("--report-every", type=int, default=1_000)
    p.add_argument("--balanced",     action="store_true",
                    help="rejection-sample into pocket-count buckets instead of plain generation")
    p.add_argument("--quota0",       type=int, default=1000, help="quota for 0-pocket bucket")
    p.add_argument("--quota1",       type=int, default=1000, help="quota for 1-pocket bucket")
    p.add_argument("--quota2",       type=int, default=1000, help="quota for 2-pocket bucket")
    p.add_argument("--max-attempts", type=int, default=2_000_000, help="safety cap for --balanced")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from simulator import BilliardsEnv
    env_tmp = BilliardsEnv(n_balls=args.n_balls)
    policy_fn, policy_tag = _make_policy(args, env_tmp)
    env_tmp.close()

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.balanced:
        quotas = {0: args.quota0, 1: args.quota1, 2: args.quota2}
        tag = f"{policy_tag}_balanced_q{args.quota0}-{args.quota1}-{args.quota2}_s{args.seed}_{ts}"
        print(f"Generating balanced dataset  quotas={quotas}  policy={policy_tag}  n_balls={args.n_balls}")
        t0    = time.time()
        shots = generate_balanced(
            quotas       = quotas,
            policy_fn    = policy_fn,
            n_balls      = args.n_balls,
            seed_start   = args.seed,
            max_attempts = args.max_attempts,
            report_every = args.report_every,
        )
        elapsed = time.time() - t0
    else:
        tag = f"{policy_tag}_n{args.n_episodes}_s{args.seed}_{ts}"
        print(f"Generating {args.n_episodes} episodes  policy={policy_tag}  n_balls={args.n_balls}")
        t0    = time.time()
        shots = generate(
            n_episodes   = args.n_episodes,
            policy_fn    = policy_fn,
            n_balls      = args.n_balls,
            seed_start   = args.seed,
            report_every = args.report_every,
        )
        elapsed = time.time() - t0

    # Stats
    n_pocket = sum(
        1 for s in shots
        if any(e.event_type == 3 for e in s.event_steps)
    )
    total_ev = sum(len(s.event_steps) for s in shots)
    print(f"\nDone in {elapsed/60:.1f}min")
    print(f"  Valid shots : {len(shots):,} / {args.n_episodes:,}")
    print(f"  Pocketed    : {n_pocket:,} ({100*n_pocket/len(shots):.1f}%)")
    print(f"  Total events: {total_ev:,}  avg={total_ev/len(shots):.1f}/shot")

    # Save
    saved_files = save_chunks(shots, out_dir, tag, args.chunk_size)

    # Metadata
    meta_path = out_dir / "metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else []
    meta.append({
        "tag"          : tag,
        "policy"       : policy_tag,
        "sac_model"    : args.sac_model,
        "n_episodes"   : args.n_episodes,
        "n_valid_shots": len(shots),
        "n_pocketed"   : n_pocket,
        "pocketed_pct" : round(100 * n_pocket / len(shots), 2),
        "total_events" : total_ev,
        "avg_events"   : round(total_ev / len(shots), 2),
        "n_balls"      : args.n_balls,
        "seed"         : args.seed,
        "elapsed_min"  : round(elapsed / 60, 1),
        "files"        : saved_files,
    })
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
