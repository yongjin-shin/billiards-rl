"""
exp16_wm/two_ball_states.py — states with exactly two balls on the table (Exp-16).

Real 2-ball states (the second shot of a game, after the first shot pocketed one
ball) are only 16% of the teacher data, and the BC policy is weakest there
(experiments.md 2026-10-11). This makes extra ones: every state is a fresh table
with one random ball slot pocketed — a "game that starts with two balls"; when the
teacher pockets one of them the game would end at one ball, so a new start is drawn
instead. Output has the teacher_data/relabel input layout (obs, game, shot, …) with
placeholder labels, so the usual pipeline labels it:
relabel (robust) → pair_labels → outcomes → pair_stats.

  placement "env"    : BilliardsEnv.reset() positions (balls in the table's interior)
  placement "uniform": cue + 2 balls uniform over the whole table (cushion/pocket-adjacent
                       spots included, like real post-shot positions)

Usage:
    python -m exp16_wm.two_ball_states --placement env --n 7500 --seed 2000 \
        --out logs/teacher_data/two_ball_A/states.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.obs_transform import N_BALLS

POCKET_CLEARANCE = 0.10      # m — uniform placement keeps ball centres this far from pocket centres
GAP = 0.005                  # m — uniform placement: extra space between balls / from the cushion


def _uniform_positions(env: BilliardsEnv, rng: np.random.Generator, n: int) -> np.ndarray:
    """n non-overlapping ball centres (m) anywhere on the playing surface, clear of the pockets."""
    R = env.system.balls["cue"].params.R
    W, L = env.table_width, env.table_length
    pockets = env._pocket_obs.reshape(-1, 2) * np.array([W, L])
    out: list[np.ndarray] = []
    while len(out) < n:
        p = rng.uniform([R + GAP, R + GAP], [W - R - GAP, L - R - GAP])
        if np.min(np.linalg.norm(pockets - p, axis=1)) < POCKET_CLEARANCE:
            continue
        if all(np.linalg.norm(p - q) > 2 * R + GAP for q in out):
            out.append(p)
    return np.asarray(out)


def make_states(n: int, seed: int, placement: str) -> np.ndarray:
    """(n, 23) raw observations, each with exactly one of the three ball slots pocketed."""
    if placement not in ("env", "uniform"):
        raise ValueError(f"unknown placement: {placement}")
    if CLEAR_ENV["n_balls"] != N_BALLS:
        raise ValueError("expects the 3-ball env")
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    W, L = env.table_width, env.table_length
    obs = np.zeros((n, 2 + 3 * N_BALLS + 2 * env._pocket_obs.size // 2), np.float32)
    for i in range(n):
        o = np.asarray(env.reset()[0], np.float32)
        if placement == "uniform":
            pos = _uniform_positions(env, rng, 1 + N_BALLS) / np.array([W, L])
            o[0:2] = pos[0]
            for k in range(N_BALLS):
                o[2 + 3 * k: 4 + 3 * k] = pos[1 + k]
        gone = int(rng.integers(N_BALLS))
        o[2 + 3 * gone: 5 + 3 * gone] = (0.0, 0.0, 1.0)
        obs[i] = o
    env.close()
    return obs


def to_relabel_input(obs: np.ndarray) -> dict[str, np.ndarray]:
    """The arrays relabel.py reads; labels are placeholders (one game per state, shot 1)."""
    n = len(obs)
    return {"obs": obs, "action": np.zeros((n, 2), np.float32),
            "good_actions": np.full((n, 1, 2), np.nan, np.float32), "n_good": np.zeros(n, np.int32),
            "pocketed": np.zeros(n, np.int32), "game": np.arange(n, dtype=np.int32),
            "shot": np.ones(n, np.int32), "win_width": np.full(n, np.nan, np.float32)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--placement", type=str, required=True, choices=["env", "uniform"])
    p.add_argument("--n",         type=int, default=7500)
    p.add_argument("--seed",      type=int, default=2000)
    p.add_argument("--out",       type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    obs = make_states(args.n, args.seed, args.placement)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = {"teacher": "none (two-ball states)", "placement": args.placement, "seed": args.seed, "env": CLEAR_ENV}
    np.savez_compressed(args.out, meta=np.array(json.dumps(meta)), **to_relabel_input(obs))
    print(f"{args.n} two-ball states ({args.placement}) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
