"""
exp16_wm/relabel.py — relabel existing teacher data with the robust teacher.

Shots start with every ball at rest, so a stored observation fixes the table:
restore_state() rebuilds it in a live env (on-table balls at their positions,
pocketed balls below the table with pooltool's pocketed state — they take no
part in a simulation). Verified before use: 340/340 actions gave the same balls
pocketed + scratch on the restored table as on the original (85 states, 55 with
pocketed balls; experiments.md 2026-10-09).

Same states, new labels: each shard is written twice with identical rows and game
ids — robust labels and the original labels — so a BC comparison differs only in
the labels (identical train/val split).

Usage:
    python -m exp16_wm.relabel --data logs/teacher_data/snap_1014/ghost_s1000.npz \
        --shard 0 --n-shards 8 --out-dir logs/teacher_data/relabel_s1000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.networks import TABLE_W, TABLE_L
from exp16_wm.teacher_data import TEACHERS, CLEAR_ENV, robust_shot

_POCKETED_POS = [-0.0295, -0.0295, -0.08]     # below the table, where pooltool parks pocketed balls
KEYS = ["obs", "action", "good_actions", "n_good", "pocketed", "game", "shot", "win_width"]


def restore_state(env: BilliardsEnv, obs: np.ndarray) -> None:
    """Put the env's table into the at-rest state described by a raw (23,) observation."""
    env.reset()
    for k, bid in enumerate(["cue"] + env._ball_ids):
        b = env.system.balls[bid]
        if bid == "cue":
            x, y, pk = obs[0], obs[1], 0.0
        else:
            i = 2 + 3 * (k - 1)
            x, y, pk = obs[i], obs[i + 1], obs[i + 2]
        b.state.rvw[1] = 0.0
        b.state.rvw[2] = 0.0
        if pk > 0.5:
            b.state.s = pt.constants.pocketed
            b.state.rvw[0] = _POCKETED_POS
        else:
            b.state.s = pt.constants.stationary
            b.state.rvw[0] = [float(x) * TABLE_W, float(y) * TABLE_L, b.params.R]
        if bid != "cue":
            env._pocketed[bid] = pk > 0.5


def shard_rows(game: np.ndarray, shard: int, n_shards: int) -> np.ndarray:
    """Row indices whose game belongs to this shard (whole games stay together)."""
    return np.where(game % n_shards == shard)[0]


def relabel_rows(data: dict[str, np.ndarray], rows: np.ndarray, seed: int, max_good: int,
                 log_every: int = 500) -> dict[str, np.ndarray]:
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = {"action": [], "good_actions": [], "n_good": [], "win_width": [], "pocketed": []}
    t0 = time.time()
    for n, r in enumerate(rows):
        restore_state(env, data["obs"][r])
        action, good, n_good, width = robust_shot(env, rng, TEACHERS["robust"], max_good)
        out["action"].append(action)
        out["good_actions"].append(good)
        out["n_good"].append(n_good)
        out["win_width"].append(width)
        out["pocketed"].append(1 if width > 0 else int(data["pocketed"][r] > 0))
        if log_every and (n + 1) % log_every == 0:
            print(f"  {n + 1}/{len(rows)} states  {time.time() - t0:.0f}s  "
                  f"median window {np.median(out['win_width']):.3f}°", flush=True)
    env.close()
    return {
        "obs": data["obs"][rows], "game": data["game"][rows], "shot": data["shot"][rows],
        "action": np.asarray(out["action"], np.float32),
        "good_actions": np.asarray(out["good_actions"], np.float32),
        "n_good": np.asarray(out["n_good"], np.int32),
        "win_width": np.asarray(out["win_width"], np.float32),
        "pocketed": np.asarray(out["pocketed"], np.int32),
    }


def original_rows(data: dict[str, np.ndarray], rows: np.ndarray, max_good: int) -> dict[str, np.ndarray]:
    """The same rows with their original labels (good_actions cut/padded to max_good)."""
    g = np.full((len(rows), max_good, 2), np.nan, np.float32)
    k = min(max_good, data["good_actions"].shape[1])
    g[:, :k] = data["good_actions"][rows, :k]
    return {"obs": data["obs"][rows], "game": data["game"][rows], "shot": data["shot"][rows],
            "action": data["action"][rows], "good_actions": g, "n_good": data["n_good"][rows],
            "pocketed": data["pocketed"][rows],
            "win_width": np.full(len(rows), np.nan, np.float32)}


def write(path: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     type=str, required=True)
    p.add_argument("--shard",    type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--max-good", type=int, default=8)
    p.add_argument("--out-dir",  type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    data = dict(np.load(args.data))
    rows = shard_rows(data["game"], args.shard, args.n_shards)
    print(f"relabel {args.data}  shard {args.shard}/{args.n_shards}: {len(rows)} states", flush=True)
    src = json.loads(str(data["meta"]))
    new = relabel_rows(data, rows, seed=10_000 + args.shard, max_good=args.max_good)
    base = os.path.splitext(os.path.basename(args.data))[0]
    write(os.path.join(args.out_dir, "robust", f"{base}_shard{args.shard}.npz"), new,
          {"teacher": "robust_relabel", "source": args.data, "plan": TEACHERS["robust"], "env": src.get("env")})
    write(os.path.join(args.out_dir, "orig", f"{base}_shard{args.shard}.npz"),
          original_rows(data, rows, args.max_good),
          {"teacher": f"{src.get('teacher')}_same_states", "source": args.data, "env": src.get("env")})
    w = new["win_width"]
    print(f"done: {len(rows)} states, window median {np.median(w):.3f}°, "
          f"speed median {np.median(new['action'][:, 1]):.2f} (orig {np.median(data['action'][rows, 1]):.2f}) m/s",
          flush=True)


if __name__ == "__main__":
    main()
