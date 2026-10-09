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

Every simulation the robust teacher runs (~300 per state, hits and misses) is also
kept as a labelled (state, action) → outcome sample for a learned success map:
sims/<file>_shard<i>.npz with sim_state (row into the shard's arrays), sim_action
(delta_angle, speed), sim_pocketed (balls), sim_scratch.

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
from exp16_wm.teacher_data import TEACHERS, CLEAR_ENV, robust_shot, margin_shot

SHOT_FNS = {"robust": robust_shot, "margin": margin_shot}

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
                 log_every: int = 500, teacher: str = "robust") -> dict[str, np.ndarray]:
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = {"action": [], "good_actions": [], "n_good": [], "win_width": [], "pocketed": []}
    sim_state, sim_action, sim_pocketed, sim_scratch = [], [], [], []   # per-state numpy chunks
    t0 = time.time()
    for n, r in enumerate(rows):
        restore_state(env, data["obs"][r])
        record: list = []
        action, good, n_good, width = SHOT_FNS[teacher](env, rng, TEACHERS[teacher], max_good, record=record)
        sim_state.append(np.full(len(record), n, np.int32))
        sim_action.append(np.asarray([a for a, _, _ in record], np.float32).reshape(-1, 2))
        sim_pocketed.append(np.asarray([p for _, p, _ in record], np.int8))
        sim_scratch.append(np.asarray([sc for _, _, sc in record], bool))
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
        "sims": {
            "sim_state":    np.concatenate(sim_state) if sim_state else np.zeros(0, np.int32),
            "sim_action":   np.concatenate(sim_action) if sim_action else np.zeros((0, 2), np.float32),
            "sim_pocketed": np.concatenate(sim_pocketed) if sim_pocketed else np.zeros(0, np.int8),
            "sim_scratch":  np.concatenate(sim_scratch) if sim_scratch else np.zeros(0, bool),
        },
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
    p.add_argument("--teacher",  type=str, default="robust", choices=list(SHOT_FNS))
    p.add_argument("--limit",    type=int, default=0, help="only the first N states of the shard (pilot)")
    return p.parse_args()


def main():
    args = parse_args()
    data = dict(np.load(args.data))
    rows = shard_rows(data["game"], args.shard, args.n_shards)
    print(f"relabel {args.data}  shard {args.shard}/{args.n_shards}: {len(rows)} states", flush=True)
    src = json.loads(str(data["meta"]))
    if args.limit:
        rows = rows[:args.limit]
    new = relabel_rows(data, rows, seed=10_000 + args.shard, max_good=args.max_good, teacher=args.teacher)
    sims = new.pop("sims")
    base = os.path.splitext(os.path.basename(args.data))[0]
    sims_dir = "sims" if args.teacher == "robust" else f"sims_{args.teacher}"
    plan = TEACHERS[args.teacher]
    write(os.path.join(args.out_dir, sims_dir, f"{base}_shard{args.shard}.npz"),
          {"obs": new["obs"], "game": new["game"], **sims},
          {"source": args.data, "shard": args.shard, "plan": plan, "env": src.get("env")})
    write(os.path.join(args.out_dir, args.teacher, f"{base}_shard{args.shard}.npz"), new,
          {"teacher": f"{args.teacher}_relabel", "source": args.data, "plan": plan, "env": src.get("env"),
           "win_width_is": "margin (angle 0.2 deg / speed 20% units)" if args.teacher == "margin" else "window width (deg)"})
    if args.teacher == "robust":
        write(os.path.join(args.out_dir, "orig", f"{base}_shard{args.shard}.npz"),
              original_rows(data, rows, args.max_good),
              {"teacher": f"{src.get('teacher')}_same_states", "source": args.data, "env": src.get("env")})
    w = new["win_width"]
    print(f"done: {len(rows)} states, {len(sims['sim_state'])} sims "
          f"({np.mean((sims['sim_pocketed'] > 0) & ~sims['sim_scratch']) * 100:.1f}% pocket clean), "
          f"{'margin' if args.teacher == 'margin' else 'window'} median {np.median(w):.3f}, "
          f"speed median {np.median(new['action'][:, 1]):.2f} (orig {np.median(data['action'][rows, 1]):.2f}) m/s",
          flush=True)


if __name__ == "__main__":
    main()
