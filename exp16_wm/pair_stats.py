"""
exp16_wm/pair_stats.py — per-pair outcome statistics for every (ball, pocket) of a state.

The teachers only keep the top-2 pairs of a state, so every other pair is an unlabeled
"negative" for the selection head even when it is just as good (experiments.md, "Label
incompleteness"). Here every pair with a ghost-ball aim (cut ≤ 80°) gets the same coarse
test — 5 aim offsets × 4 speeds = 20 simulated shots — and the event log tells what
really happened:

    pair_n        (N,18) int8  shots tried for the pair (20, or 0 if the pair has no aim / ball is gone)
    pair_made     (N,18) int8  shots that put exactly this pair's ball into this pair's pocket, no scratch
    pair_any      (N,18) int8  shots that cleanly pocketed any ball (the earlier teachers' notion of "feasible")
    pair_scratch  (N,18) int8  shots on which the cue ball was pocketed

Slot = ball_index * 6 + pocket_index (PairGNNPolicy's convention). The arrays depend on the
state only, so they are computed once and attached to every label set that shares the rows.

Usage:
    python -m exp16_wm.pair_stats --root logs/teacher_data/relabel_s1000 --dirs robust margin orig32 \
        --shard 0 --n-shards 8
"""

import argparse
import glob
import os
import sys

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.outcomes import slots_from_system, pocket_index_map, N_POCKETS, N_SLOTS
from exp16_wm.planner import aim_reference, ghost_aims, action_from_phi, strike_phi_deg
from exp16_wm.relabel import restore_state
from exp16_wm.teacher_data import CLEAR_ENV

COARSE = {"offsets_deg": [-0.4, -0.2, 0.0, 0.2, 0.4], "speeds": [0.75, 1.5, 3.0, 5.0], "max_cut_deg": 80.0}
KEYS = ["pair_n", "pair_made", "pair_any", "pair_scratch"]


def pair_stats_for_state(env: BilliardsEnv, raw_obs: np.ndarray, pocket_map: dict[str, int],
                         plan: dict = COARSE) -> dict[str, np.ndarray]:
    """Run the coarse test on every pair of one state (restored once, each shot on a copy)."""
    restore_state(env, raw_obs)
    phi_ref = aim_reference(env)
    out = {k: np.zeros(N_SLOTS, dtype=np.int8) for k in KEYS}
    for bid, j, phi in ghost_aims(env, plan["max_cut_deg"]):
        slot = (int(bid) - 1) * N_POCKETS + j
        for off in plan["offsets_deg"]:
            for v in plan["speeds"]:
                a = action_from_phi(phi + np.radians(off), phi_ref, v)
                sys_c = env.system.copy()
                sys_c.strike(phi=strike_phi_deg(phi_ref, a), V0=float(a[1]), cue_ball_id="cue")
                pt.simulate(sys_c, inplace=True, max_events=2000)
                made, scratch = slots_from_system(env, sys_c, pocket_map)
                out["pair_n"][slot] += 1
                out["pair_scratch"][slot] += int(scratch)
                out["pair_made"][slot] += int(made[slot] and not scratch)
                out["pair_any"][slot] += int(made.any() and not scratch)
    return out


def stats_for_obs(obs: np.ndarray, plan: dict = COARSE, log_every: int = 0) -> dict[str, np.ndarray]:
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pmap = pocket_index_map(env)
    res = {k: np.zeros((len(obs), N_SLOTS), dtype=np.int8) for k in KEYS}
    for i in range(len(obs)):
        st = pair_stats_for_state(env, obs[i], pmap, plan)
        for k in KEYS:
            res[k][i] = st[k]
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(obs)} states", flush=True)
    env.close()
    return res


def attach(path: str, stats: dict[str, np.ndarray]) -> None:
    """Add the pair_* arrays to a label shard (atomic rewrite); rows must line up."""
    z = dict(np.load(path))
    assert len(z["obs"]) == len(stats["pair_n"]), (path, len(z["obs"]), len(stats["pair_n"]))
    z.update(stats)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dirs", type=str, nargs="+", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--n-shards", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    name = f"ghost_s1000_shard{args.shard}.npz"
    ref = dict(np.load(os.path.join(args.root, args.dirs[0], name)))
    print(f"shard {args.shard}: {len(ref['obs'])} states", flush=True)
    stats = stats_for_obs(ref["obs"], log_every=500)
    for d in args.dirs:                       # every label set shares the same rows
        other = np.load(os.path.join(args.root, d, name))["obs"]
        assert np.array_equal(other, ref["obs"]), f"{d}: rows differ from {args.dirs[0]}"
        attach(os.path.join(args.root, d, name), stats)
    n = stats["pair_n"]
    print(f"done: tried pairs/state {(n > 0).sum(1).mean():.2f}, made-pair coverage "
          f"{(stats['pair_made'].sum(1) > 0).mean() * 100:.1f}% of states, "
          f"scratch rate {stats['pair_scratch'].sum() / max(1, n.sum()) * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
