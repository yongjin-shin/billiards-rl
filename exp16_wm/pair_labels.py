"""
exp16_wm/pair_labels.py — which (ball, pocket) pair produced each teacher label.

The teachers build every candidate around a ghost-ball aim of one (ball, pocket) pair
(angle offset ±0.4° for the ghost teacher, ≤±2° for robust/margin). That makes the pair
recoverable from the state alone, without new simulations: restore the table, list the
pairs' ghost-ball aims, and give each stored candidate the pair whose aim is nearest to
its absolute strike angle (within a tolerance).

Slot convention = PairGNNPolicy's: slot = ball_index * 6 + pocket_index, ball_index from
the observation's ball order (ids "1","2","3"), pocket_index in the observation's pocket
order. The result `good_pairs` (N, K) int8 is −1 where no pair is within tolerance
(e.g. the random fallback candidates).

Why: the GNN's winner-takes-all picks "the pair whose current output is nearest the
label" afresh on every step, a moving target — on 512 training states its pair
selection accuracy stayed at 22–48%. A fixed pair per label is a stable classification
target for the selection head and a smooth regression target for the pair's output.

Usage:
    python -m exp16_wm.pair_labels --dir logs/teacher_data/relabel_s1000/robust
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.planner import aim_reference, ghost_aims, wrap_angle
from exp16_wm.relabel import restore_state
from exp16_wm.teacher_data import CLEAR_ENV

N_POCKETS = 6


def assign_pairs(env: BilliardsEnv, good_actions: np.ndarray, tol_deg: float = 2.5,
                 max_cut_deg: float = 80.0) -> np.ndarray:
    """
    Pair slot for each candidate (K,) of one restored state; −1 if none within tol_deg.
    good_actions (K, 2) = (delta_angle, speed), NaN-padded.
    """
    out = np.full(len(good_actions), -1, dtype=np.int8)
    aims = ghost_aims(env, max_cut_deg)
    if not aims:
        return out
    phi_ref = aim_reference(env)
    slots = np.array([(int(bid) - 1) * N_POCKETS + j for bid, j, _ in aims])
    phis = np.array([phi for _, _, phi in aims])
    for k, (delta, _) in enumerate(good_actions):
        if np.isnan(delta):
            continue
        diff = np.degrees(np.abs([wrap_angle(float(phi_ref + delta - p)) for p in phis]))
        j = int(np.argmin(diff))
        if diff[j] <= tol_deg:
            out[k] = slots[j]
    return out


def pairs_for_file(path: str, tol_deg: float = 2.5) -> dict:
    """Compute good_pairs for one npz shard and write it back (atomic) as an extra array."""
    z = dict(np.load(path))
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pairs = np.full(z["good_actions"].shape[:2], -1, dtype=np.int8)
    for i in range(len(z["obs"])):
        restore_state(env, z["obs"][i])
        pairs[i] = assign_pairs(env, z["good_actions"][i], tol_deg)
    env.close()
    z["good_pairs"] = pairs
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)
    valid = ~np.isnan(z["good_actions"][..., 0])
    return {"candidates": int(valid.sum()), "assigned": int(((pairs >= 0) & valid).sum()),
            "states": len(pairs), "states_with_pair": int((pairs >= 0).any(axis=1).sum())}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--tol-deg", type=float, default=2.5)
    return p.parse_args()


def main():
    args = parse_args()
    tot = {"candidates": 0, "assigned": 0, "states": 0, "states_with_pair": 0}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.npz"))):
        st = pairs_for_file(f, args.tol_deg)
        for k in tot:
            tot[k] += st[k]
        print(f"{os.path.basename(f)}: {st}", flush=True)
    print(f"total: candidates {tot['assigned']}/{tot['candidates']} assigned "
          f"({tot['assigned'] / max(1, tot['candidates']) * 100:.1f}%), states with ≥1 pair "
          f"{tot['states_with_pair']}/{tot['states']}")


if __name__ == "__main__":
    main()
