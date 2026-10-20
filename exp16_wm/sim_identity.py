"""
exp16_wm/sim_identity.py — which (ball, pocket) each stored teacher simulation made (Exp-16).

relabel.py kept every simulation the robust teacher ran (sims/<file>_shard<i>.npz:
sim_state, sim_action, sim_pocketed = number of balls, sim_scratch) but not which
ball went into which pocket. Shots start at rest and pooltool is deterministic, so
replaying each (state, action) on the restored table recovers it. Adds to the sims file:

  sim_made (N,) int32 — bitmask over the 18 (ball, pocket) slots, slot = ball * 6 + pocket
  sim_scratch_replay (N,) bool — scratch on the replay (checked against sim_scratch)

Training data for the action-conditional made head (gnn_policy, look-ahead stage 2).

Usage:
    python -m exp16_wm.sim_identity --file logs/teacher_data/relabel_s1000/sims/ghost_s1000_shard0.npz
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.outcomes import made_slots, pocket_index_map
from exp16_wm.teacher_data import CLEAR_ENV

N_SLOTS = 18


def to_bitmask(made: np.ndarray) -> np.ndarray:
    """(..., S) bool → (...) bitmask: int32 for up to 31 slots (3 balls: 18), int64 up to 63 (9 balls: 54)."""
    s = made.shape[-1]
    if s > 63:
        raise ValueError(f"{s} slots do not fit a 64-bit mask")
    m = (made.astype(np.int64) << np.arange(s)).sum(-1)
    return m.astype(np.int32) if s <= 31 else m


def from_bitmask(mask: np.ndarray, n_slots: int = N_SLOTS) -> np.ndarray:
    """(...) int bitmask → (..., n_slots) bool."""
    return ((np.asarray(mask, np.int64)[..., None] >> np.arange(n_slots)) & 1).astype(bool)


def replay(obs: np.ndarray, sim_state: np.ndarray, sim_action: np.ndarray,
           log_every: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Replay every (obs[sim_state[i]], sim_action[i]); returns (bitmask (N,), scratch (N,))."""
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pm = pocket_index_map(env)
    mask = np.zeros(len(sim_state), np.int32)
    scratch = np.zeros(len(sim_state), bool)
    t0 = time.time()
    for i, (s, a) in enumerate(zip(sim_state, sim_action)):
        made, sc = made_slots(env, obs[s], a, pm)
        mask[i], scratch[i] = to_bitmask(made), sc
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(sim_state)}  {time.time() - t0:.0f}s", flush=True)
    env.close()
    return mask, scratch


def annotate(path: str, log_every: int = 50_000) -> dict:
    """Add sim_made / sim_scratch_replay to a sims file (atomic rewrite); returns consistency counts."""
    z = dict(np.load(path))
    mask, scratch = replay(z["obs"], z["sim_state"], z["sim_action"], log_every)
    z["sim_made"], z["sim_scratch_replay"] = mask, scratch
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)
    n_made = from_bitmask(mask).sum(1)
    return {"sims": len(mask), "scratch_match": float(np.mean(scratch == z["sim_scratch"])),
            "count_match": float(np.mean(n_made == z["sim_pocketed"]))}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    st = annotate(args.file)
    print(f"done {args.file}: {st}", flush=True)


if __name__ == "__main__":
    main()
