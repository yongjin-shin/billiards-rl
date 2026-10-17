"""
exp16_wm/outcomes.py — which (ball, pocket) pairs does an action actually make?

BilliardsEnv.step only reports how many balls went in. The simulator's event log knows
more: every BALL_POCKET event names the ball and the pocket. This module restores a table
from a stored observation, plays an action, and reads those events, so a label's intended
pair (exp16_wm/pair_labels.py, matched by aim angle) can be checked against what really
happened.

Slot convention = PairGNNPolicy's: slot = ball_index * 6 + pocket_index (ball ids "1".."3",
pocket index in the observation's pocket order). The cue ball pocketed = scratch.

Annotated arrays written into a label npz (first max_k candidates per state):
    good_made     (N, max_k, 18) bool  slots whose ball really went into that pocket
    good_scratch  (N, max_k)     bool  the cue ball was pocketed

Usage:
    python -m exp16_wm.outcomes --dir logs/teacher_data/relabel_s1000/robust --max-k 2
"""

import argparse
import glob
import os
import sys

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.planner import aim_reference, strike_phi_deg
from exp16_wm.relabel import restore_state
from exp16_wm.teacher_data import CLEAR_ENV

N_POCKETS, N_SLOTS = 6, 18


def pocket_index_map(env: BilliardsEnv) -> dict[str, int]:
    """pooltool pocket id ('lb', 'rt', ...) → index in the observation's pocket order."""
    return {pid: int(np.argmin(np.linalg.norm(env._pocket_centers - np.asarray(p.center[:2]), axis=1)))
            for pid, p in env.system.table.pockets.items()}


def slots_from_system(env: BilliardsEnv, sys_c, pocket_map: dict[str, int]) -> tuple[np.ndarray, bool]:
    """(made (18,) bool, scratch) read from the BALL_POCKET events of an already simulated system."""
    made, scratch = np.zeros(N_SLOTS, dtype=bool), False
    for e in sys_c.events:
        if e.event_type != pt.EventType.BALL_POCKET:
            continue
        ids = [a.id for a in e.agents]
        ball = next(i for i in ids if i in env._ball_ids or i == "cue")
        pocket = next(i for i in ids if i in pocket_map)
        if ball == "cue":
            scratch = True
        else:
            made[(int(ball) - 1) * N_POCKETS + pocket_map[pocket]] = True
    return made, scratch


def made_slots(env: BilliardsEnv, raw_obs: np.ndarray, action: np.ndarray,
               pocket_map: dict[str, int] | None = None) -> tuple[np.ndarray, bool]:
    """
    Restore the table from raw_obs, play action (delta_angle, speed) on a copy, and return
    (made (18,) bool, scratch). Deterministic: pooltool gives the same result every time.
    """
    pocket_map = pocket_map or pocket_index_map(env)
    restore_state(env, raw_obs)
    sys_c = env.system.copy()
    sys_c.strike(phi=strike_phi_deg(aim_reference(env), action), V0=float(action[1]), cue_ball_id="cue")
    pt.simulate(sys_c, inplace=True, max_events=2000)
    return slots_from_system(env, sys_c, pocket_map)


def annotate_file(path: str, max_k: int = 2) -> dict:
    """Add good_made / good_scratch for the first max_k candidates of every state (atomic rewrite)."""
    z = dict(np.load(path))
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pmap = pocket_index_map(env)
    n = len(z["obs"])
    made = np.zeros((n, max_k, N_SLOTS), dtype=bool)
    scratch = np.zeros((n, max_k), dtype=bool)
    for i in range(n):
        for k in range(max_k):
            a = z["good_actions"][i, k]
            if np.isnan(a[0]):
                continue
            made[i, k], scratch[i, k] = made_slots(env, z["obs"][i], a, pmap)
    env.close()
    z["good_made"], z["good_scratch"] = made, scratch
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)
    return intent_stats(z, max_k)


def intent_stats(z: dict, max_k: int = 2) -> dict:
    """Counts for: candidate present, pair assigned, intended slot really made, anything made, scratch."""
    out = {}
    for k in range(max_k):
        present = ~np.isnan(z["good_actions"][:, k, 0])
        slot = z["good_pairs"][:, k].astype(int) if "good_pairs" in z else np.full(len(present), -1)
        assigned = present & (slot >= 0)
        hit = np.zeros(len(present), bool)
        idx = np.where(assigned)[0]
        hit[idx] = z["good_made"][idx, k, slot[idx]]
        any_made = present & z["good_made"][:, k].any(axis=1)
        out[k] = {"present": int(present.sum()), "assigned": int(assigned.sum()), "intended_made": int(hit.sum()),
                  "any_ball_made": int(any_made.sum()), "scratch": int((present & z["good_scratch"][:, k]).sum())}
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--max-k", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    tot: dict[int, dict] = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.npz"))):
        st = annotate_file(f, args.max_k)
        for k, d in st.items():
            t = tot.setdefault(k, {key: 0 for key in d})
            for key, v in d.items():
                t[key] += v
        print(f"{os.path.basename(f)} done", flush=True)
    for k, t in sorted(tot.items()):
        a = max(1, t["assigned"])
        print(f"candidate {k}: present {t['present']}, pair assigned {t['assigned']}, "
              f"intended pair made {t['intended_made']} ({t['intended_made'] / a * 100:.1f}% of assigned), "
              f"any ball made {t['any_ball_made']} ({t['any_ball_made'] / max(1, t['present']) * 100:.1f}% of present), "
              f"scratch {t['scratch']}")


if __name__ == "__main__":
    main()
