"""
world_model/compare_pure_physics.py

Compare pure_physics.py free-motion propagation against real RSSM training data.

For each consecutive pair of events (k, k+1) involving the same ball:
  1. Compute post-collision state at event k  (GT delta applied to raw_rvw)
  2. Advance forward by dt_to_next[k] using pure_physics.evolve_ball_motion
  3. Compare predicted position vs raw_rvw[k+1][0] (actual pre-collision position)

This tests whether the free-motion equations in pure_physics are consistent
with pooltool's simulation (which generated the training data).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# ── add project root to path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.pure_physics import (
    evolve_ball_motion,
    STATIONARY, SPINNING, SLIDING, ROLLING,
)
from world_model.ball_motion import DEFAULT_FRICTION

DEFAULT_PARAMS = {
    "R":    DEFAULT_FRICTION.R,
    "m":    DEFAULT_FRICTION.m,
    "u_s":  DEFAULT_FRICTION.u_s,
    "u_sp": DEFAULT_FRICTION.u_sp,
    "u_r":  DEFAULT_FRICTION.u_r,
    "g":    DEFAULT_FRICTION.g,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def post_collision_rvw(raw_rvw: np.ndarray, delta) -> np.ndarray:
    """Apply gt_delta (5-dim Δvel+Δavel) to pre-collision rvw → post-collision rvw."""
    if delta is None:
        return raw_rvw.copy()
    d = delta.numpy() if hasattr(delta, "numpy") else np.asarray(delta, dtype=np.float64)
    new_rvw = raw_rvw.copy().astype(np.float64)
    new_rvw[1, :2] += d[:2]   # Δvel
    new_rvw[2, :3] += d[2:]   # Δavel
    return new_rvw


def infer_state(rvw: np.ndarray) -> int:
    """Infer ball motion state from vel/avel magnitudes."""
    speed = float(np.linalg.norm(rvw[1, :2]))
    if speed < 1e-6:
        return STATIONARY
    return SLIDING  # conservative: let evolve_ball_motion handle transitions


# ── main comparison ───────────────────────────────────────────────────────────

def compare_shots(shots, params=None, verbose: bool = False):
    if params is None:
        params = DEFAULT_PARAMS

    pos_errors          = []   # |predicted_pos - actual_pos| in cm
    pos_errors_with_dt  = []   # (err, dt) pairs
    event_counts        = []
    bad_shots           = 0

    for shot_idx, shot in enumerate(shots):
        n = len(shot.event_steps)
        if n < 2:
            continue

        # Build per-ball lookup: ball_i → list of (k, raw_rvw, delta) ordered by event index
        from collections import defaultdict
        ball_events: dict[int, list] = defaultdict(list)
        for k in range(n):
            ev = shot.event_steps[k]
            ball_events[ev.ball_i].append((k, shot.raw_rvws_i[k], shot.gt_deltas_i[k]))
            if ev.ball_j is not None:
                ball_events[ev.ball_j].append((k, shot.raw_rvws_j[k], shot.gt_deltas_j[k]))

        shot_errors = []
        for ball_idx, ev_list in ball_events.items():
            # sort by event index
            ev_list.sort(key=lambda x: x[0])
            for i in range(len(ev_list) - 1):
                k,     raw_k,    delta_k    = ev_list[i]
                k_next, raw_next, _          = ev_list[i + 1]

                # dt is the sum of inter-event gaps from global event k to k_next
                # (events are globally ordered; ball_i may skip several)
                dt = float(sum(shot.dt_to_next[j] for j in range(k, k_next)))
                if dt <= 0.0:
                    continue

                # Post-collision state at event k
                post_rvw = post_collision_rvw(raw_k, delta_k)
                s0       = infer_state(post_rvw)

                # Skip physically impossible transitions (dt > t_stop): data artifacts
                # where the ball stops before the next recorded event (≈2.9% of data)
                from world_model.pure_physics import t_stop
                ts = t_stop(post_rvw, s0, params["R"], params["u_s"],
                            params["u_sp"], params["u_r"], params["g"])
                if dt > ts + 0.1:
                    continue

                # Advance by dt using pure_physics
                try:
                    pred_rvw, _ = evolve_ball_motion(
                        s0, post_rvw,
                        R=params["R"], m=params["m"],
                        u_s=params["u_s"], u_sp=params["u_sp"],
                        u_r=params["u_r"], g=params["g"],
                        t=dt,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  shot {shot_idx} ev {k} ball {ball_idx}: evolve error: {e}")
                    continue

                pred_pos = pred_rvw[0, :2]
                true_pos = raw_next[0, :2]
                err = float(np.linalg.norm(pred_pos - true_pos)) * 100  # cm

                if verbose and err > 0.5:
                    print(
                        f"  shot={shot_idx} ev={k}→{k_next} ball={ball_idx} "
                        f"dt={dt:.4f}s  err={err:.3f}cm"
                    )
                    print(f"    post vel={post_rvw[1,:2]}  speed={np.linalg.norm(post_rvw[1,:2]):.3f}")

                shot_errors.append(err)
                pos_errors_with_dt.append((err, dt))

        if shot_errors:
            pos_errors.extend(shot_errors)
            event_counts.append(len(shot_errors))

        else:
            bad_shots += 1

    if not pos_errors:
        print("No valid consecutive same-ball event pairs found.")
        return

    pos_errors = np.array(pos_errors)
    dts_arr    = np.array([d for _, d in pos_errors_with_dt])

    print(f"Shots analysed     : {len(event_counts)}  (skipped {bad_shots})")
    print(f"Event transitions  : {len(pos_errors)}")
    print()
    print(f"Overall position error:")
    print(f"  mean   : {pos_errors.mean():.3f} cm")
    print(f"  median : {np.median(pos_errors):.3f} cm")
    print(f"  p95    : {np.percentile(pos_errors, 95):.3f} cm")
    print(f"  <1mm   : {(pos_errors < 0.1).mean()*100:.1f}%")
    print(f"  <1cm   : {(pos_errors < 1.0).mean()*100:.1f}%")
    print(f"  >5cm   : {(pos_errors > 5.0).mean()*100:.1f}%")
    print()
    print(f"Error by free-flight duration (dt):")
    buckets = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]
    for lo, hi in buckets:
        mask = (dts_arr >= lo) & (dts_arr < hi)
        if mask.sum() == 0:
            continue
        sub = pos_errors[mask]
        pct = mask.mean() * 100
        hi_str = f"{hi:.1f}" if hi < np.inf else "∞"
        print(f"  [{lo:.1f}, {hi_str})s  n={mask.sum():4d} ({pct:.0f}%)  "
              f"mean={sub.mean():.2f}cm  p95={np.percentile(sub,95):.2f}cm")


def main():
    data_dir = Path(__file__).parent / "data_rssm"
    chunk_files = sorted(data_dir.glob("*_chunk*.pkl"))
    assert chunk_files, f"No chunk files in {data_dir}"

    print(f"Loading first chunk: {chunk_files[0].name}")
    with open(chunk_files[0], "rb") as f:
        shots = pickle.load(f)

    n_sample = min(2000, len(shots))
    print(f"Using {n_sample} shots for comparison\n")

    compare_shots(shots[:n_sample], verbose=False)


if __name__ == "__main__":
    main()
