"""
exp16_wm/wide_scan.py — success-window centres from a 360° simulator scan, no aim formula (Exp-16, roadmap S1).

For every state the cue ball is shot in every direction (coarse step, default 0.05°, one speed). Each shot's outcome tells, for all
18 (ball, pocket) pairs at once, whether that pair was pocketed cleanly (target ball in that pocket, no scratch). The coarse hits of
a pair form windows (runs of consecutive successful directions, circular); every window is scanned again at a fine step around its
coarse extent, and the pair's target is the mean successful direction of its widest window — the same "window centre" as the T4 scan,
but found without the ghost-ball formula or a learned aim to start from. Pairs without any coarse hit get no target (NaN).

Output: one file per label shard (same basename) with scan_center_deg (N, 18), scan_valid (N, 18), scan_width_deg (N, 18),
scan_n_windows (N, 18) — readable by bc --geom-aux-set scan --aux-target-dir (load_scan_targets).

Usage:
    python -m exp16_wm.wide_scan --data "logs/teacher_data/policy_t2_s0/*.npz" --out logs/teacher_data/policy_t2_s0_widescan --workers 8
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_PAIRS = 18


def coarse_angles(step_deg: float) -> np.ndarray:
    """Directions (deg) covering the full circle: −180, −180 + step, …, < 180."""
    n = int(round(360.0 / step_deg))
    return -180.0 + step_deg * np.arange(n)


def find_windows(flags: np.ndarray) -> list[tuple[int, int]]:
    """flags (A,) success on a circular grid → windows as (start index, length), runs of consecutive True (a run may wrap around)."""
    f = np.asarray(flags, bool)
    A = len(f)
    if not f.any():
        return []
    if f.all():
        return [(0, A)]
    first_false = int(np.argmin(f))                           # start the walk on a False so no run is split by the seam
    out, i = [], 0
    while i < A:
        k = (first_false + i) % A
        if f[k]:
            start, length = k, 0
            while i < A and f[(first_false + i) % A]:
                length += 1; i += 1
            out.append((start, length))
        else:
            i += 1
    return out


def fine_angles(start_deg: float, length: int, coarse_step: float, fine_step: float) -> np.ndarray:
    """Fine directions over a coarse window widened by one coarse step on each side (deg, not wrapped)."""
    lo, hi = start_deg - coarse_step, start_deg + (length - 1) * coarse_step + coarse_step
    return lo + fine_step * np.arange(int(round((hi - lo) / fine_step)) + 1)


def wrap_deg(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def to_env_actions(angles_deg: np.ndarray, speed: float, act_scale: np.ndarray, act_bias: np.ndarray) -> np.ndarray:
    """Directions (deg, the policy's relative frame) → env actions (C, 2) [angle, speed], through the unit range like aim_sims."""
    d = np.radians(angles_deg) - act_bias[0]
    u = np.clip(np.arctan2(np.sin(d), np.cos(d)) / act_scale[0], -1.0, 1.0)
    return np.stack([u * act_scale[0] + act_bias[0], np.full(len(u), speed)], -1).astype(np.float32)


def scan_state(sim, obs: np.ndarray, speed: float, coarse_step: float, fine_step: float, act_scale: np.ndarray, act_bias: np.ndarray
               ) -> dict[str, np.ndarray]:
    """sim((obs, actions)) → (made (C, 18), scratch (C,)). Returns centre (18,) deg, valid, width (deg), number of windows; and the
    number of simulations used."""
    ang = coarse_angles(coarse_step)
    made, scr = sim((obs, to_env_actions(ang, speed, act_scale, act_bias)))
    ok = made & ~scr[:, None]
    centre, width = np.full(N_PAIRS, np.nan, np.float32), np.zeros(N_PAIRS, np.float32)
    n_win, n_sims = np.zeros(N_PAIRS, np.int16), len(ang)
    for j in range(N_PAIRS):
        wins = find_windows(ok[:, j])
        n_win[j] = len(wins)
        if not wins:
            continue
        start, length = max(wins, key=lambda w: w[1])
        fa = fine_angles(float(ang[start]), length, coarse_step, fine_step)
        fm, fs = sim((obs, to_env_actions(wrap_deg(fa), speed, act_scale, act_bias)))
        n_sims += len(fa)
        hit = fm[:, j] & ~fs
        if hit.any():
            centre[j] = float(wrap_deg(fa[hit].mean())); width[j] = float(hit.sum() * fine_step)
    return {"centre": centre, "valid": np.isfinite(centre), "width": width, "n_windows": n_win, "n_sims": n_sims}


def _scan_task(task: tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    from exp16_wm.frozen_verifier import _simulate_full
    obs, speed, cs, fs, sc, bi = task
    return scan_state(_simulate_full, obs, speed, cs, fs, sc, bi)


def main() -> None:
    from exp16_wm.frozen_verifier import _init_worker
    from exp16_wm.geometry_aux import aim_targets
    from exp16_wm.gnn_policy import PairGNNPolicy
    from exp16_wm.obs_transform import relative_obs
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="label files (only their states are used)")
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--speed", type=float, default=1.0); p.add_argument("--coarse-step", type=float, default=0.05)
    p.add_argument("--fine-step", type=float, default=0.01); p.add_argument("--max-states", type=int, default=0)
    args = p.parse_args()
    ref = PairGNNPolicy(hidden=8)
    sc, bi = ref.act_scale.numpy().astype(np.float64), ref.act_bias.numpy().astype(np.float64)
    os.makedirs(args.out, exist_ok=True)
    t0, total, states = time.time(), 0, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for path in sorted(glob.glob(args.data)):
            z = np.load(path)
            obs = z["obs"][:args.max_states] if args.max_states else z["obs"]
            res = pool.map(_scan_task, [(o, args.speed, args.coarse_step, args.fine_step, sc, bi) for o in obs], chunksize=1)
            N = len(z["obs"])
            out = {k: np.full((N, N_PAIRS), np.nan if k == "centre" else 0, dtype) for k, dtype in
                   (("centre", np.float32), ("width", np.float32), ("n_windows", np.int16))}
            valid = np.zeros((N, N_PAIRS), bool)
            for i, r in enumerate(res):
                out["centre"][i], out["width"][i], out["n_windows"][i], valid[i] = r["centre"], r["width"], r["n_windows"], r["valid"]
            n = sum(r["n_sims"] for r in res); total += n; states += len(obs)
            np.savez(os.path.join(args.out, os.path.basename(path)), scan_center_deg=out["centre"], scan_valid=valid,
                     scan_width_deg=out["width"], scan_n_windows=out["n_windows"], scan_speed=args.speed,
                     scan_coarse_step=args.coarse_step, scan_fine_step=args.fine_step, n_scanned=len(obs))
            exact, live = aim_targets(relative_obs(obs))
            v = valid[:len(obs)] & live
            err = np.abs(wrap_deg(out["centre"][:len(obs)][v] - exact[..., 0][v]))
            print(f"  {os.path.basename(path)}: {len(obs)} states, {n} simulations, live pairs with a window {v.sum() / max(1, live.sum()):.1%}, "
                  f"|centre − formula aim| median {np.median(err) if len(err) else float('nan'):.3f}°, ≥0.1° {np.mean(err >= 0.1) if len(err) else float('nan'):.1%}, "
                  f"{time.time() - t0:.0f}s ({total / max(1e-9, time.time() - t0):.0f} sims/s)", flush=True)
    print(f"saved {args.out}: {states} states, {total} simulations ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
