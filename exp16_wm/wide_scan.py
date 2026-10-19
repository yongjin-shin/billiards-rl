"""
exp16_wm/wide_scan.py — success-window centres from a 360° simulator scan, no aim formula (Exp-16, roadmap S1).

For every state the cue ball is shot in every direction (coarse step, default 0.05°, one speed). Each shot's outcome tells, for all
18 (ball, pocket) pairs at once, whether that pair was pocketed cleanly (target ball in that pocket, no scratch). The coarse hits of
a pair form windows (runs of consecutive successful directions, circular); every window is scanned again at a fine step around its
coarse extent, and the pair's target is the mean successful direction of its widest window — the same "window centre" as the T4 scan,
but found without the ghost-ball formula or a learned aim to start from. Pairs without any coarse hit get no target (NaN).
robust_speeds (S1b): the coarse scan is played at every listed speed and a direction counts only if it pockets the pair cleanly at all
of them — direct shots survive a change of speed, cushion and combination paths mostly do not — so the chosen window is the direct one
without asking the formula (200 states: widest window at 1 m/s alone is > 5° from the formula aim for 36% of pairs).

Output: one file per label shard (same basename) with scan_center_deg (N, 18), scan_valid (N, 18), scan_width_deg (N, 18),
scan_n_windows (N, 18), scan_any_center_deg (N, 18: widest window at the scan speed alone, any path, NaN if none); shards already
written are skipped (resume) — readable by bc --geom-aux-set scan --aux-target-dir (load_scan_targets).

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


def fine_angles(start_deg: float, length: int, coarse_step: float, fine_step: float, pad_deg: float | None = None) -> np.ndarray:
    """Fine directions over a coarse window widened by pad_deg (default one coarse step) on each side (deg, not wrapped)."""
    pad = coarse_step if pad_deg is None else pad_deg
    lo, hi = start_deg - pad, start_deg + (length - 1) * coarse_step + pad
    return lo + fine_step * np.arange(int(round((hi - lo) / fine_step)) + 1)


def wrap_deg(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def to_env_actions(angles_deg: np.ndarray, speed: float, act_scale: np.ndarray, act_bias: np.ndarray) -> np.ndarray:
    """Directions (deg, the policy's relative frame) → env actions (C, 2) [angle, speed], through the unit range like aim_sims."""
    d = np.radians(angles_deg) - act_bias[0]
    u = np.clip(np.arctan2(np.sin(d), np.cos(d)) / act_scale[0], -1.0, 1.0)
    return np.stack([u * act_scale[0] + act_bias[0], np.full(len(u), speed)], -1).astype(np.float32)


def scan_state(sim, obs: np.ndarray, speed: float, coarse_step: float, fine_step: float, act_scale: np.ndarray, act_bias: np.ndarray,
               robust_speeds: tuple[float, ...] = (), fine_pad_deg: float | None = None) -> dict[str, np.ndarray]:
    """sim((obs, actions)) → (made (C, 18), scratch (C,)). Returns centre (18,) deg, valid, width (deg), number of windows; and the
    number of simulations used. robust_speeds: coarse successes must hold at speed and at every one of these speeds."""
    ang = coarse_angles(coarse_step)
    made, scr = sim((obs, to_env_actions(ang, speed, act_scale, act_bias)))
    ok = made & ~scr[:, None]
    any_centre = np.full(N_PAIRS, np.nan, np.float32)                # widest window at `speed` alone (coarse middle), any path
    for j in range(N_PAIRS):
        w = find_windows(ok[:, j])
        if w:
            st, ln = max(w, key=lambda x: x[1])
            any_centre[j] = float(wrap_deg(ang[st] + (ln - 1) * coarse_step / 2))
    for v in robust_speeds:
        mv, sv = sim((obs, to_env_actions(ang, v, act_scale, act_bias)))
        ok &= mv & ~sv[:, None]
    centre, width = np.full(N_PAIRS, np.nan, np.float32), np.zeros(N_PAIRS, np.float32)
    n_win, n_sims = np.zeros(N_PAIRS, np.int16), len(ang) * (1 + len(robust_speeds))
    for j in range(N_PAIRS):
        wins = find_windows(ok[:, j])
        n_win[j] = len(wins)
        if not wins:
            continue
        start, length = max(wins, key=lambda w: w[1])
        fa = fine_angles(float(ang[start]), length, coarse_step, fine_step, fine_pad_deg)
        fm, fs = sim((obs, to_env_actions(wrap_deg(fa), speed, act_scale, act_bias)))
        n_sims += len(fa)
        hit = fm[:, j] & ~fs
        if hit.any():                                         # the fine run that holds the coarse window's middle (else the longest)
            runs, i = [], 0
            while i < len(hit):
                if hit[i]:
                    k = i
                    while k < len(hit) and hit[k]:
                        k += 1
                    runs.append((i, k)); i = k
                else:
                    i += 1
            mid = float(ang[start]) + (length - 1) * coarse_step / 2
            holding = [r for r in runs if fa[r[0]] - fine_step / 2 <= mid <= fa[r[1] - 1] + fine_step / 2]
            a0, a1 = holding[0] if holding else max(runs, key=lambda r: r[1] - r[0])
            centre[j] = float(wrap_deg(fa[a0:a1].mean())); width[j] = float((a1 - a0) * fine_step)
    return {"centre": centre, "valid": np.isfinite(centre), "width": width, "n_windows": n_win, "n_sims": n_sims, "any_centre": any_centre}


def _scan_task(task: tuple) -> dict[str, np.ndarray]:
    from exp16_wm.frozen_verifier import _simulate_full
    obs, speed, cs, fs, sc, bi, robust, pad = task
    return scan_state(_simulate_full, obs, speed, cs, fs, sc, bi, robust, pad)


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
    p.add_argument("--robust-speeds", type=float, nargs="*", default=[], help="coarse hits must also hold at these speeds (S1b)")
    p.add_argument("--chunk", type=int, default=256, help="states per saved partial result (resume granularity)")
    p.add_argument("--fine-pad", type=float, default=None, help="fine scan reaches this far (deg) beyond the coarse window (default one coarse step)")
    args = p.parse_args()
    ref = PairGNNPolicy(hidden=8)
    sc, bi = ref.act_scale.numpy().astype(np.float64), ref.act_bias.numpy().astype(np.float64)
    os.makedirs(args.out, exist_ok=True)
    t0, total, states = time.time(), 0, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for path in sorted(glob.glob(args.data)):
            target = os.path.join(args.out, os.path.basename(path))
            if os.path.exists(target):                                   # resume: shards already scanned (a reclaimed instance)
                print(f"  {os.path.basename(path)}: already scanned, skipped", flush=True)
                continue
            z = np.load(path)
            obs = z["obs"][:args.max_states] if args.max_states else z["obs"]
            res = []
            for c0 in range(0, len(obs), args.chunk):                    # partial results per chunk: a reclaimed instance loses minutes
                part = f"{target}.part{c0 // args.chunk:04d}.npy"
                if os.path.exists(part):
                    res.extend(np.load(part, allow_pickle=True).tolist())
                    continue
                r = pool.map(_scan_task, [(o, args.speed, args.coarse_step, args.fine_step, sc, bi, tuple(args.robust_speeds), args.fine_pad)
                                          for o in obs[c0:c0 + args.chunk]], chunksize=1)
                np.save(part + ".tmp.npy", np.array(r, dtype=object), allow_pickle=True); os.replace(part + ".tmp.npy", part)
                res.extend(r)
            N = len(z["obs"])
            out = {k: np.full((N, N_PAIRS), np.nan if k == "centre" else 0, dtype) for k, dtype in
                   (("centre", np.float32), ("width", np.float32), ("n_windows", np.int16))}
            valid = np.zeros((N, N_PAIRS), bool)
            any_c = np.full((N, N_PAIRS), np.nan, np.float32)
            for i, r in enumerate(res):
                out["centre"][i], out["width"][i], out["n_windows"][i], valid[i] = r["centre"], r["width"], r["n_windows"], r["valid"]
                any_c[i] = r["any_centre"]
            n = sum(r["n_sims"] for r in res); total += n; states += len(obs)
            tmp = target + ".tmp.npz"
            np.savez(tmp, scan_center_deg=out["centre"], scan_valid=valid, scan_any_center_deg=any_c,
                     scan_width_deg=out["width"], scan_n_windows=out["n_windows"], scan_speed=args.speed,
                     scan_coarse_step=args.coarse_step, scan_fine_step=args.fine_step, n_scanned=len(obs),
                     scan_robust_speeds=np.asarray(args.robust_speeds, np.float32), scan_fine_pad=np.float32(args.fine_pad or args.coarse_step))
            os.replace(tmp, target)                                      # never a half-written shard
            for part in glob.glob(target + ".part*.npy"):
                os.remove(part)
            exact, live = aim_targets(relative_obs(obs))
            v = valid[:len(obs)] & live
            err = np.abs(wrap_deg(out["centre"][:len(obs)][v] - exact[..., 0][v]))
            print(f"  {os.path.basename(path)}: {len(obs)} states, {n} simulations, live pairs with a window {v.sum() / max(1, live.sum()):.1%}, "
                  f"|centre − formula aim| median {np.median(err) if len(err) else float('nan'):.3f}°, ≥0.1° {np.mean(err >= 0.1) if len(err) else float('nan'):.1%}, "
                  f"{time.time() - t0:.0f}s ({total / max(1e-9, time.time() - t0):.0f} sims/s)", flush=True)
    print(f"saved {args.out}: {states} states, {total} simulations ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
