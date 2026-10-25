"""
exp16_wm/v_layout_sensitivity.py — V9: how much does a layout's value map change when one ball is moved a little?

For some layouts (raw observations of V8 files) the map of cue-position values is made for (O) the layout, (R) the same layout with other random
copies (the noise floor), (D1) one ball moved 1 cm and (D3) the same ball moved 3 cm, each with the V8 recipe: on a grid (default 10 cm), at every
free cell the policy plays out the exact position and `copies` copies on each of the 1, 2, 3, 4, 6 cm rings; the cell's value is the mean over
the radii of the ring means. Maps are compared on the cells free in both layouts: the correlation, and the correlation after a Gaussian blur.

Usage:
    python -m exp16_wm.v_layout_sensitivity --file "logs/teacher_data/v8/smoke/*.npz" --layouts 8 --runs "<PRm run dir glob>" --out result.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RADII_MM = (10.0, 20.0, 30.0, 40.0, 60.0)


def move_ball(raw: np.ndarray, which: int, dist: float, rng: np.random.Generator, w: float, l: float, ball_r: float, tries: int = 200) -> np.ndarray | None:
    """A copy of raw with the `which`-th live object ball moved `dist` metres in a random direction, staying on the table (centre one radius from
    the cushion) and at least two radii from every other ball still on it; None if no direction works."""
    from exp16_wm.obs_transform import n_balls_raw
    n = n_balls_raw(len(raw))
    live = [i for i in range(n) if raw[4 + 3 * i] < 0.5]
    i = live[which % len(live)]
    pos = np.array([raw[2 + 3 * i] * w, raw[3 + 3 * i] * l])
    others = np.array([[raw[2 + 3 * j] * w, raw[3 + 3 * j] * l] for j in live if j != i]).reshape(-1, 2)
    for _ in range(tries):
        t = rng.uniform(0.0, 2.0 * np.pi)
        q = pos + dist * np.array([np.cos(t), np.sin(t)])
        if not (ball_r <= q[0] <= w - ball_r and ball_r <= q[1] <= l - ball_r):
            continue
        if len(others) and np.min(np.linalg.norm(others - q, axis=1)) < 2.0 * ball_r + 1e-4:
            continue
        out = np.array(raw, np.float32, copy=True)
        out[2 + 3 * i], out[3 + 3 * i] = q[0] / w, q[1] / l
        return out
    return None


def map_values(raw: np.ndarray, left: int, cells: np.ndarray, seed: int, radii_mm: tuple[float, ...], copies: int, roll, w: float, l: float, ball_r: float) -> np.ndarray:
    """The label (mean over the radii of the ring means; the exact k where no copy was placed) of every cell of `cells` (M, 2 metres)."""
    from exp16_wm.v_heatmap import with_cue
    from exp16_wm.v_map_data import aggregate, ring_copies
    rng = np.random.default_rng(seed)
    radii_m = tuple(r / 1000.0 for r in radii_mm)
    per = 1 + len(radii_m) * copies
    obs = np.zeros((len(cells) * per, len(raw)), np.float32)
    placed = np.zeros((len(cells), len(radii_m) * copies), bool)
    for i, (x, y) in enumerate(cells):
        base = with_cue(raw, x, y, w, l)
        cp, pl = ring_copies(base, radii_m, copies, rng, w, l, ball_r)
        obs[i * per] = base; obs[i * per + 1:(i + 1) * per] = cp; placed[i] = pl
    jobs = [(obs[s:s + 240], left) for s in range(0, len(obs), 240)]
    ks = np.concatenate([np.asarray(r) for r in roll(jobs)]).reshape(-1, per)
    ring_k = np.where(placed, ks[:, 1:], -1).reshape(len(cells), len(radii_m), copies)
    return aggregate(ks[:, 0], ring_k)["robust"].astype(float)


def compare(a: dict, b: dict, sigma_m: float = 0.08) -> dict[str, float]:
    """Correlation of two maps ({"cells": (M, 2), "val": (M,)}) on the cells present in both (matched by position), raw and after a Gaussian blur."""
    from exp16_wm.v_map_eval import corr, smooth_scattered
    ka = {tuple(np.round(c, 5)): v for c, v in zip(a["cells"], a["val"])}
    kb = {tuple(np.round(c, 5)): v for c, v in zip(b["cells"], b["val"])}
    common = [k for k in ka if k in kb]
    xy = np.array(common)
    va, vb = np.array([ka[k] for k in common]), np.array([kb[k] for k in common])
    return {"n": len(common), "corr": corr(va, vb), "corr_blur": corr(smooth_scattered(xy, va, sigma_m), smooth_scattered(xy, vb, sigma_m))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True); p.add_argument("--layouts", type=int, default=8); p.add_argument("--step", type=float, default=0.10)
    p.add_argument("--copies", type=int, default=3); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
    args = p.parse_args()
    from multiprocessing import Pool
    import pooltool as pt
    from exp16_wm import v_heatmap as H
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_map_data import grid_cells
    rng = np.random.default_rng(args.seed)
    obs, left = [], []
    for f in sorted(glob.glob(args.file)):
        z = np.load(f, allow_pickle=True)
        for g in np.unique(z["layout_id"]):
            m = z["layout_id"] == g
            if int(z["left"][m][0]) >= 4:
                obs.append(z["obs"][m][0]); left.append(int(z["left"][m][0]))
    pick = rng.permutation(len(obs))[:args.layouts]
    ball_r = float(pt.BallParams.default().R)
    res = []
    with Pool(args.workers, initializer=H._init, initargs=([find_run(g) for g in args.runs], n_balls_raw(len(obs[0])))) as pool:
        roll = lambda jobs: pool.map(H._roll, jobs)
        for n, i in enumerate(pick):
            raw, ell = obs[i], left[i]
            which = int(rng.integers(0, 6))
            variants = {"O": raw, "R": raw, "D1": move_ball(raw, which, 0.01, rng, TABLE_W, TABLE_L, ball_r), "D3": move_ball(raw, which, 0.03, rng, TABLE_W, TABLE_L, ball_r)}
            if variants["D1"] is None or variants["D3"] is None:
                continue
            maps = {}
            for name, r in variants.items():
                cells = grid_cells(r, args.step, 0.0, np.random.default_rng(0), ball_r, TABLE_W, TABLE_L)[0]
                maps[name] = {"cells": cells, "val": map_values(r, ell, cells, args.seed * 1000 + n * 10 + {"O": 1, "R": 2, "D1": 3, "D3": 4}[name], RADII_MM, args.copies, roll, TABLE_W, TABLE_L, ball_r)}
            row = {"left": ell, **{f"{k}": compare(maps["O"], maps[k]) for k in ("R", "D1", "D3")}}
            res.append(row)
            print(f"layout {n}: left {ell}, cells {row['R']['n']} | corr R {row['R']['corr']:.3f} D1 {row['D1']['corr']:.3f} D3 {row['D3']['corr']:.3f} | blurred R {row['R']['corr_blur']:.3f} D1 {row['D1']['corr_blur']:.3f} D3 {row['D3']['corr_blur']:.3f}", flush=True)
    summ = {k: {m: float(np.nanmean([r[k][m] for r in res])) for m in ("corr", "corr_blur")} for k in ("R", "D1", "D3")}
    print("mean over layouts:", json.dumps(summ), flush=True)
    for m in ("corr", "corr_blur"):
        print(f"ratio to the replicate ({m}): D1 {summ['D1'][m] / summ['R'][m]:.2f}  D3 {summ['D3'][m] / summ['R'][m]:.2f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"layouts": res, "mean": summ}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
