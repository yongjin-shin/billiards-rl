"""
exp16_wm/v_map_data.py — V7: value maps V(n, m | s) for many object-ball layouts s, as supervised data for exp16_wm.train_v.

For a layout s the cue ball is put on ~700 free positions (a `step` grid with random jitter). At every position the policy plays out
(1) the exact position and (2) `copies` copies moved by each radius of `radii_mm` (a ring: the moved distance is that radius, the
direction random) until its first miss. A position's training label is the mean over the radii of the mean balls pocketed by the
copies of that radius — high only where the shot still goes in when the cue ball is off by 1 ... 6 cm. The output files use the format
train_v reads (obs, k, left, game = layout id, branch) with k a float, plus the exact result, every radius's mean and the position.

Usage:
    python -m exp16_wm.v_map_data --data "logs/teacher_data/v2/s30*.npz" --out-dir logs/teacher_data/v7/train --layouts-per-left 4 \
        --runs "<PRm run dir glob>" --workers 8
"""

import argparse
import glob
import os
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RADII_MM = (10.0, 20.0, 30.0, 40.0, 60.0)


def pick_layouts(paths: list[str], per_left: int, seed: int, lefts: tuple[int, ...] = (2, 3, 4, 5, 6)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """per_left states for each balls-left value in `lefts`, from distinct games, chosen at random: (obs (L, D), left (L,), game key (L,)).
    The cue ball's position in obs is irrelevant (it is replaced); a game key is "<file index>:<game id>"-unique as a number."""
    rng = np.random.default_rng(seed)
    obs, left, key = [], [], []
    for f, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        obs.append(z["obs"]); left.append(z["left"]); key.append(f * 10_000_000 + z["game"].astype(np.int64))
    obs, left, key = np.concatenate(obs), np.concatenate(left), np.concatenate(key)
    chosen: list[int] = []
    used: set[int] = set()
    for ell in lefts:
        rows = rng.permutation(np.nonzero(left == ell)[0])
        n = 0
        for r in rows:
            if int(key[r]) in used:
                continue
            used.add(int(key[r])); chosen.append(int(r)); n += 1
            if n == per_left:
                break
        if n < per_left:
            raise ValueError(f"only {n} distinct games with {ell} balls left, asked {per_left}")
    idx = np.array(chosen)
    return obs[idx], left[idx], key[idx]


def sample_cells(raw: np.ndarray, step: float, jitter: float, rng: np.random.Generator, ball_r: float, w: float, l: float) -> np.ndarray:
    """(M, 2) cue positions in metres: a step grid inside the cushions, each point shifted by up to ± jitter in x and y, kept only where
    the ball is on the table (centre one radius from the cushion) and at least two radii from every ball still on it."""
    from exp16_wm.v_heatmap import live_balls
    xs, ys = np.arange(ball_r, w - ball_r + 1e-9, step), np.arange(ball_r, l - ball_r + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1) + rng.uniform(-jitter, jitter, (gx.size, 2))
    ok = (pts[:, 0] >= ball_r) & (pts[:, 0] <= w - ball_r) & (pts[:, 1] >= ball_r) & (pts[:, 1] <= l - ball_r)
    for b in live_balls(raw, w, l):
        ok &= np.hypot(pts[:, 0] - b[0], pts[:, 1] - b[1]) >= 2.0 * ball_r + 1e-4
    return pts[ok]


def ring_copies(base: np.ndarray, radii_m: tuple[float, ...], copies: int, rng: np.random.Generator, w: float, l: float, ball_r: float) -> tuple[np.ndarray, np.ndarray]:
    """The copies of a position moved by each radius (a random direction each): obs (R·copies, D) and a placed mask (R·copies,) — a copy
    that could not be placed (v_perturb.perturb_cue found no free direction) keeps the original position and is marked False."""
    from exp16_wm.v_perturb import perturb_cue
    out = np.repeat(base[None, :], len(radii_m) * copies, axis=0).astype(np.float32)
    placed = np.zeros(len(radii_m) * copies, bool)
    for ri, r in enumerate(radii_m):
        for c in range(copies):
            moved = perturb_cue(base, r, rng, w, l, ball_r)
            if moved is not None:
                out[ri * copies + c], placed[ri * copies + c] = moved, True
    return out, placed


def aggregate(k_exact: np.ndarray, ring_k: np.ndarray) -> dict[str, np.ndarray]:
    """k_exact (C,), ring_k (C, R, copies) with −1 = unplaced copy → ring_mean (C, R) (NaN where no copy was placed), the label
    "robust" = mean over the radii that have a mean (the exact k if none has), and "worst" = min over those radii."""
    placed = ring_k >= 0
    n = placed.sum(axis=2)
    ring_mean = np.where(n > 0, np.where(placed, ring_k, 0).sum(axis=2) / np.maximum(n, 1), np.nan)
    any_ring = np.isfinite(ring_mean).any(axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                  # rows without any ring: empty mean, replaced below
        robust = np.where(any_ring, np.nanmean(ring_mean, axis=1), k_exact.astype(float))
        worst = np.where(any_ring, np.nanmin(np.where(np.isfinite(ring_mean), ring_mean, np.inf), axis=1), k_exact.astype(float))
    return {"ring_mean": ring_mean.astype(np.float32), "robust": robust.astype(np.float32), "worst": worst.astype(np.float32)}


def build_layout(raw: np.ndarray, left: int, layout_id: int, seed: int, radii_mm: tuple[float, ...], copies: int, step: float, jitter: float,
                 ball_r: float, w: float, l: float) -> dict[str, np.ndarray]:
    """Everything to simulate for one layout: the cells, and obs (rows = per cell the exact position followed by its ring copies)."""
    from exp16_wm.v_heatmap import with_cue
    rng = np.random.default_rng([seed, layout_id])
    cells = sample_cells(raw, step, jitter, rng, ball_r, w, l)
    radii_m = tuple(r / 1000.0 for r in radii_mm)
    per = 1 + len(radii_m) * copies
    obs = np.zeros((len(cells) * per, len(raw)), np.float32)
    placed = np.zeros((len(cells), len(radii_m) * copies), bool)
    for i, (x, y) in enumerate(cells):
        base = with_cue(raw, x, y, w, l)
        cp, pl = ring_copies(base, radii_m, copies, rng, w, l, ball_r)
        obs[i * per] = base
        obs[i * per + 1:(i + 1) * per] = cp
        placed[i] = pl
    return {"cells": cells, "obs": obs, "placed": placed, "per": np.array(per), "left": np.array(left)}


def finish_layout(lay: dict[str, np.ndarray], ks: np.ndarray, n_radii: int, copies: int) -> dict[str, np.ndarray]:
    """From the simulated k of a layout's obs rows to its per-cell arrays: k_exact, ring_k (−1 where the copy was not placed), labels."""
    per = int(lay["per"])
    kk = np.asarray(ks).reshape(-1, per)
    k_exact = kk[:, 0].astype(np.int64)
    ring_k = np.where(lay["placed"], kk[:, 1:], -1).reshape(-1, n_radii, copies).astype(np.int64)
    return {"k_exact": k_exact, "ring_k": ring_k, **aggregate(k_exact, ring_k)}


def run_layouts(layouts: list[dict[str, np.ndarray]], roll: Callable[[list[tuple[np.ndarray, int]]], list[np.ndarray]], chunk: int = 240) -> list[np.ndarray]:
    """Simulate every layout's obs with `roll` (a map over (obs chunk, balls left) jobs → k arrays; a worker pool in the real run)."""
    jobs, owner = [], []
    for li, lay in enumerate(layouts):
        for s in range(0, len(lay["obs"]), chunk):
            jobs.append((lay["obs"][s:s + chunk], int(lay["left"]))); owner.append(li)
    res = roll(jobs)
    out = [[] for _ in layouts]
    for li, r in zip(owner, res):
        out[li].append(np.asarray(r))
    return [np.concatenate(o) for o in out]


def write_file(path: str, raws: np.ndarray, lefts: np.ndarray, ids: np.ndarray, lays: list[dict[str, np.ndarray]], fins: list[dict[str, np.ndarray]],
               radii_mm: tuple[float, ...], target: str = "robust") -> int:
    """Write one train_v-compatible npz for these layouts: obs (cue at the cell), k (float label), left, game (= layout id), branch, plus
    k_exact, ring_mean, ring_k (every copy's result: (positions, radii, copies), −1 = not placed), cell_xy, layout_id, radii_mm.
    Returns the number of positions."""
    from exp16_wm.v_heatmap import with_cue
    from exp16_wm.networks import TABLE_L, TABLE_W
    obs, k, left, game, kx, rm, xy, rk = [], [], [], [], [], [], [], []
    for raw, ell, gid, lay, fin in zip(raws, lefts, ids, lays, fins):
        n = len(lay["cells"])
        obs.append(np.stack([with_cue(raw, x, y, TABLE_W, TABLE_L) for x, y in lay["cells"]]))
        k.append(fin[target]); left.append(np.full(n, ell)); game.append(np.full(n, gid)); kx.append(fin["k_exact"]); rm.append(fin["ring_mean"]); xy.append(lay["cells"])
        rk.append(fin["ring_k"].astype(np.int8))
    np.savez(path, obs=np.concatenate(obs).astype(np.float32), k=np.concatenate(k).astype(np.float32), left=np.concatenate(left).astype(np.int64),
             game=np.concatenate(game).astype(np.int64), branch=np.zeros(sum(len(a) for a in k), bool), k_exact=np.concatenate(kx),
             ring_mean=np.concatenate(rm), ring_k=np.concatenate(rk), cell_xy=np.concatenate(xy).astype(np.float32), layout_id=np.concatenate(game).astype(np.int64),
             radii_mm=np.array(radii_mm), meta=str({"target": target}))
    return int(sum(len(a) for a in k))


def label_from_copies(ring_k: np.ndarray, radii_mm: np.ndarray, radii_sel: list[float], copies_sel: list[int], k_exact: np.ndarray) -> np.ndarray:
    """Mean of the placed copies of the chosen radii (radii_sel, in mm) and copy numbers (copies_sel) of every position; the exact k where
    none of the chosen copies was placed. E.g. radii [10], copies [0, 1, 2]: three copies at one distance; radii [10, 30, 60], copies [0]:
    one copy at each of three distances — the same number of samples spread over distances."""
    ri = [int(np.nonzero(np.isclose(radii_mm, r))[0][0]) for r in radii_sel]
    sel = ring_k[:, ri, :][:, :, copies_sel].reshape(len(ring_k), -1).astype(float)
    ok = sel >= 0
    n = ok.sum(axis=1)
    return np.where(n > 0, np.where(ok, sel, 0.0).sum(axis=1) / np.maximum(n, 1), k_exact.astype(float)).astype(np.float32)


def relabel(src: str, dst: str, target: str) -> int:
    """Copy a V7 file with another label in k: "exact" (the exact position's k), "robust" (mean over the radii), "worst" (min over the radii),
    "ring<mm>" (that radius only, e.g. ring10; the exact k where no copy of it was placed) or "sel:<radii mm>:<copy numbers>" (the mean of
    just those copies, e.g. sel:10,30,60:0; needs the per-copy results). Returns the number of positions."""
    z = dict(np.load(src, allow_pickle=True))
    kx, rm, radii = z["k_exact"].astype(np.float32), z["ring_mean"], z["radii_mm"]
    if target == "exact":
        k = kx
    elif target in ("robust", "worst"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            k = np.where(np.isfinite(rm).any(1), np.nanmean(rm, 1) if target == "robust" else np.nanmin(np.where(np.isfinite(rm), rm, np.inf), 1), kx)
    elif target.startswith("sel:"):                                                              # sel:<radii mm, comma>:<copy numbers, comma>
        if "ring_k" not in z:
            raise ValueError(f"{src} has no per-copy results (ring_k): it was written by an older generator")
        _, rs, cs = target.split(":")
        k = label_from_copies(z["ring_k"], radii, [float(r) for r in rs.split(",")], [int(c) for c in cs.split(",")], z["k_exact"])
    elif target.startswith("ring"):
        j = int(np.nonzero(np.isclose(radii, float(target[4:])))[0][0])
        k = np.where(np.isfinite(rm[:, j]), rm[:, j], kx)
    else:
        raise ValueError(f"target must be exact, robust, worst or ring<mm>, got {target!r}")
    z["k"] = k.astype(np.float32); z["meta"] = np.array(str({"target": target}))
    np.savez(dst, **z)
    return len(k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--relabel", nargs=3, metavar=("SRC_GLOB", "DST_DIR", "TARGET"), help="instead of simulating: copy files with another label (exact | robust | worst | ring<mm>)")
    p.add_argument("--data", help="V data files to take the layouts from"); p.add_argument("--out-dir")
    p.add_argument("--layouts-per-left", type=int, default=4); p.add_argument("--per-file", type=int, default=10, help="layouts per output file (resume unit)")
    p.add_argument("--radii-mm", type=float, nargs="+", default=list(RADII_MM)); p.add_argument("--copies", type=int, default=3)
    p.add_argument("--step", type=float, default=0.05); p.add_argument("--jitter", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--id-offset", type=int, default=0, help="layout ids start here (keep train / held-out ids apart)")
    p.add_argument("--runs", nargs="+"); p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.relabel:
        src, dst, target = args.relabel
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(src)):
            print(f"{os.path.basename(f)} -> {target}: {relabel(f, os.path.join(dst, os.path.basename(f)), target)} positions", flush=True)
        return
    if not (args.data and args.out_dir and args.runs):
        raise SystemExit("simulating needs --data, --out-dir and --runs")
    from multiprocessing import Pool
    import pooltool as pt
    from exp16_wm import v_heatmap as H
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.paired_eval import find_run
    paths = sorted(glob.glob(args.data))
    raws, lefts, _ = pick_layouts(paths, args.layouts_per_left, args.seed)
    ids = args.id_offset + np.arange(len(raws))
    os.makedirs(args.out_dir, exist_ok=True)
    ball_r = float(pt.BallParams.default().R)
    radii = tuple(args.radii_mm)
    print(f"{len(raws)} layouts from {len(paths)} files; {len(radii)} radii × {args.copies} copies + exact per position", flush=True)
    with Pool(args.workers, initializer=H._init, initargs=([find_run(g) for g in args.runs], n_balls_raw(raws.shape[1]))) as pool:
        for f0 in range(0, len(raws), args.per_file):
            out = os.path.join(args.out_dir, f"layouts_{int(ids[f0]):05d}.npz")
            if os.path.exists(out):
                print(f"skip {out}", flush=True)
                continue
            t0 = time.time()
            sl = slice(f0, f0 + args.per_file)
            lays = [build_layout(raws[i], int(lefts[i]), int(ids[i]), args.seed, radii, args.copies, args.step, args.jitter, ball_r, TABLE_W, TABLE_L)
                    for i in range(len(raws))[sl]]
            ks = run_layouts(lays, lambda jobs: pool.map(H._roll, jobs))
            fins = [finish_layout(lay, k, len(radii), args.copies) for lay, k in zip(lays, ks)]
            n = write_file(out, raws[sl], lefts[sl], ids[sl], lays, fins, radii)
            print(f"{out}: {len(lays)} layouts, {n} positions, {sum(len(o) for o in (l['obs'] for l in lays))} simulations in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
