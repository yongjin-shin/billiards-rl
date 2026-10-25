"""
exp16_wm/single_ball.py — CT0: a table with only the cue ball. (angle, speed) -> where it stops and how many cushions it hit.

pooltool 0.5.0 cannot simulate a table with one ball (`min() iterable argument is empty`: the ball-ball collision cache is empty), so the
table gets a second ball in the `pocketed` state, off the table. It takes part in no event.

For some random start positions the whole (angle × speed) grid is simulated forward and the end position, the cushion count k (linear +
circular cushion events) and a scratch flag (the ball fell into a pocket) are stored. The analysis then asks (experiments.md "CT0"):
  M1  k = 0 law       — end distance / v^2 constant? sideways drift of a k = 0 shot?
  M2  k distribution  — per speed range (A = the env's [0.5, 8] m/s, B = extended [0.05, 8] m/s): share of k = 0, 1, 2, 3, 4+, scratch.
  M3  grid coverage   — per k: share of adjacent grid cells (same k, no scratch) whose end positions are <= 2 cm apart.
  M4  gap sizes       — median / 90th percentile of those gaps (cm) and the grid resolution a 2 cm gap would need.

Usage:
    python -m exp16_wm.single_ball run --starts 30 --phis 720 --speeds 60 --workers 8 --out logs/ct0/forward.npz
    python -m exp16_wm.single_ball analyze --file logs/ct0/forward.npz --out logs/ct0/result.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import NamedTuple

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import play_shot

CUSHION_EVENTS = (pt.EventType.BALL_LINEAR_CUSHION, pt.EventType.BALL_CIRCULAR_CUSHION)
OFF_TABLE = [-1.0, -1.0, -1.0]   # where the pocketed dummy ball sits
GAP_TOL_M = 0.02                 # "dense enough" = adjacent grid cells end within 2 cm
POCKET_CLEARANCE_M = 0.10        # starts closer than this to a pocket centre are not sampled
SPEED_RANGE_A = (0.5, 8.0)       # BilliardsEnv action range
SPEED_RANGE_B = (0.05, 8.0)      # extended range for the single-ball task
K_BINS = ("0", "1", "2", "3", "4+")


class ShotEnd(NamedTuple):
    x: float
    y: float
    k: int            # cushion events
    scratch: bool     # the ball went into a pocket (x, y is then the pocket position, not a standing place)
    t_end: float      # time of the last event


def make_cue_only_system(x: float, y: float) -> pt.System:
    """Cue ball at (x, y) at rest and a pocketed dummy ball (pooltool needs two balls)."""
    balls = {"cue": pt.Ball.create("cue", xy=(x, y)), "1": pt.Ball.create("1", xy=(0.9, 1.9))}
    dummy = balls["1"]
    dummy.state.s = pt.constants.pocketed
    dummy.state.rvw[0] = OFF_TABLE
    return pt.System(cue=pt.Cue(cue_ball_id="cue"), table=pt.Table.default(), balls=balls)


def shoot(system: pt.System, phi_deg: float, v0: float) -> ShotEnd:
    """Play (phi_deg, v0) on a copy of `system` (balls at rest) and read where the cue ball ends."""
    out = play_shot(system, phi_deg, v0)
    cue = out.balls["cue"]
    k = sum(1 for e in out.events if e.event_type in CUSHION_EVENTS)
    t_end = float(out.events[-1].time) if out.events else 0.0
    return ShotEnd(float(cue.state.rvw[0][0]), float(cue.state.rvw[0][1]), k, bool(cue.state.s == pt.constants.pocketed), t_end)


def make_grid(n_phi: int, n_v: int, v_min: float = 0.05, v_max: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Angles 0..360 deg in n_phi equal steps (360 deg excluded) and n_v log-spaced speeds in m/s."""
    return np.arange(n_phi) * (360.0 / n_phi), np.geomspace(v_min, v_max, n_v)


def sample_starts(n: int, seed: int) -> np.ndarray:
    """(n, 2) start positions in metres: ball fully on the table, not at a pocket mouth. Deterministic in seed."""
    table = pt.Table.default()
    pockets = np.array([np.asarray(p.center[:2], float) for p in table.pockets.values()])
    margin = 0.0286 + 0.005   # ball radius + 5 mm
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    while len(out) < n:
        xy = rng.uniform([margin, margin], [table.w - margin, table.l - margin])
        if np.min(np.linalg.norm(pockets - xy, axis=1)) >= POCKET_CLEARANCE_M:
            out.append(xy)
    return np.array(out)


def run_start(task: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """All (angle, speed) shots from one start; arrays of shape (n_phi, n_v)."""
    start, phis, speeds = task
    system = make_cue_only_system(float(start[0]), float(start[1]))
    shape = (len(phis), len(speeds))
    end = np.zeros(shape + (2,)); k = np.zeros(shape, np.int16); scratch = np.zeros(shape, bool); t_end = np.zeros(shape)
    for i, phi in enumerate(phis):
        for j, v in enumerate(speeds):
            r = shoot(system, float(phi), float(v))
            end[i, j] = (r.x, r.y); k[i, j] = r.k; scratch[i, j] = r.scratch; t_end[i, j] = r.t_end
    return {"end": end, "k": k, "scratch": scratch, "t_end": t_end}


def run_forward(starts: np.ndarray, phis: np.ndarray, speeds: np.ndarray, workers: int) -> dict[str, np.ndarray]:
    """Forward grid for every start; arrays (S, n_phi, n_v[, 2]) in the order of `starts`."""
    tasks = [(s, phis, speeds) for s in starts]
    t0 = time.time(); parts = []
    if workers <= 1:
        for n, t in enumerate(tasks):
            parts.append(run_start(t)); print(f"[ct0] start {n + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        with mp.Pool(workers) as pool:
            for n, p in enumerate(pool.imap(run_start, tasks)):
                parts.append(p); print(f"[ct0] start {n + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    return {key: np.stack([p[key] for p in parts]) for key in parts[0]}


# ----------------------------------------------------------------------------- analysis


def _k_bin(k: np.ndarray) -> np.ndarray:
    return np.minimum(k, 4)


def law_k0(starts: np.ndarray, phis: np.ndarray, speeds: np.ndarray, end: np.ndarray, k: np.ndarray, scratch: np.ndarray) -> dict:
    """M1: for the shots with no cushion and no scratch, end distance / v^2 and the drift to the side of the aim ray."""
    s_idx, p_idx, v_idx = np.nonzero((k == 0) & ~scratch)
    if len(s_idx) == 0:
        return {"n": 0}
    d = end[s_idx, p_idx, v_idx] - starts[s_idx]
    ray = np.stack([np.cos(np.radians(phis[p_idx])), np.sin(np.radians(phis[p_idx]))], axis=1)
    along = np.sum(d * ray, axis=1); lateral = d[:, 0] * ray[:, 1] - d[:, 1] * ray[:, 0]
    c = along / speeds[v_idx] ** 2
    return {"n": int(len(s_idx)), "c_mean": float(c.mean()), "c_rel_std": float(c.std() / c.mean()),
            "c_rel_max_dev": float(np.max(np.abs(c - c.mean())) / c.mean()), "lateral_max_mm": float(np.max(np.abs(lateral)) * 1000),
            "v_max": float(speeds[v_idx].max())}


def k_distribution(speeds: np.ndarray, k: np.ndarray, scratch: np.ndarray, v_range: tuple[float, float]) -> dict:
    """M2: share of k = 0, 1, 2, 3, 4+ (scratch counted apart) among the shots whose speed is in v_range."""
    sel = (speeds >= v_range[0] - 1e-12) & (speeds <= v_range[1] + 1e-12)
    kk, sc = k[:, :, sel], scratch[:, :, sel]
    n = kk.size
    out = {"n": int(n), "scratch": float(sc.mean())}
    kb = _k_bin(kk)
    for i, name in enumerate(K_BINS):
        out[name] = float(np.mean((kb == i) & ~sc))
    return out


def adjacent_gaps(end: np.ndarray, k: np.ndarray, scratch: np.ndarray, speeds: np.ndarray, v_range: tuple[float, float]) -> dict:
    """
    M3 + M4: end-position distance between adjacent grid cells, split by k (of both cells; cells with different k or a scratch are not
    comparable — they are counted in boundary_share). axis 'phi' (wraps at 360 deg) and 'v' (within the speed range).
    """
    in_range = (speeds >= v_range[0] - 1e-12) & (speeds <= v_range[1] + 1e-12)
    out: dict = {}
    for axis in ("phi", "v"):
        idx = np.nonzero(in_range)[0]
        if axis == "phi":
            e0, e1 = end[:, :, idx], np.roll(end, -1, axis=1)[:, :, idx]
            k0, k1 = k[:, :, idx], np.roll(k, -1, axis=1)[:, :, idx]
            s0, s1 = scratch[:, :, idx], np.roll(scratch, -1, axis=1)[:, :, idx]
        else:
            idx = idx[:-1][np.diff(idx) == 1]   # cells whose next speed is also in range
            e0, e1 = end[:, :, idx], end[:, :, idx + 1]
            k0, k1 = k[:, :, idx], k[:, :, idx + 1]
            s0, s1 = scratch[:, :, idx], scratch[:, :, idx + 1]
        gap = np.linalg.norm(e1 - e0, axis=-1)
        ok = (k0 == k1) & ~s0 & ~s1
        per_k = {}
        for i, name in enumerate(K_BINS):
            sel = ok & (_k_bin(k0) == i)
            g = gap[sel]
            per_k[name] = ({"n": int(sel.sum()), "within_2cm": float(np.mean(g <= GAP_TOL_M)), "median_cm": float(np.median(g) * 100),
                            "p90_cm": float(np.percentile(g, 90) * 100)} if sel.any() else {"n": 0})
        out[axis] = {"boundary_share": float(1.0 - ok.mean()), "per_k": per_k}
    return out


def analyze(npz_path: str) -> dict:
    z = np.load(npz_path)
    starts, phis, speeds, end, k, scratch = z["starts"], z["phis"], z["speeds"], z["end"], z["k"], z["scratch"]
    ranges = {"A": SPEED_RANGE_A, "B": SPEED_RANGE_B}
    return {
        "commit": str(z["commit"]), "n_starts": int(len(starts)), "n_phi": int(len(phis)), "n_v": int(len(speeds)),
        "phi_step_deg": float(phis[1] - phis[0]), "speeds_m_s": [float(speeds[0]), float(speeds[-1])],
        "M1_k0_law": law_k0(starts, phis, speeds, end, k, scratch),
        "M2_k_distribution": {name: k_distribution(speeds, k, scratch, r) for name, r in ranges.items()},
        "M3_M4_adjacent_gaps": {name: adjacent_gaps(end, k, scratch, speeds, r) for name, r in ranges.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--starts", type=int, default=30); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--phis", type=int, default=720); r.add_argument("--speeds", type=int, default=60)
    r.add_argument("--v-min", type=float, default=0.05); r.add_argument("--v-max", type=float, default=8.0)
    r.add_argument("--workers", type=int, default=8); r.add_argument("--out", required=True)
    a = sub.add_parser("analyze"); a.add_argument("--file", required=True); a.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "run":
        from exp16_wm.code_version import code_version
        starts = sample_starts(args.starts, args.seed); phis, speeds = make_grid(args.phis, args.speeds, args.v_min, args.v_max)
        data = run_forward(starts, phis, speeds, args.workers)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, starts=starts, phis=phis, speeds=speeds, commit=code_version(), seed=args.seed, **data)
        print("saved", args.out)
    else:
        res = analyze(args.file)
        text = json.dumps(res, indent=2, ensure_ascii=False)
        print(text)
        if args.out:
            open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
