"""
exp16_wm/cue_goal_data.py — CT0b: collect cue-only shots whose END positions cover the table (experiments.md "CT0b").

For one start position: seed with random shots (angle uniform, speed log-uniform 0.001 ~ 8 m/s), then repeat: find the (cushion count k, 2.5 cm
cell) entries that are empty or thin but touch a filled one, pick a shot ending next to them and shake its angle and speed by a random
amount, play the shaken shots. No speed floor, no physics formula: the table fills outward from what the simulator shows.

Every shot is a training example for the goal "where it ended" (hindsight). What is measured: for random goals on the table, the share that
has a recorded shot of a given k ending within 2 cm / 1 cm.

Usage:
    python -m exp16_wm.cue_goal_data collect --starts 4 --workers 4 --out logs/ct0b/smoke.npz
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pooltool as pt
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.single_ball import make_cue_only_system, sample_starts, shoot

_TABLE = pt.Table.default()
TABLE_W, TABLE_L = float(_TABLE.w), float(_TABLE.l)
BALL_R = 0.028575
N_K = 4   # k = 0, 1, 2, 3+


@dataclass(frozen=True)
class Params:
    cell_m: float = 0.025
    n0: int = 3000            # seed shots
    per_cell: int = 2         # a (k, cell) entry with fewer shots than this is thin
    children: int = 8         # shaken shots per thin entry and round
    max_rounds: int = 12
    max_attempts: int = 4     # rounds a thin entry may be tried before it is given up
    max_shots: int = 100_000  # per start
    v_seed: tuple[float, float] = (1e-3, 8.0)
    v_clip: tuple[float, float] = (1e-4, 8.0)


def grid_shape(cell_m: float) -> tuple[int, int]:
    return int(np.ceil(TABLE_W / cell_m)), int(np.ceil(TABLE_L / cell_m))


def standing_mask(cell_m: float) -> np.ndarray:
    """(NX, NY) bool: cells whose centre a ball can rest on (ball fully on the table)."""
    nx, ny = grid_shape(cell_m)
    cx = (np.arange(nx) + 0.5) * cell_m
    cy = (np.arange(ny) + 0.5) * cell_m
    return ((cx >= BALL_R) & (cx <= TABLE_W - BALL_R))[:, None] & ((cy >= BALL_R) & (cy <= TABLE_L - BALL_R))[None, :]


def cell_index(end: np.ndarray, cell_m: float) -> np.ndarray:
    """(n, 2) int cell (ix, iy) of each end position, clipped onto the grid."""
    nx, ny = grid_shape(cell_m)
    ij = np.floor(end / cell_m).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, nx - 1); ij[:, 1] = np.clip(ij[:, 1], 0, ny - 1)
    return ij


def count_cells(end: np.ndarray, k: np.ndarray, scratch: np.ndarray, cell_m: float) -> np.ndarray:
    """(N_K, NX, NY) number of shots per (k bin, cell); shots that fell into a pocket are not counted."""
    nx, ny = grid_shape(cell_m)
    counts = np.zeros((N_K, nx, ny), np.int32)
    ok = ~scratch
    ij = cell_index(end[ok], cell_m)
    np.add.at(counts, (np.minimum(k[ok], N_K - 1), ij[:, 0], ij[:, 1]), 1)
    return counts


def thin_entries(counts: np.ndarray, attempts: np.ndarray, prm: Params) -> np.ndarray:
    """(m, 3) (k bin, ix, iy) of entries that are thin, stand on the table, touch a filled entry of the same k and have attempts left."""
    filled = (counts >= 1).astype(np.uint8)
    touches = maximum_filter(filled, size=(1, 3, 3), mode="constant") > 0
    mask = standing_mask(prm.cell_m)[None] & (counts < prm.per_cell) & touches & (attempts < prm.max_attempts)
    return np.argwhere(mask)


def shake(phi: np.ndarray, v: np.ndarray, rng: np.random.Generator, v_clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Random angle (0.003 ~ 2 deg) and relative speed (0.1 ~ 20 %) changes, sizes log-uniform."""
    m = len(phi)
    d_phi = 10 ** rng.uniform(-2.5, 0.3, m) * rng.standard_normal(m)
    d_v = 10 ** rng.uniform(-3.0, -0.7, m) * rng.standard_normal(m)
    return (phi + d_phi) % 360.0, np.clip(v * np.exp(d_v), *v_clip)


def _play(system: pt.System, phis: np.ndarray, vs: np.ndarray) -> dict[str, np.ndarray]:
    end = np.zeros((len(phis), 2)); k = np.zeros(len(phis), np.int16); scratch = np.zeros(len(phis), bool)
    for i, (phi, v) in enumerate(zip(phis, vs)):
        r = shoot(system, float(phi), float(v))
        end[i] = (r.x, r.y); k[i] = r.k; scratch[i] = r.scratch
    return {"phi": phis, "v": vs, "end": end, "k": k, "scratch": scratch}


def collect_start(task: tuple[np.ndarray, int, Params]) -> dict:
    """All shots for one start position, plus a per-round trace of how many (k, cell) entries are filled."""
    start, seed, prm = task
    t0 = time.time()
    rng = np.random.default_rng(seed)
    system = make_cue_only_system(float(start[0]), float(start[1]))
    seed_v = np.exp(rng.uniform(np.log(prm.v_seed[0]), np.log(prm.v_seed[1]), prm.n0))
    chunks = [_play(system, rng.uniform(0.0, 360.0, prm.n0), seed_v)]
    rounds = [np.zeros(prm.n0, np.int16)]
    stand = standing_mask(prm.cell_m)
    attempts = np.zeros((N_K,) + stand.shape, np.int16)
    trace = []

    def cat(key: str) -> np.ndarray:
        return np.concatenate([c[key] for c in chunks])

    for rnd in range(0, prm.max_rounds + 1):
        end, k, scratch = cat("end"), cat("k"), cat("scratch")
        counts = count_cells(end, k, scratch, prm.cell_m)
        trace.append({"round": rnd, "shots": int(len(k)), "filled": [int(((counts[b] >= 1) & stand).sum()) for b in range(N_K)]})
        if rnd == prm.max_rounds or len(k) >= prm.max_shots:
            break
        targets = thin_entries(counts, attempts, prm)
        if len(targets) == 0:
            break
        attempts[targets[:, 0], targets[:, 1], targets[:, 2]] += 1
        # parents: non-scratch shots by (k bin, cell)
        ok = np.nonzero(~scratch)[0]
        ij = cell_index(end[ok], prm.cell_m)
        pool: dict[tuple[int, int, int], list[int]] = {}
        for idx, (a, b) in zip(ok, ij):
            pool.setdefault((min(int(k[idx]), N_K - 1), int(a), int(b)), []).append(int(idx))
        parents = []
        for kb, ix, iy in targets:
            near = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in pool.get((int(kb), int(ix) + dx, int(iy) + dy), ())]
            if near:
                parents.extend(rng.choice(near, size=prm.children).tolist())
        if not parents:
            break
        parents = np.array(parents)[: max(0, prm.max_shots - len(k))]
        new_phi, new_v = shake(cat("phi")[parents], cat("v")[parents], rng, prm.v_clip)
        chunks.append(_play(system, new_phi, new_v)); rounds.append(np.full(len(parents), rnd + 1, np.int16))
    out = {key: cat(key) for key in chunks[0]}
    out["round"] = np.concatenate(rounds)
    out["start"] = np.asarray(start, float)
    out["trace"] = trace
    out["seconds"] = time.time() - t0
    return out


def hit_rates(end: np.ndarray, k: np.ndarray, scratch: np.ndarray, goals: np.ndarray, tols: tuple[float, ...] = (0.02, 0.01)) -> dict:
    """Share of goals (n, 2) with a recorded shot ending within tol metres: per k bin ('k0' .. 'k3+') and over all k ('any')."""
    out = {}
    ok = ~scratch
    groups = {f"k{b}" if b < N_K - 1 else "k3+": ok & (np.minimum(k, N_K - 1) == b) for b in range(N_K)}
    groups["any"] = ok
    for name, sel in groups.items():
        if not sel.any():
            out[name] = {f"{int(t * 100)}cm": 0.0 for t in tols}
            continue
        d, _ = cKDTree(end[sel]).query(goals)
        out[name] = {f"{int(t * 100)}cm": float(np.mean(d <= t)) for t in tols}
    return out


def collect(starts: np.ndarray, prm: Params, seed: int, workers: int, goals: np.ndarray) -> tuple[list[dict], list[dict]]:
    tasks = [(s, seed + 1000 * i, prm) for i, s in enumerate(starts)]
    results, summaries = [], []
    t0 = time.time()

    def take(i: int, res: dict) -> None:
        results.append(res)
        summaries.append({"start": res["start"].tolist(), "shots": int(len(res["k"])), "seconds": res["seconds"], "trace": res["trace"],
                          "hit": hit_rates(res["end"], res["k"], res["scratch"], goals)})
        h = summaries[-1]["hit"]
        print(f"[ct0b] start {i + 1}/{len(tasks)} shots={len(res['k'])} {res['seconds']:.0f}s  2cm any={h['any']['2cm']:.3f} "
              f"k0={h['k0']['2cm']:.3f} k1={h['k1']['2cm']:.3f} k2={h['k2']['2cm']:.3f} k3+={h['k3+']['2cm']:.3f}  total {time.time() - t0:.0f}s", flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            take(i, collect_start(t))
    else:
        with mp.Pool(workers) as pool:
            for i, res in enumerate(pool.imap(collect_start, tasks)):
                take(i, res)
    return results, summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--starts", type=int, default=30); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--workers", type=int, default=8); c.add_argument("--goals", type=int, default=1000)
    c.add_argument("--max-shots", type=int, default=100_000); c.add_argument("--max-rounds", type=int, default=12)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    from exp16_wm.code_version import code_version
    prm = Params(max_shots=args.max_shots, max_rounds=args.max_rounds)
    starts = sample_starts(args.starts, args.seed)
    goals = sample_starts(args.goals, 12345)
    results, summaries = collect(starts, prm, args.seed, args.workers, goals)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sizes = np.array([len(r["k"]) for r in results])
    np.savez_compressed(args.out, starts=starts, start_idx=np.repeat(np.arange(len(results)), sizes),
                        **{key: np.concatenate([r[key] for r in results]) for key in ("phi", "v", "end", "k", "scratch", "round")},
                        commit=code_version(), params=json.dumps(asdict(prm)))
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump({"commit": code_version(), "params": asdict(prm), "starts": summaries}, f, indent=1)
    mean = lambda key, tol: float(np.mean([s["hit"][key][tol] for s in summaries]))
    print("mean 2cm hit:", {k_: round(mean(k_, "2cm"), 3) for k_ in ("k0", "k1", "k2", "k3+", "any")})
    print("mean 1cm hit:", {k_: round(mean(k_, "1cm"), 3) for k_ in ("k0", "k1", "k2", "k3+", "any")})
    print("shots/start mean", float(sizes.mean()), "seconds/start mean", float(np.mean([s["seconds"] for s in summaries])))


if __name__ == "__main__":
    main()
