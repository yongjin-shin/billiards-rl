"""
exp16_wm/v_decision_ceiling.py — V5-d: what picking by the next position can reach when the cue ball is only known to ε, by balls left.

No simulation and no training: it reads the V5 test set (exp16_wm.v_testset) and the V5-c perturbation result (exp16_wm.v_perturb --testset)
and scores four "next position value" sources with the test set's own metrics (evaluate_testset):
  ceiling  the mean over the ε copies' balls (what a predictor that knows the next position up to ε can best say),
  single   one ε copy (averaged over the copies),
  V        trained V runs, made-max  the made head's highest probability on the next position:
           [exact] on the simulator's next position, [ε] on the ε-moved copies (regenerated with v_perturb's seed; mean over copies / single).
The truth is always the unmoved position's result. Next positions with an unplaced copy are filled with the mean over the same
balls-left group (never the truth). The same metrics are also reported per origin balls-left.

Usage:
    python -m exp16_wm.v_decision_ceiling --testset logs/teacher_data/v5/testset_s950.npz --ceiling logs/teacher_data/v5/order_ceiling_s950.npz \
        --runs "<PRm run dir glob>" --v-runs logs/experiments_v2/v2_initkeep --out logs/teacher_data/v5/decision_ceiling_s950.json
"""

import argparse
import ast
import json
import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_testset import evaluate_testset

ROW_KEYS = ("made", "succ", "balls", "k_next", "cleared", "valid")


def fill_missing(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    """v (M,) with NaN where unknown: NaN → the mean of the finite values with the same group label (overall mean if the group has none)."""
    out = v.astype(float).copy()
    ok = np.isfinite(out)
    if not ok.any():
        raise ValueError("no finite value to fill from")
    for g in np.unique(group):
        m = group == g
        fin = m & ok
        out[m & ~ok] = out[fin].mean() if fin.any() else out[ok].mean()
    return out


def to_grid(use: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v (M,) in row-major order of the True cells of use (N, K) → (N, K) with 0 elsewhere."""
    g = np.zeros(use.shape)
    g[use] = v
    return g


def _sources(t: dict[str, np.ndarray], mat: np.ndarray) -> dict[str, np.ndarray | list[np.ndarray]]:
    """mat (M, C): a value per ε-copy of the M rankable next positions (NaN = copy missing) → (N, K) grids: the mean over copies
    ("ceiling") and each copy ("single"), holes filled per balls-left of the next position."""
    use = t["valid"] & t["succ"] & ~t["cleared"]
    if len(mat) != int(use.sum()):
        raise ValueError(f"{len(mat)} rows, but the test set has {int(use.sum())} rankable next positions")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # rows with no copy → NaN mean, filled below
        mean = np.nanmean(mat, axis=1)
    group = t["left_next"][use]
    return {"ceiling": to_grid(use, fill_missing(mean, group)),
            "single": [to_grid(use, fill_missing(mat[:, j], group)) for j in range(mat.shape[1])]}


def next_value_sources(t: dict[str, np.ndarray], k_pert: np.ndarray) -> dict[str, np.ndarray | list[np.ndarray]]:
    """k_pert (M, C): balls after the shot for each of C ε-copies of the rankable next positions (−1 = copy could not be placed).
    Returns {"ceiling": (N, K) mean over copies, "single": [(N, K) per copy]}."""
    return _sources(t, np.where(k_pert >= 0, k_pert, np.nan).astype(float))


def regenerate_perturbed(obs: np.ndarray, flat_idx: np.ndarray, eps_m: float, copies: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The ε-moved positions exp16_wm.v_perturb made for these next positions (same rng key [seed, flat index], same order):
    obs (M, D), flat_idx (M,) the position's index in the row-major (origin × candidate) test set → (M, C, D), placed (M, C).
    Copies that could not be placed keep the original position and have placed False."""
    import pooltool as pt
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.v_perturb import perturb_cue
    ball_r = float(pt.BallParams.default().R)
    out = np.repeat(obs[:, None, :], copies, axis=1).astype(np.float32)
    placed = np.zeros((len(obs), copies), bool)
    for i, (r, raw) in enumerate(zip(flat_idx, obs)):
        rng = np.random.default_rng([seed, int(r)])
        for c in range(copies):
            moved = perturb_cue(raw, eps_m, rng, TABLE_W, TABLE_L, ball_r)
            if moved is not None:
                out[i, c], placed[i, c] = moved, True
    return out, placed


def perturbed_value_sources(t: dict[str, np.ndarray], obs_pert: np.ndarray, placed: np.ndarray,
                            value_fn: Callable[[dict[str, np.ndarray]], np.ndarray]) -> dict[str, np.ndarray | list[np.ndarray]]:
    """value_fn(test set) → (N, K) values of its next_obs; applied to each ε-copy (obs_pert (M, C, D) in the rankable cells' row-major
    order, placed (M, C)). Same return as next_value_sources."""
    use = t["valid"] & t["succ"] & ~t["cleared"]
    N, K, D = t["next_obs"].shape
    cells = np.nonzero(use.reshape(-1))[0]
    cols = []
    for c in range(obs_pert.shape[1]):
        flat = t["next_obs"].reshape(-1, D).copy()
        flat[cells] = obs_pert[:, c]
        v = value_fn({**t, "next_obs": flat.reshape(N, K, D)})[use]
        cols.append(np.where(placed[:, c], v, np.nan))
    return _sources(t, np.stack(cols, axis=1))


def subset(t: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {k: t[k][rows] for k in ROW_KEYS}


def metrics_by_left(t: dict[str, np.ndarray], left: np.ndarray, value: np.ndarray | list[np.ndarray]) -> dict[str, dict[str, float]]:
    """evaluate_testset on all origins ("all") and per origin balls-left; value is one (N, K) array or a list of them (metrics averaged)."""
    vals = value if isinstance(value, list) else [value]
    out: dict[str, dict[str, float]] = {}
    for name, rows in [("all", np.ones(len(left), bool))] + [(str(int(g)), left == g) for g in np.unique(left)]:
        ts = subset(t, rows)
        ms = [evaluate_testset(ts, v[rows]) for v in vals]
        out[name] = {k: float(np.nanmean([m[k] for m in ms])) if any(np.isfinite(m[k]) for m in ms) else float("nan") for k in ms[0]}
    return out


def pooled_diff(by_left: dict[str, dict[str, float]], low: tuple[int, ...] = (2, 3), high: tuple[int, ...] = (5, 6), key: str = "pair_acc") -> float:
    """d of the V5-d table: mean of key over the few-balls-left groups minus over the many-balls-left groups (each group weighted equally)."""
    lo = [by_left[str(g)][key] for g in low if str(g) in by_left]
    hi = [by_left[str(g)][key] for g in high if str(g) in by_left]
    if not lo or not hi:
        return float("nan")
    return float(np.nanmean(lo) - np.nanmean(hi))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", required=True); p.add_argument("--ceiling", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="made run globs (for the made-max baseline)")
    p.add_argument("--v-runs", nargs="*", default=[]); p.add_argument("--out", default=""); p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_testset import made_max_values, next_values
    z = np.load(args.testset, allow_pickle=True)
    t = {k: z[k] for k in (*ROW_KEYS, "next_obs", "left_next")}
    left = z["left"]
    c = np.load(args.ceiling, allow_pickle=True)
    kp = c["k_pert"]
    if kp.shape[1] != 1:
        raise ValueError(f"expected one ε in the ceiling file, found {kp.shape[1]}")
    eps_mm = float(c["eps_mm"][0]); copies = kp.shape[2]
    use = t["valid"] & t["succ"] & ~t["cleared"]
    cells = np.nonzero(use.reshape(-1))[0]
    meta = ast.literal_eval(str(c["meta"]))                               # v_perturb's own record (seed)
    obs_p, placed = regenerate_perturbed(t["next_obs"].reshape(-1, t["next_obs"].shape[2])[cells], cells, eps_mm / 1000.0, copies, int(meta["seed"]))
    if not np.array_equal(placed, kp[:, 0, :] >= 0):
        raise RuntimeError("the regenerated ε copies are not the ones v_perturb simulated (placed masks differ)")
    print(f"regenerated {placed.size} ε copies, placed masks identical to v_perturb's", flush=True)
    src = next_value_sources(t, kp[:, 0, :])
    mm_runs = [find_run(g) for g in args.runs]
    fns: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {"made max": lambda tc: made_max_values(tc, mm_runs)}
    for r in args.v_runs:
        fns[os.path.basename(r.rstrip("/"))] = lambda tc, r=r: next_values(tc, find_run(r))
    rows: dict[str, np.ndarray | list[np.ndarray]] = {}
    for name, fn in fns.items():
        rows[f"[exact] {name}"] = fn(t)
    rows[f"[{eps_mm / 10:g}cm] ceiling ({copies} copies)"] = src["ceiling"]
    rows[f"[{eps_mm / 10:g}cm] single copy"] = src["single"]
    for name, fn in fns.items():
        ps = perturbed_value_sources(t, obs_p, placed, fn)
        rows[f"[{eps_mm / 10:g}cm] {name} (mean of {copies})"] = ps["ceiling"]
        rows[f"[{eps_mm / 10:g}cm] {name} (single)"] = ps["single"]
    result = {name: metrics_by_left(t, left, v) for name, v in rows.items()}
    cols = ("pair_acc", "pick", "best", "greedy", "random")
    for g in result[next(iter(result))]:
        print(f"\n== origin balls-left: {g}")
        for name, byl in result.items():
            print(f"  {name[:50]:50s} " + "  ".join(f"{k} {byl[g][k]:.3f}" for k in cols))
    key = next(k for k in result if "ceiling" in k)
    print(f"\nd (balls-left 2·3 minus 5·6, pair_acc) of the ceiling: {pooled_diff(result[key]):+.3f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"testset": args.testset, "ceiling": args.ceiling, "v_runs": args.v_runs, "result": result}, f, indent=1)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
