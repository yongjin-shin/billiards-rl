"""
exp16_wm/v_perturb.py — how predictable is a V label? Move the cue ball by ε and play the position out again (Exp-16, V4-0).

The simulator and the greedy policy are deterministic, so a position's V label k (balls pocketed before the first miss) is exact for
that position. But if moving the cue ball a few mm changes k, the label is a coin flip at the scale any predictor (or a cue-ball
landing model) can resolve, and no V can do better than the ε-smoothed value. This measures that:
  repeat  — Spearman (within the same number of balls left) of one moved copy's k with the original k;
  ceiling — the same with the mean k of n moved copies: roughly the best a predictor that is smooth at scale ε can do;
  same    — fraction of moved copies whose k equals the original k.
Object balls stay where they are; the cue ball moves in a random direction, staying on the table and off the other balls.

Usage:
    python -m exp16_wm.v_perturb --runs "<run glob>" --data logs/teacher_data/v2/diag_s950.npz --eps-mm 2 10 --copies 8 \
        --workers 8 --out logs/teacher_data/v4/perturb_s950.npz
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def perturb_cue(raw: np.ndarray, eps_m: float, rng: np.random.Generator, table_w: float, table_l: float, ball_r: float,
                tries: int = 50) -> np.ndarray | None:
    """Copy of raw (normalised coordinates) with the cue ball moved eps_m metres in a random direction, on the table (centre at least
    one radius from the cushion) and at least two radii from every ball still on the table; None if no direction works."""
    from exp16_wm.obs_transform import n_balls_raw
    n = n_balls_raw(len(raw))
    cue = np.array([raw[0] * table_w, raw[1] * table_l])
    others = np.array([[raw[2 + 3 * i] * table_w, raw[3 + 3 * i] * table_l] for i in range(n) if raw[4 + 3 * i] < 0.5]).reshape(-1, 2)
    for _ in range(tries):
        t = rng.uniform(0.0, 2.0 * np.pi)
        p = cue + eps_m * np.array([np.cos(t), np.sin(t)])
        if not (ball_r <= p[0] <= table_w - ball_r and ball_r <= p[1] <= table_l - ball_r):
            continue
        if len(others) and np.min(np.linalg.norm(others - p, axis=1)) < 2.0 * ball_r + 1e-4:
            continue
        out = np.array(raw, np.float32, copy=True)
        out[0], out[1] = p[0] / table_w, p[1] / table_l
        return out
    return None


def within_spearman(pred: np.ndarray, target: np.ndarray, left: np.ndarray, min_n: int = 10) -> float:
    """Spearman of pred with target inside each number of balls left, weighted by group size (NaN rows dropped)."""
    from scipy.stats import spearmanr
    s, w = 0.0, 0
    for n in np.unique(left):
        g = (left == n) & np.isfinite(pred) & np.isfinite(target)
        if g.sum() > min_n and np.std(pred[g]) > 0 and np.std(target[g]) > 0:
            s += spearmanr(pred[g], target[g]).correlation * g.sum(); w += g.sum()
    return s / w if w else float("nan")


def perturb_metrics(k: np.ndarray, left: np.ndarray, k_pert: np.ndarray, horizon: int = 0) -> dict[str, float]:
    """k (N,), left (N,), k_pert (N, C) with −1 for copies that could not be placed; labels capped at horizon (0 = no cap)."""
    c = np.minimum(left, horizon) if horizon > 0 else left
    t = np.minimum(k, c).astype(float)
    kp = np.where(k_pert >= 0, np.minimum(k_pert, c[:, None]), np.nan).astype(float)
    have = np.isfinite(kp)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)            # rows with no placed copy → NaN mean, dropped by within_spearman
        mean = np.nanmean(kp, axis=1)
    rep = [within_spearman(kp[:, j], t, left) for j in range(kp.shape[1])]
    return {"repeat": float(np.nanmean(rep)), "ceiling": within_spearman(mean, t, left),
            "same": float(np.mean(kp[have] == np.broadcast_to(t[:, None], kp.shape)[have])) if have.any() else float("nan"),
            "placed": float(have.mean())}


def testset_order_metrics(t: dict[str, np.ndarray], k_pert: np.ndarray) -> dict[str, float]:
    """V5-c: k_pert (M, C) for the test set's rankable next positions (valid & succ & ~cleared, row-major order): same-origin pair
    accuracy against the unmoved k_next of the mean over copies (ceiling) and of single copies (repeat, averaged)."""
    from exp16_wm.v_testset import pair_accuracy
    use = t["valid"] & t["succ"] & ~t["cleared"]
    kp = np.where(k_pert >= 0, k_pert, np.nan).astype(float)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(kp, axis=1)

    def grid(v: np.ndarray) -> np.ndarray:
        g = np.full(use.shape, np.nan); g[use] = v
        return g
    ok = use & np.isfinite(grid(mean))
    rep = [pair_accuracy(np.nan_to_num(grid(kp[:, j])), t["k_next"], use & np.isfinite(grid(kp[:, j]))) for j in range(kp.shape[1])]
    return {"order_ceiling": pair_accuracy(np.nan_to_num(grid(mean)), t["k_next"], ok), "order_repeat": float(np.mean(rep)),
            "placed": float(np.isfinite(kp).mean())}


def _worker(job: tuple) -> np.ndarray:
    rows, raws, lefts, eps_list, copies, runs, n_balls, seed = job
    import torch
    torch.set_num_threads(1)
    from exp16_wm.bc import TransformedPolicy
    from exp16_wm.lookahead import make_sim
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.v_data import rollout
    pol = TransformedPolicy(load_ensemble(runs, "mean"), "relative")
    act = lambda o: pol.act(o, deterministic=True)
    sim = make_sim(n_balls)
    import pooltool as pt
    ball_r = float(pt.BallParams.default().R)
    out = np.full((len(rows), len(eps_list), copies), -1, np.int64)
    for i, (r, raw, left) in enumerate(zip(rows, raws, lefts)):
        rng = np.random.default_rng([seed, int(r)])
        for e, eps in enumerate(eps_list):
            for c in range(copies):
                moved = perturb_cue(raw, eps, rng, TABLE_W, TABLE_L, ball_r)
                if moved is not None:
                    out[i, e, c] = rollout(sim, act, moved, int(left))
    return out


def main() -> None:
    from multiprocessing import Pool
    from exp16_wm.code_version import code_version
    from exp16_wm.paired_eval import find_run
    from exp16_wm.obs_transform import n_balls_raw
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--data", default="")
    p.add_argument("--testset", default="", help="V5-c: perturb the next positions of this v_testset file instead of --data")
    p.add_argument("--eps-mm", type=float, nargs="+", default=[2.0, 10.0]); p.add_argument("--copies", type=int, default=8)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N positions only (smoke test)"); p.add_argument("--out", required=True)
    args = p.parse_args()
    t = None
    if args.testset:
        z = np.load(args.testset, allow_pickle=True)
        t = {k: z[k] for k in ("made", "succ", "balls", "k_next", "cleared", "valid", "next_obs", "left_next")}
        use = t["valid"] & t["succ"] & ~t["cleared"]
        if args.limit:
            use[args.limit:] = False
            for k in t:
                t[k] = t[k][:args.limit]
            use = use[:args.limit]
        obs, k, left = t["next_obs"][use], t["k_next"][use], t["left_next"][use]
        ok = np.nonzero(use.reshape(-1))[0]
    else:
        z = np.load(args.data, allow_pickle=True)
        ok = np.nonzero(z["left"] > 0)[0]
        if args.limit:
            ok = ok[:args.limit]
        obs, k, left = z["obs"][ok], z["k"][ok], z["left"][ok]
    runs, n_balls, eps = [find_run(g) for g in args.runs], n_balls_raw(obs.shape[1]), [e / 1000.0 for e in args.eps_mm]
    chunks = np.array_split(np.arange(len(ok)), max(1, args.workers * 4))
    jobs = [(ok[c], obs[c], left[c], eps, args.copies, runs, n_balls, args.seed) for c in chunks if len(c)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        k_pert = np.concatenate(pool.map(_worker, jobs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, rows=ok, k=k, left=left, k_pert=k_pert, eps_mm=np.array(args.eps_mm),
             meta=str({**vars(args), "git_commit": code_version(), "seconds": round(time.time() - t0, 1)}))
    print(f"saved {args.out}: {len(ok)} positions × {len(eps)} ε × {args.copies} copies ({time.time() - t0:.0f}s)", flush=True)
    for e, mm in enumerate(args.eps_mm):
        if t is not None:
            m = testset_order_metrics(t, k_pert[:, e])
            print(f"  ε {mm:g} mm, same-origin order: " + "  ".join(f"{n} {v:.3f}" for n, v in m.items()), flush=True)
        for h in (0, 2):
            m = perturb_metrics(k, left, k_pert[:, e], h)
            print(f"  ε {mm:g} mm, target min(k,{h or '∞'}): " + "  ".join(f"{n} {v:.3f}" for n, v in m.items()), flush=True)


if __name__ == "__main__":
    main()
