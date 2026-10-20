"""
exp16_wm/val_set.py — fixed validation set V0 for model selection and made-head metrics (Exp-16, roadmap "고정 val 세트 V0").

Why: the final model used to be the epoch with the lowest made BCE on 10% of each experiment's own training data — dominated by easy
negatives (13–15% successes, class imbalance), not the decision the policy makes (the top candidate), and a different val set for
every data source. V0 is one fixed file for every experiment:
  states  — games of a policy with an eval seed no training data uses (val_v0_states, eval seed 900);
  windows — formula-free scan centres (exp16_wm.wide_scan, robust 1·2 m/s);
  grid    — for every live pair with a window (robust, else any path), all 217 refine candidates (31 angles ±0.3° × 7 absolute speeds)
            around the centre, simulated once: target pair pocketed cleanly or not.

Metrics:
  top1      — the policy's own choice (any inference: single model or ensemble) played on every V0 state: share that pockets a ball
              cleanly. The model-selection criterion (the decision the policy really makes).
  group_top1 — per (state, pair) grid with both outcomes: the made head's highest-scored candidate succeeds (ranking).
  ece       — calibration of the made head's probabilities on all grid candidates (10 equal-width bins).
  ceiling   — share of V0 states with at least one successful grid candidate.

Usage:
    python -m exp16_wm.val_set build --states "logs/teacher_data/val_v0_states/*.npz" --scan logs/teacher_data/val_v0_scan --out logs/teacher_data/val_v0.npz
    python -m exp16_wm.val_set score --v0 logs/teacher_data/val_v0.npz --runs "<run dir glob>" ...
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import N_PAIRS, PairGNNPolicy

ANG31 = tuple(np.round(np.arange(-0.3, 0.3 + 1e-9, 0.02), 4))
ABS7 = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)


def grid_actions(centre_deg: float, act_scale: np.ndarray, act_bias: np.ndarray) -> np.ndarray:
    """217 env actions (angle rad, speed m/s): centre ± ANG31 degrees × ABS7 speeds, through the unit range like the policy."""
    from exp16_wm.wide_scan import to_env_actions
    ang = np.repeat(np.asarray(ANG31) + centre_deg, len(ABS7))
    acts = to_env_actions(ang, 1.0, act_scale, act_bias)
    acts[:, 1] = np.tile(ABS7, len(ANG31))
    return acts


def _sim_task(task: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    from exp16_wm.frozen_verifier import _simulate_full
    obs, acts, j = task
    made, scr = _simulate_full((obs, acts))
    return made[:, j] & ~scr


def build(states: str, scan_dir: str, out: str, workers: int) -> None:
    from exp16_wm.frozen_verifier import _init_worker
    ref = PairGNNPolicy(hidden=8)
    sc, bi = ref.act_scale.numpy().astype(np.float64), ref.act_bias.numpy().astype(np.float64)
    obs_l, src_l, cen_l = [], [], []
    for p in sorted(glob.glob(states)):
        z, w = np.load(p), np.load(os.path.join(scan_dir, os.path.basename(p)))
        obs_l.append(z["obs"])
        rob, ok, anyc = w["scan_center_deg"], w["scan_valid"], w["scan_any_center_deg"]
        cen_l.append(np.where(ok, rob, anyc)); src_l.append(np.where(ok, 2, np.where(np.isfinite(anyc), 1, 0)).astype(np.int8))
    obs, centre, source = np.concatenate(obs_l), np.concatenate(cen_l), np.concatenate(src_l)
    tasks, rows = [], []
    for s, j in zip(*np.nonzero(source > 0)):
        tasks.append((obs[s], grid_actions(float(centre[s, j]), sc, bi), int(j))); rows.append((s, j))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        res = pool.map(_sim_task, tasks, chunksize=4)
    rows = np.array(rows, np.int32).reshape(-1, 2)
    acts = np.stack([t[1] for t in tasks]) if tasks else np.zeros((0, 217, 2), np.float32)
    np.savez(out, obs=obs, centre_deg=centre, source=source, grid_state=rows[:, 0], grid_pair=rows[:, 1], grid_action=acts,
             grid_ok=np.stack(res) if res else np.zeros((0, 217), bool), meta=json.dumps({"states": states, "scan": scan_dir}))
    print(f"saved {out}: {len(obs)} states, {len(rows)} pair grids, {len(rows) * 217} simulations ({time.time() - t0:.0f}s)", flush=True)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins (weighted by bin size)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


@torch.no_grad()
def grid_scores(model: PairGNNPolicy, v0: dict[str, np.ndarray], chunk: int = 256) -> np.ndarray:
    """(G, 217) made2 logits of every grid candidate for its own pair (the model's own aim / proposal as at inference)."""
    from exp16_wm.obs_transform import relative_obs
    rel = torch.as_tensor(relative_obs(v0["obs"]))
    sc, bi = model.act_scale, model.act_bias
    out = np.zeros(v0["grid_ok"].shape, np.float32)
    for lo in range(0, len(v0["grid_state"]), chunk):
        s, j = v0["grid_state"][lo:lo + chunk], v0["grid_pair"][lo:lo + chunk]
        o = model.forward_full(rel[s])
        a = (torch.as_tensor(v0["grid_action"][lo:lo + chunk]) - bi) / sc            # (g, 217, 2) unit
        r = torch.arange(len(s)); C = a.shape[1]
        jt = torch.as_tensor(j, dtype=torch.long)
        z = model.head_made2(model._made2_features(o["e"][r, jt][:, None].expand(-1, C, -1), a, o["means"][r, jt][:, None].expand(-1, C, -1),
                                                   o["aim"][r, jt][:, None].expand(-1, C))).squeeze(-1)
        out[lo:lo + chunk] = z.numpy()
    return out


def made_metrics(scores: np.ndarray, ok: np.ndarray) -> dict[str, float]:
    """group_top1 over (state, pair) grids with both outcomes, ece over all candidates."""
    both = ok.any(1) & ~ok.all(1)
    top = scores.argmax(1)
    return {"group_top1": float(ok[np.arange(len(ok)), top][both].mean()) if both.any() else float("nan"), "n_groups": int(both.sum()),
            "ece": ece(1.0 / (1.0 + np.exp(-scores.ravel())), ok.ravel())}


def top1(act_fn, v0: dict[str, np.ndarray], workers: int = 8) -> float:
    """act_fn(raw obs (N, 23)) → env actions (N, 2); share of V0 states where that action pockets a ball cleanly."""
    from exp16_wm.frozen_verifier import _init_worker, _simulate_state
    acts = act_fn(v0["obs"])
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        res = pool.map(_simulate_state, [(o, a[None]) for o, a in zip(v0["obs"], acts)], chunksize=16)
    return float(np.mean([r[0] for r in res]))


@torch.no_grad()
def env_actions(policy, obs: np.ndarray, chunk: int = 256) -> np.ndarray:
    """The policy's own choice (predict_unit: model or MadeEnsemble) for raw observations (N, 23) → env actions (N, 2)."""
    from exp16_wm.obs_transform import relative_obs
    rel = torch.as_tensor(relative_obs(obs))
    return np.concatenate([(policy.predict_unit(rel[lo:lo + chunk]) * policy.act_scale + policy.act_bias).numpy()
                           for lo in range(0, len(rel), chunk)])


def top1_serial(policy, v0: dict[str, np.ndarray]) -> float:
    """top1 in this process (no worker pool) — for model selection inside a training run that shares the machine with others."""
    import exp16_wm.frozen_verifier as F
    if getattr(F, "_ENV", None) is None:
        F._init_worker()
    acts = env_actions(policy, v0["obs"])
    return float(np.mean([F._simulate_state((o, a[None]))[0] for o, a in zip(v0["obs"], acts)]))


def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "meta"}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--states", required=True); b.add_argument("--scan", required=True)
    b.add_argument("--out", required=True); b.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score"); s.add_argument("--v0", required=True); s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--workers", type=int, default=8); s.add_argument("--ensemble", choices=["", "shared", "mean"], default="")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.states, args.scan, args.out, args.workers)
        return
    from exp16_wm import eval_by_balls as E
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.obs_transform import relative_obs
    from exp16_wm.paired_eval import find_run
    torch.set_num_threads(4)
    v0 = load(args.v0)
    print(f"V0: {len(v0['obs'])} states, ceiling {np.mean(np.isin(np.arange(len(v0['obs'])), v0['grid_state'][v0['grid_ok'].any(1)])):.1%}")
    runs = [find_run(g) for g in args.runs]
    if args.ensemble:
        ens = load_ensemble(runs, args.ensemble)
        act = lambda obs: env_actions(ens, obs)
        print(f"ensemble of {len(runs)} ({args.ensemble}): top1 {top1(act, v0, args.workers):.1%}", flush=True)
        return
    for d in runs:
        m = E.load_run(d)
        m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = "all_pairs", E.ANG31, (1.0,), E.ABS_SPEEDS
        act = lambda obs, m=m: env_actions(m, obs)
        mm = made_metrics(grid_scores(m, v0), v0["grid_ok"])
        print(f"{os.path.basename(d)[-60:]}: top1 {top1(act, v0, args.workers):.1%}  group_top1 {mm['group_top1']:.1%} ({mm['n_groups']})  "
              f"ece {mm['ece']:.3f}", flush=True)


if __name__ == "__main__":
    main()
