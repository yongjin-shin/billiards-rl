"""
exp16_wm/aim_sims.py — teacher-free training simulations for the made2 head (Exp-16, roadmap T1).

For every state of the teacher's label files (only the states are used, none of the teacher's actions, pairs or simulations)
and every live pair, sample k of the 217 refine candidates (31 angles ±0.3° × 7 absolute speeds) around the learned aim
(circular mean of frozen aux-only aim encoders) and simulate them: made for all 18 pairs and scratch. The output mirrors the
teacher sims (one file per label shard, same basename, keys sim_state / sim_action / sim_made / sim_scratch_replay) so
`python -m exp16_wm.bc --sims <out>/*.npz` reads it unchanged.

Usage:
    python -m exp16_wm.aim_sims --seed 0 --aim-runs "logs/experiments_aim_ens2000/*_i1_*_s{seed}_*" ... --out logs/teacher_data/relabel_s1000/aimsims_s0
"""

import argparse
import glob
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


def grid_units(aim_u: np.ndarray, act_scale: np.ndarray, act_bias: np.ndarray, angles_deg=ANG31, speeds=ABS7) -> np.ndarray:
    """aim_u (…,) aims as unit angles → (…, len(angles) · len(speeds), 2) unit actions: angle aim ± angles_deg, absolute speeds."""
    ang = aim_u[..., None] + np.asarray(angles_deg) / 180.0                                    # unit angle: 1 = π rad
    spd = (np.asarray(speeds) - act_bias[1]) / act_scale[1]
    a = np.broadcast_to(ang[..., :, None], ang.shape + (len(speeds),))
    v = np.broadcast_to(spd, a.shape)
    return np.clip(np.stack([a, v], axis=-1).reshape(aim_u.shape + (-1, 2)), -1.0, 1.0)


def sample_pair_candidates(valid: np.ndarray, k: int, n_cand: int, rng: np.random.Generator) -> np.ndarray:
    """valid (N, 18) live pairs → (M, 3) rows (state, pair, candidate index): k distinct candidates of every live pair."""
    out = []
    for s, j in zip(*np.nonzero(valid)):
        for c in rng.choice(n_cand, size=min(k, n_cand), replace=False):
            out.append((s, j, c))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


@torch.no_grad()
def ensemble_aim(encoders: list[PairGNNPolicy], rel: torch.Tensor, mean: float, std: float, chunk: int = 1000) -> np.ndarray:
    """(N, 18) aims (rad, relative frame): circular mean of the encoders' predictions (target standardised with mean / std degrees)."""
    out = []
    for lo in range(0, len(rel), chunk):
        x = rel[lo:lo + chunk]
        ang = torch.stack([torch.deg2rad(e.aux_head(e._encode(x)[0])[..., 0] * std + mean) for e in encoders])
        out.append(torch.atan2(torch.sin(ang).mean(0), torch.cos(ang).mean(0)))
    return torch.cat(out).numpy()


def gen(args) -> None:
    from exp16_wm.bc import build_split, load_aim_encoder, load_teacher_data
    from exp16_wm.frozen_verifier import _init_worker, _simulate_full
    from exp16_wm.geometry_aux import aim_targets, standardize
    from exp16_wm.obs_transform import relative_obs
    from exp16_wm.sim_identity import to_bitmask
    paths = sorted(glob.glob(args.data))
    dirs = []
    for pat in args.aim_runs:
        found = glob.glob(pat.format(seed=args.seed))
        if len(found) != 1:
            raise SystemExit(f"{pat}: expected one run dir, found {found}")
        dirs.append(found[0])
    encs = [load_aim_encoder(d) for d in dirs]
    data, train, _ = build_split(dict(load_teacher_data(paths)), args.val_frac, args.seed, 1.0, "none")   # the aim encoders' standardisation
    _, mean, std = standardize(*aim_targets(relative_obs(data["obs"])), train)
    ref = PairGNNPolicy(hidden=8)
    scale, bias = ref.act_scale.numpy().astype(np.float64), ref.act_bias.numpy().astype(np.float64)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(1000 + args.seed)
    t0, total = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for path in paths:
            z = np.load(path)
            obs = z["obs"][:args.max_states] if args.max_states else z["obs"]
            rel = relative_obs(obs)
            aim = ensemble_aim(encs, torch.as_tensor(rel), float(mean[0]), float(std[0]))
            d = aim - bias[0]
            aim_u = np.arctan2(np.sin(d), np.cos(d)) / scale[0]
            grid = grid_units(aim_u, scale, bias)                                             # (N, 18, 217, 2)
            pick = sample_pair_candidates(aim_targets(rel)[1], args.k, grid.shape[2], rng)
            a_unit = grid[pick[:, 0], pick[:, 1], pick[:, 2]]
            env_a = (a_unit * scale + bias).astype(np.float32)
            tasks, order = [], []
            for s in np.unique(pick[:, 0]):
                rows = np.flatnonzero(pick[:, 0] == s)
                tasks.append((obs[s], env_a[rows])); order.append(rows)
            made, scr = np.zeros((len(pick), N_PAIRS), bool), np.zeros(len(pick), bool)
            for rows, (mm, ss) in zip(order, pool.imap(_simulate_full, tasks, chunksize=8)):
                made[rows], scr[rows] = mm, ss
            out = os.path.join(args.out, os.path.basename(path))
            np.savez(out, meta=z["meta"], obs=z["obs"], game=z["game"], sim_state=pick[:, 0].astype(np.int32), sim_action=env_a,
                     sim_made=to_bitmask(made).astype(np.int32), sim_scratch_replay=scr, sim_pair=pick[:, 1].astype(np.int8),
                     sim_aim_runs=np.array(dirs), seed=args.seed)
            total += len(pick)
            print(f"  {os.path.basename(path)}: {len(obs)} states, {len(pick)} simulations "
                  f"(target pair made & no scratch {np.mean(made[np.arange(len(pick)), pick[:, 1]] & ~scr):.1%}), {time.time() - t0:.0f}s", flush=True)
    print(f"saved {args.out}: {total} simulations ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="logs/teacher_data/relabel_s1000/robust/*.npz", help="label files (only their states are used)")
    p.add_argument("--aim-runs", nargs="+", required=True, help="aux-only aim run dir globs with {seed}")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--k", type=int, default=8, help="candidates per live pair")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--max-states", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    gen(args)


if __name__ == "__main__":
    main()
