"""
exp16_wm/cue_goal_head.py — CT2a: a head that sends the cue ball to a goal position (experiments.md "CT2a").

Hindsight: every recorded shot (exp16_wm.cue_goal_data) is an example for the goal "where it ended". The MLP maps (start, goal) to
(cos angle, sin angle, log speed). Trained on the shots with one cushion count (default k = 0, the straight shots) from most of the start
positions; evaluated by playing its action in the simulator from start positions it never saw, and from ones it trained on.

The loss divides each error by its tolerance (0.1 deg, 0.5 % of speed) so both terms are O(1) at the precision that is needed.

Usage:
    python -m exp16_wm.cue_goal_head train --data logs/ct0b/collect_200.npz --out logs/ct2a/run0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import BALL_R, TABLE_L, TABLE_W
from exp16_wm.single_ball import make_cue_only_system, sample_starts, shoot

ANGLE_TOL_RAD = float(np.radians(0.1))
LOGV_TOL = 0.005
V_CLIP = (1e-4, 8.0)
SCALE = np.array([TABLE_W, TABLE_L], np.float32)


def features(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """(n, 7): start and goal as table fractions, the vector start -> goal in metres, and its length."""
    d = goal - start
    return np.concatenate([start / SCALE, goal / SCALE, d, np.linalg.norm(d, axis=1, keepdims=True)], axis=1).astype(np.float32)


def targets(phi_deg: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(n, 3): cos angle, sin angle, log speed."""
    r = np.radians(phi_deg.astype(np.float64))
    return np.stack([np.cos(r), np.sin(r), np.log(v.astype(np.float64))], axis=1).astype(np.float32)


def load_rows(path: str, k_bin: int, n_holdout: int, n_train: int, seed: int) -> dict:
    """
    Shots with cushion count k_bin (3 means 3 or more) that end on the table and are not scratches.
    Splits the start positions: n_holdout of them are kept out of training. Returns train rows (a random subsample of n_train, from the
    other starts), all rows grouped by start for the lookup baseline, and the start lists.
    """
    z = np.load(path)
    starts, sidx = z["starts"].astype(np.float32), z["start_idx"]
    end, k, scratch = z["end"], z["k"], z["scratch"]
    on_table = (end[:, 0] >= 0) & (end[:, 0] <= TABLE_W) & (end[:, 1] >= 0) & (end[:, 1] <= TABLE_L)
    kk = np.minimum(k, 3)
    ok = on_table & ~scratch & (kk == k_bin)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(starts))
    hold, train_starts = np.sort(perm[:n_holdout]), np.sort(perm[n_holdout:])
    in_train = np.isin(sidx, train_starts) & ok
    rows = np.nonzero(in_train)[0]
    rows = rng.choice(rows, size=min(n_train, len(rows)), replace=False)
    return {"starts": starts, "holdout": hold, "train_starts": train_starts, "rows": rows, "sidx": sidx, "ok": ok,
            "phi": z["phi"], "v": z["v"], "end": end}


class Head(nn.Module):
    def __init__(self, width: int = 256, depth: int = 5):
        super().__init__()
        layers, d = [], 7
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]; d = width
        layers.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*layers)
        self.register_buffer("x_mean", torch.zeros(7)); self.register_buffer("x_std", torch.ones(7))
        self.register_buffer("lv_mean", torch.zeros(())); self.register_buffer("lv_std", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net((x - self.x_mean) / self.x_std)
        return torch.cat([out[:, :2], (out[:, 2:] * self.lv_std + self.lv_mean)], dim=1)


def pseudo_huber(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1.0 + x * x) - 1.0


def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Mean over rows of angle error (as a vector distance, / 0.1 deg) and log-speed error (/ 0.005), each through a pseudo-Huber."""
    ang = torch.linalg.norm(pred[:, :2] - tgt[:, :2], dim=1) / ANGLE_TOL_RAD
    lv = (pred[:, 2] - tgt[:, 2]) / LOGV_TOL
    return (pseudo_huber(ang) + pseudo_huber(lv)).mean()


def predict_action(model: Head, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(angle in degrees 0..360, speed m/s) for each (start, goal)."""
    with torch.no_grad():
        out = model(torch.from_numpy(features(start, goal))).numpy().astype(np.float64)
    phi = np.degrees(np.arctan2(out[:, 1], out[:, 0])) % 360.0
    return phi, np.clip(np.exp(out[:, 2]), *V_CLIP)


def play_error(start: np.ndarray, goal: np.ndarray, phi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Play each (phi, v) from its start in the simulator; distance (m) from where the cue ball stops to the goal (inf if it fell in a pocket)."""
    err = np.zeros(len(start)); systems: dict[tuple[float, float], object] = {}
    for i in range(len(start)):
        key = (float(start[i, 0]), float(start[i, 1]))
        if key not in systems:
            systems[key] = make_cue_only_system(*key)
        r = shoot(systems[key], float(phi[i]), float(v[i]))
        err[i] = np.inf if r.scratch else float(np.hypot(r.x - goal[i, 0], r.y - goal[i, 1]))
    return err


def eval_cases(starts: np.ndarray, start_ids: np.ndarray, n_goals: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(start, goal) pairs: n_goals random goals on the table for every start in start_ids."""
    s, g = [], []
    for j, i in enumerate(start_ids):
        goals = sample_starts(n_goals, seed + 7919 * (j + 1)).astype(np.float32)
        s.append(np.repeat(starts[i][None], n_goals, axis=0)); g.append(goals)
    return np.concatenate(s), np.concatenate(g)


def lookup_action(data: dict, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: from the nearest training start, the recorded shot whose end is nearest the goal; its action is replayed from the new start."""
    from scipy.spatial import cKDTree
    ts = data["train_starts"]
    near = ts[np.argmin(np.linalg.norm(data["starts"][ts][None] - start[:, None], axis=2), axis=1)]
    phi, v = np.zeros(len(start)), np.zeros(len(start)); trees: dict[int, tuple] = {}
    for i in range(len(start)):
        s = int(near[i])
        if s not in trees:
            rows = np.nonzero((data["sidx"] == s) & data["ok"])[0]
            trees[s] = (cKDTree(data["end"][rows]), rows)
        tree, rows = trees[s]
        _, j = tree.query(goal[i]); r = rows[j]
        phi[i], v[i] = data["phi"][r], data["v"][r]
    return phi, v


def summarize(err: np.ndarray) -> dict:
    e = err * 100.0
    return {"n": int(len(e)), "median_cm": float(np.median(e)), "p90_cm": float(np.percentile(e, 90)),
            **{f"within_{t}cm": float(np.mean(e <= t)) for t in (1, 2, 5, 10)}}


def train(data: dict, epochs: int, batch: int, lr: float, width: int, depth: int, seed: int, log=print) -> Head:
    torch.manual_seed(seed)
    rows = data["rows"]
    s = data["starts"][data["sidx"][rows]]; g = data["end"][rows].astype(np.float32)
    x = torch.from_numpy(features(s, g)); y = torch.from_numpy(targets(data["phi"][rows], data["v"][rows]))
    model = Head(width, depth)
    model.x_mean.copy_(x.mean(0)); model.x_std.copy_(x.std(0) + 1e-6)
    model.lv_mean.copy_(y[:, 2].mean()); model.lv_std.copy_(y[:, 2].std())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * ((len(x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(x)); total = 0.0
        for b in range(0, len(x), batch):
            idx = perm[b:b + batch]
            loss = loss_fn(model(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += float(loss.detach()) * len(idx)
        log(f"[ct2a] epoch {ep + 1}/{epochs} train loss {total / len(x):.4f}  {time.time() - t0:.0f}s")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--k-bin", type=int, default=0); t.add_argument("--holdout", type=int, default=40)
    t.add_argument("--n-train", type=int, default=1_500_000); t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=4096); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--width", type=int, default=256); t.add_argument("--depth", type=int, default=5)
    t.add_argument("--n-goals", type=int, default=100); t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from exp16_wm.code_version import code_version
    os.makedirs(args.out, exist_ok=True)
    data = load_rows(args.data, args.k_bin, args.holdout, args.n_train, args.seed)
    print(f"[ct2a] train rows {len(data['rows'])} from {len(data['train_starts'])} starts, holdout {len(data['holdout'])} starts", flush=True)
    model = train(data, args.epochs, args.batch, args.lr, args.width, args.depth, args.seed, log=lambda m: print(m, flush=True))
    torch.save({"state": model.state_dict(), "width": args.width, "depth": args.depth, "args": vars(args)}, os.path.join(args.out, "head.pt"))
    result = {"commit": code_version(), "args": vars(args), "k_bin": args.k_bin}
    for name, ids in (("new_starts", data["holdout"]), ("train_starts", data["train_starts"][: len(data["holdout"])])):
        s, g = eval_cases(data["starts"], ids, args.n_goals, 1000 + args.seed)
        phi, v = predict_action(model, s, g)
        lphi, lv = lookup_action(data, s, g)
        result[name] = {"model": summarize(play_error(s, g, phi, v)), "lookup_baseline": summarize(play_error(s, g, lphi, lv))}
        print(name, json.dumps(result[name]), flush=True)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
