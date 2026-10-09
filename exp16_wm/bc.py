"""
exp16_wm/bc.py — behaviour cloning of a search-planner teacher (Exp-16 imitation).

Trains the same TanhGaussian actor SAC uses (so the result can seed SAC later)
on data from exp16_wm/teacher_data.py, then evaluates the policy alone — no
search, no simulator at decision time — on the clear-speed task.

Losses (on the deterministic action tanh(mean); per-dim errors are rescaled to
physical precision units — 1 = angle_scale_deg degrees / speed_scale m/s — so the
needle-narrow angle is not drowned out by speed, which in raw [-1, 1] units is
~50× larger per physical unit):
  teacher       MSE to the teacher's chosen action
  nearest_good  MSE to the nearest stored pocketing candidate (set loss: any
                pocketing shot is a right answer — guards against averaging
                two good aims into a miss); teacher action if none pocketed

Usage:
    python -m exp16_wm.bc --data logs/teacher_data/ghost_s100*.npz --loss teacher
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.sac import VanillaSAC
from exp16_wm.train import ACTION_DIM, ACT_LOW, ACT_HIGH, get_obs_dim, summarize_final_eval
from exp16_wm.planner import wrap_angle
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.obs_transform import TRANSFORMS, OBS_DIMS
from exp16_wm.augment import augment_flip4
from exp16_wm.gnn_policy import PairGNNPolicy

LOSSES = ("teacher", "nearest_good")


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

def to_unit(action: np.ndarray) -> np.ndarray:
    """Env action → [-1, 1] per dim (the actor's post-tanh space)."""
    return (action - (ACT_HIGH + ACT_LOW) / 2.0) / ((ACT_HIGH - ACT_LOW) / 2.0)


def load_teacher_data(paths: list[str]) -> dict[str, np.ndarray]:
    """Concatenate teacher_data files; game ids are made unique across files."""
    parts = [dict(np.load(p)) for p in sorted(paths)]
    if not parts:
        raise FileNotFoundError("no teacher data files")
    out = {k: np.concatenate([d[k] for d in parts]) for k in
           ["obs", "action", "good_actions", "n_good", "pocketed", "shot"]}
    offset, games = 0, []
    for d in parts:
        games.append(d["game"].astype(np.int64) + offset)
        offset += int(d["game"].max()) + 1 if len(d["game"]) else 0
    out["game"] = np.concatenate(games)
    return out


def split_by_game(game: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/val masks; whole games go to one side so shots of a game don't leak."""
    ids = np.unique(game)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 1 else 0
    val_ids = rng.choice(ids, size=n_val, replace=False) if n_val else np.array([], dtype=ids.dtype)
    val = np.isin(game, val_ids)
    return ~val, val


_KEYS = ["obs", "action", "good_actions", "n_good", "pocketed", "shot", "game"]


def build_split(data: dict[str, np.ndarray], val_frac: float, seed: int, data_frac: float = 1.0,
                augment: str = "none") -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Game-level train/val split, then (train only) subsample a fraction of train games and
    optionally augment. Returns (combined data, train mask, val mask) — val is never augmented,
    so no reflected copy of a val game can leak into train.
    """
    train, val = split_by_game(data["game"], val_frac, seed)
    tr_games = np.unique(data["game"][train])
    if data_frac < 1.0:
        rng = np.random.default_rng(seed + 1)
        keep = rng.choice(tr_games, size=max(1, int(round(len(tr_games) * data_frac))), replace=False)
        train = train & np.isin(data["game"], keep)
    tr_idx, va_idx = np.where(train)[0], np.where(val)[0]
    if augment == "flip4":
        tr_part = augment_flip4(data, tr_idx)
    elif augment == "none":
        tr_part = {k: data[k][tr_idx] for k in _KEYS}
    else:
        raise ValueError(f"unknown augment: {augment}")
    va_part = {k: data[k][va_idx] for k in _KEYS}
    combined = {k: np.concatenate([tr_part[k], va_part[k]]) for k in _KEYS}
    n_tr = len(tr_part["obs"])
    tr_mask = np.arange(len(combined["obs"])) < n_tr
    return combined, tr_mask, ~tr_mask


# ──────────────────────────────────────────────
# Losses / metrics
# ──────────────────────────────────────────────

def dim_weights(angle_scale_deg: float, speed_scale: float) -> torch.Tensor:
    """Per-dim multipliers turning [-1, 1]-unit differences into precision units."""
    half_range = (ACT_HIGH - ACT_LOW) / 2.0                    # [π rad, 3.75 m/s]
    return torch.tensor([np.degrees(half_range[0]) / angle_scale_deg,
                         half_range[1] / speed_scale], dtype=torch.float32)


def bc_loss(pred_unit: torch.Tensor, teacher_unit: torch.Tensor, good_unit: torch.Tensor,
            loss: str, w: torch.Tensor | None = None) -> torch.Tensor:
    """
    pred_unit (B,2), teacher_unit (B,2), good_unit (B,K,2) NaN-padded, w (2,) per-dim
    weights (None → 1). Returns the mean per-sample weighted squared error.
    """
    w = torch.ones(2) if w is None else w
    if loss == "teacher":
        return (((pred_unit - teacher_unit) * w) ** 2).sum(-1).mean()
    if loss == "nearest_good":
        valid = ~torch.isnan(good_unit[..., 0])                          # (B,K)
        d = (((pred_unit.unsqueeze(1) - torch.nan_to_num(good_unit)) * w) ** 2).sum(-1)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        nearest = d.min(dim=1).values
        fallback = (((pred_unit - teacher_unit) * w) ** 2).sum(-1)
        return torch.where(valid.any(dim=1), nearest, fallback).mean()
    raise ValueError(f"unknown loss: {loss}")


def within_rate(pred_unit: torch.Tensor, good_unit: torch.Tensor, thresh_deg: float) -> float:
    """
    Fraction of rows whose predicted delta_angle is within thresh_deg of some pocketing
    candidate (wrapped). Rows without any candidate are ignored; NaN if none have one.
    """
    valid = ~torch.isnan(good_unit[..., 0])
    if not bool(valid.any()):
        return float("nan")
    diff = (pred_unit[:, None, 0] - torch.nan_to_num(good_unit[..., 0])) * np.pi
    d = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
    d = torch.where(valid, d, torch.full_like(d, float("inf"))).min(dim=1).values
    rows = valid.any(dim=1)
    return float((d[rows] < np.radians(thresh_deg)).float().mean())


def angle_err_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|wrapped delta_angle difference| in degrees, per sample."""
    return np.degrees(np.abs([wrap_angle(float(p - t)) for p, t in zip(pred, target)]))


def nearest_good_angle_err_deg(pred: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Per sample: angle error (deg) to the nearest pocketing candidate; NaN if none."""
    out = np.full(len(pred), np.nan)
    for i, (p, g) in enumerate(zip(pred, good)):
        g = g[~np.isnan(g[:, 0])]
        if len(g):
            out[i] = np.degrees(min(abs(wrap_angle(float(p - a))) for a in g[:, 0]))
    return out


# ──────────────────────────────────────────────
# Train / evaluate
# ──────────────────────────────────────────────

def make_agent(n_balls: int = 3, hidden: list[int] | None = None,
               obs_dim: int | None = None) -> VanillaSAC:
    """hidden=None keeps VanillaSAC's default (same as SAC runs, so BC can seed SAC)."""
    kw = {} if hidden is None else {"hidden": hidden}
    return VanillaSAC(obs_dim=obs_dim or get_obs_dim(n_balls), action_dim=ACTION_DIM,
                      act_low=ACT_LOW, act_high=ACT_HIGH, **kw)


class TransformedPolicy:
    """Env-facing policy: applies the obs transform the actor was trained on."""

    def __init__(self, agent: VanillaSAC, transform: str):
        self.agent, self.fn = agent, TRANSFORMS[transform]

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.agent.act(self.fn(obs), deterministic=deterministic)


def deterministic_unit(agent: VanillaSAC, obs: torch.Tensor) -> torch.Tensor:
    """tanh(mean) — the deterministic action in [-1, 1] units, differentiable."""
    return torch.tanh(agent.actor._dist(obs).mean)


class MLPModel:
    """Adapter giving the SAC actor the interface PairGNNPolicy has (predict_unit / train_loss / params)."""

    def __init__(self, agent: VanillaSAC, loss: str):
        self.agent, self.loss = agent, loss
        self.module = agent.actor

    def predict_unit(self, obs: torch.Tensor) -> torch.Tensor:
        return deterministic_unit(self.agent, obs)

    def train_loss(self, obs: torch.Tensor, teacher_u: torch.Tensor, good_u: torch.Tensor,
                   w: torch.Tensor | None = None) -> torch.Tensor:
        return bc_loss(self.predict_unit(obs), teacher_u, good_u, self.loss, w)

    def params(self) -> list:
        return list(self.agent.actor.trunk.parameters()) + list(self.agent.actor.mean_layer.parameters())


class GNNModel:
    """Same adapter shape for PairGNNPolicy (which already implements predict_unit / train_loss)."""

    def __init__(self, gnn: PairGNNPolicy):
        self.module = gnn
        self.predict_unit, self.train_loss, self.params = gnn.predict_unit, gnn.train_loss, gnn.params


def as_model(agent: "VanillaSAC | PairGNNPolicy", loss: str) -> "MLPModel | GNNModel":
    return GNNModel(agent) if isinstance(agent, PairGNNPolicy) else MLPModel(agent, loss)


def train_bc(agent: VanillaSAC, data: dict[str, np.ndarray], train: np.ndarray, val: np.ndarray,
             loss: str, epochs: int, batch: int, lr: float, patience: int, seed: int,
             log_every: int = 10, w: torch.Tensor | None = None,
             select: str = "loss", select_thresh_deg: float = 0.2,
             lr_schedule: str = "const", lr_final_frac: float = 0.01) -> dict:
    """select: 'loss' keeps the lowest val loss; 'within' the highest val within-threshold rate
    (the val loss is dominated by a few far-off predictions and can disagree with it)."""
    if select not in ("loss", "within"):
        raise ValueError(f"unknown select: {select}")
    if lr_schedule not in ("const", "cosine"):
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = as_model(agent, loss)
    obs   = torch.as_tensor(data["obs"])
    t_u   = torch.as_tensor(to_unit(data["action"]).astype(np.float32))
    g_u   = torch.as_tensor(to_unit(data["good_actions"]).astype(np.float32))
    opt = torch.optim.Adam(model.params(), lr=lr)
    # constant lr plateaus ~1e-3 of the action range above the precision a pocket needs (~0.1°); anneal it
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * lr_final_frac)
             if lr_schedule == "cosine" else None)
    tr_idx, va_idx = np.where(train)[0], np.where(val)[0]

    best_score, best_val, best_state, best_epoch, history = -float("inf"), float("inf"), None, -1, []
    for ep in range(epochs):
        model.module.train()
        perm = rng.permutation(tr_idx)
        tr_loss = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            l = model.train_loss(obs[b], t_u[b], g_u[b], w)
            opt.zero_grad()
            l.backward()
            opt.step()
            tr_loss += l.item() * len(b)
        tr_loss /= max(1, len(perm))
        model.module.eval()
        with torch.no_grad():
            pred_va = model.predict_unit(obs[va_idx])
            va_loss = float(model.train_loss(obs[va_idx], t_u[va_idx], g_u[va_idx], w))
            va_within = within_rate(pred_va, g_u[va_idx], select_thresh_deg)
        history.append((ep, tr_loss, va_loss, va_within))
        if sched is not None:
            sched.step()
        score = -va_loss if select == "loss" else (va_within if va_within == va_within else -float("inf"))
        if score > best_score:
            best_score, best_val, best_epoch = score, va_loss, ep
            best_state = {k: v.clone() for k, v in model.module.state_dict().items()}
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:>4}  train {tr_loss:.6f}  val {va_loss:.6f}  within{select_thresh_deg:g} "
                  f"{va_within:.4f}  best@{best_epoch}", flush=True)
        if ep - best_epoch >= patience:
            break
    model.module.load_state_dict(best_state)
    model.module.eval()
    return {"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": len(history),
            "select": select, "best_score": best_score, "final_lr": opt.param_groups[0]["lr"], "history": history}


def val_metrics(agent: "VanillaSAC | PairGNNPolicy", data: dict[str, np.ndarray], val: np.ndarray) -> dict:
    with torch.no_grad():
        u = as_model(agent, "teacher").predict_unit(torch.as_tensor(data["obs"][val])).numpy()
    pred = u * (ACT_HIGH - ACT_LOW) / 2.0 + (ACT_HIGH + ACT_LOW) / 2.0
    e_t = angle_err_deg(pred[:, 0], data["action"][val, 0])
    e_g = nearest_good_angle_err_deg(pred[:, 0], data["good_actions"][val])
    return {
        "angle_err_teacher_median_deg": float(np.median(e_t)),
        "angle_err_teacher_within_0.2deg": float(np.mean(e_t < 0.2)),
        "angle_err_nearest_good_median_deg": float(np.nanmedian(e_g)),
        "angle_err_nearest_good_within_0.2deg": float(np.nanmean(e_g < 0.2)),
        "speed_err_teacher_median": float(np.median(np.abs(pred[:, 1] - data["action"][val, 1]))),
    }


def evaluate_policy(agent: "VanillaSAC | TransformedPolicy", seed: int, n_games: int, env_kw: dict) -> dict:
    """Same protocol as exp16_wm/train.py final eval: reset(seed) once, then n_games resets."""
    env = BilliardsEnv(**env_kw)
    env.reset(seed=seed)
    lens, pocketed, at5, rewards, succ = [], [], [], [], 0
    for _ in range(n_games):
        obs, _ = env.reset()
        done, k, ep_r, p5 = False, 0, 0.0, 0
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, deterministic=True))
            done, k, ep_r = term or trunc, k + 1, ep_r + r
            succ += int(info["pocketed_this_step"] > 0)
            if k <= 5:
                p5 = info["total_pocketed"]
        lens.append(k); pocketed.append(info["total_pocketed"]); at5.append(p5); rewards.append(ep_r)
    env.close()
    n_balls = env_kw["n_balls"]
    speed = summarize_final_eval(lens, pocketed, at5, n_balls)
    return {
        "clear_rate":         float(np.mean(np.array(pocketed) == n_balls)) * 100,
        "mean_shots":         speed["mean_shots"],
        "mean_shots_cleared": speed["mean_shots_cleared"],
        "pocket_rate_at5":    speed["pocket_rate_at5"] * 100,
        "final_mean_reward":  float(np.mean(rewards)),
        "shot_success_rate":  succ / sum(lens) * 100,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, nargs="+", required=True)
    p.add_argument("--loss",       type=str, default="teacher", choices=list(LOSSES))
    p.add_argument("--epochs",     type=int, default=300)
    p.add_argument("--batch",      type=int, default=512)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["const", "cosine"],
                   help="cosine anneals lr to lr * --lr-final-frac over --epochs (all runs before 2026-10-09 17:50 used const)")
    p.add_argument("--lr-final-frac", type=float, default=0.01)
    p.add_argument("--patience",   type=int, default=50)
    p.add_argument("--angle-scale-deg", type=float, default=0.2,
                   help="angle error (deg) that counts as 1 in the loss")
    p.add_argument("--speed-scale",     type=float, default=0.5,
                   help="speed error (m/s) that counts as 1 in the loss")
    p.add_argument("--val-frac",   type=float, default=0.1)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--eval-seed",  type=int, default=0)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--threads",    type=int, default=2)
    p.add_argument("--model",      type=str, default="mlp", choices=["mlp", "gnn"])
    p.add_argument("--gnn-hidden", type=int, default=128)
    p.add_argument("--wta-eps",    type=float, default=0.05)
    p.add_argument("--sel-weight", type=float, default=10.0)
    p.add_argument("--obs-transform", type=str, default="raw", choices=list(TRANSFORMS))
    p.add_argument("--augment",    type=str, default="none", choices=["none", "flip4"])
    p.add_argument("--select",     type=str, default="within", choices=["loss", "within"],
                   help="best checkpoint: lowest val loss or highest val within-0.2° rate")
    p.add_argument("--data-frac",  type=float, default=1.0, help="fraction of train games used")
    p.add_argument("--hidden",     type=int, nargs="+", default=None,
                   help="actor hidden sizes (default: VanillaSAC's, i.e. same as SAC runs)")
    p.add_argument("--out-root",   type=str, default="logs/experiments")
    return p.parse_args()


def main():
    args = parse_args()
    if args.model == "gnn" and args.obs_transform != "relative":
        raise SystemExit("--model gnn takes the relative observation: add --obs-transform relative")
    torch.set_num_threads(args.threads)
    paths = sorted({p for pat in args.data for p in glob.glob(pat)})
    data = load_teacher_data(paths)
    teacher = json.loads(str(np.load(paths[0])["meta"]))["teacher"]
    data, train, val = build_split(data, args.val_frac, args.seed, args.data_frac, args.augment)
    data["obs"] = TRANSFORMS[args.obs_transform](data["obs"])
    h_tag = ("_gnn" if args.model == "gnn" else "") + \
            ("" if args.hidden is None else "_h" + "x".join(map(str, args.hidden))) + \
            ("" if args.obs_transform == "raw" else f"_{args.obs_transform}") + \
            ("" if args.augment == "none" else f"_{args.augment}") + \
            ("" if args.data_frac >= 1.0 else f"_frac{args.data_frac:g}") + \
            ("" if args.lr_schedule == "const" else "_cos")
    name = (f"exp16_bc_{teacher}_{args.loss}{h_tag}_a{args.angle_scale_deg:g}_v{args.speed_scale:g}"
            f"_s{args.seed}_{time.strftime('%Y-%m-%d@%H%M')}")
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{name}: {len(paths)} files, {len(data['shot'])} shots "
          f"(train {train.sum()}, val {val.sum()}), teacher shot success {np.mean(data['pocketed'] > 0)*100:.1f}%",
          flush=True)

    if args.model == "gnn":
        agent = PairGNNPolicy(args.gnn_hidden, args.wta_eps, args.sel_weight)
    else:
        agent = make_agent(CLEAR_ENV["n_balls"], args.hidden, OBS_DIMS[args.obs_transform])
    t0 = time.time()
    fit = train_bc(agent, data, train, val, args.loss, args.epochs, args.batch, args.lr,
                   args.patience, args.seed, w=dim_weights(args.angle_scale_deg, args.speed_scale),
                   select=args.select, lr_schedule=args.lr_schedule, lr_final_frac=args.lr_final_frac)
    vm = val_metrics(agent, data, val)
    print(f"  trained {fit['epochs_run']} epochs ({time.time()-t0:.0f}s), best ({args.select}) @{fit['best_epoch']}"
          f" val loss {fit['best_val_loss']:.6f}  |  {json.dumps({k: round(v, 4) for k, v in vm.items()})}", flush=True)
    agent.save(os.path.join(out_dir, "bc_model.pt"))

    ev = evaluate_policy(TransformedPolicy(agent, args.obs_transform), args.eval_seed, args.eval_games, CLEAR_ENV)
    print(f"  policy alone ({args.eval_games} games): {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in ev.items()})}",
          flush=True)
    results = {"name": name, "teacher": teacher, "loss": args.loss, "data": paths,
               "n_shots": int(len(data["shot"])), "n_train": int(train.sum()), "n_val": int(val.sum()),
               "train": {k: v for k, v in fit.items() if k != "history"}, "val": vm, "eval": ev,
               "args": vars(args)}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(out_dir, "history.npy"), np.array(fit["history"]))
    print(f"  saved → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
