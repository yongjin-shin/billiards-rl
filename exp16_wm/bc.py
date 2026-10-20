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
import random
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.code_version import code_version
from exp16_wm.sac import VanillaSAC
from exp16_wm.train import ACTION_DIM, ACT_LOW, ACT_HIGH, get_obs_dim, summarize_final_eval
from exp16_wm.planner import wrap_angle
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.obs_transform import TRANSFORMS, OBS_DIMS
from exp16_wm.augment import augment_flip4
from exp16_wm.gnn_policy import PairGNNPolicy, action_diff
from exp16_wm.geometry_aux import N_AUX, aim_targets, pair_geometry, standardize

LOSSES = ("teacher", "nearest_good")

# optional per-state arrays carried through load/split when every shard has them
EXTRA_KEYS = ["good_pairs", "good_made", "good_scratch", "pair_n", "pair_made", "pair_any", "pair_scratch",
              "aux_scan_center", "aux_scan_valid"]                 # the last two: simulator aim targets (--geom-aux-set scan), not in label files
FACT_KEYS = ("good_made", "good_scratch", "pair_n", "pair_made", "pair_scratch")


def seed_everything(seed: int) -> None:
    """Seed before the model is built: weights are initialised at construction, not in train_bc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    keys = ["obs", "action", "good_actions", "n_good", "pocketed", "shot"]
    keys += [k for k in EXTRA_KEYS if all(k in d for d in parts)]
    out = {k: np.concatenate([d[k] for d in parts]) for k in keys}
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
    keys = _KEYS + ([k for k in EXTRA_KEYS if k in data] if augment == "none" else [])   # flips permute pockets
    if augment == "flip4":
        tr_part = augment_flip4(data, tr_idx)
    elif augment == "none":
        tr_part = {k: data[k][tr_idx] for k in keys}
    else:
        raise ValueError(f"unknown augment: {augment}")
    va_part = {k: data[k][va_idx] for k in keys}
    combined = {k: np.concatenate([tr_part[k], va_part[k]]) for k in keys}
    n_tr = len(tr_part["obs"])
    tr_mask = np.arange(len(combined["obs"])) < n_tr
    return combined, tr_mask, ~tr_mask


def load_sims(sims_paths: list[str], label_paths: list[str]) -> dict[str, np.ndarray]:
    """
    Teacher simulations with made-slot identity (exp16_wm.sim_identity) aligned to load_teacher_data(label_paths):
    row = global row of the state in that concatenation. Sims files pair with label files by base name; several sims files with the
    same base name (e.g. scan-centred candidates and a loop's own-aim candidates in two directories) are all used.
    """
    by_name: dict[str, list[str]] = {}
    for p in sorted(sims_paths):
        by_name.setdefault(os.path.basename(p), []).append(p)
    rows, act, made, scr, offset = [], [], [], [], 0
    for lp in sorted(label_paths):
        z_obs = np.load(lp)["obs"]
        sps = by_name.get(os.path.basename(lp))
        if not sps:
            raise FileNotFoundError(f"no sims file for {lp}")
        for sp in sps:
            z = np.load(sp)
            if "sim_made" not in z.files:
                raise ValueError(f"{sp} has no sim_made (run exp16_wm.sim_identity)")
            if not np.array_equal(z["obs"], z_obs):
                raise ValueError(f"{sp}: states differ from {lp}")
            rows.append(z["sim_state"].astype(np.int64) + offset)
            act.append(to_unit(z["sim_action"]).astype(np.float32))
            made.append(z["sim_made"]); scr.append(z["sim_scratch_replay"])
        offset += len(z_obs)
    return {"row": np.concatenate(rows), "a_unit": np.concatenate(act),
            "made": np.concatenate(made), "scratch": np.concatenate(scr)}


def split_rows(game: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Original row → row index after build_split (data_frac 1, no augmentation: train rows first, then val)."""
    train, val = split_by_game(game, val_frac, seed)
    new = np.empty(len(game), np.int64)
    tr, va = np.where(train)[0], np.where(val)[0]
    new[tr] = np.arange(len(tr))
    new[va] = len(tr) + np.arange(len(va))
    return new


def append_train_only(data: dict[str, np.ndarray], train: np.ndarray, val: np.ndarray,
                      extra: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Add extra rows to the train side only (the val set stays exactly the split's); game ids kept unique."""
    missing = [k for k in data if k not in extra]
    if missing:
        raise ValueError(f"train-only data lacks {missing}")
    ex = {k: extra[k] for k in data}
    ex["game"] = extra["game"] - extra["game"].min() + data["game"].max() + 1
    out = {k: np.concatenate([data[k], ex[k].astype(data[k].dtype)]) for k in data}
    n = len(extra["obs"])
    return out, np.concatenate([train, np.ones(n, bool)]), np.concatenate([val, np.zeros(n, bool)])


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
        return ((action_diff(pred_unit, teacher_unit) * w) ** 2).sum(-1).mean()
    if loss == "nearest_good":
        valid = ~torch.isnan(good_unit[..., 0])                          # (B,K)
        d = ((action_diff(pred_unit.unsqueeze(1), torch.nan_to_num(good_unit)) * w) ** 2).sum(-1)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        nearest = d.min(dim=1).values
        fallback = ((action_diff(pred_unit, teacher_unit) * w) ** 2).sum(-1)
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
                   w: torch.Tensor | None = None, pairs: torch.Tensor | None = None,
                   extras: dict | None = None) -> torch.Tensor:
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
             lr_schedule: str = "const", lr_final_frac: float = 0.01, use_pairs: bool = False,
             selection: str = "global", use_extras: bool = False,
             sims: dict[str, np.ndarray] | None = None, made2_weight: float = 1.0, sims_batch: int = 1024,
             aux: np.ndarray | None = None, aux_weight: float = 0.0, aux_only: bool = False,
             train_plateau: int = 0, train_plateau_tol: float = 0.01, made_only: bool = False, made_val_sims: int = 20000,
             ckpt_path: str | None = None, ckpt_every: int = 0, stop_after: int | None = None,
             select_fn=None, select_every: int = 25) -> dict:
    """select: 'loss' keeps the lowest val loss; 'within' the highest val within-threshold rate
    (the val loss is dominated by a few far-off predictions and can disagree with it).
    aux_only: train the encoder + aux head on the geometry look-ahead alone (no action / selection / made loss);
    the 'val loss' is then the val aux MSE and the checkpoint is the lowest of it.
    made_only: train on the made2 BCE of `sims` alone (no action / selection loss, teacher-free when the sims come from a search);
    the 'val loss' is then the made2 BCE of up to made_val_sims simulations of val states and the checkpoint is the lowest of it.
    train_plateau = W > 0: also stop when the training loss has converged — the lowest training loss of the last W epochs is not
    more than train_plateau_tol (relative) below the lowest before them.
    ckpt_path + ckpt_every = K > 0: every K epochs the whole training state (weights, optimiser, lr schedule, RNG states, best
    checkpoint, histories) is written to ckpt_path, and an existing ckpt_path is resumed from — a run interrupted (e.g. a reclaimed
    Spot instance) continues where it stopped and ends with the same weights as an uninterrupted run.
    stop_after = E: leave the loop after epoch E − 1 (testing resumption; the checkpoint is written there).
    select_fn(module) → score (higher is better), e.g. the fixed val set V0's top-1 success: evaluated every select_every epochs and at
    the last epoch, and the best of those epochs is the returned model instead of the lowest val loss (exp16_wm.val_set)."""
    if select not in ("loss", "within"):
        raise ValueError(f"unknown select: {select}")
    if lr_schedule not in ("const", "cosine"):
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")
    if aux_only:
        if aux is None or aux_weight <= 0:
            raise ValueError("aux_only needs aux targets and aux_weight > 0")
        if sims is not None:
            raise ValueError("aux_only does not train the made head (no sims)")
        select = "loss"
    if made_only:
        if sims is None:
            raise ValueError("made_only needs sims")
        if aux_only:
            raise ValueError("made_only and aux_only exclude each other")
        select = "loss"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = as_model(agent, loss)
    obs   = torch.as_tensor(data["obs"])
    t_u   = torch.as_tensor(to_unit(data["action"]).astype(np.float32))
    g_u   = torch.as_tensor(to_unit(data["good_actions"]).astype(np.float32))
    if use_pairs and "good_pairs" not in data:
        raise ValueError("use_pairs needs data['good_pairs'] (run exp16_wm.pair_labels; no --augment flip4)")
    pr = torch.as_tensor(data["good_pairs"].astype(np.int64)) if use_pairs else None
    ex = None
    if selection not in ("global", "factorized"):
        raise ValueError(f"unknown selection: {selection}")
    if selection == "factorized" or use_extras:
        missing = [k for k in FACT_KEYS if k not in data]
        if missing:
            raise ValueError(f"factorized / outcome targets need {missing} (exp16_wm.outcomes / exp16_wm.pair_stats)")
        ex = {k: torch.as_tensor(data[k].astype(np.float32 if k in ("good_made", "good_scratch") else np.int64)) for k in FACT_KEYS}
    aux_t = None
    if aux is not None:                                  # standardised geometry targets (N, 18, K), one row per training row
        if len(aux) != len(obs):
            raise ValueError(f"aux has {len(aux)} rows, the data {len(obs)}")
        aux_t = torch.as_tensor(aux)
    opt = torch.optim.Adam(model.params(), lr=lr)
    # constant lr plateaus ~1e-3 of the action range above the precision a pocket needs (~0.1°); anneal it
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * lr_final_frac)
             if lr_schedule == "cosine" else None)
    tr_idx, va_idx = np.where(train)[0], np.where(val)[0]
    if sims is not None:
        from exp16_wm.sim_identity import from_bitmask
        n_slots = (obs.shape[1] - 12) // 3 * 6                      # 18 for 3 balls, 54 for 9 (G1)
        sims_tr = np.where(train[sims["row"]])[0]
        s_row = torch.as_tensor(sims["row"]); s_a = torch.as_tensor(sims["a_unit"])
        s_mask = sims["made"]; s_scr = torch.as_tensor(sims["scratch"])   # made bitmask decoded per batch
        sims_va = np.where(val[sims["row"]])[0]
        if made_only and len(sims_va) == 0:
            raise ValueError("made_only needs simulations of val states for the checkpoint")
        sims_va = np.sort(np.random.default_rng(seed + 7).choice(sims_va, size=min(made_val_sims, len(sims_va)), replace=False))

    best_score, best_val, best_state, best_epoch, history = -float("inf"), float("inf"), None, -1, []
    stop_reason = None
    use_aux = aux_t is not None and aux_weight > 0
    aux_history: list[tuple[int, float, float]] = []             # (epoch, train aux MSE, val aux MSE), standardised targets
    start_ep = 0
    select_history: list[tuple[int, float]] = []
    if ckpt_path and ckpt_every > 0 and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.module.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"])
        best_score, best_val, best_state, best_epoch = ck["best_score"], ck["best_val"], ck["best_state"], ck["best_epoch"]
        history, aux_history, start_ep = ck["history"], ck["aux_history"], ck["epoch"]
        select_history = ck.get("select_history", [])
        print(f"  resumed from {ckpt_path} at epoch {start_ep}", flush=True)

    def save_ckpt(next_ep: int) -> None:
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "sched": None if sched is None else sched.state_dict(),
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "best_score": best_score, "best_val": best_val,
                    "best_state": best_state, "best_epoch": best_epoch, "history": history, "aux_history": aux_history,
                    "select_history": select_history, "epoch": next_ep}, tmp)
        os.replace(tmp, ckpt_path)                                   # never a half-written checkpoint

    for ep in range(start_ep, epochs):
        model.module.train()
        perm = rng.permutation(tr_idx)
        tr_loss = tr_aux = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            l = None if (aux_only or made_only) else model.train_loss(obs[b], t_u[b], g_u[b], w, pairs=None if pr is None else pr[b],
                                                      extras=None if ex is None else {k: v[b] for k, v in ex.items()})
            if sims is not None:
                sb = rng.choice(sims_tr, size=min(sims_batch, len(sims_tr)), replace=False)
                made_b = torch.as_tensor(from_bitmask(s_mask[sb], n_slots))
                lm = made2_weight * model.module.made2_loss(obs[s_row[sb]], s_a[sb], made_b, s_scr[sb])
                l = lm if l is None else l + lm
            if use_aux:
                la = model.module.aux_loss(obs[b], aux_t[b])
                l = aux_weight * la if l is None else l + aux_weight * la
                tr_aux += la.item() * len(b)
            opt.zero_grad()
            l.backward()
            opt.step()
            tr_loss += l.item() * len(b)
        tr_loss /= max(1, len(perm))
        tr_aux /= max(1, len(perm))
        model.module.eval()
        with torch.no_grad():
            va_aux = float(model.module.aux_loss(obs[va_idx], aux_t[va_idx])) if use_aux else float("nan")
            if aux_only:
                va_loss, va_within = va_aux, float("nan")
            elif made_only:                                          # val made2 BCE on a fixed sample of val simulations
                tot, cnt = 0.0, 0
                for lo in range(0, len(sims_va), 4096):
                    sv = sims_va[lo:lo + 4096]
                    tot += float(model.module.made2_loss(obs[s_row[sv]], s_a[sv], torch.as_tensor(from_bitmask(s_mask[sv], n_slots)), s_scr[sv])) * len(sv)
                    cnt += len(sv)
                va_loss, va_within = tot / cnt, float("nan")
            else:
                pred_va = model.predict_unit(obs[va_idx])
                va_loss = float(model.train_loss(obs[va_idx], t_u[va_idx], g_u[va_idx], w,
                                                 pairs=None if pr is None else pr[va_idx],
                                                 extras=None if ex is None else {k: v[va_idx] for k, v in ex.items()}))
                va_within = within_rate(pred_va, g_u[va_idx], select_thresh_deg)
        history.append((ep, tr_loss, va_loss, va_within))
        if use_aux:
            aux_history.append((ep, tr_aux, va_aux))
        if sched is not None:
            sched.step()
        if select_fn is not None:                                    # e.g. V0 top-1: only on its epochs, the rest never become best
            if (ep + 1) % select_every == 0 or ep + 1 == epochs:
                model.module.eval()
                score = float(select_fn(model.module)); select_history.append((ep, score))
                print(f"  epoch {ep:>4}  select {score:.4f}  (best {max(best_score, score):.4f})", flush=True)
            else:
                score = -float("inf")
        else:
            score = -va_loss if select == "loss" else (va_within if va_within == va_within else -float("inf"))
        if score > best_score:
            best_score, best_val, best_epoch = score, va_loss, ep
            best_state = {k: v.clone() for k, v in model.module.state_dict().items()}
        if log_every and ep % log_every == 0:
            kw = ("  logvar[ang,spd,sel] " + " ".join(f"{x:+.2f}" for x in model.module.log_var.detach().clamp(-3, 3).tolist())
                  if getattr(model.module, "uncertainty_weights", False) else "")
            ax = f"  aux train {tr_aux:.4f} val {va_aux:.4f}" if use_aux else ""
            print(f"  epoch {ep:>4}  train {tr_loss:.6f}  val {va_loss:.6f}  within{select_thresh_deg:g} "
                  f"{va_within:.4f}  best@{best_epoch}{ax}{kw}", flush=True)
        if ckpt_path and ckpt_every > 0 and ((ep + 1) % ckpt_every == 0 or ep + 1 == stop_after):
            save_ckpt(ep + 1)
        if ep - best_epoch >= patience:
            break
        if train_plateau and train_plateau_converged([h[1] for h in history], train_plateau, train_plateau_tol):
            stop_reason = f"training loss converged at epoch {ep}"
            break
        if stop_after is not None and ep + 1 >= stop_after:
            stop_reason = f"stop_after {stop_after}"
            break
    last_state = {k: v.clone() for k, v in model.module.state_dict().items()}      # the weights after the last epoch run
    model.module.load_state_dict(best_state)
    model.module.eval()
    return {"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": len(history),
            "select": select, "best_score": best_score, "final_lr": opt.param_groups[0]["lr"], "history": history,
            "aux_history": aux_history, "last_state": last_state, "stop_reason": stop_reason, "select_history": select_history}


SELF_AIM_NO_TARGET_STATS = (0.0, 180.0)                           # --made2-self-aim with --geom-aux-weight 0: aim (deg) = 180 · head output
ANG31 = tuple(np.round(np.arange(-0.3, 0.3 + 1e-9, 0.02), 4))     # refine grid used in the evaluations (deg around the centre)
ABS7 = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)                        # absolute refine speeds (m/s)


def train_seed(seed: int, init_seed: int | None) -> int:
    """Seed of the weight initialisation and the batch order: --init-seed if given, else --seed (the train/val split always follows --seed)."""
    return seed if init_seed is None else init_seed


def train_plateau_converged(train_losses: list[float], window: int, tol: float) -> bool:
    """True when the lowest loss of the last `window` epochs is not more than `tol` (relative) below the lowest before them."""
    if window <= 0 or len(train_losses) < 2 * window:
        return False
    before, recent = min(train_losses[:-window]), min(train_losses[-window:])
    return recent > (1.0 - tol) * before


def load_scan_targets(label_paths: list[str], target_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Simulator aim targets (exp16_wm.aim_sims scan) aligned with load_teacher_data(label_paths): (centre (N, 18) deg, valid (N, 18))."""
    cs, vs = [], []
    for lp in sorted(label_paths):
        tp = os.path.join(target_dir, os.path.basename(lp))
        z = np.load(tp)
        if len(z["scan_center_deg"]) != len(np.load(lp)["obs"]):
            raise ValueError(f"{tp}: {len(z['scan_center_deg'])} rows, the label file {len(np.load(lp)['obs'])}")
        cs.append(z["scan_center_deg"].astype(np.float32)); vs.append(z["scan_valid"].astype(bool))
    return np.concatenate(cs), np.concatenate(vs)


def aim_encoder_stats(run_dir: str) -> tuple[float, float]:
    """
    Target standardisation (mean, std in degrees) of an aux-only aim model, from its results.json if stored, else recomputed from its
    own data, split and target set — never from the data of the run that uses the encoder (whose states can differ).
    """
    r = json.load(open(os.path.join(run_dir, "results.json")))
    if "aux_mean" in r:
        return float(r["aux_mean"][0]), float(r["aux_std"][0])
    a = r["args"]
    paths = sorted({q for pat in a["data"] for q in glob.glob(pat)})
    raw = load_teacher_data(paths)
    if a.get("geom_aux_set") == "scan":
        raw["aux_scan_center"], raw["aux_scan_valid"] = load_scan_targets(paths, a["aux_target_dir"])
    data, train, _ = build_split(raw, a.get("val_frac", 0.1), a["seed"], a.get("data_frac", 1.0), "none")
    if a.get("geom_aux_set") == "scan":
        t, v = data["aux_scan_center"][..., None], data["aux_scan_valid"]
    else:
        t, v = aim_targets(TRANSFORMS["relative"](data["obs"]))
    _, mean, std = standardize(np.nan_to_num(t), v, train)
    return float(mean[0]), float(std[0])


def load_aim_encoder(run_dir: str) -> PairGNNPolicy:
    """An aux-only aim model (--geom-aux-set aim --aux-only): its last epoch's weights if saved (--save-last), else the selected best."""
    a = json.load(open(os.path.join(run_dir, "results.json")))["args"]
    if not (a.get("aux_only") and a.get("geom_aux_set") in ("aim", "scan")):
        raise ValueError(f"{run_dir} is not an aux-only aim model")
    m = PairGNNPolicy(a["gnn_hidden"], enc_layers=a.get("gnn_layers", 2), aux_dim=1, fourier=int(a.get("input_fourier", 0) or 0))
    path = os.path.join(run_dir, "bc_model_last.pt")
    m.load_state_dict(torch.load(path if os.path.exists(path) else os.path.join(run_dir, "bc_model.pt"))["gnn"])
    return m.eval()


def _save_last(agent: PairGNNPolicy, last_state: dict, path: str) -> None:
    """Save the last epoch's weights to path; the agent keeps (is restored to) its selected best weights."""
    best = {k: v.clone() for k, v in agent.state_dict().items()}
    agent.load_state_dict(last_state)
    agent.save(path)
    agent.load_state_dict(best)


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


def oracle_pair_metrics(agent: PairGNNPolicy, data: dict[str, np.ndarray], val: np.ndarray) -> dict:
    """
    Regression quality with the pair given (selection taken out): for every (candidate k, pair p) where
    clean label action k really made pair p (good_made), pair p's action vs action k. Plus how often the
    top-1 pair is one the coarse scan could make (pair_made > 0). {} without the outcome arrays.
    """
    if "good_made" not in data or "good_scratch" not in data:
        return {}
    with torch.no_grad():
        o = agent.forward_full(torch.as_tensor(data["obs"][val]))
    means, top = o["means"].numpy(), o["logits"].numpy().argmax(axis=1)
    gm = data["good_made"][val].astype(bool)
    K2 = gm.shape[1]
    g = to_unit(data["good_actions"][val][:, :K2])
    ok = ~np.isnan(g[..., 0]) & ~data["good_scratch"][val].astype(bool)
    b, k, p = np.nonzero(gm & ok[..., None])
    out = {}
    if len(b):
        d = (means[b, p, 0] - g[b, k, 0]) * np.pi
        err = np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))
        out = {"oracle_angle_err_median_deg": float(np.median(err)),
               "oracle_angle_within_0.2deg": float(np.mean(err < 0.2)),
               "oracle_speed_err_median": float(np.median(np.abs(means[b, p, 1] - g[b, k, 1])) * (ACT_HIGH[1] - ACT_LOW[1]) / 2.0)}
    if "pair_made" in data:
        out["top1_pair_made_rate"] = float(np.mean(data["pair_made"][val][np.arange(len(top)), top] > 0))
    return out


def made2_metrics(agent: PairGNNPolicy, data: dict[str, np.ndarray], val: np.ndarray,
                  sims: dict[str, np.ndarray], max_n: int = 50_000, seed: int = 0) -> dict:
    """
    Verifier quality on val states' simulations (live pairs): AUC of P(made | e_ij, a), and the same AUC
    restricted to pairs the state can make at all (coarse pair_made > 0 — telling hits from near misses).
    Trivial baseline: the state-level coarse made rate of the pair (pair_made / pair_n), blind to the action.
    """
    from sklearn.metrics import roc_auc_score
    from exp16_wm.sim_identity import from_bitmask
    idx = np.where(val[sims["row"]])[0]
    if len(idx) == 0:
        return {}
    idx = np.random.default_rng(seed).choice(idx, size=min(max_n, len(idx)), replace=False)
    rows = sims["row"][idx]
    with torch.no_grad():
        o = agent.forward_full(torch.as_tensor(data["obs"][rows]))
        z = agent.made2_logits(o, torch.as_tensor(sims["a_unit"][idx])).numpy()
    target = from_bitmask(sims["made"][idx], z.shape[1]) & ~sims["scratch"][idx][:, None]
    v = o["valid"].numpy()
    out = {"made2_auc": float(roc_auc_score(target[v], z[v])), "made2_pos_rate": float(target[v].mean())}
    if "pair_made" in data:
        rate = data["pair_made"][rows] / np.maximum(data["pair_n"][rows], 1)
        out["coarse_rate_auc"] = float(roc_auc_score(target[v], rate[v]))
        f = v & (data["pair_made"][rows] > 0)
        if target[f].any() and not target[f].all():
            out["made2_auc_feasible"] = float(roc_auc_score(target[f], z[f]))
            out["coarse_rate_auc_feasible"] = float(roc_auc_score(target[f], rate[f]))
    return out


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
    p.add_argument("--gnn-layers", type=int, default=2, help="hidden layers of each encoder MLP (ball message / ball / pair)")
    p.add_argument("--speed-loss", type=str, default="mse", choices=["mse", "l1"],
                   help="speed term of the pair loss: squared error (default) or absolute error (a median instead of a mean)")
    p.add_argument("--uncertainty-weights", action="store_true",
                   help="learn the weight of the angle / speed / selection terms of the pair loss (Kendall et al. 2018); "
                        "needs --pair-supervision --no-wta-fallback")
    p.add_argument("--geom-aux-weight", type=float, default=0.0,
                   help="> 0: auxiliary look-ahead of geometry — a head on each pair embedding predicts the pocket entry deviation, "
                        "the ball's cushion distance and the blockers on the two shot paths (training only); this is the loss weight")
    p.add_argument("--made-only", action="store_true",
                   help="train the made2 head (and the encoder) on --sims alone: no action / selection loss; evaluated with infer=all_pairs. "
                        "With search-generated sims (exp16_wm.aim_sims) nothing comes from the teacher's labels")
    p.add_argument("--made2-no-proposal", action="store_true",
                   help="made2 head input without (a − pair head proposal); for --made-only, where the pair head is not trained")
    p.add_argument("--init-seed", type=int, default=None,
                   help="seed of the weight initialisation and the batch order (default: --seed); the train/val split always follows --seed. "
                        "For ensembles of models on the same split")
    p.add_argument("--input-fourier", type=int, default=0,
                   help="GNN input representation: add sin/cos Fourier features at this many octaves to every 2-D coordinate")
    p.add_argument("--train-plateau", type=int, default=0,
                   help="> 0: also stop when the training loss has converged over this many epochs (see --train-plateau-tol)")
    p.add_argument("--train-plateau-tol", type=float, default=0.01)
    p.add_argument("--val-v0", type=str, default=None,
                   help="--made-only: choose the final model by top-1 success on this fixed val set (exp16_wm.val_set) every --v0-every epochs "
                        "instead of the lowest made BCE on the training data's own val split")
    p.add_argument("--v0-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="> 0: save the whole training state every this many epochs under <out-root>/_ckpt/ and resume from it if present "
                        "(the run dir name is kept), so an interrupted run continues")
    p.add_argument("--save-last", action="store_true", help="also save the last epoch's weights (bc_model_last.pt) next to the selected best")
    p.add_argument("--geom-aux-set", type=str, default="four", choices=["four", "aim", "scan"],
                   help="auxiliary geometry targets: 'four' = entry deviation, cushion, two blockers; 'aim' = the exact ghost-ball aim of every pair; "
                        "'scan' = the simulator's success-window centre of every pair (exp16_wm.aim_sims scan, --aux-target-dir)")
    p.add_argument("--aux-target-dir", type=str, default=None, help="--geom-aux-set scan: directory of the scan target files")
    p.add_argument("--init-from", type=str, default=None,
                   help="--model gnn: initialise the encoder and the aux head from this checkpoint (bc_model.pt); the "
                        "action / selection / made heads start fresh. E.g. an --aux-only run (pretrain on the geometry look-ahead)")
    p.add_argument("--aux-only", action="store_true",
                   help="diagnostic with --geom-aux-weight: train the encoder on the geometry look-ahead alone (no action / selection / "
                        "made loss), no policy evaluation; asks whether the encoder can represent the targets at all")
    p.add_argument("--speed-bins", type=int, default=0,
                   help="> 0: speed head = categorical over this many log-spaced bins (cross-entropy on the label's bin); "
                        "needs --pair-supervision --no-wta-fallback")
    p.add_argument("--wta-eps",    type=float, default=0.05)
    p.add_argument("--sel-weight", type=float, default=10.0)
    p.add_argument("--obs-transform", type=str, default="raw", choices=list(TRANSFORMS))
    p.add_argument("--augment",    type=str, default="none", choices=["none", "flip4"])
    p.add_argument("--selection", type=str, default="global", choices=["global", "factorized"],
                   help="factorized: ball head + per-ball pocket softmax, outcome-grounded targets, scratch head")
    p.add_argument("--fact-terms", type=str, nargs="+", default=["reg", "sel", "scratch"],
                   choices=["reg", "sel", "scratch"], help="factorized loss terms to keep (ablation)")
    p.add_argument("--sel-target", type=str, default=None, choices=["label", "made"],
                   help="modular loss (with --reg-target): selection target = labelled pairs or pair_made")
    p.add_argument("--reg-target", type=str, default=None, choices=["angle", "made", "made_same"],
                   help="modular loss (with --sel-target): regression pair = angle-matched (good_pairs), really made (good_made), "
                        "or really made and the aimed ball (made_same: no combination shots)")
    p.add_argument("--scratch-head", action="store_true", help="modular loss: + scratch head BCE")
    p.add_argument("--made-head", action="store_true",
                   help="modular loss: + auxiliary made head BCE to pair_made / pair_n (look-ahead, not used at inference)")
    p.add_argument("--scratch-penalty", type=float, default=3.0,
                   help="factorized: extra evaluation with score −= λ·σ(scratch logit)")
    p.add_argument("--pair-supervision", action="store_true",
                   help="GNN: train each pair head on the pair that produced the label (needs exp16_wm.pair_labels)")
    p.add_argument("--geom-angle", action="store_true",
                   help="GNN: the angle output comes from a pair embedding without the ball-ball message (geometry only)")
    p.add_argument("--made2-head", action="store_true",
                   help="look-ahead stage 2: action-conditional made head trained on teacher sims (--sims)")
    p.add_argument("--made2-geom", action="store_true",
                   help="diagnostic: the made2 head also gets (action angle − geometric aim)/0.2° (needs --made2-head)")
    p.add_argument("--made2-weight", type=float, default=1.0)
    p.add_argument("--made2-aim-from", type=str, nargs="+", default=None,
                   help="cascade (with --made2-geom): the made2 head's aim comes from these frozen aux-only aim models (run dir globs, "
                        "{seed} is filled in; several are averaged on the circle) instead of the ghost-ball formula")
    p.add_argument("--made2-self-aim", type=str, default=None, choices=["grad", "detach"],
                   help="end-to-end (with --made2-geom --geom-aux-set aim/scan --geom-aux-weight > 0): the made2 head's aim is this network's "
                        "own aux head, trained together; 'detach' stops the made loss at the aim (roadmap EE1 / EE2)")
    p.add_argument("--sims", type=str, nargs="+", default=["logs/teacher_data/relabel_s1000/sims/*.npz"])
    p.add_argument("--sims-batch", type=int, default=1024)
    p.add_argument("--no-wta-fallback", action="store_true",
                   help="--pair-supervision: leave rows with no labelled pair out of the loss instead of training them with WTA")
    p.add_argument("--angle-wrap", action="store_true",
                   help="wrap the angle difference in the training loss (±180° seam). Off by default: with 3 seeds "
                        "it was worse for the policy at every lr tried (experiments.md, 2026-10-09)")
    p.add_argument("--select",     type=str, default="within", choices=["loss", "within"],
                   help="best checkpoint: lowest val loss or highest val within-0.2° rate")
    p.add_argument("--data-frac",  type=float, default=1.0, help="fraction of train games used")
    p.add_argument("--train-only-data", type=str, nargs="+", default=None,
                   help="extra label files added to train only (val stays the --data split's)")
    p.add_argument("--hidden",     type=int, nargs="+", default=None,
                   help="actor hidden sizes (default: VanillaSAC's, i.e. same as SAC runs)")
    p.add_argument("--out-root",   type=str, default="logs/experiments")
    return p.parse_args()


def main():
    args = parse_args()
    if (args.pair_supervision or args.selection == "factorized") and (args.model != "gnn" or args.augment != "none"):
        raise SystemExit("--pair-supervision / --selection factorized need --model gnn and --augment none")
    if args.model == "gnn" and args.obs_transform != "relative":
        raise SystemExit("--model gnn takes the relative observation: add --obs-transform relative")
    modular = args.sel_target is not None or args.reg_target is not None or args.scratch_head or args.made_head
    if modular and (args.sel_target is None or args.reg_target is None or args.model != "gnn"):
        raise SystemExit("modular loss needs --model gnn and both --sel-target and --reg-target")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)

    import exp16_wm.gnn_policy as _gp
    _gp.ANGLE_WRAP = args.angle_wrap

    paths = sorted({p for pat in args.data for p in glob.glob(pat)})
    data = load_teacher_data(paths)
    teacher = json.loads(str(np.load(paths[0])["meta"]))["teacher"]
    if args.geom_aux_weight > 0 and args.geom_aux_set == "scan":    # simulator aim targets ride along through the split
        if not args.aux_target_dir:
            raise SystemExit("--geom-aux-set scan needs --aux-target-dir")
        data["aux_scan_center"], data["aux_scan_valid"] = load_scan_targets(paths, args.aux_target_dir)
    sims = None
    if args.made2_geom and not args.made2_head:
        raise SystemExit("--made2-geom needs --made2-head")
    aim_enc, aim_stats = None, None
    if args.made2_aim_from:
        if not args.made2_geom:
            raise SystemExit("--made2-aim-from needs --made2-geom")
        dirs = []
        for pat in args.made2_aim_from:
            found = glob.glob(pat.format(seed=args.seed))
            if len(found) != 1:
                raise SystemExit(f"--made2-aim-from {pat}: expected one run dir, found {found}")
            dirs.append(found[0])
        args.made2_aim_dirs = dirs                                    # resolved, saved in results.json for eval_by_balls.load_run
        args.made2_aim_dir = dirs[0] if len(dirs) == 1 else None
        aim_enc = load_aim_encoder(dirs[0]) if len(dirs) == 1 else [load_aim_encoder(d) for d in dirs]
    if args.made2_head:
        if args.data_frac < 1.0 or args.augment != "none" or args.model != "gnn":
            raise SystemExit("--made2-head needs --model gnn, --data-frac 1, --augment none")
        sims = load_sims(sorted({p for pat in args.sims for p in glob.glob(pat)}), paths)
        sims["row"] = split_rows(data["game"], args.val_frac, args.seed)[sims["row"]]
    data, train, val = build_split(data, args.val_frac, args.seed, args.data_frac, args.augment)
    extra_tag = ""
    if args.train_only_data:
        extra_paths = sorted({p for pat in args.train_only_data for p in glob.glob(pat)})
        data, train, val = append_train_only(data, train, val, load_teacher_data(extra_paths))
        extra_tag = "_plus-" + os.path.basename(os.path.dirname(os.path.dirname(extra_paths[0])))
    if args.made2_self_aim and (not args.made2_geom or args.made2_aim_from or args.geom_aux_set not in ("aim", "scan")):
        raise SystemExit("--made2-self-aim needs --made2-geom, --geom-aux-set aim or scan, and no --made2-aim-from")
    if args.made_only and not (args.made2_head and args.made2_geom):
        raise SystemExit("--made-only needs --made2-head --made2-geom (it chooses among candidates around the aim)")
    if args.init_from and args.model != "gnn":
        raise SystemExit("--init-from needs --model gnn")
    if args.aux_only and (args.geom_aux_weight <= 0 or args.made2_head):
        raise SystemExit("--aux-only needs --geom-aux-weight > 0 and no --made2-head")
    aux = None
    if args.geom_aux_weight > 0:                                  # geometry targets from the RAW observations (before the transform)
        if args.model != "gnn" or args.augment != "none":
            raise SystemExit("--geom-aux-weight needs --model gnn and --augment none")
        if args.geom_aux_set == "aim":                            # the exact aim, from the relative frame the policy sees
            aux, aux_mean, aux_std = standardize(*aim_targets(TRANSFORMS["relative"](data["obs"])), train)
        elif args.geom_aux_set == "scan":                         # the simulator's window centre; pairs without one are NaN (masked)
            sv = data["aux_scan_valid"] & aim_targets(TRANSFORMS["relative"](data["obs"]))[1]
            aux, aux_mean, aux_std = standardize(np.nan_to_num(data["aux_scan_center"])[..., None], sv, train)
            aux[~sv] = np.nan
        else:
            env_dims = BilliardsEnv(**CLEAR_ENV)
            aux, aux_mean, aux_std = standardize(*pair_geometry(data["obs"], (env_dims.table_width, env_dims.table_length)), train)
            env_dims.close()
    if aim_enc is not None:                                          # the aim encoders' own target standardisation
        stats = [aim_encoder_stats(d) for d in args.made2_aim_dirs]
        if max(abs(s_[0] - stats[0][0]) + abs(s_[1] - stats[0][1]) for s_ in stats) > 1e-4:
            raise SystemExit(f"--made2-aim-from encoders were standardised differently: {stats}")
        aim_stats = stats[0]
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.teacher_data import clear_env_kw
    game_env = clear_env_kw(n_balls_raw(data["obs"].shape[1]))        # the rack the data came from (G1: 3, 6, 9 balls)
    data["obs"] = TRANSFORMS[args.obs_transform](data["obs"])

    h_tag = ("_gnn" if args.model == "gnn" else "") + \
            ("" if args.hidden is None else "_h" + "x".join(map(str, args.hidden))) + \
            ("" if args.obs_transform == "raw" else f"_{args.obs_transform}") + \
            ("" if args.augment == "none" else f"_{args.augment}") + \
            ("" if args.data_frac >= 1.0 else f"_frac{args.data_frac:g}") + \
            ("" if args.lr_schedule == "const" else "_cos") + \
            (f"_mod-{args.selection[0]}{args.sel_target[0]}{dict(angle='a', made='m', made_same='M')[args.reg_target]}{'s' if args.scratch_head else ''}{'_mh' if args.made_head else ''}" if modular else
             ("_pairs" + ("-nofb" if args.no_wta_fallback else "") if args.pair_supervision else "") + ("_fact" + ("" if len(args.fact_terms) == 3 else "-" + "".join(t[0] for t in args.fact_terms))
                                                            if args.selection == "factorized" else "")) + \
            ("" if args.gnn_hidden == 128 or args.model != "gnn" else f"_H{args.gnn_hidden}") + \
            ("" if args.gnn_layers == 2 or args.model != "gnn" else f"_L{args.gnn_layers}") + \
            ("_sl1" if args.speed_loss == "l1" else "") + (f"_sb{args.speed_bins}" if args.speed_bins else "") + ("_kw" if args.uncertainty_weights else "") + (f"_ga{args.geom_aux_weight:g}" if args.geom_aux_weight > 0 else "") + ("_aim" if args.geom_aux_weight > 0 and args.geom_aux_set == "aim" else "") + ("_scan" if args.geom_aux_weight > 0 and args.geom_aux_set == "scan" else "") + ("_auxonly" if args.aux_only else "") + ("_madeonly" if args.made_only else "") + ("_noprop" if args.made2_no_proposal else "") + (f"_i{args.init_seed}" if args.init_seed is not None else "") + ("_init" if args.init_from else "") + (f"_ff{args.input_fourier}" if args.input_fourier else "") + \
            ("_geoang" if args.geom_angle else "") + extra_tag + (f"_m2w{args.made2_weight:g}" if args.made2_head else "") + ("geo" if args.made2_geom else "") + ("L" if args.made2_aim_from else "") + (f"S{args.made2_self_aim[0]}" if args.made2_self_aim else "") + ("_wrap" if args.angle_wrap else "_nowrap")
    name = (f"exp16_bc_{teacher}_{args.loss}{h_tag}_a{args.angle_scale_deg:g}_v{args.speed_scale:g}"
            f"_s{args.seed}_{time.strftime('%Y-%m-%d@%H%M')}")
    ckpt_path = None
    if args.ckpt_every > 0:                                          # key without the timestamp; a resumed run keeps its first name
        ckpt_dir = os.path.join(args.out_root, "_ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        key = name.rsplit("_", 1)[0]
        ckpt_path = os.path.join(ckpt_dir, key + ".pt")
        name_file = os.path.join(ckpt_dir, key + ".name")
        if os.path.exists(name_file):
            name = open(name_file).read().strip()
        else:
            open(name_file, "w").write(name)
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{name}: {len(paths)} files, {len(data['shot'])} shots "
          f"(train {train.sum()}, val {val.sum()}), teacher shot success {np.mean(data['pocketed'] > 0)*100:.1f}%",
          flush=True)

    if args.init_seed is not None:                                   # another initialisation on the same split
        torch.manual_seed(train_seed(args.seed, args.init_seed))
    if args.model == "gnn":
        agent = PairGNNPolicy(args.gnn_hidden, args.wta_eps, args.sel_weight,
                              factorized=args.selection == "factorized", scratch_head=args.scratch_head,
                              made_head=args.made_head, made2_head=args.made2_head,
                              geom_angle=args.geom_angle, made2_geom=args.made2_geom,
                              enc_layers=args.gnn_layers, speed_loss=args.speed_loss, speed_bins=args.speed_bins,
                              uncertainty_weights=args.uncertainty_weights, aux_dim=1 if args.made2_self_aim else ((1 if args.geom_aux_set in ("aim", "scan") else N_AUX) if args.geom_aux_weight > 0 else 0),
                              aim_encoder=aim_enc, fourier=args.input_fourier, made2_no_proposal=args.made2_no_proposal,
                              self_aim=args.made2_self_aim or "")
        if args.made2_self_aim:                                      # the own aim head regresses the standardised target
            if args.geom_aux_weight > 0:
                agent.set_aim_stats(float(aux_mean[0]), float(aux_std[0]))
            else:                                                    # λ = 0: no aim target at all, a fixed scale (head output 1 = 180°)
                agent.set_aim_stats(*SELF_AIM_NO_TARGET_STATS)
        if aim_enc is not None:
            agent.set_aim_stats(*aim_stats)
            with torch.no_grad():                                    # how good is the learned aim on the val rows (all live pairs)
                rv = torch.as_tensor(data["obs"][val])
                d_ = agent._learned_aim(rv) - agent._geometric_aim(rv)
                err = torch.rad2deg(torch.atan2(torch.sin(d_), torch.cos(d_))).abs()[torch.as_tensor(aim_targets(data["obs"][val])[1])]
            print(f"  cascade aim from {len(args.made2_aim_dirs)} encoder(s) {args.made2_aim_dirs}: val |learned − exact| median {err.median():.3f}°, "
                  f"p90 {err.quantile(0.9):.3f}°", flush=True)
        agent.fact_terms = tuple(args.fact_terms)
        agent.wta_fallback = not args.no_wta_fallback
        if args.init_from:
            loaded = agent.load_encoder(torch.load(args.init_from)["gnn"])
            print(f"  encoder initialised from {args.init_from}: {len(loaded)} tensors "
                  f"({sum(k.startswith('aux_head') for k in loaded)} of them the aux head)", flush=True)
        if modular:
            agent.mod = (args.sel_target, args.reg_target, args.scratch_head)
    else:
        agent = make_agent(CLEAR_ENV["n_balls"], args.hidden, OBS_DIMS[args.obs_transform])

    v0_select = None
    if args.val_v0:
        if not args.made_only:
            raise SystemExit("--val-v0 needs --made-only (it scores the all_pairs choice)")
        from exp16_wm.val_set import load as load_v0, top1_serial
        v0 = load_v0(args.val_v0)

        def v0_select(module: PairGNNPolicy) -> float:
            saved = (module.infer, module.refine_deg, module.refine_speed_abs)
            module.infer, module.refine_deg, module.refine_speed_abs = "all_pairs", ANG31, ABS7
            try:
                return top1_serial(module, v0)
            finally:
                module.infer, module.refine_deg, module.refine_speed_abs = saved
    t0 = time.time()
    fit = train_bc(agent, data, train, val, args.loss, args.epochs, args.batch, args.lr,
                   args.patience, train_seed(args.seed, args.init_seed), w=dim_weights(args.angle_scale_deg, args.speed_scale),
                   select=args.select, lr_schedule=args.lr_schedule, lr_final_frac=args.lr_final_frac,
                   use_pairs=args.pair_supervision or modular, selection=args.selection, use_extras=modular,
                   sims=sims, made2_weight=args.made2_weight, sims_batch=args.sims_batch,
                   aux=aux, aux_weight=args.geom_aux_weight, aux_only=args.aux_only,
                   train_plateau=args.train_plateau, train_plateau_tol=args.train_plateau_tol, made_only=args.made_only,
                   ckpt_path=ckpt_path, ckpt_every=args.ckpt_every, select_fn=v0_select, select_every=args.v0_every)
    if fit["stop_reason"]:
        print(f"  stopped: {fit['stop_reason']}", flush=True)
    if args.save_last and isinstance(agent, PairGNNPolicy):
        _save_last(agent, fit["last_state"], os.path.join(out_dir, "bc_model_last.pt"))
    if args.aux_only:                                              # no policy was trained: save the encoder + aux head and stop
        agent.save(os.path.join(out_dir, "bc_model.pt"))
        print(f"  aux-only: {fit['epochs_run']} epochs ({time.time()-t0:.0f}s), best val aux MSE {fit['best_val_loss']:.4f} @{fit['best_epoch']}", flush=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({"name": name, "train": {k: v for k, v in fit.items() if k not in ("history", "aux_history", "last_state")},
                       "val": None, "eval": None, "args": vars(args), "git_commit": code_version(),
                       "aux_mean": [float(x) for x in aux_mean], "aux_std": [float(x) for x in aux_std]}, f, indent=2)
        np.save(os.path.join(out_dir, "history.npy"), np.array(fit["history"]))
        np.save(os.path.join(out_dir, "aux_history.npy"), np.array(fit["aux_history"]))
        print(f"  saved → {out_dir}", flush=True)
        return
    vm = val_metrics(agent, data, val)
    if isinstance(agent, PairGNNPolicy):
        vm.update(oracle_pair_metrics(agent, data, val))
        if sims is not None:
            vm.update(made2_metrics(agent, data, val, sims))
    print(f"  trained {fit['epochs_run']} epochs ({time.time()-t0:.0f}s), best ({args.select}) @{fit['best_epoch']}"
          f" val loss {fit['best_val_loss']:.6f}  |  {json.dumps({k: round(v, 4) for k, v in vm.items()})}", flush=True)
    agent.save(os.path.join(out_dir, "bc_model.pt"))

    if args.made_only:                                       # no trained pair / selection head: choose with the made head over all pairs
        agent.infer, agent.refine_deg, agent.refine_speed_abs = "all_pairs", ANG31, ABS7
    ev = evaluate_policy(TransformedPolicy(agent, args.obs_transform), args.eval_seed, args.eval_games, game_env)
    print(f"  policy alone ({args.eval_games} games): {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in ev.items()})}",
          flush=True)
    ev_pen = None
    if isinstance(agent, PairGNNPolicy) and agent.has_scratch:   # the scratch head's effect, same model, penalty at inference only
        agent.scratch_lambda = args.scratch_penalty
        ev_pen = evaluate_policy(TransformedPolicy(agent, args.obs_transform), args.eval_seed, args.eval_games, game_env)
        agent.scratch_lambda = 0.0
        print(f"  policy alone with scratch penalty {args.scratch_penalty}: "
              f"{json.dumps({k: (round(v, 3) if v is not None else None) for k, v in ev_pen.items()})}", flush=True)
    ev_modes = None
    if args.made2_head and not args.made_only:              # same model, the made head used at inference
        ev_modes = {}
        for mode in ("verify", "refine", "verify_refine"):
            agent.infer = mode
            ev_modes[mode] = evaluate_policy(TransformedPolicy(agent, args.obs_transform), args.eval_seed, args.eval_games, game_env)
            print(f"  policy alone, infer={mode}: {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in ev_modes[mode].items()})}", flush=True)
        agent.infer = "select"
    results = {"name": name, "teacher": teacher, "loss": args.loss, "data": paths, "eval_infer_modes": ev_modes,
               "train_only_data": extra_paths if args.train_only_data else None, "eval_scratch_penalty": ev_pen,
               "n_shots": int(len(data["shot"])), "n_train": int(train.sum()), "n_val": int(val.sum()),
               "train": {k: v for k, v in fit.items() if k not in ("history", "aux_history", "last_state")}, "val": vm, "eval": ev,
               "args": vars(args), "git_commit": code_version()}
    if getattr(agent, "uncertainty_weights", False):      # log σ² of (angle, speed, selection) at the saved (best) epoch
        results["log_var"] = agent.log_var.detach().clamp(-3, 3).tolist()
        print(f"  learned log σ² (angle, speed, selection) at the best epoch: {results['log_var']}", flush=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(out_dir, "history.npy"), np.array(fit["history"]))
    if fit["aux_history"]:                                  # (epoch, train, val) auxiliary MSE per epoch
        np.save(os.path.join(out_dir, "aux_history.npy"), np.array(fit["aux_history"]))
    print(f"  saved → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
