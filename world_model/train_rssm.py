"""
world_model/train_rssm.py

Training script for R-SSM (Relational State-Space Model).

Features:
  - AdamW + CosineAnnealingLR (single decay cycle over max_epochs)
  - Kendall multi-task loss weighting (vel + type)
  - Bengio Scheduled Sampling: ss_prob 1.0 → 0.0 over ss_warmup epochs
  - Per-shot length-invariant RMSE evaluation (mean over shots, not events)
  - Class-weighted type CE loss
  - clip_grad_norm = 1.0
  - Early stopping with patience
  - Plotly 2x2 training dashboard

Usage:
    python world_model/train_rssm.py --out-dir world_model/results/rssm_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.ball_motion import FrictionParams, DEFAULT_FRICTION
from world_model.pure_physics import evolve_ball_motion, SLIDING
from world_model.rssm_model import (
    RSSMModel, H_DIM, N_TYPE, EVENT_BALL_BALL,
)
from world_model.rssm_dataset import ShotData, collect_dataset, load_dataset
from world_model.rssm_rollout import make_node, make_edge
from world_model.rssm_batch import iter_wavefronts, split_wavefront_by_type


TYPE_NAMES = {
    0: "ball_ball",
    1: "cue_linear",
    2: "cue_circ",
    3: "pocket",
    4: "stick_ball",
    5: "tgt_linear",
    6: "tgt_circ",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    n_balls        : int   = 1
    n_shots_train  : int   = 2000
    n_shots_val    : int   = 400
    seed_train     : int   = 0
    seed_val       : int   = 10000
    data_dir       : Optional[str] = None   # pkl dir from generate_rssm_data.py
    val_data_dir   : Optional[str] = None   # separate val pkl dir; None = on-the-fly
    delta_stats    : Optional[str] = None   # delta_stats.json for per-component scale weights
    # Model
    h_dim          : int   = H_DIM
    hidden         : list  = field(default_factory=lambda: [256, 256])
    # Optimiser
    lr             : float = 3e-4
    weight_decay   : float = 1e-4
    clip_grad      : float = 1.0
    # LR schedule (single CosineAnnealingLR cycle over max_epochs)
    # Training
    accum_steps    : int   = 32
    max_epochs     : int   = 500
    patience       : int   = 40
    eval_every     : int   = 10
    # Loss
    lam_type       : float = 0.3
    lam_pocket     : float = 0.5     # weight for per-ball pocket BCE loss
    use_kendall    : bool  = True
    focal_gamma    : float = 2.0     # focal loss gamma for type CE (0 = standard CE)
    # Bengio Scheduled Sampling
    ss_start       : float = 1.0
    ss_end         : float = 0.0
    ss_warmup      : int   = 200
    # Output
    out_dir        : str   = "world_model/results/rssm_v1"
    device         : str   = "cpu"


# ── Data helpers ──────────────────────────────────────────────────────────────

def compute_type_class_weights(
    shots   : list[ShotData],
    device  : torch.device,
    max_w   : float = 5.0,
) -> torch.Tensor:
    counts = torch.zeros(N_TYPE)
    for shot in shots:
        for t in shot.gt_types_i:
            if t is not None:
                counts[t] += 1
        for t in shot.gt_types_j:
            if t is not None:
                counts[t] += 1
    mask    = counts > 0
    weights = torch.zeros(N_TYPE)
    weights[mask] = 1.0 / counts[mask]
    weights[mask] = (weights[mask] / weights[mask].sum() * mask.sum().float()).clamp(max=max_w)
    return weights.to(device)


def compute_vel_type_weights(
    shots   : list[ShotData],
    device  : torch.device,
    max_w   : float = 5.0,
) -> torch.Tensor:
    """
    Inverse-frequency weights per event type for velocity MSE (N_TYPE,).

    Types absent from data get weight 0 (not clamped to 1, which would
    poison the mean normalization for all other types).
    Weights are capped at max_w to prevent extreme up-weighting of rare types.
    """
    counts = torch.zeros(N_TYPE)
    for shot in shots:
        for ev in shot.event_steps:
            counts[ev.event_type] += 1

    mask    = counts > 0
    weights = torch.zeros(N_TYPE)
    weights[mask] = 1.0 / counts[mask]
    weights[mask] = weights[mask] / weights[mask].mean()   # normalize over present types
    weights[mask] = weights[mask].clamp(max=max_w)
    return weights.to(device)


def load_delta_scale_weights(stats_path: str, device: torch.device) -> torch.Tensor:
    """
    Load per-component 1/std² weights from delta_stats.json.
    Returns (5,) tensor — equalizes [dvx,dvy,dwx,dwy,dwz] scale in MSE.
    """
    stats = json.load(open(stats_path))
    std   = torch.tensor(stats["global_std"], dtype=torch.float32)
    std   = std.clamp(min=1e-3)
    return (1.0 / (std ** 2)).to(device)


def _focal_cross_entropy(
    logits  : torch.Tensor,   # (1, N_TYPE)
    targets : torch.Tensor,   # (1,) long
    weight  : Optional[torch.Tensor],
    gamma   : float,
) -> torch.Tensor:
    log_p  = F.log_softmax(logits, dim=-1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt     = log_pt.exp()
    focal  = (1.0 - pt) ** gamma * (-log_pt)
    if weight is not None:
        focal = focal * weight[targets]
        return focal.sum() / (weight[targets].sum() + 1e-8)
    return focal.mean()


# ── Loss ──────────────────────────────────────────────────────────────────────

def _pick_node_i(
    ev, k: int, shot: ShotData,
    pred_rvws: dict[int, np.ndarray],
    ss_prob: float, device: torch.device,
) -> tuple[bool, np.ndarray, torch.Tensor]:
    """Decide GT-vs-predicted rvw for ball_i and build/reuse its node tensor."""
    bi     = ev.ball_i
    use_gt = (random.random() < ss_prob) or (bi not in pred_rvws)
    rvw    = shot.raw_rvws_i[k] if use_gt else pred_rvws[bi]
    # ② reuse pre-built node when GT (None-guarded for new data without stored tensors)
    if use_gt and ev.node_i is not None:
        node = ev.node_i.to(device)
    else:
        node = make_node(rvw, ev.event_type).to(device)
    return use_gt, rvw, node


def _pick_node_j(
    ev, k: int, shot: ShotData,
    pred_rvws: dict[int, np.ndarray],
    ss_prob: float, device: torch.device,
) -> tuple[bool, np.ndarray, torch.Tensor]:
    """Decide GT-vs-predicted rvw for ball_j and build/reuse its node tensor."""
    bj        = ev.ball_j
    raw_rvw_j = shot.raw_rvws_j[k]
    use_gt    = (random.random() < ss_prob) or (bj not in pred_rvws) or (raw_rvw_j is None)
    rvw       = raw_rvw_j if use_gt else pred_rvws[bj]
    if use_gt and ev.node_j is not None:
        node = ev.node_j.to(device)
    else:
        node = make_node(rvw, ev.event_type).to(device)
    return use_gt, rvw, node


def _pick_edge(
    use_gt_i: bool, use_gt_j: bool, ev,
    rvw_i: np.ndarray, rvw_j: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Reuse the pre-built edge tensor when both sides are GT, else recompute."""
    if use_gt_i and use_gt_j and ev.edge is not None:
        return ev.edge.to(device)
    return make_edge(rvw_i, rvw_j, ev.normal.numpy()).to(device)


def _advance_rvw(
    raw_rvw: np.ndarray,
    delta  : torch.Tensor,
    dt     : float,
    ss_prob: float,
    params : FrictionParams,
) -> np.ndarray:
    """
    Apply predicted Δvel/Δavel to raw_rvw and roll forward by dt for SS bookkeeping.

    Uses a fast linear approximation during warmup (ss_prob > 0.1) and full
    physics only near free-running (ss_prob <= 0.1), since SS's predicted
    rvw is only used as a rough teacher-forcing substitute, not a training target.
    """
    rvw = raw_rvw.copy()
    rvw[1, :2] += delta[:2].detach().cpu().numpy()
    rvw[2]     += delta[2:].detach().cpu().numpy()
    if dt > 1e-9:
        if ss_prob > 0.1:
            rvw[0, :2] += rvw[1, :2] * dt * 0.7
            spd = float(np.linalg.norm(rvw[1, :2]))
            if spd > 1e-4:
                decel = params.u_r * params.g
                scale = max(0.0, 1.0 - decel * dt / spd)
                rvw[1, :2] *= scale
        else:
            rvw, _ = evolve_ball_motion(
                SLIDING, rvw,
                R=params.R, m=params.m,
                u_s=params.u_s, u_sp=params.u_sp,
                u_r=params.u_r, g=params.g,
                t=dt,
            )
    return rvw


def _vel_loss_term(
    delta_i          : torch.Tensor,
    gt_i             : torch.Tensor,
    delta_j          : Optional[torch.Tensor],
    gt_j             : Optional[torch.Tensor],
    ev_type          : int,
    vel_type_weights : Optional[torch.Tensor],
    delta_scale_w    : Optional[torch.Tensor],
) -> torch.Tensor:
    """Scale-weighted + type-freq-weighted velocity MSE for one event."""
    vw = vel_type_weights[ev_type] if vel_type_weights is not None else 1.0
    if delta_scale_w is not None:
        vel_err = (((delta_i - gt_i) ** 2) * delta_scale_w).mean() * vw
    else:
        vel_err = F.mse_loss(delta_i, gt_i) * vw
    if delta_j is not None and gt_j is not None:
        if delta_scale_w is not None:
            vel_err = vel_err + (((delta_j - gt_j) ** 2) * delta_scale_w).mean() * vw
        else:
            vel_err = vel_err + F.mse_loss(delta_j, gt_j) * vw
    return vel_err


def _type_loss_term(
    type_i       : torch.Tensor,
    gt_type_i    : Optional[int],
    type_j       : Optional[torch.Tensor],
    gt_type_j    : Optional[int],
    device       : torch.device,
    type_weights : Optional[torch.Tensor],
    focal_gamma  : float,
) -> torch.Tensor:
    """Focal (or standard) CE type loss for one event, summed over ball_i/ball_j."""
    loss = torch.tensor(0.0, device=device)
    if gt_type_i is not None:
        tgt_i = torch.tensor([gt_type_i], device=device)
        if focal_gamma > 0.0:
            loss = loss + _focal_cross_entropy(type_i.unsqueeze(0), tgt_i, type_weights, focal_gamma)
        else:
            loss = loss + F.cross_entropy(type_i.unsqueeze(0), tgt_i, weight=type_weights)
    if type_j is not None and gt_type_j is not None:
        tgt_j = torch.tensor([gt_type_j], device=device)
        if focal_gamma > 0.0:
            loss = loss + _focal_cross_entropy(type_j.unsqueeze(0), tgt_j, type_weights, focal_gamma)
        else:
            loss = loss + F.cross_entropy(type_j.unsqueeze(0), tgt_j, weight=type_weights)
    return loss


def compute_shot_ss_loss(
    model             : RSSMModel,
    shot              : ShotData,
    ss_prob           : float,
    params            : FrictionParams,
    device            : torch.device,
    type_weights      : Optional[torch.Tensor] = None,
    vel_type_weights  : Optional[torch.Tensor] = None,
    delta_scale_w     : Optional[torch.Tensor] = None,
    focal_gamma       : float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Single-shot loss with Bengio Scheduled Sampling on node features.

    ss_prob=1.0 → teacher forcing  (always GT pre-collision rvw)
    ss_prob=0.0 → free running     (always predicted pre-collision rvw)

    Returns (vel_loss_sum, type_loss_sum, pocket_loss_sum, n_events)
    """
    h = model.init_hidden(shot.n_balls, device)

    # Predicted rvw per ball_idx, updated after each event for SS
    pred_rvws: dict[int, np.ndarray] = {}

    vel_loss_sum    = torch.tensor(0.0, device=device)
    type_loss_sum   = torch.tensor(0.0, device=device)
    pocket_loss_sum = torch.tensor(0.0, device=device)
    n_events = 0

    for k, ev in enumerate(shot.event_steps):
        ev_type = ev.event_type
        bi      = ev.ball_i
        bj      = ev.ball_j

        # ── Choose GT or predicted rvw ────────────────────────────────────────
        use_gt_i, rvw_i, node_i = _pick_node_i(ev, k, shot, pred_rvws, ss_prob, device)

        node_j = edge = rvw_j = None
        use_gt_j = False
        if bj is not None:
            use_gt_j, rvw_j, node_j = _pick_node_j(ev, k, shot, pred_rvws, ss_prob, device)
            edge = _pick_edge(use_gt_i, use_gt_j, ev, rvw_i, rvw_j, device)

        # ── Model step ────────────────────────────────────────────────────────
        if ev_type == EVENT_BALL_BALL and bj is not None:
            h, delta_i, delta_j, type_i, type_j = model.step_ball_ball(
                h, bi, bj, node_i, node_j, edge,
            )
        else:
            h, delta_i, type_i = model.step_single(
                h, bi, node_i, ev.normal.to(device),
            )
            delta_j = type_j = None

        # ── Velocity loss (scale-weighted + type-freq-weighted) ──────────────
        gt_i = shot.gt_deltas_i[k].to(device)
        gt_j = shot.gt_deltas_j[k].to(device) if shot.gt_deltas_j[k] is not None else None
        vel_loss_sum = vel_loss_sum + _vel_loss_term(
            delta_i, gt_i, delta_j, gt_j, ev_type, vel_type_weights, delta_scale_w,
        )

        # ── Type loss (focal CE) ───────────────────────────────────────────────
        type_loss_sum = type_loss_sum + _type_loss_term(
            type_i, shot.gt_types_i[k], type_j, shot.gt_types_j[k],
            device, type_weights, focal_gamma,
        )

        # ── Pocket prediction loss (per ball, per event) ─────────────────────
        # ③ batched BCE — one (n_balls,) call instead of n_balls 1-element calls
        pocket_probs   = model.predict_pocket(h)   # (n_balls,)
        pocket_targets = torch.tensor(
            [float(shot.will_pocket.get(bi, False)) for bi in range(shot.n_balls)],
            device=device,
        )
        pocket_loss_sum = pocket_loss_sum + F.binary_cross_entropy(
            pocket_probs, pocket_targets, reduction="sum"
        )

        n_events += 1

        # ── Update predicted rvw ──────────────────────────────────────────────
        # Skip when full teacher forcing (pred_rvws never used).
        if ss_prob < 1.0:
            dt = shot.dt_to_next[k]
            pred_rvws[bi] = _advance_rvw(rvw_i, delta_i, dt, ss_prob, params)
            if bj is not None and delta_j is not None and rvw_j is not None:
                pred_rvws[bj] = _advance_rvw(rvw_j, delta_j, dt, ss_prob, params)

    return vel_loss_sum, type_loss_sum, pocket_loss_sum, n_events


def compute_batch_ss_loss(
    model             : RSSMModel,
    shots             : list[ShotData],
    ss_prob           : float,
    params            : FrictionParams,
    device            : torch.device,
    type_weights      : Optional[torch.Tensor] = None,
    vel_type_weights  : Optional[torch.Tensor] = None,
    delta_scale_w     : Optional[torch.Tensor] = None,
    focal_gamma       : float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Batched version of compute_shot_ss_loss: runs multiple shots' event
    sequences concurrently via wavefront scheduling (world_model.rssm_batch),
    so neural-net forward calls are batched across shots instead of B=1.

    All rolling state (h, pred_rvws, loss accumulators) is kept as Python
    lists — list-element reassignment, never in-place tensor mutation — the
    same autograd-safe pattern as RSSMModel.init_hidden's per-ball h list.

    Returns a list of (vel_loss_sum, type_loss_sum, pocket_loss_sum, n_events)
    tuples, one per input shot, in the same order as `shots` — the exact
    per-shot signature of compute_shot_ss_loss, so callers can process a
    batch result the same way they process a single-shot result.
    """
    if not shots:
        return []

    n_balls_set = {s.n_balls for s in shots}
    assert len(n_balls_set) == 1, (
        f"compute_batch_ss_loss requires uniform n_balls across the batch, got {n_balls_set}"
    )
    n_balls = n_balls_set.pop()

    h_list        : list[list[torch.Tensor]]      = [model.init_hidden(n_balls, device) for _ in shots]
    pred_rvws_list: list[dict[int, np.ndarray]]    = [{} for _ in shots]

    vel_sum_list    = [torch.tensor(0.0, device=device) for _ in shots]
    type_sum_list   = [torch.tensor(0.0, device=device) for _ in shots]
    pocket_sum_list = [torch.tensor(0.0, device=device) for _ in shots]
    n_ev_list       = [0 for _ in shots]

    for wavefront in iter_wavefronts(shots):
        ball_ball_items, single_items = split_wavefront_by_type(shots, wavefront)

        if ball_ball_items:
            _run_ball_ball_batch(
                model, shots, h_list, pred_rvws_list,
                vel_sum_list, type_sum_list, n_ev_list,
                ball_ball_items, ss_prob, params, device,
                type_weights, vel_type_weights, delta_scale_w, focal_gamma,
            )

        if single_items:
            _run_single_batch(
                model, shots, h_list, pred_rvws_list,
                vel_sum_list, type_sum_list, n_ev_list,
                single_items, ss_prob, params, device,
                type_weights, vel_type_weights, delta_scale_w, focal_gamma,
            )

        # ── Pocket loss: once per active shot in this wavefront ──────────────
        # (matches compute_shot_ss_loss calling predict_pocket once per event),
        # batched across all active shots regardless of ball_ball/single type.
        active_shots = sorted({s for s, _ in wavefront})
        h_batch = torch.stack(
            [torch.stack(h_list[s], dim=0) for s in active_shots], dim=0,
        )   # (S', n_balls, h_dim)
        pocket_probs = model.predict_pocket(h_batch)   # (S', n_balls)
        for row, s in enumerate(active_shots):
            pocket_targets = torch.tensor(
                [float(shots[s].will_pocket.get(bi, False)) for bi in range(n_balls)],
                device=device,
            )
            pocket_sum_list[s] = pocket_sum_list[s] + F.binary_cross_entropy(
                pocket_probs[row], pocket_targets, reduction="sum"
            )

    return [
        (vel_sum_list[s], type_sum_list[s], pocket_sum_list[s], n_ev_list[s])
        for s in range(len(shots))
    ]


def _run_ball_ball_batch(
    model            : RSSMModel,
    shots            : list[ShotData],
    h_list           : list[list[torch.Tensor]],
    pred_rvws_list   : list[dict[int, np.ndarray]],
    vel_sum_list     : list[torch.Tensor],
    type_sum_list    : list[torch.Tensor],
    n_ev_list        : list[int],
    items            : list[tuple[int, int]],   # (shot_idx, event_idx)
    ss_prob          : float,
    params           : FrictionParams,
    device           : torch.device,
    type_weights     : Optional[torch.Tensor],
    vel_type_weights : Optional[torch.Tensor],
    delta_scale_w    : Optional[torch.Tensor],
    focal_gamma      : float,
) -> None:
    """Process one wavefront's ball_ball items as a single batched model call."""
    h_i_list, h_j_list = [], []
    node_i_list, node_j_list, edge_list = [], [], []
    meta = []   # (s, k, ev, bi, bj, rvw_i, rvw_j, use_gt_i, use_gt_j)

    for s, k in items:
        shot = shots[s]
        ev   = shot.event_steps[k]
        bi, bj = ev.ball_i, ev.ball_j
        pred_rvws = pred_rvws_list[s]

        use_gt_i, rvw_i, node_i = _pick_node_i(ev, k, shot, pred_rvws, ss_prob, device)
        use_gt_j, rvw_j, node_j = _pick_node_j(ev, k, shot, pred_rvws, ss_prob, device)
        edge = _pick_edge(use_gt_i, use_gt_j, ev, rvw_i, rvw_j, device)

        h_i_list.append(h_list[s][bi])
        h_j_list.append(h_list[s][bj])
        node_i_list.append(node_i)
        node_j_list.append(node_j)
        edge_list.append(edge)
        meta.append((s, k, ev, bi, bj, rvw_i, rvw_j))

    h_i_new, h_j_new, delta_i, delta_j, type_i, type_j = model.step_ball_ball_batch(
        torch.stack(h_i_list, dim=0),
        torch.stack(h_j_list, dim=0),
        torch.stack(node_i_list, dim=0),
        torch.stack(node_j_list, dim=0),
        torch.stack(edge_list, dim=0),
    )

    for idx, (s, k, ev, bi, bj, rvw_i, rvw_j) in enumerate(meta):
        shot = shots[s]
        # List reassignment — never in-place tensor mutation (autograd-safe).
        h_list[s][bi] = h_i_new[idx]
        h_list[s][bj] = h_j_new[idx]

        gt_i = shot.gt_deltas_i[k].to(device)
        gt_j = shot.gt_deltas_j[k].to(device) if shot.gt_deltas_j[k] is not None else None
        vel_sum_list[s] = vel_sum_list[s] + _vel_loss_term(
            delta_i[idx], gt_i, delta_j[idx], gt_j, ev.event_type,
            vel_type_weights, delta_scale_w,
        )
        type_sum_list[s] = type_sum_list[s] + _type_loss_term(
            type_i[idx], shot.gt_types_i[k], type_j[idx], shot.gt_types_j[k],
            device, type_weights, focal_gamma,
        )
        n_ev_list[s] += 1

        if ss_prob < 1.0:
            dt = shot.dt_to_next[k]
            pred_rvws_list[s][bi] = _advance_rvw(rvw_i, delta_i[idx], dt, ss_prob, params)
            pred_rvws_list[s][bj] = _advance_rvw(rvw_j, delta_j[idx], dt, ss_prob, params)


def _run_single_batch(
    model            : RSSMModel,
    shots            : list[ShotData],
    h_list           : list[list[torch.Tensor]],
    pred_rvws_list   : list[dict[int, np.ndarray]],
    vel_sum_list     : list[torch.Tensor],
    type_sum_list    : list[torch.Tensor],
    n_ev_list        : list[int],
    items            : list[tuple[int, int]],   # (shot_idx, event_idx)
    ss_prob          : float,
    params           : FrictionParams,
    device           : torch.device,
    type_weights     : Optional[torch.Tensor],
    vel_type_weights : Optional[torch.Tensor],
    delta_scale_w    : Optional[torch.Tensor],
    focal_gamma      : float,
) -> None:
    """Process one wavefront's single-ball items as a single batched model call."""
    h_i_list, node_i_list, normal_list = [], [], []
    meta = []   # (s, k, ev, bi, rvw_i)

    for s, k in items:
        shot = shots[s]
        ev   = shot.event_steps[k]
        bi   = ev.ball_i
        pred_rvws = pred_rvws_list[s]

        _, rvw_i, node_i = _pick_node_i(ev, k, shot, pred_rvws, ss_prob, device)

        h_i_list.append(h_list[s][bi])
        node_i_list.append(node_i)
        normal_list.append(ev.normal.to(device))
        meta.append((s, k, ev, bi, rvw_i))

    h_i_new, delta_i, type_i = model.step_single_batch(
        torch.stack(h_i_list, dim=0),
        torch.stack(node_i_list, dim=0),
        torch.stack(normal_list, dim=0),
    )

    for idx, (s, k, ev, bi, rvw_i) in enumerate(meta):
        shot = shots[s]
        h_list[s][bi] = h_i_new[idx]

        gt_i = shot.gt_deltas_i[k].to(device)
        vel_sum_list[s] = vel_sum_list[s] + _vel_loss_term(
            delta_i[idx], gt_i, None, None, ev.event_type,
            vel_type_weights, delta_scale_w,
        )
        type_sum_list[s] = type_sum_list[s] + _type_loss_term(
            type_i[idx], shot.gt_types_i[k], None, None,
            device, type_weights, focal_gamma,
        )
        n_ev_list[s] += 1

        if ss_prob < 1.0:
            dt = shot.dt_to_next[k]
            pred_rvws_list[s][bi] = _advance_rvw(rvw_i, delta_i[idx], dt, ss_prob, params)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model  : RSSMModel,
    shots  : list[ShotData],
    device : torch.device,
) -> tuple[float, dict[str, float], float, float]:
    """
    Length-invariant evaluation: per-shot RMSE → mean across shots.

    pocket_acc is measured at each ball's first-touch h (the earliest point
    the model could possibly know about that ball), not the final h — the
    final h is a tautology since a ball's last recorded event is the pocket
    event itself iff it gets pocketed.

    Returns (val_rmse, per_type_rmse_dict, type_acc, pocket_acc)
    """
    model.eval()
    shot_rmses  : list[float]           = []
    type_mses   : dict[int, list[float]] = defaultdict(list)
    correct = total = 0
    pocket_correct = pocket_total = 0

    with torch.no_grad():
        for shot in shots:
            h = model.init_hidden(shot.n_balls, device)
            event_mses: list[float] = []
            # h snapshot at each ball's OWN first event — earliest possible
            # lookahead point. Using final h is a tautology (a ball's last
            # recorded event IS the pocket event iff it gets pocketed, since
            # pocketed balls are removed from the sim), so it trivially
            # "predicts" pocket by re-reading the current event's one-hot type.
            first_touch_h: dict[int, torch.Tensor] = {}

            for k, ev in enumerate(shot.event_steps):
                # ⑤ node/edge may be None in new pkl data — recompute from raw_rvws
                rvw_i  = shot.raw_rvws_i[k]
                node_i = (ev.node_i if ev.node_i is not None
                          else make_node(rvw_i, ev.event_type)).to(device)
                rvw_j  = shot.raw_rvws_j[k]
                if rvw_j is not None:
                    node_j = (ev.node_j if ev.node_j is not None
                              else make_node(rvw_j, ev.event_type)).to(device)
                    edge   = (ev.edge if ev.edge is not None
                              else make_edge(rvw_i, rvw_j, ev.normal.numpy())).to(device)
                else:
                    node_j = None
                    edge   = None

                if ev.event_type == EVENT_BALL_BALL and ev.ball_j is not None:
                    h, delta_i, delta_j, type_i, type_j = model.step_ball_ball(
                        h, ev.ball_i, ev.ball_j, node_i, node_j, edge,
                    )
                else:
                    h, delta_i, type_i = model.step_single(
                        h, ev.ball_i, node_i, ev.normal.to(device),
                    )
                    delta_j = type_j = None

                gt_i = shot.gt_deltas_i[k].to(device)
                mse  = F.mse_loss(delta_i, gt_i).item()
                if delta_j is not None and shot.gt_deltas_j[k] is not None:
                    gt_j = shot.gt_deltas_j[k].to(device)
                    mse  = (mse + F.mse_loss(delta_j, gt_j).item()) / 2.0

                event_mses.append(mse)
                type_mses[ev.event_type].append(mse)

                if shot.gt_types_i[k] is not None:
                    correct += int(type_i.argmax().item() == shot.gt_types_i[k])
                    total   += 1
                if type_j is not None and shot.gt_types_j[k] is not None:
                    correct += int(type_j.argmax().item() == shot.gt_types_j[k])
                    total   += 1

                if ev.ball_i not in first_touch_h:
                    first_touch_h[ev.ball_i] = h[ev.ball_i].clone()
                if ev.ball_j is not None and ev.ball_j not in first_touch_h:
                    first_touch_h[ev.ball_j] = h[ev.ball_j].clone()

            if event_mses:
                shot_rmses.append(float(np.sqrt(np.mean(event_mses))))

            # Pocket accuracy at earliest lookahead point (each ball's first touch)
            for bi_idx, h_i in first_touch_h.items():
                gt_pock   = shot.will_pocket.get(bi_idx, False)
                pred_pock = model.predict_pocket(h_i.unsqueeze(0)).item() >= 0.5
                pocket_correct += int(pred_pock == gt_pock)
                pocket_total   += 1

    val_rmse = float(np.mean(shot_rmses)) if shot_rmses else float("inf")
    per_type = {
        TYPE_NAMES.get(t, str(t)): float(np.sqrt(np.mean(mses)))
        for t, mses in type_mses.items()
    }
    type_acc   = correct / total if total > 0 else 0.0
    pocket_acc = pocket_correct / pocket_total if pocket_total > 0 else 0.0

    model.train()
    return val_rmse, per_type, type_acc, pocket_acc


# ── Charts ────────────────────────────────────────────────────────────────────

def save_charts(history: dict, out_dir: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping charts")
        return

    epochs = history["epochs"]
    if not epochs:
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Train / Val RMSE", "LR Schedule",
                        "Per-Type Velocity RMSE", "Type Accuracy"],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(x=epochs, y=history["train_rmse"],
                             name="train", line=dict(color="#1f77b4")), row=1, col=1)
    ep_v = [e for e, v in zip(epochs, history["val_rmse"]) if v is not None]
    vl   = [v for v in history["val_rmse"] if v is not None]
    if vl:
        fig.add_trace(go.Scatter(x=ep_v, y=vl,
                                 name="val", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.update_yaxes(title_text="RMSE (m/s)", row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=history["lr"],
                             name="lr", line=dict(color="#2ca02c")), row=1, col=2)
    fig.update_yaxes(type="log", title_text="LR", row=1, col=2)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (name, rmse_list) in enumerate(history["type_rmse"].items()):
        ep_s = [e for e, v in zip(epochs, rmse_list) if v is not None]
        vs   = [v for v in rmse_list if v is not None]
        if vs:
            fig.add_trace(go.Scatter(x=ep_s, y=vs, name=name,
                                     line=dict(color=colors[idx % len(colors)])),
                          row=2, col=1)
    fig.update_yaxes(title_text="RMSE (m/s)", row=2, col=1)

    ea = [e for e, v in zip(epochs, history["type_acc"]) if v is not None]
    va = [v * 100 for v in history["type_acc"] if v is not None]
    if va:
        fig.add_trace(go.Scatter(x=ea, y=va, name="type_acc%",
                                 line=dict(color="#d62728")), row=2, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.update_layout(height=700, title_text="R-SSM Training Dashboard",
                      legend=dict(groupclick="toggleitem"))
    path = os.path.join(out_dir, "training_dashboard.html")
    fig.write_html(path)
    print(f"Charts → {path}")


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    params = DEFAULT_FRICTION

    # ── Data ──────────────────────────────────────────────────────────────────
    if cfg.data_dir is not None:
        print(f"Loading train shots from {cfg.data_dir} (max={cfg.n_shots_train})…")
        train_shots = load_dataset(cfg.data_dir, max_shots=cfg.n_shots_train)
    else:
        print(f"Collecting {cfg.n_shots_train} train shots…")
        train_shots = collect_dataset(cfg.n_shots_train, cfg.n_balls, cfg.seed_train)

    val_src = cfg.val_data_dir or cfg.data_dir
    if val_src is not None:
        print(f"Loading val shots from {val_src} (max={cfg.n_shots_val})…")
        # val shots: skip the first n_shots_train to avoid overlap
        all_val = load_dataset(val_src)
        val_shots = all_val[cfg.n_shots_train: cfg.n_shots_train + cfg.n_shots_val]
        if not val_shots:
            val_shots = all_val[-cfg.n_shots_val:]
    else:
        print(f"Collecting {cfg.n_shots_val} val shots…")
        val_shots = collect_dataset(cfg.n_shots_val, cfg.n_balls, cfg.seed_val)

    print(f"Train: {len(train_shots)} shots  Val: {len(val_shots)} shots")

    type_weights     = compute_type_class_weights(train_shots, device)
    vel_type_weights = compute_vel_type_weights(train_shots, device)
    delta_scale_w    = None
    if cfg.delta_stats is not None:
        delta_scale_w = load_delta_scale_weights(cfg.delta_stats, device)
        print(f"Delta scale weights: {delta_scale_w.cpu().numpy().round(4)}")
    else:
        print("Delta scale weights: none (raw MSE)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = RSSMModel(h_dim=cfg.h_dim, hidden=cfg.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    log_var_vel  = nn.Parameter(torch.zeros(1, device=device))
    log_var_type = nn.Parameter(torch.zeros(1, device=device))
    all_params   = list(model.parameters()) + (
        [log_var_vel, log_var_type] if cfg.use_kendall else []
    )

    opt   = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01,
    )

    # ── History ───────────────────────────────────────────────────────────────
    history: dict = {
        "epochs":      [],
        "train_rmse":  [],
        "val_rmse":    [],
        "lr":          [],
        "type_rmse":   defaultdict(list),
        "type_acc":    [],
        "pocket_acc":  [],
    }
    best_val = float("inf")
    stall    = 0

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.max_epochs + 1):
        cur_lr = opt.param_groups[0]["lr"]

        if cfg.ss_warmup > 0:
            t_frac  = min(1.0, (epoch - 1) / cfg.ss_warmup)
            ss_prob = cfg.ss_start + (cfg.ss_end - cfg.ss_start) * t_frac
        else:
            ss_prob = cfg.ss_end

        model.train()
        random.shuffle(train_shots)

        train_rmses: list[float] = []
        opt.zero_grad()

        for shot_idx, shot in enumerate(train_shots):
            if not shot.event_steps:
                continue

            vel_loss, type_loss, pocket_loss, n_ev = compute_shot_ss_loss(
                model, shot, ss_prob, params, device,
                type_weights     = type_weights,
                vel_type_weights = vel_type_weights,
                delta_scale_w    = delta_scale_w,
                focal_gamma      = cfg.focal_gamma,
            )
            if n_ev == 0:
                continue

            vel_mean    = vel_loss    / n_ev
            type_mean   = type_loss   / n_ev
            pocket_mean = pocket_loss / (n_ev * shot.n_balls)

            if cfg.use_kendall:
                lv = log_var_vel.clamp(-3, 2)
                lt = log_var_type.clamp(-3, 2)
                loss = (torch.exp(-lv) * vel_mean + lv
                        + torch.exp(-lt) * type_mean + lt
                        + cfg.lam_pocket * pocket_mean)
            else:
                loss = vel_mean + cfg.lam_type * type_mean + cfg.lam_pocket * pocket_mean

            (loss / cfg.accum_steps).backward()
            train_rmses.append(float(torch.sqrt(vel_mean).detach().item()))

            if (shot_idx + 1) % cfg.accum_steps == 0:
                nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)
                opt.step()
                opt.zero_grad()

        # Flush remaining accumulated grads
        nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)
        opt.step()
        opt.zero_grad()

        sched.step()

        train_rmse = float(np.mean(train_rmses)) if train_rmses else float("nan")

        # ── Eval ──────────────────────────────────────────────────────────────
        do_eval    = (epoch % cfg.eval_every == 0 or epoch == 1)
        val_rmse   = per_type = type_acc = pocket_acc = None

        if do_eval:
            val_rmse, per_type, type_acc, pocket_acc = evaluate(model, val_shots, device)

        # ── Checkpoint ────────────────────────────────────────────────────────
        saved = ""
        if val_rmse is not None:
            if val_rmse < best_val - 1e-5:
                best_val = val_rmse
                stall    = 0
                torch.save(
                    {"state": model.state_dict(), "epoch": epoch, "val_rmse": val_rmse},
                    out_dir / "best.pt",
                )
                saved = " *"
            else:
                stall += 1

        # ── Print ─────────────────────────────────────────────────────────────
        kw_str  = ""
        if cfg.use_kendall:
            kw_v   = float(torch.exp(-log_var_vel).item())
            kw_t   = float(torch.exp(-log_var_type).item())
            kw_str = f"  kw=[{kw_v:.2f},{kw_t:.2f}]"

        val_str    = f"  val={val_rmse:.5f}" if val_rmse is not None else ""
        acc_str    = f"  type_acc={type_acc:.3f}" if type_acc is not None else ""
        pocket_str = f"  pock_acc={pocket_acc:.3f}" if pocket_acc is not None else ""
        ts      = datetime.now().strftime("%H:%M:%S")
        line    = (
            f"[{ts}] Epoch {epoch:4d}  [lr={cur_lr:.2e}  ss={ss_prob:.3f}]"
            f"  train={train_rmse:.5f}{val_str}{acc_str}{pocket_str}{kw_str}{saved}"
        )
        if do_eval and per_type:
            line += "  [" + "  ".join(
                f"{n}={v:.4f}" for n, v in sorted(per_type.items())
            ) + "]"
        print(line)

        # History
        history["epochs"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(cur_lr)
        history["type_acc"].append(type_acc)
        history["pocket_acc"].append(pocket_acc)
        for name in TYPE_NAMES.values():
            history["type_rmse"][name].append(per_type.get(name) if per_type else None)

        if stall >= cfg.patience:
            print(f"\nEarly stop — best val_rmse={best_val:.5f}")
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({"state": model.state_dict(), "epoch": epoch}, out_dir / "last.pt")
    json.dump(
        {"best_val_rmse": best_val, "epochs_run": epoch},
        open(out_dir / "result.json", "w"), indent=2,
    )
    save_charts(dict(history), str(out_dir))
    print(f"Done → {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train R-SSM world model")
    p.add_argument("--n-balls",       type=int,   default=1)
    p.add_argument("--n-shots-train", type=int,   default=2000)
    p.add_argument("--n-shots-val",   type=int,   default=400)
    p.add_argument("--data-dir",      default=None,
                   help="pkl data dir (generate_rssm_data.py). Overrides on-the-fly sim.")
    p.add_argument("--val-data-dir",  default=None,
                   help="separate val pkl dir. If omitted, sliced from --data-dir.")
    p.add_argument("--delta-stats",   default=None,
                   help="delta_stats.json for per-component scale weights in MSE.")
    p.add_argument("--focal-gamma",   type=float, default=2.0,
                   help="Focal loss gamma for type CE. 0=standard CE.")
    p.add_argument("--h-dim",         type=int,   default=H_DIM)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--max-epochs",    type=int,   default=500)
    p.add_argument("--patience",      type=int,   default=40)
    p.add_argument("--eval-every",    type=int,   default=10)
    p.add_argument("--accum-steps",   type=int,   default=32)
    p.add_argument("--lam-type",      type=float, default=0.3)
    p.add_argument("--lam-pocket",    type=float, default=0.5,
                   help="Weight for per-ball pocket BCE loss.")
    p.add_argument("--no-kendall",    action="store_true")
    p.add_argument("--ss-start",      type=float, default=1.0)
    p.add_argument("--ss-end",        type=float, default=0.0)
    p.add_argument("--ss-warmup",     type=int,   default=200)
    p.add_argument("--out-dir",       default="world_model/results/rssm_v1")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = TrainConfig(
        n_balls       = args.n_balls,
        n_shots_train = args.n_shots_train,
        n_shots_val   = args.n_shots_val,
        data_dir      = args.data_dir,
        val_data_dir  = args.val_data_dir,
        delta_stats   = args.delta_stats,
        focal_gamma   = args.focal_gamma,
        h_dim         = args.h_dim,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        max_epochs    = args.max_epochs,
        patience      = args.patience,
        eval_every    = args.eval_every,
        accum_steps   = args.accum_steps,
        lam_type      = args.lam_type,
        lam_pocket    = args.lam_pocket,
        use_kendall   = not args.no_kendall,
        ss_start      = args.ss_start,
        ss_end        = args.ss_end,
        ss_warmup     = args.ss_warmup,
        out_dir       = args.out_dir,
        device        = args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
