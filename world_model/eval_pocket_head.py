"""
world_model/eval_pocket_head.py

Diagnostic: how good is R-SSM's per-ball pocket prediction head, really?

train_rssm.py's evaluate() reports a single pocket_acc number (~0.85 for
rssm_v4). Accuracy alone is meaningless under class imbalance — if most
balls in most shots are never pocketed, "always predict False" already
scores high. This script recomputes the same first-touch-h predictions
but reports base rate, precision/recall/F1, confusion matrix, and a
threshold sweep so we can tell signal from a majority-class default.

Usage:
    python world_model/eval_pocket_head.py --ckpt world_model/results/rssm_v4/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.rssm_model import RSSMModel
from world_model.rssm_dataset import load_dataset


EVENT_POCKET = 3

_TABLE = pt.Table.default()
POCKET_CENTERS = np.array([p.center[:2] for p in _TABLE.pockets.values()])  # (6, 2)


def geometric_baseline_score(pos: np.ndarray, vel: np.ndarray) -> float:
    """
    'Too good to be true' sanity check: does velocity direction alone,
    extrapolated as a straight ray from pos, pass close to any pocket?
    No physics, no friction, no collisions — pure geometry.

    Returns -min_distance (higher = closer to some pocket = more "pocket-like").
    """
    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        return -np.min(np.linalg.norm(POCKET_CENTERS - pos, axis=1))
    v_hat = vel / speed
    rel   = POCKET_CENTERS - pos                       # (6, 2)
    t     = np.clip(rel @ v_hat, 0.0, None)             # projection, forward only
    closest = pos + t[:, None] * v_hat                  # (6, 2)
    dists   = np.linalg.norm(POCKET_CENTERS - closest, axis=1)
    return -float(np.min(dists))


def collect_pocket_preds(
    model: RSSMModel,
    shots: list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same first-touch-h logic as train_rssm.evaluate(), but returns raw probs
    AND a `tautological` flag per sample.

    tautological=True means this ball's FIRST recorded event IS the pocket
    event itself (no prior collision) — the node feature fed into that step
    one-hot-encodes event_type=pocket directly, so predict_pocket(h) is just
    reading back an input it was already handed, not predicting the future.
    """
    probs:    list[float] = []
    labels:   list[int]   = []
    tauto:    list[bool]  = []
    baseline: list[float] = []

    model.eval()
    with torch.no_grad():
        for shot in shots:
            h = model.init_hidden(shot.n_balls, device)
            first_touch_h    : dict[int, torch.Tensor] = {}
            first_touch_type : dict[int, int]          = {}
            first_touch_posvel: dict[int, tuple]        = {}

            for k, ev in enumerate(shot.event_steps):
                node_i = ev.node_i.to(device)
                node_j = ev.node_j.to(device) if ev.node_j is not None else None
                edge   = ev.edge.to(device)   if ev.edge   is not None else None

                if ev.event_type == 0 and ev.ball_j is not None:  # EVENT_BALL_BALL
                    h, _, _, _, _ = model.step_ball_ball(
                        h, ev.ball_i, ev.ball_j, node_i, node_j, edge,
                    )
                else:
                    h, _, _ = model.step_single(
                        h, ev.ball_i, node_i, ev.normal.to(device),
                    )

                # Post-collision velocity = pre-collision vel (node_i) + ground-truth
                # Δvel for THIS event. This is the outgoing direction that actually
                # determines where the ball goes next — using the incoming (pre-
                # collision) velocity here would test the wrong physical quantity.
                gt_i = shot.gt_deltas_i[k]
                post_vel_i = (node_i[2:4] + gt_i[0:2]).numpy()
                if ev.ball_i not in first_touch_h:
                    first_touch_h[ev.ball_i]     = h[ev.ball_i].clone()
                    first_touch_type[ev.ball_i]  = ev.event_type
                    first_touch_posvel[ev.ball_i] = (node_i[0:2].numpy(), post_vel_i)
                if ev.ball_j is not None and ev.ball_j not in first_touch_h:
                    gt_j = shot.gt_deltas_j[k]
                    post_vel_j = (node_j[2:4] + gt_j[0:2]).numpy()
                    first_touch_h[ev.ball_j]     = h[ev.ball_j].clone()
                    first_touch_type[ev.ball_j]  = ev.event_type
                    first_touch_posvel[ev.ball_j] = (node_j[0:2].numpy(), post_vel_j)

            for bi_idx, h_i in first_touch_h.items():
                p = model.predict_pocket(h_i.unsqueeze(0)).item()
                probs.append(p)
                labels.append(int(shot.will_pocket.get(bi_idx, False)))
                tauto.append(first_touch_type[bi_idx] == EVENT_POCKET)
                pos, vel = first_touch_posvel[bi_idx]
                baseline.append(geometric_baseline_score(pos, vel))

    return np.array(probs), np.array(labels), np.array(tauto), np.array(baseline)


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def report(probs: np.ndarray, labels: np.ndarray) -> None:
    base_rate = labels.mean()
    n = len(labels)
    print(f"N = {n}  positive (will_pocket=True) rate = {base_rate:.3f}")
    print(f"Trivial 'always predict majority class' accuracy = {max(base_rate, 1 - base_rate):.3f}")
    print()

    print(f"{'thresh':>7} {'acc':>7} {'prec':>7} {'recall':>7} {'f1':>7} {'pred_pos%':>10}")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (probs >= thresh).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        acc  = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        pred_pos_pct = pred.mean() * 100
        marker = "  <-- train_rssm.py default" if thresh == 0.5 else ""
        print(f"{thresh:>7.1f} {acc:>7.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {pred_pos_pct:>9.1f}%{marker}")

    print()
    thresh = 0.5
    pred = (probs >= thresh).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    print(f"Confusion @0.5:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    print(f"AUC (rank-based) = {auc_score(probs, labels):.3f}  (0.5 = random, 1.0 = perfect separation)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          type=str, default="world_model/results/rssm_v4/best.pt")
    p.add_argument("--data-dir",      type=str, default="world_model/data_rssm")
    p.add_argument("--n-shots-train", type=int, default=40000)
    p.add_argument("--n-shots-val",   type=int, default=5000)
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"Checkpoint: {args.ckpt}  (epoch={ckpt.get('epoch')}, val_rmse={ckpt.get('val_rmse'):.5f})")

    model = RSSMModel()
    model.load_state_dict(ckpt["state"])
    model.to(device)

    print(f"Loading val shots from {args.data_dir}…")
    all_shots = load_dataset(args.data_dir)
    val_shots = all_shots[args.n_shots_train: args.n_shots_train + args.n_shots_val]
    if not val_shots:
        val_shots = all_shots[-args.n_shots_val:]
    print(f"Val shots: {len(val_shots)}")
    print()

    probs, labels, tauto, baseline = collect_pocket_preds(model, val_shots, device)

    n_tauto = int(tauto.sum())
    print(f"Tautological samples (first touch == pocket event itself): {n_tauto}/{len(tauto)} "
          f"({n_tauto / len(tauto) * 100:.1f}%)")
    print(f"  of which will_pocket=True: {int((tauto & (labels == 1)).sum())}")
    print()

    print("=== ALL samples (includes tautological leak) ===")
    report(probs, labels)

    print()
    print("=== GENUINE only (first touch happens BEFORE any pocket event) ===")
    mask = ~tauto
    report(probs[mask], labels[mask])

    print()
    print("=== Geometric baseline: straight-line velocity ray → nearest pocket ===")
    print("(no physics, no friction, no collisions — just does the initial direction line up)")
    print(f"Baseline AUC (all)     = {auc_score(baseline, labels):.3f}")
    print(f"Baseline AUC (genuine) = {auc_score(baseline[mask], labels[mask]):.3f}")
    print(f"Model AUC    (genuine) = {auc_score(probs[mask], labels[mask]):.3f}  <- repeated for comparison")


if __name__ == "__main__":
    main()
