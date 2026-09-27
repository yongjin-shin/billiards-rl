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
import collections
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


def collect_pocket_preds_all_touches(
    model: RSSMModel,
    shots: list,
    device: torch.device,
) -> list[dict]:
    """
    Unlike collect_pocket_preds() (first touch only), this records
    predict_pocket(h) at EVERY event a ball participates in — mirroring how
    train_rssm.py's training loss actually queries predict_pocket (line 271:
    `pocket_probs = model.predict_pocket(h)` runs inside the per-event loop,
    for every ball, not just at first touch). During a real rollout the head
    would be queried the same way — after every event — so first-touch-only
    accuracy doesn't tell us whether the prediction stays reliable, improves,
    or degrades across the rest of the sequence.

    Also records `is_last_real_touch`: True if this is the ball's last touch
    BEFORE either (a) the pocket event itself, or (b) the shot ending with
    the ball still on the table. Lets us check whether confidence rises
    monotonically as the ball approaches its actual fate.
    """
    records: list[dict] = []

    model.eval()
    with torch.no_grad():
        for shot in shots:
            h = model.init_hidden(shot.n_balls, device)
            touch_n: dict[int, int] = {}

            # Pre-scan: how many total touches does each ball get, so we know
            # which touch is "last" for it.
            total_touches: dict[int, int] = {}
            for ev in shot.event_steps:
                total_touches[ev.ball_i] = total_touches.get(ev.ball_i, 0) + 1
                if ev.ball_j is not None:
                    total_touches[ev.ball_j] = total_touches.get(ev.ball_j, 0) + 1

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

                touches = [(ev.ball_i, ev.event_type == EVENT_POCKET)]
                if ev.ball_j is not None:
                    touches.append((ev.ball_j, False))

                for b, is_pocket_ev in touches:
                    touch_n[b] = touch_n.get(b, 0) + 1
                    if is_pocket_ev:
                        continue  # tautological — the pocket event IS the label, skip
                    p = model.predict_pocket(h[b].unsqueeze(0)).item()
                    will_pocket = shot.will_pocket.get(b, False)
                    # "last real touch" = last touch BEFORE the ball's fate is
                    # settled. For balls that get pocketed, that's one touch
                    # before the (excluded) pocket event; for balls that never
                    # get pocketed, it's just their actual last touch. Getting
                    # this branch wrong makes the slice 100%-one-label by
                    # construction (label leaks into which record gets kept).
                    last_real_idx = total_touches[b] - 1 if will_pocket else total_touches[b]
                    is_last_real = touch_n[b] == last_real_idx
                    records.append({
                        "touch_idx": touch_n[b],
                        "total_touches": total_touches[b],
                        "is_last_real_touch": is_last_real,
                        "label": int(will_pocket),
                        "prob": p,
                    })

    return records


def report_by_touch_index(records: list[dict]) -> None:
    by_idx: dict[int, list[dict]] = collections.defaultdict(list)
    for r in records:
        # bucket 4+ together — sample count thins out fast
        idx = min(r["touch_idx"], 4)
        by_idx[idx].append(r)

    print(f"{'touch#':>7} {'n':>6} {'pos_rate':>9} {'AUC':>7}")
    for idx in sorted(by_idx):
        rs = by_idx[idx]
        labels = np.array([r["label"] for r in rs])
        probs  = np.array([r["prob"]  for r in rs])
        label_str = f"{idx}" if idx < 4 else "4+"
        auc = auc_score(probs, labels)
        print(f"{label_str:>7} {len(rs):>6} {labels.mean():>9.3f} {auc:>7.3f}")

    last_real = [r for r in records if r["is_last_real_touch"]]
    labels = np.array([r["label"] for r in last_real])
    probs  = np.array([r["prob"]  for r in last_real])
    print()
    print(f"Last real touch before shot ends (excl. pocket event itself, N={len(last_real)}):")
    print(f"  pos_rate={labels.mean():.3f}  AUC={auc_score(probs, labels):.3f}")


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

    print()
    print("=== By touch index: is the prediction still accurate at 2nd, 3rd, ... touch? ===")
    print("(train_rssm.py queries predict_pocket(h) after EVERY event, not just first touch —")
    print(" this checks whether accuracy holds up across the whole sequence, or was cherry-picked)")
    records = collect_pocket_preds_all_touches(model, val_shots, device)
    report_by_touch_index(records)


if __name__ == "__main__":
    main()
