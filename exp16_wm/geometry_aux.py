"""
exp16_wm/geometry_aux.py — geometric quantities used only as auxiliary training targets (Exp-16).

The pair embedding e_ij of a trained policy keeps the cut angle and the distances but drops the pocket entry deviation,
the ball's distance to the cushion and the other balls near the shot paths (probe, experiments.md 2026-10-12). Predicting
them from e_ij during training keeps that information in the encoder. The values are computed from the raw observation
for training only; nothing is added to the policy's input.
"""

import numpy as np

N_BALLS, N_POCKETS, N_PAIRS = 3, 6, 18
AUX_NAMES = ("pocket entry deviation (deg)", "ball to cushion (m)", "blocker on cue->ball path (m)", "blocker on ball->pocket path (m)")
N_AUX = len(AUX_NAMES)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between row vectors u, v (N, 2) in degrees, 0 when one is zero."""
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    c = (u * v).sum(1) / np.maximum(nu * nv, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of points p (N, 2) to the segments a→b (N, 2); a degenerate segment is the distance to the point a."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def pair_geometry(obs: np.ndarray, table: tuple[float, float], blocker_cap: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw observations (N, 23) [cue xy, 3 × (x, y, pocketed), 6 pockets xy], normalised by the table width / length →
    (targets (N, 18, N_AUX) float32, valid (N, 18) bool); pair index = ball × 6 + pocket (the policy's order). Targets of a
    pair whose ball is pocketed are 0 and valid is False.
      0 pocket entry deviation: angle between (ball − pocket) and (table centre − pocket)
      1 ball to cushion: min distance of the ball to the four rails
      2 blocker on the cue→ball path: min distance of the other unpocketed balls to that segment (capped; cap if none)
      3 blocker on the ball→pocket path: same for that segment
    """
    obs = np.asarray(obs, dtype=np.float64)
    sc = np.asarray(table, dtype=np.float64)
    n = len(obs)
    cue = obs[:, 0:2] * sc
    balls = obs[:, 2:11].reshape(n, N_BALLS, 3)
    pos, live = balls[..., :2] * sc, balls[..., 2] < 0.5
    pockets = obs[:, 11:23].reshape(n, N_POCKETS, 2) * sc
    centre = np.broadcast_to(sc / 2.0, (n, 2))
    out = np.zeros((n, N_PAIRS, N_AUX), dtype=np.float32)
    valid = np.zeros((n, N_PAIRS), dtype=bool)
    for b in range(N_BALLS):
        ball = pos[:, b]
        cushion = np.minimum.reduce([ball[:, 0], sc[0] - ball[:, 0], ball[:, 1], sc[1] - ball[:, 1]])
        for k in range(N_POCKETS):
            i, pk = b * N_POCKETS + k, pockets[:, k]
            d_cue, d_pk = np.full(n, blocker_cap), np.full(n, blocker_cap)
            for q in range(N_BALLS):
                if q == b:
                    continue
                oth = pos[:, q]
                d_cue = np.where(live[:, q], np.minimum(d_cue, _seg_dist(oth, cue, ball)), d_cue)
                d_pk = np.where(live[:, q], np.minimum(d_pk, _seg_dist(oth, ball, pk)), d_pk)
            vals = np.stack([_angle_deg(ball - pk, centre - pk), cushion, d_cue, d_pk], axis=1)
            out[:, i] = np.where(live[:, b, None], vals, 0.0)
            valid[:, i] = live[:, b]
    return out, valid


def aim_targets(rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The exact ghost-ball aim of every pair as an auxiliary target: rel (N, 21) relative observations (metres, the policy's frame)
    → (targets (N, 18, 1) float32 = aim direction in degrees in the relative frame, the same quantity as the made2_geom diagnostic;
    valid (N, 18) = the pair's ball is on the table). Pair index = ball * 6 + pocket.
    """
    import torch
    from exp16_wm.gnn_policy import PairGNNPolicy
    rel = np.asarray(rel, dtype=np.float32)
    aim = np.degrees(PairGNNPolicy._geometric_aim(torch.as_tensor(rel)).numpy())
    from exp16_wm.obs_transform import n_balls_rel
    on_table = rel[:, 2:3 * n_balls_rel(rel.shape[1]):3] == 0
    return aim[..., None].astype(np.float32), np.repeat(on_table, N_POCKETS, axis=1)


def standardize(targets: np.ndarray, valid: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise targets (N, 18, K) with the mean / std of the valid entries of the train rows only → (standardised, mean (K,), std (K,)).
    Invalid entries stay 0 (they are masked in the loss).
    """
    t = targets[train][valid[train]]
    mean, std = t.mean(0), np.maximum(t.std(0), 1e-6)
    z = np.where(valid[..., None], (targets - mean) / std, 0.0).astype(np.float32)
    return z, mean.astype(np.float32), std.astype(np.float32)
