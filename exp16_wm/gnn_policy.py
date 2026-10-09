"""
exp16_wm/gnn_policy.py — (ball, pocket)-pair GNN policy for imitation (Exp-16).

Input is the action-frame relative observation (exp16_wm/obs_transform.py, 21-dim:
3 balls × (rx, ry, pocketed) + 6 pockets × (px, py); cue at the origin, cue →
nearest ball along +x). Instead of one flat MLP over those 21 numbers:

  ball node   : [rx, ry, |r|, pocketed, is_ref]   (is_ref = the nearest live ball,
                the direction delta_angle is measured from)
  ball ↔ ball : shared message MLP on [ball_i, ball_k − ball_i, |·|, pocketed_k],
                summed over the other live balls → ball embedding h_i
  ball → pocket pair (i, j): shared MLP on [h_i, pocket_j, pocket_j − ball_i, |·|, side_j]
                → e_ij, masked when ball i is pocketed

Relations are given as plain differences (no billiards formulas) and every pair
shares weights, so what is learned for one (ball, pocket) applies to all 18.
Each pair proposes one action and a selection score; the deterministic action is
the top-scored pair's — several right answers live in different pairs instead of
being averaged into a miss. Trained with relaxed winner-takes-all: the pair
nearest to a pocketing candidate regresses to it (weight 1 − ε, others ε) and
the selection score is trained by cross-entropy to pick that pair.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp16_wm.train import ACT_LOW, ACT_HIGH
from exp16_wm.obs_transform import N_BALLS, N_POCKETS, REL_DIM

N_PAIRS = N_BALLS * N_POCKETS
SIDE_POCKETS = (1, 4)            # obs pocket order: (0,0) (0,½) (0,1) (1,0) (1,½) (1,1)
_NEG = -1e9


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class PairGNNPolicy(nn.Module):
    def __init__(self, hidden: int = 128, wta_eps: float = 0.05, sel_weight: float = 10.0):
        super().__init__()
        H = hidden
        self.wta_eps, self.sel_weight = wta_eps, sel_weight
        self.bb   = _mlp(5 + 2 + 1 + 1, H, H)
        self.ball = _mlp(5 + H, H, H)
        self.pair = _mlp(H + 2 + 2 + 1 + 1, H, H)
        self.head_a = nn.Linear(H, 2)
        self.head_s = nn.Linear(H, 1)
        side = torch.zeros(N_POCKETS)
        side[list(SIDE_POCKETS)] = 1.0
        self.register_buffer("side", side)
        self.register_buffer("act_scale", torch.tensor((ACT_HIGH - ACT_LOW) / 2.0, dtype=torch.float32))
        self.register_buffer("act_bias", torch.tensor((ACT_HIGH + ACT_LOW) / 2.0, dtype=torch.float32))

    # ── encoder ───────────────────────────────

    def _encode(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = rel.shape[0]
        balls = rel[:, :3 * N_BALLS].reshape(B, N_BALLS, 3)
        pos, pocketed = balls[..., :2], balls[..., 2]
        live = pocketed < 0.5
        dist = pos.norm(dim=-1)
        ref = torch.where(live, dist, torch.full_like(dist, float("inf"))).argmin(dim=1)
        is_ref = F.one_hot(ref, N_BALLS).float() * live.any(dim=1, keepdim=True).float()
        ball_f = torch.cat([pos, dist[..., None], pocketed[..., None], is_ref[..., None]], dim=-1)

        diff = pos[:, None, :, :] - pos[:, :, None, :]                    # (B, i, k, 2): k relative to i
        bb_in = torch.cat([ball_f[:, :, None, :].expand(B, N_BALLS, N_BALLS, 5), diff,
                           diff.norm(dim=-1, keepdim=True),
                           pocketed[:, None, :, None].expand(B, N_BALLS, N_BALLS, 1)], dim=-1)
        not_self = ~torch.eye(N_BALLS, dtype=torch.bool, device=rel.device)
        nb = (live[:, None, :] & not_self[None]).float()[..., None]
        h = self.ball(torch.cat([ball_f, (self.bb(bb_in) * nb).sum(dim=2)], dim=-1))   # (B, 3, H)

        pockets = rel[:, 3 * N_BALLS:].reshape(B, N_POCKETS, 2)
        pr = pockets[:, None, :, :] - pos[:, :, None, :]                   # (B, 3, 6, 2)
        H = h.shape[-1]
        pair_in = torch.cat([h[:, :, None, :].expand(B, N_BALLS, N_POCKETS, H),
                             pockets[:, None].expand(B, N_BALLS, N_POCKETS, 2), pr,
                             pr.norm(dim=-1, keepdim=True),
                             self.side.view(1, 1, N_POCKETS, 1).expand(B, N_BALLS, N_POCKETS, 1)], dim=-1)
        e = self.pair(pair_in).reshape(B, N_PAIRS, H)
        valid = live[:, :, None].expand(B, N_BALLS, N_POCKETS).reshape(B, N_PAIRS)
        return e, valid

    def forward(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """rel (B, 21) → per-pair actions in [-1, 1] (B, 18, 2), selection logits (B, 18), valid (B, 18)."""
        e, valid = self._encode(rel)
        means = torch.tanh(self.head_a(e))
        logits = self.head_s(e).squeeze(-1).masked_fill(~valid, _NEG)
        return means, logits, valid

    # ── interface used by exp16_wm/bc.py ──────

    def predict_unit(self, rel: torch.Tensor) -> torch.Tensor:
        """Deterministic action in [-1, 1]: the top-scored pair's proposal. (B, 2)"""
        means, logits, _ = self(rel)
        return means[torch.arange(len(means)), logits.argmax(dim=1)]

    def train_loss(self, rel: torch.Tensor, teacher_u: torch.Tensor, good_u: torch.Tensor,
                   w: torch.Tensor | None = None, pairs: torch.Tensor | None = None) -> torch.Tensor:
        """
        pairs None → relaxed winner-takes-all (below). pairs (B, K) int64, −1 = unknown →
        supervised pairs: each labelled candidate regresses its own pair's output and the
        selection head is trained to pick those pairs (fixed targets instead of a moving winner).
        Rows with no labelled pair fall back to winner-takes-all.
        """
        if pairs is None:
            return self._wta_loss(rel, teacher_u, good_u, w)
        rows = (pairs >= 0).any(dim=1)
        if bool(rows.all()):
            return self._pair_loss(rel, good_u, pairs, w)
        loss, n = rel.new_zeros(()), len(rel)
        if bool(rows.any()):
            loss = loss + self._pair_loss(rel[rows], good_u[rows], pairs[rows], w) * int(rows.sum())
        loss = loss + self._wta_loss(rel[~rows], teacher_u[~rows], good_u[~rows], w) * int((~rows).sum())
        return loss / n

    def _pair_loss(self, rel: torch.Tensor, good_u: torch.Tensor, pairs: torch.Tensor,
                   w: torch.Tensor | None) -> torch.Tensor:
        w = torch.ones(2, device=rel.device) if w is None else w
        means, logits, _ = self(rel)
        B, K = pairs.shape
        has = pairs >= 0
        pidx = pairs.clamp(min=0)
        m = means.gather(1, pidx[..., None].expand(B, K, 2))
        d = (((m - torch.nan_to_num(good_u)) * w) ** 2).sum(-1)
        n = has.sum(dim=1).clamp(min=1)
        reg = (d * has).sum(dim=1) / n
        logp = F.log_softmax(logits, dim=1).gather(1, pidx)
        ce = -(logp * has).sum(dim=1) / n
        return (reg + self.sel_weight * ce).mean()

    def _wta_loss(self, rel: torch.Tensor, teacher_u: torch.Tensor, good_u: torch.Tensor,
                  w: torch.Tensor | None = None) -> torch.Tensor:
        """Relaxed winner-takes-all regression to the nearest pocketing candidate + selection CE."""
        w = torch.ones(2, device=rel.device) if w is None else w
        means, logits, valid = self(rel)
        tgt_valid = ~torch.isnan(good_u[..., 0])                          # (B, K)
        has_good = tgt_valid.any(dim=1)
        tgt = torch.nan_to_num(good_u).clone()
        tgt[~has_good, 0] = teacher_u[~has_good]                          # no candidate → teacher action
        tgt_valid = tgt_valid.clone()
        tgt_valid[~has_good, 0] = True

        d = (((means[:, :, None, :] - tgt[:, None, :, :]) * w) ** 2).sum(-1)       # (B, 18, K)
        d = torch.where(tgt_valid[:, None, :], d, torch.full_like(d, float("inf")))
        dp = torch.where(valid, d.min(dim=2).values, torch.full_like(d[..., 0], float("inf")))
        rows = valid.any(dim=1)
        if not bool(rows.any()):
            return (means * 0).sum()
        dp, valid_r, logits_r = dp[rows], valid[rows], logits[rows]
        win = dp.argmin(dim=1)
        d_win = dp[torch.arange(len(dp)), win]
        others = valid_r & ~F.one_hot(win, N_PAIRS).bool()
        n_oth = others.sum(dim=1).clamp(min=1)
        d_oth = torch.where(others, dp, torch.zeros_like(dp)).sum(dim=1) / n_oth
        reg = (1 - self.wta_eps) * d_win + self.wta_eps * d_oth
        ce = F.cross_entropy(logits_r, win, reduction="none")
        return (reg + self.sel_weight * ce).mean()

    def params(self) -> list[nn.Parameter]:
        return list(self.parameters())

    @torch.no_grad()
    def act(self, rel_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Env action for one (21,) or a batch (B, 21) of relative observations."""
        x = torch.as_tensor(np.atleast_2d(rel_obs), dtype=torch.float32)
        a = (self.predict_unit(x) * self.act_scale + self.act_bias).numpy()
        return a[0] if np.asarray(rel_obs).ndim == 1 else a

    def save(self, path: str) -> None:
        torch.save({"gnn": self.state_dict()}, path)


assert REL_DIM == 3 * N_BALLS + 2 * N_POCKETS
