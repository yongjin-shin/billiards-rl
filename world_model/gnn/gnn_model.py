"""
world_model/gnn/gnn_model.py — GNN-based Multi-Agent World Model

Each ball is a node with shared-weight encoder/decoder.
n_balls is a hyperparameter — same network handles 2, 3, or N balls.

Graph structure:
  Nodes : each ball   (BALL_DIM=7 state + pocket_dists + is_cue flag)
  Edges : ball-ball   (directed, pairwise, geometric features)
          ball-pocket (each ball to each of 6 pockets)

Message passing at every rollout step:
  1. Decode z_i → positions/velocities for geometric edge features
  2. BallBallMsg  : j→i messages, sum-aggregated  → M_ball_i   (128-dim)
  3. BallPocketMsg: pocket→i messages, sum-aggregated → M_pocket_i (128-dim)
  4. BallUpdate   : LN(z_i + MLP(cat(z_i, M_ball, M_pocket)))
  5. BallTransition: LN(z + MLP(z))   — autonomous dynamics
  6. TypeHead     : mean-pool over balls → 5-class collision type

Forward:
  s_0        (B, N, 7)   initial ball states
  is_cue     (B, N)      1.0 for cue ball, 0.0 otherwise
  → s_hat    (B, T+1, N, 7)
  → type_logit (B, T, 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

BALL_DIM     = 7    # [x, y, vx, vy, wx, wy, wz] normalized
LATENT_DIM   = 128
N_POCKETS    = 6
N_COLL_TYPES = 5    # 0=no_coll, 1=ball_ball, 2=linear, 3=circular, 4=pocket

# Normalized pocket positions [0,1] — same as fixeddt_model.POCKET_XY_NORM
POCKET_XY_NORM = torch.tensor([
    [0.0, 0.0], [1.0, 0.0],
    [0.0, 0.5], [1.0, 0.5],
    [0.0, 1.0], [1.0, 1.0],
], dtype=torch.float32)  # (6, 2)


def _mlp(*dims: int, act=nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class BallEncoder(nn.Module):
    """
    Shared encoder for each ball.
    input: ball_state(7) + pocket_dists(6) + is_cue(1) = 14
    output: z (LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = _mlp(BALL_DIM + N_POCKETS + 1, 64, latent_dim)
        self.register_buffer("pocket_xy", POCKET_XY_NORM)  # (6, 2)

    def _pocket_dists(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (*, 2) → dists: (*, 6)
        diff = xy.unsqueeze(-2) - self.pocket_xy  # (*, 6, 2)
        return diff.norm(dim=-1)                   # (*, 6)

    def forward(self, s: torch.Tensor, is_cue: torch.Tensor) -> torch.Tensor:
        # s: (B, N, 7),  is_cue: (B, N)
        dists = self._pocket_dists(s[..., 0:2])    # (B, N, 6)
        x = torch.cat([s, dists, is_cue.unsqueeze(-1)], dim=-1)  # (B, N, 14)
        return self.net(x)                          # (B, N, LATENT_DIM)


class BallDecoder(nn.Module):
    """Shared decoder: z → ball state (7-dim)."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = _mlp(latent_dim, 64, BALL_DIM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BallBallMsg(nn.Module):
    """
    Directed message from ball j to ball i.
    input: cat(z_i, z_j, Δpos(2), Δvel(2), dist(1)) = 2*LATENT_DIM + 5
    output: message (LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        in_dim = 2 * latent_dim + 5
        self.net = _mlp(in_dim, latent_dim, latent_dim)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor,
                dpos: torch.Tensor, dvel: torch.Tensor,
                dist: torch.Tensor) -> torch.Tensor:
        # z_i, z_j: (B, N, N, LATENT_DIM) or (*, LATENT_DIM)
        x = torch.cat([z_i, z_j, dpos, dvel, dist], dim=-1)
        return self.net(x)


class BallPocketMsg(nn.Module):
    """
    Message from pocket p to ball i.
    input: cat(z_i, Δpos(2), dist(1)) = LATENT_DIM + 3
    output: message (LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        in_dim = latent_dim + 3
        self.net = _mlp(in_dim, latent_dim, latent_dim)

    def forward(self, z_i: torch.Tensor, dpos: torch.Tensor,
                dist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_i, dpos, dist], dim=-1)
        return self.net(x)


class BallUpdate(nn.Module):
    """
    Aggregate messages + residual update.
    input: z_i(LATENT_DIM), M_ball(LATENT_DIM), M_pocket(LATENT_DIM)
    output: z_i' (LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net  = _mlp(3 * latent_dim, 256, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, m_ball: torch.Tensor,
                m_pocket: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, m_ball, m_pocket], dim=-1)
        return self.norm(z + self.net(x))


class BallTransition(nn.Module):
    """Autonomous dynamics: z_{t+1} = LN(z + MLP(z))."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net  = _mlp(latent_dim, 256, 256, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(z + self.net(z))


class TypeHead(nn.Module):
    """Global 5-class collision type from mean-pooled ball latents."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = _mlp(latent_dim, 64, N_COLL_TYPES)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, LATENT_DIM) → pool → (B, LATENT_DIM) → (B, 5)
        return self.net(z.mean(dim=1))


class GNNWorldModel(nn.Module):
    """
    GNN-based world model. n_balls is dynamic at runtime.

    forward(s_0, n_steps, is_cue) → (s_hat, type_logit)
      s_0:        (B, N, 7)
      is_cue:     (B, N)      1.0 for cue ball
      s_hat:      (B, T+1, N, 7)
      type_logit: (B, T, 5)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = BallEncoder(latent_dim)
        self.decoder    = BallDecoder(latent_dim)
        self.bb_msg     = BallBallMsg(latent_dim)
        self.bp_msg     = BallPocketMsg(latent_dim)
        self.update     = BallUpdate(latent_dim)
        self.transition = BallTransition(latent_dim)
        self.type_head  = TypeHead(latent_dim)
        self.register_buffer("pocket_xy", POCKET_XY_NORM)  # (6, 2)

    def _message_pass(self, z: torch.Tensor,
                      s_decoded: torch.Tensor) -> torch.Tensor:
        """
        One round of message passing.
        z:         (B, N, D)
        s_decoded: (B, N, 7)  decoded positions/velocities for edge features
        returns z' (B, N, D)
        """
        B, N, D = z.shape
        pos = s_decoded[:, :, 0:2]  # (B, N, 2)
        vel = s_decoded[:, :, 2:4]  # (B, N, 2)

        # ── Ball-ball messages ──────────────────────────────────────────
        # (B, N, N, 2): pos_j - pos_i for all pairs
        dpos = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 2)  [i, j] = j-i
        dvel = vel.unsqueeze(2) - vel.unsqueeze(1)  # (B, N, N, 2)
        dist = dpos.norm(dim=-1, keepdim=True)       # (B, N, N, 1)

        z_i = z.unsqueeze(2).expand(B, N, N, D)     # (B, N, N, D) receiver
        z_j = z.unsqueeze(1).expand(B, N, N, D)     # (B, N, N, D) sender

        msgs_bb = self.bb_msg(z_i, z_j, dpos, dvel, dist)  # (B, N, N, D)

        # Zero out self-messages, then sum over senders (dim=2)
        mask = 1.0 - torch.eye(N, device=z.device).unsqueeze(0).unsqueeze(-1)
        M_ball = (msgs_bb * mask).sum(dim=2)         # (B, N, D)

        # ── Ball-pocket messages ────────────────────────────────────────
        # pocket_xy: (6, 2) → (1, 1, 6, 2)
        pxy = self.pocket_xy.view(1, 1, N_POCKETS, 2)
        pos_exp = pos.unsqueeze(2)                   # (B, N, 1, 2)
        dp = pxy - pos_exp                           # (B, N, 6, 2)
        dd = dp.norm(dim=-1, keepdim=True)           # (B, N, 6, 1)

        z_exp = z.unsqueeze(2).expand(B, N, N_POCKETS, D)
        msgs_bp = self.bp_msg(z_exp, dp, dd)         # (B, N, 6, D)
        M_pocket = msgs_bp.sum(dim=2)                # (B, N, D)

        return self.update(z, M_ball, M_pocket)      # (B, N, D)

    def forward(self, s_0: torch.Tensor, n_steps: int,
                is_cue: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = s_0.shape

        z = self.encoder(s_0, is_cue)   # (B, N, D)

        states   = [s_0]
        type_logits = []

        for _ in range(n_steps):
            s_dec = self.decoder(z)                    # (B, N, 7)
            z     = self._message_pass(z, s_dec)       # (B, N, D)
            type_logits.append(self.type_head(z))      # (B, 5)
            z     = self.transition(z)                 # (B, N, D)
            states.append(self.decoder(z))             # (B, N, 7)

        s_hat      = torch.stack(states, dim=1)        # (B, T+1, N, 7)
        type_logit = torch.stack(type_logits, dim=1)   # (B, T, 5)
        return s_hat, type_logit


def gnn_rollout_loss(
    s_hat:        torch.Tensor,          # (B, T+1, N, 7)
    seq_s:        torch.Tensor,          # (B, T+1, N, 7)  ground truth
    type_logit:   torch.Tensor,          # (B, T, 5)
    seq_types:    torch.Tensor,          # (B, T) int  0–4
    class_weights: torch.Tensor | None = None,
    log_sigma:    torch.Tensor | None = None,    # (2,) Kendall
    w_state: float = 1.0,
    w_type:  float = 1.0,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Multi-ball rollout loss.
    State MSE averaged over all balls, type CE same as SSM.
    """
    # State loss — mean over time steps AND balls
    loss_state = ((s_hat[:, 1:] - seq_s[:, 1:]) ** 2).mean()
    # Split for logging
    loss_cue = ((s_hat[:, 1:, 0] - seq_s[:, 1:, 0]) ** 2).mean()
    loss_tgt = ((s_hat[:, 1:, 1:] - seq_s[:, 1:, 1:]) ** 2).mean() \
               if s_hat.shape[2] > 1 else torch.zeros(1, device=s_hat.device).squeeze()

    # Type loss — identical to ssm_rollout_loss
    B, T, _ = type_logit.shape
    flat_logit = type_logit.reshape(B * T, N_COLL_TYPES)
    flat_types = seq_types.reshape(B * T).long()

    if focal_gamma > 0.0:
        from world_model.ssm_model import _focal_cross_entropy
        loss_type = _focal_cross_entropy(flat_logit, flat_types, class_weights,
                                         focal_gamma, label_smoothing)
    else:
        loss_type = F.cross_entropy(flat_logit, flat_types,
                                    weight=class_weights,
                                    label_smoothing=label_smoothing)
    loss_type_n = loss_type / math.log(N_COLL_TYPES)

    if log_sigma is not None:
        s = log_sigma.clamp(-6, 6)
        total = (torch.exp(-s[0]) * loss_state  + s[0] +
                 torch.exp(-s[1]) * loss_type_n + s[1])
        eff   = torch.exp(-s).detach()
        extra = {"kw_state": eff[0].item(), "kw_type": eff[1].item()}
    else:
        total = w_state * loss_state + w_type * loss_type_n
        extra = {}

    return total, {
        "loss_cue":   loss_cue.item(),
        "loss_tgt":   loss_tgt.item() if isinstance(loss_tgt, torch.Tensor) else loss_tgt,
        "loss_state": loss_state.item(),
        "loss_type":  loss_type.item(),
        "total":      total.item(),
        **extra,
    }
