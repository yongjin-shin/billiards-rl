"""
world_model/rssm_model.py

R-SSM: Relational State-Space Model for billiards world modelling.

Each ball carries a persistent latent h_i (H-dim) that is updated
via GNN message passing at every physical event within a shot.

Architecture:
  StateEncoder : MLP (7  → H)          shared across all balls
  MsgMLP       : MLP (2H+2*NODE+EDGE → H)  pairwise, directional
  UpdMLP       : MLP (2H → H)          residual update
  SingleMLP    : MLP (H+NODE+2 → H)   cushion / pocket events
  DecMLP       : MLP (H → 5)           Δvel(2) + Δavel(3)
  TypeMLP      : MLP (H → N_TYPE)      next-event type per ball
  QProj        : Linear (H → 1)        attention scoring
  QHead        : MLP (H → 1)           shot-level value

NODE_DIM = 14  : pos(2) + vel(2) + avel(3) + type_onehot(7)
EDGE_DIM = 9   : rel_pos(2) + rel_vel(2) + rel_avel(3) + normal(2)
N_TYPE   = 7   : 0=ball_ball 1=cue_linear 2=cue_circular
                 3=ball_pocket 4=stick_ball 5=tgt_linear 6=tgt_circular
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dimensions ────────────────────────────────────────────────────────────────

H_DIM    = 64
NODE_DIM = 14   # pos(2) + vel(2) + avel(3) + type_onehot(7)
EDGE_DIM = 9    # rel_pos(2) + rel_vel(2) + rel_avel(3) + normal(2)
N_TYPE   = 7

# Event type constants (match event_detector.py COLL_TYPE)
EVENT_BALL_BALL = 0
EVENT_CUSHION   = frozenset({1, 2, 5, 6})
EVENT_POCKET    = 3


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EventStep:
    """One physical event within a shot rollout."""
    event_type : int
    ball_i     : int
    ball_j     : int | None                # None for single-ball events
    node_i     : torch.Tensor | None       # (NODE_DIM,); None when stored in pkl
    node_j     : torch.Tensor | None       # (NODE_DIM,) or None
    edge       : torch.Tensor | None       # (EDGE_DIM,) i→j direction; None for single
    normal     : torch.Tensor              # (2,)


@dataclass
class EventOutput:
    """Predictions produced for one event."""
    delta_i  : torch.Tensor             # (5,) Δvel(2)+Δavel(3) for ball_i
    delta_j  : torch.Tensor | None      # (5,) for ball_j; None for single events
    type_i   : torch.Tensor             # (N_TYPE,) logits for ball_i
    type_j   : torch.Tensor | None      # (N_TYPE,) for ball_j; None for single


@dataclass
class RSSMOutput:
    event_outputs : list[EventOutput]
    Q             : torch.Tensor   # (1,)
    h_final       : torch.Tensor   # (N, H_DIM)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ── Model ─────────────────────────────────────────────────────────────────────

class RSSMModel(nn.Module):
    """
    Parameters
    ----------
    h_dim    : latent dim per ball (default 64)
    hidden   : hidden layer sizes for all MLPs (default [256, 256])
    """

    def __init__(
        self,
        h_dim : int       = H_DIM,
        hidden: list[int] = None,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        self.h_dim = h_dim

        msg_in = 2 * h_dim + 2 * NODE_DIM + EDGE_DIM   # 165

        self.msg_mlp       = _mlp(msg_in,    hidden, h_dim)
        self.upd_mlp       = _mlp(2 * h_dim, hidden, h_dim)
        self.single_mlp    = _mlp(h_dim + NODE_DIM + 2, hidden, h_dim)
        self.dec_mlp       = _mlp(h_dim,     hidden, 5)
        self.type_mlp      = _mlp(h_dim,     hidden, N_TYPE)
        self.pocket_mlp    = _mlp(h_dim,     [h_dim // 2], 1)   # per-ball pocket head
        self.q_proj        = nn.Linear(h_dim, 1)
        self.q_head        = _mlp(h_dim,     hidden, 1)
        self.norm          = nn.LayerNorm(h_dim)

    # ── Init ──────────────────────────────────────────────────────────────────

    def init_hidden(self, n_balls: int, device: torch.device | None = None) -> list:
        """
        Returns zero-initialised per-ball latents as a list of tensors.

        List assignment (h[i] = h_i_new) avoids tensor cloning on every event.

        Returns
        -------
        h : list of n_balls tensors, each (h_dim,)
        """
        return [torch.zeros(self.h_dim, device=device) for _ in range(n_balls)]

    # ── Step functions ────────────────────────────────────────────────────────

    def step_ball_ball(
        self,
        h      : list,           # list of (h_dim,) tensors, one per ball
        i      : int,
        j      : int,
        node_i : torch.Tensor,   # (NODE_DIM,)
        node_j : torch.Tensor,   # (NODE_DIM,)
        edge   : torch.Tensor,   # (EDGE_DIM,) — i→j direction
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h        list (updated in-place; same object returned for chaining)
        delta_i  (5,)
        delta_j  (5,)
        type_i   (N_TYPE,)
        type_j   (N_TYPE,)
        """
        h_i, h_j = h[i], h[j]

        msg_i = self.msg_mlp(torch.cat([h_i, h_j, node_i, node_j,  edge]))
        msg_j = self.msg_mlp(torch.cat([h_j, h_i, node_j, node_i, -edge]))

        h_i_new = self.norm(h_i + self.upd_mlp(torch.cat([h_i, msg_i])))
        h_j_new = self.norm(h_j + self.upd_mlp(torch.cat([h_j, msg_j])))

        # Python list assignment — no tensor clone needed
        h[i] = h_i_new
        h[j] = h_j_new

        return (
            h,
            self.dec_mlp(h_i_new),
            self.dec_mlp(h_j_new),
            self.type_mlp(h_i_new),
            self.type_mlp(h_j_new),
        )

    def step_single(
        self,
        h      : list,           # list of (h_dim,) tensors, one per ball
        i      : int,
        node_i : torch.Tensor,   # (NODE_DIM,)
        normal : torch.Tensor,   # (2,)
    ) -> tuple[list, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h       list (updated in-place; same object returned for chaining)
        delta_i (5,)
        type_i  (N_TYPE,)
        """
        h_i     = h[i]
        h_i_new = self.norm(
            h_i + self.single_mlp(torch.cat([h_i, node_i, normal]))
        )
        h[i] = h_i_new

        return (
            h,
            self.dec_mlp(h_i_new),
            self.type_mlp(h_i_new),
        )

    # ── Pocket prediction ─────────────────────────────────────────────────────

    def predict_pocket(self, h) -> torch.Tensor:
        """
        Per-ball probability of being pocketed in this shot.

        Parameters
        ----------
        h : list of (h_dim,) tensors  or  (N, h_dim) Tensor

        Returns
        -------
        (N,)  sigmoid probabilities, one per ball
        """
        if isinstance(h, list):
            h = torch.stack(h, dim=0)
        return torch.sigmoid(self.pocket_mlp(h)).squeeze(-1)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def aggregate_q(self, h) -> torch.Tensor:
        """
        Attention-pool over all ball latents → Q value.

        h : list of (h_dim,) tensors  or  (N, h_dim) Tensor
        Returns scalar
        """
        if isinstance(h, list):
            h = torch.stack(h, dim=0)
        scores  = self.q_proj(h)                          # (N, 1)
        attn    = torch.softmax(scores, dim=0)            # (N, 1)
        h_agg   = (attn * h).sum(dim=0)                  # (h_dim,)
        return self.q_head(h_agg).squeeze(-1)             # (1,) → scalar

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        n_balls : int,
        events  : list[EventStep],
        device  : torch.device | None = None,
    ) -> RSSMOutput:
        """
        Run a full shot rollout.

        Parameters
        ----------
        n_balls : number of balls in the shot
        events  : ordered list of EventStep within the shot

        Returns
        -------
        RSSMOutput with per-event predictions, Q, and final h
        """
        h = self.init_hidden(n_balls, device)   # (N, h_dim)
        event_outputs: list[EventOutput] = []

        for ev in events:
            if ev.event_type == EVENT_BALL_BALL:
                h, d_i, d_j, t_i, t_j = self.step_ball_ball(
                    h, ev.ball_i, ev.ball_j,
                    ev.node_i, ev.node_j, ev.edge,
                )
                event_outputs.append(EventOutput(d_i, d_j, t_i, t_j))

            else:   # cushion or pocket
                h, d_i, t_i = self.step_single(
                    h, ev.ball_i, ev.node_i, ev.normal,
                )
                event_outputs.append(EventOutput(d_i, None, t_i, None))

        Q = self.aggregate_q(h)
        return RSSMOutput(event_outputs=event_outputs, Q=Q, h_final=h)
