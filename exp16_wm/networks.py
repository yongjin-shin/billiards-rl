"""
Networks for Exp-16: VanillaSAC and WorldModelSAC.

VanillaSAC:
  Actor  : TanhGaussian (tanh squashing + affine rescale to env action space)
  Critic : Twin Q-networks [256, 256]

WorldModelSAC (added later, same file):
  WorldModelCritic : M(s,a)→ĥ  +  q(ĥ)→Q
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────

class Actor(nn.Module):
    """
    Squashed Gaussian Actor (SB3-compatible).

    Outputs:
      action  : rescaled to env action space [act_low, act_high]
      log_prob: in tanh-space [-1,1], consistent with SB3 target_entropy=-action_dim
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        hidden: list[int] = [256, 256],
    ):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)

        self.mean_layer    = nn.Linear(prev, action_dim)
        self.log_std_layer = nn.Linear(prev, action_dim)

        self.register_buffer("act_scale", torch.FloatTensor((act_high - act_low) / 2.0))
        self.register_buffer("act_bias",  torch.FloatTensor((act_high + act_low) / 2.0))

    # ── internal: pre-tanh sample ──────────────

    def _dist(self, obs: torch.Tensor):
        x       = self.trunk(obs)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    # ── forward (used in update) ───────────────

    def forward(self, obs: torch.Tensor):
        """
        Returns:
          action   (B, action_dim)  in env action space
          log_prob (B, 1)           in tanh-space (SB3 convention)
        """
        dist  = self._dist(obs)
        u     = dist.rsample()                        # pre-tanh, reparameterised
        a_tan = torch.tanh(u)                         # ∈ (-1, 1)
        action = a_tan * self.act_scale + self.act_bias  # env space

        # log π(a) in tanh-space  (SB3: no affine correction in log_prob)
        log_prob = dist.log_prob(u) - torch.log(1 - a_tan.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)     # (B, 1)

        return action, log_prob

    # ── no-grad version for data collection ───

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        dist  = self._dist(obs)
        u     = dist.mean if deterministic else dist.rsample()
        a_tan = torch.tanh(u)
        action = a_tan * self.act_scale + self.act_bias
        return action.cpu().numpy()


# ──────────────────────────────────────────────
# Critic  (twin Q)
# ──────────────────────────────────────────────

class Critic(nn.Module):
    """
    Twin Q-networks.  Input: (obs, action) in env action space.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int] = [256, 256]):
        super().__init__()
        self.Q1 = _mlp(obs_dim + action_dim, 1, hidden)
        self.Q2 = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x), self.Q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ──────────────────────────────────────────────
# WorldModelCritic  (added for WMSAC)
# ──────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    M: (obs, action) → ĥ
    Direct MLP predictor, no latent bottleneck, no VAE.
    ĥ shape: (B, max_events * event_dim) — flattened trajectory.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_events: int,
        event_dim: int,
        hidden: list[int] = [512, 512, 512],
    ):
        super().__init__()
        self.max_events = max_events
        self.event_dim  = event_dim
        out_dim         = max_events * event_dim
        self.net        = _mlp(obs_dim + action_dim, out_dim, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns ĥ: (B, max_events, event_dim)"""
        x    = torch.cat([obs, action], dim=-1)
        flat = self.net(x)                                   # (B, max_events*event_dim)
        return flat.view(-1, self.max_events, self.event_dim)


class WorldModelCritic(nn.Module):
    """
    Q(s,a) = q( M(s,a) )

    M : (obs, action) → ĥ   — dense physics supervision
    q : ĥ             → Q   — sparse Bellman supervision
    Twin version: Q1, Q2 each have their own M and q.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_events: int,
        event_dim: int,
        wm_hidden: list[int]  = [512, 512, 512],
        q_hidden: list[int]   = [256, 256],
    ):
        super().__init__()
        traj_dim    = max_events * event_dim

        self.M1 = WorldModel(obs_dim, action_dim, max_events, event_dim, wm_hidden)
        self.M2 = WorldModel(obs_dim, action_dim, max_events, event_dim, wm_hidden)
        self.q1 = _mlp(traj_dim, 1, q_hidden)
        self.q2 = _mlp(traj_dim, 1, q_hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (Q1, Q2, ĥ1, ĥ2)  — ĥ needed for WM loss."""
        h1 = self.M1(obs, action)          # (B, max_events, event_dim)
        h2 = self.M2(obs, action)
        q1 = self.q1(h1.flatten(1))        # (B, 1)
        q2 = self.q2(h2.flatten(1))
        return q1, q2, h1, h2

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2, _, _ = self.forward(obs, action)
        return torch.min(q1, q2)
