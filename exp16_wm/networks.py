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


# ──────────────────────────────────────────────
# GeomCritic  (Exp-16 G: pre-shot geometry features)
# ──────────────────────────────────────────────

TABLE_W, TABLE_L = 0.9906, 1.9812          # pt.Table.default() w, l (m)
BALL_R           = 0.028575
POCKET_CENTERS   = torch.tensor([
    [-0.0294864, -0.0294864], [-0.0685, 0.9906], [-0.0294864, 2.0106864],
    [ 1.0200864, -0.0294864], [ 1.0591,  0.9906], [ 1.0200864, 2.0106864],
])                                          # pt.Table.default() pocket centers (m)
GEOM_DIM         = 6
_DIAG            = float(np.hypot(TABLE_W, TABLE_L))
_NO_HIT          = (0.0, 1.0, 1.0, 1.0)     # (cos_cut, d_cue, miss, d_obj) sentinels, normalized


def geom_features(obs: torch.Tensor, action: torch.Tensor, n_balls: int) -> torch.Tensor:
    """
    Straight-line pre-shot geometry, differentiable in `action` (the first-ball
    choice is a discrete argmin; everything after it is continuous).

    Uses the env's aim rule: phi = direction to nearest unpocketed ball + action[0].
    Returns (B, GEOM_DIM): [hit, cos_cut, d_cue, miss, d_obj, speed], each ~[0, 1].
      hit     — cue ray touches a remaining target ball
      cos_cut — cos between cue direction and line of centers at contact
      d_cue   — cue travel to contact            / table diagonal
      miss    — closest approach of object-ball ray to any pocket / 0.5 m (clipped to 1)
      d_obj   — object-ball travel to that pocket / table diagonal
      speed   — action[1] rescaled from [0.5, 8] to [0, 1]
    """
    B, dev = obs.shape[0], obs.device
    scale  = torch.tensor([TABLE_W, TABLE_L], device=dev)
    cue    = obs[:, :2] * scale                                   # (B, 2)
    balls  = obs[:, 2:2 + 3 * n_balls].view(B, n_balls, 3)
    bpos   = balls[..., :2] * scale                               # (B, n, 2)
    alive  = balls[..., 2] < 0.5                                  # (B, n)

    rel    = bpos - cue[:, None]                                  # (B, n, 2)
    dist   = rel.norm(dim=-1).masked_fill(~alive, float("inf"))
    ref    = rel[torch.arange(B, device=dev), dist.argmin(dim=1)] # (B, 2)
    ref    = torch.where(alive.any(1, keepdim=True), ref, torch.tensor([1.0, 0.0], device=dev))
    phi    = torch.atan2(ref[:, 1], ref[:, 0]) + action[:, 0]
    d      = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # (B, 2)

    proj   = (rel * d[:, None]).sum(-1)                           # (B, n)
    perp2  = (rel * rel).sum(-1) - proj ** 2
    valid  = alive & (proj > 0) & (perp2 < (2 * BALL_R) ** 2)
    t_hit  = proj - torch.sqrt(torch.clamp((2 * BALL_R) ** 2 - perp2, min=1e-6))
    idx    = t_hit.masked_fill(~valid, float("inf")).argmin(dim=1)
    hit    = valid.any(dim=1)
    ar     = torch.arange(B, device=dev)

    obj     = bpos[ar, idx]                                       # (B, 2)
    t_sel   = t_hit[ar, idx]
    contact = cue + t_sel[:, None] * d
    n       = obj - contact
    n       = n / torch.clamp(n.norm(dim=-1, keepdim=True), min=1e-6)
    cos_cut = (d * n).sum(-1)

    pk      = POCKET_CENTERS.to(dev)
    rel_p   = pk[None] - obj[:, None]                             # (B, 6, 2)
    tp      = torch.clamp((rel_p * n[:, None]).sum(-1), min=0.0)  # (B, 6)
    gap     = (pk[None] - (obj[:, None] + tp[..., None] * n[:, None])).norm(dim=-1)
    pi      = gap.argmin(dim=1)
    miss    = torch.clamp(gap[ar, pi] / 0.5, max=1.0)
    d_obj   = tp[ar, pi] / _DIAG
    d_cue   = t_sel / _DIAG

    hit_vals = torch.stack([cos_cut, d_cue, miss, d_obj], dim=-1)
    no_hit   = torch.tensor(_NO_HIT, device=dev).expand(B, 4)
    body     = torch.where(hit[:, None], hit_vals, no_hit)
    speed    = (action[:, 1:2] - 0.5) / 7.5
    return torch.cat([hit[:, None].float(), body, speed], dim=-1)


class GeomCritic(nn.Module):
    """Twin Q over [obs, action, geom_features(obs, action)]."""

    def __init__(self, obs_dim: int, action_dim: int, n_balls: int,
                 hidden: list[int] = [256, 256]):
        super().__init__()
        self.n_balls = n_balls
        self.Q1 = _mlp(obs_dim + action_dim + GEOM_DIM, 1, hidden)
        self.Q2 = _mlp(obs_dim + action_dim + GEOM_DIM, 1, hidden)

    def inputs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, action, geom_features(obs, action, self.n_balls)], dim=-1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = self.inputs(obs, action)
        return self.Q1(x), self.Q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
