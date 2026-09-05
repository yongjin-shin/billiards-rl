"""
world_model/fixeddt_model.py — Fixed-Δt World Model

Architecture:
    StateEncoder φ  : s(14) + pocket_dists(12) → z(128)   ← SAC Critic이 공유
    Transition f    : z(128) → z'(128)
    StateHead       : z'(128) → ŝ(14)
    CollisionHead   : z(128) → (p_coll, type_logit[4])

포켓 거리: MarkovTransition과 동일하게 6포켓 × 2공 = 12 dim 추가.
입력 실질 dim = 14 + 12 = 26.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

STATE_DIM    = 14
POCKET_DIM   = 12   # 6 pockets × 2 balls
ENCODER_IN   = STATE_DIM + POCKET_DIM   # 26
LATENT_DIM   = 128
N_COLL_TYPES = 4   # ball_ball, linear, circular, pocket

# 6개 포켓 위치 (normalized: x/TABLE_W, y/TABLE_H)
POCKET_XY_NORM = torch.tensor([
    [0.0, 0.0], [1.0, 0.0],
    [0.0, 0.5], [1.0, 0.5],
    [0.0, 1.0], [1.0, 1.0],
], dtype=torch.float32)   # (6, 2)


def _mlp(in_dim: int, hidden: tuple, out_dim: int, act=nn.SiLU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """
    s(14) → z(128).
    포켓 거리(12)를 내부에서 계산해 26dim으로 확장 후 인코딩.
    SAC Critic이 이 모듈을 통째로 가져감.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = _mlp(ENCODER_IN, (128, 256), latent_dim)
        self.register_buffer("pocket_xy", POCKET_XY_NORM)

    def _pocket_dists(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (B, 2),  pocket_xy: (6, 2)
        diff = xy.unsqueeze(1) - self.pocket_xy.unsqueeze(0)  # (B, 6, 2)
        return diff.norm(dim=-1)  # (B, 6)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s[:, 0:2] = cue_xy,  s[:, 7:9] = tgt_xy
        cue_d = self._pocket_dists(s[:, 0:2])   # (B, 6)
        tgt_d = self._pocket_dists(s[:, 7:9])   # (B, 6)
        x = torch.cat([s, cue_d, tgt_d], dim=-1)  # (B, 26)
        return self.net(x)


class Transition(nn.Module):
    """z(128) → z'(128).  다음 타임스텝 latent 예측."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = _mlp(latent_dim, (256, 256), latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StateHead(nn.Module):
    """z'(128) → ŝ(14).  다음 상태 복원."""

    def __init__(self, latent_dim: int = LATENT_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = _mlp(latent_dim, (128,), state_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CollisionHead(nn.Module):
    """
    z(128) → (p_coll scalar, type_logit[4])
    현재 상태를 보고 다음 Δt 안에 충돌이 있는지, 있다면 어떤 타입인지 예측.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_types: int = N_COLL_TYPES):
        super().__init__()
        self.shared = _mlp(latent_dim, (128,), 64)
        self.p_head = nn.Linear(64, 1)       # 충돌 있냐 binary
        self.type_head = nn.Linear(64, n_types)  # 어떤 타입

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.shared(z))
        p_coll = self.p_head(h).squeeze(-1)        # (B,)  logit
        type_logit = self.type_head(h)             # (B, 4)
        return p_coll, type_logit


class FixedDtWorldModel(nn.Module):
    """
    Full world model.

    forward(s_t) → (z_t, z_t1, s_hat_t1, p_coll_logit, type_logit)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder   = StateEncoder(latent_dim)
        self.transition = Transition(latent_dim)
        self.state_head = StateHead(latent_dim, STATE_DIM)
        self.coll_head  = CollisionHead(latent_dim)

    def forward(self, s_t: torch.Tensor):
        z_t  = self.encoder(s_t)           # (B, 128)
        z_t1 = self.transition(z_t)        # (B, 128)
        s_hat_t1 = self.state_head(z_t1)   # (B, 14)

        # 충돌 예측은 현재 latent에서
        p_coll, type_logit = self.coll_head(z_t)

        return z_t, z_t1, s_hat_t1, p_coll, type_logit

    def rollout(self, s_0: torch.Tensor, n_steps: int):
        """
        Differentiable rollout.
        Returns:
            states_hat  (B, n_steps, 14)
            p_colls     (B, n_steps)      logits
            type_logits (B, n_steps, 4)
        """
        B = s_0.shape[0]
        states_hat  = []
        p_colls     = []
        type_logits = []

        s = s_0
        for _ in range(n_steps):
            z   = self.encoder(s)
            z1  = self.transition(z)
            s1  = self.state_head(z1)
            p_c, t_l = self.coll_head(z)

            states_hat.append(s1)
            p_colls.append(p_c)
            type_logits.append(t_l)
            s = s1

        return (
            torch.stack(states_hat, dim=1),    # (B, T, 14)
            torch.stack(p_colls, dim=1),       # (B, T)
            torch.stack(type_logits, dim=1),   # (B, T, 4)
        )


def world_model_loss(
    s_hat_t1:     torch.Tensor,          # (B, 14)
    s_t1:         torch.Tensor,          # (B, 14)
    p_coll:       torch.Tensor,          # (B,)   logit
    coll_flag:    torch.Tensor,          # (B,)   bool
    type_logit:   torch.Tensor,          # (B, 4)
    coll_type:    torch.Tensor,          # (B,)   int
    class_weights: torch.Tensor | None = None,  # (4,) inverse-freq weights
    w_state: float = 1.0,
    w_coll:  float = 2.0,
    w_type:  float = 1.0,
) -> Tuple[torch.Tensor, dict]:

    # 1. State prediction loss
    loss_state = F.mse_loss(s_hat_t1, s_t1)

    # 2. Collision detection loss (binary)
    loss_coll = F.binary_cross_entropy_with_logits(
        p_coll, coll_flag.float()
    )

    # 3. Collision type loss (only where collision happened, with class weights)
    mask = coll_flag
    if mask.any():
        loss_type = F.cross_entropy(
            type_logit[mask], coll_type[mask].long(),
            weight=class_weights,
        )
    else:
        loss_type = torch.tensor(0.0, device=p_coll.device)

    total = w_state * loss_state + w_coll * loss_coll + w_type * loss_type

    return total, {
        "loss_state": loss_state.item(),
        "loss_coll":  loss_coll.item(),
        "loss_type":  loss_type.item(),
        "total":      total.item(),
    }
