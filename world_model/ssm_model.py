"""
world_model/ssm_model.py — Deterministic SSM World Model

AR(fixeddt_model.py)과의 핵심 차이:
  AR:  s_t →enc→ z_t →trans→ z_{t+1} →dec→ ŝ_{t+1} →enc→ z_{t+1} → ...
  SSM: s_0 →enc→ z_0 →trans→ z_1 →trans→ z_2 → ...  (z space에서만)
       decode는 loss 계산용으로만.

구조식:
  z_0     = ψ(s_0)
  z_{t+1} = z_t + f_θ(z_t)   (ResTransition — skip connection)
  ŝ_t     = g_φ(z_t)
  (p̂_coll_t, ŷ_type_t) = h(z_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from world_model.fixeddt_model import (
    StateEncoder, StateHead, CollisionHead,
    POCKET_XY_NORM, STATE_DIM, N_COLL_TYPES,
)

LATENT_DIM = 64   # AR(128)에서 축소 — encode 한 번만이라 충분


class ResTransition(nn.Module):
    """
    z(64) → z(64), skip connection.
    z_{t+1} = z_t + MLP(z_t)
    gradient vanishing 방지 (T step 역전파).
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128),        nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


class SSMWorldModel(nn.Module):
    """
    Deterministic SSM.

    encode(s_0) → z_0
    rollout(z_0, T) → zs (B, T+1, 64)   — z space에서만
    forward(s_0, T) → (s_hat, p_coll, type_logit)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ResTransition(latent_dim)
        self.state_head = StateHead(latent_dim, STATE_DIM)
        self.coll_head  = CollisionHead(latent_dim)

    def encode(self, s_0: torch.Tensor) -> torch.Tensor:
        """s_0(B,14) → z_0(B,64)"""
        return self.encoder(s_0)

    def rollout(self, z_0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        z space에서만 n_steps 굴림.
        Returns zs (B, n_steps+1, latent_dim)  — z_0 포함.
        """
        zs = [z_0]
        z = z_0
        for _ in range(n_steps):
            z = self.transition(z)
            zs.append(z)
        return torch.stack(zs, dim=1)   # (B, T+1, d)

    def forward(self, s_0: torch.Tensor, n_steps: int):
        """
        s_0 → z_0 → rollout → decode.

        Returns:
            s_hat      (B, n_steps+1, 14)  — ŝ_0..ŝ_T
            p_coll     (B, n_steps)         — logit, 구간 t의 충돌 여부
            type_logit (B, n_steps, 4)
        """
        z_0 = self.encode(s_0)
        zs  = self.rollout(z_0, n_steps)        # (B, T+1, d)

        # flatten for batch processing
        B, Tp1, d = zs.shape
        zs_flat = zs.view(B * Tp1, d)

        s_hat_flat = self.state_head(zs_flat)   # (B*(T+1), 14)
        s_hat = s_hat_flat.view(B, Tp1, STATE_DIM)

        # collision: z_0..z_{T-1}가 각 구간 예측
        zs_coll = zs[:, :-1].contiguous().view(B * n_steps, d)
        p_coll_flat, type_flat = self.coll_head(zs_coll)
        p_coll     = p_coll_flat.view(B, n_steps)
        type_logit = type_flat.view(B, n_steps, N_COLL_TYPES)

        return s_hat, p_coll, type_logit


def ssm_rollout_loss(
    s_hat:        torch.Tensor,          # (B, T+1, 14)
    seq_s:        torch.Tensor,          # (B, T+1, 14)  ground truth
    p_coll:       torch.Tensor,          # (B, T)   logit
    seq_flags:    torch.Tensor,          # (B, T)   bool
    type_logit:   torch.Tensor,          # (B, T, 4)
    seq_types:    torch.Tensor,          # (B, T)   int
    class_weights: torch.Tensor | None = None,
    w_state: float = 1.0,
    w_coll:  float = 2.0,
    w_type:  float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Multi-step rollout loss.
    s_hat[:, 0]은 s_0 재구성 (선택적), 주 loss는 s_1..s_T.
    """
    # 1. State loss: s_1..s_T 예측 (s_0는 encode 입력이라 trivial)
    loss_state = F.mse_loss(s_hat[:, 1:], seq_s[:, 1:])

    # 2. Collision detection
    loss_coll = F.binary_cross_entropy_with_logits(
        p_coll, seq_flags.float()
    )

    # 3. Collision type (충돌 있는 스텝만)
    mask = seq_flags.bool()
    if mask.any():
        loss_type = F.cross_entropy(
            type_logit[mask], seq_types[mask].long(),
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
