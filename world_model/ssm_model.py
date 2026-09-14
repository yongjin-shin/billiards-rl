"""
world_model/ssm_model.py — Deterministic SSM World Model (v16: 5-class unified head)

구조식:
  z_0        = ψ(s_0)
  z_{t+1}    = LayerNorm(z_t + f_θ(z_t, ar_state_t))
  ar_state_t = [g_cue(z_t); g_tgt(z_t); softmax(h_type(z_t))]  (19dim)
             = from GT (teacher forcing) or decoded pred (inference) or BERT masking
  ŝ_cue_t    = g_cue(z_t)   절대값 (dims 0:7)
  ŝ_tgt_t    = g_tgt(z_t)   절대값 (dims 7:14)
  ŷ_type_t   = h_type(z_t)  5-class logit (0=no_coll, 1=ball_ball, 2=linear, 3=circular, 4=pocket)

Scheduled Sampling (ss_ratio):
  ss_ratio=1.0 : GT ar_state (teacher forcing); gt_states=(B,T,19)
  ss_ratio=0.0 : decoded ar_state (pure inference)
  0 < ss < 1   : per-sample random mix at each step
  ss_ratio < 0 : BERT-style per-feature masking
                  max_mask = abs(ss_ratio)
                  ss_b ~ U(0, max_mask) per batch
                  mask[b,d] ~ Bernoulli(ss_b)

Loss:
  state_loss : cue + tgt MSE (전 스텝)
  type_loss  : 5-class CE (전 스텝, no_coll=0 포함, class weights)
               정규화: ÷ log(5)
  Kendall (2,): [state, type]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from world_model.fixeddt_model import StateEncoder

LATENT_DIM   = 128
CUE_DIM      = 7     # state dims 0:7  (xy, vxvy, wxwywz)
TGT_DIM      = 7     # state dims 7:14 (xy, vxvy, wxwywz)
N_COLL_TYPES = 5     # 0=no_coll, 1=ball_ball, 2=linear, 3=circular, 4=pocket
AR_DIM       = 19    # ar state: [g_cue(7); g_tgt(7); softmax_type(5)]


class ResTransition(nn.Module):
    """
    use_ar_state=True  (v16/v17):
        z_{t+1} = LayerNorm(z_t + MLP(concat(z_t, ar_state_t)))
    use_ar_state=False (v18):
        z_{t+1} = LayerNorm(z_t + MLP(z_t))   — pure latent dynamics
    """

    def __init__(self, latent_dim: int = LATENT_DIM, use_ar_state: bool = True):
        super().__init__()
        in_dim = latent_dim + (AR_DIM if use_ar_state else 0)
        self.use_ar_state = use_ar_state
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),    nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, ar_state: torch.Tensor | None = None) -> torch.Tensor:
        inp = torch.cat([z, ar_state], dim=-1) if self.use_ar_state else z
        return self.norm(z + self.net(inp))


class CueBallHead(nn.Module):
    """z → cue ball absolute state (B, 7)  — dims 0:7"""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, CUE_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TgtBallHead(nn.Module):
    """z → target ball absolute state (B, 7)."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, TGT_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TypeHead(nn.Module):
    """z → 5-class collision type logit (B, 5)."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, N_COLL_TYPES),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SSMWorldModel(nn.Module):
    """
    Deterministic SSM with 5-class unified collision head.

    use_ar_state=True  (v16/v17): AR decoded-state feedback into transition
    use_ar_state=False (v18):     pure latent dynamics — z must be self-sufficient

    forward(s_0, T) → (s_hat, type_logit)
      s_hat[:, :, 0:7]  = cue absolute
      s_hat[:, :, 7:14] = tgt absolute
      type_logit         = (B, T, 5) collision type logit
    """

    def __init__(self, latent_dim: int = LATENT_DIM, use_ar_state: bool = True):
        super().__init__()
        self.use_ar_state = use_ar_state
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ResTransition(latent_dim, use_ar_state)
        self.cue_head   = CueBallHead(latent_dim)
        self.tgt_head   = TgtBallHead(latent_dim)
        self.type_head  = TypeHead(latent_dim)

    def encode(self, s_0: torch.Tensor) -> torch.Tensor:
        return self.encoder(s_0)

    def _decode_ar_state(self, z: torch.Tensor, gt_states, t: int,
                         ss_ratio: float) -> torch.Tensor:
        """
        매 rollout step t에서 ar_state (B, 19) 계산.

        decoded = [cue_head(z); tgt_head(z); softmax(type_head(z))]  19dim
        gt_states: (B, T, 19) = [physical GT(14); type one-hot(5)]

        ss_ratio=1.0  → GT 전체 (teacher forcing)
        ss_ratio=0.0  → decoded만 (pure inference)
        0<ss<1        → per-sample mix (use_gt ~ Bernoulli(ss_ratio))
        ss_ratio<0    → BERT-style masking:
                         max_mask = abs(ss_ratio)
                         ss_b ~ U(0, max_mask) per batch
                         mask[b,d] ~ Bernoulli(ss_b) per feature
        """
        B = z.shape[0]
        decoded = torch.cat([
            self.cue_head(z),
            self.tgt_head(z),
            torch.softmax(self.type_head(z), dim=-1),
        ], dim=-1)  # (B, 19)

        if gt_states is None or ss_ratio == 0.0:
            return decoded

        if ss_ratio < 0.0:
            max_mask = abs(ss_ratio)
            ss_b = torch.rand(1).item() * max_mask
            mask = torch.rand(B, AR_DIM, device=z.device) < ss_b
            return torch.where(mask, gt_states[:, t], decoded)

        if ss_ratio >= 1.0:
            return gt_states[:, t]

        use_gt = (torch.rand(B, device=z.device) < ss_ratio).unsqueeze(-1)  # (B, 1)
        return torch.where(use_gt, gt_states[:, t], decoded)

    def rollout(self, z_0: torch.Tensor, n_steps: int,
                gt_states=None, ss_ratio: float = 0.0) -> torch.Tensor:
        """
        Returns zs (B, n_steps+1, latent_dim) — z_0 포함.
        gt_states/ss_ratio: use_ar_state=True일 때만 유효.
        """
        zs = [z_0]
        z = z_0
        for t in range(n_steps):
            if self.use_ar_state:
                ar_state = self._decode_ar_state(z, gt_states, t, ss_ratio)
                z = self.transition(z, ar_state)
            else:
                z = self.transition(z)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def forward(self, s_0: torch.Tensor, n_steps: int,
                gt_states=None, ss_ratio: float = 0.0):
        """
        Returns:
            s_hat      (B, n_steps+1, 14)
            type_logit (B, n_steps, 5)   5-class collision type
        gt_states/ss_ratio: use_ar_state=True일 때만 유효.
        """
        z_0 = self.encode(s_0)
        zs  = self.rollout(z_0, n_steps, gt_states=gt_states,
                           ss_ratio=ss_ratio)   # (B, T+1, d)

        B, Tp1, d = zs.shape
        zf = zs.view(B * Tp1, d)

        cue_hat = self.cue_head(zf).view(B, Tp1, CUE_DIM)
        tgt_hat = self.tgt_head(zf).view(B, Tp1, TGT_DIM)
        s_hat = torch.cat([cue_hat, tgt_hat], dim=-1)  # (B, T+1, 14)

        zs_type = zs[:, :-1].contiguous().view(B * n_steps, d)
        type_logit = self.type_head(zs_type).view(B, n_steps, N_COLL_TYPES)

        return s_hat, type_logit


def _focal_cross_entropy(
    logits:  torch.Tensor,          # (N, C)
    targets: torch.Tensor,          # (N,) long
    weight:  torch.Tensor | None,   # (C,) or None
    gamma:   float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss with optional label smoothing: blends hard target and uniform distribution."""
    log_p  = F.log_softmax(logits, dim=-1)               # (N, C)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (N,)
    pt     = log_pt.exp()
    focal_factor = (1.0 - pt) ** gamma
    if label_smoothing > 0.0:
        smooth_loss = -log_p.mean(dim=-1)                # (N,) uniform target
        hard_loss   = -log_pt
        per_sample  = focal_factor * ((1 - label_smoothing) * hard_loss
                                      + label_smoothing * smooth_loss)
    else:
        per_sample = focal_factor * (-log_pt)
    if weight is not None:
        w = weight[targets]
        return (per_sample * w).sum() / (w.sum() + 1e-8)
    return per_sample.mean()


def ssm_rollout_loss(
    s_hat:        torch.Tensor,          # (B, T+1, 14)
    seq_s:        torch.Tensor,          # (B, T+1, 14)  ground truth
    type_logit:   torch.Tensor,          # (B, T, 5)     5-class logit
    seq_types:    torch.Tensor,          # (B, T)        int  0=no_coll 1-4=types
    class_weights: torch.Tensor | None = None,   # [1.0, 4.1, 4.1, 4.1, 4.1]
    log_sigma:    torch.Tensor | None = None,    # (2,) Kendall [state, type]
    w_state: float = 1.0,
    w_type:  float = 1.0,
    focal_gamma: float = 0.0,            # 0 = standard CE, >0 = focal loss
    label_smoothing: float = 0.0,        # 0 = hard labels, >0 = soft labels
) -> Tuple[torch.Tensor, dict]:
    """
    Multi-step rollout loss with 5-class unified collision head.

    state_loss : cue + tgt MSE (전 스텝)
    type_loss  : 5-class CE (전 스텝, no_coll=0 포함) ÷ log(5)
    """
    # 1. State MSE
    cue_sq   = (s_hat[:, 1:, :CUE_DIM] - seq_s[:, 1:, :CUE_DIM]) ** 2
    loss_cue = cue_sq.mean()
    tgt_sq   = (s_hat[:, 1:, CUE_DIM:] - seq_s[:, 1:, CUE_DIM:]) ** 2
    loss_tgt = tgt_sq.mean()
    loss_state = loss_cue + loss_tgt

    # 2. 5-class CE or Focal (전 스텝, no_coll=0 포함)
    B, T, _ = type_logit.shape
    flat_logit = type_logit.reshape(B * T, N_COLL_TYPES)
    flat_types = seq_types.reshape(B * T).long()
    if focal_gamma > 0.0:
        loss_type = _focal_cross_entropy(flat_logit, flat_types, class_weights,
                                         focal_gamma, label_smoothing)
    else:
        loss_type = F.cross_entropy(flat_logit, flat_types, weight=class_weights,
                                    label_smoothing=label_smoothing)
    loss_type_n = loss_type / math.log(N_COLL_TYPES)  # ÷ log(5)

    if log_sigma is not None:
        # Kendall (2018) uncertainty weighting: L = exp(-s)*L_i + s
        s = log_sigma.clamp(-6, 6)
        total = (torch.exp(-s[0]) * loss_state  + s[0] +
                 torch.exp(-s[1]) * loss_type_n + s[1])
        eff = torch.exp(-s).detach()
        extra = {
            "kw_state": eff[0].item(),
            "kw_type":  eff[1].item(),
        }
    else:
        total = w_state * loss_state + w_type * loss_type_n
        extra = {}

    return total, {
        "loss_cue":   loss_cue.item(),
        "loss_tgt":   loss_tgt.item(),
        "loss_state": loss_state.item(),
        "loss_type":  loss_type.item(),
        "total":      total.item(),
        **extra,
    }
