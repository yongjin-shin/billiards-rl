"""
world_model/spr_mdn/spr_mdn_model.py — SPR-MDN World Model (Laplace)

Self-Predictive Representations + Laplace Mixture Density Network.

Key design:
  1. MixtureHead: K-component Laplace MDN conditioned on [z; a_tilde]
  2. Self-chaining: z_{h+1}_hat = sg(sample from MDN) — never teacher-forced
  3. EMA encoder φ' provides stable NLL targets with stop-gradient
  4. L_recon grounds z to real ball states at every step
  5. LayerNorm on encoder output prevents z scale blowup
"""

import copy
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.fixeddt_model import StateEncoder
from world_model.ssm_model import CueBallHead, TgtBallHead, TypeHead, _focal_cross_entropy

LATENT_DIM    = 128
N_COMPONENTS  = 5
N_COLL_TYPES  = 5
ACTION_DIM    = 2
EMA_TAU       = 0.99


def _mlp(*dims: int, act=nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class EncoderLN(nn.Module):
    """StateEncoder + LayerNorm — prevents z scale blowup."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.enc = StateEncoder(latent_dim)
        self.ln  = nn.LayerNorm(latent_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.ln(self.enc(s))

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse)


class MixtureHead(nn.Module):
    """
    [z; a_tilde] → (π, μ, b)   K-component Laplace MDN transition.

    Input:  z (B, D), a_tilde (B, m)
    Output: π (B, K), μ (B, K, D), b (B, K, D)
      π = softmax(logits)
      b = softplus(b_raw) + ε   (ε = 0.01, b_raw bias init to 2.0)

    asym_init=True: component k gets +1.0 bias in dim k of its μ slice,
    breaking the symmetric initialization that causes π-collapse.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_components: int = N_COMPONENTS,
                 action_dim: int = ACTION_DIM, asym_init: bool = False):
        super().__init__()
        self.K = n_components
        self.D = latent_dim
        in_dim  = latent_dim + action_dim
        out_dim = n_components + n_components * latent_dim * 2   # logits + μ + b_raw
        self.net = _mlp(in_dim, 256, 256, out_dim)
        # b_raw bias → 2.0 so initial b ≈ softplus(2) + 0.01 ≈ 2.13, prevents early collapse
        b_raw_start = n_components + n_components * latent_dim
        nn.init.constant_(self.net[-1].bias[b_raw_start:], 2.0)

        if asym_init:
            # Break μ symmetry: component k gets +1.0 bias in unique dim k.
            # After encoder LN, latent dims are ~unit-variance, so scale=1.0 is meaningful.
            mu_bias_start = n_components
            with torch.no_grad():
                for k in range(n_components):
                    self.net[-1].bias[mu_bias_start + k * latent_dim + k] = 1.0

    def forward(self, z: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B    = z.shape[0]
        K, D = self.K, self.D
        h    = self.net(torch.cat([z, a], dim=-1))   # (B, K + 2*K*D)

        log_pi = h[:, :K]                            # (B, K)
        mu     = h[:, K : K + K * D].view(B, K, D)  # (B, K, D)
        b_raw  = h[:, K + K * D :].view(B, K, D)    # (B, K, D)

        pi = F.softmax(log_pi, dim=-1)               # (B, K) — sums to 1
        b  = F.softplus(b_raw) + 0.01               # (B, K, D) — strictly positive

        return pi, mu, b


def ewta_with_pi_loss(
    pi:     torch.Tensor,   # (B, K)
    mu:     torch.Tensor,   # (B, K, D)
    b:      torch.Tensor,   # (B, K, D) — unused, kept for API uniformity
    target: torch.Tensor,   # (B, D)
    kappa:  int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combined EWTA L2 + simultaneous π loss.

    loss_ewta: top-κ L2 → gradient flows only to μ (and shared MLP)
    loss_pi:   soft CE toward top-κ winners → gradient flows only to π logits

    This is the v8 fix: π learns *which component wins* at every step
    so that Categorical(π) in rollout increasingly samples the right component,
    cleaning up the reconstruction target.

    Soft CE target: uniform 1/κ over the κ closest components, 0 elsewhere.
    """
    target_e = target.unsqueeze(1)                           # (B, 1, D)
    sq_err_k = (mu - target_e).pow(2).sum(-1)               # (B, K)
    dists    = sq_err_k.detach()
    _, top_idx = dists.topk(kappa, dim=-1, largest=False)   # (B, κ)

    # μ gradient: top-κ L2
    top_sq_err = sq_err_k.gather(dim=-1, index=top_idx)     # (B, κ)
    loss_ewta  = top_sq_err.mean()

    # π gradient: soft CE — no gradient to μ via pi_target (it's from dists.detach)
    pi_target = torch.zeros_like(pi)                         # (B, K)
    pi_target.scatter_(1, top_idx, 1.0 / kappa)
    log_pi   = pi.log().clamp(min=-1e9)
    loss_pi  = -(pi_target * log_pi).sum(-1).mean()

    return loss_ewta, loss_pi


def ewta_laplace_loss(
    mu:     torch.Tensor,   # (B, K, D) — trainable
    b:      torch.Tensor,   # (B, K, D) — unused in Phase 1, kept for API compat
    target: torch.Tensor,   # (B, D)
    kappa:  int,            # top-κ closest components get gradient
) -> torch.Tensor:
    """
    EWTA (Evolving Winner-Takes-All) L2 loss for Phase 1.

    Uses squared L2 distance, NOT Laplace NLL. Reasons:
      - Laplace NLL allows b→0 as a trivial minimizer (makes log(2b)→−∞),
        driving training loss arbitrarily negative while val explodes.
      - L2 is always ≥ 0, b is not involved, no collapse risk.
      - Makansi et al. CVPR 2019 Phase 1 is also regression-based.

    π is intentionally absent — logit head receives zero gradient.
    b receives zero gradient — Phase 2 trains π via standard NLL.
    Only μ (and shared MLP upstream) receives gradient.
    """
    target_e = target.unsqueeze(1)                               # (B, 1, D)
    sq_err_k = (mu - target_e).pow(2).sum(-1)                    # (B, K) — L2²
    dists    = sq_err_k.detach()                                  # rank by detached dist
    _, top_idx = dists.topk(kappa, dim=-1, largest=False)        # (B, κ)

    top_sq_err = sq_err_k.gather(dim=-1, index=top_idx)          # (B, κ)
    return top_sq_err.mean()


def laplace_nll_mixture(
    pi:     torch.Tensor,   # (B, K)
    mu:     torch.Tensor,   # (B, K, D)
    b:      torch.Tensor,   # (B, K, D)
    target: torch.Tensor,   # (B, D)
) -> torch.Tensor:
    """
    NLL of target under Laplace mixture, batch-averaged.

    log Laplace(x; μ_k, b_k) = -Σ_i [log(2 b_ki) + |x_i - μ_ki| / b_ki]
    NLL = -logsumexp_k (log π_k + log Laplace(x; μ_k, b_k))
    """
    target = target.unsqueeze(1)                        # (B, 1, D)
    log_prob_k = -(
        torch.log(2.0 * b) + torch.abs(target - mu) / b
    ).sum(-1)                                           # (B, K)

    log_pi  = pi.log().clamp(min=-1e9)
    log_mix = torch.logsumexp(log_pi + log_prob_k, dim=-1)   # (B,)
    return -log_mix.mean()


class SPRMDNModel(nn.Module):

    def __init__(
        self,
        latent_dim:      int   = LATENT_DIM,
        n_components:    int   = N_COMPONENTS,
        action_dim:      int   = ACTION_DIM,
        ema_tau:         float = EMA_TAU,
        asym_init:       bool  = False,
        use_ema:         bool  = True,
        use_encoder_ln:  bool  = True,
    ):
        super().__init__()
        self.latent_dim      = latent_dim
        self.n_components    = n_components
        self.action_dim      = action_dim
        self.ema_tau         = ema_tau
        self.use_ema         = use_ema
        self.use_encoder_ln  = use_encoder_ln

        # Trainable modules
        self.encoder      = EncoderLN(latent_dim) if use_encoder_ln else StateEncoder(latent_dim)
        self.mixture_head = MixtureHead(latent_dim, n_components, action_dim, asym_init=asym_init)
        self.cue_head     = CueBallHead(latent_dim)
        self.tgt_head     = TgtBallHead(latent_dim)
        self.type_head    = TypeHead(latent_dim)

        # Dedicated chain LN when use_encoder_ln=False (v18-parity: no encoder LN, separate LN for chain)
        # When use_encoder_ln=True, encoder.ln is reused (already exists on EncoderLN)
        if not use_encoder_ln:
            self.chain_ln = nn.LayerNorm(latent_dim)

        # EMA target encoder — only when use_ema=True; SimSiam uses stopgrad on self.encoder
        if use_ema:
            self.ema_encoder = copy.deepcopy(self.encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self) -> None:
        if not self.use_ema:
            return
        tau = self.ema_tau
        for p, p_ema in zip(self.encoder.parameters(),
                            self.ema_encoder.parameters()):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) → s_hat: (B, 14)"""
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    def _apply_chain_ln(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize mixture mean before chaining. Mirrors SPRK1Model v14 structure."""
        if self.use_encoder_ln:
            return self.encoder.ln(z)   # type: ignore[union-attr]
        return self.chain_ln(z)

    def rollout_train(
        self,
        s_0:    torch.Tensor,   # (B, 14)
        seq_s:  torch.Tensor,   # (B, T+1, 14)
        action: torch.Tensor,   # (B, m)
        T:      int,
        p_tf:   float = 0.0,    # teacher-forcing probability (scheduled sampling)
    ) -> Tuple[List, List, List, List, List]:
        """
        Training rollout with mixture mean chaining (v19+).

        p_tf=1.0: 완전 teacher forcing
        p_tf=0.0: 완전 self-chaining (mixture mean, 완전 미분가능)

        매 스텝 h≥1마다 Bernoulli(p_tf)로 결정:
          c=1 → z_hat = sg(Enc_phi(s_{t+h}))           (teacher forcing)
          c=0 → z_hat = encoder.ln(Σ_k π_k · μ_k)     (mixture mean chain, gradient 유지)

        Target: use_ema=True → EMA encoder, use_ema=False → SimSiam (stopgrad via no_grad)

        Returns: z_hat_list[T+1], pi_list[T], mu_list[T], b_list[T], z_bar_list[T]
        """
        B      = s_0.shape[0]
        device = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_hat = self.encoder(s_0)    # (B, D) — h=0, real encoding
        z_hat_list = [z_hat]
        pi_list, mu_list, b_list, z_bar_list = [], [], [], []

        # 시퀀스 단위로 한 번만 결정 — h마다 재굴리면 GT리셋 효과 생김
        use_tf = (p_tf > 0.0 and torch.rand(1).item() < p_tf)

        for h in range(T):
            a_tilde = action if h == 0 else a_zeros

            pi, mu, b = self.mixture_head(z_hat, a_tilde)

            with torch.no_grad():
                target_enc = self.ema_encoder if self.use_ema else self.encoder
                z_bar = target_enc(seq_s[:, h + 1])
            z_bar_list.append(z_bar)

            pi_list.append(pi)
            mu_list.append(mu)
            b_list.append(b)

            if use_tf:
                with torch.no_grad():
                    z_hat = self.encoder(seq_s[:, h + 1])
            else:
                # Mixture mean chaining — gradient flows through pi and mu
                z_mean = (pi.unsqueeze(-1) * mu).sum(dim=1)  # (B, D)
                z_hat = self._apply_chain_ln(z_mean)

            z_hat_list.append(z_hat)

        return z_hat_list, pi_list, mu_list, b_list, z_bar_list

    @torch.no_grad()
    def rollout_eval(
        self,
        s_0:    torch.Tensor,          # (B, 14)
        T:      int,
        action: torch.Tensor | None = None,  # (B, m) optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic eval rollout using mixture mean: z_{h+1} = Σ_k π_k μ_k.
        Returns: s_hat (B, T+1, 14), type_logit (B, T, 5)
        """
        B      = s_0.shape[0]
        device = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_hat           = self.encoder(s_0)
        s_hat_list      = [self._decode(z_hat)]
        type_logit_list = []

        for h in range(T):
            a_tilde = action if (h == 0 and action is not None) else a_zeros
            pi, mu, b = self.mixture_head(z_hat, a_tilde)
            type_logit_list.append(self.type_head(z_hat))
            # Mixture mean, reuse encoder's LN for scale consistency
            z_hat = self._apply_chain_ln((pi.unsqueeze(-1) * mu).sum(dim=1))   # (B, D)
            s_hat_list.append(self._decode(z_hat))

        s_hat      = torch.stack(s_hat_list, dim=1)       # (B, T+1, 14)
        type_logit = torch.stack(type_logit_list, dim=1)  # (B, T, 5)
        return s_hat, type_logit

    def forward(
        self,
        s_0:    torch.Tensor,   # (B, 14)
        seq_s:  torch.Tensor,   # (B, T+1, 14)
        action: torch.Tensor,   # (B, m)
        T:      int,
        p_tf:   float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List, List, List, List]:
        """
        Training forward.
        Returns:
          s_hat       (B, T+1, 14)
          type_logit  (B, T, 5)
          z_hat_list, pi_list, mu_list, b_list, z_bar_list
        """
        z_hat_list, pi_list, mu_list, b_list, z_bar_list = \
            self.rollout_train(s_0, seq_s, action, T, p_tf=p_tf)

        z_stack    = torch.stack(z_hat_list, dim=1)           # (B, T+1, D)
        s_hat      = torch.stack(
            [self._decode(z_stack[:, t]) for t in range(T + 1)], dim=1
        )                                                     # (B, T+1, 14)
        type_logit = torch.stack(
            [self.type_head(z_stack[:, t]) for t in range(T)], dim=1
        )                                                     # (B, T, 5)

        return s_hat, type_logit, z_hat_list, pi_list, mu_list, b_list, z_bar_list


def spr_rollout_loss(
    pi_list:   List[torch.Tensor],   # [T] each (B, K)
    mu_list:   List[torch.Tensor],   # [T] each (B, K, D)
    b_list:    List[torch.Tensor],   # [T] each (B, K, D)
    z_bar_list: List[torch.Tensor],  # [T] each (B, D)  — detached EMA targets
    s_hat:     torch.Tensor,         # (B, T+1, 14)
    seq_s:     torch.Tensor,         # (B, T+1, 14)
    type_logit: torch.Tensor,        # (B, T, 5)
    seq_types: torch.Tensor,         # (B, T) int  0–4
    class_weights: torch.Tensor | None = None,
    focal_gamma:     float = 2.0,
    label_smoothing: float = 0.0,
    lam_recon:       float = 1.0,
    lam_pi:          float = 0.0,   # >0 → v8: simultaneous π training via EWTA winner
    ewta_kappa:      int   = 0,      # >0 → EWTA active; 0 → standard NLL
    ewta_phase2:     bool  = False,  # True → Phase 2 (v7): π-only NLL, sg(μ,b)
) -> Tuple[torch.Tensor, dict]:
    """
    Four modes via (ewta_kappa, lam_pi, ewta_phase2):
      (0,  0,  False) standard: full mixture NLL
      (κ>0,0,  False) v7 ph1:   EWTA L2 — only μ gets gradient, π frozen
      (0,  0,  True)  v7 ph2:   π-only NLL with sg(μ,b)
      (κ>0,>0, False) v8:       EWTA L2 for μ  +  soft-CE L_π simultaneously

    L_recon and L_type are identical across all modes.
    """
    T = len(pi_list)
    zero = pi_list[0].new_zeros(())

    # ── NLL + optional L_π (mode-dependent) ──────────────────────────────
    if ewta_kappa > 0 and lam_pi > 0:
        # v8: EWTA μ spread + simultaneous π training
        ewta_accum = zero
        pi_accum   = zero
        for h in range(T):
            l_ewta, l_pi = ewta_with_pi_loss(
                pi_list[h], mu_list[h], b_list[h], z_bar_list[h], ewta_kappa
            )
            ewta_accum = ewta_accum + l_ewta
            pi_accum   = pi_accum   + l_pi
        loss_nll = ewta_accum / T
        loss_pi  = pi_accum  / T

    elif ewta_kappa > 0:
        # v7 Phase 1: EWTA only — μ gradient, π frozen
        loss_nll = sum(
            ewta_laplace_loss(mu_list[h], b_list[h], z_bar_list[h], ewta_kappa)
            for h in range(T)
        ) / T
        loss_pi = zero

    elif ewta_phase2:
        # v7 Phase 2: π-only NLL, sg on μ and b
        loss_nll = sum(
            laplace_nll_mixture(
                pi_list[h], mu_list[h].detach(), b_list[h].detach(), z_bar_list[h]
            )
            for h in range(T)
        ) / T
        loss_pi = zero

    else:
        # Standard: full mixture NLL
        loss_nll = sum(
            laplace_nll_mixture(pi_list[h], mu_list[h], b_list[h], z_bar_list[h])
            for h in range(T)
        ) / T
        loss_pi = zero

    # ── Recon (all T+1 steps including h=0) ──────────────────────────────
    loss_cue   = F.mse_loss(s_hat[:, :, :7],  seq_s[:, :, :7])
    loss_tgt   = F.mse_loss(s_hat[:, :, 7:],  seq_s[:, :, 7:])
    loss_recon = loss_cue + loss_tgt

    # ── Type ─────────────────────────────────────────────────────────────
    B_T         = type_logit.shape[0] * type_logit.shape[1]
    logits_flat = type_logit.reshape(B_T, N_COLL_TYPES)
    types_flat  = seq_types.reshape(B_T)
    loss_type   = _focal_cross_entropy(
        logits_flat, types_flat, class_weights, focal_gamma, label_smoothing
    ) / math.log(N_COLL_TYPES)

    total = loss_nll + lam_recon * loss_recon + loss_type
    if lam_pi > 0:
        total = total + lam_pi * loss_pi

    detail = {
        "loss_nll":   loss_nll.item(),
        "loss_pi":    loss_pi.item(),
        "loss_recon": loss_recon.item(),
        "loss_cue":   loss_cue.item(),
        "loss_tgt":   loss_tgt.item(),
        "loss_type":  loss_type.item(),
    }
    return total, detail


# ════════════════════════════════════════════════════════════════════════════
# Kendall per-horizon uncertainty weighting (v21+)
# ════════════════════════════════════════════════════════════════════════════

class KendallHWeights(nn.Module):
    """
    Per-horizon Kendall uncertainty weights for NLL and recon losses.

    Formulation (Kendall & Gal 2018, extended to per-step):
        L_h = exp(-s_nll_h)  * (L_nll_h  / D) + s_nll_h
            + exp(-s_recon_h) * L_recon_h       + s_recon_h

    where s_nll_h = log_sigma_nll[h], s_recon_h = log_sigma_recon[h].

    D (LATENT_DIM=128) pre-scaling brings NLL per-dim (~0.77) to the same
    order as L_recon (~0.038), so Kendall handles the residual 20:1 imbalance
    rather than the full 3,363x structural imbalance.

    Parameters: 2 × T_max scalars (default T_max=60).
    Init at 0 → exp(-0)=1, no initial scaling shift.
    """

    def __init__(self, T_max: int = 60,
                 sigma_nll_init: float = 0.0,
                 sigma_recon_init: float = 0.0):
        super().__init__()
        self.T_max = T_max
        self.log_sigma_nll   = nn.Parameter(torch.full((T_max,), float(sigma_nll_init)))
        self.log_sigma_recon = nn.Parameter(torch.full((T_max,), float(sigma_recon_init)))

    def extra_repr(self) -> str:
        return f"T_max={self.T_max}"


def spr_rollout_loss_kendall_h(
    pi_list:       List[torch.Tensor],   # [T] each (B, K)
    mu_list:       List[torch.Tensor],   # [T] each (B, K, D)
    b_list:        List[torch.Tensor],   # [T] each (B, K, D)
    z_bar_list:    List[torch.Tensor],   # [T] each (B, D)
    s_hat:         torch.Tensor,         # (B, T+1, 14)
    seq_s:         torch.Tensor,         # (B, T+1, 14)
    type_logit:    torch.Tensor,         # (B, T, 5)
    seq_types:     torch.Tensor,         # (B, T) int
    kendall:       KendallHWeights,
    class_weights: torch.Tensor | None = None,
    focal_gamma:   float = 2.0,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Kendall-weighted per-horizon loss for SPRMDNModel.

    NLL is pre-scaled by 1/LATENT_DIM before weighting to bring it from
    ~99 (D-dim sum) to ~0.77 (per-dim), then Kendall handles the residual
    ~20x imbalance vs L_recon (~0.038).

    h=0 recon (trivial encode-decode) is excluded; only prediction steps
    h=1..T carry Kendall weights (s_hat[:, h+1, :] vs seq_s[:, h+1, :]).
    """
    T    = len(pi_list)
    zero = pi_list[0].new_zeros(())

    total_weighted = zero
    nll_raw_sum    = 0.0
    recon_raw_sum  = 0.0

    for h in range(T):
        # ── NLL: pre-scale by 1/D ─────────────────────────────────────────
        nll_h = laplace_nll_mixture(
            pi_list[h], mu_list[h], b_list[h], z_bar_list[h]
        ) / LATENT_DIM

        # ── Recon at prediction step h+1 (skip trivial h=0) ──────────────
        recon_h = (
            F.mse_loss(s_hat[:, h + 1, :7], seq_s[:, h + 1, :7]) +
            F.mse_loss(s_hat[:, h + 1, 7:], seq_s[:, h + 1, 7:])
        )

        s_nll = kendall.log_sigma_nll[h]
        s_rec = kendall.log_sigma_recon[h]

        total_weighted = (
            total_weighted
            + torch.exp(-s_nll) * nll_h + s_nll
            + torch.exp(-s_rec) * recon_h + s_rec
        )
        nll_raw_sum   += nll_h.item()
        recon_raw_sum += recon_h.item()

    total_weighted = total_weighted / T

    # ── Type loss (unchanged) ─────────────────────────────────────────────
    B_T         = type_logit.shape[0] * type_logit.shape[1]
    logits_flat = type_logit.reshape(B_T, N_COLL_TYPES)
    types_flat  = seq_types.reshape(B_T)
    loss_type   = _focal_cross_entropy(
        logits_flat, types_flat, class_weights, focal_gamma, label_smoothing
    ) / math.log(N_COLL_TYPES)

    total = total_weighted + loss_type

    sigma_nll_vals   = kendall.log_sigma_nll[:T].detach()
    sigma_recon_vals = kendall.log_sigma_recon[:T].detach()

    detail = {
        "loss_nll":         nll_raw_sum / T,          # per-dim, per-step average
        "loss_recon":       recon_raw_sum / T,
        "loss_type":        loss_type.item(),
        "sigma_nll_mean":   sigma_nll_vals.mean().item(),
        "sigma_nll_h1":     sigma_nll_vals[0].item(),
        "sigma_nll_hT":     sigma_nll_vals[T - 1].item(),
        "sigma_recon_mean": sigma_recon_vals.mean().item(),
        "sigma_recon_h1":   sigma_recon_vals[0].item(),
        "sigma_recon_hT":   sigma_recon_vals[T - 1].item(),
    }
    return total, detail


# ════════════════════════════════════════════════════════════════════════════
# EWTA loss (v23+): soft/annealed Winner-Takes-All L2 for explicit component separation
# ════════════════════════════════════════════════════════════════════════════

def spr_rollout_loss_ewta(
    pi_list:       List[torch.Tensor],   # [T] each (B, K)
    mu_list:       List[torch.Tensor],   # [T] each (B, K, D)
    b_list:        List[torch.Tensor],   # [T] each (B, K, D)  — kept for API, not used in loss
    z_bar_list:    List[torch.Tensor],   # [T] each (B, D)
    s_hat:         torch.Tensor,         # (B, T+1, 14)
    seq_s:         torch.Tensor,         # (B, T+1, 14)
    type_logit:    torch.Tensor,         # (B, T, 5)
    seq_types:     torch.Tensor,         # (B, T) int
    tau:           float = 1.0,          # current temperature (anneal high→low)
    class_weights: torch.Tensor | None = None,
    focal_gamma:   float = 2.0,
    label_smoothing: float = 0.0,
    lam_recon:     float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Soft Winner-Takes-All L2 loss for SPRMDNModel (v23+).

    Replaces NLL with:
        d_k   = ||z_bar − μ_k||² / D          (per-dim MSE, D-normalised)
        w_k   = softmax(log π_k − d_k / τ)    (soft WTA weight)
        L_wta = E_w[d_k] = Σ_k w_k × d_k

    At τ→∞: uniform weights → gradient flows to all components (mode-preserving).
    At τ→0: argmin_k → hard WTA, only winner gets gradient.

    b is not used in this loss (no b-collapse risk). b output from mixture_head
    remains available for diagnosis / future sampling-based rollout.

    L_recon and L_type are unchanged.
    """
    T = len(pi_list)
    D = mu_list[0].shape[-1]  # LATENT_DIM
    zero = pi_list[0].new_zeros(())

    total_wta   = zero
    total_recon = 0.0
    wta_sum     = 0.0

    for h in range(T):
        pi   = pi_list[h]        # (B, K)
        mu   = mu_list[h]        # (B, K, D)
        zbar = z_bar_list[h]     # (B, D)

        # Per-dim MSE per component: d_k = ||z_bar − μ_k||² / D
        zbar_exp = zbar.unsqueeze(1).expand_as(mu)   # (B, K, D)
        d = (zbar_exp - mu).pow(2).sum(dim=-1) / D   # (B, K)

        # Soft-WTA weights: combine predicted π with distance-based winner prob
        log_w = torch.log(pi.clamp(min=1e-8)) - d / tau   # (B, K)
        w = torch.softmax(log_w, dim=-1)                   # (B, K)

        # WTA loss: expected distance under soft assignment
        loss_wta_h = (w * d).sum(dim=-1).mean()   # scalar
        total_wta  = total_wta + loss_wta_h
        wta_sum   += loss_wta_h.item()

        # Recon (unchanged from spr_rollout_loss)
        recon_h = (
            F.mse_loss(s_hat[:, h + 1, :7], seq_s[:, h + 1, :7]) +
            F.mse_loss(s_hat[:, h + 1, 7:], seq_s[:, h + 1, 7:])
        )
        total_recon += recon_h.item()

    total_wta   = total_wta / T
    total_recon_t = (
        sum(
            F.mse_loss(s_hat[:, h + 1, :7], seq_s[:, h + 1, :7]) +
            F.mse_loss(s_hat[:, h + 1, 7:], seq_s[:, h + 1, 7:])
            for h in range(T)
        ) / T
    )

    # Type loss
    B_T         = type_logit.shape[0] * type_logit.shape[1]
    logits_flat = type_logit.reshape(B_T, N_COLL_TYPES)
    types_flat  = seq_types.reshape(B_T)
    loss_type   = _focal_cross_entropy(
        logits_flat, types_flat, class_weights, focal_gamma, label_smoothing
    ) / math.log(N_COLL_TYPES)

    total = total_wta + lam_recon * total_recon_t + loss_type

    # Soft assignment entropy (how spread is w — high = soft/uniform, low = hard/collapsed)
    with torch.no_grad():
        last_pi  = pi_list[-1]   # (B, K)  last step
        last_mu  = mu_list[-1]   # (B, K, D)
        last_z   = z_bar_list[-1]
        zb_exp   = last_z.unsqueeze(1).expand_as(last_mu)
        d_last   = (zb_exp - last_mu).pow(2).sum(dim=-1) / D
        lw_last  = torch.log(last_pi.clamp(min=1e-8)) - d_last / tau
        w_last   = torch.softmax(lw_last, dim=-1)
        max_pi_mean = w_last.max(dim=-1).values.mean().item()
        var_mu_mean = last_mu.var(dim=1).mean().item()

    detail = {
        "loss_wta":   wta_sum / T,
        "loss_recon": total_recon / T,
        "loss_type":  loss_type.item(),
        "tau":        tau,
        "max_pi":     max_pi_mean,
        "var_mu":     var_mu_mean,
    }
    return total, detail


# ════════════════════════════════════════════════════════════════════════════
# Categorical Latent Variable Model (v24): DreamerV3-style Prior/Posterior
# ════════════════════════════════════════════════════════════════════════════

def _unimix(logits: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """(1−α)·softmax(logits) + α/K  — uniform floor, eliminates τ schedule."""
    K = logits.shape[-1]
    return (1.0 - alpha) * torch.softmax(logits, dim=-1) + alpha / K


def _st_onehot(probs: torch.Tensor) -> torch.Tensor:
    """Straight-through: forward=onehot(argmax(probs)), backward through probs."""
    K    = probs.shape[-1]
    hard = F.one_hot(probs.argmax(dim=-1), K).to(probs)   # (B, K)
    return hard - probs.detach() + probs


class CatPriorHead(nn.Module):
    """p(c_h | trunk_out) — K logits, zero-init → perplexity=K at epoch 1."""

    def __init__(self, trunk_dim: int = 256, n_categories: int = N_COMPONENTS):
        super().__init__()
        self.fc = nn.Linear(trunk_dim, n_categories)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, trunk_out: torch.Tensor) -> torch.Tensor:
        return self.fc(trunk_out)   # (B, K)


class CatPosteriorHead(nn.Module):
    """q(c_h | trunk_out, z_target) — zero-init last layer."""

    def __init__(self, latent_dim: int = LATENT_DIM, trunk_dim: int = 256,
                 n_categories: int = N_COMPONENTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim + latent_dim, trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim, n_categories),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, trunk_out: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([trunk_out, z_tgt], dim=-1))   # (B, K)


class BranchOutputHead(nn.Module):
    """Residual correction conditioned on c — zero-init last layer → correction=0 at init."""

    def __init__(self, latent_dim: int = LATENT_DIM, n_categories: int = N_COMPONENTS,
                 trunk_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim + n_categories, trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, trunk_out: torch.Tensor, c_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([trunk_out, c_onehot], dim=-1))   # (B, D)


class SPRCatModel(nn.Module):
    """
    SPR with discrete categorical latent variable (v24).

    Per-step (training):
      trunk_out  = trunk([z_chain; a])
      prior_probs  = unimix(prior_head(trunk_out))
      post_probs   = unimix(post_head([trunk_out; z_tgt]))
      c_onehot     = ST(post_probs)
      z_chain      = chain_ln(z_chain + branch_head([trunk_out; c_onehot]))

    Per-step (eval):
      c_onehot = onehot(argmax(prior_probs))   or   sample(prior_probs)
    """

    def __init__(
        self,
        latent_dim:   int   = LATENT_DIM,
        n_categories: int   = N_COMPONENTS,
        action_dim:   int   = ACTION_DIM,
        ema_tau:      float = EMA_TAU,
        use_ema:      bool  = True,
        alpha:        float = 0.01,
    ):
        super().__init__()
        self.latent_dim   = latent_dim
        self.n_categories = n_categories
        self.action_dim   = action_dim
        self.ema_tau      = ema_tau
        self.use_ema      = use_ema
        self.alpha        = alpha

        trunk_dim = 256
        self.encoder = EncoderLN(latent_dim)
        self.trunk   = nn.Sequential(
            nn.Linear(latent_dim + action_dim, trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.SiLU(),
        )
        self.prior_head     = CatPriorHead(trunk_dim, n_categories)
        self.posterior_head = CatPosteriorHead(latent_dim, trunk_dim, n_categories)
        self.branch_head    = BranchOutputHead(latent_dim, n_categories, trunk_dim)
        self.chain_ln       = nn.LayerNorm(latent_dim)

        self.cue_head  = CueBallHead(latent_dim)
        self.tgt_head  = TgtBallHead(latent_dim)
        self.type_head = TypeHead(latent_dim)

        if use_ema:
            self.ema_encoder = copy.deepcopy(self.encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self) -> None:
        if not self.use_ema:
            return
        tau = self.ema_tau
        for p, p_ema in zip(self.encoder.parameters(),
                            self.ema_encoder.parameters()):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    def rollout_train(
        self,
        s_0:    torch.Tensor,   # (B, 14)
        seq_s:  torch.Tensor,   # (B, T+1, 14)
        action: torch.Tensor,   # (B, m)
        T:      int,
    ) -> Tuple[List, List, List, List, torch.Tensor, torch.Tensor]:
        """
        c ~ Posterior (straight-through) → chain z_pred.
        Returns: z_pred_list, z_bar_list, prior_probs_list, post_probs_list, s_hat, type_logit
        """
        B       = s_0.shape[0]
        device  = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_chain = self.encoder(s_0)
        s_hat_list      = [self._decode(z_chain)]
        type_logit_list = []
        z_pred_list, z_bar_list           = [], []
        prior_probs_list, post_probs_list = [], []

        for h in range(T):
            a_tilde   = action if h == 0 else a_zeros
            trunk_out = self.trunk(torch.cat([z_chain, a_tilde], dim=-1))  # (B, 256)

            prior_probs = _unimix(self.prior_head(trunk_out), self.alpha)
            prior_probs_list.append(prior_probs)

            with torch.no_grad():
                target_enc = self.ema_encoder if self.use_ema else self.encoder
                z_bar = target_enc(seq_s[:, h + 1])
            z_bar_list.append(z_bar)

            post_probs = _unimix(self.posterior_head(trunk_out, z_bar), self.alpha)
            post_probs_list.append(post_probs)

            c_onehot   = _st_onehot(post_probs)
            correction = self.branch_head(trunk_out, c_onehot)
            z_chain    = self.chain_ln(z_chain + correction)

            z_pred_list.append(z_chain)
            type_logit_list.append(self.type_head(z_chain))
            s_hat_list.append(self._decode(z_chain))

        s_hat      = torch.stack(s_hat_list, dim=1)       # (B, T+1, 14)
        type_logit = torch.stack(type_logit_list, dim=1)  # (B, T, 5)
        return z_pred_list, z_bar_list, prior_probs_list, post_probs_list, s_hat, type_logit

    @torch.no_grad()
    def rollout_eval(
        self,
        s_0:        torch.Tensor,
        T:          int,
        action:     torch.Tensor | None = None,
        use_sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        c ~ Prior argmax (use_sample=False) or categorical sample (use_sample=True).
        Returns: s_hat (B, T+1, 14), type_logit (B, T, 5)
        """
        B       = s_0.shape[0]
        device  = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_chain         = self.encoder(s_0)
        s_hat_list      = [self._decode(z_chain)]
        type_logit_list = []

        for h in range(T):
            a_tilde   = action if (h == 0 and action is not None) else a_zeros
            trunk_out = self.trunk(torch.cat([z_chain, a_tilde], dim=-1))

            prior_probs = _unimix(self.prior_head(trunk_out), self.alpha)
            if use_sample:
                c_idx = torch.multinomial(prior_probs, 1).squeeze(-1)
            else:
                c_idx = prior_probs.argmax(dim=-1)
            c_onehot = F.one_hot(c_idx, self.n_categories).to(z_chain)

            type_logit_list.append(self.type_head(z_chain))
            correction = self.branch_head(trunk_out, c_onehot)
            z_chain    = self.chain_ln(z_chain + correction)
            s_hat_list.append(self._decode(z_chain))

        s_hat      = torch.stack(s_hat_list, dim=1)
        type_logit = torch.stack(type_logit_list, dim=1)
        return s_hat, type_logit


def spr_rollout_loss_cat(
    z_pred_list:      List[torch.Tensor],   # [T] (B, D)
    z_bar_list:       List[torch.Tensor],   # [T] (B, D) — stop-grad targets
    prior_probs_list: List[torch.Tensor],   # [T] (B, K)
    post_probs_list:  List[torch.Tensor],   # [T] (B, K)
    s_hat:            torch.Tensor,         # (B, T+1, 14)
    seq_s:            torch.Tensor,         # (B, T+1, 14)
    type_logit:       torch.Tensor,         # (B, T, 5)
    seq_types:        torch.Tensor,         # (B, T) int
    lam_kl:           float = 1.0,
    lam_recon:        float = 1.0,
    class_weights:    torch.Tensor | None = None,
    focal_gamma:      float = 2.0,
    label_smoothing:  float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    v24 loss: L2(z_pred, z_tgt) + lam_kl·KL(post‖prior) + lam_recon·recon + type.

    Unimix ensures probs > 0 on both sides → log always finite.
    """
    T    = len(z_pred_list)
    zero = z_pred_list[0].new_zeros(())

    total_l2 = zero
    total_kl = zero

    for h in range(T):
        total_l2 = total_l2 + F.mse_loss(z_pred_list[h], z_bar_list[h])
        qp = post_probs_list[h]
        pp = prior_probs_list[h]
        total_kl = total_kl + (qp * (torch.log(qp) - torch.log(pp))).sum(-1).mean()

    total_l2 = total_l2 / T
    total_kl = total_kl / T

    total_recon = sum(
        F.mse_loss(s_hat[:, h + 1, :7], seq_s[:, h + 1, :7]) +
        F.mse_loss(s_hat[:, h + 1, 7:], seq_s[:, h + 1, 7:])
        for h in range(T)
    ) / T

    B_T         = type_logit.shape[0] * type_logit.shape[1]
    logits_flat = type_logit.reshape(B_T, N_COLL_TYPES)
    types_flat  = seq_types.reshape(B_T)
    loss_type   = _focal_cross_entropy(
        logits_flat, types_flat, class_weights, focal_gamma, label_smoothing
    ) / math.log(N_COLL_TYPES)

    total = total_l2 + lam_kl * total_kl + lam_recon * total_recon + loss_type

    with torch.no_grad():
        last_prior = prior_probs_list[-1]
        last_post  = post_probs_list[-1]
        prior_perp = torch.exp(
            -(last_prior * torch.log(last_prior.clamp(1e-8))).sum(-1).mean()
        ).item()
        post_perp = torch.exp(
            -(last_post * torch.log(last_post.clamp(1e-8))).sum(-1).mean()
        ).item()

    detail = {
        "loss_l2":    total_l2.item(),
        "loss_kl":    total_kl.item(),
        "loss_recon": total_recon.item(),
        "loss_type":  loss_type.item(),
        "prior_perp": prior_perp,
        "post_perp":  post_perp,
    }
    return total, detail


# ════════════════════════════════════════════════════════════════════════════
# SPR K=1 ablation — identical skeleton, mixture replaced by single-point MLP
# ════════════════════════════════════════════════════════════════════════════

class SPRK1Model(nn.Module):
    """
    K=1 deterministic SPR ablation (v9 / v10 / v14).

    v9  (use_ema=True):             EMA target encoder.
    v10 (use_ema=False):            no EMA, pure closed-loop.
    v14 (use_encoder_ln=False):     v18-parity — StateEncoder (no LN on encoder output)
                                    + dedicated transition LN (same as SSMWorldModel v18).
                                    Isolates encoder-LN-sharing from other variables.

    use_encoder_ln=True  (default): EncoderLN encoder; _step reuses encoder.ln.
    use_encoder_ln=False (v18 LN):  plain StateEncoder; _step uses self.transition_ln.
    """

    def __init__(
        self,
        latent_dim:      int   = LATENT_DIM,
        action_dim:      int   = ACTION_DIM,
        ema_tau:         float = EMA_TAU,
        use_ema:         bool  = True,
        use_action:      bool  = True,
        use_encoder_ln:  bool  = True,
        full_bptt:       bool  = False,  # v17: remove stop-gradient between steps
    ):
        super().__init__()
        self.latent_dim     = latent_dim
        self.action_dim     = action_dim
        self.ema_tau        = ema_tau
        self.use_ema        = use_ema
        self.use_action     = use_action
        self.use_encoder_ln = use_encoder_ln
        self.full_bptt      = full_bptt

        self.encoder = EncoderLN(latent_dim) if use_encoder_ln else StateEncoder(latent_dim)
        if use_ema:
            self.ema_encoder = copy.deepcopy(self.encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad_(False)

        # Residual transition: z' = LN(z + MLP([z; a]))
        # use_action=False: matches v18 (no action input, 128-dim only)
        trans_in = latent_dim + (action_dim if use_action else 0)
        self.transition = _mlp(trans_in, 256, 256, latent_dim)

        # Dedicated transition LN used when use_encoder_ln=False (v18-parity).
        # When use_encoder_ln=True the encoder's LN is reused instead.
        if not use_encoder_ln:
            self.transition_ln = nn.LayerNorm(latent_dim)

        self.cue_head   = CueBallHead(latent_dim)
        self.tgt_head   = TgtBallHead(latent_dim)
        self.type_head  = TypeHead(latent_dim)

    @torch.no_grad()
    def update_ema(self) -> None:
        if not self.use_ema:
            return
        tau = self.ema_tau
        for p, p_ema in zip(self.encoder.parameters(),
                            self.ema_encoder.parameters()):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    def _step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, a], dim=-1) if self.use_action else z
        delta = self.transition(inp)
        if self.use_encoder_ln:
            return self.encoder.ln(z + delta)   # type: ignore[union-attr]
        return self.transition_ln(z + delta)

    def rollout_train(
        self,
        s_0:    torch.Tensor,
        seq_s:  torch.Tensor,
        action: torch.Tensor,
        T:      int,
        p_tf:   float = 0.0,
    ) -> Tuple[List, List]:
        B      = s_0.shape[0]
        device = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_hat = self.encoder(s_0)
        z_hat_list = [z_hat]
        z_bar_list = []

        use_tf = (p_tf > 0.0 and torch.rand(1).item() < p_tf)

        for h in range(T):
            a_tilde = action if h == 0 else a_zeros
            z_pred  = self._step(z_hat, a_tilde)   # gradient flows here

            with torch.no_grad():
                target_enc = self.ema_encoder if self.use_ema else self.encoder
                z_bar = target_enc(seq_s[:, h + 1])
            z_bar_list.append(z_bar)
            z_hat_list.append(z_pred)

            if use_tf:
                with torch.no_grad():
                    z_hat = self.encoder(seq_s[:, h + 1])
            elif self.full_bptt:
                z_hat = z_pred          # full BPTT: gradient flows through all steps
            else:
                z_hat = z_pred.detach() # truncated BPTT: stop-gradient (original)

        return z_hat_list, z_bar_list

    @torch.no_grad()
    def rollout_eval(
        self,
        s_0:    torch.Tensor,
        T:      int,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B      = s_0.shape[0]
        device = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_hat           = self.encoder(s_0)
        s_hat_list      = [self._decode(z_hat)]
        type_logit_list = []

        for h in range(T):
            a_tilde = action if (h == 0 and action is not None) else a_zeros
            type_logit_list.append(self.type_head(z_hat))
            z_hat = self._step(z_hat, a_tilde)
            s_hat_list.append(self._decode(z_hat))

        s_hat      = torch.stack(s_hat_list, dim=1)
        type_logit = torch.stack(type_logit_list, dim=1)
        return s_hat, type_logit

    def forward(
        self,
        s_0:    torch.Tensor,
        seq_s:  torch.Tensor,
        action: torch.Tensor,
        T:      int,
        p_tf:   float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        z_hat_list, z_bar_list = self.rollout_train(s_0, seq_s, action, T, p_tf=p_tf)

        z_stack    = torch.stack(z_hat_list, dim=1)
        s_hat      = torch.stack(
            [self._decode(z_stack[:, t]) for t in range(T + 1)], dim=1
        )
        type_logit = torch.stack(
            [self.type_head(z_stack[:, t]) for t in range(T)], dim=1
        )
        return s_hat, type_logit, z_hat_list, z_bar_list


def sprk1_rollout_loss(
    z_hat_list:  List[torch.Tensor],   # [T+1] each (B, D) — z_hat_list[h+1] is predicted
    z_bar_list:  List[torch.Tensor],   # [T]   each (B, D) — EMA/sg targets (detached)
    s_hat:       torch.Tensor,         # (B, T+1, 14)
    seq_s:       torch.Tensor,         # (B, T+1, 14)
    type_logit:  torch.Tensor,         # (B, T, 5)
    seq_types:   torch.Tensor,         # (B, T) int
    class_weights: torch.Tensor | None = None,
    focal_gamma:   float = 2.0,
    label_smoothing: float = 0.0,
    lam_recon:     float = 1.0,
    lam_l2:        float = 1.0,        # weight on latent L2 loss
    skip_h0_recon: bool = False,       # v16: exclude trivial h=0 recon (matches v18)
) -> Tuple[torch.Tensor, dict]:
    T = len(z_bar_list)

    # L2 latent prediction loss  (z_hat_list[h+1] vs z_bar_list[h])
    loss_l2 = sum(
        F.mse_loss(z_hat_list[h + 1], z_bar_list[h]) for h in range(T)
    ) / T

    s_start = 1 if skip_h0_recon else 0
    loss_cue   = F.mse_loss(s_hat[:, s_start:, :7], seq_s[:, s_start:, :7])
    loss_tgt   = F.mse_loss(s_hat[:, s_start:, 7:], seq_s[:, s_start:, 7:])
    loss_recon = loss_cue + loss_tgt

    B_T         = type_logit.shape[0] * type_logit.shape[1]
    logits_flat = type_logit.reshape(B_T, N_COLL_TYPES)
    types_flat  = seq_types.reshape(B_T)
    loss_type   = _focal_cross_entropy(
        logits_flat, types_flat, class_weights, focal_gamma, label_smoothing
    ) / math.log(N_COLL_TYPES)

    total = lam_l2 * loss_l2 + lam_recon * loss_recon + loss_type
    detail = {
        "loss_l2":    loss_l2.item(),
        "loss_recon": loss_recon.item(),
        "loss_cue":   loss_cue.item(),
        "loss_tgt":   loss_tgt.item(),
        "loss_type":  loss_type.item(),
    }
    return total, detail
