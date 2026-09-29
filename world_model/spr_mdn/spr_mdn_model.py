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
        b  = F.softplus(b_raw) + 0.1                # (B, K, D) — b_min=0.1 prevents gradient explosion

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
        latent_dim:     int   = LATENT_DIM,
        n_components:   int   = N_COMPONENTS,
        action_dim:     int   = ACTION_DIM,
        ema_tau:        float = EMA_TAU,
        asym_init:      bool  = False,
        use_ema:        bool  = True,
        use_encoder_ln: bool  = True,
        full_bptt:      bool  = False,
        use_action:     bool  = True,
    ):
        super().__init__()
        self.latent_dim     = latent_dim
        self.n_components   = n_components
        self.action_dim     = action_dim if use_action else 0
        self.ema_tau        = ema_tau
        self.use_ema        = use_ema
        self.use_encoder_ln = use_encoder_ln
        self.full_bptt      = full_bptt
        self.use_action     = use_action

        # Encoder: with or without built-in LayerNorm
        self.encoder = EncoderLN(latent_dim) if use_encoder_ln else StateEncoder(latent_dim)
        if not use_encoder_ln:
            # Dedicated transition LN — same role as SPRK1Model.transition_ln
            self.transition_ln = nn.LayerNorm(latent_dim)

        self.mixture_head = MixtureHead(latent_dim, n_components, self.action_dim, asym_init=asym_init)
        self.cue_head     = CueBallHead(latent_dim)
        self.tgt_head     = TgtBallHead(latent_dim)
        self.type_head    = TypeHead(latent_dim)

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

    def _apply_ln(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the appropriate LayerNorm (encoder.ln or transition_ln)."""
        if self.use_encoder_ln:
            return self.encoder.ln(z)  # type: ignore[union-attr]
        return self.transition_ln(z)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) → s_hat: (B, 14)"""
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    def rollout_train(
        self,
        s_0:    torch.Tensor,   # (B, 14)
        seq_s:  torch.Tensor,   # (B, T+1, 14)
        action: torch.Tensor,   # (B, m)
        T:      int,
        p_tf:   float = 0.0,    # teacher-forcing probability (scheduled sampling)
    ) -> Tuple[List, List, List, List, List]:
        """
        Scheduled-sampling training rollout.

        full_bptt=False (default): self-chain via Laplace sample (sg), LN in grad graph.
        full_bptt=True: self-chain via mixture mean (differentiable), full gradient flow.

        use_ema=False: z_bar target = sg(encoder(s_gt)) instead of EMA encoder.
        use_encoder_ln=False: StateEncoder + transition_ln instead of EncoderLN.

        Returns: z_hat_list[T+1], pi_list[T], mu_list[T], b_list[T], z_bar_list[T]
        """
        B       = s_0.shape[0]
        device  = s_0.device
        a_zeros = torch.zeros(B, self.action_dim, device=device)

        z_hat = self.encoder(s_0)    # (B, D) — h=0, real encoding
        z_hat_list = [z_hat]
        pi_list, mu_list, b_list, z_bar_list = [], [], [], []

        # Decide once per batch (per-step Bernoulli would reset GT every step)
        use_tf = (p_tf > 0.0 and torch.rand(1).item() < p_tf)

        for h in range(T):
            a_tilde = (action if h == 0 else a_zeros) if self.use_action else a_zeros

            pi, mu, b = self.mixture_head(z_hat, a_tilde)

            # Target z_bar: EMA encoder or stop-grad of online encoder
            with torch.no_grad():
                if self.use_ema:
                    z_bar = self.ema_encoder(seq_s[:, h + 1])
                else:
                    z_bar = self.encoder(seq_s[:, h + 1])
            z_bar_list.append(z_bar)

            pi_list.append(pi)
            mu_list.append(mu)
            b_list.append(b)

            if self.full_bptt and not use_tf:
                # Differentiable chain: mixture mean, no stop-grad
                z_mean = (pi.unsqueeze(-1) * mu).sum(dim=1)   # (B, D)
                z_hat  = self._apply_ln(z_mean)
            else:
                with torch.no_grad():
                    if use_tf:
                        z_hat = self.encoder(seq_s[:, h + 1])
                        if not self.use_encoder_ln:
                            z_hat = self.transition_ln(z_hat)
                    else:
                        # Laplace sample, stop-grad
                        k      = torch.multinomial(pi.detach(), 1).squeeze(1)
                        mu_k   = mu[torch.arange(B), k]
                        b_k    = b[torch.arange(B), k]
                        u      = (torch.rand_like(b_k) - 0.5).clamp(-0.4999, 0.4999)
                        z_hat  = mu_k - b_k * u.sign() * torch.log1p(-2.0 * u.abs())
                if not use_tf:
                    # LN outside no_grad so γ,β receive gradient from chain
                    z_hat = self._apply_ln(z_hat)

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
            a_tilde = (action if (h == 0 and action is not None) else a_zeros) if self.use_action else a_zeros
            pi, mu, b = self.mixture_head(z_hat, a_tilde)
            type_logit_list.append(self.type_head(z_hat))
            z_hat = self._apply_ln((pi.unsqueeze(-1) * mu).sum(dim=1))   # (B, D)
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
    log_sigma:       "torch.Tensor | None" = None,  # (2,) Kendall [recon, type] — overrides lam_recon
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

    if log_sigma is not None:
        # Kendall uncertainty weighting (SSM-style): adaptive balance of recon vs type
        total = (loss_nll +
                 torch.exp(-log_sigma[0]) * loss_recon + log_sigma[0] +
                 torch.exp(-log_sigma[1]) * loss_type  + log_sigma[1])
    else:
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
