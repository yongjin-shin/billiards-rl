"""
world_model/spr_mdn/spr_mdn_model.py — SPR-MDN World Model

Self-Predictive Representations + Mixture Density Network on top of v18 no-AR.

Key ideas vs v18 no-AR:
  1. MixtureHead (K=5 MDN) replaces deterministic ResTransition
  2. Self-chaining: training uses own sampled ẑ as next input (not GT)
  3. EMA encoder φ' provides stable NLL targets: z̄_{h+1} = sg(Enc_φ'(s_{t+h+1}))
  4. L_recon grounds z to real ball states; prevents NLL conspiracy

Lineage: BYOL → SPR → SPR-MDN [ours]
"""

import copy
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse from existing modules — do NOT duplicate
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.fixeddt_model import StateEncoder
from world_model.ssm_model import CueBallHead, TgtBallHead, TypeHead, _focal_cross_entropy

LATENT_DIM    = 128
N_COMPONENTS  = 5
N_COLL_TYPES  = 5
EMA_TAU       = 0.99


def _mlp(*dims: int, act=nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class MixtureHead(nn.Module):
    """
    z_h → (π, μ, σ)   K-component MDN transition.

    Output sizes per sample:
      π: (K,)         mixture weights (softmax)
      μ: (K, D)       component means
      σ: (K, D)       component std-devs (softplus + ε)
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_components: int = N_COMPONENTS):
        super().__init__()
        self.K = n_components
        self.D = latent_dim
        out_dim = n_components + n_components * latent_dim * 2  # π + μ + log_σ
        self.net = _mlp(latent_dim, 256, 256, out_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D)
        B = z.shape[0]
        K, D = self.K, self.D
        h = self.net(z)                                       # (B, K + 2*K*D)

        log_pi  = h[:, :K]                                   # (B, K)
        mu      = h[:, K : K + K * D].view(B, K, D)          # (B, K, D)
        log_sig = h[:, K + K * D :].view(B, K, D)            # (B, K, D)

        pi    = F.softmax(log_pi, dim=-1)                     # (B, K)
        sigma = F.softplus(log_sig) + 1e-4                   # (B, K, D)

        return pi, mu, sigma


def mdn_nll(
    pi:     torch.Tensor,   # (B, K)
    mu:     torch.Tensor,   # (B, K, D)
    sigma:  torch.Tensor,   # (B, K, D)
    target: torch.Tensor,   # (B, D)
) -> torch.Tensor:
    """Negative log-likelihood of target under MDN, averaged over batch."""
    D = target.shape[-1]
    target = target.unsqueeze(1)                              # (B, 1, D)
    # log N(target; μ_k, σ_k²) for each component
    log_prob_k = (
        -0.5 * ((target - mu) / sigma) ** 2
        - sigma.log()
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)                                                 # (B, K)

    log_pi = pi.log().clamp(min=-1e9)
    log_mix = torch.logsumexp(log_pi + log_prob_k, dim=-1)   # (B,)
    return -log_mix.mean()


class SPRMDNModel(nn.Module):

    def __init__(
        self,
        latent_dim:    int   = LATENT_DIM,
        n_components:  int   = N_COMPONENTS,
        ema_tau:       float = EMA_TAU,
    ):
        super().__init__()
        self.latent_dim   = latent_dim
        self.n_components = n_components
        self.ema_tau      = ema_tau

        # Trainable modules
        self.encoder      = StateEncoder(latent_dim)
        self.mixture_head = MixtureHead(latent_dim, n_components)
        self.cue_head     = CueBallHead(latent_dim)
        self.tgt_head     = TgtBallHead(latent_dim)
        self.type_head    = TypeHead(latent_dim)

        # EMA encoder — same architecture, NOT in optimizer
        self.ema_encoder  = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self) -> None:
        tau = self.ema_tau
        for p, p_ema in zip(self.encoder.parameters(),
                            self.ema_encoder.parameters()):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) → s_hat: (B, 14)"""
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    def rollout_train(
        self,
        s_0:   torch.Tensor,   # (B, 14)
        seq_s: torch.Tensor,   # (B, T+1, 14)  GT states
        T:     int,
    ) -> Tuple[List, List, List, List, List]:
        """
        Self-chaining training rollout.
        Returns lists of length T+1 / T:
          z_hat_list  [T+1]: z_hat_0 has grad; z_hat_1..T are detached samples
          pi_list     [T]
          mu_list     [T]
          sigma_list  [T]
          z_bar_list  [T]: EMA-encoded GT targets (detached)
        """
        B = s_0.shape[0]
        device = s_0.device

        z_hat = self.encoder(s_0)                             # (B, D) — grad flows here
        z_hat_list  = [z_hat]
        pi_list, mu_list, sigma_list, z_bar_list = [], [], [], []

        for h in range(T):
            pi, mu, sigma = self.mixture_head(z_hat)          # uses current z_hat

            # EMA-encoded GT target — stop gradient
            with torch.no_grad():
                z_bar = self.ema_encoder(seq_s[:, h + 1])     # (B, D)
            z_bar_list.append(z_bar)

            pi_list.append(pi)
            mu_list.append(mu)
            sigma_list.append(sigma)

            # Sample next z — detached (gradient flows only via NLL above)
            with torch.no_grad():
                k = torch.multinomial(pi.detach(), 1).squeeze(1)  # (B,)
                eps = torch.randn(B, self.latent_dim, device=device)
                z_hat = mu[torch.arange(B), k] + sigma[torch.arange(B), k] * eps

            z_hat_list.append(z_hat)

        return z_hat_list, pi_list, mu_list, sigma_list, z_bar_list

    @torch.no_grad()
    def rollout_eval(
        self,
        s_0: torch.Tensor,   # (B, 14)
        T:   int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic eval rollout using mixture mean: ẑ_{h+1} = Σ_k π_k μ_k.
        Returns: s_hat (B, T+1, 14), type_logit (B, T, 5)
        """
        z_hat = self.encoder(s_0)
        s_hat_list    = [self._decode(z_hat)]
        type_logit_list = []

        for _ in range(T):
            pi, mu, sigma = self.mixture_head(z_hat)
            type_logit_list.append(self.type_head(z_hat))
            # Mixture mean as deterministic next latent
            z_hat = (pi.unsqueeze(-1) * mu).sum(dim=1)       # (B, D)
            s_hat_list.append(self._decode(z_hat))

        s_hat      = torch.stack(s_hat_list, dim=1)           # (B, T+1, 14)
        type_logit = torch.stack(type_logit_list, dim=1)      # (B, T, 5)
        return s_hat, type_logit

    def forward(
        self,
        s_0:   torch.Tensor,   # (B, 14)
        seq_s: torch.Tensor,   # (B, T+1, 14)
        T:     int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List, List, List, List]:
        """
        Training forward.
        Returns:
          s_hat       (B, T+1, 14)
          type_logit  (B, T, 5)
          z_hat_list, pi_list, mu_list, sigma_list, z_bar_list
        """
        z_hat_list, pi_list, mu_list, sigma_list, z_bar_list = \
            self.rollout_train(s_0, seq_s, T)

        z_stack    = torch.stack(z_hat_list, dim=1)           # (B, T+1, D)
        s_hat      = torch.stack(
            [self._decode(z_stack[:, t]) for t in range(T + 1)], dim=1
        )                                                     # (B, T+1, 14)
        type_logit = torch.stack(
            [self.type_head(z_stack[:, t]) for t in range(T)], dim=1
        )                                                     # (B, T, 5)

        return s_hat, type_logit, z_hat_list, pi_list, mu_list, sigma_list, z_bar_list


def spr_rollout_loss(
    pi_list:    List[torch.Tensor],   # [T] each (B, K)
    mu_list:    List[torch.Tensor],   # [T] each (B, K, D)
    sigma_list: List[torch.Tensor],   # [T] each (B, K, D)
    z_bar_list: List[torch.Tensor],   # [T] each (B, D)  — detached EMA targets
    s_hat:      torch.Tensor,         # (B, T+1, 14)
    seq_s:      torch.Tensor,         # (B, T+1, 14)
    type_logit: torch.Tensor,         # (B, T, 5)
    seq_types:  torch.Tensor,         # (B, T)  int  0–4
    class_weights: torch.Tensor | None = None,
    log_sigma:  torch.Tensor | None = None,   # (2,) Kendall [nll, recon]
    focal_gamma:     float = 2.0,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    L_NLL   = mean_t mdn_nll(π_t, μ_t, σ_t, z̄_{t+1})
    L_recon = MSE(s_hat[:, 1:], seq_s[:, 1:])  cue + tgt
    L_type  = focal_CE(type_logit, seq_types)

    Temperature taming (log_sigma (2,)):
      s = log_sigma.clamp(-6, 6)
      L = exp(-s[0])·L_NLL + s[0] + exp(-s[1])·L_recon + s[1] + L_type
    """
    T = len(pi_list)

    # ── NLL ─────────────────────────────────────────────────────────────
    nll_sum = sum(
        mdn_nll(pi_list[h], mu_list[h], sigma_list[h], z_bar_list[h])
        for h in range(T)
    )
    loss_nll = nll_sum / T

    # ── Recon ────────────────────────────────────────────────────────────
    loss_cue   = F.mse_loss(s_hat[:, 1:, :7],  seq_s[:, 1:, :7])
    loss_tgt   = F.mse_loss(s_hat[:, 1:, 7:],  seq_s[:, 1:, 7:])
    loss_recon = loss_cue + loss_tgt

    # ── Type ─────────────────────────────────────────────────────────────
    B_T = type_logit.shape[0] * type_logit.shape[1]
    logits_flat = type_logit.reshape(B_T, N_COLL_TYPES)
    types_flat  = seq_types.reshape(B_T)
    loss_type   = _focal_cross_entropy(
        logits_flat, types_flat, class_weights, focal_gamma, label_smoothing
    ) / math.log(N_COLL_TYPES)

    # ── Temperature taming ───────────────────────────────────────────────
    if log_sigma is not None:
        s = log_sigma.clamp(-6, 6)
        total = (
            torch.exp(-s[0]) * loss_nll   + s[0] +
            torch.exp(-s[1]) * loss_recon + s[1] +
            loss_type
        )
        kw = torch.exp(-s.detach())
        detail = {
            "loss_nll":   loss_nll.item(),
            "loss_recon": loss_recon.item(),
            "loss_cue":   loss_cue.item(),
            "loss_tgt":   loss_tgt.item(),
            "loss_type":  loss_type.item(),
            "kw_nll":     kw[0].item(),
            "kw_recon":   kw[1].item(),
        }
    else:
        total = loss_nll + loss_recon + loss_type
        detail = {
            "loss_nll":   loss_nll.item(),
            "loss_recon": loss_recon.item(),
            "loss_cue":   loss_cue.item(),
            "loss_tgt":   loss_tgt.item(),
            "loss_type":  loss_type.item(),
        }

    return total, detail
