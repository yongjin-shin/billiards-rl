"""
world_model/spr_mdn/gradnorm.py — GradNorm controller

Chen et al. 2018: "GradNorm: Gradient Normalization for Adaptive Loss Balancing
in Deep Multitask Networks"

Balances per-task weighted gradient norms at shared (encoder) parameters:
  G_W_i = w_i × ||∂L_i/∂θ_enc||
  Target: G_W_i → G_avg × r_i^α
where r_i = relative inverse training rate (slow tasks get higher target).

Usage:
    gn = GradNormController(["nll", "recon"], alpha=1.5).to(device)
    gn_opt = torch.optim.Adam(gn.parameters(), lr=1e-3)

    # Inside training loop (graph must still be alive):
    diag = gn.step({"nll": L_nll, "recon": L_recon},
                   list(model.encoder.parameters()), gn_opt)
    w = gn.weights.detach()
    total = w[0] * L_nll + w[1] * L_recon + L_type
    total.backward()
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class GradNormController(nn.Module):
    """
    Maintains per-task weights and updates them via GradNorm objective.

    Parameters
    ----------
    task_names : list of str
        Names of tasks to balance (e.g. ["nll", "recon"]).
    alpha : float
        Restoring force: larger → stronger pull toward equal-improvement pace.
        α=1.5 is the paper default; try 0.5 (mild) or 2.5 (aggressive).
    """

    def __init__(self, task_names: List[str], alpha: float = 1.5):
        super().__init__()
        self.task_names = task_names
        self.alpha = alpha
        n = len(task_names)
        # log_weights: exp → initial weight = 1.0 for all tasks
        self.log_weights = nn.Parameter(torch.zeros(n))
        self.register_buffer("L0",     torch.zeros(n))
        self.register_buffer("L0_set", torch.zeros(1, dtype=torch.bool))

    @property
    def weights(self) -> torch.Tensor:
        """Current task weights (always positive)."""
        return self.log_weights.exp()

    @torch.no_grad()
    def _init_L0(self, loss_vals: List[float]) -> None:
        for i, v in enumerate(loss_vals):
            self.L0[i] = max(abs(v), 1e-6)
        self.L0_set[0] = True

    def step(
        self,
        losses:        Dict[str, torch.Tensor],
        shared_params: List[nn.Parameter],
        optimizer:     torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Compute per-task gradient norms, update log_weights via GradNorm loss,
        then renormalize so sum(w) = n_tasks.

        Must be called BEFORE the main model backward (computation graph alive).
        Uses retain_graph=True internally so the caller can still do loss.backward().

        Returns diagnostic dict with weights, gradient norms, and GradNorm loss.
        """
        n = len(self.task_names)
        ordered = [losses[t] for t in self.task_names]

        if not self.L0_set[0]:
            self._init_L0([l.detach().item() for l in ordered])

        # ── per-task gradient norms at shared encoder params ─────────────
        g_norms: List[torch.Tensor] = []
        for l in ordered:
            grads = torch.autograd.grad(
                l, shared_params,
                retain_graph=True, create_graph=False, allow_unused=True,
            )
            g_sq = sum(g.pow(2).sum() for g in grads if g is not None)
            g_norms.append(g_sq.sqrt())

        # ── GradNorm objective ────────────────────────────────────────────
        w = self.weights                          # (n,) requires_grad via log_weights
        G_raw  = torch.stack([gn.detach() for gn in g_norms])     # (n,) no grad
        G_w    = w * G_raw                        # (n,) grad flows through w
        G_avg  = G_w.mean().detach()

        L_cur = torch.stack([l.detach() for l in ordered])
        r = (L_cur / self.L0.clamp(min=1e-8))
        r = r / r.mean().clamp(min=1e-8)          # normalize relative rates
        G_tgt = (G_avg * r.pow(self.alpha)).detach()

        L_gn = (G_w - G_tgt).abs().sum()

        optimizer.zero_grad()
        L_gn.backward()
        optimizer.step()

        # ── renormalize: sum(w) = n ───────────────────────────────────────
        with torch.no_grad():
            w_new = self.weights
            self.log_weights.data = torch.log(w_new / w_new.sum() * n)

        # ── diagnostics ──────────────────────────────────────────────────
        w_final = self.weights.detach()
        diag: Dict[str, float] = {
            "gn_loss": L_gn.item(),
            "G_avg":   G_avg.item(),
        }
        for i, t in enumerate(self.task_names):
            diag[f"w_{t}"]  = w_final[i].item()
            diag[f"G_{t}"]  = g_norms[i].detach().item()
            diag[f"Gt_{t}"] = G_tgt[i].item()
        return diag
