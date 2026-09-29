"""
world_model/spr_mdn/train_v27_p2.py — Phase 2: Frozen backbone + z-space MDN (EWTA)

Architecture:
  Phase 1 backbone (loaded from v26_p1/best.pt):
    encoder(s_t) → z_t          (128-dim latent)
    transition(z_t) → z_{t+1}   (deterministic, always frozen)
    decoder(z_t) → s_t

  Phase 2 addition:
    MDNHead(z_t) → (π_k, μ_k, σ_k)  K=5 isotropic Gaussian mixture over z_{t+1}

Training loss: EWTA + Entropy regularization
  EWTA (Evolved Winner-Takes-All):
    - 각 sample에서 z_target에 가장 가까운 top-k component만 학습
    - k_top curriculum: 1 → K over ewta_warmup epochs
    - μ/σ 다양성 확보 (component specialization)
  Entropy reg:
    - L_total = L_EWTA - beta_ent * H(π)
    - π collapse 방지 (dominant component π→1 억제)

Training phases (alternating):
  Phase A (9/10 epochs):
    z_t = encoder(s_t).detach()   [NLL gradient 완전 차단]
    loss = EWTA(MDN(z_t), z_t+1) - beta_ent * H(π)
  Phase B (1/10 epochs):
    loss = MSE(decoder(transition(encoder(s_0))), s_t)  [enc/dec slow follow]

Usage:
    python world_model/spr_mdn/train_v27_p2.py \
      --backbone world_model/results/spr_mdn_v26_p1/best.pt \
      --data-dir world_model/data_fixeddt \
      --out-dir  world_model/results/spr_mdn_v27_p2 \
      2>&1 | tee /tmp/v27_p2.log
"""

import math, os, sys, json, argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.fixeddt_model import StateEncoder, LATENT_DIM
from world_model.ssm_model import ResTransition, CueBallHead, TgtBallHead, TypeHead
from world_model.spr_mdn.spr_dataset import (
    SPRDataset, SPREpisodeSubset, _CapDataset, make_balanced_val_eps,
)
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H
from log_utils import Logger

T_MAX = 60
T_MIN = 10


# ─────────────────────────────────────────────────────────────────────────────
class MDNHead(nn.Module):
    """
    z_t → K-component isotropic Gaussian mixture over z_{t+1}.

    Output:
      pi_logits: (B, K)      — unnormalized mixture weights
      mu:        (B, K, D)   — component means in z-space
      log_sigma: (B, K)      — log-scale (shared across dims per component)
    """

    def __init__(self, latent_dim: int = LATENT_DIM, K: int = 5):
        super().__init__()
        self.K = K
        self.D = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),        nn.SiLU(),
        )
        self.pi_head    = nn.Linear(256, K)
        self.mu_head    = nn.Linear(256, K * latent_dim)
        self.sigma_head = nn.Linear(256, K)

    def forward(self, z: torch.Tensor):
        h         = self.net(z)
        pi_logits = self.pi_head(h)                          # (B, K)
        mu        = self.mu_head(h).view(-1, self.K, self.D) # (B, K, D)
        log_sigma = self.sigma_head(h)                        # (B, K)
        return pi_logits, mu, log_sigma


def mdn_nll(
    pi_logits:  torch.Tensor,   # (B, K)
    mu:         torch.Tensor,   # (B, K, D)
    log_sigma:  torch.Tensor,   # (B, K)
    z_target:   torch.Tensor,   # (B, D)
    sigma_min:  float = 0.1,
) -> torch.Tensor:
    """Full MDN NLL — used for validation only."""
    B, K, D = mu.shape
    sigma = log_sigma.exp().clamp(min=sigma_min)
    z_exp   = z_target.unsqueeze(1).expand_as(mu)
    sq_dist = ((z_exp - mu) ** 2).sum(-1)           # (B, K)
    log_p_k = (
        F.log_softmax(pi_logits, dim=-1)
        - D / 2 * math.log(2 * math.pi)
        - D * torch.log(sigma)
        - 0.5 * sq_dist / sigma ** 2
    )
    return -torch.logsumexp(log_p_k, dim=-1).mean()


def ewta_nll(
    pi_logits:  torch.Tensor,   # (B, K)
    mu:         torch.Tensor,   # (B, K, D)
    log_sigma:  torch.Tensor,   # (B, K)
    z_target:   torch.Tensor,   # (B, D)
    k_top:      int   = 1,
    sigma_min:  float = 0.1,
    beta_ent:   float = 0.05,
) -> torch.Tensor:
    """EWTA loss + entropy regularization.

    EWTA: only top-k closest components (by ||μ_k - z_target||) get gradient.
    Entropy reg: -beta_ent * H(π) added to prevent π collapse.
    """
    B, K, D = mu.shape
    sigma = log_sigma.exp().clamp(min=sigma_min)    # (B, K)

    z_exp   = z_target.unsqueeze(1).expand_as(mu)   # (B, K, D)
    sq_dist = ((z_exp - mu) ** 2).sum(-1)            # (B, K)

    # winner mask: top-k closest components
    k_top   = min(k_top, K)
    topk_idx = sq_dist.topk(k_top, dim=-1, largest=False).indices  # (B, k_top)
    mask = torch.zeros(B, K, device=mu.device).scatter_(1, topk_idx, 1.0)  # (B, K)

    log_p_k = (
        F.log_softmax(pi_logits, dim=-1)
        - D / 2 * math.log(2 * math.pi)
        - D * torch.log(sigma)
        - 0.5 * sq_dist / sigma ** 2
    )   # (B, K)

    # EWTA: sum only over winners, normalize by k_top
    ewta_loss = -(log_p_k * mask).sum(-1).mean()

    # entropy of π: H = -Σ π_k log π_k  (higher = more uniform → reward)
    pi = F.softmax(pi_logits, dim=-1)
    entropy = -(pi * (pi + 1e-12).log()).sum(-1).mean()

    return ewta_loss - beta_ent * entropy


def get_k_top(epoch: int, K: int, warmup_epochs: int) -> int:
    """Linear curriculum: k_top 1 → K over warmup_epochs."""
    if warmup_epochs <= 0:
        return K
    frac  = min(epoch / warmup_epochs, 1.0)
    k_top = 1 + int(frac * (K - 1) + 0.5)
    return min(k_top, K)


# ─────────────────────────────────────────────────────────────────────────────
class Phase2Model(nn.Module):
    """Phase 1 backbone (encoder/transition/decoder) + MDN stochastic head."""

    def __init__(self, latent_dim: int = LATENT_DIM, K: int = 5):
        super().__init__()
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ResTransition(latent_dim, use_ar_state=False)
        self.cue_head   = CueBallHead(latent_dim)
        self.tgt_head   = TgtBallHead(latent_dim)
        self.type_head  = TypeHead(latent_dim)
        self.mdn        = MDNHead(latent_dim, K)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    @torch.no_grad()
    def rollout_det(self, s_0: torch.Tensor, T: int):
        """Deterministic rollout via Phase 1 transition."""
        z = self.encoder(s_0)
        s_list, t_list = [self._decode(z)], []
        for _ in range(T):
            t_list.append(self.type_head(z))
            z = self.transition(z)
            s_list.append(self._decode(z))
        return (torch.stack(s_list, 1),          # (B, T+1, 14)
                torch.stack(t_list, 1))           # (B, T, 5)

    @torch.no_grad()
    def rollout_stoch(self, s_0: torch.Tensor, T: int):
        """Stochastic rollout via MDN sampling (argmax component)."""
        z = self.encoder(s_0)
        s_list = [self._decode(z)]
        for _ in range(T):
            pi_logits, mu, log_sigma = self.mdn(z)
            k_best = pi_logits.argmax(-1)          # (B,)
            z = mu[torch.arange(len(z)), k_best]   # (B, D)
            s_list.append(self._decode(z))
        return torch.stack(s_list, 1)              # (B, T+1, 14)


# ─────────────────────────────────────────────────────────────────────────────
def _eval_err(model: Phase2Model, episodes: list, device: str,
              rollout_steps: int = 60) -> dict:
    """Deterministic rollout error (Phase 1 transition)."""
    model.eval()
    cp = {"0.5s": int(0.5/DT), "1.0s": int(1.0/DT),
          "2.0s": int(2.0/DT), "3.0s": int(3.0/DT)}
    valid = [(ep_s, ep_f, ep_t)
             for ep_s, ep_f, ep_t, _, _a in episodes
             if len(ep_s) >= rollout_steps + 1]

    errs_all, errs_bb, errs_no = [], [], []
    cp_errors = {k: [] for k in cp}

    with torch.no_grad():
        s0 = torch.from_numpy(np.stack([e[0][0] for e in valid])).float().to(device)
        s_hat, _ = model.rollout_det(s0, rollout_steps)
        pr = s_hat.cpu().numpy()

    for b, (ep_s, ep_f, ep_t) in enumerate(valid):
        gt = ep_s[1:rollout_steps+1]
        p  = pr[b, 1:rollout_steps+1]
        ce = np.sqrt(((p[:,0]-gt[:,0])*TABLE_W)**2 + ((p[:,1]-gt[:,1])*TABLE_H)**2)
        te = np.sqrt(((p[:,7]-gt[:,7])*TABLE_W)**2 + ((p[:,8]-gt[:,8])*TABLE_H)**2)
        se = (ce + te) / 2 * 100
        me = se.mean()
        errs_all.append(me)
        has_bb = bool(np.any(ep_f & (ep_t == 1)))
        (errs_bb if has_bb else errs_no).append(me)
        for lbl, t in cp.items():
            cp_errors[lbl].append(se[t-1])

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    return {
        "mean_err":        _m(errs_all),
        "mean_err_has_bb": _m(errs_bb),
        "mean_err_no_bb":  _m(errs_no),
        **{k: _m(v) for k, v in cp_errors.items()},
    }


def _eval_nll(model: Phase2Model, val_loader, device: str) -> tuple[float, dict]:
    """Mean NLL + MDN component diagnostics on val set (single-step z-space).

    Returns (val_nll, stats) where stats contains:
      mu_spread : mean std of μ_k across K components (per dim, then mean over D)
                  → higher = components predict diverse z regions
      sigma_mean: mean σ across all components
                  → higher = more uncertainty / farther from point estimate
      eff_K     : exp(H(π)) — effective number of active components
                  → ideal = K, collapse = 1
    """
    model.eval()
    nlls        = []
    mu_spreads  = []   # per batch: mu.std(dim=1).mean()  scalar
    sigma_means = []   # per batch: sigma.mean()           scalar
    entropies   = []   # per batch: H(π) per sample        (B,)

    with torch.no_grad():
        for seq_s, *_ in val_loader:
            seq_s = seq_s.to(device)
            T = seq_s.shape[1] - 1
            for t in range(min(T, 5)):   # first 5 steps — representative + fast
                z_t   = model.encoder(seq_s[:, t])
                z_t1  = model.encoder(seq_s[:, t+1])
                pi_logits, mu, ls = model.mdn(z_t)

                nlls.append(mdn_nll(pi_logits, mu, ls, z_t1).item())

                sigma = ls.exp().clamp(min=0.1)              # (B, K)
                pi    = F.softmax(pi_logits, dim=-1)         # (B, K)

                # spread: std of K component means per dim, averaged over D
                mu_spreads.append(mu.std(dim=1).mean().item())   # scalar
                sigma_means.append(sigma.mean().item())
                H = -(pi * (pi + 1e-12).log()).sum(-1)           # (B,)
                entropies.append(H.mean().item())

    eff_K = float(np.exp(np.mean(entropies)))
    stats = {
        "mu_spread":  float(np.mean(mu_spreads)),
        "sigma_mean": float(np.mean(sigma_means)),
        "eff_K":      eff_K,
    }
    return float(np.mean(nlls)), stats


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(f"Device: {device}")
    logger.log(
        f"Phase2 MDN  K={args.K}  ewta_warmup={args.ewta_warmup}"
        f"  beta_ent={args.beta_ent}  phase_b_every={args.phase_b_every}"
        f"  lam_phase_b={args.lam_phase_b}"
        f"  lr_mdn={args.lr_mdn}  lr_encdec={args.lr_encdec}"
        f"  max_epochs={args.max_epochs}  patience={args.patience}"
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = SPRDataset(args.data_dir, logger=logger)
    rng_split = np.random.default_rng(0)
    perm  = rng_split.permutation(len(dataset.episodes))
    n_val = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all   = [dataset.episodes[i] for i in perm[:n_val]]
    train_eps_all = [dataset.episodes[i] for i in perm[n_val:]]

    balanced_val_eps = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    train_eps = [ep for ep in train_eps_all if len(ep[0]) >= T_MAX + 1]
    val_eps   = [ep for ep in balanced_val_eps if len(ep[0]) >= T_MAX + 1]
    logger.log(f"Train: {len(train_eps):,}  Val: {len(val_eps):,}")

    MAX_EPOCH_ITEMS = 60_000
    train_ds = _CapDataset(
        SPREpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True), MAX_EPOCH_ITEMS)
    val_ds   = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = Phase2Model(latent_dim=LATENT_DIM, K=args.K).to(device)

    # Load Phase 1 backbone
    ckpt = torch.load(args.backbone, map_location=device, weights_only=False)
    state = ckpt.get("state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.log(f"Backbone loaded: {args.backbone}")
    logger.log(f"  missing={missing}  unexpected={unexpected}")

    # Freeze transition (always)
    for p in model.transition.parameters():
        p.requires_grad_(False)
    logger.log("transition frozen")

    mdn_params    = list(model.mdn.parameters())
    encdec_params = (list(model.encoder.parameters()) +
                     list(model.cue_head.parameters()) +
                     list(model.tgt_head.parameters()) +
                     list(model.type_head.parameters()))

    n_mdn    = sum(p.numel() for p in mdn_params)
    n_encdec = sum(p.numel() for p in encdec_params if p.requires_grad)
    logger.log(f"MDN params: {n_mdn:,}  Enc/Dec params: {n_encdec:,}")

    opt_mdn    = torch.optim.Adam(mdn_params,    lr=args.lr_mdn)
    opt_encdec = torch.optim.Adam(encdec_params, lr=args.lr_encdec)

    sched_mdn    = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_mdn,    T_max=args.max_epochs, eta_min=args.lr_mdn    * 0.01)
    sched_encdec = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_encdec, T_max=args.max_epochs, eta_min=args.lr_encdec * 0.01)

    rng_T = np.random.default_rng(42)
    best_val_nll  = float("inf")
    stall_count   = 0
    last_det_err  = float("nan")
    last_nll      = float("nan")

    for epoch in range(1, args.max_epochs + 1):
        cur_T      = int(rng_T.integers(T_MIN, T_MAX + 1))
        is_phase_b = (epoch % args.phase_b_every == 0)
        phase_tag  = " [PhaseB]" if is_phase_b else " [PhaseA]"
        k_top      = get_k_top(epoch, args.K, args.ewta_warmup)

        model.train()
        nll_losses, recon_losses = [], []

        for seq_s, *_ in train_loader:
            seq_s_t = seq_s[:, :cur_T + 1].to(device)   # (B, T+1, 14)

            if not is_phase_b:
                # ── Phase A: EWTA + entropy reg, encoder fully detached ───────
                nll_sum = torch.tensor(0.0, device=device)
                for t in range(cur_T):
                    z_t  = model.encoder(seq_s_t[:, t]).detach()
                    z_t1 = model.encoder(seq_s_t[:, t+1]).detach()
                    pi, mu, ls = model.mdn(z_t)
                    nll_sum = nll_sum + ewta_nll(
                        pi, mu, ls, z_t1,
                        k_top=k_top, beta_ent=args.beta_ent,
                    )
                loss = nll_sum / cur_T

                opt_mdn.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mdn_params, 1.0)
                opt_mdn.step()
                nll_losses.append(loss.item())

            else:
                # ── Phase B: reconstruction via frozen transition ──────────────
                z_t   = model.encoder(seq_s_t[:, 0])                  # (B, D)
                recon_loss = torch.tensor(0.0, device=device)
                for t in range(cur_T):
                    z_t  = model.transition(z_t)                       # frozen
                    s_hat = torch.cat([model.cue_head(z_t),
                                       model.tgt_head(z_t)], dim=-1)
                    recon_loss = recon_loss + F.mse_loss(
                        s_hat, seq_s_t[:, t+1])
                loss = args.lam_phase_b * recon_loss / cur_T

                opt_encdec.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encdec_params, 1.0)
                opt_encdec.step()
                recon_losses.append(recon_loss.item() / cur_T)

        sched_mdn.step()
        sched_encdec.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        do_eval = (epoch % args.eval_every == 0 or epoch == 1)

        if do_eval:
            val_nll, mdn_stats = _eval_nll(model, val_loader, device)
            rerr     = _eval_err(model, balanced_val_eps, device)
            last_det_err = rerr["mean_err"]
            last_nll     = val_nll

        nll_str   = f"{np.mean(nll_losses):.4f}"   if nll_losses   else "—"
        recon_str = f"{np.mean(recon_losses):.4f}" if recon_losses else "—"

        cp_keys = [k for k in ["0.5s","1.0s","2.0s","3.0s"]
                   if do_eval and not np.isnan(rerr.get(k, float("nan")))]
        cp_str  = (" | ".join(f"{k}={rerr[k]:.1f}cm" for k in cp_keys)
                   if (do_eval and cp_keys) else "")

        log_line = (
            f"Epoch {epoch:4d}{phase_tag}"
            f"  [T={cur_T:2d}  k={k_top}/{args.K}]"
            f"  nll={nll_str}  recon={recon_str}"
            f"  val_nll={last_nll:.4f}"
            f"  det_err={last_det_err:.1f}cm"
            f"  (bb={rerr.get('mean_err_has_bb', float('nan')):.1f}"
            f"/nbb={rerr.get('mean_err_no_bb', float('nan')):.1f})"
        )
        if cp_str:
            log_line += f"  [{cp_str}]"
        if do_eval:
            log_line += (
                f"  [mdn: spread={mdn_stats['mu_spread']:.3f}"
                f"  σ={mdn_stats['sigma_mean']:.3f}"
                f"  effK={mdn_stats['eff_K']:.2f}/{args.K}]"
            )
        logger.log(log_line)

        # ── Checkpoint + early stopping ───────────────────────────────────────
        if do_eval:
            if last_nll < best_val_nll - 1e-4:
                best_val_nll = last_nll
                stall_count  = 0
                torch.save({
                    "state":       model.state_dict(),
                    "epoch":       epoch,
                    "val_nll":     last_nll,
                    "det_err":     last_det_err,
                    "K":           args.K,
                    "latent_dim":  LATENT_DIM,
                }, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")

            if stall_count >= args.patience:
                logger.log(
                    f"\nEarly stop: patience={args.patience} reached."
                    f"  best_val_nll={best_val_nll:.4f}  det_err={last_det_err:.1f}cm"
                )
                break

    json.dump({
        "K": args.K, "best_val_nll": best_val_nll,
        "det_err_at_best": last_det_err,
    }, open(out_dir / "config.json", "w"), indent=2)
    logger.log(f"Saved → {out_dir}  best_val_nll={best_val_nll:.4f}")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",       default="world_model/results/spr_mdn_v26_p1/best.pt")
    p.add_argument("--data-dir",       default="world_model/data_fixeddt")
    p.add_argument("--out-dir",        default="world_model/results/spr_mdn_v27_p2")
    p.add_argument("--K",              type=int,   default=5)
    p.add_argument("--max-epochs",     type=int,   default=2000)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--eval-every",     type=int,   default=10)
    p.add_argument("--batch-size",     type=int,   default=512)
    p.add_argument("--lr-mdn",         type=float, default=1e-4)
    p.add_argument("--lr-encdec",      type=float, default=1e-5,
                   help="enc/dec LR for Phase B (10x lower — slow follow)")
    p.add_argument("--phase-b-every",  type=int,   default=10,
                   help="run Phase B every N epochs (9:1 ratio)")
    p.add_argument("--lam-phase-b",    type=float, default=1.0,
                   help="scale factor on Phase B reconstruction loss")
    p.add_argument("--ewta-warmup",    type=int,   default=200,
                   help="epochs to linearly grow k_top from 1 → K")
    p.add_argument("--beta-ent",       type=float, default=0.05,
                   help="entropy regularization weight (H(π) bonus)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
