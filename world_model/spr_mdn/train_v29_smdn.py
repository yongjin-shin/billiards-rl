"""
world_model/spr_mdn/train_v29_smdn.py  —  S-space MDN v29

v28 대비 변경:
  1. beta_ent annealing:  beta_start → beta_end  (선형, anneal_epochs 동안)
                          → entropy reg을 초반에 강하게, 이후 약하게
                          → mu-spread 강제 (pi 균등 유지하되 mu 분산 허용)
  2. k_top 최대 k_max_top(=3):  warmup 후에도 K=5에 도달하지 않음
                          → EWTA winner-set이 항상 minority → mu collapse 방지
  3. LR: CosineAnnealingWarmRestarts (T_0, T_mult)  + 높은 초기 lr
                          → 주기적 LR 재시작으로 local minimum 탈출

Architecture (v28과 동일):
  encoder(s_t) → z_t → transition^t → z_{t+k}
  SMDNDecoder(z) → (π_k, μ_k^s, σ_k)   K=5 GMM over s-space (14-dim)
  TypeHead(z)   → shot type logits

Usage:
    python world_model/spr_mdn/train_v29_smdn.py \
      --out-dir world_model/results/spr_mdn_v29_smdn \
      2>&1 | tee /tmp/v29_smdn.log
"""

import math, os, sys, argparse, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.fixeddt_model import StateEncoder, LATENT_DIM
from world_model.ssm_model import ResTransition, TypeHead
from world_model.spr_mdn.spr_dataset import (
    SPRDataset, SPREpisodeSubset, _CapDataset, make_balanced_val_eps,
)
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.generate_data_fixeddt import DT
from log_utils import Logger

T_MAX = 60
T_MIN = 10
S_DIM = 14


# ─────────────────────────────────────────────────────────────────────────────
class SMDNDecoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, K: int = 5, s_dim: int = S_DIM):
        super().__init__()
        self.K = K; self.D = s_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),        nn.SiLU(),
        )
        self.pi_head    = nn.Linear(256, K)
        self.mu_head    = nn.Linear(256, K * s_dim)
        self.sigma_head = nn.Linear(256, K)

    def forward(self, z: torch.Tensor):
        h = self.net(z)
        return (self.pi_head(h),
                self.mu_head(h).view(-1, self.K, self.D),
                self.sigma_head(h))

    @torch.no_grad()
    def decode_det(self, z: torch.Tensor) -> torch.Tensor:
        pi, mu, _ = self(z)
        k_star = pi.argmax(dim=-1)
        return mu[torch.arange(len(z)), k_star]


class V29SMDNModel(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, K: int = 5):
        super().__init__()
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ResTransition(latent_dim, use_ar_state=False)
        self.decoder    = SMDNDecoder(latent_dim, K, S_DIM)
        self.type_head  = TypeHead(latent_dim)

    @torch.no_grad()
    def rollout_det(self, s_0: torch.Tensor, T: int):
        z = self.encoder(s_0)
        s_list, t_list = [], []
        for _ in range(T):
            z = self.transition(z)
            s_list.append(self.decoder.decode_det(z))
            t_list.append(self.type_head(z))
        return torch.stack(s_list, 1), torch.stack(t_list, 1)


# ─────────────────────────────────────────────────────────────────────────────
def ewta_nll_s(
    pi_logits:  torch.Tensor,   # (B, K)
    mu:         torch.Tensor,   # (B, K, D)
    log_sigma:  torch.Tensor,   # (B, K)
    s_target:   torch.Tensor,   # (B, D)
    k_top:      int   = 1,
    sigma_min:  float = 0.01,
) -> torch.Tensor:
    B, K, D = mu.shape
    sigma   = log_sigma.exp().clamp(min=sigma_min)

    s_exp   = s_target.unsqueeze(1).expand_as(mu)
    sq_dist = ((s_exp - mu) ** 2).sum(-1)               # (B, K)

    k_top    = min(k_top, K)
    topk_idx = sq_dist.topk(k_top, dim=-1, largest=False).indices
    mask     = torch.zeros(B, K, device=mu.device).scatter_(1, topk_idx, 1.0)

    log_p_k = (
        F.log_softmax(pi_logits, dim=-1)
        - D / 2 * math.log(2 * math.pi)
        - D * torch.log(sigma)
        - 0.5 * sq_dist / sigma ** 2
    )
    return -(log_p_k * mask).sum(-1).mean()


def mdn_nll_s(
    pi_logits:  torch.Tensor,
    mu:         torch.Tensor,
    log_sigma:  torch.Tensor,
    s_target:   torch.Tensor,
    sigma_min:  float = 0.01,
) -> torch.Tensor:
    B, K, D = mu.shape
    sigma   = log_sigma.exp().clamp(min=sigma_min)
    s_exp   = s_target.unsqueeze(1).expand_as(mu)
    sq_dist = ((s_exp - mu) ** 2).sum(-1)
    log_p_k = (
        F.log_softmax(pi_logits, dim=-1)
        - D / 2 * math.log(2 * math.pi)
        - D * torch.log(sigma)
        - 0.5 * sq_dist / sigma ** 2
    )
    return -torch.logsumexp(log_p_k, dim=-1).mean()


def get_k_top(epoch: int, k_max: int, warmup: int) -> int:
    """1 → k_max 선형 증가 (warmup epochs 동안). k_max 이상은 절대 안 됨."""
    if warmup <= 0:
        return k_max
    return min(k_max, 1 + int(min(epoch / warmup, 1.0) * (k_max - 1) + 0.5))



# ─────────────────────────────────────────────────────────────────────────────
def _eval_err(model: V29SMDNModel, episodes: list, device: str, steps: int = 60) -> dict:
    model.eval()
    valid = [(ep_s, ep_f, ep_t)
             for ep_s, ep_f, ep_t, _, _a in episodes
             if len(ep_s) >= steps + 1]
    cp = {"0.5s": int(0.5/DT), "1.0s": int(1.0/DT),
          "2.0s": int(2.0/DT), "3.0s": int(3.0/DT)}
    errs_all, errs_bb, errs_nbb = [], [], []
    cp_errs = {k: [] for k in cp}
    with torch.no_grad():
        s0 = torch.from_numpy(np.stack([e[0][0] for e in valid])).float().to(device)
        pr, _ = model.rollout_det(s0, steps)
        pr = pr.cpu().numpy()
    for b, (ep_s, ep_f, ep_t) in enumerate(valid):
        gt = ep_s[1:steps+1]
        p  = pr[b]
        ce = np.sqrt(((p[:,0]-gt[:,0])*TABLE_W)**2 + ((p[:,1]-gt[:,1])*TABLE_H)**2)
        te = np.sqrt(((p[:,7]-gt[:,7])*TABLE_W)**2 + ((p[:,8]-gt[:,8])*TABLE_H)**2)
        se = (ce + te) / 2 * 100
        errs_all.append(se.mean())
        has_bb = bool(np.any(ep_f[:steps] & (ep_t[:steps] == 1)))
        (errs_bb if has_bb else errs_nbb).append(se.mean())
        for lbl, t in cp.items():
            cp_errs[lbl].append(se[t-1])
    def _m(l): return float(np.mean(l)) if l else float("nan")
    return {"mean_err": _m(errs_all), "bb": _m(errs_bb), "nbb": _m(errs_nbb),
            **{k: _m(v) for k, v in cp_errs.items()}}


def _eval_nll(model: V29SMDNModel, val_loader, device: str) -> tuple[float, dict]:
    model.eval()
    nlls, mu_spreads, sigma_means, entropies = [], [], [], []
    with torch.no_grad():
        for seq_s, *_ in val_loader:
            seq_s = seq_s.to(device)
            T = seq_s.shape[1] - 1
            z = model.encoder(seq_s[:, 0])
            for t in range(min(T, 5)):
                z = model.transition(z)
                pi_l, mu, ls = model.decoder(z)
                nlls.append(mdn_nll_s(pi_l, mu, ls, seq_s[:, t+1]).item())
                sigma = ls.exp().clamp(min=0.01)
                pi    = F.softmax(pi_l, dim=-1)
                # mu_spread: 컴포넌트 간 pairwise mean dist (cue+tgt xy)
                mu_xy  = mu[:, :, [0,1,7,8]]          # (B,K,4)
                diffs  = mu_xy.unsqueeze(2) - mu_xy.unsqueeze(1)  # (B,K,K,4)
                spread = diffs.norm(dim=-1).mean()     # scalar
                mu_spreads.append(spread.item())
                sigma_means.append(sigma.mean().item())
                entropies.append(-(pi * (pi+1e-12).log()).sum(-1).mean().item())
    eff_K = float(np.exp(np.mean(entropies)))
    return float(np.mean(nlls)), {
        "mu_spread":  float(np.mean(mu_spreads)),
        "sigma_mean": float(np.mean(sigma_means)),
        "eff_K":      eff_K,
    }


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(
        f"V29 SMDN  K={args.K}  k_max_top={args.k_max_top}"
        f"  ewta_warmup={args.ewta_warmup}  (no entropy reg)"
        f"  lr={args.lr}  T0={args.lr_T0}  Tmult={args.lr_Tmult}"
        f"  max_epochs={args.max_epochs}  patience={args.patience}"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = SPRDataset(args.data_dir, logger=logger)
    rng     = np.random.default_rng(0)
    perm    = rng.permutation(len(dataset.episodes))
    n_val   = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all   = [dataset.episodes[i] for i in perm[:n_val]]
    train_eps_all = [dataset.episodes[i] for i in perm[n_val:]]

    balanced_val = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    train_eps    = [ep for ep in train_eps_all if len(ep[0]) >= T_MAX + 1]
    val_eps      = [ep for ep in balanced_val   if len(ep[0]) >= T_MAX + 1]
    logger.log(f"Train: {len(train_eps):,}  Val: {len(val_eps):,}")

    train_ds = _CapDataset(
        SPREpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True), 60_000)
    val_ds   = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = V29SMDNModel(K=args.K).to(device)
    n_enc  = sum(p.numel() for p in model.encoder.parameters())
    n_trans= sum(p.numel() for p in model.transition.parameters())
    n_dec  = sum(p.numel() for p in model.decoder.parameters())
    logger.log(f"Params: enc={n_enc:,}  trans={n_trans:,}  dec={n_dec:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.lr_T0, T_mult=args.lr_Tmult, eta_min=args.lr * 0.01)

    rng_T       = np.random.default_rng(42)
    best_err    = float("inf")
    stall_count = 0
    last_err    = float("nan")
    last_nll    = float("nan")
    last_stats  = {}
    last_rerr   = {}

    for epoch in range(1, args.max_epochs + 1):
        cur_T  = int(rng_T.integers(T_MIN, T_MAX + 1))
        k_top  = get_k_top(epoch, args.k_max_top, args.ewta_warmup)
        cur_lr = opt.param_groups[0]["lr"]

        model.train()
        nll_losses, type_losses = [], []

        for seq_s, seq_f, seq_t, *_ in train_loader:
            seq_s = seq_s[:, :cur_T + 1].to(device)
            seq_t = seq_t[:, :cur_T].to(device)

            z = model.encoder(seq_s[:, 0])
            L_nll  = torch.tensor(0.0, device=device)
            L_type = torch.tensor(0.0, device=device)

            for t in range(cur_T):
                z = model.transition(z)
                pi_l, mu, ls = model.decoder(z)
                L_nll  = L_nll + ewta_nll_s(
                    pi_l, mu, ls, seq_s[:, t+1],
                    k_top=k_top,
                )
                L_type = L_type + F.cross_entropy(
                    model.type_head(z), seq_t[:, t].long())

            loss = (L_nll + args.lam_type * L_type) / cur_T
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nll_losses.append(L_nll.item() / cur_T)
            type_losses.append(L_type.item() / cur_T)

        sched.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        do_eval = (epoch % args.eval_every == 0 or epoch == 1)
        if do_eval:
            last_rerr  = _eval_err(model, balanced_val, device)
            last_nll, last_stats = _eval_nll(model, val_loader, device)
            last_err   = last_rerr["mean_err"]

        cp_str = ""
        if do_eval:
            cp_str = " | ".join(f"{k}={last_rerr[k]:.1f}cm"
                                 for k in ["0.5s","1.0s","2.0s","3.0s"]
                                 if not np.isnan(last_rerr.get(k, float("nan"))))

        log_line = (
            f"Epoch {epoch:4d}"
            f"  [T={cur_T:2d}  k={k_top}/{args.K}  lr={cur_lr:.2e}]"
            f"  nll={np.mean(nll_losses):.3f}  type={np.mean(type_losses):.4f}"
            f"  val_nll={last_nll:.3f}"
            f"  err={last_err:.1f}cm"
            f"  (bb={last_rerr.get('bb', float('nan')):.1f}/nbb={last_rerr.get('nbb', float('nan')):.1f})"
        )
        if do_eval:
            log_line += (
                f"  [mdn: spread={last_stats.get('mu_spread',0):.4f}"
                f"  σ={last_stats.get('sigma_mean',0):.3f}"
                f"  effK={last_stats.get('eff_K',0):.2f}/{args.K}]"
            )
            if cp_str:
                log_line += f"  [{cp_str}]"
        logger.log(log_line)

        # ── Checkpoint + early stop ────────────────────────────────────────────
        if do_eval:
            if last_err < best_err - 0.1:
                best_err    = last_err
                stall_count = 0
                torch.save({
                    "state":    model.state_dict(),
                    "epoch":    epoch,
                    "mean_err": last_err,
                    "val_nll":  last_nll,
                    "K":        args.K,
                }, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")

            if stall_count >= args.patience:
                logger.log(
                    f"\nEarly stop — best_err={best_err:.2f}cm  val_nll={last_nll:.3f}")
                break

    json.dump({"K": args.K, "best_err": best_err}, open(out_dir / "config.json", "w"))
    logger.log(f"Done → {out_dir}")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       default="world_model/data_fixeddt")
    p.add_argument("--out-dir",        default="world_model/results/spr_mdn_v29_smdn")
    p.add_argument("--K",              type=int,   default=5)
    p.add_argument("--k-max-top",      type=int,   default=3,
                   help="EWTA k_top ceiling — never trains all K toward same target")
    p.add_argument("--ewta-warmup",    type=int,   default=150,
                   help="epochs to grow k_top 1→k_max_top")
    p.add_argument("--lr",             type=float, default=3e-3,
                   help="peak learning rate")
    p.add_argument("--lr-T0",          type=int,   default=100,
                   help="CosineAnnealingWarmRestarts initial cycle length")
    p.add_argument("--lr-Tmult",       type=int,   default=2,
                   help="cycle length multiplier per restart")
    p.add_argument("--lam-type",       type=float, default=0.1)
    p.add_argument("--max-epochs",     type=int,   default=2000)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--eval-every",     type=int,   default=10)
    p.add_argument("--batch-size",     type=int,   default=512)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
