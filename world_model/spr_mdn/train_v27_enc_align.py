"""
world_model/spr_mdn/train_v27_enc_align.py  —  Encoder Alignment (Phase 1.5)

Phase 1 backbone (transition + decoder) 은 frozen.
Encoder만 두 경로로 학습:

  Path 1 — 현재 재건:
    encoder(s_t) → z_t → sg(decoder) → s_t_hat
    Loss: MSE(s_t_hat, s_t)
    → encoder가 decoder와 호환되는 z_t 생성

  Path 2 — 다음 스텝 예측:
    encoder(s_t) → z_t → sg(transition) → z_t+1 → sg(decoder) → s_t+1_hat
    Loss: MSE(s_t+1_hat, s_{t+1})
    → encoder가 transition의 z-space에 align

gradient는 encoder로만 흐름. transition/decoder 파라미터 완전 frozen.

Usage:
    python world_model/spr_mdn/train_v27_enc_align.py \
      --backbone world_model/results/spr_mdn_v26_p1/best.pt \
      --out-dir  world_model/results/spr_mdn_v27_enc \
      2>&1 | tee /tmp/v27_enc.log
"""

import math, os, sys, argparse
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
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.generate_data_fixeddt import DT
from log_utils import Logger

T_MAX = 60
T_MIN = 10


# ─────────────────────────────────────────────────────────────────────────────
class AlignModel(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ResTransition(latent_dim, use_ar_state=False)
        self.cue_head   = CueBallHead(latent_dim)
        self.tgt_head   = TgtBallHead(latent_dim)
        self.type_head  = TypeHead(latent_dim)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)  # (B, 14)

    @torch.no_grad()
    def rollout_det(self, s_0: torch.Tensor, T: int):
        z = self.encoder(s_0)
        s_list = [self._decode(z)]
        for _ in range(T):
            z = self.transition(z)
            s_list.append(self._decode(z))
        return torch.stack(s_list, 1)   # (B, T+1, 14)


# ─────────────────────────────────────────────────────────────────────────────
def _eval_err(model: AlignModel, episodes: list, device: str,
              rollout_steps: int = 60) -> float:
    model.eval()
    valid = [(ep_s, ep_f, ep_t)
             for ep_s, ep_f, ep_t, _, _a in episodes
             if len(ep_s) >= rollout_steps + 1]
    errs = []
    with torch.no_grad():
        s0 = torch.from_numpy(np.stack([e[0][0] for e in valid])).float().to(device)
        pr = model.rollout_det(s0, rollout_steps).cpu().numpy()
    for b, (ep_s, ep_f, ep_t) in enumerate(valid):
        gt = ep_s[1:rollout_steps+1]
        p  = pr[b, 1:rollout_steps+1]
        ce = np.sqrt(((p[:,0]-gt[:,0])*TABLE_W)**2 + ((p[:,1]-gt[:,1])*TABLE_H)**2)
        te = np.sqrt(((p[:,7]-gt[:,7])*TABLE_W)**2 + ((p[:,8]-gt[:,8])*TABLE_H)**2)
        errs.append(((ce + te) / 2 * 100).mean())
    return float(np.mean(errs))


def _eval_align(model: AlignModel, val_loader, device: str) -> dict:
    """encoder(s_{t+1}) vs transition(encoder(s_t)) 평균 L2 거리."""
    model.eval()
    dists, recon_errs, dyn_errs = [], [], []
    with torch.no_grad():
        for seq_s, *_ in val_loader:
            seq_s = seq_s.to(device)
            T = seq_s.shape[1] - 1
            for t in range(min(T, 10)):
                z_t   = model.encoder(seq_s[:, t])
                z_t1_enc  = model.encoder(seq_s[:, t+1])
                z_t1_trans = model.transition(z_t)
                # alignment: L2 distance in z-space
                dists.append((z_t1_enc - z_t1_trans).norm(dim=-1).mean().item())
                # recon error
                s_hat = model._decode(z_t)
                recon_errs.append(F.mse_loss(s_hat, seq_s[:, t]).item())
                # dyn error
                s_t1_hat = model._decode(z_t1_trans)
                dyn_errs.append(F.mse_loss(s_t1_hat, seq_s[:, t+1]).item())
    return {
        "align_dist": float(np.mean(dists)),
        "recon_mse":  float(np.mean(recon_errs)),
        "dyn_mse":    float(np.mean(dyn_errs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(
        f"Encoder Alignment  lr={args.lr}  lam_dyn={args.lam_dyn}"
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

    MAX_ITEMS  = 60_000
    train_ds   = _CapDataset(SPREpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True),  MAX_ITEMS)
    val_ds     = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AlignModel().to(device)

    ckpt  = torch.load(args.backbone, map_location=device, weights_only=False)
    state = ckpt.get("state", ckpt)
    model.load_state_dict(state, strict=False)
    logger.log(f"Backbone loaded: {args.backbone}")

    # transition + decoder 완전 freeze
    frozen_modules = [model.transition, model.cue_head, model.tgt_head, model.type_head]
    for m in frozen_modules:
        for p in m.parameters():
            p.requires_grad_(False)
    logger.log("Frozen: transition, cue_head, tgt_head, type_head")

    enc_params = list(model.encoder.parameters())
    n_enc = sum(p.numel() for p in enc_params)
    logger.log(f"Encoder params: {n_enc:,}")

    opt   = torch.optim.Adam(enc_params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=args.lr * 0.01)

    rng_T = np.random.default_rng(42)
    best_dist   = float("inf")
    stall_count = 0

    for epoch in range(1, args.max_epochs + 1):
        cur_T = int(rng_T.integers(T_MIN, T_MAX + 1))
        model.train()
        losses = []

        for seq_s, *_ in train_loader:
            seq_s = seq_s[:, :cur_T + 1].to(device)   # (B, T+1, 14)
            B, Tp1, _ = seq_s.shape

            loss_recon = torch.tensor(0.0, device=device)
            loss_dyn   = torch.tensor(0.0, device=device)

            for t in range(cur_T):
                z_t = model.encoder(seq_s[:, t])          # gradient ON

                # Path 1: 현재 재건 (encoder → frozen decoder)
                s_t_hat = model._decode(z_t)
                loss_recon = loss_recon + F.mse_loss(s_t_hat, seq_s[:, t])

                # Path 2: 다음 스텝 (encoder → frozen transition → frozen decoder)
                z_t1      = model.transition(z_t)          # transition frozen (params), grad flows
                s_t1_hat  = model._decode(z_t1)
                loss_dyn  = loss_dyn + F.mse_loss(s_t1_hat, seq_s[:, t+1])

            loss = (loss_recon + args.lam_dyn * loss_dyn) / cur_T
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc_params, 1.0)
            opt.step()
            losses.append(loss.item())

        opt.step()
        sched.step()

        do_eval = (epoch % args.eval_every == 0 or epoch == 1)
        if do_eval:
            stats    = _eval_align(model, val_loader, device)
            det_err  = _eval_err(model, balanced_val, device)
            cur_dist = stats["align_dist"]

        tr_loss = float(np.mean(losses))
        log_line = (
            f"Epoch {epoch:4d}  [T={cur_T}]"
            f"  tr={tr_loss:.4f}"
        )
        if do_eval:
            log_line += (
                f"  align_dist={stats['align_dist']:.4f}"
                f"  recon_mse={stats['recon_mse']:.4f}"
                f"  dyn_mse={stats['dyn_mse']:.4f}"
                f"  det_err={det_err:.1f}cm"
            )
        logger.log(log_line)

        if do_eval:
            if cur_dist < best_dist - 1e-4:
                best_dist   = cur_dist
                stall_count = 0
                torch.save({
                    "state":      model.state_dict(),
                    "epoch":      epoch,
                    "align_dist": cur_dist,
                    "det_err":    det_err,
                }, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")

            if stall_count >= args.patience:
                logger.log(f"\nEarly stop — best align_dist={best_dist:.4f}  det_err={det_err:.1f}cm")
                break

    logger.log(f"Done → {out_dir}  best_align_dist={best_dist:.4f}")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",    default="world_model/results/spr_mdn_v26_p1/best.pt")
    p.add_argument("--data-dir",    default="world_model/data_fixeddt")
    p.add_argument("--out-dir",     default="world_model/results/spr_mdn_v27_enc")
    p.add_argument("--max-epochs",  type=int,   default=500)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--eval-every",  type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lam-dyn",     type=float, default=1.0,
                   help="weight on dynamics loss (Path 2) vs recon loss (Path 1)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
