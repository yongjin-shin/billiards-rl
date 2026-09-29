"""
world_model/spr_mdn/train_v28_p1.py  —  New Phase 1: Joint Training + Encoder Alignment

스크래치부터 전체 joint 학습. 두 loss 합산:

  L_dyn  (SSM v18 동일):
    z_0 = encoder(s_0)                 ← gradient ON (encoder까지 흐름)
    z_t = transition^t(z_0)
    loss = MSE(decoder(z_t), s_t)  t=1..T
    → encoder + transition + decoder 모두 학습

  L_enc  (encoder alignment, lam_enc 비율로):
    for t in 1..T:
      z_t = encoder(s_t)               ← gradient ON
      Path 1: [frozen dec](z_t)        → MSE with s_t       (현재 재건)
      Path 2: [frozen trans](z_t) → [frozen dec] → MSE with s_{t+1}  (다음 스텝)
    → encoder만 추가 정렬. transition/decoder는 이 path에서 frozen

  total = L_dyn + lam_enc * L_enc

"천천히 따라가기": lam_enc를 작게 (0.1~0.3) 줘서 encoder가 L_dyn gradient에
  주로 의존하되 L_enc로 t>0 정렬을 서서히 학습

Usage:
    python world_model/spr_mdn/train_v28_p1.py \
      --data-dir world_model/data_fixeddt \
      --out-dir  world_model/results/spr_mdn_v28_p1 \
      2>&1 | tee /tmp/v28_p1.log
"""

import os, sys, argparse, json
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
class V28Model(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ResTransition(latent_dim, use_ar_state=False)
        self.cue_head   = CueBallHead(latent_dim)
        self.tgt_head   = TgtBallHead(latent_dim)
        self.type_head  = TypeHead(latent_dim)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    @torch.no_grad()
    def rollout_det(self, s_0: torch.Tensor, T: int):
        z = self.encoder(s_0)
        s_list = [self._decode(z)]
        for _ in range(T):
            z = self.transition(z)
            s_list.append(self._decode(z))
        return torch.stack(s_list, 1)   # (B, T+1, 14)


# ─────────────────────────────────────────────────────────────────────────────
def _eval_err(model: V28Model, episodes: list, device: str, steps: int = 60) -> dict:
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
        pr = model.rollout_det(s0, steps).cpu().numpy()
    for b, (ep_s, ep_f, ep_t) in enumerate(valid):
        gt = ep_s[1:steps+1]
        p  = pr[b, 1:steps+1]
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


def _eval_align(model: V28Model, val_loader, device: str) -> dict:
    """encoder(s_{t+1}) vs transition(encoder(s_t)) L2 거리."""
    model.eval()
    dists, recon_l, dyn_l = [], [], []
    with torch.no_grad():
        for seq_s, *_ in val_loader:
            seq_s = seq_s.to(device)
            T = seq_s.shape[1] - 1
            for t in range(min(T, 5)):
                z_t        = model.encoder(seq_s[:, t])
                z_t1_enc   = model.encoder(seq_s[:, t+1])
                z_t1_trans = model.transition(z_t)
                dists.append((z_t1_enc - z_t1_trans).norm(dim=-1).mean().item())
                recon_l.append(F.mse_loss(model._decode(z_t), seq_s[:, t]).item())
                dyn_l.append(F.mse_loss(model._decode(z_t1_trans), seq_s[:, t+1]).item())
    return {
        "align_dist": float(np.mean(dists)),
        "recon_mse":  float(np.mean(recon_l)),
        "dyn_mse":    float(np.mean(dyn_l)),
    }


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(
        f"V28 Phase1 (joint)  lr_dyn={args.lr_dyn}  lr_enc={args.lr_enc}"
        f"  lam_enc={args.lam_enc}  lam_dyn={args.lam_dyn}  lam_recon={args.lam_recon}"
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

    # ── Model & Optimizers ────────────────────────────────────────────────────
    model = V28Model().to(device)
    logger.log(f"Params: enc={sum(p.numel() for p in model.encoder.parameters()):,}"
               f"  dyn={sum(p.numel() for p in list(model.transition.parameters()) + list(model.cue_head.parameters()) + list(model.tgt_head.parameters())):,}")

    dyn_params = (list(model.transition.parameters()) +
                  list(model.cue_head.parameters()) +
                  list(model.tgt_head.parameters()) +
                  list(model.type_head.parameters()))
    enc_params = list(model.encoder.parameters())

    opt_dyn = torch.optim.Adam(dyn_params, lr=args.lr_dyn)
    opt_enc = torch.optim.Adam(enc_params, lr=args.lr_enc)

    sched_dyn = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_dyn, T_max=args.max_epochs, eta_min=args.lr_dyn * 0.01)
    sched_enc = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_enc, T_max=args.max_epochs, eta_min=args.lr_enc * 0.01)

    rng_T = np.random.default_rng(42)
    best_err    = float("inf")
    stall_count = 0
    last_err    = float("nan")
    last_align  = {}
    last_rerr   = {}

    for epoch in range(1, args.max_epochs + 1):
        cur_T = int(rng_T.integers(T_MIN, T_MAX + 1))
        model.train()
        total_losses, dyn_losses, enc_losses = [], [], []

        # encoder loss path에서 dyn_params gradient 차단용
        for p in dyn_params:
            p.requires_grad_(True)

        for seq_s, *_ in train_loader:
            seq_s = seq_s[:, :cur_T + 1].to(device)   # (B, T+1, 14)

            # ── L_dyn: SSM v18 동일, encoder까지 gradient 흐름 ────────────────
            z = model.encoder(seq_s[:, 0])             # gradient ON
            L_dyn = torch.tensor(0.0, device=device)
            for t in range(cur_T):
                z = model.transition(z)
                L_dyn = L_dyn + F.mse_loss(model._decode(z), seq_s[:, t+1])
            L_dyn = L_dyn / cur_T
            dyn_losses.append(L_dyn.item())

            # ── L_enc: encoder(s_t) t>0 정렬, dyn_params frozen ──────────────
            for p in dyn_params:
                p.requires_grad_(False)

            L_enc = torch.tensor(0.0, device=device)
            for t in range(cur_T):
                z_t = model.encoder(seq_s[:, t])
                # Path 1: 현재 재건
                L_enc = L_enc + args.lam_recon * F.mse_loss(
                    model._decode(z_t), seq_s[:, t])
                # Path 2: 다음 스텝 (transition/decoder frozen, grad → encoder)
                z_t1 = model.transition(z_t)
                L_enc = L_enc + args.lam_dyn * F.mse_loss(
                    model._decode(z_t1), seq_s[:, t+1])
            L_enc = L_enc / cur_T
            enc_losses.append(L_enc.item())

            for p in dyn_params:
                p.requires_grad_(True)

            # ── 합산 업데이트 ─────────────────────────────────────────────────
            loss = L_dyn + args.lam_enc * L_enc
            total_losses.append(loss.item())
            opt_dyn.zero_grad(); opt_enc.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt_dyn.step(); opt_enc.step()

        sched_dyn.step()
        sched_enc.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        do_eval = (epoch % args.eval_every == 0 or epoch == 1)
        if do_eval:
            last_rerr  = _eval_err(model, balanced_val, device)
            last_align = _eval_align(model, val_loader, device)
            last_err   = last_rerr["mean_err"]

        log_line = (
            f"Epoch {epoch:4d}  [T={cur_T:2d}]"
            f"  L_dyn={np.mean(dyn_losses):.4f}  L_enc={np.mean(enc_losses):.4f}"
            f"  err={last_err:.1f}cm"
            f"  (bb={last_rerr.get('bb', float('nan')):.1f}/nbb={last_rerr.get('nbb', float('nan')):.1f})"
        )
        if do_eval:
            cp_str = " | ".join(f"{k}={last_rerr[k]:.1f}cm"
                                 for k in ["0.5s","1.0s","2.0s","3.0s"]
                                 if not np.isnan(last_rerr.get(k, float("nan"))))
            log_line += (
                f"  align={last_align['align_dist']:.4f}"
                f"  recon={last_align['recon_mse']:.4f}"
                f"  dyn={last_align['dyn_mse']:.4f}"
            )
            if cp_str:
                log_line += f"  [{cp_str}]"
        logger.log(log_line)

        # ── Checkpoint + Early stop (by det_err) ──────────────────────────────
        if do_eval:
            if last_err < best_err - 0.1:
                best_err    = last_err
                stall_count = 0
                torch.save({
                    "state":      model.state_dict(),
                    "epoch":      epoch,
                    "mean_err":   last_err,
                    "align_dist": last_align["align_dist"],
                    "latent_dim": LATENT_DIM,
                }, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")

            if stall_count >= args.patience:
                logger.log(f"\nEarly stop — best_err={best_err:.2f}cm  align={last_align['align_dist']:.4f}")
                break

    json.dump({"best_err": best_err, "latent_dim": LATENT_DIM}, open(out_dir / "config.json", "w"))
    logger.log(f"Done → {out_dir}")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="world_model/data_fixeddt")
    p.add_argument("--out-dir",     default="world_model/results/spr_mdn_v28_p1")
    p.add_argument("--max-epochs",  type=int,   default=2000)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--eval-every",  type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr-dyn",      type=float, default=1e-3,
                   help="LR for transition + decoder")
    p.add_argument("--lr-enc",      type=float, default=1e-3,
                   help="LR for encoder")
    p.add_argument("--lam-enc",     type=float, default=0.1,
                   help="weight on L_enc (encoder alignment). small = slow follow")
    p.add_argument("--lam-dyn",     type=float, default=1.0,
                   help="weight on Path 2 (next-step) inside L_enc")
    p.add_argument("--lam-recon",   type=float, default=1.0,
                   help="weight on Path 1 (current recon) inside L_enc")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
