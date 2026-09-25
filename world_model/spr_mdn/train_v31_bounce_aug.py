"""
world_model/spr_mdn/train_v31_bounce_aug.py  —  ObsTransition + Bounce Augmentation

v30_obs_b 대비 변경:
    - BounceAugDataset: t=0 + 모든 바운스 직후 지점을 sub-sequence 시작점으로 사용
      (37k 에피소드 → 235k+ 학습 샘플, +6x)
    - _CapDataset 제거: BounceAugDataset 자체가 충분히 큼
    - 나머지 동일: ObsTransition, MSE, CosineAnnealingLR, no-restart, clip=0.5

Usage:
    python world_model/spr_mdn/train_v31_bounce_aug.py \
      --out-dir world_model/results/spr_mdn_v31_bounce \
      2>&1 | tee /tmp/v31_bounce.log
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
from world_model.ssm_model import TypeHead
from world_model.spr_mdn.spr_dataset import (
    SPRDataset, SPREpisodeSubset, BounceAugDataset, make_balanced_val_eps,
)
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.generate_data_fixeddt import DT
from log_utils import Logger

T_MAX = 60
T_MIN = 2
S_DIM = 14


# ─────────────────────────────────────────────────────────────────────────────
class ObsTransition(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, s_dim: int = S_DIM):
        super().__init__()
        in_dim = latent_dim + s_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),    nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.norm(z + self.net(torch.cat([z, s], dim=-1)))


class MuHead(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, s_dim: int = S_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, s_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class V31Model(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = StateEncoder(latent_dim)
        self.transition = ObsTransition(latent_dim, S_DIM)
        self.mu_head    = MuHead(latent_dim, S_DIM)
        self.type_head  = TypeHead(latent_dim)

    @torch.no_grad()
    def rollout_det(self, s_0: torch.Tensor, T: int):
        z     = self.encoder(s_0)
        s_cur = s_0
        s_list, t_list = [], []
        for _ in range(T):
            z     = self.transition(z, s_cur)
            s_cur = self.mu_head(z)
            s_list.append(s_cur)
            t_list.append(self.type_head(z))
        return torch.stack(s_list, 1), torch.stack(t_list, 1)


# ─────────────────────────────────────────────────────────────────────────────
def _eval_err(model: V31Model, episodes: list, device: str,
              steps: int = 60) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)

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

    train_ds = BounceAugDataset(train_eps, rollout_steps=T_MAX, augment=True)
    val_ds   = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    logger.log(
        f"V31 ObsTransition + BounceAug"
        f"  train_samples={len(train_ds):,}  T curriculum {args.t_start}→{T_MAX} over {args.t_curriculum}ep"
        f"  lr={args.lr} (CosineAnnealingLR T_max={args.max_epochs})"
        f"  max_epochs={args.max_epochs}  patience={args.patience}"
    )
    logger.log(f"Train eps: {len(train_eps):,}  BounceAug samples: {len(train_ds):,}  Val: {len(val_eps):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = V31Model().to(device)
    n_enc  = sum(p.numel() for p in model.encoder.parameters())
    n_trans= sum(p.numel() for p in model.transition.parameters())
    n_dec  = sum(p.numel() for p in model.mu_head.parameters())
    logger.log(f"Params: enc={n_enc:,}  trans={n_trans:,}  dec={n_dec:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_epochs, eta_min=args.lr * 0.001)

    rng_T       = np.random.default_rng(42)
    best_err    = float("inf")
    stall_count = 0
    last_err    = float("nan")
    last_rerr   = {}

    for epoch in range(1, args.max_epochs + 1):
        if args.t_curriculum > 0:
            frac = min(epoch / args.t_curriculum, 1.0)
            t_train_max = int(args.t_start + frac * (T_MAX - args.t_start))
        else:
            t_train_max = T_MAX
        cur_T  = int(rng_T.integers(T_MIN, t_train_max + 1))
        cur_lr = opt.param_groups[0]["lr"]

        model.train()
        mse_losses, type_losses = [], []

        for seq_s, seq_f, seq_t, *_ in train_loader:
            seq_s = seq_s[:, :cur_T + 1].to(device)
            seq_t = seq_t[:, :cur_T].to(device)

            z     = model.encoder(seq_s[:, 0])
            s_cur = seq_s[:, 0]

            L_mse  = torch.tensor(0.0, device=device)
            L_type = torch.tensor(0.0, device=device)

            for t in range(cur_T):
                z     = model.transition(z, s_cur)
                s_hat = model.mu_head(z)
                L_mse  = L_mse  + F.mse_loss(s_hat, seq_s[:, t + 1])
                L_type = L_type + F.cross_entropy(
                    model.type_head(z), seq_t[:, t].long())
                s_cur = s_hat

            loss = (L_mse + args.lam_type * L_type) / cur_T
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            mse_losses.append(L_mse.item() / cur_T)
            type_losses.append(L_type.item() / cur_T)

        sched.step()

        do_eval = (epoch % args.eval_every == 0 or epoch == 1)
        if do_eval:
            last_rerr = _eval_err(model, balanced_val, device)
            last_err  = last_rerr["mean_err"]

        cp_str = ""
        if do_eval:
            cp_str = " | ".join(f"{k}={last_rerr[k]:.1f}cm"
                                 for k in ["0.5s","1.0s","2.0s","3.0s"]
                                 if not np.isnan(last_rerr.get(k, float("nan"))))

        log_line = (
            f"Epoch {epoch:4d}"
            f"  [T={cur_T:2d}/{t_train_max}  lr={cur_lr:.2e}]"
            f"  mse={np.mean(mse_losses):.5f}  type={np.mean(type_losses):.4f}"
            f"  err={last_err:.1f}cm"
            f"  (bb={last_rerr.get('bb', float('nan')):.1f}"
            f"/nbb={last_rerr.get('nbb', float('nan')):.1f})"
        )
        if do_eval and cp_str:
            log_line += f"  [{cp_str}]"
        logger.log(log_line)

        if do_eval:
            if last_err < best_err - 0.1:
                best_err    = last_err
                stall_count = 0
                torch.save({
                    "state":    model.state_dict(),
                    "epoch":    epoch,
                    "mean_err": last_err,
                }, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")

            if stall_count >= args.patience:
                logger.log(f"\nEarly stop — best_err={best_err:.2f}cm")
                break

    json.dump({"best_err": best_err}, open(out_dir / "config.json", "w"))
    logger.log(f"Done → {out_dir}")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",      default="world_model/data_fixeddt")
    p.add_argument("--out-dir",       default="world_model/results/spr_mdn_v31_bounce")
    p.add_argument("--t-start",       type=int,   default=5)
    p.add_argument("--t-curriculum",  type=int,   default=200)
    p.add_argument("--lr",            type=float, default=3e-3)
    p.add_argument("--lam-type",      type=float, default=0.1)
    p.add_argument("--max-epochs",    type=int,   default=800)
    p.add_argument("--patience",      type=int,   default=30)
    p.add_argument("--eval-every",    type=int,   default=10)
    p.add_argument("--batch-size",    type=int,   default=512)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
