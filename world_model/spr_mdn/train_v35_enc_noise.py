"""
world_model/spr_mdn/train_v35_enc_noise.py  —  V33 + Encoder Noise Injection

Problem: at segment boundaries during chaining, encoder receives the previous
  segment's predicted endpoint (not GT). The encoder was only trained on clean
  GT s_0 → covariate shift → error accumulates across segments.

Fix: inject noise into the encoder input s_0 to simulate the distribution of
  predicted states the encoder will see at test time.
  - Noise is per-dim scaled: pos (1.0x), vel (0.3x), angular (0.1x)
  - Annealed 0 → sigma_max over enc_warmup epochs
  - Warm-start from v33 best.pt to preserve per-segment accuracy

Key difference from v33_ss (abandoned at ep16):
  v33_ss: immediately starts annealing from epoch 1 with a single sigma
  v35:    first phase (no noise) lets v33 weights settle, then gradual ramp
  v35:    per-dim scaling is more realistic (velocity much smaller than pos error)

Usage:
    python world_model/spr_mdn/train_v35_enc_noise.py \
      --ckpt world_model/results/spr_mdn_v33_segment/best.pt \
      --out-dir world_model/results/spr_mdn_v35_enc_noise \
      2>&1 | tee /tmp/v35_enc.log
"""

import os, sys, argparse, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.fixeddt_model import LATENT_DIM, POCKET_XY_NORM, _mlp
from world_model.spr_mdn.spr_dataset import (
    SPRDataset, SPREpisodeSubset, SegmentDataset, collate_seg, make_balanced_val_eps,
)
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.generate_data_fixeddt import DT
from log_utils import Logger

S_DIM   = 14
N_TYPES = 5   # cue_strike, bb, lin_cush, circ_cush, pocket (matches v33 checkpoint)


class TypeHead(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, n_types: int = N_TYPES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, n_types),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# Per-dim noise scale factors (relative to sigma_max which targets position dims)
# s layout: [cue_x, cue_y, cue_vx, cue_vy, cue_wx, cue_wy, cue_wz,
#            tgt_x, tgt_y, tgt_vx, tgt_vy, tgt_wx, tgt_wy, tgt_wz]
_NOISE_SCALE = torch.tensor([
    1.0, 1.0,           # cue pos
    0.3, 0.3,           # cue vel
    0.1, 0.1, 0.1,      # cue angular vel
    1.0, 1.0,           # tgt pos
    0.3, 0.3,           # tgt vel
    0.1, 0.1, 0.1,      # tgt angular vel
], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
class CondEncoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, n_types: int = N_TYPES):
        super().__init__()
        self.net = _mlp(S_DIM + 12 + n_types, (128, 256), latent_dim)
        self.register_buffer("pocket_xy", POCKET_XY_NORM)
        self.n_types = n_types

    def _pocket_dists(self, xy: torch.Tensor) -> torch.Tensor:
        diff = xy.unsqueeze(1) - self.pocket_xy.unsqueeze(0)
        return diff.norm(dim=-1)

    def forward(self, s: torch.Tensor, seg_type: torch.Tensor) -> torch.Tensor:
        cue_d = self._pocket_dists(s[:, 0:2])
        tgt_d = self._pocket_dists(s[:, 7:9])
        t_oh  = F.one_hot(seg_type.clamp(0, self.n_types-1), self.n_types).float()
        return self.net(torch.cat([s, cue_d, tgt_d, t_oh], dim=-1))


class ObsTransition(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, s_dim: int = S_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + s_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),                nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.norm(z + self.net(torch.cat([z, s], dim=-1)))


class MuHead(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, s_dim: int = S_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(),
                                  nn.Linear(128, s_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class V35Model(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = CondEncoder(latent_dim, N_TYPES)
        self.transition = ObsTransition(latent_dim, S_DIM)
        self.mu_head    = MuHead(latent_dim, S_DIM)
        self.type_head  = TypeHead(latent_dim)

    @torch.no_grad()
    def rollout_chain(self, s_0: torch.Tensor, T: int):
        B = s_0.shape[0]
        seg_type = torch.zeros(B, dtype=torch.long, device=s_0.device)
        z     = self.encoder(s_0, seg_type)
        s_cur = s_0
        s_list, t_list = [], []

        for _ in range(T):
            z       = self.transition(z, s_cur)
            s_hat   = self.mu_head(z)
            t_logit = self.type_head(z)
            type_pred = t_logit.argmax(-1)

            s_list.append(s_hat)
            t_list.append(t_logit)
            s_cur = s_hat

            coll_mask = type_pred != 0
            if coll_mask.any():
                z_new = self.encoder(s_cur, type_pred)
                z = torch.where(coll_mask.unsqueeze(-1), z_new, z)

        return torch.stack(s_list, 1), torch.stack(t_list, 1)


# ─────────────────────────────────────────────────────────────────────────────
def _eval_err(model: V35Model, episodes: list, device: str,
              min_seg_len: int = 2) -> dict:
    TYPE_NAMES = ["cue_strike", "bb", "lin_cush", "circ_cush", "pocket", "slide_roll", "roll_stop"]
    model.eval()
    val_ds = SegmentDataset(episodes, min_len=min_seg_len, augment=False)

    errs_all: list = []
    errs_by_type: dict = {i: [] for i in range(len(TYPE_NAMES))}

    with torch.no_grad():
        for seg_s, seg_t, seg_type_start, seg_len in val_ds:
            L = int(seg_len)
            seg_s_d = seg_s.unsqueeze(0).to(device)
            st      = seg_type_start.unsqueeze(0).to(device)

            z     = model.encoder(seg_s_d[:, 0], st)
            s_cur = seg_s_d[:, 0]
            preds = []
            for t in range(L):
                z     = model.transition(z, s_cur)
                s_hat = model.mu_head(z)
                preds.append(s_hat[0].cpu().numpy())
                s_cur = s_hat

            pr = np.stack(preds)
            gt = seg_s[1:L + 1].numpy()
            ce = np.sqrt(((pr[:,0]-gt[:,0])*TABLE_W)**2 + ((pr[:,1]-gt[:,1])*TABLE_H)**2)
            te = np.sqrt(((pr[:,7]-gt[:,7])*TABLE_W)**2 + ((pr[:,8]-gt[:,8])*TABLE_H)**2)
            se = (ce + te) / 2 * 100
            errs_all.append(se.mean())
            errs_by_type[int(seg_type_start)].append(se.mean())

    def _m(l): return float(np.mean(l)) if l else float("nan")
    result = {"mean_err": _m(errs_all),
              "bb":  _m(errs_by_type[1]),
              "nbb": _m([e for i, es in errs_by_type.items() if i != 1 for e in es])}
    for i, name in enumerate(TYPE_NAMES):
        result[name] = _m(errs_by_type[i])
    return result


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)

    noise_scale = _NOISE_SCALE.to(device)   # (14,)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = SPRDataset(args.data_dir, logger=logger)
    rng     = np.random.default_rng(0)
    perm    = rng.permutation(len(dataset.episodes))
    n_val   = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all   = [dataset.episodes[i] for i in perm[:n_val]]
    train_eps_all = [dataset.episodes[i] for i in perm[n_val:]]

    balanced_val = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)

    train_ds = SegmentDataset(train_eps_all, min_len=args.min_seg_len,
                              max_seg_len=args.max_seg_len,
                              max_per_type=args.max_per_type,
                              augment=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_seg)

    logger.log(
        f"V35 Encoder Noise  segs={len(train_ds):,}  min_seg={args.min_seg_len}"
        f"  lr={args.lr}  patience={args.patience}  lam_type={args.lam_type}"
        f"  enc_noise: 0→{args.enc_sigma:.4f} over {args.enc_warmup} epochs"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = V35Model().to(device)

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        state = ck["state"] if "state" in ck else ck
        missing, unexpected = model.load_state_dict(state, strict=False)
        init_err = ck.get("mean_err", float("nan"))
        logger.log(f"Loaded {args.ckpt}  epoch={ck.get('epoch','?')}  err={init_err:.2f}cm"
                   f"  missing={len(missing)}  unexpected={len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Params: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_epochs, eta_min=args.lr * 0.01)

    best_err    = float("inf")
    stall_count = 0
    last_err    = float("nan")
    last_rerr   = {}

    for epoch in range(1, args.max_epochs + 1):
        cur_lr = opt.param_groups[0]["lr"]

        # Noise sigma annealing: 0 → enc_sigma over enc_warmup epochs
        if args.enc_warmup > 0:
            cur_sigma = args.enc_sigma * min(1.0, (epoch - 1) / args.enc_warmup)
        else:
            cur_sigma = args.enc_sigma

        model.train()
        mse_losses, type_losses = [], []

        for seg_s, seg_t, seg_type_start, seg_lens, mask in train_loader:
            seg_s          = seg_s.to(device)
            seg_t          = seg_t.to(device)
            seg_type_start = seg_type_start.to(device)
            mask           = mask.to(device)

            max_L  = seg_s.shape[1] - 1
            n_real = mask.float().sum().clamp(min=1)

            # Encoder noise: simulate predicted-state distribution at segment boundaries
            s0 = seg_s[:, 0]
            if cur_sigma > 0.0:
                noise = torch.randn_like(s0) * cur_sigma * noise_scale
                s0 = s0 + noise

            z     = model.encoder(s0, seg_type_start)
            s_cur = seg_s[:, 0]   # transition input: GT (free-running starts at step 1)

            L_mse  = torch.tensor(0.0, device=device)
            L_type = torch.tensor(0.0, device=device)

            for t in range(max_L):
                z     = model.transition(z, s_cur)
                s_hat = model.mu_head(z)
                t_logit = model.type_head(z)

                m = mask[:, t].float()
                L_mse  = L_mse  + (F.mse_loss(s_hat, seg_s[:, t+1], reduction='none')
                                    .mean(-1) * m).sum()
                L_type = L_type + (F.cross_entropy(t_logit, seg_t[:, t],
                                                    reduction='none') * m).sum()
                s_cur = s_hat   # free-running (same as v33)

            loss = (L_mse + args.lam_type * L_type) / n_real
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            mse_losses.append((L_mse / n_real).item())
            type_losses.append((L_type / n_real).item())

        sched.step()

        do_eval = (epoch % args.eval_every == 0 or epoch == 1)
        if do_eval:
            last_rerr = _eval_err(model, balanced_val, device,
                                  min_seg_len=args.min_seg_len)
            last_err  = last_rerr["mean_err"]

        log_line = (
            f"Epoch {epoch:4d}  [lr={cur_lr:.2e}  σ={cur_sigma:.4f}]"
            f"  mse={np.mean(mse_losses):.5f}  type={np.mean(type_losses):.4f}"
            f"  err={last_err:.2f}cm"
            f"  (bb={last_rerr.get('bb', float('nan')):.2f}"
            f"/nbb={last_rerr.get('nbb', float('nan')):.2f})"
        )
        if do_eval:
            type_str = "  [" + " | ".join(
                f"{k}={last_rerr[k]:.2f}"
                for k in ["cue_strike", "bb", "lin_cush", "circ_cush", "pocket"]
            ) + "]"
            log_line += type_str
        logger.log(log_line)

        if do_eval:
            if last_err < best_err - 0.05:
                best_err    = last_err
                stall_count = 0
                torch.save({"state": model.state_dict(), "epoch": epoch,
                            "mean_err": last_err}, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")
            if stall_count >= args.patience:
                logger.log(f"\nEarly stop — best_err={best_err:.3f}cm")
                break

    json.dump({"best_err": best_err}, open(out_dir / "config.json", "w"))
    logger.log(f"Done → {out_dir}")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",     default="world_model/data_fixeddt")
    p.add_argument("--out-dir",      default="world_model/results/spr_mdn_v35_enc_noise")
    p.add_argument("--ckpt",         default="world_model/results/spr_mdn_v33_segment/best.pt",
                   help="Warm-start checkpoint (v33 best.pt recommended)")
    p.add_argument("--min-seg-len",  type=int,   default=2)
    p.add_argument("--max-seg-len",  type=int,   default=0)
    p.add_argument("--max-per-type", type=int,   default=0)
    p.add_argument("--lr",           type=float, default=3e-4,
                   help="Lower LR for fine-tuning from v33")
    p.add_argument("--lam-type",     type=float, default=0.3)
    p.add_argument("--max-epochs",   type=int,   default=500)
    p.add_argument("--patience",     type=int,   default=30)
    p.add_argument("--eval-every",   type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=512)
    # Encoder noise injection
    p.add_argument("--enc-sigma",    type=float, default=0.05,
                   help="Max noise sigma for position dims (0.05 ≈ 5cm/99cm)")
    p.add_argument("--enc-warmup",   type=int,   default=100,
                   help="Epochs to anneal encoder noise 0→enc-sigma")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
