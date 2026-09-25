"""
world_model/spr_mdn/train_v26_p1.py — Phase 1: Decoupled SSM warm-up

Architecture:
  - Transition + Decoder: train like SSM v18 (deterministic, MSE + Kendall type)
    → encoder is stop-gradded from this path (uses EMA encoder for z_0 seed)
  - Encoder: chases transition's latent chain via L2
    → transition/decoder are stop-gradded from this path

Why decoupled:
  In SSM/SPR-MDN, encoder gets gradient from reconstruction (through z_0).
  Here, transition learns clean dynamics independent of encoder quality,
  and encoder separately learns to be a codec for those dynamics.

Once Phase 1 stabilises, Phase 2 introduces MDN on top of the stable backbone.

Usage:
    python world_model/spr_mdn/train_v26_p1.py \
      --data-dir world_model/data_fixeddt \
      --out-dir  world_model/results/spr_mdn_v26_p1 \
      2>&1 | tee /tmp/v26_p1.log
"""

import copy, math, os, sys, json, argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.fixeddt_model import StateEncoder, LATENT_DIM
from world_model.ssm_model import (
    ResTransition, CueBallHead, TgtBallHead, TypeHead,
    N_COLL_TYPES, _focal_cross_entropy,
)
from world_model.spr_mdn.spr_dataset import (
    SPRDataset, SPREpisodeSubset, _CapDataset, make_balanced_val_eps,
)
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H
from log_utils import Logger

T_MAX = 60
T_MIN = 10
CLASS_WEIGHTS_DEFAULT = torch.tensor([1.0, 4.1, 4.1, 4.1, 4.1])


# ─────────────────────────────────────────────────────────────────────────────
class DecoupledSSM(nn.Module):
    """
    SSM v18 backbone with decoupled encoder training.

    Gradient paths:
      L_dyn (recon + type) → transition, decoder, type_head   [encoder detached]
      L_enc (L2)           → encoder only                     [chain detached]
    """

    def __init__(self, latent_dim: int = LATENT_DIM, ema_tau: float = 0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_tau    = ema_tau

        self.encoder     = StateEncoder(latent_dim)
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)

        self.transition = ResTransition(latent_dim, use_ar_state=False)
        self.cue_head   = CueBallHead(latent_dim)
        self.tgt_head   = TgtBallHead(latent_dim)
        self.type_head  = TypeHead(latent_dim)

    @torch.no_grad()
    def update_ema(self) -> None:
        tau = self.ema_tau
        for p, p_ema in zip(self.encoder.parameters(),
                            self.ema_encoder.parameters()):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cue_head(z), self.tgt_head(z)], dim=-1)

    def forward(
        self,
        s_0:   torch.Tensor,    # (B, 14)
        seq_s: torch.Tensor,    # (B, T+1, 14)
        T:     int,
    ):
        """
        Returns:
          s_hat       (B, T+1, 14)  — decoded states from transition chain
          type_logit  (B, T, 5)     — type predictions from z_1..z_T
          enc_preds   list[T] of (B, D)   — encoder(s_t) for t=1..T
          z_targets   list[T] of (B, D)   — sg(z_chain[t]) for t=1..T

        Gradient flow (like SSM v18 + L_enc):
          L_dyn: encoder(s_0) → transition chain → decoder/type_head
                 gradient flows through entire chain back to encoder (SSM v18 identical)
          L_enc: encoder(s_t) for t=1..T → sg(z_chain[t])
                 encoder also learns to map every s_t to the chain's z_t
        """
        # ── Dynamics path: z_0 = encoder(s_0) WITH gradient (SSM v18 style) ──
        z = self.encoder(s_0)     # (B, D) — gradient flows to encoder
        z_chain = [z]
        for _ in range(T):
            z = self.transition(z)
            z_chain.append(z)

        s_hat      = torch.stack([self._decode(z) for z in z_chain], dim=1)  # (B, T+1, 14)
        type_logit = torch.stack([self.type_head(z) for z in z_chain[1:]], dim=1)  # (B, T, 5)

        # ── Encoder path: encoder(s_t) → sg(z_chain[t]) ──────────────────────
        z_targets = [z.detach() for z in z_chain]
        enc_preds = [self.encoder(seq_s[:, t]) for t in range(T + 1)]

        return s_hat, type_logit, enc_preds, z_targets

    @torch.no_grad()
    def rollout_eval(
        self,
        s_0: torch.Tensor,      # (B, 14)
        T:   int,
    ):
        """Pure inference: z_0 from online encoder, transition chain."""
        z = self.encoder(s_0)
        s_hat_list = [self._decode(z)]
        type_list  = []
        for _ in range(T):
            type_list.append(self.type_head(z))
            z = self.transition(z)
            s_hat_list.append(self._decode(z))
        s_hat      = torch.stack(s_hat_list, dim=1)   # (B, T+1, 14)
        type_logit = torch.stack(type_list,  dim=1)   # (B, T,   5)
        return s_hat, type_logit


# ─────────────────────────────────────────────────────────────────────────────
def _eval_batched(model: DecoupledSSM, episodes: list, device: str,
                  rollout_steps: int = 60) -> dict:
    model.eval()
    checkpoint_steps = {
        "0.5s": int(0.5 / DT),
        "1.0s": int(1.0 / DT),
        "2.0s": int(2.0 / DT),
        "3.0s": int(3.0 / DT),
    }
    valid = [(ep_s, ep_f, ep_t)
             for ep_s, ep_f, ep_t, _, _a in episodes
             if len(ep_s) >= rollout_steps + 1]

    errs_all, errs_bb, errs_no = [], [], []
    cp_errors = {k: [] for k in checkpoint_steps}
    coll_gt_sum = coll_tp_sum = coll_pred_sum = 0

    with torch.no_grad():
        s0    = torch.from_numpy(np.stack([ep[0][0] for ep in valid])).float().to(device)
        s_hat, type_logit = model.rollout_eval(s0, rollout_steps)
        pr = s_hat.cpu().numpy()
        pc = type_logit.argmax(-1).cpu().numpy() != 0

    for b_idx, (ep_s, ep_f, ep_t) in enumerate(valid):
        gt      = ep_s[1:rollout_steps + 1]
        p       = pr[b_idx, 1:rollout_steps + 1]
        cue_err = np.sqrt(((p[:, 0] - gt[:, 0]) * TABLE_W) ** 2 +
                          ((p[:, 1] - gt[:, 1]) * TABLE_H) ** 2)
        tgt_err = np.sqrt(((p[:, 7] - gt[:, 7]) * TABLE_W) ** 2 +
                          ((p[:, 8] - gt[:, 8]) * TABLE_H) ** 2)
        step_err = (cue_err + tgt_err) / 2 * 100
        me       = step_err.mean()

        has_bb = bool(np.any(ep_f & (ep_t == 1)))
        errs_all.append(me)
        (errs_bb if has_bb else errs_no).append(me)
        for label, t in checkpoint_steps.items():
            cp_errors[label].append(step_err[t - 1])

        gt_flags       = ep_f[:rollout_steps]
        pred_flags     = pc[b_idx, :len(gt_flags)]
        coll_gt_sum   += int(gt_flags.sum())
        coll_tp_sum   += int((gt_flags & pred_flags).sum())
        coll_pred_sum += int(pred_flags.sum())

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    recall    = coll_tp_sum / coll_gt_sum   if coll_gt_sum   > 0 else 0.0
    precision = coll_tp_sum / coll_pred_sum if coll_pred_sum > 0 else 0.0

    result = {
        "mean_err":        _m(errs_all),
        "mean_err_has_bb": _m(errs_bb),
        "mean_err_no_bb":  _m(errs_no),
        "coll_recall":     recall,
        "coll_precision":  precision,
    }
    for k, v in cp_errors.items():
        result[k] = _m(v)
    return result


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(f"Device: {device}")
    logger.log(
        f"DecoupledSSM Phase 1  latent={LATENT_DIM}"
        f"  ema_tau={args.ema_tau}  lam_enc={args.lam_enc}"
        f"  max_epochs={args.max_epochs}  patience={args.patience}"
        f"  eval_every={args.eval_every}"
    )
    logger.log("Gradient routing: L_dyn→[transition,decoder,type_head]  L_enc→[encoder]")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = SPRDataset(args.data_dir, logger=logger)

    rng_split = np.random.default_rng(0)
    perm  = rng_split.permutation(len(dataset.episodes))
    n_val = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all   = [dataset.episodes[i] for i in perm[:n_val]]
    train_eps_all = [dataset.episodes[i] for i in perm[n_val:]]

    balanced_val_eps = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    n_bb  = sum(1 for ep in balanced_val_eps if np.any(ep[1] & (ep[2] == 1)))
    n_nbb = len(balanced_val_eps) - n_bb
    logger.log(f"Balanced val: {n_bb} has_bb + {n_nbb} no_bb = {len(balanced_val_eps)} total")

    train_eps = [ep for ep in train_eps_all if len(ep[0]) >= T_MAX + 1]
    val_eps   = [ep for ep in balanced_val_eps if len(ep[0]) >= T_MAX + 1]
    logger.log(f"Train: {len(train_eps):,} eps  Val: {len(val_eps):,} eps")

    MAX_EPOCH_ITEMS = 60_000
    train_ds    = _CapDataset(SPREpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True),
                              MAX_EPOCH_ITEMS)
    val_ds      = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DecoupledSSM(latent_dim=LATENT_DIM, ema_tau=args.ema_tau).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model params (trainable): {n_params:,}")

    # Kendall uncertainty: [log_σ_recon, log_σ_type] — like SSM v18
    log_sigma = nn.Parameter(torch.zeros(2, device=device))

    w_cls = CLASS_WEIGHTS_DEFAULT.clone()
    w_cls[4] = args.pocket_weight
    CLASS_WEIGHTS = w_cls.to(device)

    # Single optimizer for all trainable params (encoder, transition, decoder, log_sigma)
    trainable = list(model.parameters()) + [log_sigma]
    opt   = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_epochs, eta_min=args.lr * 0.01,
    )

    rng_T = np.random.default_rng(42)
    best_mean_err     = float("inf")
    stall_count       = 0
    rerr = {
        "mean_err": float("nan"), "mean_err_has_bb": float("nan"),
        "mean_err_no_bb": float("nan"), "coll_recall": 0.0, "coll_precision": 0.0,
        "0.5s": float("nan"), "1.0s": float("nan"),
        "2.0s": float("nan"), "3.0s": float("nan"),
    }

    for epoch in range(1, args.max_epochs + 1):
        cur_T = int(rng_T.integers(T_MIN, T_MAX + 1))

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_losses, enc_losses, dyn_losses = [], [], []

        for seq_s, seq_f, seq_t, seq_a in train_loader:
            seq_s_t = seq_s[:, :cur_T + 1].to(device)   # (B, T+1, 14)
            seq_t_t = seq_t[:, :cur_T].to(device)        # (B, T)

            s_hat, type_logit, enc_preds, z_targets = \
                model(seq_s_t[:, 0], seq_s_t, cur_T)

            # ── L_dyn: Kendall(recon, type) → transition + decoder ────────────
            # Skip h=0 for recon (like SSM v18): trivial self-reconstruction
            loss_cue  = F.mse_loss(s_hat[:, 1:, :7], seq_s_t[:, 1:, :7])
            loss_tgt  = F.mse_loss(s_hat[:, 1:, 7:], seq_s_t[:, 1:, 7:])
            loss_recon = loss_cue + loss_tgt

            B_T        = type_logit.shape[0] * type_logit.shape[1]
            loss_type  = _focal_cross_entropy(
                type_logit.reshape(B_T, N_COLL_TYPES),
                seq_t_t.reshape(B_T),
                CLASS_WEIGHTS, args.focal_gamma, 0.0,
            ) / math.log(N_COLL_TYPES)

            L_dyn = (torch.exp(-log_sigma[0]) * loss_recon + log_sigma[0] +
                     torch.exp(-log_sigma[1]) * loss_type  + log_sigma[1])

            # ── L_enc: encoder(s_t) → sg(z_chain[t]) for t=1..T ─────────────
            # Skip t=0: enc_preds[0] == z_targets[0] = encoder(s_0), trivially 0
            L_enc = sum(
                F.mse_loss(enc_preds[t], z_targets[t])
                for t in range(1, cur_T + 1)
            ) / cur_T

            loss = L_dyn + args.lam_enc * L_enc

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
            model.update_ema()

            tr_losses.append(loss.item())
            dyn_losses.append(L_dyn.item())
            enc_losses.append(L_enc.item())

        sched.step()

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_recon, val_type_acc = [], [], []
        type_correct = type_total = 0

        with torch.no_grad():
            for seq_s, seq_f, seq_t, seq_a in val_loader:
                seq_s = seq_s.to(device)
                seq_t = seq_t.to(device)
                val_T = seq_s.shape[1] - 1

                s_hat, type_logit, _, _ = model(seq_s[:, 0], seq_s, val_T)

                lc = F.mse_loss(s_hat[:, 1:, :7], seq_s[:, 1:, :7])
                lt = F.mse_loss(s_hat[:, 1:, 7:], seq_s[:, 1:, 7:])
                val_recon.append((lc + lt).item())

                pred = type_logit.argmax(-1)
                type_correct += (pred == seq_t).sum().item()
                type_total   += pred.numel()

        val_recon_mean = float(np.mean(val_recon))
        type_acc       = type_correct / type_total if type_total > 0 else 0.0

        # ── Rollout eval ──────────────────────────────────────────────────────
        do_eval = (epoch % args.eval_every == 0 or epoch == 1 or epoch == args.max_epochs)
        if do_eval:
            rerr = _eval_batched(model, balanced_val_eps, device, rollout_steps=60)

        mean_err    = rerr["mean_err"]
        coll_recall = rerr["coll_recall"]

        s = log_sigma.detach()
        w_recon = torch.exp(-s[0]).item()
        w_type  = torch.exp(-s[1]).item()

        cp_keys = [k for k in ["0.5s", "1.0s", "2.0s", "3.0s"]
                   if not np.isnan(rerr.get(k, float("nan")))]
        cp_str = " | ".join(f"{k}={rerr[k]:.1f}cm" for k in cp_keys) if do_eval else ""

        log_line = (
            f"Epoch {epoch:4d}  [T={cur_T:2d}]"
            f"  tr={np.mean(tr_losses):.4f}"
            f"  L_dyn={np.mean(dyn_losses):.4f}  L_enc={np.mean(enc_losses):.4f}"
            f"  val_recon={val_recon_mean:.4f}"
            f"  w_recon={w_recon:.3f}  w_type={w_type:.3f}"
            f"  type_acc={type_acc:.3f}  recall={coll_recall:.3f}"
            f"  err={mean_err:.1f}cm"
            f"  (bb={rerr['mean_err_has_bb']:.1f}/nbb={rerr['mean_err_no_bb']:.1f})"
        )
        if cp_str:
            log_line += f"  [{cp_str}]"
        logger.log(log_line)

        # ── Checkpoint + early stopping ───────────────────────────────────────
        if do_eval:
            if mean_err < best_mean_err - 1e-4:
                best_mean_err = mean_err
                stall_count   = 0
                ckpt = {
                    "state":      model.state_dict(),
                    "log_sigma":  log_sigma.detach().cpu(),
                    "epoch":      epoch,
                    "mean_err":   mean_err,
                    "latent_dim": LATENT_DIM,
                }
                torch.save(ckpt, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")

            if stall_count >= args.patience:
                logger.log(
                    f"\nEarly stop: {args.patience} consecutive evals without improvement."
                    f"  best_err={best_mean_err:.1f}cm @ loaded from best.pt"
                )
                break

    cfg = {
        "latent_dim":       LATENT_DIM,
        "ema_tau":          args.ema_tau,
        "lam_enc":          args.lam_enc,
        "max_epochs":       args.max_epochs,
        "patience":         args.patience,
        "eval_every":       args.eval_every,
        "best_mean_err_cm": best_mean_err,
    }
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    logger.log(f"Saved → {out_dir}  best_mean_err={best_mean_err:.1f}cm")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="world_model/data_fixeddt")
    p.add_argument("--out-dir",     default="world_model/results/spr_mdn_v26_p1")
    p.add_argument("--max-epochs",  type=int,   default=2000,
                   help="hard cap; early stopping fires first in practice")
    p.add_argument("--patience",    type=int,   default=10,
                   help="stop after this many evals with no improvement")
    p.add_argument("--eval-every",  type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--ema-tau",     type=float, default=0.99)
    p.add_argument("--lam-enc",     type=float, default=1.0,
                   help="weight on L_enc (encoder chasing transition chain)")
    p.add_argument("--tf-decay",    type=int,   default=200,
                   help="epochs over which p_tf decays 1.0 → tf_pmin")
    p.add_argument("--tf-pmin",     type=float, default=0.1,
                   help="minimum teacher-forcing probability after decay")
    p.add_argument("--pocket-weight", type=float, default=20.0)
    p.add_argument("--focal-gamma",   type=float, default=2.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
