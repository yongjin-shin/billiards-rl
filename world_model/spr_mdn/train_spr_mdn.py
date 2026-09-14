"""
world_model/spr_mdn/train_spr_mdn.py — SPR-MDN training script

v18 no-AR 구조 위에 SPR-MDN 적용. 아이디어 검증 목적 (scratch).

Usage:
    python world_model/spr_mdn/train_spr_mdn.py \
      --data-dir world_model/data_fixeddt \
      --out-dir  world_model/results/spr_mdn_v1 \
      --epochs 400
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.spr_mdn_model import (
    SPRMDNModel, spr_rollout_loss, LATENT_DIM, N_COMPONENTS,
)
from world_model.train_ssm import (
    SSMDataset, EpisodeSubset, _CapDataset, make_balanced_val_eps,
)
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H
from log_utils import Logger

T_MAX = 60
T_MIN = 10
CLASS_WEIGHTS_DEFAULT = torch.tensor([1.0, 4.1, 4.1, 4.1, 4.1])


def _eval_batched(model: SPRMDNModel, episodes: list, device: str,
                  rollout_steps: int = 60) -> dict:
    model.eval()
    checkpoint_steps = {
        "0.5s": int(0.5 / DT),
        "1.0s": int(1.0 / DT),
        "2.0s": int(2.0 / DT),
        "3.0s": int(3.0 / DT),
    }
    valid = [(ep_s, ep_f, ep_t) for ep_s, ep_f, ep_t, _ in episodes
             if len(ep_s) >= rollout_steps + 1]

    errs_all, errs_bb, errs_no = [], [], []
    cp_errors = {k: [] for k in checkpoint_steps}
    coll_gt_sum = coll_tp_sum = coll_pred_sum = 0

    with torch.no_grad():
        s0 = torch.from_numpy(
            np.stack([ep[0][0] for ep in valid])
        ).float().to(device)

        s_hat, type_logit = model.rollout_eval(s0, rollout_steps)
        pr = s_hat.cpu().numpy()
        pc = type_logit.argmax(-1).cpu().numpy() != 0

    for b, (ep_s, ep_f, ep_t) in enumerate(valid):
        gt      = ep_s[1:rollout_steps + 1]
        p       = pr[b, 1:rollout_steps + 1]
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

        gt_flags      = ep_f[:rollout_steps]
        pred_flags    = pc[b, :len(gt_flags)]
        coll_gt_sum  += int(gt_flags.sum())
        coll_tp_sum  += int((gt_flags & pred_flags).sum())
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


def train(args):
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(f"Device: {device}")
    logger.log(
        f"Mode: SPR-MDN  K={args.n_components}  EMA_tau={args.ema_tau}"
        f"  T~Uniform({T_MIN},{T_MAX})"
        f"  pocket_w={args.pocket_weight}  focal_gamma={args.focal_gamma}"
        f"  label_smoothing={args.label_smoothing}"
    )

    # ── Data ─────────────────────────────────────────────────────────────
    dataset = SSMDataset(args.data_dir, logger=logger)

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
    train_ds = _CapDataset(EpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True),
                           MAX_EPOCH_ITEMS)
    val_ds   = EpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SPRMDNModel(
        latent_dim=LATENT_DIM,
        n_components=args.n_components,
        ema_tau=args.ema_tau,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Parameters (trainable): {n_params:,}")

    w = CLASS_WEIGHTS_DEFAULT.clone()
    w[4] = args.pocket_weight
    CLASS_WEIGHTS = w.to(device)
    logger.log(f"Class weights: {w.tolist()}")

    # log_sigma: [σ_NLL, σ_recon] — temperature taming
    log_sigma = nn.Parameter(torch.zeros(2, device=device))

    # ema_encoder is excluded from optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.Adam(trainable + [log_sigma], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    rng_T         = np.random.default_rng(42)
    best_mean_err = float("inf")
    rerr = {
        "mean_err": float("nan"), "mean_err_has_bb": float("nan"),
        "mean_err_no_bb": float("nan"), "coll_recall": 0.0,
        "coll_precision": 0.0, "0.5s": float("nan"),
        "1.0s": float("nan"), "2.0s": float("nan"), "3.0s": float("nan"),
    }

    for epoch in range(1, args.epochs + 1):
        cur_T = int(rng_T.integers(T_MIN, T_MAX + 1))

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_losses = []
        for seq_s, seq_f, seq_t in train_loader:
            seq_s_t = seq_s[:, :cur_T + 1].to(device)   # (B, T+1, 14)
            seq_t_t = seq_t[:, :cur_T].to(device)        # (B, T)

            s_hat, type_logit, _, pi_list, mu_list, sigma_list, z_bar_list = \
                model(seq_s_t[:, 0], seq_s_t, cur_T)

            loss, _ = spr_rollout_loss(
                pi_list, mu_list, sigma_list, z_bar_list,
                s_hat, seq_s_t, type_logit, seq_t_t,
                class_weights=CLASS_WEIGHTS,
                log_sigma=log_sigma,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable + [log_sigma], 1.0)
            opt.step()
            model.update_ema()
            tr_losses.append(loss.item())

        sched.step()

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_details = [], []
        type_correct = type_total = 0

        with torch.no_grad():
            for seq_s, seq_f, seq_t in val_loader:
                seq_s = seq_s.to(device)
                seq_t = seq_t.to(device)
                val_T = seq_s.shape[1] - 1

                s_hat, type_logit, _, pi_list, mu_list, sigma_list, z_bar_list = \
                    model(seq_s[:, 0], seq_s, val_T)

                loss, detail = spr_rollout_loss(
                    pi_list, mu_list, sigma_list, z_bar_list,
                    s_hat, seq_s, type_logit, seq_t,
                    class_weights=CLASS_WEIGHTS,
                    log_sigma=log_sigma,
                    focal_gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )
                val_losses.append(loss.item())
                val_details.append(detail)

                pred = type_logit.argmax(-1)
                type_correct += (pred == seq_t).sum().item()
                type_total   += pred.numel()

        val_loss = float(np.mean(val_losses))
        d        = {k: np.mean([x[k] for x in val_details]) for k in val_details[0]}
        type_acc = type_correct / type_total if type_total > 0 else 0.0

        # ── Rollout eval (every 10 epochs) ─────────────────────────────────
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            rerr = _eval_batched(model, balanced_val_eps, device, rollout_steps=60)

        mean_err    = rerr["mean_err"]
        coll_recall = rerr["coll_recall"]
        coll_prec   = rerr["coll_precision"]

        s  = log_sigma.clamp(-6, 6).detach()
        kw = torch.exp(-s)
        kw_str = f"  kw=[{kw[0].item():.2f},{kw[1].item():.2f}]"

        cp_keys = [k for k in ["0.5s", "1.0s", "2.0s", "3.0s"]
                   if not np.isnan(rerr.get(k, float("nan")))]
        cp_str  = " | ".join(f"{k}={rerr[k]:.1f}cm" for k in cp_keys) \
                  if (epoch % 10 == 0 or epoch == args.epochs) else ""

        log_line = (
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  [T={cur_T:2d}]"
            f"  tr={np.mean(tr_losses):.4f}  val={val_loss:.4f}"
            f"  | L_nll={d['loss_nll']:.4f}  L_recon={d['loss_recon']:.4f}"
            f"  L_type={d['loss_type']:.4f}"
            f"  type_acc={type_acc:.3f}"
            f"  recall={coll_recall:.3f}  prec={coll_prec:.3f}"
            f"  err={mean_err:.1f}cm"
            f"  (bb={rerr['mean_err_has_bb']:.1f}/nbb={rerr['mean_err_no_bb']:.1f})"
            f"{kw_str}"
        )
        if cp_str:
            log_line += f"  [{cp_str}]"
        logger.log(log_line)

        if (epoch % 10 == 0 or epoch == 1 or epoch == args.epochs) \
                and mean_err < best_mean_err:
            best_mean_err = mean_err
            torch.save({
                "state":        model.state_dict(),
                "log_sigma":    log_sigma.detach().cpu(),
                "epoch":        epoch,
                "mean_err":     mean_err,
                "val_loss":     val_loss,
                "latent_dim":   LATENT_DIM,
                "n_components": args.n_components,
                "ema_tau":      args.ema_tau,
            }, out_dir / "best.pt")

    torch.save({"state": model.state_dict(), "epoch": args.epochs,
                "latent_dim": LATENT_DIM, "n_components": args.n_components},
               out_dir / "final.pt")

    cfg = {
        "latent_dim":       LATENT_DIM,
        "n_components":     args.n_components,
        "ema_tau":          args.ema_tau,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "pocket_weight":    args.pocket_weight,
        "focal_gamma":      args.focal_gamma,
        "label_smoothing":  args.label_smoothing,
        "best_mean_err_cm": best_mean_err,
    }
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    logger.log(f"\nSaved → {out_dir}  best_mean_err={best_mean_err:.1f}cm")
    logger.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",        default="world_model/data_fixeddt")
    p.add_argument("--out-dir",         default="world_model/results/spr_mdn_v1")
    p.add_argument("--epochs",          type=int,   default=400)
    p.add_argument("--batch-size",      type=int,   default=512)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--n-components",    type=int,   default=N_COMPONENTS)
    p.add_argument("--ema-tau",         type=float, default=0.99)
    p.add_argument("--pocket-weight",   type=float, default=20.0)
    p.add_argument("--focal-gamma",     type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
