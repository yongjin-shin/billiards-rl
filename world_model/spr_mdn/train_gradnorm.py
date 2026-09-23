"""
world_model/spr_mdn/train_gradnorm.py — L2 warm-up → NLL + GradNorm (v25)

Two-phase curriculum:
  Phase 1 (epoch ≤ phase1_epochs): L2 on mixture mean — encoder/decoder/transition warm-up
  Phase 2 (epoch  > phase1_epochs): NLL + GradNorm    — distribution head fine-tune

v25 config (v18a-equivalent structure + NLL+GradNorm):
    python world_model/spr_mdn/train_gradnorm.py \\
      --data-dir world_model/data_fixeddt \\
      --out-dir  world_model/results/spr_gradnorm_v25 \\
      --epochs 500 --phase1-epochs 200 \\
      --K 5 --no-ema --no-encoder-ln --full-bptt \\
      --gn-alpha 1.5 --tf-decay 200 --tf-pmin 0.1

v18a reference (L2 only, no GradNorm):
    python world_model/spr_mdn/train_spr_mdn.py \\
      --k1 --no-ema --no-encoder-ln --full-bptt --lam-l2 0.01 \\
      --tf-decay 200 --tf-pmin 0.1 --epochs 500
"""

import math
import os, sys, json, argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.spr_mdn_model import (
    SPRMDNModel, spr_rollout_loss,
    laplace_nll_mixture, N_COLL_TYPES, _focal_cross_entropy,
    LATENT_DIM, N_COMPONENTS, ACTION_DIM,
)
from world_model.spr_mdn.gradnorm import GradNormController
from world_model.spr_mdn.spr_dataset import (
    SPRDataset, SPREpisodeSubset, _CapDataset, make_balanced_val_eps,
)
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H
from log_utils import Logger

T_MAX = 60
T_MIN = 10
CLASS_WEIGHTS_DEFAULT = torch.tensor([1.0, 4.1, 4.1, 4.1, 4.1])


def _eval_batched(
    model: SPRMDNModel,
    episodes: list,
    device: str,
    rollout_steps: int = 60,
) -> dict:
    model.eval()
    checkpoint_steps = {
        "0.5s": int(0.5 / DT),
        "1.0s": int(1.0 / DT),
        "2.0s": int(2.0 / DT),
        "3.0s": int(3.0 / DT),
    }
    valid = [
        (ep_s, ep_f, ep_t, ep_a)
        for ep_s, ep_f, ep_t, _, ep_a in episodes
        if len(ep_s) >= rollout_steps + 1
    ]

    errs_all, errs_bb, errs_no = [], [], []
    cp_errors = {k: [] for k in checkpoint_steps}
    coll_gt_sum = coll_tp_sum = coll_pred_sum = 0

    with torch.no_grad():
        s0   = torch.from_numpy(np.stack([ep[0][0] for ep in valid])).float().to(device)
        acts = torch.from_numpy(np.stack([ep[3]    for ep in valid])).float().to(device)
        s_hat, type_logit = model.rollout_eval(s0, rollout_steps, action=acts)
        pr = s_hat.cpu().numpy()
        pc = type_logit.argmax(-1).cpu().numpy() != 0

    for b_idx, (ep_s, ep_f, ep_t, _) in enumerate(valid):
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


def train(args: argparse.Namespace) -> None:
    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available()         else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(f"Device: {device}")
    logger.log(
        f"Phase1 (L2 warm-up): {args.phase1_epochs} epochs"
        f"  Phase2 (NLL+GradNorm): {args.epochs - args.phase1_epochs} epochs"
    )
    logger.log(
        f"K={args.K}  no_ema={args.no_ema}  no_encoder_ln={args.no_encoder_ln}"
        f"  full_bptt={args.full_bptt}  gn_alpha={args.gn_alpha}  gn_lr={args.gn_lr}"
        f"  tf_decay={args.tf_decay}  tf_pmin={args.tf_pmin}  lam_recon={args.lam_recon}"
    )

    # ── Data ─────────────────────────────────────────────────────────────
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
    train_ds = _CapDataset(
        SPREpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True),
        MAX_EPOCH_ITEMS,
    )
    val_ds      = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SPRMDNModel(
        latent_dim=LATENT_DIM,
        n_components=args.K,
        action_dim=ACTION_DIM,
        ema_tau=args.ema_tau,
        use_ema=not args.no_ema,
        use_encoder_ln=not args.no_encoder_ln,
        full_bptt=args.full_bptt,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model params (trainable): {n_params:,}")

    # ── GradNorm controller (only active in Phase 2) ──────────────────────
    gn = GradNormController(task_names=["nll", "recon"], alpha=args.gn_alpha).to(device)
    gn_opt = torch.optim.Adam(gn.parameters(), lr=args.gn_lr)

    # ── Optimizer ─────────────────────────────────────────────────────────
    w_cls = CLASS_WEIGHTS_DEFAULT.clone()
    w_cls[4] = args.pocket_weight
    CLASS_WEIGHTS = w_cls.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    rng_T         = np.random.default_rng(42)
    best_mean_err = float("inf")
    rerr = {
        "mean_err": float("nan"), "mean_err_has_bb": float("nan"),
        "mean_err_no_bb": float("nan"), "coll_recall": 0.0,
        "coll_precision": 0.0,
        "0.5s": float("nan"), "1.0s": float("nan"),
        "2.0s": float("nan"), "3.0s": float("nan"),
    }

    for epoch in range(1, args.epochs + 1):
        cur_T  = int(rng_T.integers(T_MIN, T_MAX + 1))
        p_tf   = max(args.tf_pmin, 1.0 - (epoch - 1) / args.tf_decay)
        phase1 = (epoch <= args.phase1_epochs)

        if epoch == args.phase1_epochs + 1 and args.phase1_epochs > 0:
            # Phase transition: reset GradNorm L0 with current loss scale
            gn.L0_set[0] = False
            logger.log(f"[Phase 2 START] NLL+GradNorm from epoch {epoch}")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_losses   = []
        gn_diag_acc = {}

        for seq_s, seq_f, seq_t, seq_a in train_loader:
            seq_s_t = seq_s[:, :cur_T + 1].to(device)
            seq_t_t = seq_t[:, :cur_T].to(device)
            seq_a_t = seq_a.to(device)

            s_hat, type_logit, _, pi_list, mu_list, b_list, z_bar_list = \
                model(seq_s_t[:, 0], seq_s_t, seq_a_t, cur_T, p_tf=p_tf)

            T = cur_T
            B_T = type_logit.shape[0] * type_logit.shape[1]
            L_type = _focal_cross_entropy(
                type_logit.reshape(B_T, N_COLL_TYPES),
                seq_t_t.reshape(B_T),
                CLASS_WEIGHTS, args.focal_gamma, 0.0,
            ) / math.log(N_COLL_TYPES)

            L_recon = (
                F.mse_loss(s_hat[:, :, :7], seq_s_t[:, :, :7]) +
                F.mse_loss(s_hat[:, :, 7:], seq_s_t[:, :, 7:])
            )

            if phase1:
                # ── Phase 1: L2 on mixture mean ───────────────────────────
                L_pred = sum(
                    F.mse_loss(
                        (pi_list[h].unsqueeze(-1) * mu_list[h]).sum(1),
                        z_bar_list[h],
                    )
                    for h in range(T)
                ) / T
                loss = L_pred + args.lam_recon * L_recon + L_type
                gn_diag_acc["L_pred"] = gn_diag_acc.get("L_pred", 0.0) + L_pred.item()

            else:
                # ── Phase 2: NLL + GradNorm ───────────────────────────────
                L_nll = sum(
                    laplace_nll_mixture(pi_list[h], mu_list[h], b_list[h], z_bar_list[h])
                    for h in range(T)
                ) / T

                enc_params = list(model.encoder.parameters())
                gn_diag = gn.step(
                    {"nll": L_nll, "recon": L_recon},
                    enc_params,
                    gn_opt,
                )
                for k, v in gn_diag.items():
                    gn_diag_acc[k] = gn_diag_acc.get(k, 0.0) + v

                w    = gn.weights.detach()
                loss = w[0] * L_nll + w[1] * args.lam_recon * L_recon + L_type

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            model.update_ema()
            tr_losses.append(loss.item())

        sched.step()
        n_batches = len(tr_losses)

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_details = [], []
        type_correct = type_total = 0

        with torch.no_grad():
            for seq_s, seq_f, seq_t, seq_a in val_loader:
                seq_s = seq_s.to(device)
                seq_t = seq_t.to(device)
                seq_a = seq_a.to(device)
                val_T = seq_s.shape[1] - 1

                s_hat, type_logit, _, pi_list, mu_list, b_list, z_bar_list = \
                    model(seq_s[:, 0], seq_s, seq_a, val_T)

                loss, detail = spr_rollout_loss(
                    pi_list, mu_list, b_list, z_bar_list,
                    s_hat, seq_s, type_logit, seq_t,
                    class_weights=CLASS_WEIGHTS,
                    focal_gamma=args.focal_gamma,
                    lam_recon=args.lam_recon,
                )
                val_losses.append(loss.item())
                val_details.append(detail)

                pred = type_logit.argmax(-1)
                type_correct += (pred == seq_t).sum().item()
                type_total   += pred.numel()

        val_loss = float(np.mean(val_losses))
        d        = {k: np.mean([x[k] for x in val_details]) for k in val_details[0]}
        type_acc = type_correct / type_total if type_total > 0 else 0.0

        # ── Rollout eval ───────────────────────────────────────────────────
        do_eval = (epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs
                   or epoch == args.phase1_epochs)
        if do_eval:
            rerr = _eval_batched(model, balanced_val_eps, device, rollout_steps=60)

        mean_err    = rerr["mean_err"]
        coll_recall = rerr["coll_recall"]
        coll_prec   = rerr["coll_precision"]

        cp_keys = [k for k in ["0.5s", "1.0s", "2.0s", "3.0s"]
                   if not np.isnan(rerr.get(k, float("nan")))]
        cp_str = " | ".join(f"{k}={rerr[k]:.1f}cm" for k in cp_keys) if do_eval else ""

        if phase1:
            phase_str = f"[P1:L2]"
            extra_str = f"  L_pred={gn_diag_acc.get('L_pred', 0.0) / n_batches:.4f}"
        else:
            phase_str = f"[P2:NLL+GN]"
            gn_avg    = {k: v / n_batches for k, v in gn_diag_acc.items()}
            extra_str = (
                f"  L_nll={d['loss_nll']:.4f}"
                f"  w_nll={gn_avg.get('w_nll', float('nan')):.3f}"
                f"  w_recon={gn_avg.get('w_recon', float('nan')):.3f}"
                f"  G_ratio={gn_avg.get('G_nll', 1.0) / max(gn_avg.get('G_recon', 1e-8), 1e-8):.1f}x"
            )

        log_line = (
            f"Epoch {epoch:3d}/{args.epochs}  {phase_str}"
            f"  [T={cur_T:2d}  ptf={p_tf:.2f}]"
            f"  tr={np.mean(tr_losses):.4f}  val={val_loss:.4f}"
            f"{extra_str}"
            f"  L_recon={d['loss_recon']:.4f}  L_type={d['loss_type']:.4f}"
            f"  type_acc={type_acc:.3f}"
            f"  recall={coll_recall:.3f}  err={mean_err:.1f}cm"
            f"  (bb={rerr['mean_err_has_bb']:.1f}/nbb={rerr['mean_err_no_bb']:.1f})"
        )
        if cp_str:
            log_line += f"  [{cp_str}]"
        logger.log(log_line)

        if do_eval and mean_err < best_mean_err:
            best_mean_err = mean_err
            ckpt = {
                "state":          model.state_dict(),
                "gn_state":       gn.state_dict(),
                "epoch":          epoch,
                "mean_err":       mean_err,
                "val_loss":       val_loss,
                "latent_dim":     LATENT_DIM,
                "n_components":   args.K,
                "action_dim":     ACTION_DIM,
                "use_ema":        not args.no_ema,
                "use_encoder_ln": not args.no_encoder_ln,
                "full_bptt":      args.full_bptt,
                "phase":          "phase1" if phase1 else "phase2",
            }
            torch.save(ckpt, out_dir / "best.pt")

    torch.save(
        {"state": model.state_dict(), "epoch": args.epochs,
         "latent_dim": LATENT_DIM, "n_components": args.K, "action_dim": ACTION_DIM},
        out_dir / "final.pt",
    )

    cfg = {
        "latent_dim":       LATENT_DIM,
        "n_components":     args.K,
        "no_ema":           args.no_ema,
        "no_encoder_ln":    args.no_encoder_ln,
        "full_bptt":        args.full_bptt,
        "phase1_epochs":    args.phase1_epochs,
        "gn_alpha":         args.gn_alpha,
        "gn_lr":            args.gn_lr,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "lam_recon":        args.lam_recon,
        "tf_decay":         args.tf_decay,
        "tf_pmin":          args.tf_pmin,
        "best_mean_err_cm": best_mean_err,
    }
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    logger.log(f"\nSaved → {out_dir}  best_mean_err={best_mean_err:.1f}cm")
    logger.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       default="world_model/data_fixeddt")
    p.add_argument("--out-dir",        default="world_model/results/spr_gradnorm_v25")
    p.add_argument("--epochs",         type=int,   default=500)
    p.add_argument("--phase1-epochs",  type=int,   default=200,
                   help="L2 warm-up epochs before switching to NLL+GradNorm")
    p.add_argument("--batch-size",     type=int,   default=512)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--K",              type=int,   default=5,   help="MDN components")
    p.add_argument("--ema-tau",        type=float, default=0.99)
    p.add_argument("--no-ema",         action="store_true")
    p.add_argument("--no-encoder-ln",  action="store_true")
    p.add_argument("--full-bptt",      action="store_true")
    p.add_argument("--gn-alpha",       type=float, default=1.5)
    p.add_argument("--gn-lr",          type=float, default=1e-3)
    p.add_argument("--lam-recon",      type=float, default=1.0)
    p.add_argument("--pocket-weight",  type=float, default=20.0)
    p.add_argument("--focal-gamma",    type=float, default=2.0)
    p.add_argument("--tf-decay",       type=int,   default=200)
    p.add_argument("--tf-pmin",        type=float, default=0.1)
    p.add_argument("--eval-every",     type=int,   default=10)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
