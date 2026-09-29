"""
world_model/spr_mdn/train_spr_mdn.py — SPR-MDN training script (Laplace)

Laplace MDN + action conditioning + self-chaining rollout.

Usage:
    python world_model/spr_mdn/train_spr_mdn.py \
      --data-dir world_model/data_fixeddt \
      --out-dir  world_model/results/spr_mdn_v4 \
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
    SPRMDNModel, spr_rollout_loss, LATENT_DIM, N_COMPONENTS, ACTION_DIM,
    SPRK1Model, sprk1_rollout_loss,
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


def _eval_batched(model: SPRMDNModel, episodes: list, device: str,
                  rollout_steps: int = 60) -> dict:
    model.eval()
    checkpoint_steps = {
        "0.5s": int(0.5 / DT),
        "1.0s": int(1.0 / DT),
        "2.0s": int(2.0 / DT),
        "3.0s": int(3.0 / DT),
    }
    valid = [(ep_s, ep_f, ep_t, ep_a) for ep_s, ep_f, ep_t, _, ep_a in episodes
             if len(ep_s) >= rollout_steps + 1]

    errs_all, errs_bb, errs_no = [], [], []
    cp_errors = {k: [] for k in checkpoint_steps}
    coll_gt_sum = coll_tp_sum = coll_pred_sum = 0

    with torch.no_grad():
        s0 = torch.from_numpy(
            np.stack([ep[0][0] for ep in valid])
        ).float().to(device)
        # actions for eval (action conditioning at h=0)
        acts = torch.from_numpy(
            np.stack([ep[3] for ep in valid])
        ).float().to(device)

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

        gt_flags      = ep_f[:rollout_steps]
        pred_flags    = pc[b_idx, :len(gt_flags)]
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
    K = args.n_components
    ewta_on = args.ewta_decay > 0
    v8_mode = ewta_on and args.lam_pi > 0
    k1_mode = args.k1
    v10_mode = k1_mode and args.no_ema
    v14_mode = k1_mode and getattr(args, "no_encoder_ln", False)
    if v14_mode:
        mode_str = "SPR-K1-v18LN (v14: dedicated transition LN, no encoder LN)"
    elif v10_mode:
        mode_str = "SPR-K1-noEMA (v10: pure closed-loop)"
    elif k1_mode:
        mode_str = "SPR-K1 (v9: EMA+sched-sampling)"
    else:
        mode_str = "SPR-MDN (Laplace)"
    k_str = "—" if k1_mode else f"K={K}"
    msg = (
        f"Mode: {mode_str}  {k_str}  EMA_tau={args.ema_tau}"
        f"  T~Uniform({T_MIN},{T_MAX})"
        f"  pocket_w={args.pocket_weight}  focal_gamma={args.focal_gamma}"
        f"  lam_recon={args.lam_recon}"
    )
    if k1_mode:
        msg += f"  lam_l2={args.lam_l2}"
    msg += f"  tf_decay={args.tf_decay}  tf_pmin={args.tf_pmin}"
    if not k1_mode:
        msg += f"  asym_init={args.asym_init}"
    if ewta_on and not k1_mode:
        msg += f"  EWTA: ewta_decay={args.ewta_decay}  lam_pi={args.lam_pi}"
        msg += "  [v8: single-phase EWTA+pi]" if v8_mode else f"  phase2_start={args.phase2_start}"
    logger.log(msg)

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
    use_flip_tb = getattr(args, "flip_tb", False)
    train_ds = _CapDataset(SPREpisodeSubset(train_eps, rollout_steps=T_MAX, augment=True,
                                            flip_tb=use_flip_tb),
                           MAX_EPOCH_ITEMS)
    val_ds   = SPREpisodeSubset(val_eps, rollout_steps=T_MAX, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    if k1_mode:
        model = SPRK1Model(
            latent_dim=LATENT_DIM,
            action_dim=ACTION_DIM,
            ema_tau=args.ema_tau,
            use_ema=not args.no_ema,
            use_action=not args.no_action,
            use_encoder_ln=not getattr(args, "no_encoder_ln", False),
            full_bptt=getattr(args, "full_bptt", False),
        ).to(device)
    else:
        model = SPRMDNModel(
            latent_dim=LATENT_DIM,
            n_components=args.n_components,
            action_dim=ACTION_DIM,
            ema_tau=args.ema_tau,
            asym_init=args.asym_init,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Parameters (trainable): {n_params:,}")

    w = CLASS_WEIGHTS_DEFAULT.clone()
    w[4] = args.pocket_weight
    CLASS_WEIGHTS = w.to(device)
    logger.log(f"Class weights: {w.tolist()}")

    # ema_encoder is excluded from optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    rng_T         = np.random.default_rng(42)
    best_mean_err = float("inf")
    stall_count   = 0
    rerr = {
        "mean_err": float("nan"), "mean_err_has_bb": float("nan"),
        "mean_err_no_bb": float("nan"), "coll_recall": 0.0,
        "coll_precision": 0.0, "0.5s": float("nan"),
        "1.0s": float("nan"), "2.0s": float("nan"), "3.0s": float("nan"),
    }

    for epoch in range(1, args.epochs + 1):
        cur_T = int(rng_T.integers(T_MIN, T_MAX + 1))

        # Scheduled sampling: p_tf decays from 1→p_min over tf_decay epochs
        # v10 (no-ema): pure closed-loop like v18, no teacher forcing ever
        if v10_mode:
            p_tf = 0.0
        else:
            p_tf = max(args.tf_pmin, 1.0 - (epoch - 1) / args.tf_decay)

        # EWTA schedule
        # v7 mode (lam_pi=0): Phase 1 (κ decays K→1) then Phase 2 (π-only NLL)
        # v8 mode (lam_pi>0): single phase — κ decays K→1, π trains simultaneously
        if ewta_on:
            # Phase 2 only applies in v7 mode
            in_phase2 = (epoch > args.phase2_start) and not v8_mode
            if not in_phase2:
                kappa = max(1, round(K - (K - 1) * (epoch - 1) / args.ewta_decay))
            else:
                kappa = 0   # sentinel: Phase 2 (π-only NLL, sg on μ,b)
        else:
            in_phase2 = False
            kappa     = 0

        # v7 Phase 1: force teacher forcing — π is untrained (uniform) so
        # self-chaining samples random components, corrupting EWTA rollout.
        # v8: π trains simultaneously with μ via L_π, so self-chaining is OK.
        # Non-EWTA and Phase 2: use scheduled p_tf normally.
        force_tf = ewta_on and not in_phase2 and not v8_mode
        p_tf_eff = 1.0 if force_tf else p_tf

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_losses = []
        _z_norm_logged = False
        for seq_s, seq_f, seq_t, seq_a in train_loader:
            seq_s_t = seq_s[:, :cur_T + 1].to(device)   # (B, T+1, 14)
            seq_t_t = seq_t[:, :cur_T].to(device)        # (B, T)
            seq_a_t = seq_a.to(device)                   # (B, 2)

            if k1_mode:
                s_hat, type_logit, z_hat_list, z_bar_list = \
                    model(seq_s_t[:, 0], seq_s_t, seq_a_t, cur_T, p_tf=p_tf)

                # Smoke test: log z_0 vs z_1 norms on epoch 1, first batch.
                # z_0 = encoder(s_0) is unbounded when use_encoder_ln=False;
                # z_1 = transition_ln(z_0 + MLP(z_0)) should be normalized.
                if epoch == 1 and not _z_norm_logged:
                    with torch.no_grad():
                        n0 = z_hat_list[0].norm(dim=-1).mean().item()
                        n1 = z_hat_list[1].norm(dim=-1).mean().item()
                    logger.log(f"[Smoke] z_0 norm={n0:.3f}  z_1 norm={n1:.3f}"
                               f"  ratio={n1/n0:.3f}")
                    _z_norm_logged = True

                loss, _ = sprk1_rollout_loss(
                    z_hat_list, z_bar_list,
                    s_hat, seq_s_t, type_logit, seq_t_t,
                    class_weights=CLASS_WEIGHTS,
                    focal_gamma=args.focal_gamma,
                    lam_recon=args.lam_recon,
                    lam_l2=args.lam_l2,
                    skip_h0_recon=getattr(args, "skip_h0_recon", False),
                )
            else:
                s_hat, type_logit, _, pi_list, mu_list, b_list, z_bar_list = \
                    model(seq_s_t[:, 0], seq_s_t, seq_a_t, cur_T, p_tf=p_tf_eff)
                loss, _ = spr_rollout_loss(
                    pi_list, mu_list, b_list, z_bar_list,
                    s_hat, seq_s_t, type_logit, seq_t_t,
                    class_weights=CLASS_WEIGHTS,
                    focal_gamma=args.focal_gamma,
                    lam_recon=args.lam_recon,
                    lam_pi=args.lam_pi,
                    ewta_kappa=kappa if not in_phase2 else 0,
                    ewta_phase2=in_phase2,
                )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            model.update_ema()
            tr_losses.append(loss.item())

        sched.step()

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

                if k1_mode:
                    s_hat, type_logit, z_hat_list, z_bar_list = \
                        model(seq_s[:, 0], seq_s, seq_a, val_T)
                    loss, detail = sprk1_rollout_loss(
                        z_hat_list, z_bar_list,
                        s_hat, seq_s, type_logit, seq_t,
                        class_weights=CLASS_WEIGHTS,
                        focal_gamma=args.focal_gamma,
                        lam_recon=args.lam_recon,
                        lam_l2=args.lam_l2,
                        skip_h0_recon=getattr(args, "skip_h0_recon", False),
                    )
                else:
                    s_hat, type_logit, _, pi_list, mu_list, b_list, z_bar_list = \
                        model(seq_s[:, 0], seq_s, seq_a, val_T)
                    loss, detail = spr_rollout_loss(
                        pi_list, mu_list, b_list, z_bar_list,
                        s_hat, seq_s, type_logit, seq_t,
                        class_weights=CLASS_WEIGHTS,
                        focal_gamma=args.focal_gamma,
                        lam_recon=args.lam_recon,
                        lam_pi=args.lam_pi,
                        ewta_kappa=kappa if ewta_on and not in_phase2 else 0,
                        ewta_phase2=in_phase2,
                    )
                val_losses.append(loss.item())
                val_details.append(detail)

                pred = type_logit.argmax(-1)
                type_correct += (pred == seq_t).sum().item()
                type_total   += pred.numel()

        val_loss = float(np.mean(val_losses))
        d        = {k: np.mean([x[k] for x in val_details]) for k in val_details[0]}
        type_acc = type_correct / type_total if type_total > 0 else 0.0

        # ── z_0 norm tracking (k1 + every 10 epochs) ─────────────────────
        # Monitors whether the encoder output drifts during training.
        # Only meaningful; no grad, uses the last train batch's z_hat_list.
        z0_norm_str = ""
        if k1_mode and (epoch % 10 == 0 or epoch == 1 or epoch == args.epochs):
            with torch.no_grad():
                n0 = z_hat_list[0].norm(dim=-1).mean().item()   # type: ignore[possibly-undefined]
            z0_norm_str = f"  z0_norm={n0:.2f}"

        # ── Rollout eval (every 10 epochs) ─────────────────────────────────
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            rerr = _eval_batched(model, balanced_val_eps, device, rollout_steps=60)

        mean_err    = rerr["mean_err"]
        coll_recall = rerr["coll_recall"]
        coll_prec   = rerr["coll_precision"]

        cp_keys = [k for k in ["0.5s", "1.0s", "2.0s", "3.0s"]
                   if not np.isnan(rerr.get(k, float("nan")))]
        cp_str  = " | ".join(f"{k}={rerr[k]:.1f}cm" for k in cp_keys) \
                  if (epoch % 10 == 0 or epoch == args.epochs) else ""

        phase_tag = ""
        if ewta_on:
            if in_phase2:
                phase_tag = " [P2:π-NLL]"
            elif v8_mode:
                phase_tag = f" [v8:EWTA+π κ={kappa}]"
            else:
                phase_tag = f" [P1:EWTA κ={kappa}]"

        pi_str = f"  L_pi={d['loss_pi']:.4f}" if v8_mode else ""
        if k1_mode:
            latent_str = f"  L_l2={d['loss_l2']:.4f}"
        else:
            latent_str = f"  L_nll={d['loss_nll']:.4f}{pi_str}"
        ptf_val = p_tf if k1_mode else p_tf_eff
        log_line = (
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  [T={cur_T:2d}  ptf={ptf_val:.2f}]{phase_tag}"
            f"  tr={np.mean(tr_losses):.4f}  val={val_loss:.4f}"
            f"  |{latent_str}  L_recon={d['loss_recon']:.4f}"
            f"  L_type={d['loss_type']:.4f}"
            f"  type_acc={type_acc:.3f}"
            f"  recall={coll_recall:.3f}  prec={coll_prec:.3f}"
            f"  err={mean_err:.1f}cm"
            f"  (bb={rerr['mean_err_has_bb']:.1f}/nbb={rerr['mean_err_no_bb']:.1f})"
        )
        if cp_str:
            log_line += f"  [{cp_str}]"
        log_line += z0_norm_str
        logger.log(log_line)

        if epoch % 10 == 0 or epoch == 1:
            if mean_err < best_mean_err - 1e-4:
                best_mean_err = mean_err
                stall_count   = 0
                ckpt = {
                    "state":      model.state_dict(),
                    "epoch":      epoch,
                    "mean_err":   mean_err,
                    "val_loss":   val_loss,
                    "latent_dim": LATENT_DIM,
                    "action_dim": ACTION_DIM,
                    "ema_tau":    args.ema_tau,
                    "k1_mode":    k1_mode,
                }
                if not k1_mode:
                    ckpt["n_components"] = args.n_components
                torch.save(ckpt, out_dir / "best.pt")
            else:
                stall_count += 1
                logger.log(f"  [stall {stall_count}/{args.patience}]")
            if stall_count >= args.patience:
                logger.log(
                    f"\nEarly stop: {args.patience} consecutive evals without improvement."
                    f"  best_err={best_mean_err:.1f}cm"
                )
                break

    torch.save({"state": model.state_dict(), "epoch": args.epochs,
                "latent_dim": LATENT_DIM, "n_components": args.n_components,
                "action_dim": ACTION_DIM},
               out_dir / "final.pt")

    cfg = {
        "latent_dim":       LATENT_DIM,
        "action_dim":       ACTION_DIM,
        "ema_tau":          args.ema_tau,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "pocket_weight":    args.pocket_weight,
        "focal_gamma":      args.focal_gamma,
        "lam_recon":        args.lam_recon,
        "k1_mode":          k1_mode,
        "no_ema":           args.no_ema,
        "lam_l2":           args.lam_l2,
        "best_mean_err_cm": best_mean_err,
    }
    if not k1_mode:
        cfg.update({
            "n_components": args.n_components,
            "lam_pi":       args.lam_pi,
            "asym_init":    args.asym_init,
            "ewta_decay":   args.ewta_decay,
            "phase2_start": args.phase2_start,
        })
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    logger.log(f"\nSaved → {out_dir}  best_mean_err={best_mean_err:.1f}cm")
    logger.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",        default="world_model/data_fixeddt")
    p.add_argument("--out-dir",         default="world_model/results/spr_mdn_v4")
    p.add_argument("--epochs",          type=int,   default=2000)
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--batch-size",      type=int,   default=512)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--n-components",    type=int,   default=N_COMPONENTS)
    p.add_argument("--ema-tau",         type=float, default=0.99)
    p.add_argument("--pocket-weight",   type=float, default=20.0)
    p.add_argument("--focal-gamma",     type=float, default=2.0)
    p.add_argument("--lam-recon",       type=float, default=1.0,
                   help="λ weight on L_recon (L = L_NLL + λ·L_recon + L_type)")
    p.add_argument("--tf-decay",        type=int,   default=200,
                   help="epochs over which p_tf decays from 1.0 to tf_pmin")
    p.add_argument("--tf-pmin",         type=float, default=0.1,
                   help="minimum teacher-forcing probability after decay")
    p.add_argument("--ewta-decay",      type=int,   default=0,
                   help="epochs over which κ decays K→1 (0=disable EWTA)")
    p.add_argument("--lam-pi",          type=float, default=0.0,
                   help="v8: weight for L_π = soft-CE toward EWTA winner (0=disabled, try 1.0)")
    p.add_argument("--phase2-start",    type=int,   default=300,
                   help="v7 only: epoch at which Phase 2 (π-only NLL) begins")
    p.add_argument("--asym-init",       action="store_true",
                   help="asymmetric μ bias init: component k gets +1.0 bias in dim k")
    p.add_argument("--k1",             action="store_true",
                   help="v9 ablation: replace MDN with K=1 deterministic residual transition")
    p.add_argument("--no-ema",         action="store_true",
                   help="v10 ablation (requires --k1): remove EMA encoder, use sg(encoder) target + pure closed-loop")
    p.add_argument("--no-action",      action="store_true",
                   help="remove action conditioning from transition (v18-style: z→z only)")
    p.add_argument("--no-encoder-ln",  action="store_true",
                   help="v14: use StateEncoder (no LN) + dedicated transition LN — v18-parity LN structure")
    p.add_argument("--flip-tb",        action="store_true",
                   help="v15b: add top-bottom flip augmentation (4-way: none/lr/tb/lr+tb)")
    p.add_argument("--skip-h0-recon",  action="store_true",
                   help="v16: exclude h=0 reconstruction from recon loss (matches v18)")
    p.add_argument("--full-bptt",      action="store_true",
                   help="v17: remove stop-gradient between rollout steps (full BPTT, matches v18)")
    p.add_argument("--lam-l2",         type=float, default=1.0,
                   help="weight on latent L2 prediction loss (k1 mode only; sweep 1.0→0.1→0.01)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
