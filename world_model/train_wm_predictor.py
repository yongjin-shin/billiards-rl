"""
world_model/train_wm_predictor.py — WMPredictor 학습

Usage:
    python -m world_model.train_wm_predictor
    python -m world_model.train_wm_predictor --lstm-hidden 512
    python -m world_model.train_wm_predictor --no-augment --no-wandb
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.wm_predictor import (
    WMPredictor, wm_loss,
    EVENT_DIM_V2, MAX_EVENTS, TABLE_W, TABLE_H,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class WMDataset(Dataset):
    def __init__(self, data_dir, tags=None, augment=True, verbose=True):
        all_npz = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))

        meta_path = os.path.join(data_dir, "metadata.json")
        tag_map   = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for entry in json.load(f):
                    tag_map[entry["file"]] = entry["tag"]

        obs_l, act_l, ev_l, cm_l, tm_l, len_l = [], [], [], [], [], []
        loaded = []
        for fname in all_npz:
            if tags is not None and tag_map.get(fname) not in tags:
                continue
            d = np.load(os.path.join(data_dir, fname))
            obs_l.append(d["obs"])
            act_l.append(d["actions"])
            ev_l.append(d["events"])
            cm_l.append(d["cue_masks"])
            tm_l.append(d["tgt_masks"])
            len_l.append(d["lengths"])
            loaded.append(fname)

        if not obs_l:
            raise ValueError(f"No data found in {data_dir} (tags={tags})")

        obs_raw = np.concatenate(obs_l)           # (N, 16)  raw 좌표

        # obs 정규화: x / TABLE_W, y / TABLE_H
        obs_norm = obs_raw.copy()
        obs_norm[:, 0::2] /= TABLE_W
        obs_norm[:, 1::2] /= TABLE_H

        self.obs_norm  = torch.from_numpy(obs_norm).float()
        self.actions   = torch.from_numpy(np.concatenate(act_l)).float()
        self.events    = torch.from_numpy(np.concatenate(ev_l)).float()
        self.cue_masks = torch.from_numpy(np.concatenate(cm_l)).float()
        self.tgt_masks = torch.from_numpy(np.concatenate(tm_l)).float()
        self.lengths   = torch.from_numpy(np.concatenate(len_l)).long()
        self.augment   = augment

        if verbose:
            print(f"  Files  : {loaded}")
            print(f"  Total  : {len(self.obs_norm):,} episodes")
            print(f"  Avg len: {self.lengths.float().mean():.1f}  "
                  f"max={self.lengths.max().item()}")
            print(f"  Augment: {augment}  (4× flip 조합)")

    def __len__(self):
        return len(self.obs_norm)

    def __getitem__(self, idx):
        obs_n  = self.obs_norm[idx].clone()
        act    = self.actions[idx].clone()
        events = self.events[idx].clone()
        cue_m  = self.cue_masks[idx]
        tgt_m  = self.tgt_masks[idx]
        length = self.lengths[idx]

        if self.augment:
            obs_n, act, events = self._flip(obs_n, act, events)

        return obs_n, act, events, cue_m, tgt_m, length

    def _flip(self, obs_n, act, events):
        """
        좌우(LR) / 상하(TB) 대칭 augmentation.
        각각 50% 확률로 독립 적용 → 4가지 조합.

        LR flip: x → 1-x,  delta_angle → -delta_angle
        TB flip: y → 1-y,  delta_angle → -delta_angle
        LR+TB  : 각도 두 번 반전 → 원래 값 (상쇄)
        """
        flip_lr = random.random() < 0.5
        flip_tb = random.random() < 0.5

        if flip_lr:
            obs_n[0::2]    = 1.0 - obs_n[0::2]     # 모든 x
            events[:, 0]   = 1.0 - events[:, 0]    # cue_x
            events[:, 2]   = 1.0 - events[:, 2]    # tgt_x
            act[0]         = -act[0]                # delta_angle

        if flip_tb:
            obs_n[1::2]    = 1.0 - obs_n[1::2]     # 모든 y
            events[:, 1]   = 1.0 - events[:, 1]    # cue_y
            events[:, 3]   = 1.0 - events[:, 3]    # tgt_y
            act[0]         = -act[0]                # delta_angle

        return obs_n, act, events


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--data",          type=str,   default="world_model/data_v2")
    p.add_argument("--tags",          type=str,   nargs="+",
                   default=["sac_abs", "random_abs"])
    p.add_argument("--no-augment",    action="store_true",
                   help="데이터 augmentation 비활성화")
    # model
    p.add_argument("--enc-hidden",    type=int,   nargs="+", default=[128, 256])
    p.add_argument("--lstm-hidden",   type=int,   default=256)
    p.add_argument("--lstm-layers",   type=int,   default=2)
    p.add_argument("--lstm-dropout",  type=float, default=0.1,
                   help="LSTM inter-layer dropout (lstm_layers>1 일 때만 적용)")
    p.add_argument("--event-embed-dim", type=int, default=32)
    # training
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--lambda-event",     type=float, default=1.0,
                   help="Event CE loss weight")
    p.add_argument("--lambda-pos",       type=float, default=1.0,
                   help="Position MSE loss weight")
    p.add_argument("--label-smoothing",  type=float, default=0.1,
                   help="Label smoothing for event CE (0.0 = off)")
    p.add_argument("--ss-epochs",     type=int,   default=50,
                   help="Scheduled sampling: tf_ratio 1→0 감소 구간")
    p.add_argument("--ss-min-ratio",  type=float, default=0.0)
    p.add_argument("--pct-start",     type=float, default=0.1,
                   help="OneCycleLR warmup 비율 (default: 10%%)")
    p.add_argument("--final-div-factor", type=float, default=1e3,
                   help="min_lr = max_lr / final_div_factor (default: 1e3 → 1e-6)")
    # misc
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--wandb-project", type=str,   default="billiards-wm-v2")
    p.add_argument("--no-wandb",      action="store_true")
    return p.parse_args()


# ── Experiment directory ──────────────────────────────────────────────────────

def make_exp_dir(args) -> str:
    ts    = time.strftime("%Y%m%d_%H%M%S")
    enc   = "_".join(str(h) for h in args.enc_hidden)
    aug   = "" if args.no_augment else "_aug"
    name  = (f"wmv2_enc{enc}_h{args.lstm_hidden}_l{args.lstm_layers}"
             f"_emb{args.event_embed_dim}_s{args.seed}{aug}_{ts}")
    path  = os.path.join("world_model", "checkpoints", name)
    os.makedirs(path, exist_ok=True)
    return path


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    W      = 72

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  Loading: {args.data}  tags={args.tags}")
    from torch.utils.data import Subset
    train_ds_base = WMDataset(args.data, tags=args.tags, augment=not args.no_augment)
    val_ds_base   = WMDataset(args.data, tags=args.tags, augment=False, verbose=False)

    n_val   = max(1, int(len(train_ds_base) * 0.1))
    n_train = len(train_ds_base) - n_val
    perm    = torch.randperm(len(train_ds_base),
                             generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_ds = Subset(train_ds_base, perm[:n_train])
    val_ds   = Subset(val_ds_base,   perm[n_train:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    model = WMPredictor(
        enc_hidden      = args.enc_hidden,
        lstm_hidden     = args.lstm_hidden,
        lstm_layers     = args.lstm_layers,
        lstm_dropout    = args.lstm_dropout,
        event_embed_dim = args.event_embed_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_loader)   # floor 아닌 실제 배치 수
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = args.lr,
        steps_per_epoch  = steps_per_epoch,
        epochs           = args.epochs,
        pct_start        = args.pct_start,
        anneal_strategy  = "cos",
        final_div_factor = args.final_div_factor,
    )

    # ── config ────────────────────────────────────────────────────────────────
    exp_dir  = make_exp_dir(args)
    run_name = os.path.basename(exp_dir)
    config   = {
        "data":            args.data,
        "tags":            args.tags,
        "augment":         not args.no_augment,
        "enc_hidden":      args.enc_hidden,
        "lstm_hidden":     args.lstm_hidden,
        "lstm_layers":     args.lstm_layers,
        "lstm_dropout":    args.lstm_dropout,
        "event_embed_dim": args.event_embed_dim,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "lambda_event":    args.lambda_event,
        "lambda_pos":      args.lambda_pos,
        "label_smoothing":    args.label_smoothing,
        "ss_epochs":       args.ss_epochs,
        "ss_min_ratio":    args.ss_min_ratio,
        "pct_start":       args.pct_start,
        "final_div_factor": args.final_div_factor,
        "seed":            args.seed,
        "device":          args.device,
        "n_params":        n_params,
        "n_train":         n_train,
        "n_val":           n_val,
        "exp_dir":         exp_dir,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = config,
            tags    = [f"seed:{args.seed}",
                       "aug" if not args.no_augment else "no-aug"],
        )

    # ── header ────────────────────────────────────────────────────────────────
    print(f"{'═' * W}")
    print(f"  WMPredictor v2  seed={args.seed}  device={args.device}")
    print(f"  enc={args.enc_hidden}  lstm_h={args.lstm_hidden}"
          f"  layers={args.lstm_layers}  emb={args.event_embed_dim}")
    print(f"  n_params={n_params:,}  λ_event={args.lambda_event}"
          f"  λ_pos={args.lambda_pos}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}"
          f"  pct_start={args.pct_start}  final_div={args.final_div_factor:.0e}"
          f"  (min_lr={args.lr / args.final_div_factor:.1e})")
    print(f"  ss_epochs={args.ss_epochs}  ss_min={args.ss_min_ratio}"
          f"  augment={not args.no_augment}")
    print(f"  train={n_train:,}  val={n_val:,}")
    print(f"  exp_dir: {exp_dir}")
    print(f"{'═' * W}\n")

    best_val = float("inf")
    t_start  = time.time()

    for epoch in range(1, args.epochs + 1):

        # ── tf_ratio ──────────────────────────────────────────────────────────
        progress = min(1.0, (epoch - 1) / max(1, args.ss_epochs - 1))
        tf_ratio = 1.0 - progress * (1.0 - args.ss_min_ratio)

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr_total = tr_event = tr_cue = tr_tgt = 0.0

        for obs_n, act, events, cue_m, tgt_m, lengths in train_loader:
            obs_n, act     = obs_n.to(device),   act.to(device)
            events         = events.to(device)
            cue_m, tgt_m   = cue_m.to(device),   tgt_m.to(device)
            lengths        = lengths.to(device)

            event_logits, pos_pred = model(obs_n, act, events, lengths, tf_ratio)
            loss, ev_l, cu_l, tg_l = wm_loss(
                event_logits, pos_pred, events, cue_m, tgt_m, lengths,
                args.lambda_event, args.lambda_pos, args.label_smoothing,
                init_cue_tgt=obs_n[:, 0:4],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bs        = obs_n.size(0)
            tr_total += loss.item() * bs
            tr_event += ev_l.item() * bs
            tr_cue   += cu_l.item() * bs
            tr_tgt   += tg_l.item() * bs

        tr_total /= n_train;  tr_event /= n_train
        tr_cue   /= n_train;  tr_tgt   /= n_train

        # ── val (AR inference) ────────────────────────────────────────────────
        model.eval()
        va_total = va_event = va_cue = va_tgt = 0.0

        with torch.no_grad():
            for obs_n, act, events, cue_m, tgt_m, lengths in val_loader:
                obs_n, act     = obs_n.to(device),   act.to(device)
                events         = events.to(device)
                cue_m, tgt_m   = cue_m.to(device),   tgt_m.to(device)
                lengths        = lengths.to(device)

                event_logits, pos_pred = model(obs_n, act, events, lengths,
                                               tf_ratio=0.0)
                loss, ev_l, cu_l, tg_l = wm_loss(
                    event_logits, pos_pred, events, cue_m, tgt_m, lengths,
                    args.lambda_event, args.lambda_pos, args.label_smoothing,
                    init_cue_tgt=obs_n[:, 0:4],
                )

                bs        = obs_n.size(0)
                va_total += loss.item() * bs
                va_event += ev_l.item() * bs
                va_cue   += cu_l.item() * bs
                va_tgt   += tg_l.item() * bs

        va_total /= n_val;  va_event /= n_val
        va_cue   /= n_val;  va_tgt   /= n_val

        lr_now = optimizer.param_groups[0]["lr"]

        # ── checkpoint ────────────────────────────────────────────────────────
        if va_total < best_val:
            best_val = va_total
            torch.save({
                "epoch":   epoch,
                "state":   model.state_dict(),
                "val_loss": best_val,
                "args":    vars(args),
            }, os.path.join(exp_dir, "best.pt"))

        # ── wandb log ─────────────────────────────────────────────────────────
        log = {
            "train/loss":       tr_total,
            "train/event_ce":   tr_event,
            "train/cue_mse":    tr_cue,
            "train/tgt_mse":    tr_tgt,
            "val/loss":         va_total,
            "val/event_ce":     va_event,
            "val/cue_mse":      va_cue,
            "val/tgt_mse":      va_tgt,
            "val/best_loss":    best_val,
            "train/tf_ratio":   tf_ratio,
            "train/lr":         lr_now,
        }
        if not args.no_wandb:
            wandb.log(log, step=epoch)

        # ── console ───────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"  [{epoch:>4d}/{args.epochs}]"
                f"  train={tr_total:.4f}"
                f" (ev={tr_event:.4f} cue={tr_cue:.4f} tgt={tr_tgt:.4f})"
                f"  val={va_total:.4f}"
                f" (ev={va_event:.4f} cue={va_cue:.4f} tgt={va_tgt:.4f})"
                f"  best={best_val:.4f}"
                f"  tf={tf_ratio:.2f}"
                f"  lr={lr_now:.1e}"
                f"  {elapsed:.0f}s"
            )

    # ── final ─────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    torch.save({
        "epoch":    args.epochs,
        "state":    model.state_dict(),
        "val_loss": va_total,
        "args":     vars(args),
    }, os.path.join(exp_dir, "final.pt"))

    print(f"\n{'─' * W}")
    print(f"  Done  ({elapsed_total / 60:.1f} min)")
    print(f"  best val = {best_val:.4f}  →  {exp_dir}/best.pt")
    print(f"{'─' * W}")

    if not args.no_wandb:
        wandb.log({"final/best_val_loss": best_val})
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
