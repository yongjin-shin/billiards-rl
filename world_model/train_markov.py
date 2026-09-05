"""
world_model/train_markov.py — MarkovPredictor 학습 스크립트

두 모델을 순차 학습:
    1. MarkovTransition: (event_t, event_{t+1}) 쌍으로 순수 supervised Markov 학습
       - 각 에피소드에서 인접 이벤트 쌍을 모두 flatten → 대용량 데이터셋
       - Teacher forcing 없음: 단순 supervised
    2. MarkovEncoder   : (obs, act) → 첫 이벤트 예측
       - 학습 데이터: events[:, 0, :] (첫 이벤트만)

Usage:
    python world_model/train_markov.py --data-dir world_model/data_v3
    python world_model/train_markov.py \\
        --data-dir world_model/data_v3 --tags random_v3 sac_v3 \\
        --enc-epochs 50 --trans-epochs 100 --batch-size 512
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.generate_data_v3 import EVENT_DIM_V3
from world_model.wm_predictor import TABLE_W, TABLE_H, MAX_EVENTS, EVENT_TYPES
from world_model.markov_predictor import (
    MarkovEncoder, MarkovTransition,
    S_CUE_XY, S_TGT_XY, S_TYPE_OH,
    encoder_loss, transition_loss,
)

# Collision event type indices (state-transition events 6-9 are excluded)
COLLISION_TYPES = frozenset({2, 3, 4, 5})  # ball_ball, linear/circular_cushion, ball_pocket

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    """
    충돌 이벤트 전용 (event_t, event_{t+1}) 쌍 데이터셋.
    state-transition 이벤트(6-9)를 건너뛰고 collision끼리만 쌍으로 묶음.
    max_per_class: 다음 이벤트 타입별 최대 샘플 수 (undersampling).
    class_weights: inverse-frequency 가중치 tensor (10,) — transition_loss에 전달.
    """
    def __init__(self, data: dict, indices: np.ndarray,
                 augment: bool = True, max_per_class: int | None = None):
        events    = data["events"][indices]      # (N, T, 24)
        cue_masks = data["cue_masks"][indices]   # (N, T)
        tgt_masks = data["tgt_masks"][indices]   # (N, T)
        lengths   = data["lengths"][indices]     # (N,)

        if augment:
            events, cue_masks, tgt_masks = self._augment(events, cue_masks, tgt_masks)

        # Precompute event type index per (episode, timestep)
        all_types    = events[:, :, 14:24].argmax(axis=2)          # (N, T)
        is_collision = np.isin(all_types, list(COLLISION_TYPES))    # (N, T)

        # Build collision-only pairs: skip state-transition events
        pairs_s, pairs_s1 = [], []
        pairs_cm, pairs_tm = [], []
        for i, L in enumerate(lengths):
            coll_ts = np.where(is_collision[i, :int(L)])[0]
            for k in range(len(coll_ts) - 1):
                t, t1 = coll_ts[k], coll_ts[k + 1]
                pairs_s.append(events[i, t])
                pairs_s1.append(events[i, t1])
                pairs_cm.append(cue_masks[i, t1])
                pairs_tm.append(tgt_masks[i, t1])

        pairs_s   = np.stack(pairs_s)                           # (M, 24)
        pairs_s1  = np.stack(pairs_s1)
        pairs_cm  = np.array(pairs_cm, dtype=np.float32)
        pairs_tm  = np.array(pairs_tm, dtype=np.float32)
        next_types = pairs_s1[:, 14:24].argmax(axis=1)          # (M,)

        # Per-class undersampling cap
        if max_per_class is not None:
            rng  = np.random.default_rng(0)
            keep = []
            for cls in np.unique(next_types):
                idx = np.where(next_types == cls)[0]
                if len(idx) > max_per_class:
                    idx = rng.choice(idx, max_per_class, replace=False)
                keep.append(idx)
            keep       = np.concatenate(keep)
            pairs_s    = pairs_s[keep]
            pairs_s1   = pairs_s1[keep]
            pairs_cm   = pairs_cm[keep]
            pairs_tm   = pairs_tm[keep]
            next_types = next_types[keep]

        # Class distribution summary
        counts = np.bincount(next_types, minlength=10)
        total  = counts.sum()
        print("  Next-event distribution (collision pairs):")
        for cls, cnt in enumerate(counts):
            if cnt > 0:
                print(f"    {EVENT_TYPES[cls]:28s} {cnt:7d}  ({cnt/total*100:.1f}%)")

        # Inverse-frequency class weights (balanced, 0 for absent classes)
        present = counts > 0
        weights = np.zeros(10, dtype=np.float32)
        weights[present] = total / (counts[present] * present.sum())
        self.class_weights = torch.from_numpy(weights).float()

        self.state_t  = torch.from_numpy(pairs_s).float()
        self.state_t1 = torch.from_numpy(pairs_s1).float()
        self.cue_m    = torch.from_numpy(pairs_cm).float()
        self.tgt_m    = torch.from_numpy(pairs_tm).float()

    @staticmethod
    def _augment(events, cue_masks, tgt_masks):
        """LR / TB 독립 50% flip — 좌표 공간 데이터 증강."""
        N = events.shape[0]
        out = events.copy()
        lr = np.random.rand(N) < 0.5
        tb = np.random.rand(N) < 0.5

        for s in (S_CUE_XY, S_TGT_XY):
            # x flip (LR)
            out[lr, :, s.start]     = 1.0 - events[lr, :, s.start]
            # y flip (TB)
            out[tb, :, s.start + 1] = 1.0 - events[tb, :, s.start + 1]
        # vel x/y 부호 flip (물리적으로 일관)
        for sl in (slice(2, 4), slice(9, 11)):
            out[lr, :, sl.start]     = -events[lr, :, sl.start]
            out[tb, :, sl.start + 1] = -events[tb, :, sl.start + 1]

        return out, cue_masks.copy(), tgt_masks.copy()

    def __len__(self): return len(self.state_t)

    def __getitem__(self, idx):
        return (self.state_t[idx], self.state_t1[idx],
                self.cue_m[idx],   self.tgt_m[idx])


class EncoderDataset(Dataset):
    """(obs, act) → 첫 충돌 이벤트 예측 데이터셋. length > 0 에피소드만.
    12%의 에피소드는 index 0이 sliding_rolling이므로, 첫 COLLISION 이벤트를 찾아 사용."""
    def __init__(self, data: dict, indices: np.ndarray):
        valid = data["lengths"][indices] > 0
        idx   = indices[valid]

        ev_arr = data["events"][idx]     # (N, T, 24)
        L_arr  = data["lengths"][idx]    # (N,)

        # Find first collision event index per episode
        all_types    = ev_arr[:, :, 14:24].argmax(axis=2)       # (N, T)
        is_collision = np.isin(all_types, list(COLLISION_TYPES)) # (N, T)
        first_t = np.zeros(len(idx), dtype=int)
        for i, L in enumerate(L_arr):
            coll_ts = np.where(is_collision[i, :int(L)])[0]
            if len(coll_ts) > 0:
                first_t[i] = coll_ts[0]

        obs_norm = data["obs"][idx].astype(np.float32).copy()
        obs_norm[:, 0::2] /= TABLE_W
        obs_norm[:, 1::2] /= TABLE_H

        row = np.arange(len(idx))
        self.obs_norm  = torch.from_numpy(obs_norm).float()
        self.act       = torch.from_numpy(data["actions"][idx]).float()
        self.event0    = torch.from_numpy(ev_arr[row, first_t, :]).float()
        self.cue_m     = torch.from_numpy(data["cue_masks"][idx][row, first_t]).float()
        self.tgt_m     = torch.from_numpy(data["tgt_masks"][idx][row, first_t]).float()

    def __len__(self): return len(self.obs_norm)

    def __getitem__(self, idx):
        return (self.obs_norm[idx], self.act[idx],
                self.event0[idx], self.cue_m[idx], self.tgt_m[idx])


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def load_data(data_dir: str, tags: list[str] | None) -> dict:
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    files = [m["file"] for m in meta
             if m.get("format") == "v3" and
             (tags is None or m.get("tag") in tags)]
    if not files:
        raise ValueError(f"No v3 data found for tags={tags} in {data_dir}")

    arrays: dict = {}
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        with np.load(fpath) as f:
            for k, v in f.items():
                arrays.setdefault(k, []).append(v)
        print(f"  Loaded: {fname}")

    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


# ── 학습 루틴 ─────────────────────────────────────────────────────────────────

def train_transition(model, train_loader, val_loader, args, device,
                     class_weights: torch.Tensor | None = None):
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.trans_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )

    cw      = class_weights.to(device) if class_weights is not None else None
    best_val, best_state = float("inf"), None
    out_dir = Path(args.out_dir)

    for epoch in range(1, args.trans_epochs + 1):
        model.train()
        train_losses = []
        for state_t, state_t1, cue_m, tgt_m in train_loader:
            state_t  = state_t.to(device);  state_t1 = state_t1.to(device)
            cue_m    = cue_m.to(device);    tgt_m    = tgt_m.to(device)

            logits, cue_out, tgt_out = model(state_t)
            loss, *_ = transition_loss(logits, cue_out, tgt_out,
                                       state_t, state_t1, cue_m, tgt_m,
                                       class_weights=cw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for state_t, state_t1, cue_m, tgt_m in val_loader:
                state_t  = state_t.to(device);  state_t1 = state_t1.to(device)
                cue_m    = cue_m.to(device);    tgt_m    = tgt_m.to(device)
                logits, cue_out, tgt_out = model(state_t)
                loss, *_ = transition_loss(logits, cue_out, tgt_out,
                                           state_t, state_t1, cue_m, tgt_m,
                                           class_weights=cw)
                val_losses.append(loss.item())

        tr = np.mean(train_losses);  vl = np.mean(val_losses)
        print(f"[Trans] epoch {epoch:3d}/{args.trans_epochs}  "
              f"train={tr:.4f}  val={vl:.4f}")
        if WANDB and args.wandb:
            wandb.log({"trans/train_loss": tr, "trans/val_loss": vl})

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        torch.save({"state": best_state, "val_loss": best_val},
                   out_dir / "transition_best.pt")
    torch.save({"state": model.state_dict()}, out_dir / "transition_final.pt")
    print(f"[Trans] best val={best_val:.4f}  saved → {out_dir}")


def train_encoder(model, train_loader, val_loader, args, device):
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.enc_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )

    best_val, best_state = float("inf"), None
    out_dir = Path(args.out_dir)

    for epoch in range(1, args.enc_epochs + 1):
        model.train()
        train_losses = []
        for obs_n, act, ev0, cue_m, tgt_m in train_loader:
            obs_n  = obs_n.to(device);  act   = act.to(device)
            ev0    = ev0.to(device);    cue_m = cue_m.to(device); tgt_m = tgt_m.to(device)

            logits, pos, vel, avel = model(obs_n, act)
            loss, *_ = encoder_loss(logits, pos, vel, avel, ev0, cue_m, tgt_m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs_n, act, ev0, cue_m, tgt_m in val_loader:
                obs_n  = obs_n.to(device);  act   = act.to(device)
                ev0    = ev0.to(device);    cue_m = cue_m.to(device); tgt_m = tgt_m.to(device)
                logits, pos, vel, avel = model(obs_n, act)
                loss, *_ = encoder_loss(logits, pos, vel, avel, ev0, cue_m, tgt_m)
                val_losses.append(loss.item())

        tr = np.mean(train_losses);  vl = np.mean(val_losses)
        print(f"[Enc]   epoch {epoch:3d}/{args.enc_epochs}  "
              f"train={tr:.4f}  val={vl:.4f}")
        if WANDB and args.wandb:
            wandb.log({"enc/train_loss": tr, "enc/val_loss": vl})

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        torch.save({"state": best_state, "val_loss": best_val},
                   out_dir / "encoder_best.pt")
    torch.save({"state": model.state_dict()}, out_dir / "encoder_final.pt")
    print(f"[Enc]   best val={best_val:.4f}  saved → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",     type=str, default="world_model/data_v3")
    p.add_argument("--tags",         nargs="+", default=None)
    p.add_argument("--trans-epochs", type=int, default=100)
    p.add_argument("--enc-epochs",   type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--val-frac",     type=float, default=0.1)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--trans-hidden", nargs="+", type=int, default=[256, 512, 256])
    p.add_argument("--enc-hidden",   nargs="+", type=int, default=[128, 256, 256])
    p.add_argument("--embed-dim",    type=int, default=32)
    p.add_argument("--out-dir",      type=str, default=None)
    p.add_argument("--wandb",              action="store_true")
    p.add_argument("--skip-trans",        action="store_true")
    p.add_argument("--skip-enc",          action="store_true")
    p.add_argument("--max-pairs-per-class", type=int, default=None,
                   help="Undersample: max pairs per next-event class (e.g. 30000)")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        args.out_dir = f"world_model/results/markov_{ts}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터 로딩 ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    data   = load_data(args.data_dir, args.tags)
    N      = len(data["obs"])
    rng    = np.random.default_rng(args.seed)
    perm   = rng.permutation(N)
    n_val  = max(1, int(N * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"N={N}  train={len(train_idx)}  val={n_val}")

    # ── config 저장 ───────────────────────────────────────────────────────────
    cfg = vars(args)
    with open(Path(args.out_dir) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if WANDB and args.wandb:
        wandb.init(project="billiards-markov-wm", config=cfg)

    # ── Transition 학습 ───────────────────────────────────────────────────────
    if not args.skip_trans:
        print("\n── Transition Model ─────────────────────────────────────────")
        tr_ds  = TransitionDataset(data, train_idx, augment=True,
                                   max_per_class=args.max_pairs_per_class)
        val_ds = TransitionDataset(data, val_idx,   augment=False)
        print(f"Transition pairs: train={len(tr_ds)}  val={len(val_ds)}")

        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                                num_workers=0, pin_memory=True)

        trans = MarkovTransition(tuple(args.trans_hidden), args.embed_dim).to(device)
        train_transition(trans, tr_loader, val_loader, args, device,
                         class_weights=tr_ds.class_weights)

    # ── Encoder 학습 ─────────────────────────────────────────────────────────
    if not args.skip_enc:
        print("\n── Encoder ──────────────────────────────────────────────────")
        enc_tr  = EncoderDataset(data, train_idx)
        enc_val = EncoderDataset(data, val_idx)
        print(f"Encoder samples: train={len(enc_tr)}  val={len(enc_val)}")

        enc_tr_loader  = DataLoader(enc_tr,  batch_size=args.batch_size, shuffle=True,
                                    num_workers=0, pin_memory=True)
        enc_val_loader = DataLoader(enc_val, batch_size=args.batch_size * 2,
                                    num_workers=0, pin_memory=True)

        enc = MarkovEncoder(tuple(args.enc_hidden)).to(device)
        train_encoder(enc, enc_tr_loader, enc_val_loader, args, device)

    if WANDB and args.wandb:
        wandb.finish()

    print(f"\nDone. Outputs → {args.out_dir}")


if __name__ == "__main__":
    main()
