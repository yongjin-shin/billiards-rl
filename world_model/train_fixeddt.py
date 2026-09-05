"""
world_model/train_fixeddt.py — Fixed-Δt World Model 학습

Usage:
    python world_model/train_fixeddt.py --data-dir world_model/data_fixeddt
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
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.fixeddt_model import FixedDtWorldModel, world_model_loss


def augment_state(s: torch.Tensor, flip_lr: bool, flip_tb: bool) -> torch.Tensor:
    """
    LR/TB 반전 증강.
    s dim: [cue_x, cue_y, cue_vx, cue_vy, cue_wx, cue_wy, cue_wz,
            tgt_x, tgt_y, tgt_vx, tgt_vy, tgt_wx, tgt_wy, tgt_wz]
    """
    s = s.clone()
    if flip_lr:
        s[[0, 7]]    = 1.0 - s[[0, 7]]    # x → 1-x
        s[[2, 9]]    = -s[[2, 9]]          # vx → -vx
        s[[4, 11]]   = -s[[4, 11]]         # wx → -wx
    if flip_tb:
        s[[1, 8]]    = 1.0 - s[[1, 8]]    # y → 1-y
        s[[3, 10]]   = -s[[3, 10]]         # vy → -vy
        s[[5, 12]]   = -s[[5, 12]]         # wy → -wy
    return s


class FixedDtDataset(Dataset):
    """
    (s_t, s_{t+1}, coll_flag, coll_type) 쌍을 에피소드에서 추출.
    augment=True 이면 LR/TB 50% 확률로 반전.
    """

    def __init__(self, data_dir: str, augment: bool = True):
        meta_path = Path(data_dir) / "metadata.json"
        assert meta_path.exists(), f"metadata.json not found in {data_dir}"
        meta = json.load(open(meta_path))

        s_t_list   = []
        s_t1_list  = []
        flags_list = []
        types_list = []

        for entry in meta:
            fpath = Path(data_dir) / entry["file"]
            d = np.load(fpath)
            states     = d["states"]      # (N, T_MAX, 14)
            coll_flags = d["coll_flags"]  # (N, T_MAX)
            coll_types = d["coll_types"]  # (N, T_MAX)
            lengths    = d["lengths"]     # (N,)

            for i, L in enumerate(lengths):
                L = int(L)
                if L < 2:
                    continue
                for t in range(L - 1):
                    s_t_list.append(states[i, t])
                    s_t1_list.append(states[i, t + 1])
                    flags_list.append(coll_flags[i, t])
                    types_list.append(coll_types[i, t])

        self.s_t   = torch.from_numpy(np.array(s_t_list,   dtype=np.float32))
        self.s_t1  = torch.from_numpy(np.array(s_t1_list,  dtype=np.float32))
        self.flags = torch.from_numpy(np.array(flags_list,  dtype=bool))
        types_arr  = np.array(types_list, dtype=np.int64)
        types_arr[types_arr == -1] = 0   # -1은 mask로 제거, 임시 0
        self.types = torch.from_numpy(types_arr)
        self.augment = augment

        # 충돌 타입별 역빈도 가중치 (MarkovTransition과 동일 방식)
        coll_mask = self.flags.numpy()
        type_counts = np.bincount(self.types.numpy()[coll_mask], minlength=4).astype(float)
        type_counts = np.maximum(type_counts, 1)
        inv_freq = 1.0 / type_counts
        self.class_weights = torch.from_numpy((inv_freq / inv_freq.sum() * 4).astype(np.float32))

        print(f"Dataset: {len(self.s_t):,} pairs  augment={augment}")
        print(f"  coll rate: {self.flags.float().mean():.3f}")
        type_names = ["ball_ball", "linear", "circular", "pocket"]
        for i, name in enumerate(type_names):
            n = int((self.types[self.flags] == i).sum())
            w = self.class_weights[i].item()
            print(f"  {name}: {n:,}  weight={w:.3f}")

    def __len__(self):
        return len(self.s_t)

    def __getitem__(self, idx):
        s_t  = self.s_t[idx]
        s_t1 = self.s_t1[idx]
        if self.augment:
            flip_lr = torch.rand(1).item() < 0.5
            flip_tb = torch.rand(1).item() < 0.5
            if flip_lr or flip_tb:
                s_t  = augment_state(s_t,  flip_lr, flip_tb)
                s_t1 = augment_state(s_t1, flip_lr, flip_tb)
        return s_t, s_t1, self.flags[idx], self.types[idx]


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = FixedDtDataset(args.data_dir)
    n_val   = max(1000, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=0)

    class_weights = dataset.class_weights.to(device)

    model = FixedDtWorldModel().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.1,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_losses = []
        for s_t, s_t1, flags, types in train_loader:
            s_t, s_t1  = s_t.to(device), s_t1.to(device)
            flags, types = flags.to(device), types.to(device)

            _, _, s_hat, p_coll, type_logit = model(s_t)
            loss, _ = world_model_loss(s_hat, s_t1, p_coll, flags, type_logit, types,
                                       class_weights=class_weights)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_details = [], []
        with torch.no_grad():
            for s_t, s_t1, flags, types in val_loader:
                s_t, s_t1  = s_t.to(device), s_t1.to(device)
                flags, types = flags.to(device), types.to(device)

                _, _, s_hat, p_coll, type_logit = model(s_t)
                loss, detail = world_model_loss(s_hat, s_t1, p_coll, flags, type_logit, types,
                                                class_weights=class_weights)
                val_losses.append(loss.item())
                val_details.append(detail)

        val_loss = np.mean(val_losses)
        d = {k: np.mean([x[k] for x in val_details]) for k in val_details[0]}

        # Collision type accuracy (val)
        model.eval()
        coll_correct, coll_total = 0, 0
        with torch.no_grad():
            for s_t, s_t1, flags, types in val_loader:
                s_t = s_t.to(device)
                flags = flags.to(device)
                types = types.to(device)
                _, _, _, p_coll, type_logit = model(s_t)
                mask = flags
                if mask.any():
                    pred = type_logit[mask].argmax(-1)
                    coll_correct += (pred == types[mask]).sum().item()
                    coll_total   += mask.sum().item()
        coll_acc = coll_correct / coll_total if coll_total > 0 else 0.0

        print(f"Epoch {epoch:3d}/{args.epochs}"
              f"  tr={np.mean(tr_losses):.4f}"
              f"  val={val_loss:.4f}"
              f"  state={d['loss_state']:.4f}"
              f"  coll={d['loss_coll']:.4f}"
              f"  type={d['loss_type']:.4f}"
              f"  type_acc={coll_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "best.pt")

    torch.save({"state": model.state_dict(), "epoch": args.epochs},
               out_dir / "final.pt")

    cfg = {
        "state_dim": 14, "latent_dim": 128,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "best_val_loss": best_val,
    }
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    print(f"\nSaved → {out_dir}  best_val={best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="world_model/data_fixeddt")
    p.add_argument("--out-dir",    default=None)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=2048)
    p.add_argument("--lr",         type=float, default=3e-4)
    args = p.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"world_model/results/fixeddt_{ts}"

    train(args)


if __name__ == "__main__":
    main()
