"""
world_model/train_vae.py — TrajectoryVAE 학습 (Exp-15)

Usage:
    python world_model/train_vae.py --data world_model/data/ --z-dim 16
    python world_model/train_vae.py --data world_model/data/ --z-dim 8
    python world_model/train_vae.py --data world_model/data/ --z-dim 32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.model import TrajectoryVAE, vae_loss, MAX_EVENTS, EVENT_DIM


# ── Dataset ──────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, tags=None):
        """
        data_dir: world_model/data/
        tags: None = 전체 로드, list = 해당 태그만
        """
        all_npz = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

        # 메타 필터링
        meta_path = os.path.join(data_dir, "metadata.json")
        tag_map   = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for entry in json.load(f):
                    tag_map[entry["file"]] = entry["tag"]

        obs_list, actions_list, events_list = [], [], []
        lengths_list, pocketed_list, tags_list = [], [], []

        for fname in all_npz:
            if tags is not None and tag_map.get(fname) not in tags:
                continue
            d = np.load(os.path.join(data_dir, fname))
            obs_list.append(d["obs"])
            actions_list.append(d["actions"])
            events_list.append(d["events"])
            lengths_list.append(d["lengths"])
            pocketed_list.append(d["pocketed"])
            tag = tag_map.get(fname, "unknown")
            tags_list.append(np.full(len(d["obs"]), tag))

        self.obs      = np.concatenate(obs_list)
        self.actions  = np.concatenate(actions_list)
        self.events   = np.concatenate(events_list)
        self.lengths  = np.concatenate(lengths_list)
        self.pocketed = np.concatenate(pocketed_list)
        self.tags     = np.concatenate(tags_list)

        print(f"Dataset loaded: {len(self.obs)} episodes")
        print(f"  Tags   : {np.unique(self.tags)}")
        print(f"  Pocket : {self.pocketed.mean()*100:.1f}%")
        print(f"  Avg L  : {self.lengths.mean():.1f}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "obs"     : torch.tensor(self.obs[idx],      dtype=torch.float32),
            "actions" : torch.tensor(self.actions[idx],   dtype=torch.float32),
            "events"  : torch.tensor(self.events[idx],    dtype=torch.float32),
            "lengths" : torch.tensor(self.lengths[idx],   dtype=torch.long),
            "pocketed": torch.tensor(self.pocketed[idx],  dtype=torch.float32),
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터
    dataset = TrajectoryDataset(args.data)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0)

    # 모델
    model = TrajectoryVAE(z_dim=args.z_dim,
                           hidden_enc=args.hidden_enc,
                           hidden_dec=args.hidden_dec).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: z_dim={args.z_dim}  params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5)

    # 저장 경로
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"vae_z{args.z_dim}_{ts}.pt")

    best_val_loss = float("inf")
    history = []

    print(f"\nTraining {args.epochs} epochs  |  batch={args.batch_size}  β={args.beta}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        t_loss = t_pos = t_type = t_kl = 0.0
        for batch in train_dl:
            events  = batch["events"].to(device)
            lengths = batch["lengths"].to(device)

            x_recon, mu, logvar, z = model(events, lengths)
            loss, pos_l, type_l, kl = vae_loss(events, x_recon, mu, logvar,
                                                 lengths, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item();  t_pos  += pos_l.item()
            t_type += type_l.item(); t_kl  += kl.item()

        n = len(train_dl)
        t_loss /= n; t_pos /= n; t_type /= n; t_kl /= n

        # ── val ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                events  = batch["events"].to(device)
                lengths = batch["lengths"].to(device)
                x_recon, mu, logvar, _ = model(events, lengths)
                loss, *_ = vae_loss(events, x_recon, mu, logvar, lengths, args.beta)
                v_loss += loss.item()
        v_loss /= len(val_dl)

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                "epoch"  : epoch,
                "z_dim"  : args.z_dim,
                "state"  : model.state_dict(),
                "val_loss": v_loss,
                "args"   : vars(args),
            }, ckpt_path)
            marker = " ← best"
        else:
            marker = ""

        history.append(dict(epoch=epoch, train=t_loss, val=v_loss,
                            pos=t_pos, type=t_type, kl=t_kl))

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:3d}] train={t_loss:.4f} "
                  f"(pos={t_pos:.4f} type={t_type:.4f} kl={t_kl:.4f}) "
                  f"val={v_loss:.4f}{marker}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved → {ckpt_path}")

    # history 저장
    hist_path = ckpt_path.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--z-dim",      type=int,   default=16)
    parser.add_argument("--hidden-enc", type=int,   default=64)
    parser.add_argument("--hidden-dec", type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--beta",       type=float, default=1.0,
                        help="KL weight (β-VAE)")
    parser.add_argument("--tags",       nargs="*",  default=None,
                        help="Filter dataset by tags. None = all")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
