"""
world_model/train_gnn_resolver.py

GNN collision resolver 학습.

Usage:
    python world_model/train_gnn_resolver.py --data world_model/data_collision/
"""

import os, sys, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.gnn_resolver import GNNResolver

COLL_NAMES = ["ball_ball", "linear", "circular", "pocket"]
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

class CollisionDataset(Dataset):
    def __init__(self, paths: list[str], exclude_pocket: bool = True):
        arrays = [np.load(p) for p in paths]
        pre_vel   = np.concatenate([a["pre_vel"]   for a in arrays], axis=0)
        pre_avel  = np.concatenate([a["pre_avel"]  for a in arrays], axis=0)
        post_vel  = np.concatenate([a["post_vel"]  for a in arrays], axis=0)
        post_avel = np.concatenate([a["post_avel"] for a in arrays], axis=0)
        pos       = np.concatenate([a["pos"]       for a in arrays], axis=0)
        normal    = np.concatenate([a["normal"]    for a in arrays], axis=0)
        coll_type = np.concatenate([a["coll_type"] for a in arrays], axis=0)
        has_tgt   = np.concatenate([a["has_tgt"]   for a in arrays], axis=0)

        if exclude_pocket:
            mask = coll_type != 3
            pre_vel, pre_avel = pre_vel[mask], pre_avel[mask]
            post_vel, post_avel = post_vel[mask], post_avel[mask]
            pos, normal = pos[mask], normal[mask]
            coll_type, has_tgt = coll_type[mask], has_tgt[mask]

        self.pre_vel   = torch.from_numpy(pre_vel).float()
        self.pre_avel  = torch.from_numpy(pre_avel).float()
        self.post_vel  = torch.from_numpy(post_vel).float()
        self.post_avel = torch.from_numpy(post_avel).float()
        self.pos       = torch.from_numpy(pos).float()
        self.normal    = torch.from_numpy(normal).float()
        self.coll_type = torch.from_numpy(coll_type.astype(np.int64))
        self.has_tgt   = torch.from_numpy(has_tgt.astype(bool))

        # 타입별 분포 출력
        for i, name in enumerate(COLL_NAMES[:3]):
            n = (coll_type == i).sum()
            print(f"  {name:20s}: {n:6d} ({n/len(coll_type)*100:.1f}%)")
        print(f"  Total: {len(coll_type)}")

    def __len__(self):
        return len(self.pre_vel)

    def __getitem__(self, idx):
        return (self.pos[idx], self.pre_vel[idx], self.pre_avel[idx],
                self.normal[idx], self.has_tgt[idx],
                self.post_vel[idx], self.post_avel[idx],
                self.coll_type[idx])


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(delta_vel, delta_avel, pre_vel, pre_avel, post_vel, post_avel, has_tgt):
    """
    delta_vel/avel: 모델 예측값 (B,2,2/3)
    GT delta = post - pre
    has_tgt: (B,) bool — tgt ball이 있는 경우만 tgt loss 계산
    """
    gt_dv  = post_vel  - pre_vel    # (B,2,2)
    gt_da  = post_avel - pre_avel   # (B,2,3)

    # cue ball: 항상
    loss_cue_v = nn.functional.mse_loss(delta_vel[:, 0],  gt_dv[:, 0])
    loss_cue_a = nn.functional.mse_loss(delta_avel[:, 0], gt_da[:, 0])

    # tgt ball: has_tgt인 경우만
    tgt_mask = has_tgt.float().unsqueeze(-1)  # (B,1)
    n_tgt = has_tgt.sum().clamp(min=1)

    loss_tgt_v = ((delta_vel[:, 1]  - gt_dv[:, 1]).pow(2).mean(-1) * has_tgt.float()).sum() / n_tgt
    loss_tgt_a = ((delta_avel[:, 1] - gt_da[:, 1]).pow(2).mean(-1) * has_tgt.float()).sum() / n_tgt

    return loss_cue_v + loss_cue_a + loss_tgt_v + loss_tgt_a, {
        "cue_v": loss_cue_v.item(),
        "cue_a": loss_cue_a.item(),
        "tgt_v": loss_tgt_v.item(),
        "tgt_a": loss_tgt_a.item(),
    }


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    type_err = {0: [], 1: [], 2: []}  # vel MSE per type

    for batch in loader:
        pos, pre_vel, pre_avel, normal, has_tgt, post_vel, post_avel, coll_type = \
            [x.to(device) for x in batch]

        dv, da = model(pos, pre_vel, pre_avel, normal, has_tgt)
        loss, _ = compute_loss(dv, da, pre_vel, pre_avel, post_vel, post_avel, has_tgt)
        total_loss += loss.item() * len(pos)

        # per-type cue vel error (denorm: ×12 m/s → cm 환산 후 계산)
        gt_dv = post_vel - pre_vel
        err_cue = (dv[:, 0] - gt_dv[:, 0]).pow(2).mean(-1).sqrt()  # (B,) in normalized units
        for t in range(3):
            mask = (coll_type == t)
            if mask.any():
                type_err[t].extend(err_cue[mask].cpu().tolist())

    n = sum(len(v) for v in type_err.values())
    avg_loss = total_loss / n if n > 0 else 0.0

    type_rmse = {}
    for t, errs in type_err.items():
        if errs:
            rmse_norm = float(np.mean(errs))
            rmse_ms   = rmse_norm * 12.0  # m/s
            type_rmse[COLL_NAMES[t]] = rmse_ms

    return avg_loss, type_rmse


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    # data
    paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files in {args.data}")
    print(f"Loading {len(paths)} file(s):")
    for p in paths: print(f"  {p}")

    dataset = CollisionDataset(paths)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"Train: {n_train}  Val: {n_val}")

    # model
    model = GNNResolver(hidden=args.hidden, node_dim=args.node_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}  device={DEVICE}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # resume from checkpoint if exists
    start_ep = 1
    best_val = float("inf")
    stall    = 0
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        best_val = ckpt["val_loss"]
        start_ep = ckpt["epoch"] + 1
        # fast-forward scheduler
        for _ in range(ckpt["epoch"]):
            sched.step()
        print(f"Resumed from ep{ckpt['epoch']}  best_val={best_val:.5f}")

    for ep in range(start_ep, start_ep + args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for batch in train_dl:
            pos, pre_vel, pre_avel, normal, has_tgt, post_vel, post_avel, coll_type = \
                [x.to(DEVICE) for x in batch]

            dv, da = model(pos, pre_vel, pre_avel, normal, has_tgt)
            loss, _ = compute_loss(dv, da, pre_vel, pre_avel, post_vel, post_avel, has_tgt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(pos)

        sched.step()
        avg_train = total_loss / n_train

        if ep % args.eval_every == 0 or ep == start_ep + args.epochs - 1:
            val_loss, type_rmse = evaluate(model, val_dl, DEVICE)
            dt = time.time() - t0
            rmse_str = "  ".join(f"{k}={v:.4f}m/s" for k, v in type_rmse.items())

            if val_loss < best_val:
                best_val = val_loss
                stall = 0
                torch.save({"epoch": ep, "model": model.state_dict(),
                            "val_loss": val_loss, "type_rmse": type_rmse},
                           ckpt_path)
                marker = "*"
            else:
                stall += 1
                marker = f"(stall {stall}/{args.patience})"

            print(f"ep{ep:4d}  train={avg_train:.5f}  val={val_loss:.5f}  [{rmse_str}]  {dt:.1f}s  {marker}")

            if args.patience > 0 and stall >= args.patience:
                print(f"Early stop: {stall} stalls")
                break

    print(f"\nBest val_loss: {best_val:.5f}")
    print(f"Saved → {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default="world_model/data_collision")
    p.add_argument("--out-dir",    type=str, default="world_model/results/gnn_resolver")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch",      type=int, default=1024)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--hidden",     type=int, nargs="+", default=[128, 128])
    p.add_argument("--node-dim",   type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience",   type=int, default=15)
    p.add_argument("--resume",     action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
