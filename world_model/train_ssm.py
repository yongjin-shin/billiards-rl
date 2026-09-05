"""
world_model/train_ssm.py — Deterministic SSM 학습

AR(train_fixeddt.py)과의 차이:
  - SSMDataset: (s_t, s_{t+1}) 쌍 대신 T-step episode chunk
  - 학습: single-step loss 대신 multi-step rollout loss
  - 평가: 타입분류 정확도 + rollout 위치 오차 (cm 단위, AR과 비교용)

Usage:
    python world_model/train_ssm.py --data-dir world_model/data_fixeddt
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
from world_model.ssm_model import SSMWorldModel, ssm_rollout_loss, LATENT_DIM
from world_model.train_fixeddt import augment_state
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H


class SSMDataset(Dataset):
    """
    에피소드에서 rollout_steps 길이 chunk 랜덤 샘플.

    __getitem__ 반환:
        states     (T+1, 14)  — s_{t0}, ..., s_{t0+T}
        coll_flags (T,)       — bool
        coll_types (T,)       — int
    """

    def __init__(self, data_dir: str, rollout_steps: int = 16,
                 augment: bool = True):
        meta_path = Path(data_dir) / "metadata.json"
        assert meta_path.exists(), f"metadata.json not found in {data_dir}"
        meta = json.load(open(meta_path))

        self.rollout_steps = rollout_steps
        self.augment = augment

        # 에피소드 단위로 저장 (메모리 절약: 유효 길이만)
        self.episodes = []   # list of (states, coll_flags, coll_types) arrays
        n_coll_per_type = np.zeros(4, dtype=np.int64)

        for entry in meta:
            fpath = Path(data_dir) / entry["file"]
            d = np.load(fpath)
            states     = d["states"]      # (N, T_MAX, 14)
            coll_flags = d["coll_flags"]  # (N, T_MAX) bool
            coll_types = d["coll_types"]  # (N, T_MAX) int8
            lengths    = d["lengths"]     # (N,)

            for i, L in enumerate(lengths):
                L = int(L)
                if L < rollout_steps + 1:
                    continue
                ep_s = states[i, :L].astype(np.float32)
                ep_f = coll_flags[i, :L-1].astype(bool)    # L-1 intervals
                ep_t = coll_types[i, :L-1].astype(np.int64)
                ep_t[ep_t == -1] = 0   # no-coll → 0 (mask으로 무시)
                self.episodes.append((ep_s, ep_f, ep_t))
                for c in range(4):
                    n_coll_per_type[c] += (ep_t[ep_f] == c).sum()

        # 클래스 가중치
        counts = np.maximum(n_coll_per_type, 1).astype(float)
        inv = 1.0 / counts
        self.class_weights = torch.from_numpy((inv / inv.sum() * 4).astype(np.float32))

        total_colls = n_coll_per_type.sum()
        print(f"SSMDataset: {len(self.episodes):,} episodes  rollout_steps={rollout_steps}")
        print(f"  충돌 타입 분포:")
        for i, name in enumerate(["ball_ball", "linear", "circular", "pocket"]):
            print(f"    {name}: {n_coll_per_type[i]:,}  weight={self.class_weights[i]:.3f}")

    def __len__(self):
        return len(self.episodes) * 4   # 에피소드당 여러 chunk (oversampling)

    def __getitem__(self, idx):
        ep_idx = idx % len(self.episodes)
        ep_s, ep_f, ep_t = self.episodes[ep_idx]
        L = len(ep_s)

        # 랜덤 시작점
        max_start = L - self.rollout_steps - 1
        t0 = np.random.randint(0, max(1, max_start))
        t1 = t0 + self.rollout_steps + 1   # T+1 states

        seq_s = torch.from_numpy(ep_s[t0:t1])          # (T+1, 14)
        seq_f = torch.from_numpy(ep_f[t0:t0+self.rollout_steps])   # (T,)
        seq_t = torch.from_numpy(ep_t[t0:t0+self.rollout_steps])   # (T,)

        if self.augment:
            flip_lr = torch.rand(1).item() < 0.5
            flip_tb = torch.rand(1).item() < 0.5
            if flip_lr or flip_tb:
                seq_s = torch.stack([
                    augment_state(seq_s[i], flip_lr, flip_tb)
                    for i in range(seq_s.shape[0])
                ])

        return seq_s, seq_f, seq_t


def evaluate_rollout_error(model: SSMWorldModel, episodes, device: str,
                           n_eval: int = 200, rollout_steps: int = 60):
    """
    rollout 위치 오차 (cm) — t=0.5s/1s/2s/3s 시점.
    AR과 비교용.
    """
    model.eval()
    checkpoints = {
        "0.5s": int(0.5 / DT),
        "1.0s": int(1.0 / DT),
        "2.0s": int(2.0 / DT),
        "3.0s": int(3.0 / DT),
    }
    errors = {k: [] for k in checkpoints}

    rng = np.random.default_rng(0)
    ep_indices = rng.choice(len(episodes), min(n_eval, len(episodes)), replace=False)

    with torch.no_grad():
        for ep_idx in ep_indices:
            ep_s, _, _ = episodes[ep_idx]
            L = len(ep_s)
            T = min(rollout_steps, L - 1)
            if T < 1:
                continue

            s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
            s_hat, _, _ = model(s0, n_steps=T)
            s_hat_np = s_hat[0].cpu().numpy()   # (T+1, 14)

            for label, t in checkpoints.items():
                if t > T:   # s_hat has shape (T+1,) — index T is valid
                    continue
                # cue 오차
                cue_err = np.sqrt(
                    ((s_hat_np[t, 0] - ep_s[t, 0]) * TABLE_W) ** 2 +
                    ((s_hat_np[t, 1] - ep_s[t, 1]) * TABLE_H) ** 2
                )
                # tgt 오차
                tgt_err = np.sqrt(
                    ((s_hat_np[t, 7] - ep_s[t, 7]) * TABLE_W) ** 2 +
                    ((s_hat_np[t, 8] - ep_s[t, 8]) * TABLE_H) ** 2
                )
                errors[label].append((cue_err + tgt_err) / 2 * 100)  # cm

    result = {}
    for k, v in errors.items():
        result[k] = np.mean(v) if v else float("nan")
    return result


def train(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = SSMDataset(args.data_dir, args.rollout_steps)
    n_val   = max(200, int(len(dataset.episodes) * 0.1))
    n_train = len(dataset.episodes) - n_val

    # 에피소드 단위 split
    all_eps = dataset.episodes
    rng_split = np.random.default_rng(0)
    perm = rng_split.permutation(len(all_eps))
    val_eps_idx   = perm[:n_val]
    train_eps_idx = perm[n_val:]

    # train/val dataset 분리
    class EpisodeSubset(Dataset):
        def __init__(self, eps, rollout_steps, augment):
            self.episodes = eps
            self.rollout_steps = rollout_steps
            self.augment = augment
        def __len__(self):
            return len(self.episodes) * 4
        def __getitem__(self, idx):
            ep_s, ep_f, ep_t = self.episodes[idx % len(self.episodes)]
            L = len(ep_s)
            max_start = L - self.rollout_steps - 1
            t0 = np.random.randint(0, max(1, max_start))
            seq_s = torch.from_numpy(ep_s[t0:t0+self.rollout_steps+1])
            seq_f = torch.from_numpy(ep_f[t0:t0+self.rollout_steps])
            seq_t = torch.from_numpy(ep_t[t0:t0+self.rollout_steps])
            if self.augment:
                flip_lr = torch.rand(1).item() < 0.5
                flip_tb = torch.rand(1).item() < 0.5
                if flip_lr or flip_tb:
                    seq_s = torch.stack([
                        augment_state(seq_s[i], flip_lr, flip_tb)
                        for i in range(seq_s.shape[0])
                    ])
            return seq_s, seq_f, seq_t

    train_ds = EpisodeSubset([all_eps[i] for i in train_eps_idx], args.rollout_steps, True)
    val_ds   = EpisodeSubset([all_eps[i] for i in val_eps_idx],   args.rollout_steps, False)
    val_episodes_list = [all_eps[i] for i in val_eps_idx]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_weights = dataset.class_weights.to(device)

    model = SSMWorldModel(args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.1,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        tr_losses = []
        for seq_s, seq_f, seq_t in train_loader:
            seq_s = seq_s.to(device)
            seq_f = seq_f.to(device)
            seq_t = seq_t.to(device)

            s_hat, p_coll, type_logit = model(seq_s[:, 0], args.rollout_steps)
            loss, _ = ssm_rollout_loss(
                s_hat, seq_s, p_coll, seq_f, type_logit, seq_t,
                class_weights=class_weights,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_losses, val_details = [], []
        coll_correct, coll_total = 0, 0

        with torch.no_grad():
            for seq_s, seq_f, seq_t in val_loader:
                seq_s = seq_s.to(device)
                seq_f = seq_f.to(device)
                seq_t = seq_t.to(device)

                s_hat, p_coll, type_logit = model(seq_s[:, 0], args.rollout_steps)
                loss, detail = ssm_rollout_loss(
                    s_hat, seq_s, p_coll, seq_f, type_logit, seq_t,
                    class_weights=class_weights,
                )
                val_losses.append(loss.item())
                val_details.append(detail)

                # 타입 정확도
                mask = seq_f.bool()
                if mask.any():
                    pred = type_logit[mask].argmax(-1)
                    coll_correct += (pred == seq_t[mask]).sum().item()
                    coll_total   += mask.sum().item()

        val_loss = np.mean(val_losses)
        d = {k: np.mean([x[k] for x in val_details]) for k in val_details[0]}
        type_acc = coll_correct / coll_total if coll_total > 0 else 0.0

        # rollout 오차 (10 epoch마다)
        rollout_str = ""
        if epoch % 10 == 0 or epoch == args.epochs:
            rerr = evaluate_rollout_error(model, val_episodes_list, device,
                                          n_eval=100, rollout_steps=60)
            rollout_str = "  rollout_err: " + " | ".join(
                f"{k}={v:.1f}cm" for k, v in rerr.items()
            )

        print(f"Epoch {epoch:3d}/{args.epochs}"
              f"  tr={np.mean(tr_losses):.4f}"
              f"  val={val_loss:.4f}"
              f"  state={d['loss_state']:.4f}"
              f"  coll={d['loss_coll']:.4f}"
              f"  type_acc={type_acc:.3f}"
              + rollout_str)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "latent_dim": args.latent_dim,
                "rollout_steps": args.rollout_steps,
            }, out_dir / "best.pt")

    torch.save({"state": model.state_dict(), "epoch": args.epochs,
                "latent_dim": args.latent_dim},
               out_dir / "final.pt")

    cfg = {
        "latent_dim": args.latent_dim, "rollout_steps": args.rollout_steps,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "best_val_loss": best_val,
    }
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    print(f"\nSaved → {out_dir}  best_val={best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",      default="world_model/data_fixeddt")
    p.add_argument("--out-dir",       default=None)
    p.add_argument("--rollout-steps", type=int,   default=16)
    p.add_argument("--latent-dim",    type=int,   default=LATENT_DIM)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=3e-4)
    args = p.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"world_model/results/ssm_{ts}"

    train(args)


if __name__ == "__main__":
    main()
