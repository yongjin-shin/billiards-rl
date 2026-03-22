"""
world_model/analyze.py — latent space 정량 분석 (Exp-15)

1. z_dim별 reconstruction quality 비교
2. pocket 예측 가능성 (z → pocketed, linear probe)
3. action 예측 가능성 (z → angle/speed, linear regression)
4. bounce count 예측 (z → n_bounces)

Usage:
    python world_model/analyze.py \\
        --ckpt world_model/checkpoints/vae_z16_*.pt \\
        --data world_model/data/
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.model import TrajectoryVAE
from world_model.train_vae import TrajectoryDataset


def collect_latents(model, dataset, device):
    dl = DataLoader(dataset, batch_size=512, shuffle=False)
    zs, actions, pocketeds, bounces = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            events  = batch["events"].to(device)
            lengths = batch["lengths"].to(device)
            _, mu, _, _ = model(events, lengths)
            zs.append(mu.cpu().numpy())
            actions.append(batch["actions"].numpy())
            pocketeds.append(batch["pocketed"].numpy())
    return (np.concatenate(zs),
            np.concatenate(actions),
            np.concatenate(pocketeds),
            dataset.n_bounces)


def linear_probe_classification(z, y, label="pocketed"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, z, y, cv=5, scoring="accuracy")
    print(f"  {label}: acc={scores.mean():.3f} ± {scores.std():.3f}")
    return scores.mean()


def linear_probe_regression(z, y, label="angle"):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    reg = Ridge()
    scores = cross_val_score(reg, z, y, cv=5, scoring="r2")
    print(f"  {label}: R²={scores.mean():.3f} ± {scores.std():.3f}")
    return scores.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    ckpt  = torch.load(args.ckpt, map_location=device)
    z_dim = ckpt["z_dim"]
    model = TrajectoryVAE(z_dim=z_dim).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Model: z_dim={z_dim}  val_loss={ckpt['val_loss']:.4f}")

    dataset = TrajectoryDataset(args.data)
    zs, actions, pocketeds, n_bounces = collect_latents(model, dataset, device)

    print(f"\nLatent stats: mean={zs.mean():.3f}  std={zs.std():.3f}")
    print(f"  per-dim std: {zs.std(axis=0).round(3)}")

    print("\n[Linear probes]")
    linear_probe_classification(zs, pocketeds.astype(int), "pocketed")
    linear_probe_regression(zs, actions[:, 0], "delta_angle")
    linear_probe_regression(zs, actions[:, 1], "speed")
    linear_probe_regression(zs, n_bounces.astype(float), "n_bounces")

    print("\n[Correlation matrix: z_mean_dim vs key vars]")
    corr_pocket = np.array([np.corrcoef(zs[:, k], pocketeds)[0, 1] for k in range(z_dim)])
    corr_angle  = np.array([np.corrcoef(zs[:, k], actions[:, 0])[0, 1] for k in range(z_dim)])
    corr_bounce = np.array([np.corrcoef(zs[:, k], n_bounces)[0, 1] for k in range(z_dim)])

    print(f"  {'dim':<5} {'pocket':>8} {'angle':>8} {'bounce':>8}")
    for k in range(z_dim):
        print(f"  z{k:<4} {corr_pocket[k]:>8.3f} {corr_angle[k]:>8.3f} {corr_bounce[k]:>8.3f}")


if __name__ == "__main__":
    main()
