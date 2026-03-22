"""
world_model/visualize.py — latent space 시각화 (Exp-15)

1. t-SNE 2D — pocketed / n_bounces / tag 별 색상
2. action 상관관계 — z[:, k] vs action[:, 0] (angle)
3. latent dimension traversal — z의 각 dim을 ±3σ 범위로 변화시켜 디코딩

Usage:
    python world_model/visualize.py \\
        --ckpt world_model/checkpoints/vae_z16_*.pt \\
        --data world_model/data/
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.model import TrajectoryVAE, MAX_EVENTS, EVENT_DIM, N_EVENT_TYPES
from world_model.train_vae import TrajectoryDataset


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    z_dim = ckpt["z_dim"]
    model = TrajectoryVAE(z_dim=z_dim).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded VAE  z_dim={z_dim}  val_loss={ckpt['val_loss']:.4f}")
    return model, z_dim


def collect_latents(model, dataset, device, max_samples=5000):
    """전체 데이터셋에서 z, action, pocketed, n_bounces, tags 수집"""
    dl = DataLoader(dataset, batch_size=512, shuffle=False)
    zs, actions, pocketeds, tags_raw = [], [], [], []

    with torch.no_grad():
        for batch in dl:
            events  = batch["events"].to(device)
            lengths = batch["lengths"].to(device)
            _, mu, _, _ = model(events, lengths)
            zs.append(mu.cpu().numpy())
            actions.append(batch["actions"].numpy())
            pocketeds.append(batch["pocketed"].numpy())

    zs       = np.concatenate(zs)[:max_samples]
    actions  = np.concatenate(actions)[:max_samples]
    pocketeds= np.concatenate(pocketeds)[:max_samples]

    # bounce count from dataset
    n_bounces = dataset.n_bounces[:max_samples]
    tags_arr  = dataset.tags[:max_samples]

    return zs, actions, pocketeds, n_bounces, tags_arr


def plot_tsne(zs, labels, title, fname, cmap="tab10", label_names=None):
    from sklearn.manifold import TSNE
    print(f"  t-SNE: {zs.shape} → 2D ...")
    emb = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(zs)

    fig, ax = plt.subplots(figsize=(8, 6))
    if label_names is None:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap,
                        alpha=0.5, s=8, linewidths=0)
        plt.colorbar(sc, ax=ax)
    else:
        unique = sorted(set(labels))
        colors = cm.tab10(np.linspace(0, 1, len(unique)))
        for i, u in enumerate(unique):
            mask = np.array(labels) == u
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]],
                       label=str(u), alpha=0.5, s=8, linewidths=0)
        ax.legend(markerscale=3, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved → {fname}")


def plot_action_correlation(zs, actions, out_dir, z_dim):
    """z의 각 dimension vs action[0] (delta_angle), action[1] (speed) 상관관계"""
    angle = actions[:, 0]
    speed = actions[:, 1]

    corrs_angle = [np.corrcoef(zs[:, k], angle)[0, 1] for k in range(z_dim)]
    corrs_speed = [np.corrcoef(zs[:, k], speed)[0, 1] for k in range(z_dim)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(z_dim)
    for ax, corrs, label in zip(axes, [corrs_angle, corrs_speed],
                                 ["delta_angle", "speed"]):
        ax.bar(x, corrs, color=["steelblue" if c >= 0 else "tomato" for c in corrs])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("z dimension"); ax.set_ylabel("Pearson r")
        ax.set_title(f"Correlation: z_dim vs {label}")
        ax.set_xticks(x); ax.set_xticklabels([f"z{k}" for k in x], fontsize=7)

    plt.tight_layout()
    fname = os.path.join(out_dir, "action_correlation.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved → {fname}")

    # 수치 출력
    print(f"\n  Top correlations with delta_angle:")
    for k in np.argsort(np.abs(corrs_angle))[::-1][:5]:
        print(f"    z{k}: r={corrs_angle[k]:.3f}")
    print(f"  Top correlations with speed:")
    for k in np.argsort(np.abs(corrs_speed))[::-1][:5]:
        print(f"    z{k}: r={corrs_speed[k]:.3f}")


def plot_latent_traversal(model, zs, out_dir, z_dim, device, n_steps=9):
    """
    각 latent dim을 -3σ ~ +3σ 로 변화시켜 디코딩한 trajectory 시각화.
    좌표 범위는 디코딩 결과에서 동적으로 계산 (절대/상대 좌표 모두 대응).
    """
    z_mean = zs.mean(axis=0)
    z_std  = zs.std(axis=0)

    n_cols = min(z_dim, 8)   # 최대 8 dim 시각화
    vals   = np.linspace(-3, 3, n_steps)  # σ 단위

    # ── 1단계: 모든 디코딩 결과를 미리 계산해 좌표 범위 파악 ──────────────
    all_recons = {}
    all_xy = []
    for dim in range(n_cols):
        for j, v in enumerate(vals):
            z_sample = z_mean.copy()
            z_sample[dim] = z_mean[dim] + v * z_std[dim]
            with torch.no_grad():
                z_t   = torch.tensor(z_sample, dtype=torch.float32).unsqueeze(0).to(device)
                recon = model.decoder(z_t).squeeze(0).cpu().numpy()  # (MAX_EVENTS, EVENT_DIM)
            all_recons[(dim, j)] = recon
            all_xy.append(recon[:, :2])

    all_xy = np.concatenate(all_xy, axis=0)
    # 유효한 점만 (0이 아닌 행) 사용해 범위 계산
    nonzero_mask = (all_xy != 0).any(axis=1)
    if nonzero_mask.sum() > 0:
        xy_valid = all_xy[nonzero_mask]
        pad  = max(abs(xy_valid).max() * 0.15, 0.1)
        xmin, xmax = xy_valid[:, 0].min() - pad, xy_valid[:, 0].max() + pad
        ymin, ymax = xy_valid[:, 1].min() - pad, xy_valid[:, 1].max() + pad
        # 정사각형 비율 유지
        span = max(xmax - xmin, ymax - ymin)
        cx   = (xmin + xmax) / 2
        cy   = (ymin + ymax) / 2
        xmin, xmax = cx - span / 2, cx + span / 2
        ymin, ymax = cy - span / 2, cy + span / 2
    else:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1

    # ── 2단계: 시각화 ────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_cols, n_steps, figsize=(n_steps * 1.5, n_cols * 1.8))
    alphas = np.linspace(0.3, 1.0, n_steps)

    for dim in range(n_cols):
        for j, v in enumerate(vals):
            ax    = axes[dim, j]
            recon = all_recons[(dim, j)]

            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_facecolor("#1a5c1a")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"z{dim}", fontsize=7)
            if dim == 0:
                ax.set_title(f"{v:+.1f}σ", fontsize=7)

            # 궤적 점 그리기
            xys = recon[:, :2]
            ax.plot(xys[:, 0], xys[:, 1], "w-", lw=0.6, alpha=0.5)
            ax.scatter(xys[:, 0], xys[:, 1], c="yellow",
                       s=6, alpha=alphas[j], zorder=3)
            # 원점 표시 (stick_ball 기준점)
            ax.scatter([0], [0], c="cyan", s=20, marker="x", zorder=4, linewidths=1)

    plt.suptitle("Latent Dimension Traversal (−3σ → +3σ)\n[cyan × = stick_ball origin]",
                 fontsize=10)
    plt.tight_layout()
    fname = os.path.join(out_dir, "latent_traversal.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved → {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="VAE checkpoint path (*.pt)")
    parser.add_argument("--data", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, z_dim = load_model(args.ckpt, device)
    dataset      = TrajectoryDataset(args.data)

    print("\nCollecting latents ...")
    zs, actions, pocketeds, n_bounces, tags_arr = \
        collect_latents(model, dataset, device, args.max_samples)

    print("\n[1] t-SNE: pocketed")
    plot_tsne(zs, pocketeds.astype(int), "t-SNE — pocketed (1) vs miss (0)",
              os.path.join(args.out_dir, "tsne_pocketed.png"),
              cmap="RdYlGn")

    print("[2] t-SNE: n_bounces")
    plot_tsne(zs, n_bounces, "t-SNE — number of cushion bounces",
              os.path.join(args.out_dir, "tsne_bounces.png"),
              cmap="viridis")

    print("[3] t-SNE: tag (SAC vs random)")
    unique_tags = sorted(set(tags_arr))
    tag2int     = {t: i for i, t in enumerate(unique_tags)}
    tag_ints    = np.array([tag2int[t] for t in tags_arr])
    plot_tsne(zs, tag_ints, "t-SNE — policy tag",
              os.path.join(args.out_dir, "tsne_tag.png"),
              label_names=unique_tags)

    print("[4] Action correlation")
    plot_action_correlation(zs, actions, args.out_dir, z_dim)

    print("[5] Latent traversal")
    plot_latent_traversal(model, zs, args.out_dir, z_dim, device)

    print(f"\nAll plots saved → {args.out_dir}/")


if __name__ == "__main__":
    main()
