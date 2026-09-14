"""
world_model/visualize_ssm.py — SSM rollout 시각화

각 샷을 [GT | Pred] 좌우 패널로 비교.
Usage:
    python world_model/visualize_ssm.py \
        --ckpt world_model/results/ssm_20260906_085816 \
        --data-dir world_model/data_fixeddt \
        --n-samples 8 \
        --out world_model/results/ssm_viz.png
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.ssm_model import SSMWorldModel
from world_model.train_ssm import SSMDataset
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H

# ── 색상 ──────────────────────────────────────────────────────────────────────
CUE_GT   = "#00e5ff"
CUE_PRED = "#0077aa"
TGT_GT   = "#ffee44"
TGT_PRED = "#cc8800"
BG       = "#1a2a1a"
FELT     = "#1e4d2b"
RAIL     = "#3a2a0a"

COLL_COLORS = {
    0: "#ff6600",   # ball_ball
    1: "#4488ff",   # linear
    2: "#aa44ff",   # circular
    3: "#ff2222",   # pocket
}
COLL_NAMES = ["ball_ball", "linear", "circular", "pocket"]

CHECKPOINTS = {"0.5s": int(0.5/DT), "1.0s": int(1.0/DT),
               "2.0s": int(2.0/DT), "3.0s": int(3.0/DT)}

POCKET_XY = np.array([[0,0],[1,0],[0.5,0],[0,1],[0.5,1],[1,1]], dtype=np.float32)
POCKET_R  = 0.025   # normalized radius


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def draw_table(ax):
    ax.set_facecolor(FELT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    # 쿠션 테두리
    for spine in ax.spines.values():
        spine.set_edgecolor(RAIL)
        spine.set_linewidth(4)
    # 포켓
    for px, py in POCKET_XY:
        ax.add_patch(patches.Circle((px, py), POCKET_R,
                                    color="black", zorder=5))


def draw_trajectory(ax, states, label_prefix, cue_color, tgt_color,
                    coll_flags=None, coll_types=None, alpha=1.0):
    xs_c = states[:, 0]; ys_c = states[:, 1]
    xs_t = states[:, 7]; ys_t = states[:, 8]

    ax.plot(xs_c, ys_c, color=cue_color, lw=1.5, alpha=alpha, zorder=3)
    ax.plot(xs_t, ys_t, color=tgt_color, lw=1.5, alpha=alpha, zorder=3)

    # 시작점
    ax.scatter([xs_c[0]], [ys_c[0]], color=cue_color, s=50, zorder=6, edgecolors="white", lw=0.5)
    ax.scatter([xs_t[0]], [ys_t[0]], color=tgt_color, s=50, zorder=6, edgecolors="white", lw=0.5)

    # 충돌 마커
    if coll_flags is not None and coll_types is not None:
        for t, (flag, ctype) in enumerate(zip(coll_flags, coll_types)):
            if flag and t < len(states):
                c = COLL_COLORS.get(int(ctype), "#ffffff")
                ax.scatter([xs_c[t]], [ys_c[t]], color=c, s=35,
                           marker="x", zorder=7, lw=1.2)


def draw_time_markers(ax, states_gt, states_pred):
    T = min(len(states_gt), len(states_pred)) - 1
    for label, t in CHECKPOINTS.items():
        if t > T:
            continue
        # GT marker (open circle)
        ax.scatter([states_gt[t, 0]], [states_gt[t, 1]],
                   s=60, facecolors="none", edgecolors="white",
                   lw=1.0, zorder=8)
        # distance annotation
        err_c = np.sqrt(((states_pred[t,0]-states_gt[t,0])*TABLE_W)**2 +
                        ((states_pred[t,1]-states_gt[t,1])*TABLE_H)**2) * 100
        err_t = np.sqrt(((states_pred[t,7]-states_gt[t,7])*TABLE_W)**2 +
                        ((states_pred[t,8]-states_gt[t,8])*TABLE_H)**2) * 100
        ax.annotate(f"{label}\n{(err_c+err_t)/2:.0f}cm",
                    xy=(states_gt[t,0], states_gt[t,1]),
                    xytext=(6, 4), textcoords="offset points",
                    color="white", fontsize=5, zorder=9)


# ── 메인 시각화 ───────────────────────────────────────────────────────────────

def visualize(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    ckpt_dir = Path(args.ckpt)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model = SSMWorldModel(ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded: {ckpt_dir.name}  latent_dim={ckpt['latent_dim']}  val={ckpt['val_loss']:.4f}")

    # 데이터 로드
    ds = SSMDataset(args.data_dir, rollout_steps=60, augment=False)
    rng = np.random.default_rng(args.seed)
    ep_indices = rng.choice(len(ds.episodes), args.n_samples, replace=False)

    # 그리드 레이아웃: n_samples 행 × 2 열(GT, Pred)
    n = args.n_samples
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 2.8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"SSM Rollout — GT vs Pred  ({ckpt_dir.name})",
                 color="white", fontsize=11, y=1.001)

    rollout_steps = args.rollout_steps

    with torch.no_grad():
        for row, ep_idx in enumerate(ep_indices):
            ep_s, ep_f, ep_t = ds.episodes[ep_idx]
            T = min(rollout_steps, len(ep_s) - 1)

            s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
            s_hat, _, _ = model(s0, n_steps=T)
            pred = s_hat[0].cpu().numpy()   # (T+1, 14)
            gt   = ep_s[:T+1]               # (T+1, 14)
            cf   = ep_f[:T]
            ct   = ep_t[:T]

            ax_gt   = axes[row, 0] if n > 1 else axes[0]
            ax_pred = axes[row, 1] if n > 1 else axes[1]

            # GT
            draw_table(ax_gt)
            draw_trajectory(ax_gt, gt, "GT", CUE_GT, TGT_GT, cf, ct)
            ax_gt.set_title(f"Shot {ep_idx}  GT", color="white", fontsize=7, pad=2)

            # Pred
            draw_table(ax_pred)
            draw_trajectory(ax_pred, gt, "GT",   CUE_GT,   TGT_GT,   alpha=0.25)
            draw_trajectory(ax_pred, pred, "Pred", CUE_PRED, TGT_PRED, cf, ct)
            draw_time_markers(ax_pred, gt, pred)
            ax_pred.set_title("Pred (GT=faint)", color="white", fontsize=7, pad=2)

    # 범례
    legend_items = [
        Line2D([0],[0], color=CUE_GT,   lw=2, label="Cue GT"),
        Line2D([0],[0], color=CUE_PRED, lw=2, label="Cue Pred"),
        Line2D([0],[0], color=TGT_GT,   lw=2, label="Tgt GT"),
        Line2D([0],[0], color=TGT_PRED, lw=2, label="Tgt Pred"),
    ] + [
        Line2D([0],[0], marker="x", color=c, lw=0, markersize=6,
               label=COLL_NAMES[i])
        for i, c in COLL_COLORS.items()
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               framealpha=0.3, fontsize=7, labelcolor="white",
               facecolor=BG, edgecolor="gray",
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          default="world_model/results/ssm_20260906_085816")
    p.add_argument("--data-dir",      default="world_model/data_fixeddt")
    p.add_argument("--n-samples",     type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=60)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out",           default="world_model/results/ssm_viz.png")
    visualize(p.parse_args())


if __name__ == "__main__":
    main()
