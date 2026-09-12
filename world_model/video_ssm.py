"""
world_model/video_ssm.py — SSM rollout 동영상 (GT vs Pred, 3초)

Usage:
    python world_model/video_ssm.py \
        --ckpt world_model/results/ssm_20260906_085816 \
        --data-dir world_model/data_fixeddt \
        --n-videos 5 \
        --out-dir world_model/results/ssm_videos
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
import imageio.v2 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.ssm_model import SSMWorldModel
from world_model.train_ssm import SSMDataset
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H

FPS          = int(round(1.0 / DT))   # 20
STEPS        = int(3.0 / DT)          # 60
TRAIL_ALPHA  = 0.5

CUE_GT   = "#00e5ff"
CUE_PRED = "#0099dd"
TGT_GT   = "#ffee44"
TGT_PRED = "#cc9900"
BG       = "#111811"
FELT     = "#1e4d2b"
RAIL     = "#3a2a0a"

COLL_COLORS = {0: "#ff6600", 1: "#4488ff", 2: "#aa44ff", 3: "#ff2222"}
POCKET_XY   = np.array([[0,0],[1,0],[0.5,0],[0,1],[0.5,1],[1,1]], dtype=np.float32)
POCKET_R    = 0.025


def draw_table(ax):
    ax.set_facecolor(FELT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(RAIL); spine.set_linewidth(5)
    for px, py in POCKET_XY:
        ax.add_patch(patches.Circle((px, py), POCKET_R, color="black", zorder=5))


def make_video(ep_idx, gt, pred, coll_flags, coll_types, out_path: Path):
    T = len(pred) - 1   # pred shape (T+1, 14)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Shot {ep_idx}  —  GT (left) | SSM Pred (right)",
                 color="white", fontsize=10)

    ax_gt, ax_pr = axes
    draw_table(ax_gt); draw_table(ax_pr)
    ax_gt.set_title("Ground Truth", color="white", fontsize=8, pad=3)
    ax_pr.set_title("SSM Prediction", color="white", fontsize=8, pad=3)

    # 정적 시작점
    for ax, states, cc, tc in [
        (ax_gt, gt,   CUE_GT,   TGT_GT),
        (ax_pr, pred, CUE_PRED, TGT_PRED),
    ]:
        ax.scatter([states[0,0]], [states[0,1]], color=cc,
                   s=60, zorder=6, edgecolors="white", lw=0.5)
        ax.scatter([states[0,7]], [states[0,8]], color=tc,
                   s=60, zorder=6, edgecolors="white", lw=0.5)

    # 동적 오브젝트 (line + ball)
    line_cue_gt,  = ax_gt.plot([], [], color=CUE_GT,   lw=1.5, alpha=TRAIL_ALPHA, zorder=3)
    line_tgt_gt,  = ax_gt.plot([], [], color=TGT_GT,   lw=1.5, alpha=TRAIL_ALPHA, zorder=3)
    ball_cue_gt   = ax_gt.scatter([], [], color=CUE_GT,   s=80, zorder=7, edgecolors="white", lw=0.5)
    ball_tgt_gt   = ax_gt.scatter([], [], color=TGT_GT,   s=80, zorder=7, edgecolors="white", lw=0.5)

    line_cue_pr,  = ax_pr.plot([], [], color=CUE_PRED, lw=1.5, alpha=TRAIL_ALPHA, zorder=3)
    line_tgt_pr,  = ax_pr.plot([], [], color=TGT_PRED, lw=1.5, alpha=TRAIL_ALPHA, zorder=3)
    ball_cue_pr   = ax_pr.scatter([], [], color=CUE_PRED, s=80, zorder=7, edgecolors="white", lw=0.5)
    ball_tgt_pr   = ax_pr.scatter([], [], color=TGT_PRED, s=80, zorder=7, edgecolors="white", lw=0.5)

    # 충돌 scatter (Pred 패널)
    coll_scat = ax_pr.scatter([], [], s=50, marker="x", zorder=8, lw=1.5)

    # 시간 텍스트
    txt_gt = ax_gt.text(0.02, 0.97, "", transform=ax_gt.transAxes,
                        color="white", fontsize=8, va="top")
    txt_pr = ax_pr.text(0.02, 0.97, "", transform=ax_pr.transAxes,
                        color="white", fontsize=8, va="top")

    # 오차 텍스트
    txt_err = ax_pr.text(0.98, 0.97, "", transform=ax_pr.transAxes,
                         color="#ffcc44", fontsize=8, va="top", ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    frames = []
    coll_xs, coll_ys, coll_cs = [], [], []

    for t in range(T + 1):
        # 궤적 갱신
        line_cue_gt.set_data(gt[:t+1, 0],   gt[:t+1, 1])
        line_tgt_gt.set_data(gt[:t+1, 7],   gt[:t+1, 8])
        ball_cue_gt.set_offsets([[gt[t, 0],   gt[t, 1]]])
        ball_tgt_gt.set_offsets([[gt[t, 7],   gt[t, 8]]])

        line_cue_pr.set_data(pred[:t+1, 0], pred[:t+1, 1])
        line_tgt_pr.set_data(pred[:t+1, 7], pred[:t+1, 8])
        ball_cue_pr.set_offsets([[pred[t, 0], pred[t, 1]]])
        ball_tgt_pr.set_offsets([[pred[t, 7], pred[t, 8]]])

        # 충돌 마커 누적
        if t < len(coll_flags) and coll_flags[t]:
            coll_xs.append(pred[t, 0])
            coll_ys.append(pred[t, 1])
            coll_cs.append(COLL_COLORS.get(int(coll_types[t]), "#ffffff"))
        if coll_xs:
            coll_scat.set_offsets(np.c_[coll_xs, coll_ys])
            coll_scat.set_color(coll_cs)

        # 텍스트
        elapsed = t * DT
        txt_gt.set_text(f"t={elapsed:.2f}s")

        err_c = np.sqrt(((pred[t,0]-gt[t,0])*TABLE_W)**2 +
                        ((pred[t,1]-gt[t,1])*TABLE_H)**2) * 100
        err_t = np.sqrt(((pred[t,7]-gt[t,7])*TABLE_W)**2 +
                        ((pred[t,8]-gt[t,8])*TABLE_H)**2) * 100
        txt_pr.set_text(f"t={elapsed:.2f}s")
        txt_err.set_text(f"err {(err_c+err_t)/2:.1f}cm")

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())

    plt.close(fig)

    writer = iio.get_writer(str(out_path), format="FFMPEG", fps=FPS,
                            codec="libx264", quality=7,
                            output_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default="world_model/results/ssm_20260906_085816")
    p.add_argument("--data-dir",  default="world_model/data_fixeddt")
    p.add_argument("--n-videos",  type=int, default=5)
    p.add_argument("--seed",      type=int, default=7)
    p.add_argument("--out-dir",   default="world_model/results/ssm_videos")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = Path(args.ckpt)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model = SSMWorldModel(ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded: {ckpt_dir.name}")

    ds = SSMDataset(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ep_indices = rng.choice(len(ds.episodes), args.n_videos, replace=False)

    with torch.no_grad():
        for i, ep_idx in enumerate(ep_indices):
            ep_s, ep_f, ep_t, _ = ds.episodes[ep_idx]
            T = min(STEPS, len(ep_s) - 1)
            s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
            s_hat, _, _ = model(s0, n_steps=T)
            pred = s_hat[0].cpu().numpy()
            gt   = ep_s[:T+1]
            cf   = ep_f[:T]
            ct   = ep_t[:T]
            out_path = out_dir / f"shot_{i+1:02d}_ep{ep_idx}.mp4"
            print(f"[{i+1}/{args.n_videos}] shot {ep_idx} ({T} steps)...")
            make_video(ep_idx, gt, pred, cf, ct, out_path)

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
