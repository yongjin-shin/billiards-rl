"""
world_model/spr_mdn/video_v33.py — V33 세그먼트 모델 영상 생성

타입별(cue_strike / bb / lin_cush / circ_cush / pocket)로 N개씩 GT vs Pred 영상 출력.

Usage:
    python world_model/spr_mdn/video_v33.py \
        --ckpt world_model/results/spr_mdn_v33_segment \
        --data-dir world_model/data_fixeddt \
        --n-per-type 3 \
        --out-dir world_model/results/v33_videos \
        2>&1 | tee /tmp/v33_video.log
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import imageio.v2 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.train_v33_segment import V33Model
from world_model.spr_mdn.spr_dataset import SPRDataset, SegmentDataset, make_balanced_val_eps
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H

FPS          = int(round(1.0 / DT))   # 20
TRAIL_ALPHA  = 0.5

CUE_GT   = "#00e5ff"
CUE_PRED = "#0099dd"
TGT_GT   = "#ffee44"
TGT_PRED = "#cc9900"
BG       = "#111811"
FELT     = "#1e4d2b"
RAIL     = "#3a2a0a"

TYPE_NAMES  = ["cue_strike", "bb", "lin_cush", "circ_cush", "pocket"]
TYPE_COLORS = ["#aaffaa", "#4488ff", "#aa44ff", "#ff6600", "#ff2222"]
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


def make_video(seg_idx: int, seg_type: int, gt: np.ndarray, pred: np.ndarray,
               out_path: Path):
    """
    gt, pred: (L+1, 14) — index 0 = 공유 시작 상태
    """
    T    = len(pred) - 1
    name = TYPE_NAMES[seg_type]
    col  = TYPE_COLORS[seg_type]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Segment #{seg_idx}  |  type: {name}  |  GT (left) vs V33 Pred (right)",
                 color=col, fontsize=10, fontweight="bold")

    ax_gt, ax_pr = axes
    draw_table(ax_gt); draw_table(ax_pr)
    ax_gt.set_title("Ground Truth",   color="white", fontsize=9, pad=3)
    ax_pr.set_title("V33 Prediction", color="white", fontsize=9, pad=3)

    # 시작 마커
    for ax, states, cc, tc in [
        (ax_gt, gt,   CUE_GT,   TGT_GT),
        (ax_pr, pred, CUE_PRED, TGT_PRED),
    ]:
        ax.scatter([states[0, 0]], [states[0, 1]], color=cc,
                   s=60, zorder=6, edgecolors="white", lw=0.5, marker="o")
        ax.scatter([states[0, 7]], [states[0, 8]], color=tc,
                   s=60, zorder=6, edgecolors="white", lw=0.5, marker="o")

    # 동적 오브젝트
    line_cue_gt, = ax_gt.plot([], [], color=CUE_GT,   lw=1.5, alpha=TRAIL_ALPHA)
    line_tgt_gt, = ax_gt.plot([], [], color=TGT_GT,   lw=1.5, alpha=TRAIL_ALPHA)
    ball_cue_gt   = ax_gt.scatter([], [], color=CUE_GT,   s=90, zorder=7, edgecolors="white", lw=0.5)
    ball_tgt_gt   = ax_gt.scatter([], [], color=TGT_GT,   s=90, zorder=7, edgecolors="white", lw=0.5)

    line_cue_pr, = ax_pr.plot([], [], color=CUE_PRED, lw=1.5, alpha=TRAIL_ALPHA)
    line_tgt_pr, = ax_pr.plot([], [], color=TGT_PRED, lw=1.5, alpha=TRAIL_ALPHA)
    ball_cue_pr   = ax_pr.scatter([], [], color=CUE_PRED, s=90, zorder=7, edgecolors="white", lw=0.5)
    ball_tgt_pr   = ax_pr.scatter([], [], color=TGT_PRED, s=90, zorder=7, edgecolors="white", lw=0.5)

    txt_time = ax_gt.text(0.02, 0.97, "", transform=ax_gt.transAxes,
                          color="white", fontsize=9, va="top")
    txt_err  = ax_pr.text(0.98, 0.97, "", transform=ax_pr.transAxes,
                          color="#ffcc44", fontsize=9, va="top", ha="right")
    txt_seg  = ax_pr.text(0.02, 0.97, f"{name}", transform=ax_pr.transAxes,
                          color=col, fontsize=9, va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    frames = []
    for t in range(T + 1):
        line_cue_gt.set_data(gt[:t+1, 0],   gt[:t+1, 1])
        line_tgt_gt.set_data(gt[:t+1, 7],   gt[:t+1, 8])
        ball_cue_gt.set_offsets([[gt[t, 0],  gt[t, 1]]])
        ball_tgt_gt.set_offsets([[gt[t, 7],  gt[t, 8]]])

        line_cue_pr.set_data(pred[:t+1, 0], pred[:t+1, 1])
        line_tgt_pr.set_data(pred[:t+1, 7], pred[:t+1, 8])
        ball_cue_pr.set_offsets([[pred[t, 0], pred[t, 1]]])
        ball_tgt_pr.set_offsets([[pred[t, 7], pred[t, 8]]])

        elapsed = t * DT
        txt_time.set_text(f"t={elapsed:.2f}s  ({t}/{T})")

        err_c = np.sqrt(((pred[t,0]-gt[t,0])*TABLE_W)**2 +
                        ((pred[t,1]-gt[t,1])*TABLE_H)**2) * 100
        err_t = np.sqrt(((pred[t,7]-gt[t,7])*TABLE_W)**2 +
                        ((pred[t,8]-gt[t,8])*TABLE_H)**2) * 100
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
    print(f"  Saved → {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default="world_model/results/spr_mdn_v33_segment")
    p.add_argument("--data-dir",    default="world_model/data_fixeddt")
    p.add_argument("--n-per-type",  type=int, default=3)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out-dir",     default="world_model/results/v33_videos")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(Path(args.ckpt) / "best.pt", map_location=device, weights_only=False)
    model = V33Model().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded ep{ckpt['epoch']}  mean_err={ckpt['mean_err']:.2f}cm")

    dataset = SPRDataset(args.data_dir)
    rng     = np.random.default_rng(0)
    perm    = rng.permutation(len(dataset.episodes))
    n_val   = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all = [dataset.episodes[i] for i in perm[:n_val]]
    balanced_val = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)

    seg_ds = SegmentDataset(balanced_val, min_len=2, augment=False)

    # 타입별로 인덱스 분류
    rng2 = np.random.default_rng(args.seed)
    by_type: dict = {i: [] for i in range(len(TYPE_NAMES))}
    for idx, (_, _, st) in enumerate(seg_ds.segments):
        by_type[st].append(idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for t_id, t_name in enumerate(TYPE_NAMES):
        indices = by_type[t_id]
        if not indices:
            print(f"[{t_name}] no segments, skip")
            continue
        chosen = rng2.choice(indices, min(args.n_per_type, len(indices)), replace=False)
        print(f"\n[{t_name}] {len(chosen)} videos")

        for rank, idx in enumerate(chosen):
            seg_s, seg_t, seg_type_start, seg_len = seg_ds[idx]
            L  = int(seg_len)
            st = seg_type_start.unsqueeze(0).to(device)
            s0 = seg_s[0:1].to(device)

            with torch.no_grad():
                z     = model.encoder(s0, st)
                s_cur = s0
                preds = [s0[0].cpu().numpy()]
                for _ in range(L):
                    z     = model.transition(z, s_cur)
                    s_hat = model.mu_head(z)
                    preds.append(s_hat[0].cpu().numpy())
                    s_cur = s_hat

            pred = np.stack(preds)          # (L+1, 14)
            gt   = seg_s.numpy()            # (L+1, 14)

            fname = out_dir / f"{t_id:02d}_{t_name}_{rank+1:02d}.mp4"
            make_video(idx, t_id, gt, pred, fname)
            total += 1

    print(f"\nDone. {total} videos → {out_dir}/")


if __name__ == "__main__":
    main()
