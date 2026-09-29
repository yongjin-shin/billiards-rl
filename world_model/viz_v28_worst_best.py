"""
v28_smdn — Ball-Ball(bb) case worst/best top5 video

Usage:
    cd /Users/yj/Documents/billiards-rl
    python /tmp/viz_v28_worst_best.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

import torch

sys.path.insert(0, "/Users/yj/Documents/billiards-rl")
from world_model.fixeddt_model import LATENT_DIM
from world_model.spr_mdn.train_v28_smdn import V28SMDNModel
from world_model.spr_mdn.spr_dataset import SPRDataset, make_balanced_val_eps
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.generate_data_fixeddt import DT

DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"
OUT     = Path("/tmp/viz_v28_bb")
OUT.mkdir(exist_ok=True)
STEPS   = 60   # rollout length to evaluate
FFMPEG  = "/opt/homebrew/bin/ffmpeg"

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading v28_smdn...")
model = V28SMDNModel(K=5).to(DEVICE)
ck = torch.load("world_model/results/spr_mdn_v28_smdn/best.pt",
                map_location=DEVICE, weights_only=False)
model.load_state_dict(ck.get("state", ck), strict=False)
model.eval()

# ── Load validation data ──────────────────────────────────────────────────────
print("Loading data...")
ds  = SPRDataset("world_model/data_fixeddt")
rng = np.random.default_rng(0)
perm = rng.permutation(len(ds.episodes))
n_val = max(200, int(len(ds.episodes) * 0.1))
val_eps_all = [ds.episodes[i] for i in perm[:n_val]]
balanced_val = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)

# bb case: ep_f & ep_t==1 (ball-ball collision)
bb_eps = [ep for ep in balanced_val
          if len(ep[0]) >= STEPS + 1
          and bool(np.any(ep[1][:STEPS] & (ep[2][:STEPS] == 1)))]
print(f"  BB episodes (len≥{STEPS}): {len(bb_eps)}")

# ── Run rollout on all bb episodes ───────────────────────────────────────────
s0_np = np.stack([ep[0][0] for ep in bb_eps]).astype(np.float32)
with torch.no_grad():
    s0_t = torch.from_numpy(s0_np).to(DEVICE)
    pred, _ = model.rollout_det(s0_t, STEPS)
    pred = pred.cpu().numpy()   # (N, STEPS, 14)

# per-episode mean error (cm)
def ep_err(pred_i, ep):
    gt = ep[0][1:STEPS+1]
    p  = pred_i
    ce = np.sqrt(((p[:,0]-gt[:,0])*TABLE_W)**2 + ((p[:,1]-gt[:,1])*TABLE_H)**2)
    te = np.sqrt(((p[:,7]-gt[:,7])*TABLE_W)**2 + ((p[:,8]-gt[:,8])*TABLE_H)**2)
    return float(((ce+te)/2*100).mean())

errs = [ep_err(pred[i], bb_eps[i]) for i in range(len(bb_eps))]
errs = np.array(errs)

# sort: worst (highest err) and best (lowest err)
sorted_idx = np.argsort(errs)
best_idx  = sorted_idx[:5].tolist()
worst_idx = sorted_idx[-5:][::-1].tolist()

print("\nBest 5 (bb):  " + ", ".join(f"{errs[i]:.1f}cm" for i in best_idx))
print("Worst 5 (bb): " + ", ".join(f"{errs[i]:.1f}cm" for i in worst_idx))


# ── Animation helper ──────────────────────────────────────────────────────────
FPS      = 15      # frames per second in output video
SKIP     = 1       # render every N timesteps (1=all)
TRAIL    = 20      # trail length in frames

def make_video(ep_idx: int, label: str, rank: int, tag: str):
    ep  = bb_eps[ep_idx]
    ep_s, ep_f, ep_t = ep[0], ep[1], ep[2]
    pr  = pred[ep_idx]            # (STEPS, 14)
    gt  = ep_s[1:STEPS+1]        # (STEPS, 14)
    err = errs[ep_idx]

    # cm coordinates
    def to_cm(arr):
        """(T, 14) → cue_xy (T,2) cm, tgt_xy (T,2) cm"""
        cue = arr[:, :2] * np.array([TABLE_W, TABLE_H]) * 100
        tgt = arr[:, 7:9] * np.array([TABLE_W, TABLE_H]) * 100
        return cue, tgt

    gt_cue, gt_tgt   = to_cm(gt)
    pr_cue, pr_tgt   = to_cm(pr)

    W_cm = TABLE_W * 100
    H_cm = TABLE_H * 100

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(-3, W_cm + 3)
    ax.set_ylim(-3, H_cm + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a6b3c")    # 당구대 초록

    # 테이블 테두리
    for spine in ax.spines.values():
        spine.set_visible(False)
    table_rect = patches.FancyBboxPatch(
        (0, 0), W_cm, H_cm, linewidth=3,
        edgecolor="#5a3008", facecolor="#1a6b3c",
        boxstyle="round,pad=0")
    ax.add_patch(table_rect)

    # 포켓 (6개: 네 모서리 + 양쪽 중간)
    pocket_positions = [
        (0, 0), (W_cm/2, 0), (W_cm, 0),
        (0, H_cm), (W_cm/2, H_cm), (W_cm, H_cm),
    ]
    for px, py in pocket_positions:
        pocket = plt.Circle((px, py), 2.5, color="black", zorder=10)
        ax.add_patch(pocket)

    # 이동 경로 선 (dim)
    gt_cue_line,  = ax.plot([], [], "w-",  alpha=0.25, lw=1.2, label="GT cue")
    gt_tgt_line,  = ax.plot([], [], "y-",  alpha=0.25, lw=1.2, label="GT tgt")
    pr_cue_line,  = ax.plot([], [], "c--", alpha=0.55, lw=1.5, label="Pred cue")
    pr_tgt_line,  = ax.plot([], [], "m--", alpha=0.55, lw=1.5, label="Pred tgt")

    # 현재 공 (원)
    gt_cue_ball  = plt.Circle((0,0), 2.85, color="white",     zorder=8)
    gt_tgt_ball  = plt.Circle((0,0), 2.85, color="gold",      zorder=8)
    pr_cue_ball  = plt.Circle((0,0), 2.85, color="cyan",      zorder=9, fill=False, lw=2)
    pr_tgt_ball  = plt.Circle((0,0), 2.85, color="magenta",   zorder=9, fill=False, lw=2)
    for c in [gt_cue_ball, gt_tgt_ball, pr_cue_ball, pr_tgt_ball]:
        ax.add_patch(c)

    # 충돌 표시 (bb 타이밍)
    bb_times = np.where(ep_f[:STEPS] & (ep_t[:STEPS] == 1))[0]

    title_txt = ax.set_title(
        f"[{tag.upper()}  #{rank+1}]  BB episode  err={err:.1f}cm\n"
        f"t=0/{STEPS}  (white=GT cue  gold=GT tgt  cyan=pred cue  magenta=pred tgt)",
        fontsize=9, color="white")
    ax.set_facecolor("#0d4a29")
    fig.patch.set_facecolor("#0d2b1a")
    ax.tick_params(colors="white")

    # 범례
    ax.legend(loc="upper right", fontsize=7, framealpha=0.4,
              labelcolor="white", facecolor="#0d2b1a")

    # BB 충돌 시점 vertical marker (plot 바깥 시간축 영역에서 표시)
    # → 타이틀에 표시로 대체

    n_frames = STEPS // SKIP

    def init():
        gt_cue_line.set_data([], [])
        gt_tgt_line.set_data([], [])
        pr_cue_line.set_data([], [])
        pr_tgt_line.set_data([], [])
        return (gt_cue_line, gt_tgt_line, pr_cue_line, pr_tgt_line,
                gt_cue_ball, gt_tgt_ball, pr_cue_ball, pr_tgt_ball)

    def update(frame):
        t = frame * SKIP
        t_start = max(0, t - TRAIL)

        gt_cue_line.set_data(gt_cue[t_start:t+1, 0], gt_cue[t_start:t+1, 1])
        gt_tgt_line.set_data(gt_tgt[t_start:t+1, 0], gt_tgt[t_start:t+1, 1])
        pr_cue_line.set_data(pr_cue[t_start:t+1, 0], pr_cue[t_start:t+1, 1])
        pr_tgt_line.set_data(pr_tgt[t_start:t+1, 0], pr_tgt[t_start:t+1, 1])

        gt_cue_ball.center = (gt_cue[t, 0], gt_cue[t, 1])
        gt_tgt_ball.center = (gt_tgt[t, 0], gt_tgt[t, 1])
        pr_cue_ball.center = (pr_cue[t, 0], pr_cue[t, 1])
        pr_tgt_ball.center = (pr_tgt[t, 0], pr_tgt[t, 1])

        is_bb = any(abs(t - bt) <= SKIP for bt in bb_times)
        suffix = "  *** BB COLLISION ***" if is_bb else ""
        t_sec  = t * DT
        title_txt.set_text(
            f"[{tag.upper()}  #{rank+1}]  BB episode  err={err:.1f}cm\n"
            f"t={t}/{STEPS}  ({t_sec:.2f}s){suffix}")

        return (gt_cue_line, gt_tgt_line, pr_cue_line, pr_tgt_line,
                gt_cue_ball, gt_tgt_ball, pr_cue_ball, pr_tgt_ball, title_txt)

    ani = FuncAnimation(fig, update, frames=n_frames,
                        init_func=init, blit=True, interval=1000/FPS)

    fname = OUT / f"{tag}_{rank+1:02d}_err{err:.0f}cm.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=1200,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    ani.save(str(fname), writer=writer, dpi=100,
             savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ── Generate videos ───────────────────────────────────────────────────────────
import matplotlib as mpl
mpl.rcParams["animation.ffmpeg_path"] = FFMPEG

print("\n--- WORST 5 ---")
for rank, idx in enumerate(worst_idx):
    make_video(idx, "worst", rank, "worst")

print("\n--- BEST 5 ---")
for rank, idx in enumerate(best_idx):
    make_video(idx, "best", rank, "best")

print(f"\nAll done → {OUT}")
