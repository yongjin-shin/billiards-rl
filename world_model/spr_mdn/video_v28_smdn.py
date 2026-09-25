"""
world_model/spr_mdn/video_v28_smdn.py — v28_smdn 예측 영상

2-panel: GT | Pred (argmax component mean)

Usage:
    python world_model/spr_mdn/video_v28_smdn.py \
        --ckpt world_model/results/spr_mdn_v28_dt01 \
        --data-dir world_model/data_dt01 \
        --n-videos 8 \
        --out-dir world_model/results/v28_smdn_videos
"""

import os, sys, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.train_v28_smdn import V28SMDNModel
from world_model.spr_mdn.spr_dataset import SPRDataset, make_balanced_val_eps
from world_model.wm_predictor import TABLE_W, TABLE_H

FPS    = 15
STEPS  = 60
DT     = 0.05
TRAIL  = 20
FFMPEG = "/opt/homebrew/bin/ffmpeg"

W_cm = TABLE_W * 100
H_cm = TABLE_H * 100
BALL_R = 2.85

POCKETS = [
    (0,    0),      (W_cm, 0),
    (0,    H_cm/2), (W_cm, H_cm/2),
    (0,    H_cm),   (W_cm, H_cm),
]

TYPE_NAMES  = {0: "none", 1: "bb", 2: "lin", 3: "circ", 4: "pkt", 5: "s_r", 6: "stop"}
COLL_COLORS = {0: "#888888", 1: "#4488ff", 2: "#cc66ff", 3: "#ff8800", 4: "#ff3333",
               5: "#44ffaa", 6: "#ffff44"}


def to_cm(arr):
    cue = arr[:, :2]  * np.array([W_cm, H_cm])
    tgt = arr[:, 7:9] * np.array([W_cm, H_cm])
    return cue, tgt


def rollout_pred(model, ep_s, device):
    s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
    preds, pred_types = [], []
    with torch.no_grad():
        z = model.encoder(s0)
        for _ in range(STEPS):
            z = model.transition(z)
            s_hat = model.decoder.decode_det(z)
            tp    = int(model.type_head(z).argmax(-1).item())
            preds.append(s_hat[0].cpu().numpy())
            pred_types.append(tp)
    return np.stack(preds), pred_types


def draw_table(ax, title, title_color="white"):
    ax.set_xlim(-3, W_cm + 3)
    ax.set_ylim(-3, H_cm + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d4a29")
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch(
        (0, 0), W_cm, H_cm, linewidth=3,
        edgecolor="#5a3008", facecolor="#1a6b3c", boxstyle="round,pad=0"))
    for px, py in POCKETS:
        ax.add_patch(plt.Circle((px, py), 2.5, color="black", zorder=10))
    ax.set_title(title, color=title_color, fontsize=9, pad=4)


def make_video(ep_idx, ep_s, ep_f, ep_t, pred_arr, pred_types, out_path, err):
    T = STEPS
    gt        = ep_s[:T + 1]
    gt_cue, gt_tgt   = to_cm(gt[1:])
    pred_cue, pred_tgt = to_cm(pred_arr)
    gt_cue0 = gt[:1, :2]  * np.array([W_cm, H_cm])
    gt_tgt0 = gt[:1, 7:9] * np.array([W_cm, H_cm])

    has_bb = bool(np.any(ep_f[:T] & (ep_t[:T] == 1)))
    tag    = "bb" if has_bb else "no-bb"

    import matplotlib as mpl
    mpl.rcParams["animation.ffmpeg_path"] = FFMPEG

    fig, (ax_gt, ax_pr) = plt.subplots(1, 2, figsize=(13, 11))
    fig.patch.set_facecolor("#0d2b1a")
    fig.suptitle(f"v28_smdn  Shot {ep_idx}  [{tag}]  err={err:.1f}cm",
                 color="white", fontsize=10)

    draw_table(ax_gt, "Ground Truth",            "white")
    draw_table(ax_pr, f"v28_smdn Pred  {err:.1f}cm", "#66aaff")

    for ax in (ax_gt, ax_pr):
        ax.scatter([gt_cue0[0,0]], [gt_cue0[0,1]], color="white", s=40, zorder=11, marker="x", lw=1.5)
        ax.scatter([gt_tgt0[0,0]], [gt_tgt0[0,1]], color="gold",  s=40, zorder=11, marker="x", lw=1.5)

    # GT static trail + collision markers
    ax_gt.plot(gt_cue[:, 0], gt_cue[:, 1], color="white", lw=0.6, alpha=0.15, zorder=2)
    ax_gt.plot(gt_tgt[:, 0], gt_tgt[:, 1], color="gold",  lw=0.6, alpha=0.15, zorder=2)
    ax_pr.plot(pred_cue[:, 0], pred_cue[:, 1], color="#66aaff", lw=0.6, alpha=0.15, zorder=2)
    ax_pr.plot(pred_tgt[:, 0], pred_tgt[:, 1], color="#ff88cc", lw=0.6, alpha=0.15, zorder=2)

    for t in range(T):
        if t < len(ep_f) and ep_f[t]:
            tp  = int(ep_t[t])
            col = COLL_COLORS.get(tp, "#fff")
            ax_gt.scatter([gt_cue[t, 0]], [gt_cue[t, 1]], color=col, s=60, marker="o", zorder=8, alpha=0.7)
            ax_gt.text(gt_cue[t, 0]+1.5, gt_cue[t, 1]+1.5, TYPE_NAMES.get(tp,"?"), color=col, fontsize=5, zorder=9)
        if t < len(pred_types) and pred_types[t] != 0:
            tp  = pred_types[t]
            col = COLL_COLORS.get(tp, "#fff")
            ax_pr.scatter([pred_cue[t, 0]], [pred_cue[t, 1]], color=col, s=60, marker="o", zorder=8, alpha=0.7)
            ax_pr.text(pred_cue[t, 0]+1.5, pred_cue[t, 1]+1.5, TYPE_NAMES.get(tp,"?"), color=col, fontsize=5, zorder=9)

    # dynamic elements
    lc_gt, = ax_gt.plot([], [], "w-",  lw=1.8, alpha=0.6, zorder=4)
    lt_gt, = ax_gt.plot([], [], "y-",  lw=1.8, alpha=0.6, zorder=4)
    bc_gt  = ax_gt.add_patch(plt.Circle((0,0), BALL_R, color="white", zorder=9))
    bt_gt  = ax_gt.add_patch(plt.Circle((0,0), BALL_R, color="gold",  zorder=9))

    lc_pr, = ax_pr.plot([], [], color="#66aaff", lw=1.8, alpha=0.6, zorder=4)
    lt_pr, = ax_pr.plot([], [], color="#ff88cc", lw=1.8, alpha=0.6, zorder=4)
    bc_pr  = ax_pr.add_patch(plt.Circle((0,0), BALL_R, color="#66aaff", zorder=9))
    bt_pr  = ax_pr.add_patch(plt.Circle((0,0), BALL_R, color="#ff88cc", zorder=9))

    txt_t  = ax_gt.text(1, H_cm + 4, "", color="white", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def init():
        for artist in (lc_gt, lt_gt, lc_pr, lt_pr):
            artist.set_data([], [])
        return []

    def update(frame):
        t  = frame
        t0 = max(0, t - TRAIL)
        lc_gt.set_data(gt_cue[t0:t+1, 0], gt_cue[t0:t+1, 1])
        lt_gt.set_data(gt_tgt[t0:t+1, 0], gt_tgt[t0:t+1, 1])
        bc_gt.center = (gt_cue[t, 0], gt_cue[t, 1])
        bt_gt.center = (gt_tgt[t, 0], gt_tgt[t, 1])
        lc_pr.set_data(pred_cue[t0:t+1, 0], pred_cue[t0:t+1, 1])
        lt_pr.set_data(pred_tgt[t0:t+1, 0], pred_tgt[t0:t+1, 1])
        bc_pr.center = (pred_cue[t, 0], pred_cue[t, 1])
        bt_pr.center = (pred_tgt[t, 0], pred_tgt[t, 1])
        txt_t.set_text(f"t={t}/{T}  ({t*DT:.2f}s)")
        return []

    ani = FuncAnimation(fig, update, frames=T,
                        init_func=init, blit=False, interval=1000/FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=1500,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    ani.save(str(out_path), writer=writer, dpi=100,
             savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  → {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default="world_model/results/spr_mdn_v28_dt01")
    p.add_argument("--data-dir", default="world_model/data_dt01")
    p.add_argument("--n-videos", type=int, default=8)
    p.add_argument("--seed",     type=int, default=7)
    p.add_argument("--out-dir",  default="world_model/results/v28_smdn_videos")
    p.add_argument("--steps",    type=int, default=None)
    p.add_argument("--fps",      type=int, default=None)
    p.add_argument("--bb-only",  action="store_true")
    args = p.parse_args()

    global STEPS, FPS, DT
    # DT from metadata
    meta_path = Path(args.data_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        DT = float((meta[-1] if isinstance(meta, list) else meta).get("dt", 0.05))
    if args.steps:
        STEPS = args.steps
    if args.fps:
        FPS = args.fps

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw    = torch.load(Path(args.ckpt) / "best.pt", map_location=device, weights_only=False)
    model  = V28SMDNModel(K=raw["K"]).to(device)
    model.load_state_dict(raw["state"])
    model.eval()
    print(f"Loaded ep{raw['epoch']}  err={raw['mean_err']:.2f}cm  STEPS={STEPS}")

    dataset      = SPRDataset(args.data_dir)
    rng          = np.random.default_rng(0)
    perm         = rng.permutation(len(dataset.episodes))
    n_val        = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all  = [dataset.episodes[i] for i in perm[:n_val]]
    balanced_val = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    valid        = [(i, ep) for i, ep in enumerate(balanced_val) if len(ep[0]) >= STEPS + 1]

    rng2   = np.random.default_rng(args.seed)
    has_bb  = [(i, ep) for i, ep in valid if np.any(ep[1][:STEPS] & (ep[2][:STEPS] == 1))]
    no_bb   = [(i, ep) for i, ep in valid if not np.any(ep[1][:STEPS] & (ep[2][:STEPS] == 1))]
    if args.bb_only:
        n_bb   = min(args.n_videos, len(has_bb))
        chosen = [has_bb[j] for j in rng2.choice(len(has_bb), n_bb, replace=False)]
    else:
        n_bb   = min(args.n_videos // 2, len(has_bb))
        n_nbb  = min(args.n_videos - n_bb, len(no_bb))
        chosen = ([has_bb[j] for j in rng2.choice(len(has_bb), n_bb,  replace=False)] +
                  [no_bb[j]  for j in rng2.choice(len(no_bb),  n_nbb, replace=False)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (i, (ep_s, ep_f, ep_t, _, _)) in enumerate(chosen):
        pred_arr, pred_types = rollout_pred(model, ep_s, device)
        gt  = ep_s[1:STEPS + 1]
        ce  = np.sqrt(((pred_arr[:,0]-gt[:,0])*W_cm)**2 + ((pred_arr[:,1]-gt[:,1])*H_cm)**2)
        te  = np.sqrt(((pred_arr[:,7]-gt[:,7])*W_cm)**2 + ((pred_arr[:,8]-gt[:,8])*H_cm)**2)
        err = float(((ce + te) / 2).mean())
        tag = "bb" if np.any(ep_f[:STEPS] & (ep_t[:STEPS] == 1)) else "nbb"
        print(f"[{rank+1}/{len(chosen)}] ep{i}  [{tag}]  err={err:.1f}cm")
        out_path = out_dir / f"v28_{rank+1:02d}_ep{i}.mp4"
        make_video(i, ep_s, ep_f, ep_t, pred_arr, pred_types, out_path, err)

    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
