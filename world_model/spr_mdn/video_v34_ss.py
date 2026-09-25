"""
world_model/spr_mdn/video_v34_ss.py
— 3-panel 비교: GT | Chain Oracle | Chain Pred  (full 60-step rollout)
  v33_validation 스타일, V34Model 사용.

Usage:
    python world_model/spr_mdn/video_v34_ss.py \
        --ckpt world_model/results/spr_mdn_v34_ss \
        --n-videos 5 \
        --out-dir world_model/results/v34_ss_videos
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.train_v34_ss import V34Model
from world_model.spr_mdn.spr_dataset import SPRDataset, make_balanced_val_eps
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H

FPS   = 15
STEPS = 60
TRAIL = 20
FFMPEG = "/opt/homebrew/bin/ffmpeg"

W_cm = TABLE_W * 100
H_cm = TABLE_H * 100
BALL_R = 2.85

POCKETS = [
    (0,    0),     (W_cm, 0),
    (0,    H_cm/2),(W_cm, H_cm/2),
    (0,    H_cm),  (W_cm, H_cm),
]

TYPE_NAMES  = {0: "none", 1: "bb", 2: "lin", 3: "circ", 4: "pkt"}
COLL_COLORS = {0: "#888888", 1: "#4488ff", 2: "#cc66ff", 3: "#ff8800", 4: "#ff3333"}


def to_cm(arr: np.ndarray):
    cue = arr[:, :2]  * np.array([W_cm, H_cm])
    tgt = arr[:, 7:9] * np.array([W_cm, H_cm])
    return cue, tgt


def draw_table(ax, title: str, title_color: str = "white"):
    ax.set_xlim(-3, W_cm + 3)
    ax.set_ylim(-3, H_cm + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d4a29")
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch(
        (0, 0), W_cm, H_cm, linewidth=3,
        edgecolor="#5a3008", facecolor="#1a6b3c",
        boxstyle="round,pad=0"))
    for px, py in POCKETS:
        ax.add_patch(plt.Circle((px, py), 2.5, color="black", zorder=10))
    ax.set_title(title, color=title_color, fontsize=9, pad=4)


def rollout_oracle(model, ep_s, ep_f, ep_t, device):
    """GT bounce type으로 encoder 재초기화하는 oracle rollout."""
    s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
    z  = model.encoder(s0, torch.zeros(1, dtype=torch.long, device=device))
    sc = s0
    preds = []
    with torch.no_grad():
        for t in range(STEPS):
            z     = model.transition(z, sc)
            s_hat = model.mu_head(z)
            preds.append(s_hat[0].cpu().numpy())
            sc = s_hat
            if t < len(ep_f) and ep_f[t]:
                gt_type = torch.tensor([int(ep_t[t])], dtype=torch.long, device=device)
                z = model.encoder(sc, gt_type)
    return np.stack(preds)


def rollout_pred(model, ep_s, device):
    """TypeHead 예측으로 encoder 재초기화하는 자율 rollout."""
    s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
    z  = model.encoder(s0, torch.zeros(1, dtype=torch.long, device=device))
    sc = s0
    preds, pred_types = [], []
    with torch.no_grad():
        for _ in range(STEPS):
            z       = model.transition(z, sc)
            s_hat   = model.mu_head(z)
            tp      = int(model.type_head(z).argmax(-1).item())
            preds.append(s_hat[0].cpu().numpy())
            pred_types.append(tp)
            sc = s_hat
            if tp != 0:
                z = model.encoder(sc, torch.tensor([tp], dtype=torch.long, device=device))
    return np.stack(preds), pred_types


def make_video(ep_idx, ep_s, ep_f, ep_t, pred_ora, pred_chain, pred_types,
               out_path: Path, err_ora: float, err_chain: float):
    gt = ep_s[:STEPS + 1]
    T  = STEPS

    gt_cue, gt_tgt   = to_cm(gt[1:])
    ora_cue, ora_tgt = to_cm(pred_ora)
    ch_cue,  ch_tgt  = to_cm(pred_chain)
    gt_cue0 = gt[:1, :2]  * np.array([W_cm, H_cm])
    gt_tgt0 = gt[:1, 7:9] * np.array([W_cm, H_cm])

    has_bb = bool(np.any(ep_f[:T] & (ep_t[:T] == 1)))
    tag    = "bb" if has_bb else "no-bb"

    import matplotlib as mpl
    mpl.rcParams["animation.ffmpeg_path"] = FFMPEG

    fig, axes = plt.subplots(1, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0d2b1a")
    fig.suptitle(
        f"Shot {ep_idx}  [{tag}]     "
        f"Chain Oracle err={err_ora:.1f}cm   |   Chain Pred err={err_chain:.1f}cm",
        color="white", fontsize=10)

    ax_gt, ax_ora, ax_ch = axes
    draw_table(ax_gt,  "Ground Truth",                            "white")
    draw_table(ax_ora, f"Chain Oracle\n(GT bounce type)  {err_ora:.1f}cm",  "#66ff99")
    draw_table(ax_ch,  f"Chain Pred\n(TypeHead)  {err_chain:.1f}cm",        "#66aaff")

    for ax, c0, t0 in [(ax_gt, gt_cue0[0], gt_tgt0[0]),
                        (ax_ora, gt_cue0[0], gt_tgt0[0]),
                        (ax_ch,  gt_cue0[0], gt_tgt0[0])]:
        ax.scatter([c0[0]], [c0[1]], color="white", s=40, zorder=11, marker="x", lw=1.5)
        ax.scatter([t0[0]], [t0[1]], color="gold",  s=40, zorder=11, marker="x", lw=1.5)

    def _setup_panel(ax, cc, pc):
        lc, = ax.plot([], [], color=cc, lw=1.5, alpha=0.5, zorder=4)
        lt, = ax.plot([], [], color=pc, lw=1.5, alpha=0.5, zorder=4)
        bc  = ax.add_patch(plt.Circle((0, 0), BALL_R, color=cc, zorder=8))
        bt  = ax.add_patch(plt.Circle((0, 0), BALL_R, color=pc, zorder=8))
        cs  = ax.scatter([], [], s=70, marker="x", zorder=12, lw=2)
        return lc, lt, bc, bt, cs

    p_gt  = _setup_panel(ax_gt,  "white",   "gold")
    p_ora = _setup_panel(ax_ora, "#66ff99", "#ccff66")
    p_ch  = _setup_panel(ax_ch,  "#66aaff", "#ff88cc")

    txt_t = ax_gt.text(-2, H_cm + 5, "", color="white", fontsize=8)

    gt_coll_xs, gt_coll_ys, gt_coll_cs   = [], [], []
    ora_coll_xs, ora_coll_ys              = [], []
    ch_coll_xs, ch_coll_ys, ch_coll_cs   = [], [], []

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    panels   = [p_gt, p_ora, p_ch]
    cue_arrs = [gt_cue, ora_cue, ch_cue]
    tgt_arrs = [gt_tgt, ora_tgt, ch_tgt]

    def init():
        arts = []
        for lc, lt, bc, bt, cs in panels:
            lc.set_data([], []); lt.set_data([], [])
            arts += [lc, lt, bc, bt]
        return arts

    def update(frame):
        t  = frame
        t0 = max(0, t - TRAIL)
        for (lc, lt, bc, bt, cs), cue_a, tgt_a in zip(panels, cue_arrs, tgt_arrs):
            lc.set_data(cue_a[t0:t+1, 0], cue_a[t0:t+1, 1])
            lt.set_data(tgt_a[t0:t+1, 0], tgt_a[t0:t+1, 1])
            bc.center = (cue_a[t, 0], cue_a[t, 1])
            bt.center = (tgt_a[t, 0], tgt_a[t, 1])

        if t < len(ep_f) and ep_f[t]:
            tp  = int(ep_t[t])
            col = COLL_COLORS.get(tp, "#ffffff")
            gt_coll_xs.append(gt_cue[t, 0]); gt_coll_ys.append(gt_cue[t, 1]); gt_coll_cs.append(col)
            ora_coll_xs.append(ora_cue[t, 0]); ora_coll_ys.append(ora_cue[t, 1])

        if t < len(pred_types) and pred_types[t] != 0:
            col = COLL_COLORS.get(pred_types[t], "#ffffff")
            ch_coll_xs.append(ch_cue[t, 0]); ch_coll_ys.append(ch_cue[t, 1]); ch_coll_cs.append(col)

        if gt_coll_xs:
            p_gt[4].set_offsets(np.c_[gt_coll_xs, gt_coll_ys]); p_gt[4].set_color(gt_coll_cs)
            p_ora[4].set_offsets(np.c_[ora_coll_xs, ora_coll_ys])
            p_ora[4].set_color(gt_coll_cs[:len(ora_coll_xs)])
        if ch_coll_xs:
            p_ch[4].set_offsets(np.c_[ch_coll_xs, ch_coll_ys]); p_ch[4].set_color(ch_coll_cs)

        is_bb  = (t < len(ep_f) and ep_f[t] and ep_t[t] == 1)
        suffix = "  *** BB ***" if is_bb else ""
        txt_t.set_text(f"t={t}/{T}  ({t*DT:.2f}s){suffix}")
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
    p.add_argument("--ckpt",     default="world_model/results/spr_mdn_v34_ss")
    p.add_argument("--data-dir", default="world_model/data_fixeddt")
    p.add_argument("--n-videos", type=int, default=5)
    p.add_argument("--steps",    type=int, default=None)
    p.add_argument("--seed",     type=int, default=7)
    p.add_argument("--out-dir",  default="world_model/results/v34_ss_videos")
    args = p.parse_args()

    global STEPS
    if args.steps is not None:
        STEPS = args.steps

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt   = torch.load(Path(args.ckpt) / "best.pt", map_location=device, weights_only=False)
    model  = V34Model().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded epoch={ckpt['epoch']}  err={ckpt['mean_err']:.2f}cm  STEPS={STEPS}")

    dataset      = SPRDataset(args.data_dir)
    rng          = np.random.default_rng(0)
    perm         = rng.permutation(len(dataset.episodes))
    n_val        = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all  = [dataset.episodes[i] for i in perm[:n_val]]
    balanced_val = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    valid = [(i, ep) for i, ep in enumerate(balanced_val) if len(ep[0]) >= STEPS + 1]

    rng2   = np.random.default_rng(args.seed)
    chosen = rng2.choice(len(valid), min(args.n_videos, len(valid)), replace=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _mean_err(pred_arr, ep_s):
        gt = ep_s[1:STEPS+1]
        ce = np.sqrt(((pred_arr[:,0]-gt[:,0])*W_cm)**2 + ((pred_arr[:,1]-gt[:,1])*H_cm)**2)
        te = np.sqrt(((pred_arr[:,7]-gt[:,7])*W_cm)**2 + ((pred_arr[:,8]-gt[:,8])*H_cm)**2)
        return float(((ce+te)/2).mean())

    for rank, ci in enumerate(chosen):
        i, (ep_s, ep_f, ep_t, _, _) = valid[ci]
        pred_ora               = rollout_oracle(model, ep_s, ep_f, ep_t, device)
        pred_chain, pred_types = rollout_pred(model, ep_s, device)
        err_ora   = _mean_err(pred_ora,   ep_s)
        err_chain = _mean_err(pred_chain, ep_s)
        print(f"[{rank+1}/{len(chosen)}] ep{i}  oracle={err_ora:.1f}cm  pred={err_chain:.1f}cm")
        out_path = out_dir / f"val_{rank+1:02d}_ep{i}.mp4"
        make_video(i, ep_s, ep_f, ep_t, pred_ora, pred_chain, pred_types,
                   out_path, err_ora, err_chain)

    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
