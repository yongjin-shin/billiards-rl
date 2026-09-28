"""
world_model/viz_rssm.py

RSSM rollout visualization — worst/best top-5 video

GT(white/gold) vs Predicted(cyan/magenta) trajectory comparison.
Teacher forcing: correct pre-collision states, predicted velocity deltas.

Usage:
    python world_model/viz_rssm.py
    python world_model/viz_rssm.py --ckpt world_model/results/rssm_v2/best.pt
    python world_model/viz_rssm.py --out-dir /tmp/viz_rssm
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

import torch
import torch.nn.functional as F

from world_model.ball_motion import DEFAULT_FRICTION
from world_model.pure_physics import evolve_ball_motion, SLIDING
from world_model.rssm_model import RSSMModel, H_DIM, EVENT_BALL_BALL
from world_model.rssm_dataset import ShotData, generate_shot_data, load_dataset
from world_model.rssm_rollout import make_node, make_edge
from simulator import BilliardsEnv


# ── Constants ─────────────────────────────────────────────────────────────────

TABLE_W = 0.9906   # m (width)
TABLE_H = 1.9812   # m (length)
PARAMS  = DEFAULT_FRICTION
BALL_IDS = ["cue", "1"]

# Correct pocket positions: 4 corners + 2 long-side centers
POCKET_POS = [
    (0,       0),
    (TABLE_W, 0),
    (0,       TABLE_H),
    (TABLE_W, TABLE_H),
    (0,       TABLE_H / 2),   # left center
    (TABLE_W, TABLE_H / 2),   # right center
]
POCKET_R = 0.062   # m

FPS   = 20
TRAIL = 40         # trail length in frames
DT_RENDER = 0.025  # seconds between rendered frames


# ── Helpers ───────────────────────────────────────────────────────────────────

def _evolve(rvw: np.ndarray, dt: float) -> np.ndarray:
    rvw_new, _ = evolve_ball_motion(
        SLIDING, np.array(rvw, dtype=np.float64),
        R=PARAMS.R, m=PARAMS.m, u_s=PARAMS.u_s, u_sp=PARAMS.u_sp,
        u_r=PARAMS.u_r, g=PARAMS.g, t=dt,
    )
    return rvw_new


def shot_rmse(model: RSSMModel, shot: ShotData, device: torch.device) -> float:
    h    = model.init_hidden(shot.n_balls, device)
    mses = []
    with torch.no_grad():
        for k, ev in enumerate(shot.event_steps):
            ni = ev.node_i.to(device)
            nj = ev.node_j.to(device) if ev.node_j is not None else None
            eg = ev.edge.to(device)   if ev.edge   is not None else None
            if ev.event_type == EVENT_BALL_BALL and ev.ball_j is not None:
                h, di, dj, _, _ = model.step_ball_ball(h, ev.ball_i, ev.ball_j, ni, nj, eg)
            else:
                h, di, _ = model.step_single(h, ev.ball_i, ni, ev.normal.to(device))
                dj = None
            mses.append(F.mse_loss(di, shot.gt_deltas_i[k].to(device)).item())
            if dj is not None and shot.gt_deltas_j[k] is not None:
                mses.append(F.mse_loss(dj, shot.gt_deltas_j[k].to(device)).item())
    return float(np.sqrt(np.mean(mses))) if mses else float("inf")


def reconstruct(model: RSSMModel, shot: ShotData, device: torch.device):
    """
    Teacher-forcing trajectory reconstruction.

    Returns list of frame dicts:
        {'gt': {ball_idx: (x,y)}, 'pred': {ball_idx: (x,y)}, 'is_event': bool}
    """
    h = model.init_hidden(shot.n_balls, device)

    # Current rvw per ball: GT is snapped to exact collision state each event;
    # pred accumulates (velocity drift, position drift from predicted motion).
    gt_cur   = {}
    pred_cur = {}
    frames   = []

    with torch.no_grad():
        for k, ev in enumerate(shot.event_steps):
            bi = ev.ball_i
            bj = ev.ball_j
            dt = shot.dt_to_next[k]

            gt_pre_i = shot.raw_rvws_i[k]
            gt_pre_j = (shot.raw_rvws_j[k]
                        if bj is not None and shot.raw_rvws_j[k] is not None
                        else None)

            # Initialise balls on first appearance
            if bi not in gt_cur:
                gt_cur[bi]   = gt_pre_i.copy()
                pred_cur[bi] = gt_pre_i.copy()
            if bj is not None and gt_pre_j is not None and bj not in gt_cur:
                gt_cur[bj]   = gt_pre_j.copy()
                pred_cur[bj] = gt_pre_j.copy()

            # Snap GT to exact collision position
            gt_cur[bi] = gt_pre_i.copy()
            if bj is not None and gt_pre_j is not None:
                gt_cur[bj] = gt_pre_j.copy()

            # ── Model step (teacher forcing: GT pre-collision state) ───────────
            ni = make_node(gt_pre_i, ev.event_type)
            if ev.event_type == EVENT_BALL_BALL and bj is not None and gt_pre_j is not None:
                nj   = make_node(gt_pre_j, ev.event_type)
                edge = make_edge(gt_pre_i, gt_pre_j, ev.normal.numpy())
                h, di, dj, _, _ = model.step_ball_ball(
                    h, bi, bj, ni.to(device), nj.to(device), edge.to(device))
                pred_delta_j = dj.cpu().numpy()
            else:
                h, di, _ = model.step_single(h, bi, ni.to(device), ev.normal.to(device))
                pred_delta_j = None

            pred_delta_i = di.cpu().numpy()

            # ── GT post-collision state ────────────────────────────────────────
            gt_post_i = gt_pre_i.copy()
            gt_post_i[1, :2] += shot.gt_deltas_i[k].numpy()[:2]
            gt_post_i[2]     += shot.gt_deltas_i[k].numpy()[2:]
            gt_cur[bi] = gt_post_i

            if bj is not None and gt_pre_j is not None and shot.gt_deltas_j[k] is not None:
                gt_post_j = gt_pre_j.copy()
                gt_post_j[1, :2] += shot.gt_deltas_j[k].numpy()[:2]
                gt_post_j[2]     += shot.gt_deltas_j[k].numpy()[2:]
                gt_cur[bj] = gt_post_j

            # ── Predicted post-collision state ─────────────────────────────────
            # Position: pred ball's own accumulated position (no GT snap)
            # Velocity: replace with GT pre-collision vel + predicted delta
            pred_post_i = pred_cur[bi].copy()
            pred_post_i[1, :2] = gt_pre_i[1, :2] + pred_delta_i[:2]
            pred_post_i[2]     = gt_pre_i[2]      + pred_delta_i[2:]
            pred_cur[bi] = pred_post_i

            if bj is not None and gt_pre_j is not None and pred_delta_j is not None:
                pred_post_j = pred_cur[bj].copy()
                pred_post_j[1, :2] = gt_pre_j[1, :2] + pred_delta_j[:2]
                pred_post_j[2]     = gt_pre_j[2]      + pred_delta_j[2:]
                pred_cur[bj] = pred_post_j

            # ── Sample frames for this segment ────────────────────────────────
            n_pts = max(2, int(dt / DT_RENDER))
            for i_pt, tau in enumerate(np.linspace(0, dt, n_pts)):
                frame = {
                    "is_event": (i_pt == 0),
                    "gt":       {bidx: _evolve(rv, tau)[0, :2].copy()
                                 for bidx, rv in gt_cur.items()},
                    "pred":     {bidx: _evolve(rv, tau)[0, :2].copy()
                                 for bidx, rv in pred_cur.items()},
                }
                frames.append(frame)

            # Advance all tracked balls to segment end
            for bidx in list(gt_cur.keys()):
                gt_cur[bidx]   = _evolve(gt_cur[bidx],   dt)
                pred_cur[bidx] = _evolve(pred_cur[bidx], dt)

    return frames


# ── Video ─────────────────────────────────────────────────────────────────────

BALL_COLOR = {
    0: ("white",   "cyan"),     # (GT, Pred) for cue ball
    1: ("gold",    "magenta"),  # for target ball
}


def make_video(
    shot    : ShotData,
    score   : float,
    frames  : list,
    tag     : str,
    rank    : int,
    out_dir : Path,
) -> None:
    if not frames:
        return

    W_cm = TABLE_W * 100
    H_cm = TABLE_H * 100

    fig, ax = plt.subplots(figsize=(5, 9))
    ax.set_xlim(-5, W_cm + 5)
    ax.set_ylim(-5, H_cm + 5)
    ax.set_aspect("equal")
    fig.patch.set_facecolor("#0d2b1a")
    ax.set_facecolor("#0d4a29")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors="white", labelsize=7)

    ax.add_patch(patches.FancyBboxPatch(
        (0, 0), W_cm, H_cm,
        linewidth=3, edgecolor="#5a3008", facecolor="none",
        boxstyle="round,pad=0",
    ))
    for px, py in POCKET_POS:
        ax.add_patch(plt.Circle((px * 100, py * 100), POCKET_R * 100,
                                color="black", zorder=5))

    gt_trails   = {i: ax.plot([], [], color=BALL_COLOR[i][0],
                               alpha=0.3, lw=1.2)[0] for i in range(2)}
    pred_trails = {i: ax.plot([], [], color=BALL_COLOR[i][1],
                               alpha=0.45, lw=1.5, ls="--")[0] for i in range(2)}
    gt_balls    = {i: plt.Circle((0, 0), 2.85,
                                  color=BALL_COLOR[i][0], zorder=8) for i in range(2)}
    pred_balls  = {i: plt.Circle((0, 0), 2.85, color=BALL_COLOR[i][1],
                                  zorder=9, fill=False, lw=2) for i in range(2)}
    for c in list(gt_balls.values()) + list(pred_balls.values()):
        ax.add_patch(c)

    ax.legend(handles=[
        plt.Line2D([0],[0], color="white",   lw=2,       label="GT cue"),
        plt.Line2D([0],[0], color="gold",    lw=2,       label="GT tgt"),
        plt.Line2D([0],[0], color="cyan",    lw=2, ls="--", label="Pred cue"),
        plt.Line2D([0],[0], color="magenta", lw=2, ls="--", label="Pred tgt"),
    ], loc="upper right", fontsize=6, framealpha=0.4,
       labelcolor="white", facecolor="#0d2b1a")

    title = ax.set_title("", fontsize=8, color="white")

    # Per-ball history for trails
    history = {i: [] for i in range(2)}

    def init():
        for i in range(2):
            gt_trails[i].set_data([], [])
            pred_trails[i].set_data([], [])
        title.set_text("")
        return (list(gt_trails.values()) + list(pred_trails.values()) +
                list(gt_balls.values()) + list(pred_balls.values()) + [title])

    def update(fi):
        frm    = frames[fi]
        is_ev  = frm["is_event"]

        for bidx in range(2):
            gp = frm["gt"].get(bidx)
            pp = frm["pred"].get(bidx)
            if gp is not None:
                history[bidx].append((gp, pp if pp is not None else gp))

            h_slice = history[bidx][max(0, len(history[bidx]) - TRAIL):]
            if h_slice:
                gt_xy   = np.array([p[0] for p in h_slice]) * 100
                pred_xy = np.array([p[1] for p in h_slice]) * 100
                gt_trails[bidx].set_data(gt_xy[:, 0], gt_xy[:, 1])
                pred_trails[bidx].set_data(pred_xy[:, 0], pred_xy[:, 1])

            if gp is not None:
                gt_balls[bidx].center   = (gp[0] * 100, gp[1] * 100)
            if pp is not None:
                pred_balls[bidx].center = (pp[0] * 100, pp[1] * 100)

        suffix = "  *** COLLISION ***" if is_ev else ""
        title.set_text(
            f"[{tag.upper()} #{rank+1}]  RMSE={score:.4f}  events={len(shot.event_steps)}{suffix}\n"
            f"white/gold=GT   cyan/magenta=Pred   f={fi+1}/{len(frames)}"
        )
        return (list(gt_trails.values()) + list(pred_trails.values()) +
                list(gt_balls.values()) + list(pred_balls.values()) + [title])

    ani = FuncAnimation(fig, update, frames=len(frames),
                        init_func=init, blit=True, interval=1000 / FPS)

    fname = out_dir / f"{tag}_{rank+1:02d}_rmse{score:.3f}.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=1500,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    ani.save(str(fname), writer=writer, dpi=100,
             savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default="world_model/results/rssm_v2/best.pt")
    p.add_argument("--out-dir",     default="world_model/results/viz_rssm")
    p.add_argument("--n-shots",     type=int, default=300)
    p.add_argument("--seed-off",    type=int, default=20000)
    p.add_argument("--top-k",       type=int, default=5)
    p.add_argument("--data-dir",    default=None,
                   help="pkl data dir (generate_rssm_data.py output). "
                        "If omitted, shots are generated on-the-fly.")
    p.add_argument("--pocket-only", action="store_true",
                   help="Only score/render shots that contain a pocket event.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    import matplotlib as mpl
    mpl.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading {args.ckpt} ...")
    model = RSSMModel(h_dim=H_DIM, hidden=[256, 256]).to(device)
    ck    = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    model.eval()

    # ── Collect test shots ─────────────────────────────────────────────────────
    if args.data_dir is not None:
        print(f"Loading shots from {args.data_dir} (pocket_only={args.pocket_only})...")
        shots = load_dataset(args.data_dir, max_shots=args.n_shots,
                             pocket_only=args.pocket_only)
        print(f"  Got {len(shots)} shots")
    else:
        print(f"Collecting {args.n_shots} test shots (seed_off={args.seed_off})...")
        env   = BilliardsEnv(n_balls=1)
        shots = []
        for seed in range(args.n_shots * 2):
            env.reset(seed=args.seed_off + seed)
            env.step(env.action_space.sample())
            shot = generate_shot_data(env.system, BALL_IDS)
            if len(shot.event_steps) >= 2:
                if not args.pocket_only or any(e.event_type == 3 for e in shot.event_steps):
                    shots.append(shot)
            if len(shots) >= args.n_shots:
                break
        env.close()
        print(f"  Got {len(shots)} valid shots")

    # ── Score ──────────────────────────────────────────────────────────────────
    print("Scoring...")
    scores     = np.array([shot_rmse(model, s, device) for s in shots])
    sorted_idx = np.argsort(scores)
    best_idx   = sorted_idx[:args.top_k].tolist()
    worst_idx  = sorted_idx[-args.top_k:][::-1].tolist()

    print(f"  RMSE  min={scores.min():.4f}  max={scores.max():.4f}"
          f"  mean={scores.mean():.4f}  median={np.median(scores):.4f}")
    print(f"  Best  {args.top_k}: " + "  ".join(f"{scores[i]:.3f}" for i in best_idx))
    print(f"  Worst {args.top_k}: " + "  ".join(f"{scores[i]:.3f}" for i in worst_idx))

    # ── Render ─────────────────────────────────────────────────────────────────
    print("\n--- WORST ---")
    for rank, idx in enumerate(worst_idx):
        frms = reconstruct(model, shots[idx], device)
        make_video(shots[idx], scores[idx], frms, "worst", rank, out_dir)

    print("\n--- BEST ---")
    for rank, idx in enumerate(best_idx):
        frms = reconstruct(model, shots[idx], device)
        make_video(shots[idx], scores[idx], frms, "best",  rank, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
