"""
world_model/viz_pure_physics.py

pure_physics.py(직접 재구현한 물리 롤아웃) vs pooltool(실제 시뮬레이터) 비교 영상.

실제 RSSM 학습 데이터(data_rssm)의 각 이벤트에서 동일한 충돌-후 상태를 만든 뒤,
pooltool의 ph.evolve_ball_motion과 world_model.pure_physics.evolve_ball_motion
양쪽으로 자유주행시켜 궤적을 겹쳐 그린다. 충돌 resolution은 항상 실제 GT delta를
그대로 적용하므로, 두 궤적의 차이는 오직 free-flight 공식/마찰계수 불일치만 반영한다.

Usage:
    python world_model/viz_pure_physics.py
    python world_model/viz_pure_physics.py --out-dir /tmp/viz_pp --top-k 5
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

import pooltool.constants as const
import pooltool.physics as ph

from world_model.ball_motion import DEFAULT_FRICTION
from world_model import pure_physics as pp
from world_model.rssm_dataset import ShotData


# ── Constants ─────────────────────────────────────────────────────────────────

TABLE_W = 0.9906   # m
TABLE_H = 1.9812   # m
PARAMS  = DEFAULT_FRICTION
PARAMS_DICT = dict(R=PARAMS.R, m=PARAMS.m, u_s=PARAMS.u_s,
                    u_sp=PARAMS.u_sp, u_r=PARAMS.u_r, g=PARAMS.g)

POCKET_POS = [
    (0,       0),
    (TABLE_W, 0),
    (0,       TABLE_H),
    (TABLE_W, TABLE_H),
    (0,       TABLE_H / 2),
    (TABLE_W, TABLE_H / 2),
]
POCKET_R = 0.062   # m

FPS       = 20
TRAIL     = 40      # trail length in frames
DT_RENDER = 0.025   # seconds between rendered frames


# ── Physics evolution helpers ───────────────────────────────────────────────────

def _speed(rvw: np.ndarray) -> float:
    return float(np.linalg.norm(rvw[1, :2]))


def _evolve_pt(rvw: np.ndarray, dt: float) -> np.ndarray:
    """pooltool's own evolve_ball_motion (real simulator's free-flight formula)."""
    if _speed(rvw) < 1e-6:
        return rvw.copy()
    rvw_new, _ = ph.evolve_ball_motion(const.sliding, np.asarray(rvw, dtype=np.float64),
                                        **PARAMS_DICT, t=dt)
    return rvw_new


def _evolve_pp(rvw: np.ndarray, dt: float) -> np.ndarray:
    """world_model.pure_physics re-implementation."""
    if _speed(rvw) < 1e-6:
        return rvw.copy()
    rvw_new, _ = pp.evolve_ball_motion(pp.SLIDING, np.asarray(rvw, dtype=np.float64),
                                        **PARAMS_DICT, t=dt)
    return rvw_new


def _apply_delta(raw_rvw: np.ndarray, delta) -> np.ndarray:
    out = np.array(raw_rvw, dtype=np.float64).copy()
    if delta is None:
        return out
    d = delta.numpy() if hasattr(delta, "numpy") else np.asarray(delta, dtype=np.float64)
    out[1, :2] += d[:2]   # Δvel
    out[2]     += d[2:]   # Δavel
    return out


# ── Reconstruction ───────────────────────────────────────────────────────────────

def reconstruct(shot: ShotData) -> list[dict]:
    """
    Both arms are re-anchored to the identical true post-collision state at
    every event a ball is directly involved in (collision resolution is
    always the real GT delta) — the only thing under test is whether
    pure_physics's free-flight formula/params reproduce pooltool's between events.
    """
    pt_cur: dict[int, np.ndarray] = {}
    pp_cur: dict[int, np.ndarray] = {}
    frames: list[dict] = []

    for k, ev in enumerate(shot.event_steps):
        bi = ev.ball_i
        bj = ev.ball_j
        dt = float(shot.dt_to_next[k])

        gt_pre_i = shot.raw_rvws_i[k]
        gt_pre_j = (shot.raw_rvws_j[k]
                    if bj is not None and shot.raw_rvws_j[k] is not None
                    else None)

        post_i = _apply_delta(gt_pre_i, shot.gt_deltas_i[k])
        pt_cur[bi] = post_i.copy()
        pp_cur[bi] = post_i.copy()

        if bj is not None and gt_pre_j is not None:
            post_j = _apply_delta(gt_pre_j, shot.gt_deltas_j[k])
            pt_cur[bj] = post_j.copy()
            pp_cur[bj] = post_j.copy()

        if dt <= 0:
            continue

        n_pts = max(2, int(dt / DT_RENDER))
        for i_pt, tau in enumerate(np.linspace(0, dt, n_pts)):
            frames.append({
                "is_event": (i_pt == 0),
                "pooltool": {b: _evolve_pt(rv, tau)[0, :2].copy() for b, rv in pt_cur.items()},
                "physics":  {b: _evolve_pp(rv, tau)[0, :2].copy() for b, rv in pp_cur.items()},
            })

        for b in list(pt_cur.keys()):
            pt_cur[b] = _evolve_pt(pt_cur[b], dt)
            pp_cur[b] = _evolve_pp(pp_cur[b], dt)

    return frames


def shot_error_cm(frames: list[dict]) -> float:
    """Mean pooltool-vs-physics position discrepancy across all frames/balls."""
    errs = []
    for frm in frames:
        for b, pt_xy in frm["pooltool"].items():
            pp_xy = frm["physics"].get(b)
            if pp_xy is not None:
                errs.append(np.linalg.norm(pt_xy - pp_xy) * 100)
    return float(np.mean(errs)) if errs else 0.0


# ── Video ─────────────────────────────────────────────────────────────────────

BALL_COLOR = {
    0: ("white", "cyan"),     # (pooltool, pure_physics) — cue
    1: ("gold",  "magenta"),  # (pooltool, pure_physics) — target
}


def make_video(shot: ShotData, err_cm: float, frames: list[dict],
               tag: str, rank: int, out_dir: Path) -> None:
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

    pt_trails = {i: ax.plot([], [], color=BALL_COLOR[i][0],
                             alpha=0.3, lw=1.2)[0] for i in range(2)}
    pp_trails = {i: ax.plot([], [], color=BALL_COLOR[i][1],
                             alpha=0.45, lw=1.5, ls="--")[0] for i in range(2)}
    pt_balls  = {i: plt.Circle((0, 0), 2.85,
                                color=BALL_COLOR[i][0], zorder=8) for i in range(2)}
    pp_balls  = {i: plt.Circle((0, 0), 2.85, color=BALL_COLOR[i][1],
                                zorder=9, fill=False, lw=2) for i in range(2)}
    for c in list(pt_balls.values()) + list(pp_balls.values()):
        ax.add_patch(c)

    ax.legend(handles=[
        plt.Line2D([0], [0], color="white",   lw=2,          label="pooltool cue"),
        plt.Line2D([0], [0], color="gold",    lw=2,          label="pooltool tgt"),
        plt.Line2D([0], [0], color="cyan",    lw=2, ls="--", label="pure_physics cue"),
        plt.Line2D([0], [0], color="magenta", lw=2, ls="--", label="pure_physics tgt"),
    ], loc="upper right", fontsize=6, framealpha=0.4,
       labelcolor="white", facecolor="#0d2b1a")

    title = ax.set_title("", fontsize=8, color="white")
    history = {i: [] for i in range(2)}

    def init():
        for i in range(2):
            pt_trails[i].set_data([], [])
            pp_trails[i].set_data([], [])
        title.set_text("")
        return (list(pt_trails.values()) + list(pp_trails.values()) +
                list(pt_balls.values()) + list(pp_balls.values()) + [title])

    def update(fi):
        frm   = frames[fi]
        is_ev = frm["is_event"]

        for bidx in range(2):
            gp    = frm["pooltool"].get(bidx)
            pp_xy = frm["physics"].get(bidx)
            if gp is not None:
                history[bidx].append((gp, pp_xy if pp_xy is not None else gp))

            h_slice = history[bidx][max(0, len(history[bidx]) - TRAIL):]
            if h_slice:
                pt_xy_arr = np.array([p[0] for p in h_slice]) * 100
                pp_xy_arr = np.array([p[1] for p in h_slice]) * 100
                pt_trails[bidx].set_data(pt_xy_arr[:, 0], pt_xy_arr[:, 1])
                pp_trails[bidx].set_data(pp_xy_arr[:, 0], pp_xy_arr[:, 1])

            if gp is not None:
                pt_balls[bidx].center = (gp[0] * 100, gp[1] * 100)
            if pp_xy is not None:
                pp_balls[bidx].center = (pp_xy[0] * 100, pp_xy[1] * 100)

        suffix = "  *** EVENT ***" if is_ev else ""
        title.set_text(
            f"[{tag.upper()} #{rank+1}]  mean_err={err_cm:.4f}cm  events={len(shot.event_steps)}{suffix}\n"
            f"white/gold=pooltool   cyan/magenta=pure_physics   f={fi+1}/{len(frames)}"
        )
        return (list(pt_trails.values()) + list(pp_trails.values()) +
                list(pt_balls.values()) + list(pp_balls.values()) + [title])

    ani = FuncAnimation(fig, update, frames=len(frames),
                        init_func=init, blit=True, interval=1000 / FPS)

    fname  = out_dir / f"{tag}_{rank+1:02d}_err{err_cm:.4f}cm.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=1500,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    ani.save(str(fname), writer=writer, dpi=100,
             savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(Path(__file__).parent / "data_rssm"))
    p.add_argument("--out-dir",  default="world_model/results/viz_pure_physics")
    p.add_argument("--n-shots",  type=int, default=300)
    p.add_argument("--top-k",    type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl
    mpl.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

    data_dir = Path(args.data_dir)
    chunk_files = sorted(data_dir.glob("*_chunk*.pkl"))
    assert chunk_files, f"No chunk files in {data_dir}"
    print(f"Loading {chunk_files[0].name} ...")
    with open(chunk_files[0], "rb") as f:
        all_shots = pickle.load(f)

    shots = [s for s in all_shots if len(s.event_steps) >= 2][:args.n_shots]
    print(f"Using {len(shots)} shots")

    print("Reconstructing + scoring...")
    all_frames = []
    scores     = []
    for s in shots:
        frms = reconstruct(s)
        all_frames.append(frms)
        scores.append(shot_error_cm(frms))
    scores = np.array(scores)

    sorted_idx = np.argsort(scores)
    best_idx   = sorted_idx[:args.top_k].tolist()
    worst_idx  = sorted_idx[-args.top_k:][::-1].tolist()

    print(f"  err(cm)  min={scores.min():.4f}  max={scores.max():.4f}"
          f"  mean={scores.mean():.4f}  median={np.median(scores):.4f}")
    print(f"  Best  {args.top_k}: " + "  ".join(f"{scores[i]:.4f}" for i in best_idx))
    print(f"  Worst {args.top_k}: " + "  ".join(f"{scores[i]:.4f}" for i in worst_idx))

    print("\n--- WORST ---")
    for rank, idx in enumerate(worst_idx):
        make_video(shots[idx], scores[idx], all_frames[idx], "worst", rank, out_dir)

    print("\n--- BEST ---")
    for rank, idx in enumerate(best_idx):
        make_video(shots[idx], scores[idx], all_frames[idx], "best", rank, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
