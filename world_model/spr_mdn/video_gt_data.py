"""
world_model/spr_mdn/video_gt_data.py — GT 데이터 시각화

v28_bb 스타일: 실물 cm 좌표, portrait, 실제 공 크기, trail.
왼쪽: 테이블 (충돌 타입 마커), 오른쪽: 속도 타임시리즈.
데이터 정합성 체크용.

Usage:
    python world_model/spr_mdn/video_gt_data.py \
        --data-dir world_model/data_fixeddt \
        --n-videos 8 \
        --out-dir world_model/results/v33_gt_videos
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.spr_dataset import SPRDataset, make_balanced_val_eps
from world_model.wm_predictor import TABLE_W, TABLE_H

FPS    = 15
STEPS  = 60    # overridden by --steps arg
TRAIL  = 20
DT     = 0.05  # overridden by --dt arg
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
# coll_ball 비트마스크: bit0=cue, bit1=target
_CUE_BIT = 1; _TGT_BIT = 2


def speed_ms(states, col_vx, col_vy):
    vx = states[:, col_vx] * TABLE_W
    vy = states[:, col_vy] * TABLE_H
    return np.sqrt(vx**2 + vy**2)


def make_gt_video(ep_idx, ep_s, ep_f, ep_t, out_path: Path, ep_b=None):
    """ep_b: (T,) int8 coll_ball bitmask; None → mark both balls for any collision"""
    T      = min(STEPS, len(ep_s) - 1)
    states = ep_s[:T + 1]
    flags  = ep_f[:T]
    types  = ep_t[:T]
    balls  = ep_b[:T] if ep_b is not None else None

    cue_cm = states[:, :2]  * np.array([W_cm, H_cm])
    tgt_cm = states[:, 7:9] * np.array([W_cm, H_cm])
    cue_spd = speed_ms(states, 2, 3)
    tgt_spd = speed_ms(states, 9, 10)
    time_ax = np.arange(T + 1) * DT

    n_cush = int((flags & ((types == 2) | (types == 3))).sum())
    has_bb = bool(np.any(flags & (types == 1)))
    tag    = "bb" if has_bb else "no-bb"

    import matplotlib as mpl
    mpl.rcParams["animation.ffmpeg_path"] = FFMPEG

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0d2b1a")
    fig.suptitle(
        f"GT Shot {ep_idx}  [{tag}]  T={T}steps  n_cush={n_cush}",
        color="white", fontsize=10)

    ax_tbl = fig.add_axes([0.02, 0.06, 0.46, 0.88])
    ax_spd = fig.add_axes([0.54, 0.12, 0.42, 0.72])

    # ── 테이블 ──────────────────────────────────────────────────────────────
    ax_tbl.set_xlim(-3, W_cm + 3)
    ax_tbl.set_ylim(-3, H_cm + 3)
    ax_tbl.set_aspect("equal")
    ax_tbl.set_facecolor("#0d4a29")
    ax_tbl.axis("off")
    ax_tbl.add_patch(patches.FancyBboxPatch(
        (0, 0), W_cm, H_cm, linewidth=3,
        edgecolor="#5a3008", facecolor="#1a6b3c", boxstyle="round,pad=0"))
    for px, py in POCKETS:
        ax_tbl.add_patch(plt.Circle((px, py), 2.5, color="black", zorder=10))

    ax_tbl.plot(cue_cm[:, 0], cue_cm[:, 1], color="white", lw=0.8, alpha=0.18, zorder=2)
    ax_tbl.plot(tgt_cm[:, 0], tgt_cm[:, 1], color="gold",  lw=0.8, alpha=0.18, zorder=2)

    # 충돌 마커 미리 그리기 (정적) — coll_ball 기반으로 해당 공만 표시
    for t in range(T):
        if flags[t]:
            tp   = int(types[t])
            col  = COLL_COLORS.get(tp, "#ffffff")
            mask = int(balls[t]) if balls is not None else 3   # 3 = both
            if mask & _CUE_BIT:
                ax_tbl.scatter([cue_cm[t+1, 0]], [cue_cm[t+1, 1]],
                               color=col, s=80, marker="o", zorder=8, alpha=0.7)
                ax_tbl.text(cue_cm[t+1, 0] + 1.5, cue_cm[t+1, 1] + 1.5,
                            TYPE_NAMES.get(tp, "?"), color=col, fontsize=6, zorder=9)
            if mask & _TGT_BIT:
                ax_tbl.scatter([tgt_cm[t+1, 0]], [tgt_cm[t+1, 1]],
                               color=col, s=80, marker="o", zorder=8, alpha=0.7)
                if not (mask & _CUE_BIT):   # cue에 이미 라벨 없으면 tgt에 표시
                    ax_tbl.text(tgt_cm[t+1, 0] + 1.5, tgt_cm[t+1, 1] + 1.5,
                                TYPE_NAMES.get(tp, "?"), color=col, fontsize=6, zorder=9)

    # 동적 trail + 공
    trail_cue, = ax_tbl.plot([], [], "w-",  lw=1.8, alpha=0.6, zorder=4)
    trail_tgt, = ax_tbl.plot([], [], "y-",  lw=1.8, alpha=0.6, zorder=4)
    ball_cue   = ax_tbl.add_patch(plt.Circle((0,0), BALL_R, color="white", zorder=9))
    ball_tgt   = ax_tbl.add_patch(plt.Circle((0,0), BALL_R, color="gold",  zorder=9))
    txt_time   = ax_tbl.text(1, H_cm + 4, "", color="white", fontsize=9)
    txt_coll   = ax_tbl.text(1, -6, "", color="#ffcc44", fontsize=8)

    # ── 속도 그래프 ──────────────────────────────────────────────────────────
    ax_spd.set_facecolor("#0d1a0d")
    ax_spd.set_xlim(0, T * DT)
    ymax = max(cue_spd.max(), tgt_spd.max()) * 1.15
    ax_spd.set_ylim(-0.1, ymax)
    ax_spd.set_xlabel("time (s)", color="white", fontsize=8)
    ax_spd.set_ylabel("speed (m/s)", color="white", fontsize=8)
    ax_spd.tick_params(colors="white", labelsize=7)
    for sp in ax_spd.spines.values():
        sp.set_edgecolor("#444")

    # 전체 속도 궤적 (배경)
    ax_spd.plot(time_ax, cue_spd, color="white", lw=0.8, alpha=0.2)
    ax_spd.plot(time_ax, tgt_spd, color="gold",  lw=0.8, alpha=0.2)

    # 충돌 수직선 (정적)
    for t in range(T):
        if flags[t]:
            tp  = int(types[t])
            col = COLL_COLORS.get(tp, "#ffffff")
            xt  = (t + 1) * DT
            ax_spd.axvline(xt, color=col, lw=1.2, alpha=0.75, zorder=2)
            ax_spd.text(xt + 0.01, ymax * 0.93,
                        TYPE_NAMES.get(tp, "?"),
                        color=col, fontsize=6, va="top", rotation=90)

    # 범례
    ax_spd.legend(
        [Line2D([0],[0],color="white",lw=2), Line2D([0],[0],color="gold",lw=2)],
        ["cue speed", "target speed"],
        loc="upper right", framealpha=0.3, labelcolor="white", fontsize=8)

    # 동적 속도 선 + 현재 시간 마커
    spd_cue, = ax_spd.plot([], [], color="white", lw=1.8, zorder=4)
    spd_tgt, = ax_spd.plot([], [], color="gold",  lw=1.8, zorder=4)
    cur_line  = ax_spd.axvline(0, color="#aaffaa", lw=1.0, alpha=0.8, zorder=5)

    n_frames = T + 1

    def init():
        trail_cue.set_data([], [])
        trail_tgt.set_data([], [])
        return trail_cue, trail_tgt

    last_coll_type = [None]

    def update(frame):
        t   = frame
        t0  = max(0, t - TRAIL)

        trail_cue.set_data(cue_cm[t0:t+1, 0], cue_cm[t0:t+1, 1])
        trail_tgt.set_data(tgt_cm[t0:t+1, 0], tgt_cm[t0:t+1, 1])
        ball_cue.center = (cue_cm[t, 0], cue_cm[t, 1])
        ball_tgt.center = (tgt_cm[t, 0], tgt_cm[t, 1])

        # 충돌 직후 라벨 (잠시 표시)
        if t > 0 and flags[t-1]:
            tp = int(types[t-1])
            last_coll_type[0] = (t, TYPE_NAMES.get(tp, "?"),
                                  COLL_COLORS.get(tp, "#fff"))
        if last_coll_type[0] and t - last_coll_type[0][0] < 8:
            _, name, col = last_coll_type[0]
            txt_coll.set_text(f"⚡ {name}")
            txt_coll.set_color(col)
        else:
            txt_coll.set_text("")

        spd_cue.set_data(time_ax[:t+1], cue_spd[:t+1])
        spd_tgt.set_data(time_ax[:t+1], tgt_spd[:t+1])
        cur_line.set_xdata([t * DT, t * DT])

        txt_time.set_text(f"t={t}/{T}  ({t*DT:.2f}s)")
        return []

    ani = FuncAnimation(fig, update, frames=n_frames,
                        init_func=init, blit=False, interval=1000/FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=1500,
                          extra_args=["-vcodec","libx264","-pix_fmt","yuv420p"])
    ani.save(str(out_path), writer=writer, dpi=100,
             savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  → {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="world_model/data_fixeddt")
    p.add_argument("--pkl-file",  default=None,
                   help="load episodes directly from a .pkl file (e.g. late_bounce_episodes.pkl)")
    p.add_argument("--n-videos",  type=int, default=8)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out-dir",   default="world_model/results/v33_gt_videos")
    p.add_argument("--dt",        type=float, default=None,
                   help="DT of the data (auto-detected from metadata if not set)")
    p.add_argument("--steps",     type=int, default=None,
                   help="rollout steps for visualization (default: 3s worth)")
    args = p.parse_args()

    # DT / STEPS override
    global DT, STEPS
    if args.dt is not None:
        DT = args.dt
    else:
        import json
        meta = json.load(open(Path(args.data_dir) / "metadata.json"))
        DT = float(meta[-1].get("dt", 0.05))
    STEPS = args.steps if args.steps else max(60, int(3.0 / DT))
    print(f"DT={DT}s  STEPS={STEPS} ({STEPS*DT:.1f}s)")

    # episode source: pkl or SPRDataset
    coll_balls_map = {}   # idx → coll_ball array (or None)
    if args.pkl_file is not None:
        import pickle
        with open(args.pkl_file, "rb") as f:
            all_eps = pickle.load(f)
        valid = [(i, ep) for i, ep in enumerate(all_eps) if len(ep[0]) >= STEPS + 1]
    else:
        dataset = SPRDataset(args.data_dir)
        rng     = np.random.default_rng(0)
        perm    = rng.permutation(len(dataset.episodes))
        n_val   = max(50, int(len(dataset.episodes) * 0.2))
        val_idxs = perm[:n_val]
        val_eps  = [dataset.episodes[j] for j in val_idxs]
        for rank_i, j in enumerate(val_idxs):
            coll_balls_map[rank_i] = dataset.coll_balls[j]
        valid = [(i, ep) for i, ep in enumerate(val_eps) if len(ep[0]) >= STEPS + 1]

    rng2  = np.random.default_rng(args.seed)

    has_bb = [(i,ep) for i,ep in valid if np.any(ep[1][:STEPS] & (ep[2][:STEPS]==1))]
    no_bb  = [(i,ep) for i,ep in valid if not np.any(ep[1][:STEPS] & (ep[2][:STEPS]==1))]
    n_bb   = min(args.n_videos // 2, len(has_bb))
    n_nbb  = min(args.n_videos - n_bb, len(no_bb))
    chosen = ([has_bb[j] for j in rng2.choice(len(has_bb), n_bb,  replace=False)] +
              [no_bb[j]  for j in rng2.choice(len(no_bb),  n_nbb, replace=False)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (i, (ep_s, ep_f, ep_t, n_cush, _)) in enumerate(chosen):
        T = min(STEPS, len(ep_s) - 1)
        has_bb_ep = bool(np.any(ep_f[:T] & (ep_t[:T] == 1)))
        idxs = np.where(ep_f[:T])[0]
        fb = int(idxs[0]) if len(idxs) > 0 else -1
        print(f"[{rank+1}/{len(chosen)}] ep{i}  T={T}  ({T*DT:.1f}s)  n_cush={n_cush}  bb={has_bb_ep}  first_bounce=step{fb}")
        fname = out_dir / f"gt_{rank+1:02d}_ep{i}.mp4"
        ep_b = coll_balls_map.get(i)
        make_gt_video(i, ep_s, ep_f, ep_t, fname, ep_b=ep_b)

    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
