"""
world_model/debug_sim_viz.py
시뮬레이터 직접 실행 → 이벤트 순서 시각화 (GT ground truth 확인용).

Usage:
    python world_model/debug_sim_viz.py [--seed 42] [--out debug_sim.png]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from simulator import BilliardsEnv
from world_model.wm_predictor import TABLE_W, TABLE_H

# ── 이벤트 타입 색상 ─────────────────────────────────────────────────────────
TYPE_COLOR = {
    "stick_ball":             "#00bfff",
    "ball_ball":              "#ff6600",
    "ball_linear_cushion":    "#ffdd00",
    "ball_circular_cushion":  "#ffaa00",
    "ball_pocket":            "#ff2222",
    "sliding_rolling":        "#aaffaa",
    "rolling_spinning":       "#88ff88",
    "rolling_stationary":     "#44ff44",
    "spinning_stationary":    "#22cc22",
    "none":                   "#888888",
}
CUE_LINE = "#00e5ff"
TGT_LINE = "#ffee44"
BG       = "#1a1a1a"


def get_xy(agent):
    if hasattr(agent.initial, "xyz"):
        return float(agent.initial.xyz[0]), float(agent.initial.xyz[1])
    if hasattr(agent.initial, "state"):
        rv = agent.initial.state.rvw[0]
        return float(rv[0]), float(rv[1])
    return None, None


def draw_table(ax):
    ax.add_patch(patches.Rectangle(
        (0, 0), TABLE_W, TABLE_H,
        facecolor="#2d7a2d", edgecolor="#1a4a1a", linewidth=2,
    ))
    for px, py in [(0,0),(TABLE_W,0),(0,TABLE_H/2),(TABLE_W,TABLE_H/2),(0,TABLE_H),(TABLE_W,TABLE_H)]:
        ax.add_patch(plt.Circle((px, py), 0.038, color="black", zorder=5))
    ax.set_xlim(-0.06, TABLE_W + 0.06)
    ax.set_ylim(-0.06, TABLE_H + 0.06)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def run_and_viz(seed, out_path):
    env = BilliardsEnv(n_balls=1)
    obs, _ = env.reset(seed=seed)
    action  = env.action_space.sample()
    np.random.seed(seed)  # 재현성
    action  = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    system  = env.system

    # ── 이벤트 파싱 ────────────────────────────────────────────────────────
    # 공별로 이벤트 위치 수집 (시간 순서)
    cue_pts  = []   # [(x, y, event_type_str), ...]
    tgt_pts  = []

    print(f"\nseed={seed}  action={action}  reward={reward:.3f}")
    print(f"{'idx':>4}  {'event_type':30}  agents")
    print("-" * 70)

    for i, e in enumerate(system.events):
        et = str(e.event_type)
        row_info = []
        for ag in e.agents:
            if not (hasattr(ag, "agent_type") and ag.agent_type == "ball"):
                continue
            x, y = get_xy(ag)
            if x is None:
                continue
            row_info.append(f"{ag.id}@({x:.3f},{y:.3f})")
            if ag.id == "cue":
                cue_pts.append((x, y, et))
            else:
                tgt_pts.append((x, y, et))
        print(f"{i:4d}  {et:30}  {', '.join(row_info)}")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 9), facecolor=BG)
    draw_table(ax)

    # 초기 공 위치 (obs에서)
    cue0_x, cue0_y = float(obs[0]), float(obs[1])   # raw coords (env 정규화)
    tgt0_x, tgt0_y = float(obs[2]), float(obs[3])

    # obs 좌표가 [0,1]이면 TABLE 스케일로 복원
    if cue0_x <= 1.0 and cue0_y <= 1.0:
        cue0_x *= TABLE_W;  cue0_y *= TABLE_H
        tgt0_x *= TABLE_W;  tgt0_y *= TABLE_H

    ax.add_patch(plt.Circle((cue0_x, cue0_y), 0.030,
                             facecolor="white", edgecolor="#aaaaaa", lw=1.2, zorder=10))
    ax.add_patch(plt.Circle((tgt0_x, tgt0_y), 0.030,
                             facecolor="#ffee44", edgecolor="#888800", lw=1.2, zorder=10))

    def draw_ball_path(pts, init_xy, line_color, label):
        """초기 위치 → 모든 이벤트 위치를 이어서 하나의 연속 경로로 그린다."""
        if not pts:
            return
        xs = [init_xy[0]] + [p[0] for p in pts]
        ys = [init_xy[1]] + [p[1] for p in pts]
        types = [None]     + [p[2] for p in pts]

        # 전체 path 선 (연속)
        ax.plot(xs, ys, color=line_color, lw=1.8, alpha=0.85, zorder=6, label=label)

        # 초기 위치 표시
        ax.scatter(xs[0], ys[0], s=60, marker="*", color=line_color,
                   zorder=9, edgecolors="white", linewidths=0.5)

        # 이벤트 점 (타입별 색상)
        for x, y, et in pts:
            c = TYPE_COLOR.get(et, "#cccccc")
            ax.scatter(x, y, s=28, c=c, zorder=8,
                       linewidths=0.4, edgecolors="white", alpha=0.95)

        # 이벤트 번호 텍스트
        for k, (x, y, et) in enumerate(pts):
            ax.text(x + 0.015, y + 0.015, str(k+1),
                    fontsize=5, color="white", zorder=11, alpha=0.8)

    draw_ball_path(cue_pts, (cue0_x, cue0_y), CUE_LINE, "cue")
    draw_ball_path(tgt_pts, (tgt0_x, tgt0_y), TGT_LINE, "tgt")

    # 범례
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elems = (
        [Line2D([0],[0], color=CUE_LINE, lw=2, label="cue path"),
         Line2D([0],[0], color=TGT_LINE, lw=2, label="tgt path")]
        + [Patch(facecolor=c, label=et.replace("ball_","").replace("_"," "))
           for et, c in TYPE_COLOR.items() if et not in ("none","stick_ball")]
    )
    ax.legend(handles=legend_elems, loc="upper left", fontsize=6,
              framealpha=0.6, facecolor="#333", labelcolor="white",
              bbox_to_anchor=(0, 1))

    ax.set_title(f"seed={seed}  φ={action[0]:.2f}  v={action[1]:.2f}  r={reward:.2f}",
                 fontsize=9, color="white", pad=6)
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n→ saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",  type=str, default="world_model/results/debug_sim.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_and_viz(args.seed, args.out)
