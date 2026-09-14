"""
plot_placement.py — legacy vs current 볼 초기 위치 분포 시각화
Usage: python plot_placement.py --samples 200 --out outputs/placement_dist.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from simulator import BilliardsEnv

FELT   = "#276221"
RAIL   = "#5a3a1a"
CUE_C  = "#f0f0f0"
BALL_C = "#e74c3c"
POCKET_C = "#111111"

def collect_positions(n_samples, legacy):
    env = BilliardsEnv(n_balls=1, legacy_placement=legacy)
    cue_pts, ball_pts = [], []
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        env.reset(seed=int(rng.integers(0, 2**31)))
        cue_pts.append(env.system.balls["cue"].state.rvw[0, :2].copy())
        ball_pts.append(env.system.balls[env._ball_ids[0]].state.rvw[0, :2].copy())
    env.close()
    return np.array(cue_pts), np.array(ball_pts)

def draw_table(ax, env):
    W, L = env.table.w, env.table.l
    ax.set_facecolor(RAIL)
    ax.add_patch(patches.Rectangle((0, 0), W, L, color=FELT, zorder=1))
    # pockets
    pocket_centers = env._pocket_centers
    for px, py in pocket_centers:
        ax.add_patch(plt.Circle((px, py), 0.03, color=POCKET_C, zorder=3))
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-0.05, L + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--out", default="outputs/placement_dist.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env_cur = BilliardsEnv(n_balls=1, legacy_placement=False)
    env_leg = BilliardsEnv(n_balls=1, legacy_placement=True)

    cur_cue, cur_ball   = collect_positions(args.samples, legacy=False)
    leg_cue, leg_ball   = collect_positions(args.samples, legacy=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a1a")

    titles = ["Current Placement", "Legacy Placement"]
    cue_data  = [cur_cue,  leg_cue]
    ball_data = [cur_ball, leg_ball]
    envs = [env_cur, env_leg]

    for ax, title, cue_pts, ball_pts, env in zip(axes, titles, cue_data, ball_data, envs):
        draw_table(ax, env)
        ax.scatter(cue_pts[:, 0],  cue_pts[:, 1],
                   c=CUE_C,  s=18, alpha=0.5, zorder=4, label="Cue ball")
        ax.scatter(ball_pts[:, 0], ball_pts[:, 1],
                   c=BALL_C, s=18, alpha=0.5, zorder=4, label="Target ball")
        ax.set_title(title, color="white", fontsize=13, pad=8)
        ax.legend(loc="lower right", fontsize=8,
                  facecolor="#333", labelcolor="white", framealpha=0.7)

    fig.suptitle(f"Ball Placement Distribution  (n={args.samples})",
                 color="white", fontsize=14, y=0.97)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {args.out}")

    env_cur.close()
    env_leg.close()

if __name__ == "__main__":
    main()
