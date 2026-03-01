#!/usr/bin/env python3
"""
visualize.py — Load best SAC model and render shots as a top-down table grid.

Usage:
    python visualize.py                     # 20 shots, best_model
    python visualize.py --model sac_billiards --shots 12
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pooltool as pt
from stable_baselines3 import SAC
from simulator import BilliardsEnv


# ── Visual constants ──────────────────────────────────────────────────────────
BALL_R      = 0.028575          # standard pool ball radius (m)
FELT_COLOR  = "#1a7a3a"
RAIL_COLOR  = "#5c3a1e"
POCKET_CLR  = "#0a0a0a"
CUE_CLR     = "#f0f0f0"
EIGHT_CLR   = "#1a1a1a"
AIM_CLR     = "#ffff00"
HIT_CLR     = "#00e676"
MISS_CLR    = "#ff4444"
BG_CLR      = "#111111"


# ── Trajectory extraction ─────────────────────────────────────────────────────
def _trajectory(ball):
    """Return (N, 2) x-y trajectory from ball history, or None."""
    for attr in ("history_cts", "history"):
        h = getattr(ball, attr, None)
        if h is None:
            continue
        rvw = getattr(h, "rvw", None)
        if rvw is None:
            continue
        arr = np.asarray(rvw)
        if arr.ndim == 3 and arr.shape[1] >= 1:
            return arr[:, 0, :2]        # (N, 3, 3) layout
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]           # (N, ≥2) layout
    return None


# ── Draw one shot ─────────────────────────────────────────────────────────────
def draw_shot(ax, env, model, seed=None):
    obs, _ = env.reset(seed=seed)

    # Initial positions before the shot
    cue_i    = env.system.balls["cue"].state.rvw[0, :2].copy()
    target_i = env.system.balls["8"].state.rvw[0, :2].copy()

    # Agent decision
    action, _ = model.predict(obs, deterministic=True)
    delta_angle = float(action[0])
    speed       = float(action[1])

    # Absolute shot angle (for the aim arrow)
    phi_direct = np.arctan2(target_i[1] - cue_i[1], target_i[0] - cue_i[0])
    phi_shot   = phi_direct + delta_angle

    # Step environment
    _, _, _, _, info = env.step(action)
    pocketed = info["pocketed"]

    W = env.table_width
    L = env.table_length

    # ── Background & table ────────────────────────────────────────────────────
    ax.set_facecolor(RAIL_COLOR)
    ax.set_xlim(-0.07, W + 0.07)
    ax.set_ylim(-0.07, L + 0.07)
    ax.set_aspect("equal")
    ax.axis("off")

    # Felt surface
    ax.add_patch(patches.Rectangle((0, 0), W, L, color=FELT_COLOR, zorder=1))

    # Cushion lines
    for lw, clr, inset in [(6, RAIL_COLOR, -0.04), (1.5, "#2e9c50", 0.02)]:
        ax.add_patch(patches.Rectangle(
            (inset, inset), W - 2*inset, L - 2*inset,
            linewidth=lw, edgecolor=clr, facecolor="none", zorder=2
        ))

    # Pockets
    for pc in env._pocket_centers:
        ax.add_patch(plt.Circle((pc[0], pc[1]), 0.06, color=POCKET_CLR, zorder=3))
        ax.add_patch(plt.Circle((pc[0], pc[1]), 0.05, color="#222222", zorder=3))

    # ── Trajectories ──────────────────────────────────────────────────────────
    for ball_id, clr, start in [("cue", CUE_CLR, cue_i), ("8", "#888888", target_i)]:
        ball = env.system.balls[ball_id]
        traj = _trajectory(ball)

        if traj is not None and len(traj) > 1:
            # Filter out points outside table (pocketed ball teleports to (0,0))
            mask = (traj[:, 0] >= -0.05) & (traj[:, 0] <= W + 0.05) & \
                   (traj[:, 1] >= -0.05) & (traj[:, 1] <= L + 0.05)
            traj = traj[mask]
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1],
                        color=clr, alpha=0.55, linewidth=1.5,
                        solid_capstyle="round", zorder=4)
        else:
            # Fallback: dashed line start → end
            end = ball.state.rvw[0, :2]
            if not (ball_id == "8" and pocketed):
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color=clr, alpha=0.45, linewidth=1.5,
                        linestyle="--", dashes=(5, 3), zorder=4)

    # ── Aim arrow ─────────────────────────────────────────────────────────────
    arrow_len = min(0.18, np.linalg.norm(target_i - cue_i) * 0.35)
    ax.annotate(
        "", xy=(cue_i[0] + np.cos(phi_shot) * arrow_len,
                cue_i[1] + np.sin(phi_shot) * arrow_len),
        xytext=(cue_i[0], cue_i[1]),
        arrowprops=dict(arrowstyle="-|>", color=AIM_CLR,
                        lw=1.2, mutation_scale=8),
        zorder=6,
    )

    # ── Initial ball positions ────────────────────────────────────────────────
    # Cue ball
    ax.add_patch(plt.Circle(cue_i, BALL_R, color=CUE_CLR,
                            ec="#999999", lw=0.8, zorder=7))

    # 8-ball (black with white "8" label)
    ax.add_patch(plt.Circle(target_i, BALL_R, color=EIGHT_CLR,
                            ec="#444444", lw=0.8, zorder=7))
    ax.text(target_i[0], target_i[1], "8",
            ha="center", va="center", fontsize=5.5,
            fontweight="bold", color="white", zorder=8)

    # Pocket destination marker (if pocketed)
    if pocketed:
        nearest_pocket = env._pocket_centers[
            np.argmin(np.linalg.norm(env._pocket_centers - target_i, axis=1))
        ]
        ax.add_patch(plt.Circle(nearest_pocket, 0.04,
                                color=HIT_CLR, alpha=0.9, zorder=3))

    # ── Title ─────────────────────────────────────────────────────────────────
    label = "✓  POCKETED" if pocketed else "✗  MISSED"
    spd_label = f"  δ={np.degrees(delta_angle):.0f}°  v={speed:.1f}m/s"
    ax.set_title(label + spd_label,
                 color=HIT_CLR if pocketed else MISS_CLR,
                 fontsize=7.5, fontweight="bold", pad=3)

    return pocketed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize billiards RL agent shots")
    parser.add_argument("--model",  default="logs/best_model/best_model",
                        help="Model path (without .zip)")
    parser.add_argument("--shots",  type=int, default=20,
                        help="Number of shots to render")
    parser.add_argument("--cols",   type=int, default=5,
                        help="Grid columns")
    parser.add_argument("--out",    default="outputs/agent_visualization.png",
                        help="Output image filename")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    n_shots = args.shots
    n_cols  = args.cols
    n_rows  = (n_shots + n_cols - 1) // n_cols   # ceil division

    print(f"Loading model: {args.model} ...")
    model = SAC.load(args.model)
    env   = BilliardsEnv()
    rng   = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.0, n_rows * 5.2))
    fig.patch.set_facecolor(BG_CLR)

    # Flatten and hide any extra axes
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n_shots:]:
        ax.set_visible(False)

    hit = 0
    for i, ax in enumerate(axes_flat[:n_shots]):
        seed = int(rng.integers(0, 2**31)) if args.seed is not None else None
        p = draw_shot(ax, env, model, seed=seed)
        hit += int(p)
        sys.stdout.write(f"\r  Rendering shot {i+1}/{n_shots}  ({hit} pocketed)")
        sys.stdout.flush()

    pct = hit / n_shots * 100
    fig.suptitle(
        f"SAC agent  ·  {hit} / {n_shots} shots pocketed  ({pct:.0f}%)\n"
        f"White = cue ball,  Black = 8-ball,  Yellow arrow = aim direction",
        color="white", fontsize=11, y=1.01, fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    import os; os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=BG_CLR)
    print(f"\n\nSaved → {args.out}  ({hit}/{n_shots} pocketed, {pct:.0f}%)")


if __name__ == "__main__":
    main()
