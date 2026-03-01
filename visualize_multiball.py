#!/usr/bin/env python3
"""
visualize_multiball.py — Visualize multi-ball episodes (random or trained agent).

Each subplot = one full episode (all shots overlaid on the table).
Ball trajectories fade in opacity as steps progress.

Usage:
    python visualize_multiball.py                      # random agent, 9 episodes
    python visualize_multiball.py --episodes 6         # fewer episodes
    python visualize_multiball.py --model <path> --algo TQC
"""

import argparse, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simulator import BilliardsEnv

# ── Visual constants ───────────────────────────────────────────────────────────
FELT_COLOR   = "#1a7a3a"
RAIL_COLOR   = "#5c3a1e"
POCKET_CLR   = "#0a0a0a"
CUE_CLR      = "#f0f0f0"
BG_CLR       = "#111111"
BALL_COLORS  = ["#e74c3c", "#3498db", "#f1c40f"]   # red, blue, yellow
BALL_R       = 0.028575


def _trajectory(ball):
    """Extract (N, 2) xy trajectory from a pooltool ball's history."""
    for attr in ("history_cts", "history"):
        h = getattr(ball, attr, None)
        if h is None:
            continue
        states = getattr(h, "states", None)
        if states and len(states) > 1:
            return np.array([s.rvw[0, :2] for s in states])
    return None


def draw_episode(ax, env, policy_fn, seed=None):
    """Run one full episode and draw all shots on ax."""
    obs, _ = env.reset(seed=seed)

    W = env.table_width
    L = env.table_length

    # Snapshot initial positions before any shot
    init_cue = env.system.balls["cue"].state.rvw[0, :2].copy()
    init_balls = {
        bid: env.system.balls[bid].state.rvw[0, :2].copy()
        for bid in env._ball_ids
    }

    # ── Table background ──────────────────────────────────────────────────────
    ax.set_facecolor(RAIL_COLOR)
    ax.set_xlim(-0.07, W + 0.07)
    ax.set_ylim(-0.07, L + 0.07)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), W, L, color=FELT_COLOR, zorder=1))

    for lw, clr, inset in [(6, RAIL_COLOR, -0.04), (1.5, "#2e9c50", 0.02)]:
        ax.add_patch(patches.Rectangle(
            (inset, inset), W - 2*inset, L - 2*inset,
            linewidth=lw, edgecolor=clr, facecolor="none", zorder=2
        ))

    for pc in env._pocket_centers:
        ax.add_patch(plt.Circle((pc[0], pc[1]), 0.055, color=POCKET_CLR, zorder=3))

    # ── Run episode, collecting per-step trajectories ─────────────────────────
    step_data = []
    done = False
    steps = 0
    ep_reward = 0.0

    while not done:
        action = policy_fn(obs)
        # Save system reference BEFORE step() — _respawn_cue() may replace
        # env.system on scratch, destroying trajectory history.
        system_before = env.system
        obs, r, term, trunc, info = env.step(action)
        ep_reward += r
        steps += 1
        done = term or trunc

        # Read trajectories from the pre-step system (has simulate() history)
        cue_traj = _trajectory(system_before.balls["cue"])
        ball_trajs = {}
        for bid in env._ball_ids:
            if bid in system_before.balls:
                t = _trajectory(system_before.balls[bid])
                if t is not None:
                    ball_trajs[bid] = t

        step_data.append({
            "step":       steps,
            "cue_traj":   cue_traj,
            "ball_trajs": ball_trajs,
        })

    # ── Draw trajectories (early steps faded, later steps more opaque) ────────
    def clip_traj(traj):
        if traj is None or len(traj) < 2:
            return None
        mask = ((traj[:, 0] >= -0.05) & (traj[:, 0] <= W + 0.05) &
                (traj[:, 1] >= -0.05) & (traj[:, 1] <= L + 0.05))
        t = traj[mask]
        return t if len(t) > 1 else None

    for sd in step_data:
        alpha = 0.25 + 0.65 * (sd["step"] / max(steps, 1))

        t = clip_traj(sd["cue_traj"])
        if t is not None:
            ax.plot(t[:, 0], t[:, 1], color=CUE_CLR,
                    alpha=alpha * 0.75, linewidth=1.2, zorder=4)

        for i, bid in enumerate(env._ball_ids):
            t = clip_traj(sd["ball_trajs"].get(bid))
            if t is not None:
                ax.plot(t[:, 0], t[:, 1], color=BALL_COLORS[i],
                        alpha=alpha * 0.65, linewidth=1.2, zorder=4)

    # ── Draw initial ball positions ───────────────────────────────────────────
    ax.add_patch(plt.Circle(init_cue, BALL_R,
                            color=CUE_CLR, ec="#aaaaaa", lw=0.8, zorder=7))

    total_pocketed = sum(env._pocketed.values())
    for i, bid in enumerate(env._ball_ids):
        pos     = init_balls[bid]
        pocketed = env._pocketed[bid]
        ec      = "#44ff44" if pocketed else "#888888"
        lw      = 1.8 if pocketed else 0.8
        ax.add_patch(plt.Circle(pos, BALL_R,
                                color=BALL_COLORS[i], ec=ec, lw=lw, zorder=7))
        ax.text(pos[0], pos[1], bid,
                ha="center", va="center",
                fontsize=5, fontweight="bold", color="white", zorder=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    color = "#00e676" if total_pocketed == 3 else \
            ("#ffb300" if total_pocketed > 0 else "#ff5555")
    ax.set_title(
        f"{total_pocketed}/3 pocketed  ·  {steps} steps  ·  r={ep_reward:.2f}",
        color=color, fontsize=7, fontweight="bold", pad=3
    )

    return total_pocketed, steps, ep_reward


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=9,
                        help="Number of episodes to show")
    parser.add_argument("--cols",     type=int, default=3,
                        help="Grid columns")
    parser.add_argument("--model",    default=None,
                        help="Trained model path (without .zip). Omit for random agent.")
    parser.add_argument("--algo",     default="TQC",
                        choices=["SAC", "TQC", "PPO"],
                        help="Algorithm (needed when --model is set)")
    parser.add_argument("--out",      default="outputs/multiball_random.png")
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()

    n_episodes = args.episodes
    n_cols     = args.cols
    n_rows     = (n_episodes + n_cols - 1) // n_cols

    # ── Policy ────────────────────────────────────────────────────────────────
    env = BilliardsEnv(n_balls=3)

    if args.model:
        if args.algo == "TQC":
            from sb3_contrib import TQC
            model = TQC.load(args.model)
        elif args.algo == "SAC":
            from stable_baselines3 import SAC
            model = SAC.load(args.model)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
        agent_label = f"{args.algo} (trained)"
    else:
        rng = np.random.default_rng(args.seed)
        policy_fn = lambda obs: env.action_space.sample()
        agent_label = "Random agent"

    # ── Plot grid ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 5.5))
    fig.patch.set_facecolor(BG_CLR)

    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n_episodes:]:
        ax.set_visible(False)

    rng = np.random.default_rng(args.seed)
    total_pocketed_all, total_clears = 0, 0
    avg_steps = []

    for i, ax in enumerate(axes_flat[:n_episodes]):
        seed = int(rng.integers(0, 2**31))
        p, s, r = draw_episode(ax, env, policy_fn, seed=seed)
        total_pocketed_all += p
        total_clears       += int(p == 3)
        avg_steps.append(s)
        sys.stdout.write(
            f"\r  Episode {i+1}/{n_episodes}  "
            f"({total_pocketed_all} balls pocketed, {total_clears} clears)"
        )
        sys.stdout.flush()

    mean_pocketed = total_pocketed_all / n_episodes
    clear_rate    = total_clears / n_episodes * 100
    mean_steps    = sum(avg_steps) / len(avg_steps)

    fig.suptitle(
        f"{agent_label}  ·  3-ball env (max 15 steps)\n"
        f"avg {mean_pocketed:.1f}/3 pocketed  ·  clear rate {clear_rate:.0f}%  "
        f"·  avg {mean_steps:.1f} steps/episode",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=BG_CLR)
    print(f"\n\nSaved → {args.out}")
    print(f"  avg pocketed : {mean_pocketed:.2f} / 3")
    print(f"  clear rate   : {clear_rate:.0f}%  ({total_clears}/{n_episodes})")
    print(f"  avg steps    : {mean_steps:.1f}")


if __name__ == "__main__":
    main()
