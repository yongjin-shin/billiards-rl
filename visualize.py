#!/usr/bin/env python3
"""
visualize.py — Unified visualization for billiards-rl.

Supports single-ball (n_balls=1) and multi-ball (n_balls=3), image grid or MP4 video.
Without --model, uses a random agent.

Usage:
    python visualize.py                                         # random, 1 ball, image
    python visualize.py --n-balls 3                            # random, 3 balls, image
    python visualize.py --mode video                           # random, 1 ball, video
    python visualize.py --mode video --n-balls 3               # random, 3 balls, video
    python visualize.py --model <path> --algo TQC              # trained TQC, image
    python visualize.py --model <path> --algo SAC --mode video # trained SAC, video

    # Before/after comparison (concatenates two MP4s with title cards):
    python visualize.py --mode compare --before before.mp4 --after after.mp4
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pooltool as pt

from simulator import BilliardsEnv

# ── Visual constants ───────────────────────────────────────────────────────────
FPS         = 30
BALL_R      = 0.028575
FELT_COLOR  = "#1a7a3a"
RAIL_COLOR  = "#5c3a1e"
POCKET_CLR  = "#0a0a0a"
CUE_CLR     = "#f0f0f0"
AIM_CLR     = "#ffff00"
HIT_CLR     = "#00e676"
MISS_CLR    = "#ff4444"
BG_CLR      = "#111111"

# One color per target ball (supports up to 3)
BALL_COLORS = ["#e74c3c", "#3498db", "#f1c40f"]   # red, blue, yellow


# =============================================================================
# Shared helpers
# =============================================================================

def load_model(model_path, algo):
    if algo == "TQC":
        from sb3_contrib import TQC
        return TQC.load(model_path)
    elif algo == "SAC":
        from stable_baselines3 import SAC
        return SAC.load(model_path)
    else:
        from stable_baselines3 import PPO
        return PPO.load(model_path)


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


def extract_trajectory_timed(ball):
    """Return (t_array, xy_array) for physics-time animation."""
    try:
        states = ball.history.states
        if not states:
            raise ValueError
        t_arr  = np.array([s.t for s in states], dtype=float)
        xy_arr = np.array([s.rvw[0, :2] for s in states], dtype=float)
        return t_arr, xy_arr
    except Exception:
        pos = ball.state.rvw[0, :2]
        return np.array([0.0, 0.0]), np.vstack([pos, pos])


def interp_xy(t_query, t_arr, xy_arr):
    """Linear interpolation of xy at scalar time t_query."""
    if t_query <= t_arr[0]:  return xy_arr[0].copy()
    if t_query >= t_arr[-1]: return xy_arr[-1].copy()
    idx  = np.clip(np.searchsorted(t_arr, t_query) - 1, 0, len(t_arr) - 2)
    frac = (t_query - t_arr[idx]) / max(t_arr[idx + 1] - t_arr[idx], 1e-12)
    return (1 - frac) * xy_arr[idx] + frac * xy_arr[idx + 1]


def _clip_traj(traj, W, L):
    """Remove out-of-bounds points from trajectory array."""
    if traj is None or len(traj) < 2:
        return None
    mask = ((traj[:, 0] >= -0.05) & (traj[:, 0] <= W + 0.05) &
            (traj[:, 1] >= -0.05) & (traj[:, 1] <= L + 0.05))
    t = traj[mask]
    return t if len(t) > 1 else None


def _draw_table_bg(ax, env):
    """Draw felt, rails, and pockets onto ax."""
    W, L = env.table_width, env.table_length
    ax.set_facecolor(RAIL_COLOR)
    ax.set_xlim(-0.07, W + 0.07)
    ax.set_ylim(-0.07, L + 0.07)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), W, L, color=FELT_COLOR, zorder=1))
    for lw, clr, inset in [(6, RAIL_COLOR, -0.04), (1.5, "#2e9c50", 0.02)]:
        ax.add_patch(patches.Rectangle(
            (inset, inset), W - 2*inset, L - 2*inset,
            linewidth=lw, edgecolor=clr, facecolor="none", zorder=2))
    for pc in env._pocket_centers:
        ax.add_patch(plt.Circle((pc[0], pc[1]), 0.055, color=POCKET_CLR, zorder=3))


# =============================================================================
# Image mode
# =============================================================================

def draw_episode(ax, env, policy_fn, seed=None):
    """
    Run one episode and draw everything on ax.
    Works for n_balls=1 (single shot) and n_balls>1 (multi-step).
    Returns (total_pocketed, steps, ep_reward).
    """
    obs, _ = env.reset(seed=seed)
    W, L   = env.table_width, env.table_length

    init_cue   = env.system.balls["cue"].state.rvw[0, :2].copy()
    init_balls = {bid: env.system.balls[bid].state.rvw[0, :2].copy()
                  for bid in env._ball_ids}

    _draw_table_bg(ax, env)

    # Run episode and collect per-step trajectory data
    step_data    = []
    first_action = None
    phi_shot     = 0.0
    done = False
    steps, ep_reward = 0, 0.0

    while not done:
        action = policy_fn(obs)

        if first_action is None:
            first_action = action.copy()
            # Aim direction for single-ball aim arrow
            cue_pos = env.system.balls["cue"].state.rvw[0, :2]
            ref_pos = env.system.balls[env._ball_ids[0]].state.rvw[0, :2]
            phi_direct = np.arctan2(ref_pos[1] - cue_pos[1], ref_pos[0] - cue_pos[0])
            phi_shot   = phi_direct + float(action[0])

        system_before = env.system          # capture ref BEFORE step (scratch replaces system)
        obs, r, term, trunc, info = env.step(action)
        ep_reward += r
        steps += 1
        done = term or trunc

        cue_traj = _trajectory(system_before.balls["cue"])
        ball_trajs = {}
        for bid in env._ball_ids:
            if bid in system_before.balls:
                t = _trajectory(system_before.balls[bid])
                if t is not None:
                    ball_trajs[bid] = t
        step_data.append({"step": steps, "cue_traj": cue_traj, "ball_trajs": ball_trajs})

    # ── Draw trajectories (earlier steps more transparent) ────────────────────
    for sd in step_data:
        alpha = 0.25 + 0.65 * (sd["step"] / max(steps, 1))

        t = _clip_traj(sd["cue_traj"], W, L)
        if t is not None:
            ax.plot(t[:, 0], t[:, 1], color=CUE_CLR,
                    alpha=alpha * 0.75, linewidth=1.2, zorder=4)

        for i, bid in enumerate(env._ball_ids):
            t = _clip_traj(sd["ball_trajs"].get(bid), W, L)
            if t is not None:
                ax.plot(t[:, 0], t[:, 1], color=BALL_COLORS[i],
                        alpha=alpha * 0.65, linewidth=1.2, zorder=4)

    # ── Aim arrow (single-ball only — one shot, clear direction) ─────────────
    if env.n_balls == 1 and first_action is not None:
        ref = init_balls[env._ball_ids[0]]
        arrow_len = min(0.18, np.linalg.norm(ref - init_cue) * 0.35)
        ax.annotate("",
            xy=(init_cue[0] + np.cos(phi_shot) * arrow_len,
                init_cue[1] + np.sin(phi_shot) * arrow_len),
            xytext=tuple(init_cue),
            arrowprops=dict(arrowstyle="-|>", color=AIM_CLR, lw=1.2, mutation_scale=8),
            zorder=6)

    # ── Ball positions (initial) ───────────────────────────────────────────────
    ax.add_patch(plt.Circle(init_cue, BALL_R,
                            color=CUE_CLR, ec="#aaaaaa", lw=0.8, zorder=7))
    total_pocketed = sum(env._pocketed.values())
    for i, bid in enumerate(env._ball_ids):
        pos      = init_balls[bid]
        pocketed = env._pocketed[bid]
        ax.add_patch(plt.Circle(pos, BALL_R, color=BALL_COLORS[i],
                                ec=HIT_CLR if pocketed else "#888888",
                                lw=1.8 if pocketed else 0.8, zorder=7))
        ax.text(pos[0], pos[1], bid,
                ha="center", va="center",
                fontsize=5, fontweight="bold", color="white", zorder=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    if env.n_balls == 1:
        pocketed = total_pocketed > 0
        da = float(first_action[0]) if first_action is not None else 0
        sp = float(first_action[1]) if first_action is not None else 0
        color = HIT_CLR if pocketed else MISS_CLR
        title = ("✓ POCKETED" if pocketed else "✗ MISSED") + \
                f"  δ={np.degrees(da):.0f}°  v={sp:.1f}m/s"
    else:
        color = HIT_CLR if total_pocketed == 3 else \
                ("#ffb300" if total_pocketed > 0 else MISS_CLR)
        title = f"{total_pocketed}/3 pocketed  ·  {steps} steps  ·  r={ep_reward:.2f}"

    ax.set_title(title, color=color, fontsize=7, fontweight="bold", pad=3)
    return total_pocketed, steps, ep_reward


def render_image(env, policy_fn, args):
    n      = args.count
    n_cols = args.cols
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 5.5))
    fig.patch.set_facecolor(BG_CLR)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    rng = np.random.default_rng(args.seed)
    total_pocketed, total_clears, step_list = 0, 0, []

    for i, ax in enumerate(axes_flat[:n]):
        seed = int(rng.integers(0, 2**31))
        p, s, _ = draw_episode(ax, env, policy_fn, seed=seed)
        total_pocketed += p
        n_target = env.n_balls
        total_clears += int(p == n_target)
        step_list.append(s)
        sys.stdout.write(f"\r  {i+1}/{n}  ({total_pocketed} pocketed, {total_clears} clears)")
        sys.stdout.flush()

    mean_p     = total_pocketed / n
    clear_rate = total_clears / n * 100
    mean_steps = sum(step_list) / len(step_list)
    agent_lbl  = args.model if args.model else "Random agent"

    if env.n_balls == 1:
        suptitle = (f"{agent_lbl}  ·  1-ball env\n"
                    f"pocket rate {total_pocketed}/{n} ({total_pocketed/n*100:.0f}%)")
    else:
        suptitle = (f"{agent_lbl}  ·  {env.n_balls}-ball env (max {env.max_steps} steps)\n"
                    f"avg {mean_p:.1f}/{env.n_balls} pocketed  ·  "
                    f"clear rate {clear_rate:.0f}%  ·  avg {mean_steps:.1f} steps/episode")

    fig.suptitle(suptitle, color="white", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=BG_CLR)
    print(f"\n\nSaved → {args.out}")
    if env.n_balls == 1:
        print(f"  pocket rate : {total_pocketed}/{n} ({total_pocketed/n*100:.0f}%)")
    else:
        print(f"  avg pocketed : {mean_p:.2f}/{env.n_balls}")
        print(f"  clear rate   : {clear_rate:.0f}%  ({total_clears}/{n})")
        print(f"  avg steps    : {mean_steps:.1f}")


# =============================================================================
# Video mode
# =============================================================================

def _clip_end_time(traj_dict):
    """Estimate when meaningful motion ends (last time any ball moved)."""
    ends = []
    for t_arr, xy_arr in traj_dict.values():
        if len(t_arr) < 2:
            continue
        diffs = np.diff(xy_arr, axis=0)
        moving = np.where(diffs.any(axis=1))[0]
        if len(moving):
            ends.append(t_arr[moving[-1] + 1])
        else:
            ends.append(t_arr[0])
    if not ends:
        return 0.5
    return min(max(ends) + 0.25, 8.0)


def run_episode_shots(env, policy_fn):
    """
    Run one full episode and collect physics data for each shot.
    Returns list of shot dicts ready for animation.
    """
    obs, _ = env.reset()
    shots  = []
    done   = False

    while not done:
        action = policy_fn(obs)

        cue_pos = env.system.balls["cue"].state.rvw[0, :2].copy()
        ref_id  = env._ball_ids[0]
        for bid in env._ball_ids:
            if not env._pocketed.get(bid, False) and bid in env.system.balls:
                ref_id = bid
                break
        ref_pos     = env.system.balls[ref_id].state.rvw[0, :2]
        phi_direct  = np.arctan2(ref_pos[1] - cue_pos[1], ref_pos[0] - cue_pos[0])
        phi_shot    = phi_direct + float(action[0])

        # Active ball positions before shot
        init_pos = {bid: env.system.balls[bid].state.rvw[0, :2].copy()
                    for bid in env._ball_ids
                    if bid in env.system.balls and not env._pocketed.get(bid, False)}
        init_cue = env.system.balls["cue"].state.rvw[0, :2].copy()

        system_before = env.system
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

        # Collect timed trajectories from pre-step system
        traj_dict = {"cue": extract_trajectory_timed(system_before.balls["cue"])}
        for bid in env._ball_ids:
            if bid in system_before.balls:
                traj_dict[bid] = extract_trajectory_timed(system_before.balls[bid])

        scratch      = info.get("scratch", False)
        new_cue_pos  = (env.system.balls["cue"].state.rvw[0, :2].copy()
                        if scratch and not done else None)
        pocketed_now = dict(env._pocketed)

        shots.append(dict(
            phi_shot      = phi_shot,
            delta_angle   = float(action[0]),
            speed         = float(action[1]),
            init_cue      = init_cue,
            init_pos      = init_pos,
            traj_dict     = traj_dict,
            clip_end      = _clip_end_time(traj_dict),
            scratch       = scratch,
            new_cue_pos   = new_cue_pos,
            pocketed_now  = pocketed_now,
            reward        = r,
            info          = info,
        ))

    return shots


class TableRenderer:
    """Animatable table renderer. Handles cue + up to 3 target balls."""

    def __init__(self, env, dpi=130):
        W, L      = env.table_width, env.table_length
        self.W, self.L = W, L
        self.pc   = env._pocket_centers
        self.n_balls = env.n_balls
        self._ball_ids = env._ball_ids

        ratio  = L / W
        fig_w  = 5.0
        fig_h  = fig_w * ratio + 1.4
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        self.fig.patch.set_facecolor(BG_CLR)
        self.ax.set_position([0.05, 0.08, 0.90, 0.84])
        self._draw_static()

        # Cue ball
        self.cue_patch = plt.Circle((0, 0), BALL_R, color=CUE_CLR, ec="#999", lw=0.8, zorder=7)
        self.ax.add_patch(self.cue_patch)
        self.cue_line, = self.ax.plot([], [], color=CUE_CLR, alpha=0.45, lw=1.4, zorder=4)

        # Target balls
        self.ball_patches, self.ball_labels, self.ball_lines = [], [], []
        for i, bid in enumerate(self._ball_ids):
            p = plt.Circle((0, 0), BALL_R, color=BALL_COLORS[i], ec="#888", lw=0.8, zorder=7)
            self.ax.add_patch(p)
            lbl = self.ax.text(0, 0, bid, ha="center", va="center",
                               fontsize=6, fontweight="bold", color="white", zorder=8)
            line, = self.ax.plot([], [], color=BALL_COLORS[i], alpha=0.45, lw=1.4, zorder=4)
            self.ball_patches.append(p)
            self.ball_labels.append(lbl)
            self.ball_lines.append(line)

        # Aim arrow
        self.aim_arrow = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=AIM_CLR, lw=1.2, mutation_scale=9),
            zorder=6)

        self.title_txt = self.fig.text(0.5, 0.97, "", ha="center", va="top",
                                       fontsize=11, color="white", fontweight="bold")
        self.info_txt  = self.fig.text(0.5, 0.03, "", ha="center", va="bottom",
                                       fontsize=9, color="#cccccc")

    def _draw_static(self):
        ax = self.ax
        ax.set_facecolor(RAIL_COLOR)
        ax.set_xlim(-0.07, self.W + 0.07)
        ax.set_ylim(-0.07, self.L + 0.07)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(patches.Rectangle((0, 0), self.W, self.L, color=FELT_COLOR, zorder=1))
        ax.add_patch(patches.Rectangle((0.012, 0.012),
                                       self.W - 0.024, self.L - 0.024,
                                       lw=1.5, edgecolor="#2e9c50",
                                       facecolor="none", zorder=2))
        for pc in self.pc:
            ax.add_patch(plt.Circle((pc[0], pc[1]), 0.062, color="#060606", zorder=3))
            ax.add_patch(plt.Circle((pc[0], pc[1]), 0.050, color=POCKET_CLR, zorder=3))

    def render_frame(self, shot, t_real, trail_cue, ball_trails, pocketed_state,
                     shot_num, running_info):
        W, L = self.W, self.L
        cue_t,  cue_xy  = shot["traj_dict"]["cue"]
        cue_pos = interp_xy(t_real, cue_t, cue_xy)

        self.cue_patch.center = cue_pos
        if len(trail_cue) > 1:
            self.cue_line.set_data(trail_cue[:, 0], trail_cue[:, 1])
        else:
            self.cue_line.set_data([], [])

        for i, bid in enumerate(self._ball_ids):
            pocketed = pocketed_state.get(bid, False)
            if pocketed:
                self.ball_patches[i].set_visible(False)
                self.ball_labels[i].set_visible(False)
                self.ball_lines[i].set_data([], [])
                continue

            if bid in shot["traj_dict"]:
                bt, bxy = shot["traj_dict"][bid]
                bpos = interp_xy(t_real, bt, bxy)
                near_pocket = np.min(np.linalg.norm(self.pc - bpos, axis=1)) < 0.08
                visible = not near_pocket
            else:
                bpos    = shot["init_pos"].get(bid, np.array([0.0, 0.0]))
                visible = True

            self.ball_patches[i].center = bpos
            self.ball_patches[i].set_visible(visible)
            self.ball_labels[i].set_position(bpos)
            self.ball_labels[i].set_visible(visible)

            tr = ball_trails.get(bid)
            if tr is not None and len(tr) > 1 and visible:
                self.ball_lines[i].set_data(tr[:, 0], tr[:, 1])
            else:
                self.ball_lines[i].set_data([], [])

        # Aim arrow (only at start of each shot)
        arrow_len = 0.14
        ax_end    = (shot["init_cue"][0] + np.cos(shot["phi_shot"]) * arrow_len,
                     shot["init_cue"][1] + np.sin(shot["phi_shot"]) * arrow_len)
        self.aim_arrow.xy     = ax_end
        self.aim_arrow.xytext = tuple(shot["init_cue"])
        self.aim_arrow.set_visible(t_real < 0.06)

        # Texts
        self.title_txt.set_text(running_info["title"])
        self.title_txt.set_color(running_info["title_color"])
        self.info_txt.set_text(
            f"Shot {shot_num}  |  "
            f"δ={np.degrees(shot['delta_angle']):.0f}°  "
            f"v={shot['speed']:.1f} m/s  |  {running_info['stat']}"
        )

        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf  = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, :3]

    def close(self):
        plt.close(self.fig)


def _animate_shot(writer, renderer, shot, shot_num, pocketed_state,
                  running_info, slow, pause_sec):
    """Animate a single shot and append frames to writer."""
    cue_t, cue_xy = shot["traj_dict"]["cue"]
    clip_end = shot["clip_end"]
    dt_real  = 1.0 / (FPS * slow)
    t_frames = np.arange(0, clip_end + dt_real, dt_real)

    trail_cue   = np.empty((0, 2))
    ball_trails = {bid: np.empty((0, 2)) for bid in renderer._ball_ids}

    for t_real in t_frames:
        trail_cue = np.vstack([trail_cue, interp_xy(t_real, cue_t, cue_xy)])

        for bid in renderer._ball_ids:
            if bid in shot["traj_dict"] and not pocketed_state.get(bid, False):
                bt, bxy = shot["traj_dict"][bid]
                ball_trails[bid] = np.vstack([ball_trails[bid], interp_xy(t_real, bt, bxy)])

        frame = renderer.render_frame(shot, t_real, trail_cue, ball_trails,
                                      pocketed_state, shot_num, running_info)
        writer.append_data(frame)

    # Pause on result
    last = renderer.render_frame(shot, clip_end, trail_cue, ball_trails,
                                 pocketed_state, shot_num, running_info)
    for _ in range(int(pause_sec * FPS)):
        writer.append_data(last)


def render_video(env, policy_fn, args):
    import imageio.v2 as imageio

    n_episodes = args.count
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    renderer = TableRenderer(env, dpi=130)
    writer   = imageio.get_writer(args.out, fps=FPS, codec="libx264",
                                  output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    total_pocketed, total_clears = 0, 0
    global_shot = 0

    for ep in range(n_episodes):
        shots = run_episode_shots(env, policy_fn)
        ep_pocketed = sum(s["pocketed_now"].get(bid, False)
                          for s in shots[-1:]
                          for bid in env._ball_ids) if shots else 0
        # Use final pocketed state from last shot
        final_pocketed = shots[-1]["pocketed_now"] if shots else {}
        ep_pocketed = sum(final_pocketed.values())
        total_pocketed += ep_pocketed
        total_clears   += int(ep_pocketed == env.n_balls)

        # Animate each shot within the episode
        pocketed_state = {bid: False for bid in env._ball_ids}  # tracks during episode

        for s_idx, shot in enumerate(shots):
            global_shot += 1
            # Update pocketed state to what was true BEFORE this shot
            # (so we animate balls that were still active)
            prev_pocketed = {}
            for bid in env._ball_ids:
                # A ball was pocketed before this shot if it's not in init_pos
                prev_pocketed[bid] = bid not in shot["init_pos"]
            pocketed_state_before = prev_pocketed

            if env.n_balls == 1:
                info_stat    = (f"{total_pocketed}/{global_shot} pocketed "
                                f"({total_pocketed/global_shot*100:.0f}%)")
                newly_p      = shot["pocketed_now"].get(env._ball_ids[0], False)
                title        = "✓ POCKETED" if newly_p else "✗ MISSED"
                title_color  = HIT_CLR if newly_p else MISS_CLR
            else:
                cur_p        = sum(shot["pocketed_now"].values())
                info_stat    = f"ep {ep+1}/{n_episodes}  ·  {cur_p}/{env.n_balls} pocketed"
                title        = f"Episode {ep+1}"
                title_color  = "white"

            running_info = {"title": title, "title_color": title_color, "stat": info_stat}
            _animate_shot(writer, renderer, shot, global_shot,
                          pocketed_state_before, running_info, args.slow, args.pause)

            # Update pocketed_state for next shot
            pocketed_state = dict(shot["pocketed_now"])

        sys.stdout.write(
            f"\r  Episode {ep+1}/{n_episodes}  "
            f"({ep_pocketed}/{env.n_balls} pocketed, {total_clears} clears)"
        )
        sys.stdout.flush()

    writer.close()
    renderer.close()

    print(f"\n\nSaved → {args.out}")
    if env.n_balls == 1:
        print(f"  pocket rate : {total_pocketed}/{global_shot} ({total_pocketed/max(global_shot,1)*100:.0f}%)")
    else:
        print(f"  avg pocketed : {total_pocketed/n_episodes:.2f}/{env.n_balls}")
        print(f"  clear rate   : {total_clears/n_episodes*100:.0f}%  ({total_clears}/{n_episodes})")


# =============================================================================
# Compare mode — concatenate two MP4s with title cards
# =============================================================================

def _make_title_frames(text, subtext, w, h, n_frames, bg="#111111",
                       text_color="white", sub_color="#aaaaaa"):
    """Render a title card as a list of identical RGB frames."""
    dpi   = 100
    fig_w = w / dpi
    fig_h = h / dpi
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg)
    fig.text(0.5, 0.58, text,    ha="center", va="center",
             fontsize=22, fontweight="bold", color=text_color)
    fig.text(0.5, 0.42, subtext, ha="center", va="center",
             fontsize=13, color=sub_color)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return [frame] * n_frames


def render_compare(args):
    """Concatenate before.mp4 + after.mp4 with title cards into one video."""
    import imageio.v2 as imageio

    if not args.before or not args.after:
        print("ERROR: --mode compare requires --before <path> and --after <path>")
        sys.exit(1)

    def _count_and_meta(path):
        r = imageio.get_reader(path)
        meta = r.get_meta_data()
        count = 0
        first = None
        for f in r:
            if first is None:
                first = f
            count += 1
        r.close()
        return meta, count, first

    print(f"Scanning before: {args.before}")
    meta_b, n_b, first_b = _count_and_meta(args.before)
    print(f"Scanning after : {args.after}")
    meta_a, n_a, first_a = _count_and_meta(args.after)

    h, w = first_b.shape[:2]
    fps  = int(meta_b.get("fps", FPS))
    card_sec = getattr(args, "card_sec", 2.5)
    n_card   = int(card_sec * fps)

    before_label = getattr(args, "before_label", "Before  ·  Random Agent")
    after_label  = getattr(args, "after_label",  "After  ·  Trained SAC")
    before_sub   = f"{n_b // fps}s  ·  {n_b} frames"
    after_sub    = f"{n_a // fps}s  ·  {n_a} frames"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264",
                                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    # Title card → before section (stream frames one by one)
    print("Writing before section ...")
    for f in _make_title_frames(before_label, before_sub, w, h, n_card):
        writer.append_data(f)
    reader_b = imageio.get_reader(args.before)
    for i, f in enumerate(reader_b):
        writer.append_data(f)
        if (i + 1) % 100 == 0:
            sys.stdout.write(f"\r  Before: {i+1}/{n_b} frames")
            sys.stdout.flush()
    reader_b.close()
    print(f"\r  Before: {n_b}/{n_b} frames ✓")

    # Short black gap
    gap = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(int(fps * 0.5)):
        writer.append_data(gap)

    # Title card → after section
    print("Writing after section ...")
    for f in _make_title_frames(after_label, after_sub, w, h, n_card,
                                text_color=HIT_CLR):
        writer.append_data(f)
    reader_a = imageio.get_reader(args.after)
    for i, f in enumerate(reader_a):
        writer.append_data(f)
        if (i + 1) % 100 == 0:
            sys.stdout.write(f"\r  After : {i+1}/{n_a} frames")
            sys.stdout.flush()
    reader_a.close()
    print(f"\r  After : {n_a}/{n_a} frames ✓")

    writer.close()
    total_sec = (n_card * 2 + n_b + n_a + int(fps * 0.5)) / fps
    print(f"\nSaved → {args.out}  ({total_sec:.0f}s total)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize billiards-rl agent (image grid or MP4 video)")

    parser.add_argument("--n-balls", type=int, default=1, choices=[1, 3],
                        help="Number of target balls (default: 1)")
    parser.add_argument("--mode",    default="image", choices=["image", "video", "compare"],
                        help="image grid / MP4 video / before-after comparison (default: image)")
    parser.add_argument("--model",   default=None,
                        help="Trained model path without .zip. Omit for random agent.")
    parser.add_argument("--algo",    default="SAC", choices=["SAC", "TQC", "PPO"],
                        help="Algorithm — needed when --model is set (default: SAC)")
    parser.add_argument("--count",   type=int, default=None,
                        help="Episodes/shots to render (default: 9 for image, 6 for video)")
    parser.add_argument("--cols",    type=int, default=3,
                        help="Grid columns for image mode (default: 3)")
    parser.add_argument("--slow",    type=float, default=3.0,
                        help="Slow-motion factor for video (default: 3)")
    parser.add_argument("--pause",   type=float, default=0.8,
                        help="Pause seconds between shots in video (default: 0.8)")
    parser.add_argument("--out",     default=None,
                        help="Output file path (auto-named if omitted)")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Random seed (default: 0)")
    # compare mode args
    parser.add_argument("--before",        default=None, help="Before MP4 path (compare mode)")
    parser.add_argument("--after",         default=None, help="After MP4 path (compare mode)")
    parser.add_argument("--before-label",  default="Before  ·  Random Agent",
                        help="Title text for before section")
    parser.add_argument("--after-label",   default="After  ·  Trained SAC",
                        help="Title text for after section")
    parser.add_argument("--card-sec",      type=float, default=2.5,
                        help="Title card duration in seconds (default: 2.5)")
    args = parser.parse_args()

    # compare mode — just concatenate, no env needed
    if args.mode == "compare":
        if args.out is None:
            args.out = "outputs/comparison.mp4"
        args.before_label = args.before_label
        args.after_label  = args.after_label
        args.card_sec     = args.card_sec
        render_compare(args)
        return

    # Defaults for count and out
    if args.count is None:
        args.count = 9 if args.mode == "image" else 6
    if args.out is None:
        agent = os.path.basename(args.model) if args.model else "random"
        ext   = "png" if args.mode == "image" else "mp4"
        args.out = f"outputs/{args.n_balls}ball_{agent}.{ext}"

    # Environment
    env = BilliardsEnv(n_balls=args.n_balls)

    # Policy
    if args.model:
        print(f"Loading {args.algo} model: {args.model} ...")
        model     = load_model(args.model, args.algo)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        rng       = np.random.default_rng(args.seed)
        policy_fn = lambda obs: env.action_space.sample()
        print(f"Using random agent  |  n_balls={args.n_balls}  mode={args.mode}")

    print(f"Output: {args.out}  ({args.count} {'shots' if args.n_balls == 1 else 'episodes'})")

    if args.mode == "image":
        render_image(env, policy_fn, args)
    else:
        render_video(env, policy_fn, args)


if __name__ == "__main__":
    main()
