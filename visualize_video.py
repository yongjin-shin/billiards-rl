#!/usr/bin/env python3
"""
visualize_video.py — Render trained SAC agent shots as an MP4 video.

Each shot animates at slow motion so ball movement is clearly visible.
Ball positions are interpolated from pooltool's physics event timestamps.

Usage:
    python visualize_video.py                      # 10 shots, best_model
    python visualize_video.py --shots 5 --slow 4  # 5 shots, 4x slow motion
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
import pooltool as pt
from stable_baselines3 import SAC
from simulator import BilliardsEnv


# ── Constants ─────────────────────────────────────────────────────────────────
FPS         = 30
BALL_R      = 0.028575
FELT_COLOR  = "#1a7a3a"
RAIL_COLOR  = "#5c3a1e"
POCKET_CLR  = "#0a0a0a"
CUE_CLR     = "#f0f0f0"
EIGHT_CLR   = "#1a1a1a"
HIT_CLR     = "#00e676"
MISS_CLR    = "#ff4444"
AIM_CLR     = "#ffff00"
BG_CLR      = "#111111"


# ── Physics trajectory extraction ─────────────────────────────────────────────
def extract_trajectory(ball):
    """
    Extract time-indexed (x, y) positions from ball.history.states.
    Returns (t_array, xy_array) both as numpy arrays.
    Falls back to start+end positions if history unavailable.
    """
    try:
        states = ball.history.states
        if not states:
            raise ValueError("empty history")
        t_arr  = np.array([s.t            for s in states], dtype=float)
        xy_arr = np.array([[s.rvw[0, 0], s.rvw[0, 1]] for s in states], dtype=float)
        return t_arr, xy_arr
    except Exception:
        pos = ball.state.rvw[0, :2]
        return np.array([0.0, 0.0]), np.array([pos, pos])


def interp_xy(t_query, t_arr, xy_arr):
    """Linear interpolation of xy at scalar time t_query."""
    if t_query <= t_arr[0]:
        return xy_arr[0].copy()
    if t_query >= t_arr[-1]:
        return xy_arr[-1].copy()
    idx  = np.searchsorted(t_arr, t_query) - 1
    idx  = np.clip(idx, 0, len(t_arr) - 2)
    frac = (t_query - t_arr[idx]) / max(t_arr[idx + 1] - t_arr[idx], 1e-12)
    return (1 - frac) * xy_arr[idx] + frac * xy_arr[idx + 1]


# ── Shot simulation ────────────────────────────────────────────────────────────
def run_shot(env, model):
    """Simulate one shot and return trajectory data + result."""
    obs, _ = env.reset()

    cue_init    = env.system.balls["cue"].state.rvw[0, :2].copy()
    target_init = env.system.balls["8"].state.rvw[0, :2].copy()

    action, _   = model.predict(obs, deterministic=True)
    delta_angle = float(action[0])
    speed       = float(action[1])

    phi_direct  = np.arctan2(target_init[1] - cue_init[1],
                             target_init[0] - cue_init[0])
    phi_shot    = phi_direct + delta_angle

    _, _, _, _, info = env.step(action)
    pocketed = info["pocketed"]

    cue_t,    cue_xy    = extract_trajectory(env.system.balls["cue"])
    eight_t,  eight_xy  = extract_trajectory(env.system.balls["8"])

    # Duration: until last meaningful motion (skip long deceleration tail)
    # Find when 8-ball last moved significantly OR was pocketed
    if pocketed:
        # 8-ball disappears at pocket — clip at that moment
        eight_states = env.system.balls["8"].history.states
        pocket_t = next((s.t for s in eight_states if s.s == pt.constants.pocketed),
                        eight_t[-1])
        clip_end = min(pocket_t + 0.3, eight_t[-1])
    else:
        # Find last moment either ball moved (state != 0)
        ball_end = max(
            cue_t[np.where(np.diff(cue_xy, axis=0).sum(axis=1) != 0)[0][-1] + 1]
                if (np.diff(cue_xy, axis=0) != 0).any() else 0,
            eight_t[np.where(np.diff(eight_xy, axis=0).sum(axis=1) != 0)[0][-1] + 1]
                if (np.diff(eight_xy, axis=0) != 0).any() else 0,
        )
        clip_end = min(ball_end + 0.2, max(cue_t[-1], eight_t[-1]))

    return dict(
        cue_init    = cue_init,
        target_init = target_init,
        phi_shot    = phi_shot,
        delta_angle = delta_angle,
        speed       = speed,
        pocketed    = pocketed,
        cue_t       = cue_t,
        cue_xy      = cue_xy,
        eight_t     = eight_t,
        eight_xy    = eight_xy,
        clip_end    = clip_end,
        pocket_centers = env._pocket_centers.copy(),
        W           = env.table_width,
        L           = env.table_length,
    )


# ── Table renderer ────────────────────────────────────────────────────────────
class TableRenderer:
    def __init__(self, W, L, pocket_centers, dpi=120):
        self.W = W
        self.L = L
        self.pc = pocket_centers

        # Portrait aspect — pool table is taller than wide
        ratio   = L / W
        fig_w   = 5.0
        fig_h   = fig_w * ratio + 1.4    # extra space for text
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        self.fig.patch.set_facecolor(BG_CLR)
        self.ax.set_position([0.05, 0.08, 0.90, 0.84])

        self._draw_static()

        # Movable elements
        self.cue_patch   = plt.Circle((0, 0), BALL_R, color=CUE_CLR,
                                      ec="#999999", lw=0.8, zorder=7)
        self.eight_patch = plt.Circle((0, 0), BALL_R, color=EIGHT_CLR,
                                      ec="#555555", lw=0.8, zorder=7)
        self.eight_lbl   = self.ax.text(0, 0, "8", ha="center", va="center",
                                         fontsize=6, fontweight="bold",
                                         color="white", zorder=8)
        self.ax.add_patch(self.cue_patch)
        self.ax.add_patch(self.eight_patch)

        self.cue_line,   = self.ax.plot([], [], color=CUE_CLR,
                                         alpha=0.45, linewidth=1.4, zorder=4)
        self.eight_line, = self.ax.plot([], [], color="#888888",
                                         alpha=0.45, linewidth=1.4, zorder=4)

        self.aim_arrow = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=AIM_CLR,
                            lw=1.2, mutation_scale=9),
            zorder=6,
        )

        self.title_txt = self.fig.text(0.5, 0.97, "", ha="center", va="top",
                                        fontsize=11, color="white",
                                        fontweight="bold")
        self.info_txt  = self.fig.text(0.5, 0.03, "", ha="center", va="bottom",
                                        fontsize=9, color="#cccccc")

    def _draw_static(self):
        W, L = self.W, self.L
        ax   = self.ax
        ax.set_facecolor(RAIL_COLOR)
        ax.set_xlim(-0.07, W + 0.07)
        ax.set_ylim(-0.07, L + 0.07)
        ax.set_aspect("equal")
        ax.axis("off")

        # Felt surface
        ax.add_patch(patches.Rectangle((0, 0), W, L, color=FELT_COLOR, zorder=1))
        # Inner cushion line
        ax.add_patch(patches.Rectangle((0.012, 0.012), W - 0.024, L - 0.024,
                                        lw=1.5, edgecolor="#2e9c50",
                                        facecolor="none", zorder=2))
        # Pockets
        for pc in self.pc:
            ax.add_patch(plt.Circle((pc[0], pc[1]), 0.062, color="#060606", zorder=3))
            ax.add_patch(plt.Circle((pc[0], pc[1]), 0.050, color=POCKET_CLR, zorder=3))

    def render_frame(self, shot, t_real, trail_cue, trail_eight):
        cue_pos   = interp_xy(t_real, shot["cue_t"],   shot["cue_xy"])
        eight_pos = interp_xy(t_real, shot["eight_t"], shot["eight_xy"])

        # Is 8-ball still on table? Hide it once it's near a pocket
        eight_visible = True
        if shot["pocketed"]:
            min_pocket_dist = np.min(np.linalg.norm(
                shot["pocket_centers"] - eight_pos, axis=1))
            if min_pocket_dist < 0.08:
                eight_visible = False

        # Update balls
        self.cue_patch.center   = cue_pos
        self.eight_patch.center = eight_pos
        self.eight_lbl.set_position(eight_pos)
        self.eight_patch.set_visible(eight_visible)
        self.eight_lbl.set_visible(eight_visible)

        # Trails (only up to current time)
        if len(trail_cue) > 1:
            self.cue_line.set_data(trail_cue[:, 0], trail_cue[:, 1])
        else:
            self.cue_line.set_data([], [])
        if len(trail_eight) > 1 and eight_visible:
            self.eight_line.set_data(trail_eight[:, 0], trail_eight[:, 1])
        else:
            self.eight_line.set_data([], [])

        # Aim arrow (only at t=0)
        arrow_len = 0.14
        ax_end = (shot["cue_init"][0] + np.cos(shot["phi_shot"]) * arrow_len,
                  shot["cue_init"][1] + np.sin(shot["phi_shot"]) * arrow_len)
        self.aim_arrow.xy     = ax_end
        self.aim_arrow.xytext = tuple(shot["cue_init"])
        self.aim_arrow.set_visible(t_real < 0.05)

        # Texts
        pct    = shot.get("running_pct", "")
        result = ("✓  POCKETED" if shot["pocketed"] else "✗  MISSED")
        self.title_txt.set_text(result)
        self.title_txt.set_color(HIT_CLR if shot["pocketed"] else MISS_CLR)

        info = (f"Shot {shot['shot_num']}  |  "
                f"δ={np.degrees(shot['delta_angle']):.0f}°  "
                f"v={shot['speed']:.1f} m/s  |  {pct}")
        self.info_txt.set_text(info)

        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, :3]   # drop alpha → RGB

    def close(self):
        plt.close(self.fig)


# ── Video writer ───────────────────────────────────────────────────────────────
def write_video(shots_data, out_path, slow=3.0, pause_sec=1.0):
    if not shots_data:
        return

    W  = shots_data[0]["W"]
    L  = shots_data[0]["L"]
    pc = shots_data[0]["pocket_centers"]

    renderer = TableRenderer(W, L, pc, dpi=130)

    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264",
                                 output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    n_total  = len(shots_data)
    n_hit    = 0
    pause_f  = int(pause_sec * FPS)

    for idx, shot in enumerate(shots_data):
        if shot["pocketed"]:
            n_hit += 1
        shot["shot_num"]     = idx + 1
        shot["running_pct"]  = f"{n_hit}/{idx+1} pocketed ({n_hit/(idx+1)*100:.0f}%)"

        # Build time grid for this shot (real physics seconds)
        clip_end   = shot["clip_end"]
        dt_real    = 1.0 / (FPS * slow)          # real-time step per frame
        t_frames   = np.arange(0, clip_end + dt_real, dt_real)

        trail_c = np.empty((0, 2))
        trail_e = np.empty((0, 2))

        for frame_idx, t_real in enumerate(t_frames):
            # Grow trails
            cue_pos   = interp_xy(t_real, shot["cue_t"],   shot["cue_xy"])
            eight_pos = interp_xy(t_real, shot["eight_t"], shot["eight_xy"])
            trail_c   = np.vstack([trail_c, cue_pos])
            trail_e   = np.vstack([trail_e, eight_pos])

            frame = renderer.render_frame(shot, t_real, trail_c, trail_e)
            writer.append_data(frame)

        # Pause on result
        last_frame = renderer.render_frame(shot, clip_end, trail_c, trail_e)
        for _ in range(pause_f):
            writer.append_data(last_frame)

        sys.stdout.write(
            f"\r  Shot {idx+1}/{n_total}  ({n_hit} pocketed)  frames={len(t_frames)}"
        )
        sys.stdout.flush()

    writer.close()
    renderer.close()
    print(f"\n\nSaved → {out_path}  ({n_hit}/{n_total} pocketed, "
          f"{n_hit/n_total*100:.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="logs/best_model/best_model")
    parser.add_argument("--shots",  type=int,   default=10)
    parser.add_argument("--slow",   type=float, default=3.0,
                        help="Slow-motion factor (default 3 = 3× slower)")
    parser.add_argument("--pause",  type=float, default=1.0,
                        help="Seconds to pause on result after each shot")
    parser.add_argument("--out",    default="outputs/agent_shots.mp4")
    args = parser.parse_args()

    print(f"Loading model: {args.model} ...")
    model = SAC.load(args.model)
    env   = BilliardsEnv()

    print(f"Simulating {args.shots} shots ...")
    shots_data = []
    for i in range(args.shots):
        shot = run_shot(env, model)
        shots_data.append(shot)
        icon = "✓" if shot["pocketed"] else "✗"
        sys.stdout.write(f"\r  {icon} Shot {i+1}/{args.shots}")
        sys.stdout.flush()

    n_hit = sum(s["pocketed"] for s in shots_data)
    print(f"\nSimulated {args.shots} shots: {n_hit} pocketed ({n_hit/args.shots*100:.0f}%)")
    print(f"Rendering video at {FPS}fps, {args.slow}× slow motion ...")

    import os; os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_video(shots_data, args.out, slow=args.slow, pause_sec=args.pause)


if __name__ == "__main__":
    main()
