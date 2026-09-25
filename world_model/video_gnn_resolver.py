"""
world_model/video_gnn_resolver.py

GNN collision resolver validation 영상.
GT (pooltool physics) vs GNN predicted trajectory 비교.

τ는 pooltool 해석적 계산 그대로, 충돌 resolution만 GNN으로 교체.

Usage:
    python world_model/video_gnn_resolver.py \
        --ckpt world_model/results/gnn_resolver/best.pt \
        --n-videos 10 \
        --out-dir world_model/results/gnn_resolver_videos
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import imageio.v2 as iio

import pooltool as pt
import pooltool.physics.evolve as evolve
import pooltool.constants as const

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from world_model.gnn_resolver import GNNResolver
from world_model.generate_collision_data import (
    extract_collisions, normalize,
    _contact_normal_linear, _contact_normal_circular,
    _contact_normal_pocket, _contact_normal_ball_ball,
    TABLE_W, TABLE_H, MAX_SPEED, MAX_AVEL,
    COLL_TYPES,
)

FPS  = 20
DT   = 1.0 / FPS

CUE_GT   = "#00e5ff"
CUE_PRED = "#0099dd"
TGT_GT   = "#ffee44"
TGT_PRED = "#cc9900"
BG       = "#111811"
FELT     = "#1e4d2b"
RAIL     = "#3a2a0a"

TYPE_COLORS = {0:"#4488ff", 1:"#aaffaa", 2:"#ff6600", 3:"#ff2222", 4:"#ffffff"}
TYPE_NAMES  = {0:"ball_ball", 1:"linear", 2:"circular", 3:"pocket", 4:"stick_ball"}
POCKET_XY   = np.array([[0,0],[1,0],[0.5,0],[0,1],[0.5,1],[1,1]], dtype=np.float32)
POCKET_R    = 0.025
BALL_PARAMS = dict(R=0.028575, m=0.17, u_s=0.2, u_sp=0.044, u_r=0.016, g=9.81)


def draw_table(ax):
    ax.set_facecolor(FELT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(RAIL); spine.set_linewidth(5)
    for px, py in POCKET_XY:
        ax.add_patch(patches.Circle((px, py), POCKET_R, color="black", zorder=5))


def evolve_trajectory(rvw0: np.ndarray, s0: int, duration: float, fps: int = FPS):
    """ball state를 duration 동안 fps로 샘플링한 position 배열 반환."""
    n  = max(2, int(round(duration * fps)) + 1)
    ts = np.linspace(0, duration, n)
    xs, ys = [], []
    R = BALL_PARAMS["R"]
    for t in ts:
        rvw, _ = evolve.evolve_ball_motion(
            state=s0, rvw=rvw0.copy(),
            R=R, m=BALL_PARAMS["m"],
            u_s=BALL_PARAMS["u_s"], u_sp=BALL_PARAMS["u_sp"],
            u_r=BALL_PARAMS["u_r"], g=BALL_PARAMS["g"], t=t,
        )
        xs.append(rvw[0, 0] / TABLE_W)
        ys.append(rvw[0, 1] / TABLE_H)
    return np.column_stack([xs, ys])


def run_gt_trajectory(system):
    """pooltool system에서 cue/tgt 연속 궤적 추출 (normalized [0,1])."""
    cue_path, tgt_path = [], []

    events = [e for e in system.events if str(e.event_type) != "none"]

    for i, ev in enumerate(events):
        t_start = ev.time
        t_end   = events[i+1].time if i+1 < len(events) else t_start + 0.5

        duration = float(t_end - t_start)
        if duration <= 0:
            continue

        # 이벤트 직후 state
        for agent in ev.agents:
            if getattr(agent, "agent_type", "") != "ball":
                continue
            fin = agent.final
            rvw = np.array(fin.state.rvw if hasattr(fin, "state") else
                           [[fin.xyz[0], fin.xyz[1], 0],
                            [fin.vel[0], fin.vel[1], 0],
                            [fin.avel[0], fin.avel[1], fin.avel[2]]])
            s   = int(fin.state.s) if hasattr(fin, "state") else const.sliding

            traj = evolve_trajectory(rvw, s, duration)
            if agent.id == "cue":
                cue_path.append(traj)
            elif agent.id == "1":
                tgt_path.append(traj)

    cue = np.concatenate(cue_path, axis=0) if cue_path else np.zeros((2, 2))
    tgt = np.concatenate(tgt_path, axis=0) if tgt_path else np.zeros((2, 2))
    # 길이 맞추기
    n = min(len(cue), len(tgt))
    return cue[:n], tgt[:n]


def run_gnn_trajectory(system, model, device):
    """GNN resolver로 충돌 해결, pooltool kinematic으로 사이 적분."""
    cue_path, tgt_path = [], []

    events_raw = [e for e in system.events if str(e.event_type) != "none"]

    # 초기 state (stick_ball 이전: cue는 정지 상태)
    cue_rvw = None
    cue_s   = const.stationary
    tgt_rvw = None
    tgt_s   = const.stationary

    # stick_ball에서 초기 cue state 읽기
    for ev in events_raw:
        if str(ev.event_type) == "stick_ball":
            for agent in ev.agents:
                if getattr(agent, "agent_type", "") == "ball" and agent.id == "cue":
                    ini = agent.initial
                    tgt_for_ini = next(
                        (a for a in ev.agents
                         if getattr(a, "agent_type", "") == "ball" and a.id == "1"),
                        None
                    )
                    cue_rvw = np.array([[ini.xyz[0], ini.xyz[1], 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]])
                    cue_s = const.stationary
            # tgt 초기 위치
            for ev2 in events_raw:
                if str(ev2.event_type) == "ball_ball":
                    for ag2 in ev2.agents:
                        if getattr(ag2, "agent_type", "") == "ball" and ag2.id == "1":
                            ini2 = ag2.initial
                            tgt_rvw = np.array([[ini2.xyz[0], ini2.xyz[1], 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]])
                            tgt_s = const.stationary
                    break
            break

    if cue_rvw is None:
        return np.zeros((2, 2)), np.zeros((2, 2))

    model.eval()

    for i, ev in enumerate(events_raw):
        et = str(ev.event_type)
        if et not in COLL_TYPES:
            continue
        ct = COLL_TYPES[et]

        t_start = ev.time
        t_end   = events_raw[i+1].time if i+1 < len(events_raw) else t_start + 0.5
        duration = max(0.0, float(t_end - t_start))

        # pre-collision: pooltool GT에서 읽기
        cue_pre_vel  = np.zeros(2)
        cue_pre_avel = np.zeros(3)
        tgt_pre_vel  = np.zeros(2)
        tgt_pre_avel = np.zeros(3)
        cue_pos = np.zeros(2)
        tgt_pos = np.zeros(2)
        has_tgt = False
        normal  = np.array([1.0, 0.0])

        for agent in ev.agents:
            atype = getattr(agent, "agent_type", "")
            if atype == "ball":
                ini = agent.initial
                pos = np.array([ini.xyz[0], ini.xyz[1]])
                vel = np.array(ini.vel[:2])
                avl = np.array(ini.avel[:3])
                if agent.id == "cue":
                    cue_pre_vel, cue_pre_avel, cue_pos = vel, avl, pos
                elif agent.id == "1":
                    tgt_pre_vel, tgt_pre_avel, tgt_pos = vel, avl, pos
                    has_tgt = True
            elif atype == "linear_cushion_segment":
                normal = _contact_normal_linear(agent)
            elif atype == "circular_cushion_segment":
                normal = _contact_normal_circular(agent, cue_pos)
            elif atype == "pocket":
                normal = _contact_normal_pocket(agent, cue_pos)
            elif atype == "cue":
                import math
                phi_rad = float(agent.initial.phi) * math.pi / 180.0
                normal  = np.array([math.cos(phi_rad), math.sin(phi_rad)])

        if et == "ball_ball" and has_tgt:
            normal = _contact_normal_ball_ball(cue_pos, tgt_pos)

        # GNN 예측
        with torch.no_grad():
            pv  = torch.tensor([[cue_pre_vel  / MAX_SPEED,
                                  tgt_pre_vel  / MAX_SPEED]], dtype=torch.float32).to(device)
            pa  = torch.tensor([[cue_pre_avel / MAX_AVEL,
                                  tgt_pre_avel / MAX_AVEL]], dtype=torch.float32).to(device)
            pos_t = torch.tensor([[[cue_pos[0]/TABLE_W, cue_pos[1]/TABLE_H],
                                    [tgt_pos[0]/TABLE_W, tgt_pos[1]/TABLE_H]]],
                                  dtype=torch.float32).to(device)
            nrm = torch.tensor([normal], dtype=torch.float32).to(device)
            htg = torch.tensor([has_tgt], dtype=torch.bool).to(device)

            dv, da = model(pos_t, pv, pa, nrm, htg)
            dv = dv[0].cpu().numpy()
            da = da[0].cpu().numpy()

        cue_post_vel  = cue_pre_vel  + dv[0] * MAX_SPEED
        cue_post_avel = cue_pre_avel + da[0] * MAX_AVEL
        tgt_post_vel  = tgt_pre_vel  + dv[1] * MAX_SPEED  if has_tgt else tgt_pre_vel
        tgt_post_avel = tgt_pre_avel + da[1] * MAX_AVEL    if has_tgt else tgt_pre_avel

        # post-collision state 업데이트
        if et != "ball_pocket":
            cue_rvw = np.array([[cue_pos[0], cue_pos[1], 0.0],
                                  [cue_post_vel[0], cue_post_vel[1], 0.0],
                                  [cue_post_avel[0], cue_post_avel[1], cue_post_avel[2]]])
            cue_s = const.sliding if np.linalg.norm(cue_post_vel) > 1e-4 else const.stationary

        if has_tgt and et != "ball_pocket":
            tgt_rvw = np.array([[tgt_pos[0], tgt_pos[1], 0.0],
                                  [tgt_post_vel[0], tgt_post_vel[1], 0.0],
                                  [tgt_post_avel[0], tgt_post_avel[1], tgt_post_avel[2]]])
            tgt_s = const.sliding if np.linalg.norm(tgt_post_vel) > 1e-4 else const.stationary

        if duration > 0 and cue_rvw is not None:
            cue_path.append(evolve_trajectory(cue_rvw, cue_s, duration))
        if duration > 0 and tgt_rvw is not None:
            tgt_path.append(evolve_trajectory(tgt_rvw, tgt_s, duration))

    cue = np.concatenate(cue_path, axis=0) if cue_path else np.zeros((2, 2))
    tgt = np.concatenate(tgt_path, axis=0) if tgt_path else np.zeros((2, 2))
    n = min(len(cue), len(tgt))
    return cue[:n], tgt[:n]


def make_video(ep_idx, cue_gt, tgt_gt, cue_pr, tgt_pr, pocketed, out_path: Path):
    T = min(len(cue_gt), len(cue_pr)) - 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)
    result = "POCKET ✓" if pocketed else "miss"
    fig.suptitle(f"Episode #{ep_idx}  |  {result}  |  GT (left) vs GNN Resolver (right)",
                 color="white", fontsize=10, fontweight="bold")

    ax_gt, ax_pr = axes
    draw_table(ax_gt); draw_table(ax_pr)
    ax_gt.set_title("Ground Truth",        color="white", fontsize=9, pad=3)
    ax_pr.set_title("GNN Resolver Pred",   color="white", fontsize=9, pad=3)

    # 시작 마커
    for ax, cue, tgt in [(ax_gt, cue_gt, tgt_gt), (ax_pr, cue_pr, tgt_pr)]:
        ax.scatter([cue[0,0]], [cue[0,1]], color=CUE_GT if ax==ax_gt else CUE_PRED,
                   s=60, zorder=6, edgecolors="white", lw=0.5)
        ax.scatter([tgt[0,0]], [tgt[0,1]], color=TGT_GT if ax==ax_gt else TGT_PRED,
                   s=60, zorder=6, edgecolors="white", lw=0.5)

    lc_gt, = ax_gt.plot([], [], color=CUE_GT,   lw=1.5, alpha=0.6)
    lt_gt, = ax_gt.plot([], [], color=TGT_GT,   lw=1.5, alpha=0.6)
    bc_gt  = ax_gt.scatter([], [], color=CUE_GT,   s=90, zorder=7, edgecolors="white", lw=0.5)
    bt_gt  = ax_gt.scatter([], [], color=TGT_GT,   s=90, zorder=7, edgecolors="white", lw=0.5)

    lc_pr, = ax_pr.plot([], [], color=CUE_PRED, lw=1.5, alpha=0.6)
    lt_pr, = ax_pr.plot([], [], color=TGT_PRED, lw=1.5, alpha=0.6)
    bc_pr  = ax_pr.scatter([], [], color=CUE_PRED, s=90, zorder=7, edgecolors="white", lw=0.5)
    bt_pr  = ax_pr.scatter([], [], color=TGT_PRED, s=90, zorder=7, edgecolors="white", lw=0.5)

    txt_t   = ax_gt.text(0.02, 0.97, "", transform=ax_gt.transAxes,
                         color="white", fontsize=9, va="top")
    txt_err = ax_pr.text(0.98, 0.97, "", transform=ax_pr.transAxes,
                         color="#ffcc44", fontsize=9, va="top", ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    frames = []
    for t in range(T + 1):
        lc_gt.set_data(cue_gt[:t+1, 0], cue_gt[:t+1, 1])
        lt_gt.set_data(tgt_gt[:t+1, 0], tgt_gt[:t+1, 1])
        bc_gt.set_offsets([[cue_gt[t, 0], cue_gt[t, 1]]])
        bt_gt.set_offsets([[tgt_gt[t, 0], tgt_gt[t, 1]]])

        lc_pr.set_data(cue_pr[:t+1, 0], cue_pr[:t+1, 1])
        lt_pr.set_data(tgt_pr[:t+1, 0], tgt_pr[:t+1, 1])
        bc_pr.set_offsets([[cue_pr[t, 0], cue_pr[t, 1]]])
        bt_pr.set_offsets([[tgt_pr[t, 0], tgt_pr[t, 1]]])

        txt_t.set_text(f"t={t/FPS:.2f}s")

        ec = np.sqrt(((cue_pr[t,0]-cue_gt[t,0])*TABLE_W)**2 +
                     ((cue_pr[t,1]-cue_gt[t,1])*TABLE_H)**2) * 100
        et_ = np.sqrt(((tgt_pr[t,0]-tgt_gt[t,0])*TABLE_W)**2 +
                      ((tgt_pr[t,1]-tgt_gt[t,1])*TABLE_H)**2) * 100
        txt_err.set_text(f"err cue={ec:.1f}cm  tgt={et_:.1f}cm")

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())

    plt.close(fig)

    writer = iio.get_writer(str(out_path), format="FFMPEG", fps=FPS,
                            codec="libx264", quality=7,
                            output_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved → {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default="world_model/results/gnn_resolver/best.pt")
    p.add_argument("--n-videos",  type=int, default=10)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out-dir",   default="world_model/results/gnn_resolver_videos")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GNNResolver().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded  val_loss={ckpt['val_loss']:.5f}  type_rmse={ckpt['type_rmse']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BilliardsEnv(n_balls=1)
    rng = np.random.default_rng(args.seed)

    # 포켓 성공 절반 / 실패 절반
    pocketed_eps, missed_eps = [], []
    for seed in range(10000):
        if len(pocketed_eps) >= args.n_videos // 2 and \
           len(missed_eps)   >= args.n_videos - args.n_videos // 2:
            break
        obs, _ = env.reset(seed=seed)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        pocketed = bool(info.get("pocketed", False))
        if pocketed and len(pocketed_eps) < args.n_videos // 2:
            pocketed_eps.append((seed, action, env.system.copy()))
        elif not pocketed and len(missed_eps) < args.n_videos - args.n_videos // 2:
            missed_eps.append((seed, action, env.system.copy()))

    episodes = pocketed_eps + missed_eps
    print(f"\n{len(pocketed_eps)} pocketed + {len(missed_eps)} missed = {len(episodes)} videos\n")

    for ep_i, (seed, action, system) in enumerate(episodes):
        pocketed = any(str(e.event_type) == "ball_pocket" for e in system.events)
        print(f"[{ep_i+1}/{len(episodes)}] seed={seed}  pocketed={pocketed}")

        cue_gt, tgt_gt = run_gt_trajectory(system)
        cue_pr, tgt_pr = run_gnn_trajectory(system, model, device)

        tag  = "pocket" if pocketed else "miss"
        fname = out_dir / f"{ep_i+1:02d}_{tag}_seed{seed}.mp4"
        make_video(ep_i, cue_gt, tgt_gt, cue_pr, tgt_pr, pocketed, fname)

    env.close()
    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
