"""
world_model/visualize_predictor.py — LSTMPredictor 샷 inference 시각화

1. 이미지 : N 샘플 그리드  (GT vs 예측 궤적 + obs 공 위치)
2. 영상   : 각 샘플을 스텝별로 애니메이션 (mp4, imageio)

Usage:
    python world_model/visualize_predictor.py \\
        --ckpt world_model/checkpoints/pred_lstm_ctx128_h512_l1_tf50_s2_20260329_061733 \\
        [--data world_model/data_abs] \\
        [--n-samples 16] \\
        [--n-video 4] \\
        [--out-dir world_model/results/pred_lstm_ctx128_h512_l1_tf50_s2_20260329_061733]
"""

import os
import sys
import json
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.predictor import LSTMPredictor, EVENT_DIM, MAX_EVENTS, N_EVENT_TYPES

# ── 테이블 절대 좌표 ──────────────────────────────────────────────────────────
TABLE_W = 0.991   # 너비 (x)
TABLE_L = 1.981   # 길이 (y)

# 이벤트 타입 이름 (predictor.py / generate_data.py 와 일치)
EVENT_NAMES = [
    "none",
    "stick_ball",
    "ball_ball",
    "ball_linear_cushion",
    "ball_circular_cushion",
    "ball_pocket",
    "sliding_rolling",
    "rolling_spinning",
    "rolling_stationary",
    "spinning_stationary",
]

# 이벤트 타입별 색상
TYPE_COLORS = {
    0: "#888888",  # none
    1: "#00bfff",  # stick_ball
    2: "#ff6600",  # ball_ball
    3: "#ffdd00",  # ball_linear_cushion
    4: "#ffaa00",  # ball_circular_cushion
    5: "#ff2222",  # ball_pocket
    6: "#aaffaa",  # sliding_rolling
    7: "#88ff88",  # rolling_spinning
    8: "#44ff44",  # rolling_stationary
    9: "#22cc22",  # spinning_stationary
}


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_predictor(ckpt_dir: str, device):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    pt_path  = os.path.join(ckpt_dir, "best.pt")

    with open(cfg_path) as f:
        cfg = json.load(f)

    ckpt = torch.load(pt_path, map_location=device)

    model = LSTMPredictor(
        obs_dim     = cfg["obs_dim"],
        act_dim     = 2,
        ctx_hidden  = cfg["ctx_hidden"],
        lstm_hidden = cfg["lstm_hidden"],
        lstm_layers = cfg["lstm_layers"],
    ).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded  {os.path.basename(ckpt_dir)}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model, cfg


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data(data_dir: str, tags):
    meta_path = os.path.join(data_dir, "metadata.json")
    tag_set   = set(tags)
    tag_map   = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for entry in json.load(f):
                tag_map[entry["file"]] = entry["tag"]

    obs_list, act_list, events_list, lengths_list, pocketed_list = [], [], [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        if tag_map and tag_map.get(fname) not in tag_set:
            continue
        d = np.load(os.path.join(data_dir, fname))
        obs_list.append(d["obs"])
        act_list.append(d["actions"])
        events_list.append(d["events"])
        lengths_list.append(d["lengths"])
        pocketed_list.append(d["pocketed"])

    obs      = np.concatenate(obs_list)
    actions  = np.concatenate(act_list)
    events   = np.concatenate(events_list)
    lengths  = np.concatenate(lengths_list)
    pocketed = np.concatenate(pocketed_list)
    print(f"  Data: {len(obs):,} episodes  (tags={tags})")
    return obs, actions, events, lengths, pocketed


# ── 테이블 배경 그리기 ────────────────────────────────────────────────────────

def draw_table(ax, alpha=1.0):
    """초록 테이블 + 포켓 그리기"""
    # 테이블 배경
    rect = patches.Rectangle(
        (0, 0), TABLE_W, TABLE_L,
        facecolor="#2d7a2d", edgecolor="#1a4a1a", linewidth=2, alpha=alpha,
    )
    ax.add_patch(rect)

    # 포켓 (6개: 상중하 × 좌우)
    pocket_r = 0.04
    pocket_xy = [
        (0,        0       ),  # BL
        (TABLE_W,  0       ),  # BR
        (0,        TABLE_L / 2),  # ML
        (TABLE_W,  TABLE_L / 2),  # MR
        (0,        TABLE_L ),  # TL
        (TABLE_W,  TABLE_L ),  # TR
    ]
    for px, py in pocket_xy:
        circ = plt.Circle((px, py), pocket_r, color="black", zorder=5, alpha=alpha)
        ax.add_patch(circ)

    ax.set_xlim(-0.05, TABLE_W + 0.05)
    ax.set_ylim(-0.05, TABLE_L + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_obs_balls(ax, obs, obs_dim, alpha=0.9):
    """
    obs 앞 절반: cue(x,y) + ball들 (x,y) — 절대 좌표로 변환.
    obs 는 [0,1] 정규화 기준, 역변환: x *= TABLE_W, y *= TABLE_L
    """
    # obs 구조: cue_x, cue_y, b1_x, b1_y, [b1_pocketed], b2_x, ...  포켓 위치
    # 단순히 앞에 나오는 x,y 쌍들을 파싱 (짝수 index 가 x, 홀수가 y)
    # 1-ball env: obs_dim=16 → cue(2) + b1(2) + b1_flag(0 for n_balls=1) + 12 pocket_obs
    ball_xy = []
    coords = obs[:4]   # cue(x,y) + b1(x,y)
    for i in range(0, len(coords), 2):
        bx = float(coords[i])   * TABLE_W
        by = float(coords[i+1]) * TABLE_L
        ball_xy.append((bx, by))

    colors = ["white", "yellow"]
    for k, (bx, by) in enumerate(ball_xy):
        if 0 <= bx <= TABLE_W and 0 <= by <= TABLE_L:
            circ = plt.Circle(
                (bx, by), 0.025,
                color=colors[k % len(colors)],
                zorder=10, alpha=alpha,
                linewidth=1.5, edgecolor="black",
            )
            ax.add_patch(circ)


def draw_trajectory(ax, events_enc, length, color, lw=1.2, alpha=0.85,
                    marker_size=30, label=None):
    """events_enc: (MAX_EVENTS, EVENT_DIM) — 유효한 length 개만 그린다."""
    n = min(length, MAX_EVENTS)
    if n == 0:
        return
    xs = events_enc[:n, 0]
    ys = events_enc[:n, 1]
    types = events_enc[:n, 2:].argmax(axis=-1)

    # 선
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=6,
            label=label if label else None)

    # 점 (이벤트 타입별 색상)
    for i, (x, y, t) in enumerate(zip(xs, ys, types)):
        c = TYPE_COLORS.get(int(t), "gray")
        ax.scatter(x, y, s=marker_size, color=c, zorder=7,
                   linewidths=0.5, edgecolors="white", alpha=alpha)

    # 첫 이벤트 (start) 표시
    ax.scatter(xs[0], ys[0], s=55, marker="^", color=color,
               zorder=8, edgecolors="black", linewidths=0.8)


# ── 이미지 시각화 ─────────────────────────────────────────────────────────────

def visualize_grid(model, obs_arr, act_arr, events_arr, lengths_arr,
                   pocketed_arr, indices, out_dir, device, prefix="grid"):
    """
    indices 에 해당하는 샘플들을 그리드로 시각화.
    각 칸: 왼쪽=GT, 오른쪽=Pred (같은 칸에 두 궤적을 오버레이).
    """
    n = len(indices)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 5.5))
    axes = np.array(axes).reshape(nrows, ncols)

    obs_t   = torch.from_numpy(obs_arr[indices]).float().to(device)
    act_t   = torch.from_numpy(act_arr[indices]).float().to(device)

    with torch.no_grad():
        pred_t = model(obs_t, act_t)   # 완전 AR inference
    pred_np = pred_t.cpu().numpy()     # (n, MAX_EVENTS, EVENT_DIM)

    for k, idx in enumerate(indices):
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        draw_table(ax)
        draw_obs_balls(ax, obs_arr[idx], obs_arr.shape[1])

        gt_len   = int(lengths_arr[idx])
        pred_len = MAX_EVENTS   # 예측은 항상 MAX_EVENTS 반환 (type으로 end 추정 어려움)

        # GT 궤적
        draw_trajectory(ax, events_arr[idx], gt_len,
                        color="#00e5ff", lw=1.8, alpha=0.9, label="GT")
        # 예측 궤적
        draw_trajectory(ax, pred_np[k], gt_len,  # 같은 길이 기준으로 비교
                        color="#ff4466", lw=1.4, alpha=0.85, label="Pred")

        pocket_str = "✓" if pocketed_arr[idx] else "✗"
        act_str    = f"φ={act_arr[idx,0]:.2f} v={act_arr[idx,1]:.1f}"
        ax.set_title(
            f"#{idx}  {pocket_str}  L={gt_len}\n{act_str}",
            fontsize=7, pad=3,
        )

    # 빈 칸 숨기기
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    # 범례
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#00e5ff", lw=2, label="GT"),
        Line2D([0], [0], color="#ff4466", lw=2, label="Pred (AR)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.8)
    fig.suptitle("LSTMPredictor — Shot Inference  (GT vs AR Prediction)",
                 fontsize=11, y=1.01)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{prefix}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Image saved → {fname}")


# ── 영상 시각화 ───────────────────────────────────────────────────────────────

def _fig_to_rgb(fig):
    """matplotlib figure → RGB numpy array (even dimensions for libx264)"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    img = imageio.imread(buf)   # RGBA or RGB
    buf.close()

    # RGBA → RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # libx264 requires even width & height
    h, w = img.shape[:2]
    if h % 2 != 0:
        img = np.pad(img, ((0, 1), (0, 0), (0, 0)), mode="edge")
    if w % 2 != 0:
        img = np.pad(img, ((0, 0), (0, 1), (0, 0)), mode="edge")

    return img


def visualize_video(model, obs_arr, act_arr, events_arr, lengths_arr,
                    pocketed_arr, indices, out_dir, device, fps=6):
    """
    각 샘플에 대해 GT vs Pred 궤적을 스텝별로 재생하는 mp4 생성.
    """
    obs_t = torch.from_numpy(obs_arr[indices]).float().to(device)
    act_t = torch.from_numpy(act_arr[indices]).float().to(device)
    with torch.no_grad():
        pred_t = model(obs_t, act_t)
    pred_np = pred_t.cpu().numpy()

    for k, idx in enumerate(indices):
        gt_len   = int(lengths_arr[idx])
        max_len  = max(gt_len, 4)

        frames = []
        for step in range(1, max_len + 1):
            fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))

            for ax, traj, label, color in [
                (axes[0], events_arr[idx], "Ground Truth", "#00e5ff"),
                (axes[1], pred_np[k],      "Prediction (AR)", "#ff4466"),
            ]:
                draw_table(ax)
                draw_obs_balls(ax, obs_arr[idx], obs_arr.shape[1])
                n_draw = min(step, MAX_EVENTS if label.startswith("P") else gt_len)
                draw_trajectory(ax, traj, n_draw, color=color, lw=1.8, alpha=0.9)
                ax.set_title(label, fontsize=9)

            pocket_str = "Pocketed ✓" if pocketed_arr[idx] else "Miss ✗"
            act_str    = f"φ={act_arr[idx,0]:.2f}  v={act_arr[idx,1]:.1f}"
            fig.suptitle(
                f"Sample #{idx}  |  {pocket_str}  |  {act_str}  |  step {step}/{max_len}",
                fontsize=9,
            )
            plt.tight_layout()

            frames.append(_fig_to_rgb(fig))
            plt.close(fig)

        # 마지막 프레임 1초 정지
        frames += [frames[-1]] * fps

        fname = os.path.join(out_dir, f"video_sample{idx:04d}.mp4")
        imageio.mimwrite(fname, frames, fps=fps, macro_block_size=1)
        print(f"  Video saved → {fname}  ({len(frames)} frames)")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize LSTMPredictor shot inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str,
                   default="world_model/checkpoints/pred_lstm_ctx128_h512_l1_tf50_s2_20260329_061733",
                   help="Checkpoint directory (contains config.json + best.pt)")
    p.add_argument("--data", type=str, default="world_model/data_abs")
    p.add_argument("--tags", type=str, nargs="+", default=["sac_abs", "random_abs"])
    p.add_argument("--n-samples", type=int, default=16,
                   help="Number of samples for grid image")
    p.add_argument("--n-video", type=int, default=4,
                   help="Number of samples for video")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: results/<ckpt_name>)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 출력 디렉터리 ──
    ckpt_name = os.path.basename(args.ckpt.rstrip("/"))
    out_dir   = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", ckpt_name
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput → {out_dir}")

    # ── 모델 ──
    model, cfg = load_predictor(args.ckpt, device)

    # ── 데이터 ──
    obs, actions, events, lengths, pocketed = load_data(args.data, args.tags)

    # ── 샘플 인덱스 선택 ──
    # 포켓 성공 / 실패 절반씩 섞어 선택
    pos_idx = np.where( pocketed)[0]
    neg_idx = np.where(~pocketed)[0]
    rng     = np.random.default_rng(args.seed)

    def sample_balanced(n):
        n_pos = n // 2
        n_neg = n - n_pos
        chosen_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(idx)
        return idx.tolist()

    grid_idx  = sample_balanced(args.n_samples)
    video_idx = sample_balanced(args.n_video)

    print(f"\n  Grid  : {len(grid_idx)} samples")
    print(f"  Video : {len(video_idx)} samples")

    # ── 이미지 저장 ──
    print("\n[1] Generating grid image ...")
    visualize_grid(model, obs, actions, events, lengths, pocketed,
                   grid_idx, out_dir, device, prefix="shot_grid")

    # 포켓 성공만 모아서 별도 그리드
    pos_grid = rng.choice(pos_idx, size=min(8, len(pos_idx)), replace=False).tolist()
    neg_grid = rng.choice(neg_idx, size=min(8, len(neg_idx)), replace=False).tolist()

    print("[2] Generating pocketed-only grid ...")
    visualize_grid(model, obs, actions, events, lengths, pocketed,
                   pos_grid, out_dir, device, prefix="shot_grid_pocketed")

    print("[3] Generating miss-only grid ...")
    visualize_grid(model, obs, actions, events, lengths, pocketed,
                   neg_grid, out_dir, device, prefix="shot_grid_miss")

    # ── 영상 저장 ──
    print("\n[4] Generating videos ...")
    visualize_video(model, obs, actions, events, lengths, pocketed,
                    video_idx, out_dir, device)

    print(f"\nDone. All results → {out_dir}/")


if __name__ == "__main__":
    main()
