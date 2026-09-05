"""
world_model/visualize_markov.py — MarkovPredictor GT vs Pred 시각화

각 샘플을 [GT | Pred] 2-패널로 비교.
v3 포맷의 velocity 정보를 활용해 각 이벤트에 방향 화살표 추가.

Usage:
    python world_model/visualize_markov.py \\
        --ckpt-dir world_model/results/markov_20260905_092337 \\
        --data-dir world_model/data_v3 \\
        [--tags random_v3 sac_v3] \\
        [--n-samples 12] \\
        [--out visualize_markov_out.png]
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.markov_predictor import (
    MarkovEncoder, MarkovTransition,
    S_CUE_XY, S_CUE_VEL, S_CUE_AVEL,
    S_TGT_XY, S_TGT_VEL, S_TGT_AVEL, S_TYPE_OH,
)
from world_model.wm_predictor import (
    EVENT_TYPES, N_EVENT_TYPES, BALL_POCKET_IDX,
    MAX_EVENTS, TABLE_W, TABLE_H,
)
from world_model.generate_data_v3 import MAX_SPEED_V3, MAX_AVEL

# ── 상수 ──────────────────────────────────────────────────────────────────────

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

CUE_COLOR = "#00e5ff"
TGT_COLOR = "#ffee44"
BG_COLOR  = "#1a1a1a"


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, device):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)

    encoder = MarkovEncoder(tuple(cfg["enc_hidden"])).to(device)
    ckpt_e  = torch.load(os.path.join(ckpt_dir, "encoder_best.pt"),
                         map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt_e["state"])
    encoder.eval()

    trans  = MarkovTransition(tuple(cfg["trans_hidden"]), cfg["embed_dim"]).to(device)
    ckpt_t = torch.load(os.path.join(ckpt_dir, "transition_best.pt"),
                        map_location=device, weights_only=False)
    trans.load_state_dict(ckpt_t["state"])
    trans.eval()

    n_enc   = sum(p.numel() for p in encoder.parameters())
    n_trans = sum(p.numel() for p in trans.parameters())
    print(f"Loaded : {os.path.basename(ckpt_dir)}")
    print(f"  encoder val_loss={ckpt_e['val_loss']:.4f}  params={n_enc:,}")
    print(f"  trans   val_loss={ckpt_t['val_loss']:.4f}  params={n_trans:,}")
    return encoder, trans, cfg


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data(data_dir: str, tags):
    meta_path = os.path.join(data_dir, "metadata.json")
    tag_set   = set(tags) if tags else None
    tag_map   = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for entry in json.load(f):
                if entry.get("format") == "v3":
                    tag_map[entry["file"]] = entry["tag"]

    arrays: dict = {}
    loaded = 0
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        if tag_set and tag_map.get(fname) not in tag_set:
            continue
        d = np.load(os.path.join(data_dir, fname))
        for k, v in d.items():
            arrays.setdefault(k, []).append(v)
        loaded += 1

    if loaded == 0:
        raise ValueError(f"No v3 data found in {data_dir} (tags={tags})")

    data = {k: np.concatenate(v) for k, v in arrays.items()}
    print(f"Data: {len(data['obs']):,} episodes  pocketed={100*data['pocketed'].mean():.1f}%")
    return data


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    n = obs.copy()
    n[:, 0::2] /= TABLE_W
    n[:, 1::2] /= TABLE_H
    return n


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_sequence(encoder, trans, obs_norm_t, act_t, max_steps=MAX_EVENTS):
    """
    단일 샘플 inference.
    obs_norm_t, act_t: (1, D) tensors
    Returns:
        types   : (T,) int array
        cue_xys : (T, 2) float — normalized
        tgt_xys : (T, 2) float — normalized
        cue_vels: (T, 2) float — normalized
        tgt_vels: (T, 2) float — normalized
    """
    state = encoder.predict_event(obs_norm_t, act_t)  # (1, 24)

    types, cue_xys, tgt_xys, cue_vels, tgt_vels = [], [], [], [], []
    for _ in range(max_steps):
        types.append(int(state[0, S_TYPE_OH].argmax().item()))
        cue_xys.append(state[0, S_CUE_XY].cpu().numpy())
        tgt_xys.append(state[0, S_TGT_XY].cpu().numpy())
        cue_vels.append(state[0, S_CUE_VEL].cpu().numpy())
        tgt_vels.append(state[0, S_TGT_VEL].cpu().numpy())

        if types[-1] == BALL_POCKET_IDX:
            break
        state = trans.step(state)

    return (np.array(types),
            np.array(cue_xys), np.array(tgt_xys),
            np.array(cue_vels), np.array(tgt_vels))


# ── 테이블 그리기 ─────────────────────────────────────────────────────────────

def draw_table(ax):
    ax.add_patch(patches.Rectangle(
        (0, 0), TABLE_W, TABLE_H,
        facecolor="#2d7a2d", edgecolor="#1a4a1a", linewidth=1.5,
    ))
    for px, py in [(0, 0), (TABLE_W, 0),
                   (0, TABLE_H/2), (TABLE_W, TABLE_H/2),
                   (0, TABLE_H),   (TABLE_W, TABLE_H)]:
        ax.add_patch(plt.Circle((px, py), 0.038, color="black", zorder=5))
    ax.set_xlim(-0.05, TABLE_W + 0.05)
    ax.set_ylim(-0.05, TABLE_H + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)


# ── 패널 그리기 ───────────────────────────────────────────────────────────────

def draw_panel(ax, obs, events_v3, cue_masks, tgt_masks, length,
               cue_color, tgt_color, label, title,
               cue_vels=None, tgt_vels=None, show_arrows=True):
    """
    events_v3: (MAX_EVENTS, 24) — v3 포맷
    cue_vels : (T, 2) normalized vel  (GT면 events_v3에서 추출, Pred면 별도)
    tgt_vels : (T, 2) normalized vel
    """
    draw_table(ax)
    L = int(length)

    # 초기 공 위치
    cue0 = (obs[0] * TABLE_W, obs[1] * TABLE_H)
    tgt0 = (obs[2] * TABLE_W, obs[3] * TABLE_H)
    ax.add_patch(plt.Circle(cue0, 0.028, facecolor="white",
                             edgecolor="#aaaaaa", lw=1.0, zorder=10))
    ax.add_patch(plt.Circle(tgt0, 0.028, facecolor="#ffee44",
                             edgecolor="#888800", lw=1.0, zorder=10))

    if L == 0:
        ax.set_title(f"{label}\n{title}", fontsize=7, color="white", pad=3)
        return

    # v3 슬라이스
    cue_abs_x = events_v3[:L, S_CUE_XY.start]     * TABLE_W
    cue_abs_y = events_v3[:L, S_CUE_XY.start + 1] * TABLE_H
    tgt_abs_x = events_v3[:L, S_TGT_XY.start]     * TABLE_W
    tgt_abs_y = events_v3[:L, S_TGT_XY.start + 1] * TABLE_H
    types     = events_v3[:L, S_TYPE_OH].argmax(axis=-1).astype(int)
    cm        = cue_masks[:L].astype(bool)
    tm        = tgt_masks[:L].astype(bool)

    # velocity (정규화 → m/s 역환산은 시각화용이니 normalized 그대로 사용)
    if cue_vels is None:
        cue_vels = events_v3[:L, S_CUE_VEL.start:S_CUE_VEL.stop]
    if tgt_vels is None:
        tgt_vels = events_v3[:L, S_TGT_VEL.start:S_TGT_VEL.stop]

    # 경로선
    def draw_path(init_x, init_y, xs, ys, mask, color):
        valid = np.where(mask)[0]
        if not len(valid):
            return
        px = [init_x] + [xs[i] for i in valid]
        py = [init_y] + [ys[i] for i in valid]
        ax.plot(px, py, color=color, lw=1.6, alpha=0.85, zorder=6)

    draw_path(cue0[0], cue0[1], cue_abs_x, cue_abs_y, cm, cue_color)
    draw_path(tgt0[0], tgt0[1], tgt_abs_x, tgt_abs_y, tm, tgt_color)

    # 이벤트 마커 + velocity 화살표
    arrow_scale = 0.12   # normalized vel → 테이블 단위 스케일
    for i in range(L):
        c = TYPE_COLORS.get(types[i], "gray")
        if cm[i]:
            ax.scatter(cue_abs_x[i], cue_abs_y[i], s=22, c=c,
                       marker="o", zorder=8, lw=0.4, edgecolors="white", alpha=0.9)
            if show_arrows and cue_vels is not None:
                vx, vy = cue_vels[i] * arrow_scale
                ax.annotate("", xy=(cue_abs_x[i] + vx * TABLE_W,
                                    cue_abs_y[i] + vy * TABLE_H),
                            xytext=(cue_abs_x[i], cue_abs_y[i]),
                            arrowprops=dict(arrowstyle="->", color=cue_color,
                                            lw=0.8, alpha=0.6), zorder=7)
        if tm[i]:
            ax.scatter(tgt_abs_x[i], tgt_abs_y[i], s=22, c=c,
                       marker="s", zorder=8, lw=0.4, edgecolors="white", alpha=0.9)
            if show_arrows and tgt_vels is not None:
                vx, vy = tgt_vels[i] * arrow_scale
                ax.annotate("", xy=(tgt_abs_x[i] + vx * TABLE_W,
                                    tgt_abs_y[i] + vy * TABLE_H),
                            xytext=(tgt_abs_x[i], tgt_abs_y[i]),
                            arrowprops=dict(arrowstyle="->", color=tgt_color,
                                            lw=0.8, alpha=0.6), zorder=7)

    ax.set_title(f"{label}\n{title}", fontsize=6.5, color="white", pad=3)


# ── 그리드 이미지 ─────────────────────────────────────────────────────────────

def visualize_grid(encoder, trans, data, indices, out_path, device,
                   show_arrows=True):
    n         = len(indices)
    n_per_row = 2
    n_rows    = (n + n_per_row - 1) // n_per_row
    n_cols    = n_per_row * 2

    obs_norm = normalize_obs(data["obs"])

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.6, n_rows * 5.2),
        facecolor=BG_COLOR,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for k, idx in enumerate(indices):
        row      = k // n_per_row
        col_base = (k %  n_per_row) * 2
        L        = int(data["lengths"][idx])

        poc  = "✓" if data["pocketed"][idx] else "✗"
        info = (f"#{idx}  {poc}  L={L}  nb={data['n_bounces'][idx]}\n"
                f"φ={data['actions'][idx,0]:.2f}  v={data['actions'][idx,1]:.1f}")

        # GT panel
        draw_panel(
            axes[row, col_base],
            obs       = data["obs"][idx],
            events_v3 = data["events"][idx],
            cue_masks = data["cue_masks"][idx],
            tgt_masks = data["tgt_masks"][idx],
            length    = L,
            cue_color = CUE_COLOR,
            tgt_color = TGT_COLOR,
            label     = "GT",
            title     = info,
            show_arrows = show_arrows,
        )

        # Pred inference
        obs_t = torch.from_numpy(obs_norm[idx:idx+1]).float().to(device)
        act_t = torch.from_numpy(data["actions"][idx:idx+1]).float().to(device)
        p_types, p_cue, p_tgt, p_cvels, p_tvels = predict_sequence(encoder, trans, obs_t, act_t)

        T_pred = len(p_types)
        pred_events = np.zeros((MAX_EVENTS, 24), dtype=np.float32)
        pred_events[:T_pred, S_CUE_XY]  = p_cue
        pred_events[:T_pred, S_TGT_XY]  = p_tgt
        pred_events[:T_pred, S_CUE_VEL] = p_cvels
        pred_events[:T_pred, S_TGT_VEL] = p_tvels
        for t, ty in enumerate(p_types):
            pred_events[t, S_TYPE_OH.start + ty] = 1.0

        # 마스크: ball_ball이면 tgt도 유효, 아니면 cue만
        p_cm = np.zeros(MAX_EVENTS, dtype=np.int8)
        p_tm = np.zeros(MAX_EVENTS, dtype=np.int8)
        for t, ty in enumerate(p_types):
            p_cm[t] = 1
            if ty == 2:  # ball_ball
                p_tm[t] = 1

        draw_panel(
            axes[row, col_base + 1],
            obs       = data["obs"][idx],
            events_v3 = pred_events,
            cue_masks = p_cm,
            tgt_masks = p_tm,
            length    = T_pred,
            cue_color = "#ff9900",
            tgt_color = "#ff44ff",
            label     = "Pred",
            title     = info,
            cue_vels  = p_cvels,
            tgt_vels  = p_tvels,
            show_arrows = show_arrows,
        )

    # 범례
    legend_items = [
        Line2D([0],[0], color=CUE_COLOR,   lw=2, label="GT cue"),
        Line2D([0],[0], color=TGT_COLOR,   lw=2, label="GT tgt"),
        Line2D([0],[0], color="#ff9900",   lw=2, label="Pred cue"),
        Line2D([0],[0], color="#ff44ff",   lw=2, label="Pred tgt"),
    ] + [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
               markersize=6, label=EVENT_TYPES[i])
        for i, c in TYPE_COLORS.items() if i > 0
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=7,
               fontsize=6, facecolor="#333333", labelcolor="white",
               framealpha=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved → {out_path}")


# ── 정량 평가 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, trans, data, n_eval, device, seed=0):
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(data["obs"]), min(n_eval, len(data["obs"])), replace=False)
    obs_norm = normalize_obs(data["obs"])

    type_correct, pos_errs_cue, pos_errs_tgt = [], [], []

    for idx in idxs:
        L = int(data["lengths"][idx])
        if L == 0:
            continue
        obs_t = torch.from_numpy(obs_norm[idx:idx+1]).float().to(device)
        act_t = torch.from_numpy(data["actions"][idx:idx+1]).float().to(device)
        p_types, p_cue, p_tgt, _, _ = predict_sequence(encoder, trans, obs_t, act_t)

        gt_types = data["events"][idx, :L, 14:].argmax(axis=-1)
        gt_cue   = data["events"][idx, :L, 0:2]
        gt_tgt   = data["events"][idx, :L, 7:9]
        cm       = data["cue_masks"][idx, :L].astype(bool)
        tm       = data["tgt_masks"][idx, :L].astype(bool)

        T = min(len(p_types), L)
        for t in range(T):
            type_correct.append(int(p_types[t] == gt_types[t]))
            if cm[t]:
                pos_errs_cue.append(np.linalg.norm(p_cue[t] - gt_cue[t]))
            if tm[t]:
                pos_errs_tgt.append(np.linalg.norm(p_tgt[t] - gt_tgt[t]))

    print(f"\n── 정량 평가 (n={len(idxs)}) ──────────────────────────────────")
    print(f"Event type accuracy : {100*np.mean(type_correct):.1f}%")
    print(f"Cue pos MSE (norm)  : {np.mean(pos_errs_cue):.4f}"
          f"  ({np.mean(pos_errs_cue)*TABLE_W*100:.1f} cm avg)")
    print(f"Tgt pos MSE (norm)  : {np.mean(pos_errs_tgt):.4f}"
          f"  ({np.mean(pos_errs_tgt)*TABLE_H*100:.1f} cm avg)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir",    type=str, required=True)
    p.add_argument("--data-dir",    type=str, default="world_model/data_v3")
    p.add_argument("--tags",        nargs="+", default=None)
    p.add_argument("--n-samples",   type=int, default=12)
    p.add_argument("--n-eval",      type=int, default=500)
    p.add_argument("--out",         type=str, default="visualize_markov_out.png")
    p.add_argument("--no-arrows",   action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--pocketed-only", action="store_true")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, trans, cfg = load_model(args.ckpt_dir, device)
    data = load_data(args.data_dir, args.tags)

    rng = np.random.default_rng(args.seed)
    if args.pocketed_only:
        pool = np.where(data["pocketed"] & (data["lengths"] > 0))[0]
    else:
        pool = np.where(data["lengths"] > 0)[0]

    indices = rng.choice(pool, min(args.n_samples, len(pool)), replace=False)
    indices = sorted(indices)

    evaluate(encoder, trans, data, args.n_eval, device)
    visualize_grid(encoder, trans, data, indices, args.out, device,
                   show_arrows=not args.no_arrows)


if __name__ == "__main__":
    main()
