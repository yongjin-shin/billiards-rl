"""
world_model/visualize_wmv2.py — WMPredictor v2 샷 inference 시각화

GT(Ground Truth) vs Pred(AR Inference) 궤적을 비교.
- 이미지: N 샘플 그리드 (cue path + tgt path 각각 표시)
- 영상  : 스텝별 애니메이션 (GT | Pred 좌우 분할)

Usage:
    python world_model/visualize_wmv2.py \\
        --ckpt world_model/checkpoints/wmv2_enc128_128_h256_l1_emb32_s0_aug_20260329_164446 \\
        [--data world_model/data_v2] \\
        [--tags sac_abs_test] \\
        [--n-samples 16] \\
        [--n-video 4] \\
        [--seed 42] \\
        [--device cpu]
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
from matplotlib.lines import Line2D
import imageio.v2 as imageio
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.wm_predictor import (
    WMPredictor, EVENT_TYPES, N_EVENT_TYPES, BALL_POCKET_IDX,
    MAX_EVENTS, TABLE_W, TABLE_H,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

# 이벤트 타입별 색상 (type index → hex)
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

# GT / Pred 색상 테마
GT_CUE_COLOR   = "#00e5ff"   # 청록  — GT cue path
GT_TGT_COLOR   = "#aaff44"   # 연두  — GT tgt path
PR_CUE_COLOR   = "#ff4466"   # 분홍  — Pred cue path
PR_TGT_COLOR   = "#ff9900"   # 주황  — Pred tgt path


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, device):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    pt_path  = os.path.join(ckpt_dir, "best.pt")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = WMPredictor(
        enc_hidden      = cfg["enc_hidden"],
        lstm_hidden     = cfg["lstm_hidden"],
        lstm_layers     = cfg["lstm_layers"],
        event_embed_dim = cfg["event_embed_dim"],
    ).to(device)

    ckpt = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state"])
    model.eval()

    print(f"Loaded  : {os.path.basename(ckpt_dir)}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    print(f"  n_params={sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data(data_dir: str, tags):
    meta_path = os.path.join(data_dir, "metadata.json")
    tag_set   = set(tags) if tags else None
    tag_map   = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for entry in json.load(f):
                tag_map[entry["file"]] = entry["tag"]

    obs_l, act_l, ev_l, cm_l, tm_l, len_l, poc_l, nb_l = [], [], [], [], [], [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        if tag_set and tag_map.get(fname) not in tag_set:
            continue
        d = np.load(os.path.join(data_dir, fname))
        obs_l.append(d["obs"])
        act_l.append(d["actions"])
        ev_l.append(d["events"])
        cm_l.append(d["cue_masks"])
        tm_l.append(d["tgt_masks"])
        len_l.append(d["lengths"])
        poc_l.append(d["pocketed"])
        nb_l.append(d["n_bounces"])

    if not obs_l:
        raise ValueError(f"No data found in {data_dir} (tags={tags})")

    obs      = np.concatenate(obs_l)
    actions  = np.concatenate(act_l)
    events   = np.concatenate(ev_l)
    cue_masks= np.concatenate(cm_l)
    tgt_masks= np.concatenate(tm_l)
    lengths  = np.concatenate(len_l)
    pocketed = np.concatenate(poc_l)
    n_bounces= np.concatenate(nb_l)

    print(f"  Data: {len(obs):,} episodes  (tags={tags})")
    print(f"  Pocketed: {pocketed.sum()} / {len(pocketed)}"
          f"  ({100*pocketed.mean():.1f}%)")
    return obs, actions, events, cue_masks, tgt_masks, lengths, pocketed, n_bounces


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """obs (raw) → [0,1] 정규화. train_wm_predictor.py 와 동일 로직."""
    obs_n = obs.copy()
    obs_n[:, 0::2] /= TABLE_W
    obs_n[:, 1::2] /= TABLE_H
    return obs_n


# ── 테이블 그리기 ─────────────────────────────────────────────────────────────

def draw_table(ax, alpha=1.0):
    rect = patches.Rectangle(
        (0, 0), TABLE_W, TABLE_H,
        facecolor="#2d7a2d", edgecolor="#1a4a1a", linewidth=2, alpha=alpha,
    )
    ax.add_patch(rect)

    pocket_r  = 0.04
    pocket_xy = [
        (0,        0          ),  # BL
        (TABLE_W,  0          ),  # BR
        (0,        TABLE_H / 2),  # ML
        (TABLE_W,  TABLE_H / 2),  # MR
        (0,        TABLE_H    ),  # TL
        (TABLE_W,  TABLE_H    ),  # TR
    ]
    for px, py in pocket_xy:
        ax.add_patch(plt.Circle((px, py), pocket_r,
                                color="black", zorder=5, alpha=alpha))

    ax.set_xlim(-0.06, TABLE_W + 0.06)
    ax.set_ylim(-0.06, TABLE_H + 0.06)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_obs_balls(ax, obs_norm, alpha=0.9):
    """
    obs_norm[0:2] = cue (white),  obs_norm[2:4] = tgt (yellow)
    obs_norm 은 [0,1] → 실제 좌표로 변환 후 그림.
    """
    ball_info = [
        (obs_norm[0] * TABLE_W, obs_norm[1] * TABLE_H, "white"),
        (obs_norm[2] * TABLE_W, obs_norm[3] * TABLE_H, "#ffee44"),
    ]
    for bx, by, color in ball_info:
        if 0 <= bx <= TABLE_W and 0 <= by <= TABLE_H:
            ax.add_patch(plt.Circle((bx, by), 0.025,
                                    facecolor=color, edgecolor="black",
                                    linewidth=1.2, zorder=10, alpha=alpha))


# ── 궤적 그리기 (v2) ──────────────────────────────────────────────────────────

def _draw_masked_line(ax, xs, ys, mask, color, lw, alpha, ls="-"):
    """mask=1 인 점들만 선으로 연결 (불연속 구간은 분리)."""
    seg_x, seg_y = [], []
    for x, y, m in zip(xs, ys, mask):
        if m:
            seg_x.append(x)
            seg_y.append(y)
        else:
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=color, lw=lw,
                        alpha=alpha, zorder=6, linestyle=ls)
            seg_x, seg_y = [], []
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=color, lw=lw,
                alpha=alpha, zorder=6, linestyle=ls)


def draw_trajectory_v2(ax, events_enc, cue_masks, tgt_masks, types_idx, n,
                       cue_color, tgt_color, lw=1.4, alpha=0.85, ls="-"):
    """
    events_enc : (MAX_EVENTS, 14)  — 정규화된 좌표
    cue_masks  : (MAX_EVENTS,)
    tgt_masks  : (MAX_EVENTS,)
    types_idx  : (MAX_EVENTS,)   int  event type index
    n          : 유효 길이
    """
    if n == 0:
        return

    cue_xs = events_enc[:n, 0] * TABLE_W
    cue_ys = events_enc[:n, 1] * TABLE_H
    tgt_xs = events_enc[:n, 2] * TABLE_W
    tgt_ys = events_enc[:n, 3] * TABLE_H
    cm     = cue_masks[:n]
    tm     = tgt_masks[:n]
    typs   = types_idx[:n]

    # ── 선 ────────────────────────────────────────────────────────────────────
    _draw_masked_line(ax, cue_xs, cue_ys, cm, cue_color, lw, alpha, ls)
    _draw_masked_line(ax, tgt_xs, tgt_ys, tm, tgt_color, lw, alpha, ls)

    # ── 이벤트 점 ──────────────────────────────────────────────────────────────
    for i in range(n):
        c = TYPE_COLORS.get(int(typs[i]), "gray")
        if cm[i]:
            ax.scatter(cue_xs[i], cue_ys[i], s=22, color=c,
                       marker="o", zorder=8, linewidths=0.4,
                       edgecolors="white", alpha=alpha)
        if tm[i]:
            ax.scatter(tgt_xs[i], tgt_ys[i], s=22, color=c,
                       marker="s", zorder=8, linewidths=0.4,
                       edgecolors="white", alpha=alpha)

    # ── 시작점 마커 ────────────────────────────────────────────────────────────
    # cue 시작
    if cm[0]:
        ax.scatter(cue_xs[0], cue_ys[0], s=50, marker="^",
                   color=cue_color, zorder=9, edgecolors="black", linewidths=0.8)
    # tgt 시작
    if tm[0]:
        ax.scatter(tgt_xs[0], tgt_ys[0], s=50, marker="^",
                   color=tgt_color, zorder=9, edgecolors="black", linewidths=0.8)


# ── 배치 inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_predict(model, obs_arr, act_arr, events_arr, lengths_arr, device):
    """
    model.forward(tf_ratio=0.0) 으로 완전 AR inference.
    Returns:
        pred_types : (N, MAX_EVENTS)  int
        pred_cue   : (N, MAX_EVENTS, 2)  normalized
        pred_tgt   : (N, MAX_EVENTS, 2)  normalized
    """
    obs_n   = torch.from_numpy(normalize_obs(obs_arr)).float().to(device)
    act_t   = torch.from_numpy(act_arr).float().to(device)
    ev_t    = torch.from_numpy(events_arr).float().to(device)
    len_t   = torch.from_numpy(lengths_arr).long().to(device)

    event_logits, pos_pred = model(obs_n, act_t, ev_t, len_t, tf_ratio=0.0)
    pred_types = event_logits.argmax(dim=-1).cpu().numpy()   # (N, T)
    pred_cue   = pos_pred[:, :, 0:2].cpu().numpy()           # (N, T, 2)
    pred_tgt   = pos_pred[:, :, 2:4].cpu().numpy()           # (N, T, 2)
    return pred_types, pred_cue, pred_tgt


def make_pred_events(pred_types, pred_cue, pred_tgt):
    """pred 결과를 v2 events 포맷 (N, T, 14) 으로 합치기 (시각화 편의)."""
    N, T = pred_types.shape
    one_hot = np.zeros((N, T, N_EVENT_TYPES), dtype=np.float32)
    one_hot[np.arange(N)[:, None], np.arange(T)[None, :], pred_types] = 1.0
    return np.concatenate([pred_cue, pred_tgt, one_hot], axis=-1)   # (N, T, 14)


# ── 그리드 이미지 ─────────────────────────────────────────────────────────────

def visualize_grid(
    model, obs_arr, act_arr, events_arr, cue_masks_arr, tgt_masks_arr,
    lengths_arr, pocketed_arr, n_bounces_arr, indices, out_dir, device,
    prefix="grid"
):
    n     = len(indices)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.4, nrows * 5.8),
                             facecolor="#1a1a1a")
    axes = np.array(axes).reshape(nrows, ncols)

    # ── 배치 inference ────────────────────────────────────────────────────────
    sub_obs = obs_arr[indices]
    sub_act = act_arr[indices]
    sub_ev  = events_arr[indices]
    sub_len = lengths_arr[indices]

    pred_types, pred_cue, pred_tgt = batch_predict(
        model, sub_obs, sub_act, sub_ev, sub_len, device
    )
    pred_events = make_pred_events(pred_types, pred_cue, pred_tgt)

    for k, idx in enumerate(indices):
        r, c  = divmod(k, ncols)
        ax    = axes[r, c]
        ax.set_facecolor("#1a1a1a")

        draw_table(ax)

        draw_obs_balls(ax, obs_arr[idx])

        gt_len = int(lengths_arr[idx])
        gt_types = events_arr[idx, :, 4:].argmax(axis=-1)

        # GT 궤적
        draw_trajectory_v2(ax,
                           events_arr[idx], cue_masks_arr[idx], tgt_masks_arr[idx],
                           gt_types, gt_len,
                           cue_color=GT_CUE_COLOR, tgt_color=GT_TGT_COLOR,
                           lw=1.8, alpha=0.9, ls="-")

        # Pred 궤적 (GT 길이 기준으로 표시 — 비교 편의)
        draw_trajectory_v2(ax,
                           pred_events[k],
                           np.ones(MAX_EVENTS, dtype=np.int8),   # pred는 mask 없음
                           np.ones(MAX_EVENTS, dtype=np.int8),
                           pred_types[k], gt_len,
                           cue_color=PR_CUE_COLOR, tgt_color=PR_TGT_COLOR,
                           lw=1.4, alpha=0.80, ls="--")

        poc  = "✓" if pocketed_arr[idx] else "✗"
        title = (f"#{idx}  {poc}  L={gt_len}  nb={n_bounces_arr[idx]}\n"
                 f"φ={act_arr[idx,0]:.2f}  v={act_arr[idx,1]:.1f}")
        ax.set_title(title, fontsize=7, pad=3, color="white")

    # 빈 칸 숨기기
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor("#1a1a1a")

    # 범례
    legend_elems = [
        Line2D([0], [0], color=GT_CUE_COLOR, lw=2,   label="GT  cue"),
        Line2D([0], [0], color=GT_TGT_COLOR, lw=2,   label="GT  tgt"),
        Line2D([0], [0], color=PR_CUE_COLOR, lw=2, ls="--", label="Pred cue (AR)"),
        Line2D([0], [0], color=PR_TGT_COLOR, lw=2, ls="--", label="Pred tgt (AR)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.7, facecolor="#333333", labelcolor="white")
    fig.suptitle("WMPredictor v2 — AR Inference  (GT vs Pred)",
                 fontsize=11, color="white", y=1.002)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{prefix}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"  Image → {fname}")


# ── 영상 시각화 ───────────────────────────────────────────────────────────────

def _fig_to_rgb(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="#1a1a1a")
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    if h % 2: img = np.pad(img, ((0,1),(0,0),(0,0)), mode="edge")
    if w % 2: img = np.pad(img, ((0,0),(0,1),(0,0)), mode="edge")
    return img


def visualize_video(
    model, obs_arr, act_arr, events_arr, cue_masks_arr, tgt_masks_arr,
    lengths_arr, pocketed_arr, n_bounces_arr, indices, out_dir, device, fps=5
):
    sub_obs = obs_arr[indices]
    sub_act = act_arr[indices]
    sub_ev  = events_arr[indices]
    sub_len = lengths_arr[indices]

    pred_types, pred_cue, pred_tgt = batch_predict(
        model, sub_obs, sub_act, sub_ev, sub_len, device
    )
    pred_events = make_pred_events(pred_types, pred_cue, pred_tgt)

    for k, idx in enumerate(indices):
        gt_len  = int(lengths_arr[idx])
        max_len = max(gt_len, 3)

        frames = []
        for step in range(1, max_len + 1):
            fig, axes = plt.subplots(1, 2, figsize=(8.4, 6.0),
                                     facecolor="#1a1a1a")
            gt_types   = events_arr[idx, :, 4:].argmax(axis=-1)
            pred_type_ = pred_types[k]

            for ax, ev, cm, tm, typs, cue_c, tgt_c, lbl in [
                (axes[0],
                 events_arr[idx], cue_masks_arr[idx], tgt_masks_arr[idx],
                 gt_types, GT_CUE_COLOR, GT_TGT_COLOR, "Ground Truth"),
                (axes[1],
                 pred_events[k],
                 np.ones(MAX_EVENTS, dtype=np.int8),
                 np.ones(MAX_EVENTS, dtype=np.int8),
                 pred_type_, PR_CUE_COLOR, PR_TGT_COLOR, "Prediction (AR)"),
            ]:
                ax.set_facecolor("#1a1a1a")
                draw_table(ax)
                draw_obs_balls(ax, obs_arr[idx])
                draw_trajectory_v2(ax, ev, cm, tm, typs,
                                   min(step, gt_len if lbl.startswith("G") else max_len),
                                   cue_c, tgt_c, lw=1.8, alpha=0.9)
                ax.set_title(lbl, fontsize=9, color="white", pad=4)

            poc_str = "Pocketed ✓" if pocketed_arr[idx] else "Miss ✗"
            fig.suptitle(
                f"#{idx}  {poc_str}  nb={n_bounces_arr[idx]}"
                f"  |  φ={act_arr[idx,0]:.2f}  v={act_arr[idx,1]:.1f}"
                f"  |  step {step}/{max_len}",
                fontsize=9, color="white",
            )
            plt.tight_layout()
            frames.append(_fig_to_rgb(fig))
            plt.close(fig)

        frames += [frames[-1]] * fps  # 마지막 프레임 1초 정지

        fname = os.path.join(out_dir, f"video_{idx:04d}.mp4")
        imageio.mimwrite(fname, frames, fps=fps, macro_block_size=1)
        print(f"  Video → {fname}  ({len(frames)} frames)")


# ── 이벤트 타입 정확도 분석 ────────────────────────────────────────────────────

def analyze_accuracy(model, obs_arr, act_arr, events_arr, cue_masks_arr,
                     tgt_masks_arr, lengths_arr, device):
    """event type accuracy + pos MSE 를 콘솔에 출력."""
    pred_types, pred_cue, pred_tgt = batch_predict(
        model, obs_arr, act_arr, events_arr, lengths_arr, device
    )

    N = len(obs_arr)
    correct_total = 0
    valid_total   = 0
    cue_mse_sum = tgt_mse_sum = 0.0
    cue_n = tgt_n = 0

    for i in range(N):
        L = int(lengths_arr[i])
        gt_types = events_arr[i, :L, 4:].argmax(axis=-1)
        correct_total += (pred_types[i, :L] == gt_types).sum()
        valid_total   += L

        for t in range(L):
            if cue_masks_arr[i, t]:
                dx = pred_cue[i, t, 0] - events_arr[i, t, 0]
                dy = pred_cue[i, t, 1] - events_arr[i, t, 1]
                cue_mse_sum += dx*dx + dy*dy
                cue_n += 1
            if tgt_masks_arr[i, t]:
                dx = pred_tgt[i, t, 0] - events_arr[i, t, 2]
                dy = pred_tgt[i, t, 1] - events_arr[i, t, 3]
                tgt_mse_sum += dx*dx + dy*dy
                tgt_n += 1

    type_acc   = correct_total / max(valid_total, 1)
    cue_mse    = cue_mse_sum / max(cue_n, 1) / 2
    tgt_mse    = tgt_mse_sum / max(tgt_n, 1) / 2

    print(f"\n  ── Accuracy on {N} samples ──")
    print(f"  Event type accuracy : {type_acc:.4f}  ({correct_total}/{valid_total})")
    print(f"  Cue pos MSE         : {cue_mse:.4f}  ({cue_n} valid steps)")
    print(f"  Tgt pos MSE         : {tgt_mse:.4f}  ({tgt_n} valid steps)")
    print(f"  Pos MSE (avg)       : {(cue_mse+tgt_mse)/2:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize WMPredictor v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str,
                   default="world_model/checkpoints/"
                           "wmv2_enc128_128_h256_l1_emb32_s0_aug_20260329_164446")
    p.add_argument("--data",      type=str,   default="world_model/data_v2")
    p.add_argument("--tags",      type=str,   nargs="+",
                   default=["sac_abs_test"],
                   help="None 이면 전체 데이터 사용")
    p.add_argument("--n-samples", type=int,   default=16,
                   help="그리드 이미지 샘플 수")
    p.add_argument("--n-video",   type=int,   default=4,
                   help="영상으로 저장할 샘플 수")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    type=str,   default="cpu")
    p.add_argument("--out-dir",   type=str,   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_name = os.path.basename(args.ckpt.rstrip("/"))
    out_dir   = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", ckpt_name
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput  → {out_dir}")

    # ── 모델 ──
    model, cfg = load_model(args.ckpt, device)

    # ── 데이터 ──
    (obs, actions, events,
     cue_masks, tgt_masks, lengths,
     pocketed, n_bounces) = load_data(args.data, args.tags)

    # ── 정확도 분석 ──
    analyze_accuracy(model, obs, actions, events, cue_masks,
                     tgt_masks, lengths, device)

    # ── 샘플 선택: 포켓 성공 / 실패 절반씩 ──
    pos_idx = np.where( pocketed)[0]
    neg_idx = np.where(~pocketed)[0]
    rng     = np.random.default_rng(args.seed)

    def sample_balanced(n):
        n_pos = n // 2
        n_neg = n - n_pos
        chosen = np.concatenate([
            rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
        ])
        rng.shuffle(chosen)
        return chosen.tolist()

    grid_idx  = sample_balanced(args.n_samples)
    video_idx = sample_balanced(args.n_video)

    kw = dict(
        obs_arr=obs, act_arr=actions,
        events_arr=events, cue_masks_arr=cue_masks, tgt_masks_arr=tgt_masks,
        lengths_arr=lengths, pocketed_arr=pocketed, n_bounces_arr=n_bounces,
        device=device,
    )

    # ── 그리드 ──
    print("\n[1] Mixed grid ...")
    visualize_grid(model, indices=grid_idx, out_dir=out_dir,
                   prefix="grid_mixed", **kw)

    pos_grid = rng.choice(pos_idx, size=min(8, len(pos_idx)), replace=False).tolist()
    neg_grid = rng.choice(neg_idx, size=min(8, len(neg_idx)), replace=False).tolist()

    print("[2] Pocketed grid ...")
    visualize_grid(model, indices=pos_grid, out_dir=out_dir,
                   prefix="grid_pocketed", **kw)

    print("[3] Miss grid ...")
    visualize_grid(model, indices=neg_grid, out_dir=out_dir,
                   prefix="grid_miss", **kw)

    # ── 영상 ──
    print("\n[4] Videos ...")
    visualize_video(model, indices=video_idx, out_dir=out_dir, fps=5, **kw)

    print(f"\nDone  →  {out_dir}/")


if __name__ == "__main__":
    main()
