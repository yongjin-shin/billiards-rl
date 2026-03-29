"""
world_model/visualize_wmv2.py — WMPredictor v2 샷 inference 시각화

각 샘플을 [GT | Pred] 좌우 패널로 분리해서 비교.

Usage:
    python world_model/visualize_wmv2.py \\
        --ckpt world_model/checkpoints/wmv2_enc128_128_h256_l1_emb32_s0_aug_20260329_164446 \\
        [--data world_model/data_v2] \\
        [--tags sac_abs_test] \\
        [--n-samples 8] \\
        [--n-video 4]
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
TYPE_NAMES = EVENT_TYPES

CUE_COLOR = "#00e5ff"    # cue ball path
TGT_COLOR = "#ffee44"    # target ball path
BG_COLOR  = "#1a1a1a"


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, device):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)

    model = WMPredictor(
        enc_hidden      = cfg["enc_hidden"],
        lstm_hidden     = cfg["lstm_hidden"],
        lstm_layers     = cfg["lstm_layers"],
        event_embed_dim = cfg["event_embed_dim"],
    ).to(device)

    ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded  : {os.path.basename(ckpt_dir)}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}"
          f"  n_params={sum(p.numel() for p in model.parameters()):,}")
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

    data = dict(
        obs       = np.concatenate(obs_l),
        actions   = np.concatenate(act_l),
        events    = np.concatenate(ev_l),
        cue_masks = np.concatenate(cm_l),
        tgt_masks = np.concatenate(tm_l),
        lengths   = np.concatenate(len_l),
        pocketed  = np.concatenate(poc_l),
        n_bounces = np.concatenate(nb_l),
    )
    print(f"  Data: {len(data['obs']):,} episodes  (tags={tags})")
    print(f"  Pocketed: {data['pocketed'].sum()}/{len(data['pocketed'])} "
          f"({100*data['pocketed'].mean():.1f}%)")
    return data


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """WMDataset 과 동일 로직: obs → /[TABLE_W, TABLE_H, ...]"""
    n = obs.copy()
    n[:, 0::2] /= TABLE_W
    n[:, 1::2] /= TABLE_H
    return n


# ── 배치 inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_predict(model, data, indices, device):
    """tf_ratio=0.0 (완전 AR) 로 배치 inference."""
    obs_n = torch.from_numpy(normalize_obs(data["obs"][indices])).float().to(device)
    act   = torch.from_numpy(data["actions"][indices]).float().to(device)
    ev    = torch.from_numpy(data["events"][indices]).float().to(device)
    lens  = torch.from_numpy(data["lengths"][indices]).long().to(device)

    event_logits, pos_pred = model(obs_n, act, ev, lens, tf_ratio=0.0)

    pred_types = event_logits.argmax(dim=-1).cpu().numpy()   # (N, T)
    pred_cue   = pos_pred[:, :, 0:2].cpu().numpy()           # (N, T, 2) normalized
    pred_tgt   = pos_pred[:, :, 2:4].cpu().numpy()           # (N, T, 2) normalized
    return pred_types, pred_cue, pred_tgt


# ── 테이블 그리기 ─────────────────────────────────────────────────────────────

def draw_table(ax):
    ax.add_patch(patches.Rectangle(
        (0, 0), TABLE_W, TABLE_H,
        facecolor="#2d7a2d", edgecolor="#1a4a1a", linewidth=1.5,
    ))
    pocket_r  = 0.038
    for px, py in [(0, 0), (TABLE_W, 0),
                   (0, TABLE_H/2), (TABLE_W, TABLE_H/2),
                   (0, TABLE_H),   (TABLE_W, TABLE_H)]:
        ax.add_patch(plt.Circle((px, py), pocket_r, color="black", zorder=5))
    ax.set_xlim(-0.05, TABLE_W + 0.05)
    ax.set_ylim(-0.05, TABLE_H + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)


# ── 궤적 그리기 ───────────────────────────────────────────────────────────────

def _to_abs(xy_norm, scale_x, scale_y):
    """(N, 2) normalized → actual coords"""
    return xy_norm[:, 0] * scale_x, xy_norm[:, 1] * scale_y


def draw_panel(ax, obs, events, cue_masks, tgt_masks, length,
               cue_color, tgt_color, label, title):
    """
    단일 패널 그리기: 테이블 + 공 초기 위치 + shot path + 궤적.

    obs        : (16,)  raw [0,1]  (env normalized)
    events     : (MAX_EVENTS, 14)  normalized
    cue_masks  : (MAX_EVENTS,)
    tgt_masks  : (MAX_EVENTS,)
    length     : int
    """
    draw_table(ax)
    L = int(length)

    # ── 초기 공 위치 (obs) ────────────────────────────────────────────────────
    cue0_x, cue0_y = obs[0] * TABLE_W, obs[1] * TABLE_H
    tgt0_x, tgt0_y = obs[2] * TABLE_W, obs[3] * TABLE_H

    ax.add_patch(plt.Circle((cue0_x, cue0_y), 0.028,
                             facecolor="white", edgecolor="#aaaaaa",
                             linewidth=1.0, zorder=10))
    ax.add_patch(plt.Circle((tgt0_x, tgt0_y), 0.028,
                             facecolor="#ffee44", edgecolor="#888800",
                             linewidth=1.0, zorder=10))

    if L == 0:
        ax.set_title(f"{label}\n{title}", fontsize=7, color="white", pad=3)
        return

    # ── events → actual coords ────────────────────────────────────────────────
    cue_abs_x = events[:L, 0] * TABLE_W
    cue_abs_y = events[:L, 1] * TABLE_H
    tgt_abs_x = events[:L, 2] * TABLE_W
    tgt_abs_y = events[:L, 3] * TABLE_H
    types     = events[:L, 4:].argmax(axis=-1).astype(int)
    cm        = cue_masks[:L].astype(bool)
    tm        = tgt_masks[:L].astype(bool)

    # ── shot path: obs → 첫 번째 유효 이벤트 위치 ────────────────────────────
    # cue shot path (obs_cue → first valid cue event)
    first_cm = np.where(cm)[0]
    if len(first_cm) > 0:
        fi = first_cm[0]
        ax.plot([cue0_x, cue_abs_x[fi]], [cue0_y, cue_abs_y[fi]],
                color=cue_color, lw=0.8, alpha=0.45, ls="--", zorder=5)

    # tgt shot path: only if first valid tgt event != obs_tgt (safety)
    first_tm = np.where(tm)[0]
    if len(first_tm) > 0:
        fi = first_tm[0]
        ax.plot([tgt0_x, tgt_abs_x[fi]], [tgt0_y, tgt_abs_y[fi]],
                color=tgt_color, lw=0.8, alpha=0.45, ls="--", zorder=5)

    # ── cue 궤적 선 ───────────────────────────────────────────────────────────
    seg_x, seg_y = [], []
    for i in range(L):
        if cm[i]:
            seg_x.append(cue_abs_x[i])
            seg_y.append(cue_abs_y[i])
        else:
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=cue_color, lw=1.5, alpha=0.9, zorder=6)
            seg_x, seg_y = [], []
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=cue_color, lw=1.5, alpha=0.9, zorder=6)

    # ── tgt 궤적 선 ───────────────────────────────────────────────────────────
    seg_x, seg_y = [], []
    for i in range(L):
        if tm[i]:
            seg_x.append(tgt_abs_x[i])
            seg_y.append(tgt_abs_y[i])
        else:
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=tgt_color, lw=1.5, alpha=0.9, zorder=6)
            seg_x, seg_y = [], []
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=tgt_color, lw=1.5, alpha=0.9, zorder=6)

    # ── 이벤트 점 ──────────────────────────────────────────────────────────────
    for i in range(L):
        c = TYPE_COLORS.get(types[i], "gray")
        if cm[i]:
            ax.scatter(cue_abs_x[i], cue_abs_y[i], s=20, c=c,
                       marker="o", zorder=8, linewidths=0.3, edgecolors="white", alpha=0.9)
        if tm[i]:
            ax.scatter(tgt_abs_x[i], tgt_abs_y[i], s=20, c=c,
                       marker="s", zorder=8, linewidths=0.3, edgecolors="white", alpha=0.9)

    ax.set_title(f"{label}\n{title}", fontsize=6.5, color="white", pad=3)


# ── 그리드 이미지 ─────────────────────────────────────────────────────────────

def visualize_grid(model, data, indices, out_dir, device, prefix="grid"):
    """
    각 샘플을 [GT | Pred] 2-패널로.
    n 샘플 → n행 × 2열 subplots.
    """
    n = len(indices)

    # GT 데이터 수집
    pred_types, pred_cue, pred_tgt = batch_predict(model, data, indices, device)

    # 레이아웃: 한 행에 2개 샘플씩 (GT|Pred GT|Pred), 즉 4열
    n_per_row = 2   # 한 행에 몇 개 샘플
    n_rows    = (n + n_per_row - 1) // n_per_row
    n_cols    = n_per_row * 2   # 샘플당 GT + Pred 2열

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.6, n_rows * 5.2),
        facecolor=BG_COLOR,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for k, idx in enumerate(indices):
        row      = k // n_per_row
        col_base = (k %  n_per_row) * 2
        L        = int(data["lengths"][idx])

        poc  = "✓" if data["pocketed"][idx] else "✗"
        info = (f"#{idx}  {poc}  L={L}  nb={data['n_bounces'][idx]}\n"
                f"φ={data['actions'][idx,0]:.2f}  v={data['actions'][idx,1]:.1f}")

        # ── GT panel ──────────────────────────────────────────────────────────
        ax_gt = axes[row, col_base]
        draw_panel(
            ax_gt,
            obs       = data["obs"][idx],
            events    = data["events"][idx],
            cue_masks = data["cue_masks"][idx],
            tgt_masks = data["tgt_masks"][idx],
            length    = L,
            cue_color = CUE_COLOR,
            tgt_color = TGT_COLOR,
            label     = "GT",
            title     = info,
        )

        # ── Pred panel ────────────────────────────────────────────────────────
        # pred_events 를 v2 format 으로 재구성
        one_hot = np.zeros((MAX_EVENTS, N_EVENT_TYPES), dtype=np.float32)
        one_hot[np.arange(MAX_EVENTS), pred_types[k]] = 1.0
        pred_events = np.concatenate([pred_cue[k], pred_tgt[k], one_hot], axis=-1)

        ax_pr = axes[row, col_base + 1]
        draw_panel(
            ax_pr,
            obs       = data["obs"][idx],
            events    = pred_events,
            # Pred는 GT mask 그대로 적용 — 같은 이벤트 슬롯에서 위치만 비교
            cue_masks = data["cue_masks"][idx],
            tgt_masks = data["tgt_masks"][idx],
            length    = L,
            cue_color = "#ff4466",
            tgt_color = "#ff9900",
            label     = "Pred (AR)",
            title     = info,
        )

    # 빈 칸 숨기기
    for k in range(n, n_rows * n_per_row):
        row      = k // n_per_row
        col_base = (k %  n_per_row) * 2
        for dc in range(2):
            ax = axes[row, col_base + dc]
            ax.axis("off")
            ax.set_facecolor(BG_COLOR)

    # 분리선 (GT | Pred 사이에 세로 구분)
    for c in range(1, n_cols, 2):
        for r in range(n_rows):
            ax = axes[r, c]
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

    # 범례
    legend_elems = [
        Line2D([0],[0], color=CUE_COLOR,  lw=2, label="cue path"),
        Line2D([0],[0], color=TGT_COLOR,  lw=2, label="tgt path"),
        Line2D([0],[0], marker="o", color="#ff6600", ms=5, ls="none",
               label="ball_ball"),
        Line2D([0],[0], marker="o", color="#ffdd00", ms=5, ls="none",
               label="cushion"),
        Line2D([0],[0], marker="o", color="#ff2222", ms=5, ls="none",
               label="ball_pocket"),
        Line2D([0],[0], marker="o", color="#44ff44", ms=5, ls="none",
               label="rolling/*"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=6,
               fontsize=8, framealpha=0.6, facecolor="#333333",
               labelcolor="white", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("WMPredictor v2  |  GT (left)  vs  Pred (right)",
                 fontsize=11, color="white", y=1.003)

    plt.tight_layout(pad=0.5)
    fname = os.path.join(out_dir, f"{prefix}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Image → {fname}")


# ── 영상 시각화 ───────────────────────────────────────────────────────────────

def _fig_to_rgb(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=BG_COLOR)
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    if h % 2: img = np.pad(img, ((0,1),(0,0),(0,0)), mode="edge")
    if w % 2: img = np.pad(img, ((0,0),(0,1),(0,0)), mode="edge")
    return img


def visualize_video(model, data, indices, out_dir, device, fps=5):
    pred_types, pred_cue, pred_tgt = batch_predict(model, data, indices, device)

    for k, idx in enumerate(indices):
        L        = int(data["lengths"][idx])
        max_step = max(L, 2)

        one_hot = np.zeros((MAX_EVENTS, N_EVENT_TYPES), dtype=np.float32)
        one_hot[np.arange(MAX_EVENTS), pred_types[k]] = 1.0
        pred_events = np.concatenate([pred_cue[k], pred_tgt[k], one_hot], axis=-1)

        poc_str = "Pocketed ✓" if data["pocketed"][idx] else "Miss ✗"
        act_str = (f"φ={data['actions'][idx,0]:.2f}  "
                   f"v={data['actions'][idx,1]:.1f}")

        frames = []
        for step in range(1, max_step + 1):
            fig, (ax_gt, ax_pr) = plt.subplots(
                1, 2, figsize=(7.0, 5.5), facecolor=BG_COLOR,
            )

            draw_panel(ax_gt,
                       data["obs"][idx],
                       data["events"][idx],
                       data["cue_masks"][idx], data["tgt_masks"][idx],
                       min(step, L),
                       CUE_COLOR, TGT_COLOR,
                       label=f"GT  (step {step}/{L})",
                       title="")
            draw_panel(ax_pr,
                       data["obs"][idx],
                       pred_events,
                       data["cue_masks"][idx], data["tgt_masks"][idx],
                       min(step, L),
                       "#ff4466", "#ff9900",
                       label=f"Pred (step {step}/{L})",
                       title="")

            fig.suptitle(f"#{idx}  {poc_str}  |  {act_str}  |  nb={data['n_bounces'][idx]}",
                         fontsize=9, color="white")
            plt.tight_layout()
            frames.append(_fig_to_rgb(fig))
            plt.close(fig)

        frames += [frames[-1]] * fps
        fname = os.path.join(out_dir, f"video_{idx:04d}.mp4")
        imageio.mimwrite(fname, frames, fps=fps, macro_block_size=1)
        print(f"  Video → {fname}  ({len(frames)} frames)")


# ── 정확도 분석 ───────────────────────────────────────────────────────────────

def analyze_accuracy(model, data, device, n_eval=None):
    N   = len(data["obs"]) if n_eval is None else min(n_eval, len(data["obs"]))
    idx = np.arange(N)
    pred_types, pred_cue, pred_tgt = batch_predict(model, data, idx, device)

    correct = valid = 0
    cue_mse_sum = tgt_mse_sum = cue_n = tgt_n = 0.0

    for i in range(N):
        L = int(data["lengths"][i])
        gt_t = data["events"][i, :L, 4:].argmax(axis=-1)
        correct += (pred_types[i, :L] == gt_t).sum()
        valid   += L
        for t in range(L):
            if data["cue_masks"][i, t]:
                dx = pred_cue[i,t,0] - data["events"][i,t,0]
                dy = pred_cue[i,t,1] - data["events"][i,t,1]
                cue_mse_sum += dx*dx + dy*dy
                cue_n += 1
            if data["tgt_masks"][i, t]:
                dx = pred_tgt[i,t,0] - data["events"][i,t,2]
                dy = pred_tgt[i,t,1] - data["events"][i,t,3]
                tgt_mse_sum += dx*dx + dy*dy
                tgt_n += 1

    print(f"\n  ── Accuracy ({N} samples) ──────────────────")
    print(f"  Event type acc: {correct/max(valid,1):.4f} "
          f"({correct}/{valid})")
    print(f"  Cue  pos MSE : {cue_mse_sum/max(cue_n,1)/2:.4f}  "
          f"({cue_n} valid)")
    print(f"  Tgt  pos MSE : {tgt_mse_sum/max(tgt_n,1)/2:.4f}  "
          f"({tgt_n} valid)")
    print(f"  Avg  pos MSE : {(cue_mse_sum/max(cue_n,1)+tgt_mse_sum/max(tgt_n,1))/4:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str,
                   default="world_model/checkpoints/"
                           "wmv2_enc128_128_h256_l1_emb32_s0_aug_20260329_164446")
    p.add_argument("--data",      type=str,   default="world_model/data_v2")
    p.add_argument("--tags",      type=str,   nargs="+", default=["sac_abs_test"])
    p.add_argument("--n-samples", type=int,   default=8)
    p.add_argument("--n-video",   type=int,   default=4)
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
        os.path.dirname(__file__), "results", ckpt_name,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput → {out_dir}")

    model, cfg = load_model(args.ckpt, device)
    data       = load_data(args.data, args.tags)

    # 정확도
    analyze_accuracy(model, data, device)

    # 샘플 선택: pocketed / miss 절반씩
    pos_idx = np.where( data["pocketed"])[0]
    neg_idx = np.where(~data["pocketed"])[0]
    rng     = np.random.default_rng(args.seed)

    def balanced(n):
        np_  = n // 2
        nn_  = n - np_
        chosen = np.concatenate([
            rng.choice(pos_idx, size=min(np_, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(nn_, len(neg_idx)), replace=False),
        ])
        rng.shuffle(chosen)
        return chosen.tolist()

    grid_idx  = balanced(args.n_samples)
    video_idx = balanced(args.n_video)

    print(f"\n[1] Mixed grid ({args.n_samples} samples) ...")
    visualize_grid(model, data, grid_idx, out_dir, device, prefix="grid_mixed")

    pos_grid = rng.choice(pos_idx, size=min(args.n_samples, len(pos_idx)),
                          replace=False).tolist()
    neg_grid = rng.choice(neg_idx, size=min(args.n_samples, len(neg_idx)),
                          replace=False).tolist()

    print("[2] Pocketed grid ...")
    visualize_grid(model, data, pos_grid, out_dir, device, prefix="grid_pocketed")

    print("[3] Miss grid ...")
    visualize_grid(model, data, neg_grid, out_dir, device, prefix="grid_miss")

    print(f"\n[4] Videos ({args.n_video}) ...")
    visualize_video(model, data, video_idx, out_dir, device, fps=5)

    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
