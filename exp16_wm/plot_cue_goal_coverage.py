"""
exp16_wm/plot_cue_goal_coverage.py — how varied are the start positions and the goals of the cue-only shots with a given cushion count?

Three panels on the table (portrait): the start positions (training vs held-out starts of CT2a), for each 2.5 cm cell the number of starts
that sent a shot there (the goals the data can serve), and the goals of one start.

Usage:
    python -m exp16_wm.plot_cue_goal_coverage --data logs/ct0b/collect_200.npz --k-bin 0 --out logs/ct0b/k0_coverage.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pooltool as pt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import TABLE_L, TABLE_W, cell_index, grid_shape, standing_mask
from exp16_wm.cue_goal_head import load_rows

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CELL = 0.025


def _font() -> None:
    path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False


def draw_table(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((0, 0), TABLE_W, TABLE_L, fill=False, ec=INK2, lw=1.2, zorder=5))
    for p in pt.Table.default().pockets.values():
        c = np.asarray(p.center[:2], float)
        ax.add_patch(Circle((np.clip(c[0], 0, TABLE_W), np.clip(c[1], 0, TABLE_L)), 0.05, fc=INK2, ec="none", alpha=0.35, zorder=4))
    ax.set_xlim(-0.04, TABLE_W + 0.04); ax.set_ylim(-0.04, TABLE_L + 0.04); ax.set_aspect("equal"); ax.axis("off")


def coverage_counts(sidx: np.ndarray, end: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(NX, NY): number of different starts with at least one of `rows` ending in the cell."""
    nx, ny = grid_shape(CELL)
    ij = cell_index(end[rows], CELL)
    key = np.unique(sidx[rows].astype(np.int64) * (nx * ny) + ij[:, 0] * ny + ij[:, 1])
    cell = key % (nx * ny)
    out = np.zeros(nx * ny, int); np.add.at(out, cell, 1)
    return out.reshape(nx, ny)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k-bin", type=int, default=0); ap.add_argument("--example", type=int, default=-1)
    args = ap.parse_args()
    _font()
    d = load_rows(args.data, args.k_bin, 40, 1000, 0)
    starts, sidx, end, ok = d["starts"], d["sidx"], d["end"], d["ok"]
    rows = np.nonzero(ok)[0]
    cov = coverage_counts(sidx, end, rows)
    stand = standing_mask(CELL)
    dist = np.linalg.norm(end[rows] - starts[sidx[rows]], axis=1)
    centre = np.array([TABLE_W / 2, TABLE_L / 2])
    ex = d["train_starts"][np.argmin(np.linalg.norm(starts[d["train_starts"]] - centre, axis=1))] if args.example < 0 else args.example   # default: the start nearest the table centre
    ex_rows = rows[sidx[rows] == ex]
    print(f"k={args.k_bin}: {len(rows):,} shots, {len(rows) / len(starts):,.0f} per start")
    print(f"cells that stand: {int(stand.sum())}; covered by >=1 start {np.mean(cov[stand] >= 1):.3f}, >=100 starts {np.mean(cov[stand] >= 100):.3f}, "
          f">=190 starts {np.mean(cov[stand] >= 190):.3f}")
    print(f"start -> goal distance: p10 {np.percentile(dist, 10):.2f} m, median {np.median(dist):.2f} m, p90 {np.percentile(dist, 90):.2f} m")
    print(f"example start {ex} at {starts[ex].round(3)}: {len(ex_rows):,} shots")

    ramp = LinearSegmentedColormap.from_list("blue", ["#eaf2fc", BLUE, "#123f7a"]); ramp.set_bad("#efefec")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.08})
    for ax in axes:
        ax.set_facecolor(SURFACE)

    ax = axes[0]; draw_table(ax)
    hold = np.zeros(len(starts), bool); hold[d["holdout"]] = True
    ax.scatter(*starts[~hold].T, s=26, c=BLUE, ec=SURFACE, lw=0.8, zorder=6, label=f"학습에 쓴 시작 {int((~hold).sum())}곳")
    ax.scatter(*starts[hold].T, s=26, c=ORANGE, ec=SURFACE, lw=0.8, zorder=6, label=f"평가용으로 뺀 시작 {int(hold.sum())}곳")
    ax.set_title("① 수구 시작 위치", color=INK, fontsize=12, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9, labelcolor=INK2)

    ax = axes[1]; draw_table(ax)
    img = np.ma.masked_where(~stand | (cov == 0), cov).T
    im = ax.imshow(img, origin="lower", extent=(0, grid_shape(CELL)[0] * CELL, 0, grid_shape(CELL)[1] * CELL), cmap=ramp, vmin=0, vmax=200, zorder=2)
    ax.set_title("② 목표 위치: 그 칸에 보낸 시작 수", color=INK, fontsize=12, pad=8)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, ticks=[0, 100, 200])
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, labelsize=9, length=0); cb.set_label(f"시작 200곳 중 (2.5 cm 칸)\n서 있을 수 있는 칸 {int(stand.sum()):,}개: 최소 {int(cov[stand].min())}곳, 중앙값 {int(np.median(cov[stand]))}곳이 보냄", color=INK2, fontsize=9)

    ax = axes[2]; draw_table(ax)
    pick = np.random.default_rng(0).choice(ex_rows, size=min(8000, len(ex_rows)), replace=False)
    ax.scatter(*end[pick].T, s=3, c=BLUE, alpha=0.35, lw=0, zorder=3)
    ax.scatter(*starts[ex][None].T, s=90, facecolors=SURFACE, edgecolors=ORANGE, linewidths=2.2, zorder=7)
    ax.set_title("③ 시작 한 곳(주황 원)에서 간 목표", color=INK, fontsize=12, pad=8)
    ax.text(0.5, -0.025, f"이 시작의 샷 {len(ex_rows):,}개 중 8,000개를 점으로", transform=ax.transAxes, ha="center", va="top", color=INK2, fontsize=9)

    kname = {0: "쿠션 0번", 1: "쿠션 1번", 2: "쿠션 2번", 3: "쿠션 3번 이상"}[args.k_bin]
    fig.suptitle(f"{kname}으로 선 샷의 시작 위치와 목표 위치 (시작 {len(starts)}곳, 샷 {len(rows) / 1e4:,.0f}만 개)", color=INK, fontsize=13, y=0.97)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
