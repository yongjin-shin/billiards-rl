"""
exp16_wm/v_heatmap_explain.py — what "① actual" and "② 1 cm average" of the value map mean, drawn for one real cue position.

Takes the cache of exp16_wm.v_heatmap, picks a cue position whose eight 1 cm-moved copies disagree, replays those copies in the
simulator (the same copies and policy as the cache: their mean must equal the cache's ② value there), and draws
  left   the position, the 1 cm circle, the eight copies at their real places, each with the balls it pocketed in a row,
  right  the three maps along the table row through that position: ① jumps from cell to cell, ② is what the average of the copies gives.

Usage:
    python -m exp16_wm.v_heatmap_explain --cache logs/teacher_data/v6/heatmap_left5.npz --runs "<PRm run dir glob>" \
        --out logs/teacher_data/v6/heatmap_explain.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pick_story_cell(copies: dict[int, np.ndarray], lo: float = 1.5, hi: float = 3.5) -> int:
    """Among the cells in `copies` (cell → the balls pocketed by each placed copy), the one whose copies disagree most (largest standard
    deviation) while their mean is between lo and hi and at least four copies were placed; raises if none qualifies."""
    best, best_sd = -1, 0.0                                                    # the copies must actually disagree (sd > 0)
    for c, ks in copies.items():
        m, sd = float(np.mean(ks)), float(np.std(ks))
        if len(ks) >= 4 and lo <= m <= hi and sd > best_sd:
            best, best_sd = c, sd
    if best < 0:
        raise ValueError("no candidate cell has disagreeing copies with a mean in range")
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--candidates", type=int, default=40); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import pooltool as pt
    from exp16_wm import v_heatmap as H
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.paired_eval import find_run
    c = np.load(args.cache, allow_pickle=True)
    raw, xs, ys, valid, truth, smooth = c["raw"], c["xs"], c["ys"], c["valid"], c["truth"], c["smooth"]
    ball_r = float(pt.BallParams.default().R)
    left = int(len(H.live_balls(raw, TABLE_W, TABLE_L)))
    cells, obs, owner = H.rollout_inputs(raw, xs, ys, valid, 0.01, 8, args.seed, TABLE_W, TABLE_L, ball_r)
    t_c, s_c = truth[cells[:, 0], cells[:, 1]], smooth[cells[:, 0], cells[:, 1]]
    interior = (cells[:, 1] > 2) & (cells[:, 1] < len(xs) - 3) & (cells[:, 0] > 2) & (cells[:, 0] < len(ys) - 3)
    pool = np.nonzero(interior & np.isfinite(t_c) & np.isfinite(s_c) & (s_c >= 1.8) & (s_c <= 3.2))[0]            # copies' mean is mid-range
    cand = pool[np.argsort(-np.abs(t_c[pool] - s_c[pool]))[:args.candidates]]                                     # ① and ② differ most
    H._init([find_run(g) for g in args.runs], n_balls_raw(len(raw)))
    results: dict[int, np.ndarray] = {}
    positions: dict[int, np.ndarray] = {}
    for m in cand:
        sel = np.nonzero((owner[:, 0] == m) & (owner[:, 1] >= 0))[0]
        results[int(m)] = H._roll((obs[sel], left))
        positions[int(m)] = np.stack([obs[sel, 0] * TABLE_W, obs[sel, 1] * TABLE_L], axis=1)
    m = pick_story_cell(results)
    iy, ix = cells[m]
    own = np.nonzero((owner[:, 0] == m) & (owner[:, 1] < 0))[0][0]
    own_k = int(H._roll((obs[own:own + 1], left))[0])
    assert own_k == int(truth[iy, ix]), (own_k, truth[iy, ix])                            # the replay is the cache's ①
    assert abs(float(np.mean(results[m])) - float(smooth[iy, ix])) < 1e-6, (np.mean(results[m]), smooth[iy, ix])   # and its ②
    story = {"x": float(xs[ix]), "y": float(ys[iy]), "own": own_k, "copies_xy": positions[m], "copies_k": results[m], "iy": int(iy), "ix": int(ix)}
    plot_explain(c, story, args.out)


def plot_explain(c, story: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.v_heatmap import live_balls
    xs, ys, left = c["xs"], c["ys"], int(c["left"])
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"                                         # categorical slots 1-3 (① ② ③), fixed order
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False})
    ramp = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.2, 6.2), facecolor=surface, gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.18})
    fig.suptitle("① 실제 결과와 ② 1 cm 평균은 무엇이 다른가", color=ink, fontsize=15, x=0.02, ha="left", y=0.99)
    cx, cy = story["x"], story["y"]
    # ---- left: one position, the 1 cm circle and its eight copies
    a.set_facecolor(surface)
    w = 1.8
    a.set_xlim(-w, w); a.set_ylim(-w, w); a.set_aspect("equal")
    a.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=ink2, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2))
    a.text(0.72, 0.78, "1 cm", color=ink2, fontsize=10.5, ha="left", va="bottom", zorder=7)
    for k, (x, y) in zip(story["copies_k"], story["copies_xy"]):
        dx, dy = (x - cx) * 100, (y - cy) * 100
        a.plot([0, dx], [0, dy], color=grid, linewidth=1.0, zorder=1)
        a.scatter([dx], [dy], s=330, color=ramp(k / left), edgecolor=surface, linewidth=1.6, zorder=4)
        a.scatter([dx], [dy], s=330, facecolor="none", edgecolor=ink, linewidth=0.9, zorder=5)
        a.text(dx, dy, str(int(k)), ha="center", va="center", color="#ffffff" if k >= 2 else ink, fontsize=11.5, fontweight="bold", zorder=6)
    a.scatter([0], [0], s=420, marker="X", color=ramp(story["own"] / left), edgecolor=ink, linewidth=1.6, zorder=6)
    for bx, by in live_balls(c["raw"], TABLE_W, TABLE_L):
        if abs(bx - cx) < w / 100 * 1.5 and abs(by - cy) < w / 100 * 1.5:
            a.add_patch(Circle(((bx - cx) * 100, (by - cy) * 100), 2.86, facecolor=surface, edgecolor=ink, linewidth=1.5, zorder=3))
    mean = float(np.mean(story["copies_k"]))
    a.set_title("한 자리를 1 cm 흔들면 (확대, 단위 cm)", color=ink, fontsize=13, loc="left", pad=10)
    a.annotate(f"X: 원래 자리에서 한 번 친 값 = {story['own']}개 (①의 이 칸)", xy=(0, 0.12), xytext=(-1.7, 1.62), color=ink, fontsize=11, va="center",
               arrowprops={"arrowstyle": "-", "color": ink2, "linewidth": 1.0}, zorder=8)
    a.text(0.0, -0.17, "● 8개 점: 수구를 1 cm(점선 원) 안의 무작위 방향으로 옮긴 자리.\n   점 안의 숫자는 그 자리에서 새로 쳐 본 결과(공 수).\n"
           f"② = 8개의 평균 = {mean:.2f}개", transform=a.transAxes, color=ink2, fontsize=10.5, va="top", linespacing=1.5)
    a.tick_params(colors=ink2, labelsize=9); a.set_xlabel("가로로 옮긴 거리 (cm)", color=ink2); a.set_ylabel("세로로 옮긴 거리 (cm)", color=ink2)
    a.grid(color=grid, linewidth=0.8, zorder=0)
    for sp in a.spines.values():
        sp.set_color(ink2); sp.set_linewidth(0.8)
    # ---- right: the same table row, three maps
    b.set_facecolor(surface)
    row = story["iy"]
    xcm = xs * 100
    t, s_, v = c["truth"][row], c["smooth"][row], c["v"][row]
    b.axvspan((cx - 0.01) * 100, (cx + 0.01) * 100, color=grid, zorder=0)
    b.plot(xcm, t, color=c1, linewidth=1.3, marker="o", markersize=4.5, label="① 실제 결과 (칸마다 한 번)", zorder=3)
    b.plot(xcm, s_, color=c2, linewidth=2.2, label="② 1 cm 평균 (칸마다 8번의 평균)", zorder=4)
    b.plot(xcm, v, color=c3, linewidth=2.2, linestyle=(0, (5, 2)), label="③ V의 예측", zorder=4)
    b.scatter([cx * 100], [story["own"]], s=120, marker="X", color=c1, edgecolor=ink, linewidth=1.3, zorder=6)
    b.scatter([cx * 100], [mean], s=90, color=c2, edgecolor=ink, linewidth=1.3, zorder=6)
    b.annotate(f"같은 자리인데\n① {story['own']}개, ② {mean:.2f}개", xy=(cx * 100 + 0.8, (story["own"] + mean) / 2), xytext=(cx * 100 + 4.5, 1.25), color=ink, fontsize=10.5,
               va="center", bbox={"facecolor": surface, "edgecolor": grid, "boxstyle": "round,pad=0.3"}, arrowprops={"arrowstyle": "-", "color": ink2, "linewidth": 1.0}, zorder=9)
    b.set_ylim(-0.3, left + 0.4); b.set_yticks(range(0, left + 1)); b.set_xlim(0, TABLE_W * 100)
    b.set_title(f"같은 줄(수구를 놓는 세로 위치 y = {cy * 100:.0f} cm)을 가로로 따라가며 본 값", color=ink, fontsize=13, loc="left", pad=24)
    b.text(0.0, 1.02, "회색 띠는 왼쪽 그림의 1 cm 범위(가운데 X 기준 좌우 1 cm)", transform=b.transAxes, color=ink2, fontsize=10.5, va="bottom")
    b.set_xlabel("가로 위치 (cm)", color=ink2); b.set_ylabel("그 자리에서 놓칠 때까지 넣는 공 수", color=ink2)
    b.tick_params(colors=ink2, labelsize=9); b.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    b.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1, frameon=False, fontsize=10.5, labelcolor=ink)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        b.spines[sp].set_color(ink2)
    fig.savefig(out, dpi=150, facecolor=surface, bbox_inches="tight")
    print(f"saved {out}  (picked cell x {cx * 100:.0f} cm, y {cy * 100:.0f} cm: ① {story['own']}, copies {list(map(int, story['copies_k']))}, ② {mean:.2f})", flush=True)


if __name__ == "__main__":
    main()
