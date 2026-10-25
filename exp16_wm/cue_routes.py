"""
exp16_wm/cue_routes.py — for one start: with one cushion, how many different walls can send the cue ball to the same goal?

Replays the one-cushion shots of one start (exp16_wm.cue_goal_data) to read which wall each hit, then for random goals counts the walls that
have a recorded shot ending within 2 cm of the goal, and draws the routes to one goal.

Usage:
    python -m exp16_wm.cue_routes --data logs/ct0b/collect_200.npz --out logs/ct2a/k1_routes.png
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import TABLE_L, TABLE_W
from exp16_wm.cue_goal_head import load_rows
from exp16_wm.cue_goal_data import standing_mask  # noqa: F401  (kept for parity with the other plots)
from exp16_wm.plot_cue_goal_coverage import INK, INK2, SURFACE, _font, draw_table
from exp16_wm.single_ball import CUSHION_EVENTS, make_cue_only_system, sample_starts
from simulator import play_shot

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, fixed order


def replay(start: np.ndarray, phi: np.ndarray, v: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per shot: id of the cushion hit, the cue ball position at that contact, and where it ended."""
    system = make_cue_only_system(float(start[0]), float(start[1]))
    ids, contact, end = [], np.zeros((len(phi), 2)), np.zeros((len(phi), 2))
    for i in range(len(phi)):
        out = play_shot(system, float(phi[i]), float(v[i]))
        hit = [e for e in out.events if e.event_type in CUSHION_EVENTS][0]
        ids.append(next(str(a.id) for a in hit.agents if str(a.id) != "cue"))
        contact[i] = next(a.final.state.rvw[0][:2] for a in hit.agents if str(a.id) == "cue")
        end[i] = out.balls["cue"].state.rvw[0][:2]
    return ids, contact, end


def wall_names(ids: list[str], contact: np.ndarray) -> dict[str, str]:
    """Name each cushion by where its contacts lie (bottom / top / left / right, lower or upper half of a long side); '...t' = pocket jaw."""
    names = {}
    for cid in sorted(set(ids)):
        if cid.endswith("t"):
            names[cid] = "포켓 모서리"; continue
        c = np.median(contact[[i for i, x in enumerate(ids) if x == cid]], axis=0)
        if c[1] < 0.2: names[cid] = "아래 짧은 변"
        elif c[1] > TABLE_L - 0.2: names[cid] = "위 짧은 변"
        else: names[cid] = ("왼쪽 " if c[0] < TABLE_W / 2 else "오른쪽 ") + ("아래" if c[1] < TABLE_L / 2 else "위")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-goals", type=int, default=1000)
    args = ap.parse_args()
    _font()
    d = load_rows(args.data, 1, 40, 1000, 0)
    starts = d["starts"]
    centre = np.array([TABLE_W / 2, TABLE_L / 2])
    ex = d["train_starts"][np.argmin(np.linalg.norm(starts[d["train_starts"]] - centre, axis=1))]
    rows = np.nonzero(d["ok"] & (d["sidx"] == ex))[0]
    t0 = time.time()
    ids, contact, end = replay(starts[ex], d["phi"][rows], d["v"][rows])
    print(f"start {ex} at {starts[ex].round(3)}: replayed {len(rows):,} one-cushion shots in {time.time() - t0:.0f}s")
    names = wall_names(ids, contact)
    group = np.array([names[c] for c in ids])                     # several ids can be one wall (the pocket jaws are many)
    order = ["왼쪽 아래", "왼쪽 위", "오른쪽 아래", "오른쪽 위", "아래 짧은 변", "위 짧은 변", "포켓 모서리"]
    color = dict(zip(order, COLORS + ["#8a8984"]))
    print("shots per wall:", {w: int((group == w).sum()) for w in order})
    trees = {w: cKDTree(end[group == w]) for w in order if (group == w).any()}
    goals = sample_starts(args.n_goals, 12345)
    reach = np.array([[trees[w].query(g)[0] <= 0.02 for w in trees] for g in goals])
    n_walls = reach.sum(1)
    print("goals by number of different walls that have a recorded one-cushion shot within 2 cm:",
          {int(n): int((n_walls == n).sum()) for n in range(0, len(trees) + 1)})
    print(f"  at least one: {np.mean(n_walls >= 1):.3f}, two or more: {np.mean(n_walls >= 2):.3f}, mean {n_walls.mean():.2f}")

    pick = np.nonzero((n_walls >= 3) & (np.linalg.norm(goals - starts[ex], axis=1) > 0.5))[0]
    g = goals[pick[0]] if len(pick) else goals[int(np.argmax(n_walls))]
    fig, ax = plt.subplots(figsize=(4.6, 8.2), facecolor=SURFACE); ax.set_facecolor(SURFACE); draw_table(ax)
    s = starts[ex]
    for w in trees:
        sel = np.nonzero(group == w)[0]
        j = sel[np.argmin(np.linalg.norm(end[sel] - g, axis=1))]
        if np.linalg.norm(end[j] - g) > 0.02:
            continue
        ax.plot([s[0], contact[j, 0], end[j, 0]], [s[1], contact[j, 1], end[j, 1]], color=color[w], lw=1.8, zorder=3)
        ax.scatter(*contact[j], s=22, color=color[w], zorder=4)
        ax.annotate(w, contact[j], xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color=INK2, zorder=6)
    ax.scatter(*g, marker="x", s=120, color=INK, linewidths=2.2, zorder=7)
    ax.scatter(*s, s=110, facecolors=SURFACE, edgecolors="#eb6834", linewidths=2.4, zorder=7)
    ax.set_title("같은 목표(×)에 쿠션 1번으로 가는 길들\n주황 원: 시작, 선 색: 맞힌 벽", color=INK, fontsize=11, pad=8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("saved", args.out, " goal", g.round(3))


if __name__ == "__main__":
    main()
