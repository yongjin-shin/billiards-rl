"""
exp16_wm/v_heatmap.py — the value of the cue ball's position as a map: one table layout, the cue ball put on a grid of positions.

For a fixed layout of the object balls it shows, for every free cue position (grid, `--step` metres apart):
  truth   the balls the policy pockets in a row before its first miss, played in the simulator from that position,
  smooth  the mean of the same at `--eps-mm` away in random directions (`--copies` times) — what a predictor that knows the position
          to that precision can best say (the V5-c ceiling),
  V       a trained V run's prediction (balls) for that position.
The simulator part is cached in an npz (`--cache`), so the figure can be redrawn without it.

Usage:
    python -m exp16_wm.v_heatmap --testset logs/teacher_data/v5/testset_s950.npz --runs "<PRm run dir glob>" \
        --v-run logs/experiments_v2/v6a_ctrl --cache logs/teacher_data/v6/heatmap.npz --out logs/teacher_data/v6/heatmap.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_G: dict = {}


def live_balls(raw: np.ndarray, w: float, l: float) -> np.ndarray:
    """(n, 2) positions in metres of the object balls still on the table of a raw observation (normalised coordinates)."""
    from exp16_wm.obs_transform import n_balls_raw
    n = n_balls_raw(len(raw))
    return np.array([[raw[2 + 3 * i] * w, raw[3 + 3 * i] * l] for i in range(n) if raw[4 + 3 * i] < 0.5]).reshape(-1, 2)


def valid_grid(raw: np.ndarray, step: float, ball_r: float, w: float, l: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cue positions on a regular grid: xs (nx,), ys (ny,), valid (ny, nx) — on the table (centre one radius from the cushion) and at least
    two radii from every ball still on it."""
    xs, ys = np.arange(ball_r, w - ball_r + 1e-9, step), np.arange(ball_r, l - ball_r + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    valid = np.ones(gx.shape, bool)
    for b in live_balls(raw, w, l):
        valid &= np.hypot(gx - b[0], gy - b[1]) >= 2.0 * ball_r + 1e-4
    return xs, ys, valid


def with_cue(raw: np.ndarray, x: float, y: float, w: float, l: float) -> np.ndarray:
    out = np.array(raw, np.float32, copy=True)
    out[0], out[1] = x / w, y / l
    return out


def rollout_inputs(raw: np.ndarray, xs: np.ndarray, ys: np.ndarray, valid: np.ndarray, eps_m: float, copies: int, seed: int,
                   w: float, l: float, ball_r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Everything to simulate: cells (M, 2) = (iy, ix) of the valid positions; obs (R, D) the position itself and its ε-moved copies of each
    cell (a copy that cannot be placed is left out); owner (R, 2) = (cell index, −1 for the position itself / copy number)."""
    from exp16_wm.v_perturb import perturb_cue
    cells = np.argwhere(valid)
    obs, owner = [], []
    for m, (iy, ix) in enumerate(cells):
        base = with_cue(raw, xs[ix], ys[iy], w, l)
        obs.append(base); owner.append((m, -1))
        rng = np.random.default_rng([seed, m])
        for c in range(copies):
            moved = perturb_cue(base, eps_m, rng, w, l, ball_r)
            if moved is not None:
                obs.append(moved); owner.append((m, c))
    return cells, np.stack(obs), np.array(owner)


def _init(runs: list[str], n_balls: int) -> None:
    import torch
    torch.set_num_threads(1)
    from exp16_wm.bc import TransformedPolicy
    from exp16_wm.lookahead import make_sim
    from exp16_wm.made_ensemble import load_ensemble
    pol = TransformedPolicy(load_ensemble(runs, "mean"), "relative")
    _G["act"], _G["sim"] = (lambda o: pol.act(o, deterministic=True)), make_sim(n_balls)


def _roll(job: tuple[np.ndarray, int]) -> np.ndarray:
    from exp16_wm.v_data import rollout
    obs, left = job
    return np.array([rollout(_G["sim"], _G["act"], o, left) for o in obs], np.int64)


def to_maps(shape: tuple[int, int], cells: np.ndarray, owner: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """truth (ny, nx): the position's own k; smooth: the mean over its placed copies; NaN where the cell is not valid (or has no copy)."""
    truth, smooth = np.full(shape, np.nan), np.full(shape, np.nan)
    own = owner[:, 1] < 0
    truth[cells[owner[own, 0], 0], cells[owner[own, 0], 1]] = k[own]
    s, n = np.zeros(len(cells)), np.zeros(len(cells))
    np.add.at(s, owner[~own, 0], k[~own]); np.add.at(n, owner[~own, 0], 1)
    ok = n > 0
    smooth[cells[ok, 0], cells[ok, 1]] = s[ok] / n[ok]
    return truth, smooth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", required=True); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--v-run", required=True)
    p.add_argument("--row", type=int, default=-1, help="test-set origin (default: the first with 5 balls left)")
    p.add_argument("--left", type=int, default=5); p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--eps-mm", type=float, default=10.0); p.add_argument("--copies", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--out3d", default="", help="also draw the three maps as 3-D surfaces (height = value) to this file")
    args = p.parse_args()
    if not os.path.exists(args.cache):
        compute(args)
    plot(args)
    if args.out3d:
        plot3d(args.cache, args.out3d)


def compute(args: argparse.Namespace) -> None:
    import time
    from multiprocessing import Pool
    import pooltool as pt
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_testset import next_values
    z = np.load(args.testset, allow_pickle=True)
    row = args.row if args.row >= 0 else int(np.nonzero(z["left"] == args.left)[0][0])
    raw = z["obs"][row].astype(np.float32)
    ball_r = float(pt.BallParams.default().R)
    left = int(len(live_balls(raw, TABLE_W, TABLE_L)))
    xs, ys, valid = valid_grid(raw, args.step, ball_r, TABLE_W, TABLE_L)
    cells, obs, owner = rollout_inputs(raw, xs, ys, valid, args.eps_mm / 1000.0, args.copies, args.seed, TABLE_W, TABLE_L, ball_r)
    print(f"origin row {row}: {left} balls, grid {valid.shape}, {len(cells)} free positions, {len(obs)} simulations", flush=True)
    t0 = time.time()
    runs = [find_run(g) for g in args.runs]
    with Pool(args.workers, initializer=_init, initargs=(runs, n_balls_raw(len(raw)))) as pool:
        ks = pool.map(_roll, [(c, left) for c in np.array_split(obs, args.workers * 8) if len(c)])
    truth, smooth = to_maps(valid.shape, cells, owner, np.concatenate(ks))
    print(f"simulated in {time.time() - t0:.0f}s", flush=True)
    base = np.stack([with_cue(raw, xs[ix], ys[iy], TABLE_W, TABLE_L) for iy, ix in cells])
    t = {"next_obs": base[:, None, :], "valid": np.ones((len(cells), 1), bool), "succ": np.ones((len(cells), 1), bool),
         "cleared": np.zeros((len(cells), 1), bool), "left_next": np.full((len(cells), 1), left)}
    v = np.full(valid.shape, np.nan)
    v[cells[:, 0], cells[:, 1]] = next_values(t, find_run(args.v_run))[:, 0]
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    np.savez(args.cache, raw=raw, xs=xs, ys=ys, valid=valid, truth=truth, smooth=smooth, v=v, left=left, row=row,
             meta=str({**vars(args), "table": (TABLE_W, TABLE_L), "ball_r": ball_r}))


def plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    from exp16_wm.networks import TABLE_L, TABLE_W
    c = np.load(args.cache, allow_pickle=True)
    xs, ys, left = c["xs"], c["ys"], int(c["left"])
    import pooltool as pt
    ball_r = float(pt.BallParams.default().R)
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False})
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    cmap.set_bad(surface)
    panels = [("① 실제 결과", "시뮬레이터로 쳐 본 값, 위치마다 한 번", c["truth"], False),
              ("② 1 cm 평균", "수구를 1 cm씩 흔들어 8번 친 평균", c["smooth"], True),
              ("③ V의 예측", "지금 V가 내놓는 값", c["v"], True)]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 8.2), facecolor=surface)
    fig.suptitle("수구를 어디에 놓느냐에 따라 값이 어떻게 달라지나 (공 배치는 고정)", color=ink, fontsize=15, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.935, f"값 = 그 자리에서 놓칠 때까지 연속으로 넣는 공 수 (남은 공 {left}개). 흰 원은 공, 검은 점은 포켓.",
             color=ink2, fontsize=11, ha="left")
    ext = (xs[0] - 0.01, xs[-1] + 0.01, ys[0] - 0.01, ys[-1] + 0.01)
    pockets = [(0, 0), (TABLE_W, 0), (0, TABLE_L / 2), (TABLE_W, TABLE_L / 2), (0, TABLE_L), (TABLE_W, TABLE_L)]
    for ax, (title, sub, z, contour) in zip(axes, panels):
        ax.set_facecolor(surface)
        im = ax.imshow(np.ma.masked_invalid(z), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=left, interpolation="nearest")
        if contour:
            gx, gy = np.meshgrid(xs, ys)
            ax.contour(gx, gy, np.nan_to_num(z, nan=0.0), levels=np.arange(1, left), colors=ink2, linewidths=0.9)
        for b in live_balls(c["raw"], TABLE_W, TABLE_L):
            ax.add_patch(Circle(b, ball_r, facecolor=surface, edgecolor=ink, linewidth=1.4, zorder=5))
        for pk in pockets:
            ax.add_patch(Circle(pk, 0.032, facecolor=ink, edgecolor="none", zorder=6))
        ax.set_xlim(-0.04, TABLE_W + 0.04); ax.set_ylim(-0.04, TABLE_L + 0.04); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(ink2); s.set_linewidth(0.8)
        ax.set_title(title, color=ink, fontsize=13, loc="left", pad=22)
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=ink2, fontsize=10, va="bottom")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=range(0, left + 1))
    cb.ax.set_title("넣는 공 수", color=ink2, fontsize=11, pad=10); cb.ax.tick_params(colors=ink2)
    cb.outline.set_edgecolor(ink2)
    fig.savefig(args.out, dpi=150, facecolor=surface, bbox_inches="tight")
    print(f"saved {args.out}", flush=True)


def plot3d(cache: str, out: str, elev: float = 38.0, azim: float = -58.0) -> None:
    """The three maps of `cache` as surfaces over the table: x, y = the cue ball's position, height and colour = the value (balls)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from exp16_wm.networks import TABLE_L, TABLE_W
    c = np.load(cache, allow_pickle=True)
    xs, ys, left = c["xs"], c["ys"], int(c["left"])
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False})
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(16, 6.6), facecolor=surface)
    fig.suptitle("같은 지도를 3차원으로: 바닥 = 수구 위치, 높이 = 그 자리에서 놓칠 때까지 넣는 공 수", color=ink, fontsize=14, x=0.02, ha="left", y=0.98)
    panels = [("① 실제 결과 (위치마다 한 번)", c["truth"]), ("② 1 cm 평균 (8번)", c["smooth"]), ("③ V의 예측", c["v"])]
    balls = live_balls(c["raw"], TABLE_W, TABLE_L)
    for i, (title, z) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", facecolor=surface)
        ax.plot_surface(X, Y, np.where(np.isfinite(z), z, np.nan), cmap=cmap, vmin=0, vmax=left, rcount=len(ys), ccount=len(xs),
                        linewidth=0, antialiased=False, shade=False)
        ax.scatter(balls[:, 0], balls[:, 1], np.zeros(len(balls)), s=120, facecolor=surface, edgecolor=ink, linewidth=1.5, depthshade=False, zorder=10)
        ax.set_zlim(0, left); ax.set_zticks(range(0, left + 1)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_box_aspect((TABLE_W, TABLE_L, 1.3)); ax.view_init(elev=elev, azim=azim)
        ax.tick_params(axis="z", colors=ink2, labelsize=9)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor(surface); a.pane.set_edgecolor("#d8d7d2")
        ax.set_title(title, color=ink, fontsize=12, loc="left", pad=2)
    fig.subplots_adjust(left=0.0, right=1.0, wspace=-0.02, top=0.93, bottom=0.0)
    fig.savefig(out, dpi=150, facecolor=surface, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
