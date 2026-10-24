"""
exp16_wm/v_heatmap_radii.py — the value map of one table layout averaged over a growing radius, side by side in 2-D.

From the cache of exp16_wm.v_heatmap: ① the actual result (one shot per cue position), the 1 cm average (the cache's real simulations,
8 moved copies per position), 2 ~ 5 cm averages (approximation: ① averaged over the cue positions within that radius on the 2 cm grid,
no new simulation) and V's prediction, all on one colour scale. Each panel is captioned with how often neighbouring cells differ by at
least one ball.

`--simulate` replaces the 2 ~ 5 cm approximations with real simulations: for every free cue position, 8 copies moved by that radius in a
random direction are played out by the policy (the same recipe as the cache's 1 cm map) and averaged.

Usage:
    python -m exp16_wm.v_heatmap_radii --cache logs/teacher_data/v6/heatmap_left5.npz --out logs/teacher_data/v6/heatmap_radii.png
    python -m exp16_wm.v_heatmap_radii --cache logs/teacher_data/v6/heatmap_left5.npz --simulate --runs "<PRm run dir glob>" \
        --real logs/teacher_data/v6/heatmap_left5_radii.npz --out logs/teacher_data/v6/heatmap_radii_real.png
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import convolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def disk_average(z: np.ndarray, radius_cells: float) -> np.ndarray:
    """Mean of z over the finite cells within radius_cells of each finite cell (NaN cells are neither used nor filled)."""
    ok = np.isfinite(z)
    n = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    kernel = (xx ** 2 + yy ** 2 <= radius_cells ** 2 + 1e-9).astype(float)
    num = convolve(np.where(ok, z, 0.0), kernel, mode="constant")
    den = convolve(ok.astype(float), kernel, mode="constant")
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan)


def jump_share(z: np.ndarray) -> float:
    """Share of adjacent cell pairs (both finite) whose values differ by at least 1 ball."""
    d = np.concatenate([np.abs(z[:, 1:] - z[:, :-1]).ravel(), np.abs(z[1:, :] - z[:-1, :]).ravel()])
    d = d[np.isfinite(d)]
    return float((d >= 1.0).mean()) if len(d) else float("nan")


def panels_of(c, radii_cm: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0), real: dict[float, np.ndarray] | None = None) -> list[tuple[str, str, np.ndarray]]:
    """(title, note, map) for every panel: ①, the 1 cm simulation average, the other radii, and V. A radius found in `real` (radius in cm →
    its simulated map) is the simulation; the others are the grid approximation (① averaged over the cells within the radius)."""
    step_cm = float(c["xs"][1] - c["xs"][0]) * 100.0
    real = real or {}
    out = [("① 실제 결과", "칸마다 한 번", c["truth"]), ("1 cm 평균", "8번 실제 시뮬레이션", c["smooth"])]
    for r in radii_cm:
        if r in real:
            out.append((f"{r:g} cm 평균", "8번 실제 시뮬레이션", real[r]))
        else:
            out.append((f"{r:g} cm 평균", "근사(주변 칸 평균)", disk_average(c["truth"], r / step_cm)))
    out.append(("③ V의 예측", "지금 V", c["v"]))
    return out


def correlation_table(panels: list[tuple[str, str, np.ndarray]]) -> list[tuple[str, float, float, float]]:
    """Per panel: (title, share of adjacent cells that differ by ≥ 1 ball, correlation with ①, correlation with V), over the cells finite in all."""
    truth, v = panels[0][2], panels[-1][2]
    rows = []
    for title, _, z in panels:
        m = np.isfinite(z) & np.isfinite(truth) & np.isfinite(v)
        rows.append((title, jump_share(z), float(np.corrcoef(z[m], truth[m])[0, 1]), float(np.corrcoef(z[m], v[m])[0, 1])))
    return rows


def simulate_radii(cache: str, runs: list[str], radii_cm: tuple[float, ...], out: str, workers: int = 8, seed: int = 0, copies: int = 8) -> None:
    """For each radius: the mean k over `copies` copies moved by that radius, at every free cue position of the cache (npz: smooth_<cm> maps)."""
    import time
    from multiprocessing import Pool
    import pooltool as pt
    from exp16_wm import v_heatmap as H
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.paired_eval import find_run
    c = np.load(cache, allow_pickle=True)
    raw, xs, ys, valid = c["raw"], c["xs"], c["ys"], c["valid"]
    ball_r = float(pt.BallParams.default().R)
    left = int(len(H.live_balls(raw, TABLE_W, TABLE_L)))
    maps = {}
    with Pool(workers, initializer=H._init, initargs=([find_run(g) for g in runs], n_balls_raw(len(raw)))) as pool:
        for r in radii_cm:
            t0 = time.time()
            cells, obs, owner = H.rollout_inputs(raw, xs, ys, valid, r / 100.0, copies, seed, TABLE_W, TABLE_L, ball_r)
            sel = owner[:, 1] >= 0                                              # the copies only (the unmoved position is the cache's ①)
            ks = np.concatenate(pool.map(H._roll, [(ch, left) for ch in np.array_split(obs[sel], workers * 8) if len(ch)]))
            _, smooth = H.to_maps(valid.shape, cells, owner[sel], ks)
            maps[f"smooth_{r:g}"] = smooth
            print(f"{r:g} cm: {int(sel.sum())} simulations in {time.time() - t0:.0f}s, cells without a copy {int(np.isnan(smooth[valid]).sum())}", flush=True)
    np.savez(out, radii_cm=np.array(radii_cm), **maps)


def load_real(path: str, radii_cm: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)) -> dict[float, np.ndarray]:
    z = np.load(path)
    return {r: z[f"smooth_{r:g}"] for r in radii_cm if f"smooth_{r:g}" in z.files}


def plot_radii(cache: str, out: str, real_path: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.v_heatmap import live_balls
    c = np.load(cache, allow_pickle=True)
    xs, ys, left = c["xs"], c["ys"], int(c["left"])
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False})
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    cmap.set_bad(surface)
    panels = panels_of(c, real=load_real(real_path) if real_path and os.path.exists(real_path) else None)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels) + 0.8, 5.9), facecolor=surface)
    fig.subplots_adjust(left=0.01, right=0.935, top=0.80, bottom=0.14, wspace=0.10)
    fig.text(0.01, 0.965, "수구 위치 값 지도를 평균 반지름별로 (같은 공 배치, 같은 색 척도)", color=ink, fontsize=15, ha="left", va="top")
    ext = (xs[0] - 0.01, xs[-1] + 0.01, ys[0] - 0.01, ys[-1] + 0.01)
    gx, gy = np.meshgrid(xs, ys)
    pockets = [(0, 0), (TABLE_W, 0), (0, TABLE_L / 2), (TABLE_W, TABLE_L / 2), (0, TABLE_L), (TABLE_W, TABLE_L)]
    balls = live_balls(c["raw"], TABLE_W, TABLE_L)
    for i, (ax, (title, note, z)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(surface)
        im = ax.imshow(np.ma.masked_invalid(z), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=left, interpolation="nearest")
        if i > 0:
            ax.contour(gx, gy, np.nan_to_num(z, nan=0.0), levels=np.arange(1, left), colors=ink2, linewidths=0.7)
        for b in balls:
            ax.add_patch(Circle(b, 0.028575, facecolor=surface, edgecolor=ink, linewidth=1.2, zorder=5))
        for pk in pockets:
            ax.add_patch(Circle(pk, 0.032, facecolor=ink, edgecolor="none", zorder=6))
        ax.set_xlim(-0.04, TABLE_W + 0.04); ax.set_ylim(-0.04, TABLE_L + 0.04); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(ink2); sp.set_linewidth(0.8)
        ax.set_title(title, color=ink, fontsize=12, pad=24, loc="center")
        ax.text(0.5, 1.012, note, transform=ax.transAxes, color=ink2, fontsize=9, ha="center", va="bottom")
        ax.text(0.5, -0.02, f"이웃 칸이 1개 이상\n다른 비율 {jump_share(z) * 100:.1f}%", transform=ax.transAxes, color=ink, fontsize=10, ha="center", va="top")
    cax = fig.add_axes([0.948, 0.14, 0.011, 0.66])
    cb = fig.colorbar(im, cax=cax, ticks=range(0, left + 1))
    cb.ax.set_title("넣는 공 수", color=ink2, fontsize=10, pad=8); cb.ax.tick_params(colors=ink2); cb.outline.set_edgecolor(ink2)
    fig.savefig(out, dpi=140, facecolor=surface, bbox_inches="tight")
    print(f"saved {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--real", default="", help="npz of simulated 2 ~ 5 cm maps (written by --simulate, read when it exists)")
    p.add_argument("--simulate", action="store_true", help="simulate the 2 ~ 5 cm maps into --real first")
    p.add_argument("--runs", nargs="+", default=[]); p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.simulate:
        if not args.real or not args.runs:
            raise SystemExit("--simulate needs --real and --runs")
        simulate_radii(args.cache, args.runs, (2.0, 3.0, 4.0, 5.0), args.real, args.workers)
    plot_radii(args.cache, args.out, args.real)
    c = np.load(args.cache, allow_pickle=True)
    print("title | adjacent-cell jump share | corr with ① | corr with V")
    for t, j, r_t, r_v in correlation_table(panels_of(c, real=load_real(args.real) if args.real and os.path.exists(args.real) else None)):
        print(f"{t:12s} | {j * 100:5.1f}% | {r_t:5.2f} | {r_v:5.2f}")


if __name__ == "__main__":
    main()
