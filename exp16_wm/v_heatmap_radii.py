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
from scipy.ndimage import convolve, gaussian_filter

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


def zone_auc(pred: np.ndarray, zone_map: np.ndarray, threshold: float) -> float:
    """AUC of `pred` for telling the cells whose zone_map value is at least `threshold` ("good zone") from the others, over cells finite in
    both; NaN if one of the two groups is empty."""
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(pred) & np.isfinite(zone_map)
    y = zone_map[m] >= threshold
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, pred[m]))


def noise_report(a: np.ndarray, b: np.ndarray, v: np.ndarray, threshold: float = 4.0) -> dict[str, float]:
    """Two independent simulated maps a, b of the same radius (different copies) and V's map v: how much of a map is sampling noise.
    reproducibility = corr(a, b); the 16-copy average (a + b) / 2 has reliability 2r / (1 + r) (Spearman-Brown); the zone agreement is
    the share of cells on which both maps put the cell on the same side of `threshold`; replicate_auc is how well b finds a's good zone
    (a perfect predictor of the smooth map, handicapped by the noise in both maps)."""
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(v)
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    avg = (a + b) / 2
    rel = 2 * r / (1 + r)
    za, zb = a[m] >= threshold, b[m] >= threshold
    cv = lambda z: float(np.corrcoef(v[m], z[m])[0, 1])
    return {"reproducibility": r, "reliability_16": rel, "jump_single": (jump_share(a) + jump_share(b)) / 2, "jump_16": jump_share(avg),
            "corr_v_single": (cv(a) + cv(b)) / 2, "corr_v_16": cv(avg), "corr_v_16_corrected": cv(avg) / float(np.sqrt(rel)),
            "zone_agreement": float(np.mean(za == zb)), "replicate_auc": float(roc_auc_score(za, b[m])) if 0 < za.mean() < 1 else float("nan"),
            "v_auc": float(roc_auc_score(za, v[m])) if 0 < za.mean() < 1 else float("nan")}


def blur(z: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian blur of a map that ignores NaN cells (they stay NaN); sigma 0 returns the map unchanged."""
    if sigma_cells <= 0:
        return z.copy()
    ok = np.isfinite(z)
    num = gaussian_filter(np.where(ok, z, 0.0), sigma_cells, mode="constant")
    den = gaussian_filter(ok.astype(float), sigma_cells, mode="constant")
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan)


def boltzmann(z: np.ndarray, beta: float) -> np.ndarray:
    """The distribution over the finite cells of z with probability ∝ exp(beta · z) (NaN cells get 0)."""
    ok = np.isfinite(z)
    w = np.exp(beta * (z[ok] - z[ok].max()))
    p = np.zeros_like(z, dtype=float)
    p[ok] = w / w.sum()
    return p


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))))


def trend_fit(pred: np.ndarray, ref: np.ndarray, sigma_cells: float = 1.0, tau_ref: float = 1.0) -> dict:
    """Compare a predictor map with a reference map as trends: both blurred by sigma_cells over the cells finite in both; the reference
    becomes the Boltzmann distribution p ∝ exp(ref / tau_ref) (tau_ref in balls) over the cue positions; the predictor becomes
    q ∝ exp(beta · standardised pred) with beta ≥ 0 fitted to minimise KL(p ‖ q) (so only WHERE the predictor puts mass counts, not the
    scale of its values). Returns the blurred maps, p, q, beta, the KLs and explained = 1 − KL(p ‖ q) / KL(p ‖ uniform): 0 = no better
    than a flat map, 1 = the same distribution."""
    m = np.isfinite(pred) & np.isfinite(ref)
    pr, rf = blur(np.where(m, pred, np.nan), sigma_cells), blur(np.where(m, ref, np.nan), sigma_cells)
    p = boltzmann(rf, 1.0 / tau_ref)
    uniform = np.where(m, 1.0 / m.sum(), 0.0)
    kl_uni = kl_div(p, uniform)
    std = float(np.nanstd(pr))
    z = (pr - np.nanmean(pr)) / std if std > 0 else np.zeros_like(pr)
    betas = np.concatenate([[0.0], np.logspace(-2, 1.5, 120)])
    kls = [kl_div(p, boltzmann(z, b)) for b in betas]
    i = int(np.argmin(kls))
    ok = np.isfinite(pr)
    return {"pred_blur": pr, "ref_blur": rf, "p": p, "q": boltzmann(z, float(betas[i])), "uniform": uniform, "beta": float(betas[i]), "kl": float(kls[i]),
            "kl_uniform": float(kl_uni), "explained": float(1.0 - kls[i] / kl_uni) if kl_uni > 0 else float("nan"),
            "corr": float(np.corrcoef(pr[ok], rf[ok])[0, 1])}


def trend_report(pred: np.ndarray, ref: np.ndarray, sigma_cells: float = 1.0, tau_ref: float = 1.0) -> dict[str, float]:
    """The numbers of trend_fit (explained, beta, kl, kl_uniform, corr)."""
    f = trend_fit(pred, ref, sigma_cells, tau_ref)
    return {k: f[k] for k in ("explained", "beta", "kl", "kl_uniform", "corr")}


def plot_trend(c, ref: np.ndarray, preds: dict[str, np.ndarray], out: str, sigma_cells: float = 2.0, tau_ref: float = 1.0, ref_name: str = "3 cm 평균") -> None:
    """Draw the Boltzmann fit in 2-D: the blurred reference and its distribution p, then for each predictor its fitted distribution q (top)
    and where q puts more (red) or less (blue) mass than p (bottom). Distributions are shown as multiples of the uniform distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Circle
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.v_heatmap import live_balls
    xs, ys, left = c["xs"], c["ys"], int(c["left"])
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False})
    blue = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    orange = LinearSegmentedColormap.from_list("orange_seq", ["#fdebdf", "#f9cdb0", "#f3a273", "#eb6834", "#c24e1d", "#8a3512"])
    div = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
    for cm in (blue, orange, div):
        cm.set_bad(surface)
    fits = {n: trend_fit(pm, ref, sigma_cells, tau_ref) for n, pm in preds.items()}
    first = next(iter(fits.values()))
    N = int(np.isfinite(first["p"] + first["q"]).sum())
    ratio = lambda d: np.where(d > 0, d * N, np.nan)                          # multiples of the uniform distribution
    top_max = max(np.nanmax(ratio(first["p"])), *[np.nanmax(ratio(f["q"])) for f in fits.values()])
    dmax = max(np.nanmax(np.abs(ratio(f["q"]) - ratio(f["p"]))) for f in fits.values())
    ncol = 1 + len(fits)
    fig, axes = plt.subplots(2, ncol, figsize=(2.55 * ncol + 1.2, 11.2), facecolor=surface)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.86, bottom=0.04, wspace=0.12, hspace=0.20)
    fig.text(0.02, 0.975, "Boltzmann으로 맞췄을 때 (2D)", color=ink, fontsize=15, ha="left", va="top")
    fig.text(0.02, 0.945, f"기준 = {ref_name}을 블러 σ {sigma_cells * 2:g} cm로 매끄럽게 한 경향, 분포 p ∝ exp(값 / {tau_ref:g}개). 예측자는 같은 블러 뒤 온도 하나를 맞춘 분포 q.\n"
             "색 = 균등 분포의 몇 배로 그 자리에 확률이 몰리는가(위). 아래는 q가 p보다 많이(빨강) 또는 적게(파랑) 몰린 곳.", color=ink2, fontsize=10.5, ha="left", va="top")
    ext = (xs[0] - 0.01, xs[-1] + 0.01, ys[0] - 0.01, ys[-1] + 0.01)
    pockets = [(0, 0), (TABLE_W, 0), (0, TABLE_L / 2), (TABLE_W, TABLE_L / 2), (0, TABLE_L), (TABLE_W, TABLE_L)]
    balls = live_balls(c["raw"], TABLE_W, TABLE_L)

    def frame(ax):
        ax.set_facecolor(surface)
        for b in balls:
            ax.add_patch(Circle(b, 0.028575, facecolor=surface, edgecolor=ink, linewidth=1.1, zorder=5))
        for pk in pockets:
            ax.add_patch(Circle(pk, 0.032, facecolor=ink, edgecolor="none", zorder=6))
        ax.set_xlim(-0.04, TABLE_W + 0.04); ax.set_ylim(-0.04, TABLE_L + 0.04); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(ink2); sp.set_linewidth(0.8)

    def show(ax, z, cmap, norm=None, vmin=None, vmax=None):
        return ax.imshow(np.ma.masked_invalid(z), origin="lower", extent=ext, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")

    im_b = show(axes[0, 0], first["ref_blur"], blue, vmin=0, vmax=left)
    frame(axes[0, 0]); axes[0, 0].set_title("기준의 값 (블러)\n공 수", color=ink, fontsize=11.5, pad=8)
    im_p = show(axes[1, 0], ratio(first["p"]), orange, vmin=0, vmax=top_max)
    frame(axes[1, 0]); axes[1, 0].set_title("기준의 분포 p\n(Boltzmann)", color=ink, fontsize=11.5, pad=8)
    for j, (name, f) in enumerate(fits.items(), start=1):
        show(axes[0, j], ratio(f["q"]), orange, vmin=0, vmax=top_max); frame(axes[0, j])
        axes[0, j].set_title(f"{name}\n설명하는 몫 {f['explained']:.2f} · 상관 {f['corr']:.2f}", color=ink, fontsize=11.5, pad=8)
        im_d = show(axes[1, j], ratio(f["q"]) - ratio(f["p"]), div, norm=TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)); frame(axes[1, j])
        axes[1, j].set_title(f"q-p (역온도 {f['beta']:.2f})", color=ink2, fontsize=10.5, pad=8)
    cb0 = fig.colorbar(im_b, ax=axes[0, 0], orientation="horizontal", fraction=0.04, pad=0.02, ticks=range(0, left + 1))
    cb0.ax.tick_params(colors=ink2, labelsize=8); cb0.outline.set_edgecolor(ink2)
    cax1 = fig.add_axes([0.945, 0.50, 0.012, 0.26]); cb1 = fig.colorbar(im_p, cax=cax1); cb1.ax.set_title("균등의\n배수", color=ink2, fontsize=9.5, pad=6)
    cax2 = fig.add_axes([0.945, 0.08, 0.012, 0.26]); cb2 = fig.colorbar(im_d, cax=cax2); cb2.ax.set_title("q-p", color=ink2, fontsize=9.5, pad=6)
    for cb in (cb1, cb2):
        cb.ax.tick_params(colors=ink2, labelsize=9); cb.outline.set_edgecolor(ink2)
    fig.savefig(out, dpi=140, facecolor=surface, bbox_inches="tight")
    print(f"saved {out}", flush=True)


def v_map_of_run(c, run_dir: str) -> np.ndarray:
    """A trained V run's prediction (balls) with the cue ball on every free cell of the cache."""
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_heatmap import with_cue
    from exp16_wm.v_testset import next_values
    xs, ys, valid = c["xs"], c["ys"], c["valid"]
    cells = np.argwhere(valid)
    base = np.stack([with_cue(c["raw"], xs[ix], ys[iy], TABLE_W, TABLE_L) for iy, ix in cells])
    n = len(cells)
    t = {"next_obs": base[:, None, :], "valid": np.ones((n, 1), bool), "succ": np.ones((n, 1), bool), "cleared": np.zeros((n, 1), bool),
         "left_next": np.full((n, 1), int(c["left"]))}
    out = np.full(valid.shape, np.nan)
    out[cells[:, 0], cells[:, 1]] = next_values(t, find_run(run_dir))[:, 0]
    return out


def made_max_map(c, runs: list[str]) -> np.ndarray:
    """The made head's highest pair probability with the cue ball on every free cell of the cache (the "is there an easy shot" baseline)."""
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_heatmap import with_cue
    from exp16_wm.v_testset import made_max_values
    xs, ys, valid = c["xs"], c["ys"], c["valid"]
    cells = np.argwhere(valid)
    base = np.stack([with_cue(c["raw"], xs[ix], ys[iy], TABLE_W, TABLE_L) for iy, ix in cells])
    n = len(cells)
    t = {"next_obs": base[:, None, :], "valid": np.ones((n, 1), bool), "succ": np.ones((n, 1), bool), "cleared": np.zeros((n, 1), bool)}
    out = np.full(valid.shape, np.nan)
    out[cells[:, 0], cells[:, 1]] = made_max_values(t, [find_run(g) for g in runs])[:, 0]
    return out


def zone_table(panels: list[tuple[str, str, np.ndarray]], extra: dict[str, np.ndarray], threshold: float = 4.0) -> list[tuple[str, float, dict[str, float]]]:
    """For every averaged map (all panels but ① and V): the share of cells in its good zone (≥ threshold) and the AUC for finding that zone
    of V, of ① (one exact shot per cell) and of each predictor in `extra`."""
    truth, v = panels[0][2], panels[-1][2]
    preds = {"V": v, "① 한 번 친 결과": truth, **extra}
    rows = []
    for title, _, z in panels[1:-1]:
        ok = np.isfinite(z)
        rows.append((title, float((z[ok] >= threshold).mean()), {n: zone_auc(pv, z, threshold) for n, pv in preds.items()}))
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
    p.add_argument("--replicate", default="", help="npz of the same radius simulated with another seed (smooth_<--replicate-cm>): print the noise report")
    p.add_argument("--replicate-cm", type=float, default=3.0)
    p.add_argument("--trend-plot", default="", help="with --trend: also draw the Boltzmann fit of V runs / baselines to this PNG")
    p.add_argument("--trend", action="store_true", help="compare V runs (--v-runs) and baselines with the simulated --replicate-cm map as trends (blur + Boltzmann + KL)")
    p.add_argument("--v-runs", nargs="*", default=[])
    args = p.parse_args()
    if args.trend:
        c = np.load(args.cache, allow_pickle=True)
        cm = args.replicate_cm
        ref, rep = np.load(args.real)[f"smooth_{cm:g}"], np.load(args.replicate)[f"smooth_{cm:g}"]
        preds = {"V v6a_ctrl (정확한 정답)": c["v"]}
        for r in args.v_runs:
            preds[os.path.basename(r.rstrip("/"))] = v_map_of_run(c, r)
        if args.runs:
            preds["made 최댓값"] = made_max_map(c, args.runs)
        preds["① 한 번 친 결과"] = c["truth"]
        preds["같은 지도 다른 seed (잡음 한계)"] = rep
        print(f"\n== 기준 = {cm:g} cm 평균(seed 0). 경향: 지도와 예측자를 블러 σ 2 cm / 4 cm로 매끄럽게 한 뒤 Boltzmann 분포로 바꿔 온도 하나를 맞춘 KL이 설명하는 몫(상관). 구역: 평균 4개 이상인 좋은 구역의 AUC")
        print(f"  {'':34s} {'경향 σ2cm':>16s} {'경향 σ4cm':>16s}   구역 AUC")
        for n, pm in preds.items():
            r1, r2 = trend_report(pm, ref, 1.0), trend_report(pm, ref, 2.0)
            print(f"  {n:34s} {r1['explained']:7.3f} ({r1['corr']:.2f}) {r2['explained']:7.3f} ({r2['corr']:.2f})   {zone_auc(pm, ref, 4.0):.3f}")
        if args.trend_plot:
            keep = {n: preds[n] for n in ("V v6a_ctrl (정확한 정답)", "made 최댓값", "같은 지도 다른 seed (잡음 한계)") if n in preds}
            plot_trend(c, ref, keep, args.trend_plot, sigma_cells=2.0, ref_name=f"{cm:g} cm 평균")
        return
    if args.replicate:
        c = np.load(args.cache, allow_pickle=True)
        a, b = np.load(args.real)[f"smooth_{args.replicate_cm:g}"], np.load(args.replicate)[f"smooth_{args.replicate_cm:g}"]
        for k, val in noise_report(a, b, c["v"]).items():
            print(f"{k:22s} {val:.3f}")
        return
    if args.simulate:
        if not args.real or not args.runs:
            raise SystemExit("--simulate needs --real and --runs")
        simulate_radii(args.cache, args.runs, (2.0, 3.0, 4.0, 5.0), args.real, args.workers)
    plot_radii(args.cache, args.out, args.real)
    c = np.load(args.cache, allow_pickle=True)
    panels = panels_of(c, real=load_real(args.real) if args.real and os.path.exists(args.real) else None)
    print("title | adjacent-cell jump share | corr with ① | corr with V")
    for t, j, r_t, r_v in correlation_table(panels):
        print(f"{t:12s} | {j * 100:5.1f}% | {r_t:5.2f} | {r_v:5.2f}")
    extra = {"made 최댓값": made_max_map(c, args.runs)} if args.runs else {}
    print("\ngood zone = average >= 4 balls; AUC of each predictor for finding that zone")
    for t, share, aucs in zone_table(panels, extra):
        print(f"{t:10s} zone share {share * 100:4.1f}% | " + " | ".join(f"{n} {a:.3f}" for n, a in aucs.items()))


if __name__ == "__main__":
    main()
