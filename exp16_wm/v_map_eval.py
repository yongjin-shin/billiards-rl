"""
exp16_wm/v_map_eval.py — V7: compare V models on held-out layouts (value maps V(n, m | s)), many layouts at once.

For every held-out layout the model's value on its ~700 cue positions is compared with the reference map (the mean of all 15 ring copies,
"robust"): per-layout Pearson correlation (primary), the same after a Gaussian blur of both maps (σ 4 and 8 cm), the share of the
Boltzmann KL it explains (blurred maps, p ∝ exp(ref), q ∝ exp(β · standardised model), β fitted), the AUC for finding the good zone
(reference ≥ 4 balls) and, as a side check, the correlation with the exact k. Layout scores are averaged over layouts with a bootstrap
95% interval (over layouts), and arms are compared by paired per-layout differences. The noise limit of the reference comes from its own
copies: the correlation of the first and second copy of every ring (5 samples each) → reliability of the 15-sample mean.

Usage:
    python -m exp16_wm.v_map_eval --heldout logs/teacher_data/v7/heldout --runs "<PRm run dir glob>" \
        --model a_ctrl2=logs/experiments_v2/v6c_ctrl2 --model b=logs/experiments_v2/v7_b_s0 ...
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIGMAS_M = (0.04, 0.08)
THRESHOLD = 4.0


def load_heldout(path: str) -> dict[str, np.ndarray]:
    """Concatenate the V7 files of a directory: obs, left, layout, xy, k_exact, ring_k, radii_mm."""
    parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(path, "*.npz")))]
    if not parts:
        raise FileNotFoundError(f"no V7 files in {path}")
    cat = lambda key: np.concatenate([p[key] for p in parts])
    return {"obs": cat("obs"), "left": cat("left"), "layout": cat("layout_id"), "xy": cat("cell_xy"), "k_exact": cat("k_exact").astype(float),
            "ring_k": cat("ring_k"), "radii_mm": parts[0]["radii_mm"]}


def reference_labels(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The 15-sample reference ("robust"), the two single-copy replicates (copy 0 and copy 1 of every ring: 5 samples each) and the exact k."""
    from exp16_wm.v_map_data import label_from_copies
    r, rk, kx = h["radii_mm"], h["ring_k"], h["k_exact"]
    allr = [float(x) for x in r]
    return {"ref": label_from_copies(rk, r, allr, [0, 1, 2], kx).astype(float), "copy0": label_from_copies(rk, r, allr, [0], kx).astype(float),
            "copy1": label_from_copies(rk, r, allr, [1], kx).astype(float), "exact": kx}


def smooth_scattered(xy: np.ndarray, v: np.ndarray, sigma_m: float) -> np.ndarray:
    """Gaussian-weighted average of v over the positions of one layout within 3σ of each position (itself included)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    out = np.empty(len(v))
    for i, nb in enumerate(tree.query_ball_point(xy, 3.0 * sigma_m)):
        d2 = np.sum((xy[nb] - xy[i]) ** 2, axis=1)
        w = np.exp(-d2 / (2 * sigma_m ** 2))
        out[i] = np.sum(w * v[nb]) / np.sum(w)
    return out


def trend_vectors(pred: np.ndarray, ref: np.ndarray, tau_ref: float = 1.0) -> dict[str, float]:
    """The Boltzmann comparison of two (already blurred) value vectors: p ∝ exp(ref / tau_ref), q ∝ exp(β · standardised pred) with β ≥ 0 fitted to
    minimise KL(p ‖ q); explained = 1 − KL(p ‖ q) / KL(p ‖ uniform)."""
    from exp16_wm.v_heatmap_radii import boltzmann, kl_div
    p = boltzmann(ref, 1.0 / tau_ref)
    kl_uni = kl_div(p, np.full(len(p), 1.0 / len(p)))
    std = pred.std()
    z = (pred - pred.mean()) / std if std > 0 else np.zeros_like(pred)
    betas = np.concatenate([[0.0], np.logspace(-2, 1.5, 80)])
    kl = min(kl_div(p, boltzmann(z, b)) for b in betas)
    return {"explained": float(1.0 - kl / kl_uni) if kl_uni > 0 else float("nan")}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def layout_metrics(pred: np.ndarray, ref: np.ndarray, exact: np.ndarray, xy: np.ndarray, sigmas_m: tuple[float, ...] = SIGMAS_M,
                   threshold: float = THRESHOLD) -> dict[str, float]:
    """All scores of one layout: corr (raw), corr_exact, then per blur σ (cm): corr_<σ>, explained_<σ>, and the zone AUC (raw)."""
    from sklearn.metrics import roc_auc_score
    out = {"corr": corr(pred, ref), "corr_exact": corr(pred, exact)}
    zone = ref >= threshold
    out["auc"] = float(roc_auc_score(zone, pred)) if 0 < zone.mean() < 1 else float("nan")
    for s in sigmas_m:
        ps, rs = smooth_scattered(xy, pred, s), smooth_scattered(xy, ref, s)
        tag = f"{s * 100:g}"
        out[f"corr_{tag}"] = corr(ps, rs)
        out[f"explained_{tag}"] = trend_vectors(ps, rs)["explained"]
    return out


def per_layout(h: dict[str, np.ndarray], pred: np.ndarray, ref: np.ndarray) -> dict[str, np.ndarray]:
    """layout_metrics for every layout: metric → (n_layouts,) in the order of np.unique(h["layout"])."""
    rows = [layout_metrics(pred[m], ref[m], h["k_exact"][m], h["xy"][m]) for m in (h["layout"] == l for l in np.unique(h["layout"]))]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def bootstrap_ci(x: np.ndarray, B: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 2.5%, 97.5%) of the mean over layouts (NaN layouts dropped), by resampling layouts."""
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(x: np.ndarray, y: np.ndarray, B: int = 4000, seed: int = 0) -> dict[str, float]:
    """x − y per layout (layouts where either is NaN dropped): the mean, its bootstrap 95% interval over layouts, and the share of layouts with x > y."""
    m = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[m]
    mean, lo, hi = bootstrap_ci(d, B, seed)
    return {"diff": mean, "lo": lo, "hi": hi, "share_better": float(np.mean(d > 0))}


def noise_limit(h: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, float]:
    """Reliability of the 15-sample reference from the correlation r of copy 0 and copy 1 (5 samples each, independent): the mean of three such
    components has reliability 3r / (1 + 2r); a noise-free map would correlate with the reference at most sqrt(reliability)."""
    rs = [corr(labels["copy0"][m], labels["copy1"][m]) for m in (h["layout"] == l for l in np.unique(h["layout"]))]
    r = float(np.nanmean(rs))
    rel = 3 * r / (1 + 2 * r)
    return {"copy_corr": r, "reliability15": rel, "max_corr_with_ref": float(np.sqrt(rel))}


def predict_model(run_dir: str, obs: np.ndarray, left: np.ndarray) -> np.ndarray:
    from exp16_wm import eval_by_balls as E
    from exp16_wm.paired_eval import find_run
    from exp16_wm.train_v import predict_balls
    return predict_balls(E.load_run(find_run(run_dir)), obs, left, 0)


def predict_made_max(runs: list[str], obs: np.ndarray) -> np.ndarray:
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_testset import made_max_values
    n = len(obs)
    t = {"next_obs": obs[:, None, :], "valid": np.ones((n, 1), bool), "succ": np.ones((n, 1), bool), "cleared": np.zeros((n, 1), bool)}
    return made_max_values(t, [find_run(g) for g in runs])[:, 0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", required=True); p.add_argument("--runs", nargs="+", default=[], help="made run globs for the made-max baseline")
    p.add_argument("--model", action="append", default=[], help="name=run_dir (repeatable)"); p.add_argument("--out", default="")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--diff", action="append", default=[], help="A-B (model names): paired per-layout difference of every score, A minus B (repeatable)")
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    h = load_heldout(args.heldout)
    lab = reference_labels(h)
    ids = np.unique(h["layout"])
    print(f"{len(ids)} held-out layouts, {len(h['obs']):,} positions | noise limit of the reference: {noise_limit(h, lab)}", flush=True)
    preds: dict[str, np.ndarray] = {}
    if args.runs:
        preds["made 최댓값"] = predict_made_max(args.runs, h["obs"])
    for spec in args.model:
        name, path = spec.split("=", 1)
        preds[name] = predict_model(path, h["obs"], h["left"])
    preds["(참고) 복사본 0번: 기준과 표본이 겹쳐 부풀려짐"] = lab["copy0"]
    res = {n: per_layout(h, pv, lab["ref"]) for n, pv in preds.items()}
    cols = ["corr", "corr_4", "corr_8", "explained_4", "explained_8", "auc", "corr_exact"]
    print(f"{'':30s} " + " ".join(f"{c:>20s}" for c in cols))
    for n, r in res.items():
        print(f"{n:30s} " + " ".join("{:6.3f} [{:5.3f},{:5.3f}]".format(*bootstrap_ci(r[c])) for c in cols), flush=True)
    for spec in args.diff:
        a, b = spec.split("-", 1) if spec.count("-") == 1 else spec.rsplit("-", 1)
        print(f"\n{a} − {b} (paired over layouts; share = layouts where {a} is higher)")
        for c in cols:
            d = paired_diff(res[a][c], res[b][c])
            print(f"  {c:12s} {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  better in {d['share_better'] * 100:.0f}% of layouts", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"layouts": ids.tolist(), "noise": noise_limit(h, lab), "metrics": {n: {c: r[c].tolist() for c in r} for n, r in res.items()}}, open(args.out, "w"))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
