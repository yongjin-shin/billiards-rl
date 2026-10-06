"""
exp16_wm/compare_runs.py — compare finished Exp-16 runs across arms (seed-paired).

Reads each run's results.json (final 500-ep eval of the best checkpoint) and the
per-eval pocket rate from train.log, then prints:
  1. final pocket / clear / best_mean_reward per arm (mean ± std over seeds)
  2. eval pocket rate averaged over training-step windows
  3. paired t-test p (by seed) of every arm vs the reference arm, for both

Usage:
    python -m exp16_wm.compare_runs --ref vanilla
    python -m exp16_wm.compare_runs --ref rssm --arms rssm hmix
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats

WINDOWS = [(5e3, 50e3), (50e3, 100e3), (100e3, 200e3), (200e3, 300e3), (300e3, 500e3),
           (500e3, 1e6), (1e6, 1.5e6), (1.5e6, 2e6)]
_STEP = re.compile(r"\s*\[\s*([\d,]+) /")
_POCKET = re.compile(r"\s*pocket\s*:\s*([\d.]+)%")


def arm_of(results: dict, config: dict) -> str:
    """vanilla | geom | traj | rssm | hmix — same names as exp16_wm/run_grid.sh."""
    if results["agent"] in ("vanilla", "geom"):
        return results["agent"]
    if config.get("h_mix_start", 0.0) > 0:
        return "hmix"
    return config.get("wm_target", "traj")


def parse_eval_curve(log_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(env_steps, eval pocket %) from train.log lines; pocket lines before any step header are dropped."""
    steps, pocket, cur = [], [], None
    for line in log_lines:
        m = _STEP.match(line)
        if m:
            cur = int(m.group(1).replace(",", ""))
            continue
        m = _POCKET.match(line)
        if m and cur is not None:
            steps.append(cur)
            pocket.append(float(m.group(1)))
    return np.array(steps), np.array(pocket)


def window_means(steps: np.ndarray, pocket: np.ndarray,
                 windows: list[tuple[float, float]] = WINDOWS) -> list[float]:
    """Mean eval pocket % inside each (lo, hi] step window; nan if the window is empty."""
    out = []
    for lo, hi in windows:
        sel = (steps > lo) & (steps <= hi)
        out.append(float(pocket[sel].mean()) if sel.any() else float("nan"))
    return out


def load_runs(root: str) -> dict[str, dict[int, dict]]:
    """{arm: {seed: run}} — latest finished, best-ckpt-MA run per (arm, seed); aborted dirs skipped."""
    runs: dict[str, dict[int, dict]] = {}
    for d in sorted(glob.glob(os.path.join(root, "exp16_*_multi3_ms5_s*_*@*"))):
        if not os.path.exists(os.path.join(d, "results.json")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg.get("best_ckpt_window") is None:
            continue
        res = json.load(open(os.path.join(d, "results.json")))
        steps, pocket = parse_eval_curve(open(os.path.join(d, "train.log")).readlines())
        runs.setdefault(arm_of(res, cfg), {})[res["seed"]] = {
            "dir": d, "results": res, "windows": window_means(steps, pocket)}
    return runs


def paired_p(a: dict[int, float], b: dict[int, float]) -> float:
    """Paired t-test over seeds present in both; nan if fewer than 2 shared seeds."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return float("nan")
    return float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="logs/experiments")
    p.add_argument("--ref", default="vanilla")
    p.add_argument("--arms", nargs="+", default=None)
    args = p.parse_args()

    runs = load_runs(args.root)
    arms = args.arms or sorted(runs, key=lambda a: (a != args.ref, a))
    print("runs per arm:", {a: sorted(runs.get(a, {})) for a in arms})

    print("\n== final (best ckpt, 500 ep) — mean ± std over seeds, p vs", args.ref)
    for key in ("trained_pocket_rate", "clear_rate", "best_mean_reward"):
        print(f"-- {key}")
        ref = {s: r["results"][key] for s, r in runs.get(args.ref, {}).items()}
        for a in arms:
            v = {s: r["results"][key] for s, r in runs.get(a, {}).items()}
            if not v:
                continue
            p_str = "" if a == args.ref else f"  p={paired_p(v, ref):.3f}"
            print(f"  {a:8s} {np.mean(list(v.values())):7.3f} ± {np.std(list(v.values()), ddof=1) if len(v) > 1 else 0:.3f}"
                  f"  (n={len(v)}){p_str}")

    print("\n== eval pocket % by training window (mean over seeds)")
    print("  " + " ".join(f"{int(lo/1e3)}-{int(hi/1e3)}k".rjust(10) for lo, hi in WINDOWS))
    for a in arms:
        if a not in runs:
            continue
        w = np.array([r["windows"] for r in runs[a].values()])
        print(f"  {a:8s}" + " ".join(f"{x:10.1f}" for x in np.nanmean(w, axis=0)))
        if a != args.ref and args.ref in runs:
            ps = [paired_p({s: r["windows"][i] for s, r in runs[a].items()},
                           {s: r["windows"][i] for s, r in runs[args.ref].items()})
                  for i in range(len(WINDOWS))]
            print(f"  {'  p':8s}" + " ".join(f"{x:10.3f}" for x in ps))


if __name__ == "__main__":
    main()
