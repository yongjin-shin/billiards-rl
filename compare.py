"""
compare.py — Compare RL experiments from logs/experiments/.

Reads config.json + results.json + eval/evaluations.npz from each experiment,
prints a summary table, and saves a learning-curve comparison plot.

Usage:
    python compare.py                          # all experiments in logs/experiments/
    python compare.py --out my_comparison.png
    python compare.py --filter SAC             # only experiments whose name contains SAC
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXP_ROOT = os.path.join("logs", "experiments")

# Colour palette — cycles if more than len(COLOURS) experiments
COLOURS = ["#4fc3f7", "#ff7043", "#66bb6a", "#ffd54f", "#ce93d8", "#80deea"]


# =============================================================================
# Loaders
# =============================================================================

def load_experiment(exp_dir: str) -> dict | None:
    """
    Load one experiment directory.
    Returns a dict with keys: name, config, results, timesteps, pocket_rates
    (or None if essential files are missing).
    """
    results_path = os.path.join(exp_dir, "results.json")
    config_path  = os.path.join(exp_dir, "config.json")
    eval_path    = os.path.join(exp_dir, "eval", "evaluations.npz")

    if not os.path.isfile(results_path):
        return None   # experiment not finished yet

    with open(results_path) as f:
        results = json.load(f)

    config = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Learning curve from EvalCallback
    timesteps    = None
    pocket_rates = None
    if os.path.isfile(eval_path):
        data    = np.load(eval_path)
        timesteps = data["timesteps"]                      # (n_evals,)
        rewards   = data["results"]                        # (n_evals, n_episodes)
        n_balls   = config.get("n_balls", 1)
        if n_balls == 1:
            # Binary reward: mean reward == pocket rate
            pocket_rates = rewards.mean(axis=1) * 100
        else:
            # Multi-ball: episode reward is sum of per-ball rewards + step penalties.
            # Normalize to [0,100] using n_balls as ceiling (max reward ≈ n_balls).
            # This gives an approximate "fraction of balls pocketed" curve.
            pocket_rates = np.clip(rewards.mean(axis=1) / n_balls * 100, 0, 100)

    return {
        "name"        : os.path.basename(exp_dir),
        "exp_dir"     : exp_dir,
        "config"      : config,
        "results"     : results,
        "timesteps"   : timesteps,
        "pocket_rates": pocket_rates,
    }


def list_experiments(root: str = EXP_ROOT, name_filter: str = "") -> list[dict]:
    """Load all finished experiments under `root`, optionally filtered by name substring."""
    if not os.path.isdir(root):
        return []

    experiments = []
    for name in sorted(os.listdir(root)):
        if name_filter and name_filter.upper() not in name.upper():
            continue
        exp_dir = os.path.join(root, name)
        if not os.path.isdir(exp_dir):
            continue
        exp = load_experiment(exp_dir)
        if exp is not None:
            experiments.append(exp)

    return experiments


# =============================================================================
# Display
# =============================================================================

def print_summary_table(experiments: list[dict]):
    """Pretty-print a comparison table to stdout."""
    if not experiments:
        print("  (no experiments found)")
        return

    header = f"  {'Name':<35}  {'Algo':>4}  {'Steps':>7}  {'Random':>7}  {'Trained':>7}  {'Δ':>6}  {'Time':>8}  {'FPS':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for exp in experiments:
        r = exp["results"]
        # Support both standard train.py results and transfer experiment results
        trained_rate = (r.get("trained_pocket_rate")
                        or r.get("finetuned_pocket%")
                        or r.get("trained_pocket%")
                        or 0.0)
        random_rate  = r.get("random_pocket_rate", 0.0)
        algo         = r.get("algo", r.get("strategy", "?"))
        print(
            f"  {exp['name']:<40}  "
            f"{algo:>12}  "
            f"{r.get('steps', 0)//1000:>5}k  "
            f"{random_rate:>6.1f}%  "
            f"{trained_rate:>6.1f}%  "
            f"{trained_rate - random_rate:>+5.1f}pp  "
            f"{r.get('training_time_sec', 0)/60:>6.1f}m  "
            f"{r.get('avg_fps', 0):>6.0f}"
        )


def plot_learning_curves(experiments: list[dict], out: str = "comparison.png"):
    """
    Plot pocket rate vs training steps for each experiment.
    Saves to `out`.
    """
    BG   = "#111111"
    GRID = "#333333"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    has_curves = False
    for i, exp in enumerate(experiments):
        colour = COLOURS[i % len(COLOURS)]
        label  = exp["name"]
        r      = exp["results"]

        # ── Learning curve (if evaluations.npz exists) ──────────────────────
        if exp["timesteps"] is not None and exp["pocket_rates"] is not None:
            ax.plot(
                exp["timesteps"] / 1_000,
                exp["pocket_rates"],
                color=colour, linewidth=2.0, label=label, alpha=0.9,
            )
            # Smoothed version
            if len(exp["pocket_rates"]) >= 5:
                from numpy.lib.stride_tricks import sliding_window_view
                k    = min(5, len(exp["pocket_rates"]))
                smooth = np.convolve(exp["pocket_rates"],
                                     np.ones(k) / k, mode="valid")
                t_s  = exp["timesteps"][k - 1:] / 1_000
                ax.plot(t_s, smooth, color=colour, linewidth=1.0, alpha=0.4)
            has_curves = True
        else:
            # No curve data — just a horizontal line at final result
            final = r.get("trained_pocket_rate", 0)
            steps = r.get("steps", 1_000_000)
            ax.hlines(final, 0, steps / 1_000,
                      colors=colour, linewidths=2.0, linestyles="--",
                      label=f"{label}  (final: {final:.1f}%)")
            has_curves = True

    ax.set_xlabel("Timesteps (k)", color="white", fontsize=12)
    ax.set_ylabel("Pocket rate (%)", color="white", fontsize=12)
    ax.set_title("SAC vs PPO — Pocket rate over training", color="white",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-2, 102)

    if has_curves:
        legend = ax.legend(facecolor="#222222", edgecolor=GRID,
                           labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Saved → {out}")
    return out


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare billiards RL experiments")
    parser.add_argument("--filter", default="",
                        help="Only include experiments whose name contains this string")
    parser.add_argument("--out", default="comparison.png",
                        help="Output plot filename (default: comparison.png)")
    parser.add_argument("--list", action="store_true",
                        help="Only list experiments, skip plotting")
    args = parser.parse_args()

    print(f"\nScanning {EXP_ROOT} ...")
    experiments = list_experiments(name_filter=args.filter)

    if not experiments:
        print(f"  No finished experiments found in {EXP_ROOT}")
        print(f"  Run:  python train.py --algo SAC")
        print(f"        python train.py --algo PPO")
        return

    print(f"\nFound {len(experiments)} experiment(s):\n")
    print_summary_table(experiments)

    if not args.list:
        plot_learning_curves(experiments, out=args.out)
        print(f"\nOpen {args.out} to view learning curves.")


if __name__ == "__main__":
    main()
