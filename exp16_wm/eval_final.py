"""
exp16_wm/eval_final.py — standalone final eval for exp16 experiments.

Loads best_model.pt from an existing exp_dir and runs the final evaluation
(random baseline + 500 trained episodes), then updates results.json.

Usage:
    python -m exp16_wm.eval_final --exp-dir logs/experiments/exp16_vanilla_multi3_ms5_s1_2026-03-28@0909
    python -m exp16_wm.eval_final --exp-dir logs/experiments/exp16_vanilla_multi3_ms5_s2_2026-03-28@1105
"""

import argparse
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.sac import VanillaSAC, WMSAC

try:
    from simulator import TRAJ_MAX_EVENTS, TRAJ_EVENT_DIM
except ImportError:
    TRAJ_MAX_EVENTS = TRAJ_EVENT_DIM = None

ACTION_DIM = 2
ACT_LOW    = np.array([-np.pi, 0.5],  dtype=np.float32)
ACT_HIGH   = np.array([ np.pi, 8.0],  dtype=np.float32)


def get_obs_dim(n_balls: int) -> int:
    return 2 + n_balls * (2 if n_balls == 1 else 3) + 12


def build_agent_from_config(cfg: dict):
    obs_dim = get_obs_dim(cfg["n_balls"])
    if cfg["agent"] == "vanilla":
        return VanillaSAC(
            obs_dim    = obs_dim,
            action_dim = ACTION_DIM,
            act_low    = ACT_LOW,
            act_high   = ACT_HIGH,
            lr         = cfg["lr"],
            tau        = cfg["tau"],
            gamma      = cfg["gamma"],
            device     = cfg.get("device", "cpu"),
        )
    else:
        return WMSAC(
            obs_dim    = obs_dim,
            action_dim = ACTION_DIM,
            act_low    = ACT_LOW,
            act_high   = ACT_HIGH,
            max_events = TRAJ_MAX_EVENTS,
            event_dim  = TRAJ_EVENT_DIM,
            wm_coef    = cfg.get("wm_coef", 1.0),
            lr         = cfg["lr"],
            tau        = cfg["tau"],
            gamma      = cfg["gamma"],
            device     = cfg.get("device", "cpu"),
        )


def run_final_eval(exp_dir: str, no_wandb: bool = False):
    # ── load config ───────────────────────────────────────────────────────────
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    n_balls       = cfg["n_balls"]
    max_steps     = cfg["max_steps"]
    step_penalty  = cfg.get("step_penalty",  0.1)
    trunc_penalty = cfg.get("trunc_penalty", 1.0)
    seed          = cfg["seed"]

    print(f"\n{'─' * 60}")
    print(f"  Final eval: {os.path.basename(exp_dir)}")
    print(f"  agent={cfg['agent']}  seed={seed}  n_balls={n_balls}  max_steps={max_steps}")
    print(f"{'─' * 60}")

    env_kwargs = dict(
        n_balls       = n_balls,
        max_steps     = max_steps,
        step_penalty  = step_penalty,
        trunc_penalty = trunc_penalty,
    )

    # ── [1/2] random baseline ─────────────────────────────────────────────────
    print(f"  [1/2] Random baseline (500 episodes)...")
    rng_env = BilliardsEnv(**env_kwargs)
    rng_env.reset(seed=seed)
    random_pocketed = 0
    for _ in range(500):
        rng_env.reset()
        done = False
        while not done:
            _, _, term, trunc, info = rng_env.step(rng_env.action_space.sample())
            done = term or trunc
        if n_balls > 1:
            random_pocketed += info.get("total_pocketed", 0)
        else:
            random_pocketed += int(info.get("pocketed", False))
    rng_env.close()
    random_rate = random_pocketed / 500 / n_balls * 100
    print(f"      random pocket rate: {random_rate:.1f}%")

    # ── [2/2] trained best checkpoint ─────────────────────────────────────────
    print(f"  [2/2] Final eval — best model (500 episodes)...")
    best_model_path = os.path.join(exp_dir, "best_model", "best_model.pt")
    agent = build_agent_from_config(cfg)
    agent.load(best_model_path)

    eval_env = BilliardsEnv(**env_kwargs)
    eval_env.reset(seed=seed)

    total_pocketed, clears = 0, 0
    for _ in range(500):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(action)
            done = term or trunc
        if n_balls > 1:
            total_pocketed += info.get("total_pocketed", 0)
            clears += int(info.get("total_pocketed", 0) == n_balls)
        else:
            total_pocketed += int(info.get("pocketed", False))
            clears += int(info.get("pocketed", False))
    eval_env.close()

    trained_rate = total_pocketed / 500 / n_balls * 100
    clear_rate   = clears / 500 * 100
    improvement  = trained_rate - random_rate

    print(f"\n  {'─' * 58}")
    print(f"  pocket rate  : {trained_rate:.1f}%   (random: {random_rate:.1f}%   Δ: {improvement:+.1f}pp)")
    print(f"  clear  rate  : {clear_rate:.1f}%")

    # ── update results.json ───────────────────────────────────────────────────
    results_path = os.path.join(exp_dir, "results.json")
    try:
        with open(results_path) as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    results.update({
        "random_pocket_rate":  round(random_rate,  2),
        "trained_pocket_rate": round(trained_rate, 2),
        "clear_rate":          round(clear_rate,   2),
        "improvement_pp":      round(improvement,  2),
    })
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  results.json updated → {results_path}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    if not no_wandb:
        try:
            import wandb
            run_id = cfg.get("wandb_run_id")
            if run_id:
                wandb.init(id=run_id, resume="must", project="billiards-rl")
            else:
                wandb.init(
                    project = "billiards-rl",
                    name    = os.path.basename(exp_dir),
                    config  = cfg,
                    resume  = "allow",
                )
            wandb.log({
                "final/trained_pocket_rate": trained_rate,
                "final/clear_rate":          clear_rate,
                "final/random_pocket_rate":  random_rate,
                "final/improvement_pp":      improvement,
            })
            wandb.finish()
            print("  wandb logged.")
        except Exception as e:
            print(f"  wandb skipped: {e}")

    return trained_rate, clear_rate, random_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir",   required=True, help="Path to experiment directory")
    p.add_argument("--no-wandb",  action="store_true")
    args = p.parse_args()

    run_final_eval(args.exp_dir, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
