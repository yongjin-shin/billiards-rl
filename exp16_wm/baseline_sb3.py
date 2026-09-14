"""
exp16_wm/baseline_sb3.py — SB3 SAC baseline for Exp-16 comparison.

Hyperparameters matched to root train.py SAC preset:
  lr=3e-4, buffer_size=200k, batch_size=512, tau=0.005, gamma=0.99,
  train_freq=1, gradient_steps=1, learning_starts=5000, ent_coef='auto'

Usage:
    python -m exp16_wm.baseline_sb3 --seed 0
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from train import _tee_output


# ──────────────────────────────────────────────
# Eval + logging callback
# ──────────────────────────────────────────────

class EvalLogCallback(BaseCallback):
    """Evaluate every eval_freq steps; log to wandb + evaluations.npz."""

    def __init__(self, eval_env, exp_dir, eval_freq, n_eval_episodes,
                 no_wandb=False, verbose=1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.exp_dir         = exp_dir
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.no_wandb        = no_wandb

        self.best_pocket_rate = 0.0
        self.best_mean_reward = float("-inf")
        self._t_start         = None
        self._eval_timesteps: list = []
        self._eval_results:   list = []

    def _on_training_start(self):
        self._t_start = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        n_balls = self.eval_env.n_balls
        rewards, ep_lens, pocketed, clears = [], [], [], []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_r, ep_len = 0.0, 0
            ep_pocketed = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_r    += r
                ep_len  += 1
                if n_balls == 1:
                    ep_pocketed = int(info.get("pocketed", False))
                else:
                    ep_pocketed = info.get("total_pocketed", 0)
            rewards.append(ep_r)
            ep_lens.append(ep_len)
            pocketed.append(ep_pocketed / n_balls)
            clears.append(float(ep_pocketed == n_balls))

        mean_r      = float(np.mean(rewards))
        std_r       = float(np.std(rewards))
        pocket_rate = float(np.mean(pocketed))
        clear_rate  = float(np.mean(clears))
        elapsed     = time.time() - self._t_start
        fps         = self.num_timesteps / elapsed

        if mean_r      > self.best_mean_reward:  self.best_mean_reward  = mean_r
        if pocket_rate > self.best_pocket_rate:
            self.best_pocket_rate = pocket_rate
            self.model.save(os.path.join(self.exp_dir, "best_model", "best_model"))

        # SB3 internal train metrics (actor_loss, critic_loss, ent_coef, …)
        sb3_train = {k: float(v)
                     for k, v in self.model.logger.name_to_value.items()
                     if isinstance(v, (int, float))}

        log_dict = {
            "eval/mean_reward":      mean_r,
            "eval/std_reward":       std_r,
            "eval/pocket_rate":      pocket_rate * 100,
            "eval/clear_rate":       clear_rate  * 100,
            "eval/best_mean_reward": self.best_mean_reward,
            "eval/best_pocket_rate": self.best_pocket_rate * 100,
            "eval/ep_len_mean":      float(np.mean(ep_lens)),
            "train/fps":             fps,
        }
        log_dict.update(sb3_train)

        if self.verbose:
            print(
                f"[{self.num_timesteps:>8,}] "
                f"reward={mean_r:+.3f}±{std_r:.3f}  "
                f"pocket={pocket_rate*100:.1f}%  "
                f"fps={fps:.0f}"
            )

        if not self.no_wandb:
            wandb.log(log_dict, step=self.num_timesteps)

        # evaluations.npz (compare.py 호환)
        self._eval_timesteps.append(self.num_timesteps)
        self._eval_results.append(rewards)
        np.savez(
            os.path.join(self.exp_dir, "eval", "evaluations.npz"),
            timesteps = np.array(self._eval_timesteps),
            results   = np.array(self._eval_results),
        )

        return True


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--n-envs",         type=int,   default=10)
    p.add_argument("--total-steps",    type=int,   default=1_000_000)
    p.add_argument("--eval-freq",      type=int,   default=10_000)
    p.add_argument("--eval-episodes",   type=int,   default=50)
    p.add_argument("--learning-starts", type=int,   default=5_000)
    p.add_argument("--wandb-project",  type=str,   default="billiards-rl-exp16")
    p.add_argument("--no-wandb",       action="store_true")
    # env params
    p.add_argument("--n-balls",        type=int,   default=1, choices=[1, 3])
    p.add_argument("--max-steps",      type=int,   default=5)
    p.add_argument("--step-penalty",   type=float, default=0.01)
    p.add_argument("--trunc-penalty",  type=float, default=0.0)
    return p.parse_args()


def main(args):
    exp_dir = _make_exp_dir(args)
    with _tee_output(os.path.join(exp_dir, "train.log")):
        _main_inner(args, exp_dir)


def _make_exp_dir(args) -> str:
    ts      = time.strftime("%Y-%m-%d@%H%M")
    env_tag = f"_multi{args.n_balls}_ms{args.max_steps}" if args.n_balls > 1 else ""
    name    = f"exp16_sb3_sac{env_tag}_s{args.seed}_{ts}"
    path = os.path.join("logs", "experiments", name)
    os.makedirs(os.path.join(path, "eval"),       exist_ok=True)
    os.makedirs(os.path.join(path, "best_model"), exist_ok=True)
    return path


def _main_inner(args, exp_dir):
    # ── reproducibility ───────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── config ────────────────────────────────
    config = {
        "agent":           "sb3_sac",
        "seed":            args.seed,
        "n_envs":          args.n_envs,
        "total_steps":     args.total_steps,
        "buffer_size":     200_000,
        "batch_size":      512,
        "lr":              3e-4,
        "tau":             0.005,
        "gamma":           0.99,
        "ent_coef":        "auto",
        "learning_starts": args.learning_starts,
        "gradient_steps":  1,
        "eval_freq":       args.eval_freq,
        "eval_episodes":   args.eval_episodes,
        "n_balls":         args.n_balls,
        "max_steps":       args.max_steps,
        "step_penalty":    args.step_penalty,
        "trunc_penalty":   args.trunc_penalty,
        "exp_dir":         exp_dir,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── wandb ─────────────────────────────────
    run_name = os.path.basename(exp_dir)
    if not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = config,
            tags    = ["agent:sb3_sac", f"seed:{args.seed}"],
        )

    # ── environments ──────────────────────────
    env_kwargs = {
        "n_balls":       args.n_balls,
        "max_steps":     args.max_steps,
        "step_penalty":  args.step_penalty,
        "trunc_penalty": args.trunc_penalty,
    }
    vec_env = make_vec_env(
        BilliardsEnv,
        n_envs      = args.n_envs,
        vec_env_cls = SubprocVecEnv,
        env_kwargs  = env_kwargs,
        seed        = args.seed,
    )
    eval_env = BilliardsEnv(**env_kwargs)
    eval_env.reset(seed=args.seed)

    # ── model ─────────────────────────────────
    model = SAC(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = 3e-4,
        buffer_size     = 200_000,
        batch_size      = 512,
        tau             = 0.005,
        gamma           = 0.99,
        train_freq      = 1,
        gradient_steps  = 1,
        learning_starts = args.learning_starts,
        ent_coef        = "auto",
        policy_kwargs   = {"net_arch": [256, 256]},
        seed            = args.seed,
        verbose         = 0,
    )

    callback = EvalLogCallback(
        eval_env        = eval_env,
        exp_dir         = exp_dir,
        eval_freq       = args.eval_freq,
        n_eval_episodes = args.eval_episodes,
        no_wandb        = args.no_wandb,
    )

    print(f"Training sb3_sac — {args.total_steps:,} steps × {args.n_envs} envs  exp_dir={exp_dir}")
    model.learn(total_timesteps=args.total_steps, callback=callback)

    model.save(os.path.join(exp_dir, "final_model"))

    results = {
        "best_pocket_rate": callback.best_pocket_rate,
        "best_mean_reward": callback.best_mean_reward,
        "total_steps":      args.total_steps,
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if not args.no_wandb:
        wandb.finish()

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
