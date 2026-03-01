"""
train_pretrained.py — Pretrained transfer experiments for billiards-rl.

Transfer a pretrained n_balls=1 SAC model to BilliardsEnv(n_balls=3)
using one of two strategies:

  A. obs-collapse  : Wrap n_balls=3 env → 16-dim obs (same as n_balls=1).
                     Load pretrained weights directly.  Zero-shot eval or
                     fine-tuning.  The wrapper always presents the nearest
                     unpocketed ball as "the ball", so the model applies
                     its learned aiming skill without seeing ball2/ball3.

  B. weight-copy   : Train fresh SAC on full 23-dim n_balls=3 obs, but
                     initialise network weights by copying from pretrained
                     n_balls=1 model (shared dims only; new ball dims → 0).
                     Warm-start: immediate reward signal from known aiming,
                     while freely adapting to multi-ball strategy.

Obs layout reference
---------------------
  n_balls=1 (16-dim): [cue_x, cue_y,
                        b1x, b1y,
                        p0x,p0y, ..., p5x,p5y]

  n_balls=3 (23-dim): [cue_x, cue_y,
                        b1x, b1y, b1_flag,
                        b2x, b2y, b2_flag,
                        b3x, b3y, b3_flag,
                        p0x,p0y, ..., p5x,p5y]

Weight copy mapping (first input layer, obs columns)
-----------------------------------------------------
  new[:, 0:4]   ← old[:, 0:4]    (cue + ball1 pos)
  new[:, 4:11]   = 0              (b1_flag, b2, b3 — new dims)
  new[:, 11:23] ← old[:, 4:16]   (pocket features)

Usage
-----
  # Zero-shot evaluation only (obs-collapse, no fine-tuning)
  python train_pretrained.py \\
      --strategy obs-collapse \\
      --pretrained logs/experiments/SAC_1000k_s42_.../best_model/best_model \\
      --eval-only

  # Fine-tune with obs-collapse (500k steps)
  python train_pretrained.py \\
      --strategy obs-collapse \\
      --pretrained <path> --steps 500000 --seed 42

  # Weight-copy warm-start (1M steps, full 23-dim obs)
  python train_pretrained.py \\
      --strategy weight-copy \\
      --pretrained <path> --steps 1000000 --seed 42
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from train import ETACallback, make_exp_dir, save_json, set_global_seed, ALGO_CONFIGS, N_ENVS, DEVICE
from simulator import BilliardsEnv


# =============================================================================
# Strategy A: Obs-collapse wrapper
# =============================================================================

class ObsCollapseWrapper(gym.ObservationWrapper):
    """
    Wraps BilliardsEnv(n_balls=3) to emit 16-dim obs (identical to n_balls=1).

    For each step the wrapper finds the nearest unpocketed ball and presents
    [cue_x, cue_y, nearest_x, nearest_y, pocket_features_12] to the policy.
    The pretrained n_balls=1 model can be loaded directly into this env.
    """

    def __init__(self, env: BilliardsEnv):
        super().__init__(env)
        assert isinstance(env, BilliardsEnv), "ObsCollapseWrapper requires BilliardsEnv"
        assert env.n_balls == 3, "ObsCollapseWrapper is designed for n_balls=3"

        # Override observation space to 16-dim (same as n_balls=1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Collapse 23-dim obs → 16-dim by selecting nearest unpocketed ball."""
        # obs layout: [cue_x(0), cue_y(1),
        #              b1x(2), b1y(3), b1_flag(4),
        #              b2x(5), b2y(6), b2_flag(7),
        #              b3x(8), b3y(9), b3_flag(10),
        #              p0x..p5y(11-22)]
        cue = obs[0:2]
        pockets = obs[11:23]

        # Collect unpocketed balls (flag == 0) and their positions
        candidates = []
        for start, flag_idx in [(2, 4), (5, 7), (8, 10)]:
            if obs[flag_idx] == 0.0:  # not pocketed
                candidates.append(obs[start:start + 2])

        if candidates:
            dists = [float(np.linalg.norm(b - cue)) for b in candidates]
            nearest = candidates[int(np.argmin(dists))]
        else:
            nearest = np.array([0.5, 0.5], dtype=np.float32)

        return np.concatenate([cue, nearest, pockets]).astype(np.float32)


def make_collapse_env(n_balls: int = 3, max_steps: int = 5,
                      step_penalty: float = 0.01, trunc_penalty: float = 0.0,
                      progressive_penalty: bool = False,
                      clear_bonus: float = 0.0):
    """Create an ObsCollapseWrapper env (single, not vectorised)."""
    base = BilliardsEnv(n_balls=n_balls, max_steps=max_steps,
                        step_penalty=step_penalty, trunc_penalty=trunc_penalty,
                        progressive_penalty=progressive_penalty,
                        clear_bonus=clear_bonus)
    return ObsCollapseWrapper(base)


# =============================================================================
# Strategy B: Weight copy helper
# =============================================================================

def copy_weights_to_23dim(src_model: SAC, dst_model: SAC) -> None:
    """
    Copy pretrained n_balls=1 SAC weights into a fresh n_balls=3 SAC model.

    Shared dims mapping (obs):
        new[:,  0: 4] ← old[:,  0: 4]   cue + ball1
        new[:,  4:11]  = 0              new dims (ball flags / ball2 / ball3)
        new[:, 11:23] ← old[:,  4:16]   pocket features

    Shared dims mapping (obs+action for critic):
        Same obs mapping, then action cols are appended (same 2-dim → copy as-is)
    """
    import torch

    src_sd = src_model.policy.state_dict()
    dst_sd = dst_model.policy.state_dict()

    # ── Actor first layer: weight shape [256, obs_dim] ────────────────────────
    # SB3 MlpPolicy actor structure: latent_pi.0 is first Linear
    for key in list(src_sd.keys()):
        if key not in dst_sd:
            continue

        src_w = src_sd[key]
        dst_w = dst_sd[key]

        if src_w.shape == dst_w.shape:
            # All hidden layers + output layers are the same shape → copy directly
            dst_sd[key] = src_w.clone()
        else:
            # First input layers differ in obs_dim columns
            # src shape: [256, 16] or [256, 18]  (16 obs or 16 obs + 2 action)
            # dst shape: [256, 23] or [256, 25]  (23 obs or 23 obs + 2 action)
            src_in = src_w.shape[1]   # 16 or 18
            dst_in = dst_w.shape[1]   # 23 or 25

            new_w = torch.zeros_like(dst_w)

            if src_in == 16 and dst_in == 23:
                # Actor first layer
                new_w[:, 0:4]   = src_w[:, 0:4]    # cue + ball1
                new_w[:, 4:11]  = 0                  # new ball dims
                new_w[:, 11:23] = src_w[:, 4:16]    # pockets
            elif src_in == 18 and dst_in == 25:
                # Critic first layer (obs + action cols)
                new_w[:, 0:4]   = src_w[:, 0:4]     # cue + ball1
                new_w[:, 4:11]  = 0                   # new ball dims
                new_w[:, 11:23] = src_w[:, 4:16]    # pockets
                new_w[:, 23:25] = src_w[:, 16:18]   # action dims
            else:
                print(f"  [WARN] Unexpected shape mismatch for {key}: "
                      f"{src_w.shape} → {dst_w.shape}. Skipping.")
                continue

            dst_sd[key] = new_w

            if "bias" in key:
                # Weight key was processed; bias same shape → handled above
                pass

    dst_model.policy.load_state_dict(dst_sd)
    print("  [weight-copy] Done. Shared obs/action dims copied; new dims zeroed.")


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_model(model, env_factory, n_eval: int = 200, n_balls: int = 3):
    """Run n_eval episodes and return (pocket_rate_pct, clear_rate_pct)."""
    env = env_factory()
    total_pocketed, clears = 0, 0
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        if n_balls == 1:
            total_pocketed += int(info.get("pocketed", False))
            clears += int(info.get("pocketed", False))
        else:
            total_pocketed += info.get("total_pocketed", 0)
            clears += int(info.get("total_pocketed", 0) == n_balls)
    env.close()
    return total_pocketed / n_eval / n_balls * 100, clears / n_eval * 100


# =============================================================================
# Main experiment runner
# =============================================================================

def run_transfer(
    strategy: str,
    pretrained_path: str,
    steps: int = 500_000,
    seed: int = 42,
    max_steps: int = 5,
    step_penalty: float = 0.01,
    trunc_penalty: float = 0.0,
    progressive_penalty: bool = False,
    clear_bonus: float = 0.0,
    eval_only: bool = False,
    n_eval: int = 500,
):
    assert strategy in ("obs-collapse", "weight-copy"), \
        f"Unknown strategy '{strategy}'"
    set_global_seed(seed)

    algo = "SAC"
    AlgoClass = SAC
    algo_cfg = ALGO_CONFIGS["SAC"]

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    strat   = strategy.replace("-", "_")
    rew_tag = f"_sp{step_penalty}_tp{trunc_penalty}" if (step_penalty != 0.01 or trunc_penalty != 0.0) else ""
    if progressive_penalty:
        rew_tag += "_pp"
    if clear_bonus > 0.0:
        rew_tag += f"_cb{clear_bonus}"
    name    = f"SAC_transfer_{strat}_ms{max_steps}{rew_tag}_s{seed}_{ts}"
    exp_dir = os.path.join("logs", "experiments", name)
    os.makedirs(os.path.join(exp_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "eval"),       exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "train"),      exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  billiards-rl  transfer — strategy={strategy}")
    print(f"  pretrained   : {pretrained_path}")
    print(f"  max_steps    : {max_steps}  |  seed {seed}")
    print(f"  step_penalty : {step_penalty}  |  trunc_penalty {trunc_penalty}  |  progressive {progressive_penalty}  |  clear_bonus {clear_bonus}")
    if not eval_only:
        print(f"  fine-tune    : {steps:,} steps")
    print(f"  exp_dir      : {exp_dir}")
    print(f"{'='*60}\n")

    # ── Load pretrained model ─────────────────────────────────────────────────
    pretrained = AlgoClass.load(pretrained_path)
    print(f"[1] Pretrained model loaded from {pretrained_path}")

    # ── Strategy A: obs-collapse ──────────────────────────────────────────────
    if strategy == "obs-collapse":

        def _env_factory():
            return ObsCollapseWrapper(BilliardsEnv(n_balls=3, max_steps=max_steps,
                                                   step_penalty=step_penalty, trunc_penalty=trunc_penalty,
                                                   progressive_penalty=progressive_penalty,
                                                   clear_bonus=clear_bonus))

        # Zero-shot eval (no training)
        print("[2] Zero-shot evaluation (obs-collapse, 23→16-dim)...")
        zs_pocket, zs_clear = evaluate_model(
            pretrained, _env_factory, n_eval=n_eval, n_balls=3)
        print(f"    Zero-shot pocket rate : {zs_pocket:.1f}%")
        print(f"    Zero-shot clear rate  : {zs_clear:.1f}%")

        if eval_only:
            save_json(os.path.join(exp_dir, "results.json"), {
                "strategy"            : strategy,
                "pretrained"          : pretrained_path,
                "max_steps"           : max_steps,
                "step_penalty"        : step_penalty,
                "trunc_penalty"       : trunc_penalty,
                "progressive_penalty" : progressive_penalty,
                "clear_bonus"         : clear_bonus,
                "seed"                : seed,
                "zeroshot_pocket%"    : round(zs_pocket, 2),
                "zeroshot_clear%"     : round(zs_clear, 2),
                "exp_dir"             : exp_dir,
            })
            print(f"\n  Saved → {exp_dir}")
            return exp_dir

        # Fine-tune
        print(f"\n[3] Fine-tuning pretrained model ({steps:,} steps, obs-collapse)...")
        vec_env = make_vec_env(
            lambda: ObsCollapseWrapper(BilliardsEnv(n_balls=3, max_steps=max_steps,
                                                    step_penalty=step_penalty, trunc_penalty=trunc_penalty,
                                                    progressive_penalty=progressive_penalty,
                                                    clear_bonus=clear_bonus)),
            n_envs      = N_ENVS,
            vec_env_cls = SubprocVecEnv,
            monitor_dir = os.path.join(exp_dir, "train"),
            seed        = seed,
        )
        _eval_env = Monitor(
            ObsCollapseWrapper(BilliardsEnv(n_balls=3, max_steps=max_steps,
                                            step_penalty=step_penalty, trunc_penalty=trunc_penalty,
                                            progressive_penalty=progressive_penalty,
                                            clear_bonus=clear_bonus)),
            filename=os.path.join(exp_dir, "eval", "monitor"),
        )

        # Set new env on the pretrained model
        pretrained.set_env(vec_env)

        eval_cb = EvalCallback(
            _eval_env,
            best_model_save_path = os.path.join(exp_dir, "best_model"),
            log_path             = os.path.join(exp_dir, "eval"),
            eval_freq            = 10_000 // N_ENVS,
            n_eval_episodes      = 50,
            deterministic        = True,
            verbose              = 0,
        )
        eta_cb = ETACallback(total_timesteps=steps, log_freq=10_000)

        t0 = time.time()
        pretrained.learn(
            total_timesteps = steps,
            callback        = CallbackList([eval_cb, eta_cb]),
            tb_log_name     = "SAC_transfer_collapse",
            reset_num_timesteps = False,   # continue timestep counter
        )
        elapsed = time.time() - t0
        pretrained.save(os.path.join(exp_dir, "sac_finetuned"))

        # Eval best checkpoint
        best = AlgoClass.load(os.path.join(exp_dir, "best_model", "best_model"))
        ft_pocket, ft_clear = evaluate_model(best, _env_factory, n_eval=n_eval, n_balls=3)

        save_json(os.path.join(exp_dir, "results.json"), {
            "strategy"            : strategy,
            "pretrained"          : pretrained_path,
            "steps"               : steps,
            "max_steps"           : max_steps,
            "step_penalty"        : step_penalty,
            "trunc_penalty"       : trunc_penalty,
            "progressive_penalty" : progressive_penalty,
            "clear_bonus"         : clear_bonus,
            "seed"                : seed,
            "zeroshot_pocket%"    : round(zs_pocket, 2),
            "zeroshot_clear%"     : round(zs_clear,  2),
            "finetuned_pocket%"   : round(ft_pocket,  2),
            "finetuned_clear%"    : round(ft_clear,   2),
            "training_time_sec"   : round(elapsed, 1),
            "avg_fps"             : round(steps / elapsed, 0),
            "exp_dir"             : exp_dir,
        })
        print(f"\n  Zero-shot pocket : {zs_pocket:.1f}%  →  Fine-tuned : {ft_pocket:.1f}%")
        print(f"  Clear rate       : {zs_clear:.1f}%  →  Fine-tuned : {ft_clear:.1f}%")
        print(f"  Training time    : {elapsed/60:.1f} min")
        print(f"  Saved → {exp_dir}")

    # ── Strategy B: weight-copy ───────────────────────────────────────────────
    elif strategy == "weight-copy":

        def _env_factory():
            return BilliardsEnv(n_balls=3, max_steps=max_steps, step_penalty=step_penalty,
                                trunc_penalty=trunc_penalty, progressive_penalty=progressive_penalty,
                                clear_bonus=clear_bonus)

        # Build fresh 23-dim SAC
        print("[2] Building fresh 23-dim SAC and copying pretrained weights...")
        vec_env = make_vec_env(
            BilliardsEnv,
            n_envs      = N_ENVS,
            env_kwargs  = {"n_balls": 3, "max_steps": max_steps,
                           "step_penalty": step_penalty, "trunc_penalty": trunc_penalty,
                           "progressive_penalty": progressive_penalty,
                           "clear_bonus": clear_bonus},
            vec_env_cls = SubprocVecEnv,
            monitor_dir = os.path.join(exp_dir, "train"),
            seed        = seed,
        )
        fresh_model = AlgoClass(
            "MlpPolicy",
            vec_env,
            device          = DEVICE,
            verbose         = 0,
            tensorboard_log = "logs/tensorboard",
            **algo_cfg,
        )
        copy_weights_to_23dim(pretrained, fresh_model)

        # Zero-shot eval on full 23-dim env
        print("[3] Zero-shot evaluation (weight-copy, full 23-dim obs)...")
        zs_pocket, zs_clear = evaluate_model(
            fresh_model, _env_factory, n_eval=n_eval, n_balls=3)
        print(f"    Zero-shot pocket rate : {zs_pocket:.1f}%")
        print(f"    Zero-shot clear rate  : {zs_clear:.1f}%")

        if eval_only:
            save_json(os.path.join(exp_dir, "results.json"), {
                "strategy"            : strategy,
                "pretrained"          : pretrained_path,
                "max_steps"           : max_steps,
                "step_penalty"        : step_penalty,
                "trunc_penalty"       : trunc_penalty,
                "progressive_penalty" : progressive_penalty,
                "clear_bonus"         : clear_bonus,
                "seed"                : seed,
                "zeroshot_pocket%"    : round(zs_pocket, 2),
                "zeroshot_clear%"     : round(zs_clear, 2),
                "exp_dir"             : exp_dir,
            })
            print(f"\n  Saved → {exp_dir}")
            return exp_dir

        # Warm-start training
        print(f"\n[4] Warm-start training ({steps:,} steps, full 23-dim obs)...")
        _eval_env = Monitor(
            BilliardsEnv(n_balls=3, max_steps=max_steps, step_penalty=step_penalty,
                         trunc_penalty=trunc_penalty, progressive_penalty=progressive_penalty,
                         clear_bonus=clear_bonus),
            filename=os.path.join(exp_dir, "eval", "monitor"),
        )
        eval_cb = EvalCallback(
            _eval_env,
            best_model_save_path = os.path.join(exp_dir, "best_model"),
            log_path             = os.path.join(exp_dir, "eval"),
            eval_freq            = 10_000 // N_ENVS,
            n_eval_episodes      = 50,
            deterministic        = True,
            verbose              = 0,
        )
        eta_cb = ETACallback(total_timesteps=steps, log_freq=10_000)

        t0 = time.time()
        fresh_model.learn(
            total_timesteps = steps,
            callback        = CallbackList([eval_cb, eta_cb]),
            tb_log_name     = "SAC_transfer_weightcopy",
        )
        elapsed = time.time() - t0
        fresh_model.save(os.path.join(exp_dir, "sac_warmstart"))

        # Eval best checkpoint
        best = AlgoClass.load(os.path.join(exp_dir, "best_model", "best_model"))
        ft_pocket, ft_clear = evaluate_model(best, _env_factory, n_eval=n_eval, n_balls=3)

        save_json(os.path.join(exp_dir, "results.json"), {
            "strategy"            : strategy,
            "pretrained"          : pretrained_path,
            "steps"               : steps,
            "max_steps"           : max_steps,
            "step_penalty"        : step_penalty,
            "trunc_penalty"       : trunc_penalty,
            "progressive_penalty" : progressive_penalty,
            "clear_bonus"         : clear_bonus,
            "seed"                : seed,
            "zeroshot_pocket%"    : round(zs_pocket, 2),
            "zeroshot_clear%"     : round(zs_clear,  2),
            "trained_pocket%"     : round(ft_pocket,  2),
            "trained_clear%"      : round(ft_clear,   2),
            "training_time_sec"   : round(elapsed, 1),
            "avg_fps"             : round(steps / elapsed, 0),
            "exp_dir"             : exp_dir,
        })
        print(f"\n  Zero-shot pocket : {zs_pocket:.1f}%  →  Trained : {ft_pocket:.1f}%")
        print(f"  Clear rate       : {zs_clear:.1f}%  →  Trained : {ft_clear:.1f}%")
        print(f"  Training time    : {elapsed/60:.1f} min")
        print(f"  Saved → {exp_dir}")

    return exp_dir


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transfer learning for billiards-rl")
    parser.add_argument(
        "--strategy", default="obs-collapse",
        choices=["obs-collapse", "weight-copy"],
        help="Transfer strategy (default: obs-collapse)",
    )
    parser.add_argument(
        "--pretrained", required=True,
        help="Path to pretrained n_balls=1 SAC model (without .zip extension)",
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Fine-tuning / warm-start steps (default: 500k)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5,
        help="Max episode steps for n_balls=3 env (default: 5)",
    )
    parser.add_argument(
        "--step-penalty", type=float, default=0.01,
        help="Reward penalty per step (default: 0.01)",
    )
    parser.add_argument(
        "--trunc-penalty", type=float, default=0.0,
        help="Extra penalty when episode truncated by step limit (default: 0.0)",
    )
    parser.add_argument(
        "--progressive-penalty", action="store_true",
        help="Use progressive step penalty: step i costs step_penalty × i (default: flat)",
    )
    parser.add_argument(
        "--clear-bonus", type=float, default=0.0,
        help="Bonus added at termination scaled by 1/steps_used — rewards faster clears (default: 0.0)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate zero-shot performance, no fine-tuning",
    )
    parser.add_argument(
        "--n-eval", type=int, default=500,
        help="Number of evaluation episodes (default: 500)",
    )
    args = parser.parse_args()

    run_transfer(
        strategy             = args.strategy,
        pretrained_path      = args.pretrained,
        steps                = args.steps,
        seed                 = args.seed,
        max_steps            = args.max_steps,
        step_penalty         = args.step_penalty,
        trunc_penalty        = args.trunc_penalty,
        progressive_penalty  = args.progressive_penalty,
        clear_bonus          = args.clear_bonus,
        eval_only            = args.eval_only,
        n_eval               = args.n_eval,
    )


if __name__ == "__main__":
    main()
