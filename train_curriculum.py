"""
train_curriculum.py — SAC curriculum training: ms=5 → ms=4 → ms=3.

각 stage에서 이전 stage의 best_model 가중치를 불러와 더 어려운 환경에서 fine-tune.
obs 차원은 23-dim으로 동일(shots_taken=False)하므로 weight 재사용 가능.

Stage 1  ms=5  steps1 steps  → best_model 저장
Stage 2  ms=4  steps2 steps  → Stage 1 best 로드 후 fine-tune → best_model 저장
Stage 3  ms=3  steps3 steps  → Stage 2 best 로드 후 fine-tune → best_model 저장

총 steps = steps1 + steps2 + steps3 (기본 2M)

Usage:
    source .venv/bin/activate
    python train_curriculum.py                              # default: 1M+500k+500k, seed=42
    python train_curriculum.py --steps1 1000000 --steps2 500000 --steps3 500000 --seed 0
    python train_curriculum.py --steps1 500000 --steps2 500000 --steps3 1000000 --seed 42
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from simulator import BilliardsEnv
from train import ETACallback, set_global_seed, save_json, ALGO_CONFIGS, N_ENVS, DEVICE, _tee_output


# =============================================================================
# Helpers
# =============================================================================

def make_stage_dir(parent_dir: str, stage: int, max_steps: int, steps: int) -> str:
    path = os.path.join(parent_dir, f"stage{stage}_ms{max_steps}_{steps//1000}k")
    os.makedirs(os.path.join(path, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(path, "eval"),       exist_ok=True)
    os.makedirs(os.path.join(path, "train"),      exist_ok=True)
    return path


def evaluate_model(model: SAC, max_steps: int, seed: int,
                   step_penalty: float = 0.1, trunc_penalty: float = 1.0,
                   n_eval: int = 500) -> dict:
    """500-episode deterministic evaluation. Returns pocket/clear/ep_len stats."""
    eval_env = BilliardsEnv(n_balls=3, max_steps=max_steps,
                            step_penalty=step_penalty, trunc_penalty=trunc_penalty)
    eval_env.reset(seed=seed)
    total_pocketed, clears, total_steps = 0, 0, 0
    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(action)
            done = term or trunc
            steps += 1
        total_pocketed += info["total_pocketed"]
        clears += int(info["total_pocketed"] == 3)
        total_steps += steps
    eval_env.close()
    return {
        "pocket_rate": round(total_pocketed / n_eval / 3 * 100, 2),
        "clear_rate":  round(clears / n_eval * 100, 2),
        "ep_len_mean": round(total_steps / n_eval, 3),
    }


# =============================================================================
# Single-stage training (load or create SAC, train, return best_model path)
# =============================================================================

def train_stage(stage: int, max_steps: int, steps: int, seed: int,
                parent_dir: str,
                step_penalty: float = 0.1,
                trunc_penalty: float = 1.0,
                pretrained_path: str | None = None) -> str:
    """
    Train one curriculum stage.
    pretrained_path: if given, load weights from this SAC .zip before training.
    Returns path to best_model .zip (without extension).
    """
    stage_dir = make_stage_dir(parent_dir, stage, max_steps, steps)
    env_label = f"ms={max_steps}, {steps//1000}k steps"
    print(f"\n{'='*55}")
    print(f"  STAGE {stage}  |  {env_label}  |  seed {seed}")
    if pretrained_path:
        print(f"  Loading weights from: {pretrained_path}")
    print(f"  stage_dir: {stage_dir}")
    print(f"{'='*55}\n")

    set_global_seed(seed)
    algo_cfg = ALGO_CONFIGS["SAC"]

    env_kwargs = dict(n_balls=3, max_steps=max_steps,
                      step_penalty=step_penalty, trunc_penalty=trunc_penalty)

    vec_env = make_vec_env(
        BilliardsEnv,
        n_envs      = N_ENVS,
        env_kwargs  = env_kwargs,
        vec_env_cls = SubprocVecEnv,
        monitor_dir = os.path.join(stage_dir, "train"),
        seed        = seed,
    )

    _eval_env = Monitor(
        BilliardsEnv(**env_kwargs),
        filename=os.path.join(stage_dir, "eval", "monitor"),
    )
    _eval_env.reset(seed=seed)

    eval_callback = EvalCallback(
        _eval_env,
        best_model_save_path = os.path.join(stage_dir, "best_model"),
        log_path             = os.path.join(stage_dir, "eval"),
        eval_freq            = 10_000 // N_ENVS,
        n_eval_episodes      = 50,
        deterministic        = True,
        verbose              = 0,
    )
    eta_callback = ETACallback(total_timesteps=steps, log_freq=10_000)

    # TensorBoard: curriculum_sp0.1_tp1.0/SAC/s42/stage1_2026-03-03@0900
    _ts = time.strftime("%Y-%m-%d@%H%M")
    tb_log_name = f"curriculum_sp{step_penalty}_tp{trunc_penalty}/SAC/s{seed}/stage{stage}_{_ts}"

    if pretrained_path:
        # Load weights into a new model with the new environment
        # Replay buffer is intentionally reset — old ms transitions are off-distribution
        model = SAC.load(
            pretrained_path,
            env             = vec_env,
            device          = DEVICE,
            verbose         = 0,
            tensorboard_log = "logs/tensorboard",
        )
        # Override learning_rate/gradient_steps to defaults (loaded model keeps its params)
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            device          = DEVICE,
            verbose         = 0,
            tensorboard_log = "logs/tensorboard",
            **algo_cfg,
        )

    t0 = time.time()
    model.learn(
        total_timesteps   = steps,
        callback          = CallbackList([eval_callback, eta_callback]),
        tb_log_name       = tb_log_name,
        reset_num_timesteps = True,   # each stage starts at step 0 in TensorBoard
    )
    elapsed = time.time() - t0

    vec_env.close()

    best_model_path = os.path.join(stage_dir, "best_model", "best_model")
    print(f"\n  Stage {stage} done — {elapsed/60:.1f} min")

    # Evaluate best model
    print(f"  Evaluating stage {stage} best model (500 eps, ms={max_steps})...")
    best_model = SAC.load(best_model_path)
    metrics = evaluate_model(best_model, max_steps=max_steps, seed=seed,
                             step_penalty=step_penalty, trunc_penalty=trunc_penalty)
    print(f"  pocket={metrics['pocket_rate']:.1f}%  "
          f"clear={metrics['clear_rate']:.1f}%  "
          f"ep_len={metrics['ep_len_mean']:.2f}")

    save_json(os.path.join(stage_dir, "results.json"), {
        "stage"        : stage,
        "max_steps"    : max_steps,
        "steps"        : steps,
        "seed"         : seed,
        "step_penalty" : step_penalty,
        "trunc_penalty": trunc_penalty,
        "pretrained"   : pretrained_path,
        "training_time_sec": round(elapsed, 1),
        **metrics,
    })
    return best_model_path


# =============================================================================
# Curriculum runner
# =============================================================================

def run_curriculum(steps1: int = 1_000_000,
                   steps2: int = 500_000,
                   steps3: int = 500_000,
                   seed: int = 42,
                   step_penalty: float = 0.1,
                   trunc_penalty: float = 1.0) -> str:
    """
    Run full curriculum: ms=5 → ms=4 → ms=3.
    Returns the experiment directory path.
    """
    total_steps = steps1 + steps2 + steps3
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        "logs", "experiments",
        f"SAC_curriculum_ms5-4-3_{total_steps//1000}k_s{seed}_{ts}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "train.log")

    def _run():
        print(f"\n{'#'*55}")
        print(f"  CURRICULUM  ms=5 → ms=4 → ms=3")
        print(f"  total steps : {total_steps:,}  ({steps1//1000}k + {steps2//1000}k + {steps3//1000}k)")
        print(f"  seed        : {seed}")
        print(f"  exp_dir     : {exp_dir}")
        print(f"{'#'*55}")

        t_total = time.time()

        # Stage 1 — ms=5, train from scratch
        s1_best = train_stage(
            stage=1, max_steps=5, steps=steps1, seed=seed,
            parent_dir=exp_dir,
            step_penalty=step_penalty, trunc_penalty=trunc_penalty,
            pretrained_path=None,
        )

        # Stage 2 — ms=4, warm-start from Stage 1
        s2_best = train_stage(
            stage=2, max_steps=4, steps=steps2, seed=seed,
            parent_dir=exp_dir,
            step_penalty=step_penalty, trunc_penalty=trunc_penalty,
            pretrained_path=s1_best,
        )

        # Stage 3 — ms=3, warm-start from Stage 2
        s3_best = train_stage(
            stage=3, max_steps=3, steps=steps3, seed=seed,
            parent_dir=exp_dir,
            step_penalty=step_penalty, trunc_penalty=trunc_penalty,
            pretrained_path=s2_best,
        )

        elapsed_total = time.time() - t_total

        # Cross-stage evaluation: test Stage 3 model on ms=3 (primary metric)
        print(f"\n{'='*55}")
        print(f"  CURRICULUM COMPLETE — {elapsed_total/60:.1f} min total")
        print(f"  Loading Stage 3 best model for final evaluation (ms=3, 500 eps)...")
        final_model = SAC.load(s3_best)
        final_metrics = evaluate_model(final_model, max_steps=3, seed=seed,
                                       step_penalty=step_penalty,
                                       trunc_penalty=trunc_penalty)
        print(f"  Final (ms=3) pocket={final_metrics['pocket_rate']:.1f}%  "
              f"clear={final_metrics['clear_rate']:.1f}%  "
              f"ep_len={final_metrics['ep_len_mean']:.2f}")
        print(f"  Saved → {exp_dir}")
        print(f"{'='*55}\n")

        save_json(os.path.join(exp_dir, "results.json"), {
            "algo"           : "SAC",
            "method"         : "curriculum_ms5-4-3",
            "steps1"         : steps1,
            "steps2"         : steps2,
            "steps3"         : steps3,
            "total_steps"    : total_steps,
            "seed"           : seed,
            "step_penalty"   : step_penalty,
            "trunc_penalty"  : trunc_penalty,
            "final_ms"       : 3,
            "training_time_sec": round(elapsed_total, 1),
            **{f"final_{k}": v for k, v in final_metrics.items()},
        })

    with _tee_output(log_file):
        _run()

    return exp_dir


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAC curriculum training: ms=5 → ms=4 → ms=3"
    )
    parser.add_argument("--steps1", type=int, default=1_000_000,
                        help="Stage 1 (ms=5) timesteps (default: 1M)")
    parser.add_argument("--steps2", type=int, default=500_000,
                        help="Stage 2 (ms=4) timesteps (default: 500k)")
    parser.add_argument("--steps3", type=int, default=500_000,
                        help="Stage 3 (ms=3) timesteps (default: 500k)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--step-penalty",  type=float, default=0.1,
                        help="Flat step penalty (default: 0.1)")
    parser.add_argument("--trunc-penalty", type=float, default=1.0,
                        help="Truncation penalty (default: 1.0)")
    args = parser.parse_args()
    run_curriculum(
        steps1=args.steps1,
        steps2=args.steps2,
        steps3=args.steps3,
        seed=args.seed,
        step_penalty=args.step_penalty,
        trunc_penalty=args.trunc_penalty,
    )


if __name__ == "__main__":
    main()
