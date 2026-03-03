"""
train.py — Train SAC, PPO, or TQC on BilliardsEnv.

Each run creates a self-contained experiment directory:
  logs/experiments/{ALGO}_{steps}k_{timestamp}/
    ├── config.json        ← all hyperparams
    ├── results.json       ← final pocket rate, fps, training time
    ├── best_model/        ← best checkpoint (EvalCallback)
    ├── eval/              ← evaluations.npz (for compare.py)
    └── train/             ← per-worker monitor CSVs

Usage:
    python train.py                              # SAC, 1M steps, seed 42
    python train.py --algo PPO --seed 42         # PPO, same seed → fair comparison
    python train.py --algo TQC --seed 42         # TQC (requires: pip install sb3-contrib)
    python train.py --algo SAC --steps 500000 --seed 0

Fair comparison rule:
    All algorithms must use the same --seed so eval episodes see identical ball positions.
"""

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

try:
    from sb3_contrib import TQC
    HAS_TQC = True
except ImportError:
    HAS_TQC = False

from simulator import BilliardsEnv


# =============================================================================
# Algorithm hyper-parameter presets
# (network arch kept identical for fair comparison)
# =============================================================================

ALGO_CONFIGS = {
    "SAC": dict(
        learning_starts = 5_000,
        batch_size      = 512,
        buffer_size     = 200_000,
        policy_kwargs   = dict(net_arch=[256, 256]),
    ),
    "PPO": dict(
        n_steps       = 1024,   # rollout per env per update (×10 envs = 10,240 total)
        batch_size    = 512,
        n_epochs      = 10,
        clip_range    = 0.2,
        ent_coef      = 0.005,  # small entropy bonus prevents premature collapse
        gae_lambda    = 0.95,
        policy_kwargs = dict(net_arch=[256, 256]),
    ),
    # TQC: Truncated Quantile Critics (sb3-contrib)
    # Distributional Q-learning — drops top quantiles to reduce overestimation.
    # n_quantiles=25, top_quantiles_to_drop_per_net=2 → drops 4 of 50 total quantiles.
    # Otherwise identical setup to SAC for fair comparison.
    "TQC": dict(
        learning_starts              = 5_000,
        batch_size                   = 512,
        buffer_size                  = 200_000,
        top_quantiles_to_drop_per_net = 2,
        policy_kwargs                = dict(net_arch=[256, 256], n_quantiles=25),
    ),
}

# Map algo name → class (TQC registered only if sb3-contrib is installed)
def _build_algo_map():
    m = {"SAC": SAC, "PPO": PPO}
    if HAS_TQC:
        m["TQC"] = TQC
    return m

N_ENVS  = 10
DEVICE  = "cpu"   # cpu beats MPS on M-series for small MLPs


# =============================================================================
# File logging — tee stdout/stderr to exp_dir/train.log
# =============================================================================

class _Tee:
    """Write to both an existing stream and a log file simultaneously."""
    def __init__(self, stream, filepath):
        self._stream = stream
        self._file   = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):          # needed by subprocess / multiprocessing
        return self._stream.fileno()

    def isatty(self):
        return False

    def close(self):
        self._file.close()

    def __getattr__(self, name):   # forward anything else (e.g. .encoding)
        return getattr(self._stream, name)


@contextmanager
def _tee_output(filepath):
    """Context manager: redirect stdout+stderr to both console and filepath."""
    tee = _Tee(sys.stdout, filepath)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = tee
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        tee.close()


# =============================================================================
# ETA callback
# =============================================================================

class ETACallback(BaseCallback):
    """Prints timestep progress + estimated remaining time every log_freq steps."""

    def __init__(self, total_timesteps: int, log_freq: int = 10_000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_freq        = log_freq
        self._start          = None
        self._last_log       = 0

    def _on_training_start(self):
        self._start = time.time()

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t - self._last_log >= self.log_freq:
            elapsed  = time.time() - self._start
            frac     = t / self.total_timesteps
            eta      = (elapsed / frac - elapsed) if frac > 0 else 0
            eta_str  = time.strftime("%H:%M:%S", time.gmtime(eta))
            fps      = t / elapsed if elapsed > 0 else 0
            print(
                f"[ETA] step {t:>8,} / {self.total_timesteps:,} "
                f"({frac*100:.1f}%)  "
                f"elapsed {elapsed/60:.1f}m  ETA {eta_str}  "
                f"fps {fps:.0f}",
                flush=True,
            )
            self._last_log = t
        return True


# =============================================================================
# Experiment directory helpers
# =============================================================================

def set_global_seed(seed: int):
    """Seed numpy, torch, and Python random for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_exp_dir(algo: str, steps: int, seed: int, n_balls: int = 1,
                 max_steps: int = 5, step_penalty: float = 0.01,
                 trunc_penalty: float = 0.0,
                 progressive_penalty: bool = False,
                 clear_bonus: float = 0.0,
                 shots_taken: bool = False,
                 learning_rate: float = 3e-4,
                 gradient_steps: int = 1,
                 abs_angle: bool = False) -> str:
    """Create and return a unique experiment directory path."""
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = f"_multi{n_balls}_ms{max_steps}" if n_balls > 1 else ""
    rew_tag = f"_sp{step_penalty}_tp{trunc_penalty}" if (step_penalty != 0.01 or trunc_penalty != 0.0) else ""
    if progressive_penalty:
        rew_tag += "_pp"
    if clear_bonus > 0.0:
        rew_tag += f"_cb{clear_bonus}"
    if shots_taken:
        rew_tag += "_st"
    if learning_rate != 3e-4:
        rew_tag += f"_lr{learning_rate}"
    if gradient_steps != 1:
        rew_tag += f"_gs{gradient_steps}"
    if abs_angle:
        rew_tag += "_aa"
    name    = f"{algo}_{steps // 1000}k_s{seed}{env_tag}{rew_tag}_{ts}"
    path    = os.path.join("logs", "experiments", name)
    os.makedirs(os.path.join(path, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(path, "eval"),       exist_ok=True)
    os.makedirs(os.path.join(path, "train"),      exist_ok=True)
    return path


def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Training
# =============================================================================

def train(algo: str = "SAC", steps: int = 1_000_000, seed: int = 42,
          n_balls: int = 1, max_steps: int = 5,
          step_penalty: float = 0.01, trunc_penalty: float = 0.0,
          progressive_penalty: bool = False,
          clear_bonus: float = 0.0,
          shots_taken: bool = False,
          learning_rate: float = 3e-4,
          gradient_steps: int = 1,
          abs_angle: bool = False) -> str:
    """
    Train one algorithm for `steps` timesteps with a fixed seed.
    Returns the experiment directory path.

    n_balls=1            → single-shot env  (Phase 0, backward-compatible)
    n_balls=3            → multi-ball env   (Phase 1a)
    max_steps            → episode horizon for multi-ball env (ignored when n_balls=1)
    step_penalty         → base reward penalty per step (default 0.01)
    trunc_penalty        → extra penalty when episode truncated by step limit (default 0.0)
    progressive_penalty  → if True, step i costs step_penalty×i (later steps more expensive)
    clear_bonus          → +clear_bonus/steps_used added on termination (all balls cleared)
    shots_taken          → if True, append shots_taken/max_steps to obs (24-dim for n_balls=3)
    learning_rate        → critic/actor learning rate (default 3e-4; try 1e-4 for stability)
    gradient_steps       → gradient updates per env step (default 1; set to N_ENVS=10 for 1:1 ratio)
    """
    algo     = algo.upper()
    algo_map = _build_algo_map()
    if algo not in algo_map:
        raise ValueError(f"Unknown algo '{algo}'. Available: {list(algo_map.keys())}")
    if algo == "TQC" and not HAS_TQC:
        raise ImportError("TQC requires sb3-contrib: pip install sb3-contrib")

    set_global_seed(seed)

    exp_dir   = make_exp_dir(algo, steps, seed, n_balls, max_steps, step_penalty, trunc_penalty, progressive_penalty, clear_bonus, shots_taken, learning_rate, gradient_steps, abs_angle)

    with _tee_output(os.path.join(exp_dir, "train.log")):
        _train_inner(algo, steps, seed, n_balls, max_steps, step_penalty,
                     trunc_penalty, progressive_penalty, clear_bonus,
                     shots_taken, learning_rate, gradient_steps, abs_angle,
                     exp_dir)

    return exp_dir


def _train_inner(algo, steps, seed, n_balls, max_steps, step_penalty,
                 trunc_penalty, progressive_penalty, clear_bonus,
                 shots_taken, learning_rate, gradient_steps, abs_angle, exp_dir):
    AlgoClass = _build_algo_map()[algo]
    algo_cfg  = ALGO_CONFIGS[algo]

    env_label = f"multi{n_balls}(ms={max_steps})" if n_balls > 1 else "single"
    print(f"\n{'='*55}")
    print(f"  billiards-rl — {algo}  |  {steps:,} steps  |  seed {seed}  |  env {env_label}")
    print(f"  exp_dir : {exp_dir}")
    print(f"{'='*55}\n")

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "algo"       : algo,
        "steps"      : steps,
        "seed"       : seed,
        "n_balls"    : n_balls,
        "n_envs"     : N_ENVS,
        "device"     : DEVICE,
        "network"    : [256, 256],
        "algo_kwargs": {k: v for k, v in algo_cfg.items() if k != "policy_kwargs"},
        "timestamp"  : datetime.now().isoformat(timespec="seconds"),
        "max_steps"           : max_steps,
        "step_penalty"        : step_penalty,
        "trunc_penalty"       : trunc_penalty,
        "progressive_penalty" : progressive_penalty,
        "clear_bonus"         : clear_bonus,
        "shots_taken"         : shots_taken,
        "abs_angle"           : abs_angle,
        "learning_rate"       : learning_rate,
        "gradient_steps"      : gradient_steps,
        "env"                 : f"BilliardsEnv-n{n_balls}-ms{max_steps}",
        "exp_dir"    : exp_dir,
    }
    save_json(os.path.join(exp_dir, "config.json"), config)

    # ── Random baseline ───────────────────────────────────────────────────────
    print("[1/3] Random agent baseline (500 episodes)...")
    baseline_env = BilliardsEnv(n_balls=n_balls, max_steps=max_steps, step_penalty=step_penalty, trunc_penalty=trunc_penalty, progressive_penalty=progressive_penalty, clear_bonus=clear_bonus, shots_taken=shots_taken, abs_angle=abs_angle)
    baseline_env.reset(seed=seed)
    total_pocketed_baseline = 0
    for _ in range(500):
        baseline_env.reset()
        done = False
        while not done:
            _, _, term, trunc, info = baseline_env.step(baseline_env.action_space.sample())
            done = term or trunc
        if n_balls == 1:
            total_pocketed_baseline += int(info["pocketed"])
        else:
            total_pocketed_baseline += info["total_pocketed"]
    random_rate = total_pocketed_baseline / 500 / n_balls * 100
    print(f"      Random pocket rate: {random_rate:.1f}%  "
          f"(avg {total_pocketed_baseline/500:.2f}/{n_balls} balls)\n")

    # ── Vectorised training envs ──────────────────────────────────────────────
    # seed=seed gives worker i the seed (seed + i) → reproducible but diverse
    print(f"[2/3] Training {algo} — {steps:,} steps × {N_ENVS} envs ...")
    vec_env = make_vec_env(
        BilliardsEnv,
        n_envs      = N_ENVS,
        env_kwargs  = {"n_balls": n_balls, "max_steps": max_steps,
                       "step_penalty": step_penalty, "trunc_penalty": trunc_penalty,
                       "progressive_penalty": progressive_penalty,
                       "clear_bonus": clear_bonus, "shots_taken": shots_taken,
                       "abs_angle": abs_angle},
        vec_env_cls = SubprocVecEnv,
        monitor_dir = os.path.join(exp_dir, "train"),
        seed        = seed,
    )

    _eval_env = Monitor(BilliardsEnv(n_balls=n_balls, max_steps=max_steps, step_penalty=step_penalty, trunc_penalty=trunc_penalty, progressive_penalty=progressive_penalty, clear_bonus=clear_bonus, shots_taken=shots_taken, abs_angle=abs_angle),
                        filename=os.path.join(exp_dir, "eval", "monitor"))
    _eval_env.reset(seed=seed)

    eval_callback = EvalCallback(
        _eval_env,
        best_model_save_path = os.path.join(exp_dir, "best_model"),
        log_path             = os.path.join(exp_dir, "eval"),
        eval_freq            = 10_000 // N_ENVS,   # every 10k total steps
        n_eval_episodes      = 50,
        deterministic        = True,
        verbose              = 0,   # silent: ETACallback handles progress printing
    )
    eta_callback = ETACallback(total_timesteps=steps, log_freq=10_000)

    model_kwargs = dict(
        device          = DEVICE,
        verbose         = 0,        # silent: ETACallback handles progress printing
        tensorboard_log = "logs/tensorboard",
        learning_rate   = learning_rate,
    )
    if algo != "PPO":   # gradient_steps is off-policy only (SAC, TQC)
        model_kwargs["gradient_steps"] = gradient_steps

    model = AlgoClass("MlpPolicy", vec_env, **model_kwargs, **algo_cfg)

    # ── Descriptive TensorBoard run name (hierarchy: config/algo/seed/run) ──────
    # Tree: ms3_sp0.1_tp1.0 → SAC → s0 → 2026-03-03@0752
    # ※ macOS APFS에서 ':' 는 경로 구분자로 처리되므로 시·분 사이 구분자 생략
    cfg_parts = [f"ms{max_steps}", f"sp{step_penalty}"]
    if trunc_penalty > 0.0:     cfg_parts.append(f"tp{trunc_penalty}")
    if progressive_penalty:     cfg_parts.append("pp")
    if clear_bonus > 0.0:       cfg_parts.append(f"cb{clear_bonus}")
    if shots_taken:             cfg_parts.append("st")
    if learning_rate != 3e-4:   cfg_parts.append(f"lr{learning_rate}")
    if gradient_steps != 1:     cfg_parts.append(f"gs{gradient_steps}")
    if abs_angle:               cfg_parts.append("aa")
    _ts_str = time.strftime("%Y-%m-%d@%H%M")   # e.g. 2026-03-03@0752
    tb_log_name = f"{'_'.join(cfg_parts)}/{algo}/s{seed}/{_ts_str}"

    t0 = time.time()
    model.learn(
        total_timesteps = steps,
        callback        = CallbackList([eval_callback, eta_callback]),
        tb_log_name     = tb_log_name,
    )
    elapsed = time.time() - t0

    model.save(os.path.join(exp_dir, f"{algo.lower()}_final"))

    # ── Final evaluation using BEST checkpoint, not final weights ─────────────
    # SAC (and sometimes PPO) can collapse near the end of training.
    # EvalCallback saved the peak model → use that for the reported pocket rate.
    best_model_path = os.path.join(exp_dir, "best_model", "best_model")
    best_model = AlgoClass.load(best_model_path)
    print(f"\n[3/3] Evaluating best {algo} checkpoint (500 episodes)...")

    final_eval_env = BilliardsEnv(n_balls=n_balls, max_steps=max_steps, step_penalty=step_penalty, trunc_penalty=trunc_penalty, progressive_penalty=progressive_penalty, clear_bonus=clear_bonus, shots_taken=shots_taken, abs_angle=abs_angle)
    final_eval_env.reset(seed=seed)
    n_eval = 500
    total_pocketed_eval, clears = 0, 0
    for _ in range(n_eval):
        obs, _ = final_eval_env.reset()
        done = False
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = final_eval_env.step(action)
            done = term or trunc
        if n_balls == 1:
            total_pocketed_eval += int(info["pocketed"])
            clears += int(info["pocketed"])
        else:
            total_pocketed_eval += info["total_pocketed"]
            clears += int(info["total_pocketed"] == n_balls)

    trained_rate = total_pocketed_eval / n_eval / n_balls * 100
    clear_rate   = clears / n_eval * 100
    avg_fps      = steps / elapsed

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "algo"               : algo,
        "n_balls"            : n_balls,
        "max_steps"          : max_steps,
        "step_penalty"        : step_penalty,
        "trunc_penalty"       : trunc_penalty,
        "progressive_penalty" : progressive_penalty,
        "clear_bonus"         : clear_bonus,
        "shots_taken"         : shots_taken,
        "learning_rate"       : learning_rate,
        "gradient_steps"      : gradient_steps,
        "steps"               : steps,
        "seed"               : seed,
        "random_pocket_rate" : round(random_rate,  2),
        "trained_pocket_rate": round(trained_rate, 2),
        "clear_rate"         : round(clear_rate,   2),
        "improvement_pp"     : round(trained_rate - random_rate, 2),
        "training_time_sec"  : round(elapsed, 1),
        "avg_fps"            : round(avg_fps, 0),
        "exp_dir"            : exp_dir,
    }
    save_json(os.path.join(exp_dir, "results.json"), results)

    print(f"\n  {'─'*45}")
    print(f"  {algo} pocket rate  : {trained_rate:.1f}%")
    if n_balls > 1:
        print(f"  Clear rate (all {n_balls}): {clear_rate:.1f}%")
    print(f"  Random            : {random_rate:.1f}%")
    print(f"  Improvement       : {trained_rate - random_rate:+.1f}pp")
    print(f"  Training time     : {elapsed/60:.1f} min  ({avg_fps:.0f} fps)")
    print(f"  Saved → {exp_dir}")
    print(f"  TensorBoard → tensorboard --logdir logs/tensorboard")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train billiards RL agent")
    _choices = ["SAC", "PPO", "TQC"] if HAS_TQC else ["SAC", "PPO"]
    parser.add_argument("--algo",   default="SAC", choices=_choices,
                        help="Algorithm (default: SAC)")
    parser.add_argument("--steps",  type=int, default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n-balls", type=int, default=1, choices=[1, 3],
                        help="Number of target balls: 1=single-shot (default), 3=multi-ball Phase 1a")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max steps per episode for multi-ball env (default: 5, ignored for n-balls=1)")
    parser.add_argument("--step-penalty", type=float, default=0.01,
                        help="Reward penalty per step (default: 0.01)")
    parser.add_argument("--trunc-penalty", type=float, default=0.0,
                        help="Extra reward penalty when episode truncated by step limit (default: 0.0)")
    parser.add_argument("--progressive-penalty", action="store_true",
                        help="Use progressive step penalty: step i costs step_penalty × i (default: flat)")
    parser.add_argument("--clear-bonus", type=float, default=0.0,
                        help="Bonus added at termination scaled by 1/steps_used — rewards faster clears (default: 0.0)")
    parser.add_argument("--shots-taken", action="store_true",
                        help="Append shots_taken/max_steps to obs (24-dim for n_balls=3) — urgency ablation")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Critic/actor learning rate (default: 3e-4; try 1e-4 for stability)")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient updates per env step (default: 1; set to N_ENVS=10 for 1:1 ratio)")
    parser.add_argument("--abs-angle", action="store_true",
                        help="Use absolute cue angle [0, 2π] instead of delta from nearest ball (Exp-12)")
    args = parser.parse_args()
    train(args.algo, args.steps, args.seed, args.n_balls, args.max_steps,
          args.step_penalty, args.trunc_penalty, args.progressive_penalty,
          args.clear_bonus, args.shots_taken, args.learning_rate, args.gradient_steps,
          args.abs_angle)


if __name__ == "__main__":
    main()
