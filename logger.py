"""
logger.py — Structured experiment logger with Aim integration.

Provides:
  ExperimentLogger       — timestamped logging + Aim Run (metrics, exceptions)
  BilliardsEvalCallback  — 직접 평가 loop: pocket_rate / clear_rate / ep_len → Aim
  TrainMetricsCallback   — ep_info_buffer 샘플링 → train 지표 → Aim
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

_LOG_FMT  = "%(asctime)s [%(levelname)-5s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# ExperimentLogger
# =============================================================================

class ExperimentLogger:
    """
    Central logger for one training run / curriculum stage.

    Output:
      exp_dir/structured.log  — timestamped, leveled log (this logger)
      exp_dir/train.log       — all stdout/stderr (via _Tee, unchanged)

    Usage:
        exp_log = ExperimentLogger(exp_dir, run_name="SAC_s42", config={...})
        exp_log.info("training started")
        exp_log.log_metrics({"eval/mean_reward": 3.5}, step=10_000, context="eval")
        try:
            model.learn(...)
        except Exception:
            exp_log.log_exception("model.learn")
            raise
        finally:
            exp_log.finish(summary=results_dict)
    """

    def __init__(self, exp_dir: str, run_name: str, config: dict):
        self.exp_dir  = Path(exp_dir)
        self.run_name = run_name

        # ── Python logger ──────────────────────────────────────────────────────
        self._log = logging.getLogger(f"billiards.{run_name}.{id(self)}")
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False

        fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

        _ch = logging.StreamHandler()
        _ch.setLevel(logging.INFO)
        _ch.setFormatter(fmt)

        _fh = logging.FileHandler(
            self.exp_dir / "structured.log", mode="a", encoding="utf-8"
        )
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(fmt)

        self._log.addHandler(_ch)
        self._log.addHandler(_fh)

        self._log.info(f"=== Experiment started: {run_name} ===")
        self._log.debug(f"config: {json.dumps(config, indent=2, default=str)}")

    # ── Logging API ────────────────────────────────────────────────────────────

    def info(self, msg: str):    self._log.info(msg)
    def warning(self, msg: str): self._log.warning(msg)
    def error(self, msg: str):   self._log.error(msg)
    def debug(self, msg: str):   self._log.debug(msg)

    def log_metrics(self, metrics: dict, step: int, context: str = "eval") -> None:
        parts = []
        for k, v in metrics.items():
            try:
                parts.append(f"{k}={float(v):.4g}")
            except (TypeError, ValueError):
                continue
        if parts:
            self._log.info(f"[{context}] step={step:,}  " + "  ".join(parts))

    def log_exception(self, context: str = "") -> None:
        tb  = traceback.format_exc()
        tag = f" [{context}]" if context else ""
        self._log.error(f"EXCEPTION{tag}:\n{tb}")

    def finish(self, summary: Optional[dict] = None) -> None:
        if summary:
            self._log.info(
                f"=== FINAL RESULTS ===\n"
                f"{json.dumps(summary, indent=2, default=str)}"
            )
        self._log.info(f"=== Experiment finished: {self.run_name} ===")
        for h in list(self._log.handlers):
            try:
                h.close()
            except Exception:
                pass
            self._log.removeHandler(h)


# =============================================================================
# Callbacks
# =============================================================================

class BilliardsEvalCallback(BaseCallback):
    """
    직접 평가 loop 기반 eval callback.
    EvalCallback과 달리 info dict에 직접 접근해 pocket_rate / clear_rate를 집계.

    wandb에 기록되는 지표:
      eval/pocket_rate      — 포켓된 공 수 / (episodes × n_balls) [%]
      eval/clear_rate       — 모든 공 클리어한 에피소드 비율 [%]
      eval/mean_reward      — 평균 에피소드 reward
      eval/std_reward       — std 에피소드 reward
      eval/best_mean_reward — 지금까지 최고 mean_reward
      eval/best_pocket_rate — 지금까지 최고 pocket_rate
      eval/ep_len_mean      — 평균 에피소드 길이

    best_model 저장 기준: mean_reward 개선 시.
    """

    def __init__(
        self,
        exp_logger: ExperimentLogger,
        eval_env,
        n_balls: int,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int = 1_000,
        n_eval_episodes: int = 50,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._exp_logger         = exp_logger
        self._eval_env           = eval_env
        self._n_balls            = n_balls
        self._best_model_path    = os.path.join(best_model_save_path, "best_model")
        self._log_path           = log_path
        self._eval_freq          = eval_freq
        self._n_eval_episodes    = n_eval_episodes
        self._deterministic      = deterministic
        self._best_mean_reward   = -np.inf
        self._best_pocket_rate   = 0.0

        os.makedirs(best_model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # evaluations.npz 누적 (compare.py 호환)
        self._eval_timesteps: list = []
        self._eval_results: list   = []   # shape: (n_evals, n_eval_episodes)

    def _on_step(self) -> bool:
        if self._eval_freq > 0 and self.n_calls % self._eval_freq != 0:
            return True

        ep_rewards, ep_lengths = [], []
        total_pocketed, total_clears = 0, 0

        obs, _ = self._eval_env.reset()
        for _ in range(self._n_eval_episodes):
            obs, _ = self._eval_env.reset()
            ep_reward, ep_len = 0.0, 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self._deterministic)
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                ep_reward += reward
                ep_len    += 1
                done = terminated or truncated
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            if self._n_balls == 1:
                total_pocketed += int(info.get("pocketed", False))
                total_clears   += int(info.get("pocketed", False))
            else:
                total_pocketed += info.get("total_pocketed", 0)
                total_clears   += int(info.get("total_pocketed", 0) == self._n_balls)

        mean_reward  = float(np.mean(ep_rewards))
        std_reward   = float(np.std(ep_rewards))
        pocket_rate  = total_pocketed / (self._n_eval_episodes * self._n_balls) * 100
        clear_rate   = total_clears   /  self._n_eval_episodes * 100
        ep_len_mean  = float(np.mean(ep_lengths))

        # best model 저장
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            self.model.save(self._best_model_path)
        if pocket_rate > self._best_pocket_rate:
            self._best_pocket_rate = pocket_rate

        metrics = {
            "eval/pocket_rate":      pocket_rate,
            "eval/clear_rate":       clear_rate,
            "eval/mean_reward":      mean_reward,
            "eval/std_reward":       std_reward,
            "eval/best_mean_reward": self._best_mean_reward,
            "eval/best_pocket_rate": self._best_pocket_rate,
            "eval/ep_len_mean":      ep_len_mean,
        }

        # structured.log
        self._exp_logger.log_metrics(metrics, step=self.num_timesteps, context="eval")

        # wandb
        if HAS_WANDB and wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)

        # evaluations.npz 누적 (compare.py 호환)
        self._eval_timesteps.append(self.num_timesteps)
        self._eval_results.append(ep_rewards)
        np.savez(
            os.path.join(self._log_path, "evaluations.npz"),
            timesteps=np.array(self._eval_timesteps),
            results=np.array(self._eval_results),
        )

        return True


class TrainMetricsCallback(BaseCallback):
    """
    Samples ep_info_buffer every log_freq steps and logs to ExperimentLogger.

    Logs:
      train/ep_reward_mean  — mean episodic reward over recent episodes
      train/ep_reward_std   — std
      train/ep_len_mean     — mean episode length
    """

    def __init__(self, exp_logger: ExperimentLogger, log_freq: int = 10_000):
        super().__init__()
        self._exp_logger = exp_logger
        self._log_freq   = log_freq
        self._last_log   = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self._log_freq:
            return True
        buf = self.model.ep_info_buffer
        if buf and len(buf) > 0:
            rews = [ep["r"] for ep in buf]
            lens = [ep["l"] for ep in buf]
            metrics = {
                "train/ep_reward_mean": float(np.mean(rews)),
                "train/ep_reward_std":  float(np.std(rews)),
                "train/ep_len_mean":    float(np.mean(lens)),
            }
            self._exp_logger.log_metrics(metrics, step=self.num_timesteps, context="train")
            if HAS_WANDB and wandb.run is not None:
                wandb.log(metrics, step=self.num_timesteps)
        self._last_log = self.num_timesteps
        return True
