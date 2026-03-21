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
    from aim import Run as AimRun
    HAS_AIM = True
except ImportError:
    HAS_AIM = False

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
      Aim Run                 — real-time metrics + exception capture

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

    def __init__(self, exp_dir: str, run_name: str, config: dict,
                 aim_experiment: str = "billiards-rl",
                 use_aim: bool = True):
        self.exp_dir  = Path(exp_dir)
        self.run_name = run_name

        # ── Python logger ──────────────────────────────────────────────────────
        # Unique name per instance so multi-stage runs don't cross-contaminate.
        self._log = logging.getLogger(f"billiards.{run_name}.{id(self)}")
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False

        fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

        # Console / train.log handler (writes to sys.stderr at call time → caught by _Tee)
        _ch = logging.StreamHandler()   # default stream = sys.stderr
        _ch.setLevel(logging.INFO)
        _ch.setFormatter(fmt)

        # Dedicated structured log file (exp_dir/structured.log)
        _fh = logging.FileHandler(
            self.exp_dir / "structured.log", mode="a", encoding="utf-8"
        )
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(fmt)

        self._log.addHandler(_ch)
        self._log.addHandler(_fh)

        # ── Aim ────────────────────────────────────────────────────────────────
        self._aim: Optional[AimRun] = None
        if use_aim and HAS_AIM:
            try:
                self._aim = AimRun(
                    repo=".",
                    experiment=aim_experiment,
                    capture_terminal_logs=False,
                )
                self._aim["run_name"] = run_name
                self._aim["config"]   = config
                self._log.info(f"Aim run: {self._aim.hash}  (aim up → localhost:43800)")
            except Exception:
                self._log.warning(
                    "Aim init failed — continuing without Aim:\n" + traceback.format_exc()
                )

        self._log.info(f"=== Experiment started: {run_name} ===")
        self._log.debug(f"config: {json.dumps(config, indent=2, default=str)}")

    # ── Logging API ────────────────────────────────────────────────────────────

    def info(self, msg: str):    self._log.info(msg)
    def warning(self, msg: str): self._log.warning(msg)
    def error(self, msg: str):   self._log.error(msg)
    def debug(self, msg: str):   self._log.debug(msg)

    def log_metrics(self, metrics: dict, step: int, context: str = "eval") -> None:
        """Log a metrics dict to Aim and structured.log.

        Args:
            metrics: dict of {metric_name: float_value}
            step:    global training step
            context: 'train' or 'eval' (Aim context tag)
        """
        aim_ctx = {"subset": context}
        parts   = []
        for k, v in metrics.items():
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                continue
            parts.append(f"{k}={v_f:.4g}")
            if self._aim is not None:
                try:
                    self._aim.track(v_f, name=k, step=step, context=aim_ctx)
                except Exception:
                    pass
        if parts:
            self._log.info(f"[{context}] step={step:,}  " + "  ".join(parts))

    def log_exception(self, context: str = "") -> None:
        """Call from inside an except block. Logs exception type + full traceback."""
        tb  = traceback.format_exc()
        tag = f" [{context}]" if context else ""
        self._log.error(f"EXCEPTION{tag}:\n{tb}")
        if self._aim is not None:
            try:
                prev = self._aim.get("exceptions") or ""
                self._aim["exceptions"] = prev + f"\n--- {context} ---\n{tb}"
            except Exception:
                pass

    def finish(self, summary: Optional[dict] = None) -> None:
        """Finalize the logger. Logs summary metrics to Aim and closes the run."""
        if summary:
            self._log.info(
                f"=== FINAL RESULTS ===\n"
                f"{json.dumps(summary, indent=2, default=str)}"
            )
            if self._aim is not None:
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        try:
                            self._aim.set(k, v, strict=False)
                        except Exception:
                            pass
        if self._aim is not None:
            try:
                self._aim.close()
            except Exception:
                pass
        self._log.info(f"=== Experiment finished: {self.run_name} ===")
        # Remove all handlers to prevent log leaks across stages
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

    Aim에 기록되는 지표:
      eval/pocket_rate      — 포켓된 공 수 / (episodes × n_balls) [%]
      eval/clear_rate       — 모든 공 클리어한 에피소드 비율 [%]
      eval/mean_reward      — 평균 에피소드 reward
      eval/best_mean_reward — 지금까지 최고 mean_reward
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
        pocket_rate  = total_pocketed / (self._n_eval_episodes * self._n_balls) * 100
        clear_rate   = total_clears   /  self._n_eval_episodes * 100
        ep_len_mean  = float(np.mean(ep_lengths))

        # best model 저장
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            self.model.save(self._best_model_path)

        # Aim + structured.log
        self._exp_logger.log_metrics(
            {
                "eval/pocket_rate":      pocket_rate,
                "eval/clear_rate":       clear_rate,
                "eval/mean_reward":      mean_reward,
                "eval/best_mean_reward": self._best_mean_reward,
                "eval/ep_len_mean":      ep_len_mean,
            },
            step=self.num_timesteps,
            context="eval",
        )

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
            self._exp_logger.log_metrics(
                {
                    "train/ep_reward_mean": float(np.mean(rews)),
                    "train/ep_reward_std":  float(np.std(rews)),
                    "train/ep_len_mean":    float(np.mean(lens)),
                },
                step=self.num_timesteps,
                context="train",
            )
        self._last_log = self.num_timesteps
        return True
