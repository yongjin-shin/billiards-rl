"""
logger.py — Structured experiment logger with Aim integration.

Provides:
  ExperimentLogger     — timestamped logging + Aim Run (metrics, exceptions)
  AimEvalCallback      — EvalCallback subclass that logs eval metrics to Aim
  TrainMetricsCallback — samples ep_info_buffer every N steps → logs to Aim
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

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

class AimEvalCallback(EvalCallback):
    """
    EvalCallback subclass that additionally logs eval metrics to ExperimentLogger.

    Logs after every evaluation:
      eval/mean_reward    — mean episodic reward over n_eval_episodes
      eval/best_mean_reward — running best
    """

    def __init__(self, exp_logger: ExperimentLogger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exp_logger = exp_logger

    def _on_step(self) -> bool:
        result = super()._on_step()
        # EvalCallback evaluates when n_calls % eval_freq == 0
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if hasattr(self, "last_mean_reward") and self.last_mean_reward is not None:
                self._exp_logger.log_metrics(
                    {
                        "eval/mean_reward":      float(self.last_mean_reward),
                        "eval/best_mean_reward": float(self.best_mean_reward),
                    },
                    step=self.num_timesteps,
                    context="eval",
                )
        return result


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
