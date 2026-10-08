"""
tests/test_exp16_wm_train.py

Unit tests for exp16_wm/train.py::update_best_ma — the moving-average
best-checkpoint selection helper — plus the --best-ckpt-metric helpers and the
final-eval speed summary (clear-speed task, 2026-10-08).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest

from types import SimpleNamespace

from exp16_wm.train import (update_best_ma, initial_best_ckpt, ckpt_value,
                             summarize_final_eval, target_entropy_at, make_exp_dir)


class TestUpdateBestMa:
    def test_window_fills_up_before_capping(self):
        history: list = []
        best = 0.0
        for v in [0.1, 0.2, 0.3]:
            history, ma, best, _ = update_best_ma(history, v, window=5, best_ma_so_far=best)
        assert history == [0.1, 0.2, 0.3]
        assert ma == (0.1 + 0.2 + 0.3) / 3

    def test_window_caps_at_max_length(self):
        history: list = []
        best = 0.0
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            history, ma, best, _ = update_best_ma(history, v, window=3, best_ma_so_far=best)
        assert history == [0.4, 0.5, 0.6]
        assert ma == (0.4 + 0.5 + 0.6) / 3

    def test_smooths_a_single_noisy_spike(self):
        # a lone high value shouldn't immediately become "best" if the rest
        # of the window is low — this is the whole point of the moving avg.
        history, ma, best, is_best = update_best_ma([0.5, 0.5, 0.5, 0.5], 0.9, window=5, best_ma_so_far=0.5)
        assert is_best  # ma did rise (0.58 > 0.5) ...
        assert ma < 0.9  # ...but far less than the raw spike itself

    def test_monotonic_increase_always_new_best(self):
        history: list = []
        best = 0.0
        results = []
        for v in [0.1, 0.2, 0.3, 0.4]:
            history, ma, best, is_best = update_best_ma(history, v, window=5, best_ma_so_far=best)
            results.append(is_best)
        assert all(results)

    def test_decrease_is_not_new_best(self):
        history, ma, best, is_best = update_best_ma([0.1, 0.2, 0.3, 0.4, 0.5], 0.0, window=5, best_ma_so_far=0.5)
        assert not is_best
        assert best == 0.5  # unchanged

    def test_empty_history_edge_case(self):
        history, ma, best, is_best = update_best_ma([], 0.4, window=5, best_ma_so_far=0.0)
        assert history == [0.4]
        assert ma == 0.4
        assert is_best

    def test_window_one_behaves_like_raw_value(self):
        history, ma, best, is_best = update_best_ma([0.7], 0.3, window=1, best_ma_so_far=0.7)
        assert history == [0.3]
        assert ma == 0.3
        assert not is_best


class TestBestCkptMetric:
    def test_pocket_rate_keeps_original_start(self):
        assert initial_best_ckpt("pocket_rate") == 0.0

    def test_mean_reward_starts_at_minus_inf(self):
        assert initial_best_ckpt("mean_reward") == float("-inf")

    def test_value_picks_requested_metric(self):
        assert ckpt_value("pocket_rate", 0.9, -3.0) == 0.9
        assert ckpt_value("mean_reward", 0.9, -3.0) == -3.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            initial_best_ckpt("clear_rate")
        with pytest.raises(ValueError):
            ckpt_value("clear_rate", 0.5, 0.0)

    def test_negative_mean_reward_first_eval_is_saved(self):
        # Edge case: early in the clear-speed task every eval reward is negative
        # (3 - ~30 shots). Starting from 0.0 would never save a best_model.
        best = initial_best_ckpt("mean_reward")
        _, ma, best, is_best = update_best_ma([], -27.0, window=5, best_ma_so_far=best)
        assert is_best
        assert ma == best == -27.0

    def test_less_negative_reward_is_new_best(self):
        history, _, best, _ = update_best_ma([], -20.0, window=2, best_ma_so_far=float("-inf"))
        _, ma, best, is_best = update_best_ma(history, -10.0, window=2, best_ma_so_far=best)
        assert is_best
        assert ma == -15.0


class TestSummarizeFinalEval:
    def test_happy_path(self):
        # 3 games: cleared in 4 shots, cleared in 6 shots, 2/3 balls at the 30-shot cap
        out = summarize_final_eval(ep_lens=[4, 6, 30], ep_pocketed=[3, 3, 2],
                                   ep_pocketed_at_k=[3, 2, 1], n_balls=3)
        assert out["mean_shots"] == pytest.approx(40 / 3)
        assert out["mean_shots_cleared"] == pytest.approx(5.0)
        assert out["pocket_rate_at5"] == pytest.approx(6 / 9)

    def test_no_cleared_games(self):
        out = summarize_final_eval(ep_lens=[30, 30], ep_pocketed=[1, 0],
                                   ep_pocketed_at_k=[0, 0], n_balls=3)
        assert out["mean_shots"] == 30.0
        assert out["mean_shots_cleared"] is None
        assert out["pocket_rate_at5"] == 0.0

    def test_empty_eval(self):
        out = summarize_final_eval([], [], [], n_balls=3)
        assert out == {"mean_shots": None, "mean_shots_cleared": None, "pocket_rate_at5": None}


class TestTargetEntropySchedule:
    def test_constant_when_no_end(self):
        assert target_entropy_at(0, -2.0, None, 1_000_000) == -2.0
        assert target_entropy_at(5_000_000, -2.0, None, 1_000_000) == -2.0

    def test_linear_anneal_then_hold(self):
        assert target_entropy_at(0, -2.0, -7.0, 1_000_000) == -2.0
        assert target_entropy_at(500_000, -2.0, -7.0, 1_000_000) == pytest.approx(-4.5)
        assert target_entropy_at(1_000_000, -2.0, -7.0, 1_000_000) == -7.0
        assert target_entropy_at(2_000_000, -2.0, -7.0, 1_000_000) == -7.0

    def test_zero_anneal_steps_jumps_to_end(self):
        # edge case: no anneal window → use the end value immediately
        assert target_entropy_at(0, -2.0, -7.0, 0) == -7.0


class TestExpDirName:
    def _args(self, **kw):
        base = dict(agent="vanilla", n_balls=3, max_steps=30, seed=0, h_mix_start=0.0,
                    target_entropy=None, target_entropy_end=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_default_name_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        name = os.path.basename(make_exp_dir(self._args()))
        assert name.startswith("exp16_vanilla_multi3_ms30_s0_")

    def test_entropy_anneal_tag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        name = os.path.basename(make_exp_dir(self._args(target_entropy=-2.0, target_entropy_end=-7.0)))
        assert name.startswith("exp16_vanilla_ent-2to-7_multi3_ms30_s0_")
