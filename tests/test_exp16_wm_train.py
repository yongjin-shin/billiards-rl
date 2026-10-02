"""
tests/test_exp16_wm_train.py

Unit tests for exp16_wm/train.py::update_best_ma — the moving-average
best-checkpoint selection helper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp16_wm.train import update_best_ma


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
