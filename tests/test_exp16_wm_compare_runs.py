"""
tests/test_exp16_wm_compare_runs.py

Unit tests for exp16_wm/compare_runs.py helpers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp16_wm.compare_runs import arm_of, load_runs, paired_p, parse_eval_curve, window_means

LOG = """
  [    1,000 / 2,000,000]    0.1%  elapsed=00:00:02
  EVAL
    pocket   :   34.7%    ma(5): 34.7%    best_ma: 34.7%
  [   60,000 / 2,000,000]    3.0%  elapsed=00:01:00
    pocket   :   50.0%    ma(5): 40.0%    best_ma: 40.0%
""".splitlines()


class TestParseEvalCurve:
    def test_steps_and_pocket(self):
        steps, pocket = parse_eval_curve(LOG)
        assert steps.tolist() == [1000, 60000]
        assert pocket.tolist() == [34.7, 50.0]

    def test_empty_log(self):
        steps, pocket = parse_eval_curve([])
        assert steps.size == 0 and pocket.size == 0


class TestWindowMeans:
    def test_window_assignment_and_empty_is_nan(self):
        w = window_means(np.array([1000, 60000]), np.array([34.7, 50.0]),
                         windows=[(0, 50e3), (50e3, 100e3), (100e3, 200e3)])
        assert w[0] == 34.7 and w[1] == 50.0 and np.isnan(w[2])


class TestArmOf:
    def test_arms(self):
        assert arm_of({"agent": "vanilla"}, {}) == "vanilla"
        assert arm_of({"agent": "geom"}, {}) == "geom"
        assert arm_of({"agent": "wm"}, {"wm_target": "rssm", "h_mix_start": 0.5}) == "hmix"
        assert arm_of({"agent": "wm"}, {"wm_target": "rssm", "h_mix_start": 0.0}) == "rssm"
        assert arm_of({"agent": "wm"}, {"wm_target": "traj"}) == "traj"


class TestPairedP:
    def test_needs_two_shared_seeds(self):
        assert np.isnan(paired_p({0: 1.0}, {0: 2.0, 1: 3.0}))

    def test_consistent_shift_is_significant(self):
        assert paired_p({0: 1.0, 1: 2.0, 2: 3.0}, {0: 1.5, 1: 2.4, 2: 3.6}) < 0.05


class TestLoadRuns:
    def test_empty_root(self, tmp_path):
        assert load_runs(str(tmp_path)) == {}
