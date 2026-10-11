"""Tests for exp16_wm/eval_by_balls.py (pure helpers; the env-based measurements are checked by the run itself)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.eval_by_balls import balls_left, by_balls, scan_metrics, wrapped_angle_error_deg, summarize


def test_balls_left_counts_unpocketed_slots():
    obs = np.zeros((3, 23))
    obs[0, [4, 7, 10]] = 0            # nothing pocketed
    obs[1, 4] = 1                     # one pocketed
    obs[2, [4, 7]] = 1                # two pocketed
    assert balls_left(obs).tolist() == [3, 2, 1]


def test_by_balls_groups_and_leaves_empty_groups_nan():
    v = np.array([1.0, 3.0, 5.0, 7.0]); n = np.array([1, 1, 3, 3])
    r = by_balls(v, n)
    assert r[1] == 2.0 and r[3] == 6.0 and np.isnan(r[2])
    assert by_balls(v, n, np.max)[3] == 7.0


def test_wrapped_angle_error_takes_the_short_way():
    e = wrapped_angle_error_deg(np.array([0.999, 0.0]), np.array([-0.999, 0.1]))
    assert abs(e[0] - 0.36) < 1e-6 and abs(e[1] - 18.0) < 1e-6          # 0.002π rad = 0.36°, 0.1π rad = 18°


def test_scan_metrics_perfect_and_wrong_heads():
    truth = np.zeros((2, 7), bool); truth[:, 2:5] = True               # window in the middle
    good = np.where(truth, 0.9, 0.1)
    m = scan_metrics(good, truth)
    assert m["auc"] == 1.0 and m["argmax_in_window"] == 1.0 and m["n_states_partial"] == 2 and m["brier"] < 0.02
    bad = np.where(truth, 0.1, 0.9)
    m = scan_metrics(bad, truth)
    assert m["auc"] == 0.0 and m["argmax_in_window"] == 0.0


def test_scan_metrics_ignores_states_without_a_partial_window():
    truth = np.array([[True] * 5, [False] * 5, [False, True, True, False, False]])
    prob = np.array([[0.5] * 5, [0.5] * 5, [0.1, 0.8, 0.2, 0.1, 0.1]])
    m = scan_metrics(prob, truth)
    assert m["n_states_partial"] == 1 and m["argmax_in_window"] == 1.0
    assert np.isnan(scan_metrics(np.full((1, 3), 0.5), np.ones((1, 3), bool))["argmax_in_window"])     # no partial state


def test_summarize_pools_seeds_per_balls_left():
    def seed(err, ok, nl):
        r = {"n_left": np.array(nl), "labelled": np.ones(len(nl), bool), "angle_err": np.array(err, float),
             "clean": {"select": np.array(ok, bool)}}
        return {"run": r, "_truth": None}
    s = summarize([seed([0.1, 0.5], [True, False], [1, 2]), seed([0.1, 0.3], [True, True], [1, 2])])["run"]
    assert s["angle_err_median_deg"][1] == pytest.approx(0.1) and s["angle_err_median_deg"][2] == pytest.approx(0.4)
    assert s["clean"]["select"][2] == 0.5 and s["clean"]["select"]["all"] == 0.75
    assert s["clean"]["select"]["by_seed"] == [0.5, 1.0]
