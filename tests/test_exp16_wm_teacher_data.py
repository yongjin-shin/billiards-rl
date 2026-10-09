"""
tests/test_exp16_wm_teacher_data.py

Unit tests for exp16_wm/teacher_data.py — imitation-learning data from a
search-planner teacher (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pytest

from simulator import BilliardsEnv
from exp16_wm.teacher_data import (pack_good_actions, collect, save, to_arrays, CLEAR_ENV,
                                     widest_window, robust_shot, TEACHERS, best_margin_cell, margin_shot)
from exp16_wm.planner import aim_reference, simulate_candidate

A = [np.array([0.1 * i, 3.0], dtype=np.float32) for i in range(5)]


class TestPackGoodActions:
    def test_keeps_only_pocketing_sorted_best_first(self):
        results = [(0, False, 0.1), (1, False, 0.5), (2, False, 0.9), (1, False, 0.2), (0, True, 0.0)]
        good, n = pack_good_actions(A, results, max_good=4)
        assert n == 3
        np.testing.assert_array_equal(good[0], A[2])   # 2 balls
        np.testing.assert_array_equal(good[1], A[3])   # 1 ball, remaining nearer a pocket
        np.testing.assert_array_equal(good[2], A[1])
        assert np.isnan(good[3]).all()

    def test_cap_counts_all_but_stores_max_good(self):
        results = [(1, False, 0.1 * i) for i in range(5)]
        good, n = pack_good_actions(A, results, max_good=2)
        assert n == 5 and good.shape == (2, 2) and not np.isnan(good).any()

    def test_no_pocketing_candidates(self):
        good, n = pack_good_actions(A, [(0, False, 0.3)] * 5, max_good=3)
        assert n == 0 and np.isnan(good).all()


class TestCollect:
    def test_small_run_shapes_and_teacher_choice(self):
        data = collect(seed=7, n_games=2, teacher="ghost", env_kw=CLEAR_ENV, max_good=8, log_every=0)
        n = len(data["shot"])
        assert n >= 2
        assert data["obs"].shape == (n, 23)
        assert data["action"].shape == (n, 2)
        assert data["good_actions"].shape == (n, 8, 2)
        assert set(np.unique(data["game"])) == {0, 1}
        assert (data["action"][:, 1] >= 0.5).all() and (data["action"][:, 1] <= 8.0).all()
        # when the teacher's shot pocketed, it is the first (best) stored good action
        hit = data["n_good"] > 0
        np.testing.assert_allclose(data["good_actions"][hit, 0], data["action"][hit])

    def test_empty_to_arrays(self):
        out = to_arrays({k: [] for k in ["obs", "action", "good_actions", "n_good",
                                          "pocketed", "game", "shot"]})
        assert len(out["shot"]) == 0 and out["action"].shape == (0, 2)

    def test_save_roundtrip(self, tmp_path):
        data = collect(seed=8, n_games=1, teacher="ghost", env_kw=CLEAR_ENV, max_good=4, log_every=0)
        path = str(tmp_path / "t.npz")
        save(path, data, seed=8, teacher="ghost", env_kw=CLEAR_ENV)
        z = np.load(path)
        assert json.loads(str(z["meta"]))["teacher"] == "ghost"
        np.testing.assert_array_equal(z["action"], data["action"])


class TestWidestWindow:
    OFFS = np.round(np.arange(-0.2, 0.2 + 1e-9, 0.04), 4)       # 11 points

    def test_picks_widest_run_and_its_centre(self):
        hits = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], bool)
        w, c = widest_window(hits, self.OFFS)
        assert w == pytest.approx(0.12) and c == pytest.approx(-0.04)   # cells 3..5 = -0.08..0.00

    def test_tie_keeps_first_run(self):
        hits = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], bool)
        w, c = widest_window(hits, self.OFFS)
        assert w == pytest.approx(0.08) and c == pytest.approx(-0.18)

    def test_none_and_all(self):
        w, c = widest_window(np.zeros(11, bool), self.OFFS)
        assert w == 0.0 and np.isnan(c)
        w, c = widest_window(np.ones(11, bool), self.OFFS)
        assert w == pytest.approx(0.44) and c == pytest.approx(0.0, abs=1e-9)


class TestRobustTeacher:
    def test_chosen_shot_pockets_without_scratch(self):
        env = BilliardsEnv(**CLEAR_ENV)
        env.reset(seed=31)
        rng = np.random.default_rng(0)
        for _ in range(2):
            env.reset()
            action, good, n_good, width = robust_shot(env, rng, TEACHERS["robust"], max_good=4)
            n, scratch, _ = simulate_candidate(env, aim_reference(env), action)
            if width > 0:
                assert n >= 1 and not scratch and n_good >= 1
                np.testing.assert_allclose(good[0], action)
            assert 0.5 <= action[1] <= 8.0
        env.close()

    def test_collect_records_window_width(self):
        data = collect(seed=32, n_games=1, teacher="robust", env_kw=CLEAR_ENV, max_good=4, log_every=0)
        assert len(data["win_width"]) == len(data["shot"]) >= 1
        assert (data["win_width"] >= 0).all()


class TestBestMarginCell:
    ARGS = dict(angle_step_deg=0.1, ln_speed_step=0.2, angle_scale_deg=0.2, speed_scale_ln=0.2)

    def _rect(self, rows, cols, shape=(21, 9)):
        h = np.zeros(shape, bool)
        h[rows[0]:rows[1], cols[0]:cols[1]] = True
        return h

    def test_centre_of_rectangle_has_largest_margin(self):
        m, (i, j) = best_margin_cell(self._rect((6, 15), (2, 7)), **self.ARGS)
        assert (i, j) == (10, 4) and m > 0

    def test_prefers_the_speed_direction_when_speed_is_the_tight_axis(self):
        # wide in angle (21 cells = 2°), narrow in speed (3 cells): margin is limited by speed, not angle
        h = self._rect((0, 21), (3, 6))
        m, (i, j) = best_margin_cell(h, **self.ARGS)
        assert j == 4
        assert m == pytest.approx(2.0, abs=0.01)        

    def test_tied_ridge_picks_its_centre(self):
        h = self._rect((2, 19), (3, 6))
        _, (i, j) = best_margin_cell(h, **self.ARGS)
        assert j == 4 and abs(i - 10) <= 1

    def test_no_hits_and_all_hits(self):
        assert best_margin_cell(np.zeros((5, 5), bool), **self.ARGS) == (0.0, (-1, -1))
        m, (i, j) = best_margin_cell(np.ones((5, 5), bool), **self.ARGS)
        assert m > 0 and (i, j) == (2, 2)

    def test_angle_grid_edge_is_not_a_wall(self):
        # a window touching the angle edge looks wider than the same window floating inside
        edge = np.zeros((21, 9), bool); edge[0:8, 2:7] = True
        inner = np.zeros((21, 9), bool); inner[6:14, 2:7] = True
        assert best_margin_cell(edge, **self.ARGS)[0] >= best_margin_cell(inner, **self.ARGS)[0]


class TestMarginTeacher:
    def test_shot_pockets_cleanly_and_reports_margin(self):
        env = BilliardsEnv(**CLEAR_ENV)
        env.reset(seed=51)
        rng = np.random.default_rng(0)
        env.reset()
        rec: list = []
        action, good, n_good, margin = margin_shot(env, rng, TEACHERS["margin"], max_good=4, record=rec)
        n, scratch, _ = simulate_candidate(env, aim_reference(env), action)
        assert len(rec) > 1000                                  # the full 2D scan was recorded
        assert 0.5 - 1e-6 <= action[1] <= 8.0 + 1e-6
        if margin > 0:
            assert n >= 1 and not scratch and n_good >= 1
            np.testing.assert_allclose(good[0], action)
        env.close()

    def test_collect_with_margin_teacher(self):
        data = collect(seed=52, n_games=1, teacher="margin", env_kw=CLEAR_ENV, max_good=4, log_every=0)
        assert len(data["win_width"]) == len(data["shot"]) >= 1 and (data["win_width"] >= 0).all()
