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

from exp16_wm.teacher_data import pack_good_actions, collect, save, to_arrays, CLEAR_ENV

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
