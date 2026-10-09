"""
tests/test_exp16_wm_outcomes.py

Unit tests for exp16_wm/outcomes.py — which (ball, pocket) pairs an action really makes (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulator import BilliardsEnv
from exp16_wm.outcomes import pocket_index_map, made_slots, annotate_file, intent_stats
from exp16_wm.pair_labels import pairs_for_file
from exp16_wm.planner import aim_reference, simulate_candidate
from exp16_wm.teacher_data import collect, save, CLEAR_ENV


def _env():
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=71)
    return env


class TestPocketMap:
    def test_six_ids_map_to_six_distinct_indices(self):
        m = pocket_index_map(_env())
        assert sorted(m.values()) == [0, 1, 2, 3, 4, 5]
        assert m["lb"] == 0 and m["rt"] == 5          # (0,0) and (1,1) in the observation's pocket order


class TestMadeSlots:
    def test_count_matches_the_env_pocket_count_and_is_deterministic(self):
        env = _env()
        data = collect(seed=72, n_games=2, teacher="ghost", env_kw=CLEAR_ENV, max_good=4, log_every=0)
        pmap = pocket_index_map(env)
        checked = 0
        for i in range(len(data["shot"])):
            a = data["good_actions"][i, 0]
            if np.isnan(a[0]):
                continue
            made, scratch = made_slots(env, data["obs"][i], a, pmap)
            n, sc, _ = simulate_candidate(env, aim_reference(env), a)
            assert made.sum() == n and scratch == sc
            made2, _ = made_slots(env, data["obs"][i], a, pmap)
            np.testing.assert_array_equal(made, made2)
            assert n >= 1                              # a stored good candidate pockets at least one ball
            checked += 1
        assert checked >= 2
        env.close()

    def test_miss_makes_nothing(self):
        env = _env()
        env.reset()
        obs = env._get_obs()
        made, scratch = made_slots(env, obs, np.array([np.radians(179.0), 0.5], dtype=np.float32))
        assert made.sum() == 0 and not scratch
        env.close()


class TestAnnotate:
    def test_file_roundtrip_and_stats(self, tmp_path):
        data = collect(seed=73, n_games=2, teacher="ghost", env_kw=CLEAR_ENV, max_good=4, log_every=0)
        path = str(tmp_path / "s.npz")
        save(path, data, seed=73, teacher="ghost", env_kw=CLEAR_ENV)
        pairs_for_file(path)
        st = annotate_file(path, max_k=2)
        z = np.load(path)
        assert z["good_made"].shape == (len(z["obs"]), 2, 18) and z["good_scratch"].shape == (len(z["obs"]), 2)
        assert st[0]["present"] == len(z["obs"]) - int(np.isnan(z["good_actions"][:, 0, 0]).sum())
        assert 0 <= st[0]["intended_made"] <= st[0]["assigned"] <= st[0]["present"]
        assert st[0]["any_ball_made"] == st[0]["present"]          # good[0] of the ghost teacher always pockets a ball

    def test_intent_stats_without_pairs_counts_none_assigned(self):
        z = {"good_actions": np.array([[[0.1, 1.0], [np.nan, np.nan]]], dtype=np.float32),
             "good_made": np.zeros((1, 2, 18), bool), "good_scratch": np.zeros((1, 2), bool)}
        st = intent_stats(z, 2)
        assert st[0]["assigned"] == 0 and st[1]["present"] == 0
