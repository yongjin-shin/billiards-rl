"""
tests/test_exp16_wm_relabel.py

Unit tests for exp16_wm/relabel.py — relabel stored states with the robust teacher (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulator import BilliardsEnv
from exp16_wm.relabel import restore_state, shard_rows, relabel_rows, original_rows
from exp16_wm.teacher_data import collect, CLEAR_ENV
from exp16_wm.planner import aim_reference, simulate_candidate, plan_shot


class TestRestoreState:
    def test_restored_table_gives_same_outcomes(self):
        src, dst = BilliardsEnv(**CLEAR_ENV), BilliardsEnv(**CLEAR_ENV)
        src.reset(seed=11)
        dst.reset(seed=999)
        rng = np.random.default_rng(0)
        kw = dict(offsets_deg=[0.0], speeds=[3.0], max_cut_deg=80.0, n_random=10)
        checked = 0
        for _ in range(3):
            obs, _ = src.reset()
            done = False
            while not done:
                restore_state(dst, obs)
                assert dst._pocketed == src._pocketed
                np.testing.assert_allclose(dst._get_obs(), obs, atol=1e-6)
                a = plan_shot(src, rng, **kw)[0]
                for act in (a, rng.uniform(src.action_space.low, src.action_space.high).astype(np.float32)):
                    assert (simulate_candidate(src, aim_reference(src), act)[:2]
                            == simulate_candidate(dst, aim_reference(dst), act)[:2])
                    checked += 1
                obs, _, term, trunc, _ = src.step(a)
                done = term or trunc
        assert checked >= 6


class TestShards:
    def test_whole_games_per_shard_and_full_cover(self):
        game = np.repeat(np.arange(10), 3)
        parts = [shard_rows(game, s, 4) for s in range(4)]
        assert sorted(np.concatenate(parts).tolist()) == list(range(30))
        for p in parts:
            for g in np.unique(game[p]):
                assert (game[p] == g).sum() == 3            # a game is never split across shards

    def test_empty_shard(self):
        assert len(shard_rows(np.array([0, 0, 2, 2]), 1, 2)) == 0


class TestRelabel:
    def test_same_rows_two_label_sets(self):
        data = collect(seed=41, n_games=2, teacher="ghost", env_kw=CLEAR_ENV, max_good=8, log_every=0)
        rows = np.arange(len(data["shot"]))
        new = relabel_rows(data, rows, seed=0, max_good=4, log_every=0)
        old = original_rows(data, rows, max_good=4)
        for k in ["obs", "game", "shot"]:
            np.testing.assert_array_equal(new[k], old[k])
        assert new["good_actions"].shape == old["good_actions"].shape == (len(rows), 4, 2)
        assert (new["win_width"] >= 0).all() and np.isnan(old["win_width"]).all()
        np.testing.assert_array_equal(old["action"], data["action"])
