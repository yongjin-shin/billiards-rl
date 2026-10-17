"""
tests/test_exp16_wm_planner.py

Unit tests for exp16_wm/planner.py — the pooltool search-planner reference
for the clear-speed task (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from simulator import BilliardsEnv
from exp16_wm.planner import (wrap_angle, ghost_ball_aim, aim_reference, action_from_phi,
                              strike_phi_deg, simulate_candidate, candidate_score, pick_best,
                              ghost_candidates, play_game)

R = 0.028575
PLAN_KW = {"offsets_deg": [-0.2, 0.0, 0.2], "speeds": [3.0], "max_cut_deg": 80.0, "n_random": 20}


def _env(max_steps: int = 30, seed: int = 0) -> BilliardsEnv:
    env = BilliardsEnv(n_balls=3, max_steps=max_steps, step_penalty=1.0, trunc_penalty=0.0)
    env.reset(seed=seed)
    return env


class TestGeometry:
    def test_wrap_angle_range(self):
        for x in [-10.0, -np.pi, 0.0, np.pi, 3 * np.pi, 7.5]:
            w = wrap_angle(x)
            assert -np.pi <= w < np.pi
            assert np.isclose(np.cos(w), np.cos(x)) and np.isclose(np.sin(w), np.sin(x))

    def test_straight_shot_aims_at_ball_with_zero_cut(self):
        phi, cut = ghost_ball_aim(np.array([0.5, 0.5]), np.array([0.5, 1.0]),
                                  np.array([0.5, 2.0]), R)
        assert phi == pytest.approx(np.pi / 2)
        assert cut == pytest.approx(0.0, abs=1e-6)

    def test_45_degree_cut(self):
        # object ball must travel along (1,1)/√2 while the cue comes in roughly along +x
        _, cut = ghost_ball_aim(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                np.array([2.0, 1.0]), R)
        assert 40.0 < np.degrees(cut) < 50.0

    def test_back_cut_is_over_90_degrees(self):
        # pocket "behind" the ball relative to the cue → impossible cut, filtered by max_cut
        _, cut = ghost_ball_aim(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                np.array([0.0, 0.1]), R)
        assert np.degrees(cut) > 90.0

    def test_degenerate_ball_on_pocket(self):
        assert ghost_ball_aim(np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                              np.array([1.0, 1.0]), R) is None


class TestPickBest:
    def test_more_pocketed_wins(self):
        assert pick_best([(0, False, 0.1), (2, False, 0.9), (1, False, 0.0)]) == 1

    def test_scratch_costs_half_a_ball(self):
        assert candidate_score(1, True) == 0.5
        # a clean miss (0) loses to pocket+scratch (0.5) but a clean pocket (1) beats it
        assert pick_best([(0, False, 0.0), (1, True, 0.5)]) == 1
        assert pick_best([(1, True, 0.0), (1, False, 0.5)]) == 1

    def test_tie_break_prefers_balls_near_pocket(self):
        assert pick_best([(0, False, 0.40), (0, False, 0.05), (0, False, 0.20)]) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pick_best([])


class TestEnvAimRule:
    def test_action_reproduces_absolute_phi(self):
        env = _env()
        phi_ref = aim_reference(env)
        for phi in [-3.0, -0.5, 0.0, 1.2, 3.1]:
            a = action_from_phi(phi, phi_ref, 3.0)
            assert -np.pi <= a[0] <= np.pi
            d = np.radians(strike_phi_deg(phi_ref, a)) - phi
            assert np.isclose(np.cos(d), 1.0, atol=1e-9)
        env.close()

    def test_simulate_does_not_mutate_env(self):
        env = _env()
        before = {bid: b.state.rvw.copy() for bid, b in env.system.balls.items()}
        phi_ref = aim_reference(env)
        simulate_candidate(env, phi_ref, np.array([0.0, 5.0], dtype=np.float32))
        for bid, rvw in before.items():
            np.testing.assert_array_equal(env.system.balls[bid].state.rvw, rvw)
        env.close()

    def test_prediction_matches_env_step(self):
        # pooltool is deterministic: what the planner sees on the copy must be what env.step does
        env = _env(seed=1)
        for _ in range(6):
            phi_ref = aim_reference(env)
            cands = ghost_candidates(env, phi_ref, [0.0], [3.0], 80.0)
            action = cands[0] if cands else np.array([0.0, 3.0], dtype=np.float32)
            n_pred, scratch_pred, _ = simulate_candidate(env, phi_ref, action)
            _, _, term, trunc, info = env.step(action)
            assert info["pocketed_this_step"] == n_pred
            assert info["scratch"] == scratch_pred
            if term or trunc:
                break
        env.close()


class TestPlayGame:
    def test_ghost_candidates_respect_cut_limit(self):
        env = _env()
        phi_ref = aim_reference(env)
        loose = ghost_candidates(env, phi_ref, [0.0], [3.0], 89.0)
        tight = ghost_candidates(env, phi_ref, [0.0], [3.0], 10.0)
        assert len(tight) <= len(loose) <= 3 * 6
        env.close()

    def test_game_stops_at_cap_and_reports(self):
        # edge case: a 2-shot cap truncates unless the planner clears 3 balls in 2 shots
        env = _env(max_steps=2)
        out = play_game(env, np.random.default_rng(0), **PLAN_KW)
        assert 1 <= out["shots"] <= 2
        assert 0 <= out["pocketed_at5"] == out["pocketed"] <= 3
        assert out["mismatches"] == 0
        env.close()
