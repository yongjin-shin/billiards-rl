"""
tests/test_exp16_wm_obs_transform.py

Unit tests for exp16_wm/obs_transform.py — relative-coordinate observation in
the action's frame (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from simulator import BilliardsEnv
from exp16_wm.obs_transform import relative_obs, aim_reference_from_obs, RAW_DIM, REL_DIM
from exp16_wm.networks import TABLE_W, TABLE_L
from exp16_wm.planner import aim_reference

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _obs(cue, balls, pockets=POCKETS):
    """balls: list of (x, y, pocketed) in normalised units."""
    return np.array(list(cue) + [v for b in balls for v in b] + pockets, dtype=np.float32)


class TestRelativeObs:
    def test_shapes(self):
        o = _obs((0.5, 0.5), [(0.6, 0.7, 0), (0.2, 0.2, 0), (0, 0, 1)])
        assert relative_obs(o).shape == (REL_DIM,)
        assert relative_obs(np.stack([o, o, o])).shape == (3, REL_DIM)

    def test_nearest_ball_on_positive_x_axis(self):
        o = _obs((0.5, 0.5), [(0.6, 0.7, 0), (0.1, 0.1, 0), (0, 0, 1)])
        r = relative_obs(o)
        d = np.hypot(0.1 * TABLE_W, 0.2 * TABLE_L)
        assert r[0] == pytest.approx(d, rel=1e-5) and r[1] == pytest.approx(0.0, abs=1e-6)

    def test_rotation_preserves_distances(self):
        o = _obs((0.5, 0.5), [(0.6, 0.7, 0), (0.1, 0.1, 0), (0, 0, 1)])
        r = relative_obs(o)
        d2 = np.hypot(0.4 * TABLE_W, 0.4 * TABLE_L)
        assert np.hypot(r[3], r[4]) == pytest.approx(d2, rel=1e-5)
        dp = np.hypot(0.5 * TABLE_W, 0.5 * TABLE_L)            # pocket (1, 1)
        assert np.hypot(r[9 + 10], r[9 + 11]) == pytest.approx(dp, rel=1e-5)

    def test_pocketed_ball_is_zero_with_flag(self):
        r = relative_obs(_obs((0.5, 0.5), [(0.6, 0.7, 0), (0.1, 0.1, 0), (0, 0, 1)]))
        np.testing.assert_array_equal(r[6:9], [0, 0, 1])

    def test_all_pocketed_uses_zero_reference(self):
        # edge case: no ball left → reference angle 0, pockets only translated
        o = _obs((0.5, 0.5), [(0, 0, 1)] * 3)
        assert aim_reference_from_obs(o)[0] == 0.0
        r = relative_obs(o)
        np.testing.assert_allclose(r[9:11], [-0.5 * TABLE_W, -0.5 * TABLE_L], rtol=1e-5)

    def test_wrong_dim_rejected(self):
        with pytest.raises(ValueError):                 # a length that is 2 + 3n + 12 for no n (G1: any n is accepted)
            relative_obs(np.zeros(RAW_DIM + 1))


class TestMatchesEnvAimRule:
    def test_reference_angle_equals_env(self):
        env = BilliardsEnv(n_balls=3, max_steps=30, step_penalty=1.0, trunc_penalty=0.0)
        env.reset(seed=3)
        for _ in range(8):
            obs, _ = env.reset()
            assert aim_reference_from_obs(obs)[0] == pytest.approx(aim_reference(env), abs=1e-5)
        env.close()
