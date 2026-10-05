"""
tests/test_exp16_wm_diagnose_q.py

Unit tests for the pure metric helpers and critic access in exp16_wm/diagnose_q.py.
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp16_wm.diagnose_q import (
    behaviour_action, critic_features, critic_q, discounted_returns,
    probe_auc, r2_score_flat, safe_auc, within_step_pearson,
)
from exp16_wm.sac import VanillaSAC, WMSAC
from exp16_wm.train import ACT_HIGH, ACT_LOW


class TestSafeAuc:
    def test_perfect_separation(self):
        assert safe_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0

    def test_single_class_is_nan(self):
        assert np.isnan(safe_auc(np.array([0.1, 0.5]), np.array([1, 1])))


class TestR2:
    def test_exact_prediction(self):
        t = np.arange(10.0)
        assert r2_score_flat(t, t) == pytest.approx(1.0)

    def test_mean_prediction_is_zero(self):
        t = np.arange(10.0)
        assert r2_score_flat(np.full(10, t.mean()), t) == pytest.approx(0.0)

    def test_constant_target_is_nan(self):
        assert np.isnan(r2_score_flat(np.ones(5), np.zeros(5)))


class TestProbeAuc:
    def test_informative_feature_beats_noise(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        good = (y + rng.normal(0, 0.5, 400))[:, None]
        noise = rng.normal(0, 1, (400, 1))
        assert probe_auc(good, y) > 0.8
        assert abs(probe_auc(noise, y) - 0.5) < 0.1

    def test_too_few_positives_is_nan(self):
        y = np.array([0] * 20 + [1] * 2)
        assert np.isnan(probe_auc(np.random.randn(22, 3), y))


class TestDiscountedReturns:
    def test_values(self):
        np.testing.assert_allclose(discounted_returns([1.0, 0.0, 1.0], 0.5), [1.25, 0.5, 1.0])

    def test_empty_episode(self):
        assert discounted_returns([], 0.99).shape == (0,)


class TestBehaviourAction:
    def test_stays_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = behaviour_action(np.array([3.1, 7.9], dtype=np.float32), rng)
            assert np.all(a >= ACT_LOW) and np.all(a <= ACT_HIGH)

    def test_p_random_zero_keeps_near_policy(self):
        rng = np.random.default_rng(0)
        a = np.stack([behaviour_action(np.array([0.0, 4.0]), rng, p_random=0.0) for _ in range(200)])
        assert abs(a[:, 0].mean()) < 0.05


class TestCriticAccess:
    def test_vanilla_shapes(self):
        ag = VanillaSAC(obs_dim=23, action_dim=2, act_low=ACT_LOW, act_high=ACT_HIGH)
        obs, act = np.random.rand(7, 23).astype(np.float32), np.random.rand(7, 2).astype(np.float32)
        assert critic_q(ag, obs, act).shape == (7,)
        assert critic_features(ag, obs, act).shape == (7, 256)

    def test_wm_features_are_h_hat(self):
        ag = WMSAC(obs_dim=23, action_dim=2, act_low=ACT_LOW, act_high=ACT_HIGH,
                   max_events=4, event_dim=64)
        obs, act = np.random.rand(3, 23).astype(np.float32), np.random.rand(3, 2).astype(np.float32)
        assert critic_features(ag, obs, act).shape == (3, 4 * 64)
        assert critic_q(ag, obs, act).shape == (3,)


class TestWithinStepPearson:
    def test_removes_between_step_offset(self):
        # Q tracks G inside each t, but a per-t offset makes the pooled corr negative
        rng = np.random.default_rng(0)
        t = np.repeat(np.arange(3), 50)
        g = rng.normal(size=150)
        q = g - 10.0 * t
        assert stats.pearsonr(q, g + 10.0 * t).statistic < 0.5
        assert within_step_pearson(q, g + 10.0 * t, t) == pytest.approx(1.0)

    def test_too_few_samples_is_nan(self):
        assert np.isnan(within_step_pearson(np.arange(5.0), np.arange(5.0), np.zeros(5), min_n=20))
