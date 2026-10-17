"""
tests/test_exp16_wm_label_gap.py

Unit tests for exp16_wm/label_gap.py — policy-vs-label gap analysis (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from exp16_wm.label_gap import unit_to_env, nearest_good_idx, err_stats, pick_rows
from exp16_wm.bc import dim_weights, to_unit
from exp16_wm.train import ACT_LOW, ACT_HIGH

W = dim_weights(0.2, 0.5).numpy()


class TestUnits:
    def test_unit_roundtrip(self):
        a = np.array([[0.3, 4.0], [-2.0, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(unit_to_env(to_unit(a)), a, atol=1e-5)
        np.testing.assert_allclose(unit_to_env(np.array([1.0, 1.0])), ACT_HIGH)


class TestNearestGood:
    def test_picks_candidate_nearest_in_weighted_space(self):
        pred = np.array([[0.0, 3.0]], dtype=np.float32)
        good = np.array([[[np.radians(0.5), 3.0], [np.radians(0.1), 5.0], [np.nan, np.nan]]], dtype=np.float32)
        # 0.5° off at equal speed (2.5 units) beats 2 m/s off at near-equal angle (4 units)
        assert nearest_good_idx(pred, good, W)[0] == 0

    def test_rows_without_candidates_give_minus_one(self):
        pred = np.zeros((2, 2), dtype=np.float32) + [0.0, 3.0]
        good = np.full((2, 2, 2), np.nan, dtype=np.float32)
        good[0, 1] = [0.01, 3.0]
        np.testing.assert_array_equal(nearest_good_idx(pred, good, W), [1, -1])


class TestErrStats:
    def test_values(self):
        ang = np.array([0.1, 0.3, 2.0, 0.05])
        spd_err = np.array([0.1, 0.5, 1.0, 0.0])
        spd_label = np.array([1.0, 1.0, 2.0, 4.0])
        s = err_stats(ang, spd_err, spd_label)
        assert s["angle_within_0.2"] == 0.5 and s["angle_within_1.0"] == 0.75
        assert s["speed_median_ms"] == pytest.approx(0.3)
        assert s["speed_within_20pct"] == 0.5               # rel errors 0.1, 0.5, 0.5, 0.0 → 2 of 4 below 0.2

    def test_zero_label_speed_does_not_divide_by_zero(self):
        s = err_stats(np.array([0.0]), np.array([0.1]), np.array([0.0]))
        assert np.isfinite(s["speed_rel_median"])


class TestPickRows:
    MASK = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)

    def test_no_limit_returns_all_masked_rows(self):
        np.testing.assert_array_equal(pick_rows(self.MASK, 0, "val"), [1, 2, 4, 5, 6, 8, 9])
        np.testing.assert_array_equal(pick_rows(self.MASK, 100, "train"), [1, 2, 4, 5, 6, 8, 9])

    def test_val_takes_the_first_n(self):
        np.testing.assert_array_equal(pick_rows(self.MASK, 3, "val"), [1, 2, 4])

    def test_train_sample_is_seeded_sorted_and_inside_the_mask(self):
        a, b = pick_rows(self.MASK, 4, "train", seed=0), pick_rows(self.MASK, 4, "train", seed=0)
        np.testing.assert_array_equal(a, b)
        assert len(a) == 4 and (np.diff(a) > 0).all() and self.MASK[a].all()


class TestNearestGoodWrap:
    def test_candidate_across_the_seam_is_nearest(self):
        pred = np.array([[np.radians(179.5), 3.0]], dtype=np.float32)
        good = np.array([[[np.radians(-179.5), 3.0], [np.radians(120.0), 3.0]]], dtype=np.float32)
        assert nearest_good_idx(pred, good, W)[0] == 0
