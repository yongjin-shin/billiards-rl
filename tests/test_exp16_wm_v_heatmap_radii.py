"""Tests for exp16_wm.v_heatmap_radii: disk average with holes, jump share, panel list, figure."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_heatmap_radii import disk_average, jump_share, panels_of, plot_radii


def test_disk_average_radius_zero_is_identity_and_constant_stays_constant_around_holes():
    z = np.arange(20, dtype=float).reshape(4, 5)
    np.testing.assert_allclose(disk_average(z, 0.0), z)
    c = np.full((5, 5), 3.0); c[2, 2] = np.nan
    out = disk_average(c, 2.0)
    assert np.isnan(out[2, 2]) and np.allclose(out[np.isfinite(out)], 3.0)          # holes neither used nor filled, no edge bias


def test_disk_average_spreads_a_spike_over_the_cells_within_the_radius():
    z = np.zeros((7, 7)); z[3, 3] = 5.0
    out = disk_average(z, 1.0)                                                     # centre + 4 neighbours → 5 cells share it
    assert out[3, 3] == pytest.approx(1.0) and out[3, 4] == pytest.approx(5.0 / 5) and out[3, 5] == 0.0 and out[2, 2] == 0.0


def test_jump_share_counts_steps_of_a_ball_and_ignores_holes():
    z = np.array([[0.0, 1.0, np.nan], [0.0, 0.5, 0.5]])
    assert jump_share(z) == 0.2                                                    # same case as the page's stats
    assert np.isnan(jump_share(np.full((2, 2), np.nan)))


def _cache():
    RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.1, 0.1, 1.0] + [0.0] * 12, np.float32)
    xs, ys = np.arange(6) * 0.02 + 0.03, np.arange(9) * 0.02 + 0.03
    z = np.tile(np.arange(6) % 3, (9, 1)).astype(float)
    return {"raw": RAW, "xs": xs, "ys": ys, "left": 2, "truth": z, "smooth": z * 0.8, "v": z * 0 + 1.2}


def test_panels_are_actual_one_cm_four_approximations_and_v_in_that_order():
    ps = panels_of(_cache())
    assert [p[0] for p in ps] == ["① 실제 결과", "1 cm 평균", "2 cm 평균", "3 cm 평균", "4 cm 평균", "5 cm 평균", "③ V의 예측"]
    c = _cache()
    np.testing.assert_array_equal(ps[0][2], c["truth"]); np.testing.assert_array_equal(ps[1][2], c["smooth"])
    assert jump_share(ps[4][2]) <= jump_share(ps[2][2]) <= jump_share(ps[0][2])    # a larger radius is never rougher


def test_plot_radii_writes_a_png(tmp_path):
    c = _cache(); cache = tmp_path / "c.npz"
    np.savez(cache, **c, valid=np.isfinite(c["truth"]), row=0, meta="{}")
    out = tmp_path / "r.png"
    plot_radii(str(cache), str(out))
    assert out.exists() and out.stat().st_size > 5000
