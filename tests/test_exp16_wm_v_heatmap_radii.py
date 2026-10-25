"""Tests for exp16_wm.v_heatmap_radii: disk average with holes, jump share, panel list, figure."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_heatmap_radii import blur, boltzmann, correlation_table, kl_div, trend_report, disk_average, jump_share, load_real, panels_of, noise_report, plot_radii, zone_auc, zone_table


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


def test_panels_use_real_simulated_radii_when_given_and_the_approximation_otherwise():
    c = _cache()
    real3 = np.full_like(c["truth"], 7.0)
    ps = panels_of(c, real={3.0: real3})
    assert ps[3][1] == "8번 실제 시뮬레이션" and np.all(ps[3][2] == 7.0)                  # 3 cm is the simulation
    assert ps[2][1].startswith("근사") and ps[4][1].startswith("근사")                   # the others stay approximated


def test_correlation_table_reports_jumps_and_correlations_with_truth_and_v():
    c = _cache(); c["v"] = c["truth"] * 0.5 + 1.0                                        # V is a rescaled copy of ① → corr 1 with ① and V itself
    rows = correlation_table(panels_of(c))
    assert rows[0][2] == pytest.approx(1.0) and rows[0][3] == pytest.approx(1.0) and rows[-1][3] == pytest.approx(1.0)
    assert rows[0][1] >= rows[-2][1]                                                      # ① is no smoother than the widest average


def test_load_real_reads_the_saved_radii_and_skips_missing_ones(tmp_path):
    z = np.ones((3, 3))
    np.savez(tmp_path / "r.npz", radii_cm=np.array([2.0, 4.0]), **{"smooth_2": z * 2, "smooth_4": z * 4})
    got = load_real(str(tmp_path / "r.npz"))
    assert sorted(got) == [2.0, 4.0] and got[4.0][0, 0] == 4.0


def test_zone_auc_perfect_reversed_and_degenerate_cases():
    zone = np.array([[0.0, 1.0, 4.0], [5.0, np.nan, 4.5]])
    assert zone_auc(np.array([[0.1, 0.2, 0.9], [1.0, 0.0, 0.8]]), zone, 4.0) == 1.0          # high exactly where the zone is
    assert zone_auc(-np.array([[0.1, 0.2, 0.9], [1.0, 0.0, 0.8]]), zone, 4.0) == 0.0
    assert np.isnan(zone_auc(np.zeros((2, 3)), zone, 9.0))                                     # nobody is in the zone → no AUC
    pred = np.array([[0.1, 0.2, 0.9], [1.0, 5.0, 0.8]])                                         # the NaN cell of zone_map is ignored (its pred 5.0 too)
    assert zone_auc(pred, zone, 4.0) == 1.0


def test_zone_table_rows_for_the_averaged_maps_only():
    c = _cache()
    rows = zone_table(panels_of(c), {"x": c["truth"]}, threshold=1.0)
    assert len(rows) == 5 and rows[0][0] == "1 cm 평균"                                       # 1 cm, 2, 3, 4, 5 cm
    assert set(rows[0][2]) == {"V", "① 한 번 친 결과", "x"} and 0.0 <= rows[0][1] <= 1.0


def test_noise_report_identical_maps_are_noise_free_and_independent_noise_lowers_reproducibility():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 6, (40, 30)).astype(float)
    v = base * 0.5 + rng.normal(0, 1, base.shape)
    same = noise_report(base, base, v)
    assert same["reproducibility"] == pytest.approx(1.0) and same["reliability_16"] == pytest.approx(1.0) and same["zone_agreement"] == 1.0
    assert same["replicate_auc"] == pytest.approx(1.0) and same["jump_single"] == pytest.approx(same["jump_16"])
    noisy = noise_report(base + rng.normal(0, 1.5, base.shape), base + rng.normal(0, 1.5, base.shape), v)
    assert noisy["reproducibility"] < 0.9 and noisy["jump_16"] < noisy["jump_single"] and noisy["zone_agreement"] < 1.0   # averaging two copies smooths
    assert noisy["corr_v_16_corrected"] > noisy["corr_v_16"]                                                              # the correction raises the correlation


def test_blur_keeps_constants_and_holes_and_zero_sigma_is_identity():
    z = np.full((6, 6), 2.0); z[2, 3] = np.nan
    out = blur(z, 1.0)
    assert np.isnan(out[2, 3]) and np.allclose(out[np.isfinite(out)], 2.0)
    np.testing.assert_array_equal(blur(np.arange(6.0).reshape(2, 3), 0), np.arange(6.0).reshape(2, 3))


def test_boltzmann_is_a_distribution_over_finite_cells_and_beta_zero_is_uniform():
    z = np.array([[0.0, 1.0], [np.nan, 3.0]])
    p = boltzmann(z, 1.0)
    assert p.sum() == pytest.approx(1.0) and p[1, 0] == 0.0 and p[1, 1] > p[0, 1] > p[0, 0]
    np.testing.assert_allclose(boltzmann(z, 0.0)[np.isfinite(z)], 1 / 3)
    assert kl_div(p, p) == pytest.approx(0.0)


def test_trend_report_a_rescaled_copy_is_perfect_a_flat_or_random_map_explains_nothing():
    rng = np.random.default_rng(0)
    ref = blur(rng.normal(0, 1, (30, 20)) * 2 + 3, 2.0)
    same = trend_report(ref * 0.2 + 7.0, ref, sigma_cells=0.0)                       # only the scale and offset differ: temperature is fitted away
    assert same["explained"] > 0.97 and same["corr"] == pytest.approx(1.0)
    flat = trend_report(np.zeros_like(ref) + 1.0, ref, sigma_cells=0.0)
    assert abs(flat["explained"]) < 1e-9
    rand = trend_report(rng.normal(0, 1, ref.shape), ref, sigma_cells=0.0)
    assert rand["explained"] < 0.2 and rand["kl_uniform"] > 0
