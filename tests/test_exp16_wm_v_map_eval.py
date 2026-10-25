"""Tests for exp16_wm.v_map_eval: scattered blur, per-layout scores, bootstrap interval, noise limit, loading."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_map_eval import bootstrap_ci, corr, layout_metrics, load_heldout, noise_limit, per_layout, reference_labels, smooth_scattered, trend_vectors


def _xy(n=400, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, (n, 2))


def test_smooth_scattered_keeps_constants_averages_neighbours_and_shrinks_noise():
    xy = _xy()
    np.testing.assert_allclose(smooth_scattered(xy, np.full(len(xy), 3.0), 0.1), 3.0)
    noise = np.random.default_rng(1).normal(0, 1, len(xy))
    assert smooth_scattered(xy, noise, 0.1).std() < 0.5 * noise.std()                  # averaging many neighbours cancels independent noise
    spike = np.zeros(len(xy)); spike[0] = 10.0
    out = smooth_scattered(xy, spike, 0.05)
    assert out[0] < 10.0 and out.max() == out[0] and (out >= 0).all()                  # the spike spreads out but stays highest at its own position


def test_trend_vectors_a_rescaled_copy_is_perfect_and_a_constant_explains_nothing():
    ref = np.sin(np.linspace(0, 6, 300)) * 2 + 3
    assert trend_vectors(ref * 0.1 + 5, ref)["explained"] > 0.97
    assert abs(trend_vectors(np.ones_like(ref), ref)["explained"]) < 1e-9


def test_layout_metrics_perfect_model_scores_one_and_a_reversed_model_scores_below_chance():
    xy = _xy()
    ref = np.clip(3 + 2 * np.sin(6 * xy[:, 0]) + np.cos(5 * xy[:, 1]), 0, 5)
    good = layout_metrics(ref * 0.2 + 1, ref, ref, xy)
    assert good["corr"] == pytest.approx(1.0) and good["auc"] == pytest.approx(1.0) and good["corr_4"] > 0.999 and good["explained_4"] > 0.95
    bad = layout_metrics(-ref, ref, ref, xy)
    assert bad["corr"] == pytest.approx(-1.0) and bad["auc"] == pytest.approx(0.0)
    assert np.isnan(layout_metrics(ref, np.zeros_like(ref), ref, xy)["auc"])           # no good zone at all → no AUC


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_layouts():
    x = np.random.default_rng(0).normal(0.5, 0.2, 100)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and m == pytest.approx(x.mean()) and hi - lo < 0.15
    assert bootstrap_ci(x[:10])[2] - bootstrap_ci(x[:10])[1] > hi - lo
    assert bootstrap_ci(np.array([1.0, np.nan, 3.0]))[0] == 2.0                        # NaN layouts are dropped


def _fake_heldout(tmp_path, n_layouts=3, per=40, seed=0):
    rng = np.random.default_rng(seed)
    for f in range(2):
        lay, xy, rk, kx, obs = [], [], [], [], []
        for j in range(n_layouts):
            lid = 100 + f * 10 + j
            base = rng.uniform(0, 5, per)                                               # the layout's smooth truth
            rk.append(np.clip(np.round(base[:, None, None] + rng.normal(0, 1, (per, 5, 3))), 0, 6).astype(np.int8))
            kx.append(np.clip(np.round(base), 0, 6)); lay.append(np.full(per, lid)); xy.append(rng.uniform(0, 1, (per, 2))); obs.append(rng.normal(0, 1, (per, 8)))
        c = np.concatenate
        np.savez(tmp_path / f"layouts_{f}.npz", obs=c(obs).astype(np.float32), left=np.full(n_layouts * per, 5), layout_id=c(lay), cell_xy=c(xy),
                 k_exact=c(kx).astype(int), ring_k=c(rk), radii_mm=np.array([10.0, 20.0, 30.0, 40.0, 60.0]))
    return str(tmp_path)


def test_load_reference_noise_limit_and_per_layout_on_fake_files(tmp_path):
    h = load_heldout(_fake_heldout(tmp_path))
    assert len(h["obs"]) == 240 and len(np.unique(h["layout"])) == 6
    lab = reference_labels(h)
    np.testing.assert_allclose(lab["ref"], h["ring_k"].reshape(240, -1).astype(float).mean(axis=1), atol=1e-5)      # all 15 copies, none unplaced
    assert lab["copy0"].shape == (240,) and not np.allclose(lab["copy0"], lab["copy1"])
    nl = noise_limit(h, lab)
    assert 0.0 < nl["copy_corr"] < 1.0 and nl["copy_corr"] < nl["reliability15"] < 1.0 and nl["max_corr_with_ref"] == pytest.approx(np.sqrt(nl["reliability15"]))
    r = per_layout(h, lab["ref"], lab["ref"])
    assert len(r["corr"]) == 6 and np.allclose(r["corr"], 1.0)                          # the reference predicts itself perfectly
    r2 = per_layout(h, lab["copy0"], lab["ref"])
    assert (r2["corr"] < 1.0).all() and r2["corr"].mean() > 0.3                         # one noisy copy: good but not perfect


def test_corr_returns_nan_for_a_constant_vector():
    assert np.isnan(corr(np.ones(5), np.arange(5.0))) and corr(np.arange(5.0), np.arange(5.0) * 2) == pytest.approx(1.0)
