"""Tests for exp16_wm.wide_scan: 360° scan → success windows → window centres, without the aim formula."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.wide_scan import coarse_angles, find_windows, fine_angles, scan_state, to_env_actions, wrap_deg

REF = PairGNNPolicy(hidden=8)
SC, BI = REF.act_scale.numpy().astype(np.float64), REF.act_bias.numpy().astype(np.float64)


def _deg(env_actions: np.ndarray) -> np.ndarray:
    return wrap_deg(np.degrees(env_actions[:, 0]))


def test_coarse_angles_cover_the_circle_once():
    a = coarse_angles(0.05)
    assert len(a) == 7200 and a[0] == -180.0 and a[-1] < 180.0
    np.testing.assert_allclose(np.diff(a), 0.05)


def test_find_windows_runs_including_one_across_the_seam():
    assert find_windows(np.zeros(10, bool)) == []
    assert find_windows(np.ones(10, bool)) == [(0, 10)]
    f = np.zeros(10, bool); f[[2, 3, 4, 7]] = True
    assert sorted(find_windows(f)) == [(2, 3), (7, 1)]
    g = np.zeros(10, bool); g[[8, 9, 0, 1]] = True; g[5] = True
    assert sorted(find_windows(g)) == [(5, 1), (8, 4)]                       # 8, 9, 0, 1 is one window


def test_fine_angles_widen_the_coarse_window_by_one_step():
    fa = fine_angles(10.0, 3, 0.05, 0.01)                                    # coarse 10.00, 10.05, 10.10
    assert fa[0] == pytest.approx(9.95) and fa[-1] == pytest.approx(10.15)
    np.testing.assert_allclose(np.diff(fa), 0.01, atol=1e-9)


def test_env_actions_keep_the_direction():
    ang = np.array([-179.99, -90.0, 0.0, 45.5, 179.95])
    a = to_env_actions(ang, 1.0, SC, BI)
    np.testing.assert_allclose(wrap_deg(_deg(a) - ang), 0, atol=1e-3)
    assert (a[:, 1] == 1.0).all()


def _fake_sim(windows: dict[int, tuple[float, float]], scratch: tuple[float, float] | None = None):
    """pair j is made for directions in [lo, hi] deg (lo > hi wraps across ±180); scratch inside `scratch`."""
    def inside(d, lo, hi):
        return (d >= lo) & (d <= hi) if lo <= hi else (d >= lo) | (d <= hi)
    def sim(task):
        _, acts = task
        d = _deg(acts)
        made = np.zeros((len(d), 18), bool)
        for j, (lo, hi) in windows.items():
            made[:, j] = inside(d, lo, hi)
        scr = inside(d, *scratch) if scratch else np.zeros(len(d), bool)
        return made, scr
    return sim


def test_scan_state_finds_window_centres_widths_and_skips_scratch_and_missing_pairs():
    sim = _fake_sim({0: (10.0, 10.12), 5: (179.9, -179.9), 7: (-30.0, -29.5)}, scratch=(-29.8, -29.5))
    r = scan_state(sim, np.zeros(23, np.float32), 1.0, 0.05, 0.01, SC, BI)
    assert r["valid"][[0, 5, 7]].all() and not r["valid"][[1, 2, 3, 4, 6]].any()
    assert r["centre"][0] == pytest.approx(10.06, abs=0.006)
    assert abs(wrap_deg(r["centre"][5] - 180.0)) < 0.006                      # across the seam: centre at ±180, not 0
    assert r["centre"][7] == pytest.approx(-29.9, abs=0.006)                  # scratch part excluded
    assert r["width"][0] == pytest.approx(0.13, abs=0.011)
    assert r["n_windows"][0] == 1 and r["n_sims"] > 7200


def test_scan_state_keeps_the_widest_of_several_windows():
    sim = _fake_sim({3: (20.0, 20.06)})
    def two(task):
        made, scr = sim(task)
        d = _deg(task[1]); made[:, 3] |= (d >= -40.0) & (d <= -39.5)          # a second, wider window
        return made, scr
    r = scan_state(two, np.zeros(23, np.float32), 1.0, 0.05, 0.01, SC, BI)
    assert r["n_windows"][3] == 2 and r["centre"][3] == pytest.approx(-39.75, abs=0.006)


def test_robust_speeds_keep_only_directions_that_work_at_every_speed():
    def sim(task):
        _, acts = task
        d, v = _deg(acts), float(acts[0, 1])
        made = np.zeros((len(d), 18), bool)
        made[:, 2] = (d >= 30.0) & (d <= 31.0) if v == 1.0 else (d >= -60.0) & (d <= -59.0)    # a wide window at one speed only
        made[:, 2] |= (d >= 5.0) & (d <= 5.2)                                                  # the direct window, both speeds
        return made, np.zeros(len(d), bool)
    plain = scan_state(sim, np.zeros(23, np.float32), 1.0, 0.05, 0.01, SC, BI)
    robust = scan_state(sim, np.zeros(23, np.float32), 1.0, 0.05, 0.01, SC, BI, robust_speeds=(2.0,))
    assert plain["centre"][2] == pytest.approx(30.5, abs=0.006)                                   # widest at 1 m/s alone
    assert robust["centre"][2] == pytest.approx(5.1, abs=0.006) and robust["n_windows"][2] == 1
    assert robust["n_sims"] >= 2 * 7200


def test_fine_pad_recovers_the_full_window_at_the_scan_speed():
    def sim(task):
        _, acts = task
        d, v = _deg(acts), float(acts[0, 1])
        made = np.zeros((len(d), 18), bool)
        made[:, 4] = (d >= 10.0) & (d <= 10.6) if v == 1.0 else (d >= 10.0) & (d <= 10.2)    # wide at 1 m/s, narrow at 2 m/s
        return made, np.zeros(len(d), bool)
    narrow = scan_state(sim, np.zeros(23, np.float32), 1.0, 0.1, 0.01, SC, BI, robust_speeds=(2.0,))
    padded = scan_state(sim, np.zeros(23, np.float32), 1.0, 0.1, 0.01, SC, BI, robust_speeds=(2.0,), fine_pad_deg=0.5)
    assert padded["centre"][4] == pytest.approx(10.3, abs=0.006) and padded["width"][4] == pytest.approx(0.61, abs=0.011)
    assert narrow["centre"][4] < padded["centre"][4]                                         # without the pad the centre is pulled

