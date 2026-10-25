"""Tests for exp16_wm.single_ball (CT0: cue-only table, forward (angle, speed) grid, cushion count, grid-gap analysis)."""

import os
import sys

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.single_ball import (adjacent_gaps, k_distribution, law_k0, make_cue_only_system, make_grid, run_start,
                                  sample_starts, shoot)

START = (0.15, 0.4)   # a 20-degree line from here has ~0.89 m of free run before the first cushion and passes no pocket


def _ray(phi_deg: float) -> np.ndarray:
    return np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])


# ---------------------------------------------------------------- the cue-only table


def test_dummy_ball_is_pocketed_and_cue_only_system_simulates():
    system = make_cue_only_system(*START)
    assert system.balls["1"].state.s == pt.constants.pocketed
    r = shoot(system, 20.0, 0.2)   # would raise `min() iterable argument is empty` without the dummy ball
    assert not r.scratch


def test_k0_shot_is_straight_and_distance_scales_with_v_squared():
    system = make_cue_only_system(*START)
    ray = _ray(20.0)
    along = {}
    for v in (0.1, 0.2):
        r = shoot(system, 20.0, v)
        d = np.array([r.x, r.y]) - START
        assert r.k == 0
        assert abs(d[0] * ray[1] - d[1] * ray[0]) < 1e-3   # sideways drift < 1 mm
        along[v] = d @ ray
    assert abs(along[0.2] / along[0.1] - 4.0) < 0.02       # d ∝ v²


def test_shoot_does_not_change_the_system_it_was_given():
    system = make_cue_only_system(*START)
    before = system.balls["cue"].state.rvw.copy()
    shoot(system, 20.0, 1.0)
    assert np.array_equal(system.balls["cue"].state.rvw, before)


def test_cushion_count_grows_with_speed():
    system = make_cue_only_system(0.5, 0.2)
    ks = [shoot(system, 90.0, v).k for v in (0.5, 2.0, 8.0)]
    assert ks[0] == 1 and ks[1] >= ks[0] and ks[2] >= 2


def test_tiny_speed_stops_almost_where_it_started_with_no_cushion():
    system = make_cue_only_system(0.5, 0.5)
    r = shoot(system, 0.0, 0.05)
    assert r.k == 0 and not r.scratch
    assert 0.0 < r.x - 0.5 < 0.03 and abs(r.y - 0.5) < 1e-6


def test_start_next_to_a_cushion_aimed_at_it_hits_it_once_at_low_speed():
    system = make_cue_only_system(0.04, 0.6)   # 4 cm from the x = 0 cushion
    r = shoot(system, 180.0, 0.3)
    assert r.k >= 1 and not r.scratch


def test_shot_into_the_corner_pocket_is_marked_scratch():
    system = make_cue_only_system(0.5, 0.5)
    r = shoot(system, 225.0, 2.0)   # straight at the (0, 0) corner pocket
    assert r.scratch


# ---------------------------------------------------------------- grid and starts


def test_make_grid_shapes_and_spacing():
    phis, speeds = make_grid(720, 60)
    assert phis.shape == (720,) and speeds.shape == (60,)
    assert np.allclose(np.diff(phis), 0.5) and phis[-1] == 359.5
    assert np.isclose(speeds[0], 0.05) and np.isclose(speeds[-1], 8.0)
    assert np.allclose(speeds[1:] / speeds[:-1], speeds[1] / speeds[0])   # log spacing


def test_sample_starts_are_deterministic_on_the_table_and_away_from_pockets():
    a, b = sample_starts(40, seed=3), sample_starts(40, seed=3)
    assert np.array_equal(a, b) and not np.array_equal(a, sample_starts(40, seed=4))
    table = pt.Table.default()
    assert (a[:, 0] > 0.03).all() and (a[:, 0] < table.w - 0.03).all() and (a[:, 1] > 0.03).all() and (a[:, 1] < table.l - 0.03).all()
    pockets = np.array([np.asarray(p.center[:2], float) for p in table.pockets.values()])
    assert np.linalg.norm(a[:, None] - pockets[None], axis=-1).min() >= 0.10


def test_run_start_matches_single_shots():
    phis, speeds = np.array([20.0, 90.0]), np.array([0.2, 1.0, 3.0])
    out = run_start((np.array(START), phis, speeds))
    assert out["end"].shape == (2, 3, 2) and out["k"].shape == (2, 3)
    system = make_cue_only_system(*START)
    for i, phi in enumerate(phis):
        for j, v in enumerate(speeds):
            r = shoot(system, float(phi), float(v))
            assert np.allclose(out["end"][i, j], (r.x, r.y)) and out["k"][i, j] == r.k and out["scratch"][i, j] == r.scratch


# ---------------------------------------------------------------- analysis on synthetic grids


def _synthetic(n_phi=8, n_v=5, gap=0.01):
    """One start at the origin, end positions along x spaced `gap` metres per angle step, y per speed step."""
    speeds = np.geomspace(0.05, 8.0, n_v)
    end = np.zeros((1, n_phi, n_v, 2))
    end[0, :, :, 0] = np.arange(n_phi)[:, None] * gap
    end[0, :, :, 1] = np.arange(n_v)[None, :] * gap
    return speeds, end, np.zeros((1, n_phi, n_v), np.int16), np.zeros((1, n_phi, n_v), bool)


def test_adjacent_gaps_dense_grid_is_fully_within_2cm():
    speeds, end, k, sc = _synthetic(gap=0.01)
    g = adjacent_gaps(end, k, sc, speeds, (0.05, 8.0))
    # the angle axis wraps: the last cell is next to cell 0, a jump of 7 gaps — only that column is wide
    assert g["v"]["per_k"]["0"]["within_2cm"] == 1.0 and g["v"]["boundary_share"] == 0.0
    assert np.isclose(g["phi"]["per_k"]["0"]["within_2cm"], 7 / 8)
    assert np.isclose(g["v"]["per_k"]["0"]["median_cm"], 1.0)


def test_adjacent_gaps_separates_cells_with_different_k_and_scratch():
    speeds, end, k, sc = _synthetic(gap=0.5)   # gaps of 50 cm: never within 2 cm
    k[0, 3:, :] = 1
    sc[0, 0, 0] = True
    g = adjacent_gaps(end, k, sc, speeds, (0.05, 8.0))
    assert g["phi"]["per_k"]["0"]["within_2cm"] == 0.0 and g["phi"]["per_k"]["1"]["n"] > 0
    assert g["phi"]["boundary_share"] > 0.0   # the k = 0 -> 1 step, the wrap-around 1 -> 0 and the scratch cells


def test_adjacent_gaps_respects_the_speed_range():
    speeds, end, k, sc = _synthetic(n_v=60, gap=0.01)
    both = adjacent_gaps(end, k, sc, speeds, (0.05, 8.0))["v"]["per_k"]["0"]["n"]
    a_only = adjacent_gaps(end, k, sc, speeds, (0.5, 8.0))["v"]["per_k"]["0"]["n"]
    assert 0 < a_only < both


def test_k_distribution_sums_to_one_and_respects_range():
    speeds = np.array([0.1, 1.0, 2.0])
    k = np.array([[[0, 1, 5], [0, 2, 3]]], np.int16)   # (1 start, 2 angles, 3 speeds)
    sc = np.zeros(k.shape, bool); sc[0, 1, 2] = True
    full = k_distribution(speeds, k, sc, (0.05, 8.0))
    assert np.isclose(sum(full[n] for n in ("0", "1", "2", "3", "4+")) + full["scratch"], 1.0)
    a = k_distribution(speeds, k, sc, (0.5, 8.0))
    assert a["n"] == 4 and a["0"] == 0.0   # no k = 0 shot at v >= 0.5


def test_law_k0_perfect_and_empty():
    starts = np.array([[0.2, 0.2]]); phis = np.array([0.0, 90.0]); speeds = np.array([0.1, 0.2])
    end = np.zeros((1, 2, 2, 2))
    for i, phi in enumerate(phis):
        for j, v in enumerate(speeds):
            end[0, i, j] = starts[0] + 7.76 * v ** 2 * _ray(phi)
    out = law_k0(starts, phis, speeds, end, np.zeros((1, 2, 2), np.int16), np.zeros((1, 2, 2), bool))
    assert out["n"] == 4 and abs(out["c_mean"] - 7.76) < 1e-9 and out["c_rel_max_dev"] < 1e-9 and out["lateral_max_mm"] < 1e-6
    none = law_k0(starts, phis, speeds, end, np.ones((1, 2, 2), np.int16), np.zeros((1, 2, 2), bool))
    assert none == {"n": 0}
