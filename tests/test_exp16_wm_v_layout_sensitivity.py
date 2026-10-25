"""Tests for exp16_wm.v_layout_sensitivity: moving a ball, matching cells of two maps, and the map recipe with a fake simulator."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_layout_sensitivity import compare, map_values, move_ball

W, L, R = 0.9906, 1.9812, 0.028575
RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.3, 0.7, 0.0] + [0.0] * 12, np.float32)        # two live balls (the cue ball is replaced anyway)


def test_move_ball_moves_exactly_the_distance_keeps_the_others_and_stays_legal():
    out = move_ball(RAW, 0, 0.03, np.random.default_rng(0), W, L, R)
    assert out is not None
    old, new = np.array([RAW[2] * W, RAW[3] * L]), np.array([out[2] * W, out[3] * L])
    assert np.linalg.norm(new - old) == pytest.approx(0.03, abs=1e-5)
    np.testing.assert_array_equal(np.delete(out, [2, 3]), np.delete(RAW, [2, 3]))          # nothing else changes
    assert R <= new[0] <= W - R and R <= new[1] <= L - R
    other = np.array([RAW[5] * W, RAW[6] * L])
    assert np.linalg.norm(new - other) >= 2 * R
    assert move_ball(RAW, 1, 0.03, np.random.default_rng(1), W, L, R) is not None and move_ball(RAW, 3, 0.03, np.random.default_rng(1), W, L, R) is not None  # `which` wraps around


def test_move_ball_returns_none_when_no_direction_keeps_it_on_the_table():
    assert move_ball(RAW, 0, 5.0, np.random.default_rng(0), W, L, R, tries=50) is None        # 5 m is farther than the table is wide in every direction


def test_compare_matches_cells_by_position_and_reports_correlation():
    cells = np.array([[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1], [0.6, 0.1], [0.7, 0.1], [0.8, 0.1]])
    a = {"cells": cells, "val": np.arange(8.0)}
    b = {"cells": cells[1:], "val": 2 * np.arange(1.0, 8.0) + 3}                          # one cell missing, values a rescaled copy on the common ones
    r = compare(a, b)
    assert r["n"] == 7 and r["corr"] == pytest.approx(1.0) and r["corr_blur"] == pytest.approx(1.0, abs=1e-6)
    c = compare(a, {"cells": cells, "val": -np.arange(8.0)})
    assert c["corr"] == pytest.approx(-1.0)


def test_map_values_gives_one_label_per_cell_from_a_fake_simulator():
    cells = np.array([[0.2, 0.5], [0.4, 0.9], [0.7, 1.4]])
    fake = lambda jobs: [np.full(len(o), 2) for o, _ in jobs]                              # every shot pockets 2 balls
    v = map_values(RAW, 3, cells, 0, (10.0, 30.0), 2, fake, W, L, R)
    np.testing.assert_allclose(v, [2.0, 2.0, 2.0])
    rows = lambda jobs: [np.floor(o[:, 0] * 10).astype(int) for o, _ in jobs]              # k = floor(10 · cue x): the exact row equals the cell's own x
    v2 = map_values(RAW, 3, cells, 0, (10.0, 30.0), 2, rows, W, L, R)
    assert v2.shape == (3,) and np.all(np.abs(v2 - np.floor(cells[:, 0] / W * 10)) <= 1.0)
