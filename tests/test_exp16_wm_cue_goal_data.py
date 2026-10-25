"""Tests for exp16_wm.cue_goal_data (CT0b: collect cue-only shots whose end positions cover the table)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import (N_K, TABLE_L, TABLE_W, Params, cell_index, collect_start, count_cells, grid_shape, hit_rates, shake,
                                    standing_mask, thin_entries)

CELL = 0.025


def test_grid_shape_covers_the_table():
    nx, ny = grid_shape(CELL)
    assert nx * CELL >= TABLE_W > (nx - 1) * CELL and ny * CELL >= TABLE_L > (ny - 1) * CELL


def test_cell_index_buckets_and_clips_onto_the_grid():
    nx, ny = grid_shape(CELL)
    ij = cell_index(np.array([[0.0, 0.0], [0.026, 0.051], [TABLE_W + 0.5, TABLE_L + 0.5], [-0.1, -0.1]]), CELL)
    assert ij.tolist() == [[0, 0], [1, 2], [nx - 1, ny - 1], [0, 0]]


def test_standing_mask_excludes_the_cushion_strip_but_not_the_middle():
    m = standing_mask(CELL)
    assert not m[0].any() and not m[:, 0].any() and m[10, 10]


def test_count_cells_ignores_scratch_and_bins_k():
    end = np.array([[0.305, 0.305], [0.31, 0.31], [0.305, 0.305], [0.305, 0.305]])   # all in cell (12, 12); 0.30 sits on a float boundary
    k = np.array([0, 0, 5, 1], np.int16); scratch = np.array([False, False, False, True])
    c = count_cells(end, k, scratch, CELL)
    assert c.shape[0] == N_K and c.sum() == 3
    assert c[0, 12, 12] == 2 and c[N_K - 1, 12, 12] == 1   # k = 5 goes to the "3+" bin; the scratch shot is not counted


def test_thin_entries_touch_filled_ones_and_respect_attempts_and_standing():
    prm = Params()
    counts = np.zeros((N_K,) + grid_shape(CELL), np.int32)
    counts[0, 12, 12] = 5                                  # full: not thin itself
    attempts = np.zeros(counts.shape, np.int16)
    t = thin_entries(counts, attempts, prm)
    assert {tuple(x) for x in t} == {(0, 12 + dx, 12 + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - {(0, 12, 12)}
    attempts[0, 13, 12] = prm.max_attempts                 # given up
    assert (0, 13, 12) not in {tuple(x) for x in thin_entries(counts, attempts, prm)}
    counts2 = np.zeros_like(counts); counts2[0, 0, 0] = 1  # a filled entry in the cushion strip: its neighbours that stand are still targets
    t2 = {tuple(x) for x in thin_entries(counts2, np.zeros_like(attempts), prm)}
    assert (0, 0, 0) not in t2 and (0, 1, 1) in t2


def test_shake_keeps_ranges_and_is_deterministic():
    phi, v = np.full(200, 359.99), np.full(200, 7.99)
    a = shake(phi, v, np.random.default_rng(1), (1e-4, 8.0)); b = shake(phi, v, np.random.default_rng(1), (1e-4, 8.0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert (a[0] >= 0).all() and (a[0] < 360).all() and (a[1] >= 1e-4).all() and (a[1] <= 8.0).all()
    assert not np.array_equal(a[0], phi)


def test_hit_rates_per_k_and_any():
    end = np.array([[0.30, 0.30], [0.50, 0.50], [0.70, 0.70]]); k = np.array([0, 1, 1], np.int16); scratch = np.array([False, False, True])
    goals = np.array([[0.305, 0.30], [0.50, 0.515], [0.70, 0.70], [0.9, 1.5]])
    h = hit_rates(end, k, scratch, goals)
    assert h["k0"]["2cm"] == 0.25 and h["k0"]["1cm"] == 0.25      # only goal 0 is near the k = 0 shot
    assert h["k1"]["2cm"] == 0.25 and h["k1"]["1cm"] == 0.0       # goal 1 is 1.5 cm from the k = 1 shot; the scratch shot at goal 2 does not count
    assert h["any"]["2cm"] == 0.5
    assert h["k2"] == {"2cm": 0.0, "1cm": 0.0}                    # no shot with k = 2 at all


def test_collect_start_tiny_budget_is_consistent_and_fills_more_cells_each_round():
    prm = Params(n0=300, children=2, max_rounds=3, max_shots=3000)
    res = collect_start((np.array([0.5, 0.8]), 7, prm))
    n = len(res["k"])
    assert 300 < n <= 3000
    for key in ("phi", "v", "end", "scratch", "round"):
        assert len(res[key]) == n
    assert (res["v"] >= 1e-4).all() and (res["v"] <= 8.0).all()
    assert (res["round"][:300] == 0).all() and res["round"].max() >= 1
    filled = [sum(t["filled"]) for t in res["trace"]]
    assert all(b >= a for a, b in zip(filled, filled[1:])) and filled[-1] > filled[0]
    assert res["trace"][0]["shots"] == 300
