"""Tests for exp16_wm.v_heatmap: free-position grid, simulation inputs, truth / smoothed maps."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_heatmap import live_balls, rollout_inputs, to_maps, valid_grid, with_cue

W, L, R = 1.0, 2.0, 0.03
# 2 balls: ball 0 at the table centre (live), ball 1 pocketed; 12 pocket values (20 = 2 + 3·2 + 12)
RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.1, 0.1, 1.0] + [0.0] * 12, np.float32)


def test_live_balls_ignores_pocketed_ones():
    np.testing.assert_allclose(live_balls(RAW, W, L), [[0.5, 1.0]])


def test_valid_grid_stays_on_the_table_and_clears_the_balls():
    xs, ys, valid = valid_grid(RAW, 0.05, R, W, L)
    assert xs[0] == R and xs[-1] <= W - R and ys[0] == R and ys[-1] <= L - R
    gx, gy = np.meshgrid(xs, ys)
    d = np.hypot(gx - 0.5, gy - 1.0)
    assert not valid[d < 2 * R].any() and valid[d >= 2 * R + 1e-3].all()           # exactly the cells near the live ball are cut out
    assert (~valid).any() and valid.sum() > 0


def test_valid_grid_without_live_balls_is_all_free():
    raw = RAW.copy(); raw[4] = 1.0                                                    # pocket the only live ball
    assert valid_grid(raw, 0.1, R, W, L)[2].all()


def test_rollout_inputs_cover_each_cell_once_plus_placed_copies():
    xs, ys, valid = valid_grid(RAW, 0.1, R, W, L)
    cells, obs, owner = rollout_inputs(RAW, xs, ys, valid, 0.01, 3, 0, W, L, R)
    assert len(cells) == valid.sum()
    own = owner[:, 1] < 0
    assert own.sum() == len(cells) and obs.shape[1] == len(RAW)
    first = obs[own][0]
    np.testing.assert_allclose(first[:2], [xs[cells[0, 1]] / W, ys[cells[0, 0]] / L], atol=1e-6)
    cp = obs[(owner[:, 0] == 0) & ~own]
    d = np.hypot((cp[:, 0] - first[0]) * W, (cp[:, 1] - first[1]) * L)
    np.testing.assert_allclose(d, 0.01, atol=1e-5)                                    # copies are exactly eps away
    a = rollout_inputs(RAW, xs, ys, valid, 0.01, 3, 0, W, L, R)[1]
    np.testing.assert_array_equal(a, obs)                                             # same seed → same copies


def test_to_maps_truth_smooth_and_empty_cells_are_nan():
    cells = np.array([[0, 0], [0, 1]])
    owner = np.array([[0, -1], [0, 0], [0, 1], [1, -1]])                              # cell 0 has two copies, cell 1 none
    truth, smooth = to_maps((1, 3), cells, owner, np.array([4, 2, 4, 1]))
    np.testing.assert_array_equal(truth[0, :2], [4, 1])
    assert np.isnan(truth[0, 2])
    assert smooth[0, 0] == 3.0 and np.isnan(smooth[0, 1]) and np.isnan(smooth[0, 2])


def test_with_cue_moves_only_the_cue_ball():
    out = with_cue(RAW, 0.3, 0.8, W, L)
    assert out[0] == np.float32(0.3) and out[1] == np.float32(0.4)
    np.testing.assert_array_equal(out[2:], RAW[2:])
