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


def test_plot3d_writes_a_png_from_a_cache(tmp_path):
    from exp16_wm.v_heatmap import plot3d
    xs, ys, valid = valid_grid(RAW, 0.1, R, W, L)
    z = np.where(valid, np.random.default_rng(0).integers(0, 3, valid.shape).astype(float), np.nan)
    cache = tmp_path / "c.npz"
    np.savez(cache, raw=RAW, xs=xs, ys=ys, valid=valid, truth=z, smooth=z, v=z, left=2, row=0, meta="{}")
    out = tmp_path / "o.png"
    plot3d(str(cache), str(out))
    assert out.exists() and out.stat().st_size > 5000


def _cache(tmp_path, ny=5, nx=4):
    xs, ys = np.arange(nx) * 0.02 + R, np.arange(ny) * 0.02 + R
    truth = np.arange(ny * nx, dtype=float).reshape(ny, nx) % 3
    truth[0, 0] = np.nan
    cache = tmp_path / "c.npz"
    np.savez(cache, raw=RAW, xs=xs, ys=ys, valid=np.isfinite(truth), truth=truth, smooth=truth * 0.9, v=truth * 0 + 1.5, left=2, row=0, meta="{}")
    return str(cache)


def test_html_page_embeds_the_maps_as_valid_json_with_nulls_for_holes(tmp_path):
    import json
    import re
    from exp16_wm.v_heatmap_html import TITLE, build_html
    html = build_html(_cache(tmp_path))
    d = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    assert f"<title>{TITLE}</title>" in html and "NaN" not in html.split("<script>")[1].split("</script>")[0].split("const D = ")[1].split(";\n")[0]
    assert d["nx"] == 4 and d["ny"] == 5 and [p["key"] for p in d["panels"]] == ["truth", "smooth", "v"]
    assert all(len(p["z"]) == 20 for p in d["panels"]) and d["panels"][0]["z"][0] is None      # the hole is null, not NaN
    assert d["panels"][2]["stats"]["std"] == 0.0 and d["panels"][2]["stats"]["jump"] == 0.0       # a flat map has no jumps


def test_neighbour_jump_share_counts_steps_of_one_ball_and_ignores_holes():
    from exp16_wm.v_heatmap_html import neighbour_jump_share
    z = np.array([[0.0, 1.0, np.nan], [0.0, 0.5, 0.5]])
    # horizontal pairs: (0,1)=1 ✓, (1,nan) skipped, (0,0.5)=0.5 ✗, (0.5,0.5)=0 ✗ ; vertical: (0,0)=0, (1,0.5)=0.5, (nan,.5) skipped → 1 of 5
    assert neighbour_jump_share(z) == 0.2
    assert np.isnan(neighbour_jump_share(np.full((2, 2), np.nan)))
