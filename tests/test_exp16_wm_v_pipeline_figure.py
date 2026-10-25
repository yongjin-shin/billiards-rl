"""Tests for exp16_wm.v_pipeline_figure: grid maps, the story cell, and the figure from a V8 file."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_pipeline_figure import draw, layout_arrays, story_cell, to_map

W, L, R = 0.9906, 1.9812, 0.028575
RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.3, 0.7, 0.0] + [0.0] * 12, np.float32)


def _file(tmp_path):
    from exp16_wm.v_map_data import build_layout, finish_layout, run_layouts, write_file
    lay = build_layout(RAW, 2, 31, 0, (10.0, 20.0, 30.0, 40.0, 60.0), 1, 0.07, 0.0, R, W, L)
    ks = run_layouts([lay], lambda jobs: [np.floor(o[:, 0] * 4 + o[:, 1] * 3).astype(int) % 3 for o, _ in jobs])
    out = str(tmp_path / "g.npz")
    write_file(out, np.stack([RAW]), np.array([2]), np.array([31]), [lay], [finish_layout(lay, ks[0], 5, 1)], (10.0, 20.0, 30.0, 40.0, 60.0))
    return dict(np.load(out, allow_pickle=True))


def test_to_map_puts_values_on_cells_and_leaves_the_rest_nan():
    m = to_map(np.array([1.0, 2.0]), np.array([[0, 1], [2, 0]]), (3, 2))
    assert m[0, 1] == 1.0 and m[2, 0] == 2.0 and np.isnan(m[1, 1]) and m.shape == (3, 2)


def test_layout_arrays_and_story_cell_pick_a_cell_with_disagreeing_rings(tmp_path):
    z = _file(tmp_path)
    arr = layout_arrays(z, 31)
    assert arr["nxy"] == (14, 28) and len(arr["ij"]) == len(arr["label"]) == arr["ring_k"].shape[0] and arr["left"] == 2
    c = story_cell(arr)
    rk = arr["ring_k"][c, :, 0]
    assert len(set(rk.tolist())) > 1 and 1 < arr["ij"][c, 0] < 12 and 1 < arr["ij"][c, 1] < 26      # rings disagree, away from the cushions


def test_figure_is_written_with_and_without_a_model_prediction(tmp_path):
    z = _file(tmp_path)
    arr = layout_arrays(z, 31)
    draw(arr, arr["obs"][0], None, 1.14, str(tmp_path / "a.png"))
    draw(arr, arr["obs"][0], np.clip(arr["label"] + 0.3, 0, 2), 1.14, str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").stat().st_size > 8000 and (tmp_path / "b.png").stat().st_size > 8000
