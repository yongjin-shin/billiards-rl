"""Tests for exp16_wm.v_heatmap_explain: the story cell is the one whose copies disagree most; the figure draws from a story."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_heatmap_explain import pick_story_cell, plot_explain


def test_pick_story_cell_takes_the_most_disagreeing_copies_with_a_mean_in_range():
    copies = {0: np.array([2, 2, 2, 2, 2]),                   # agree: sd 0
              1: np.array([5, 5, 0, 0, 5, 0]),                # mean 2.5, sd 2.5  ← picked
              2: np.array([5, 5, 5, 0]),                      # mean 3.75: out of range
              3: np.array([0, 5, 0]),                         # too few copies
              4: np.array([4, 1, 3, 2])}                      # mean 2.5, sd 1.1
    assert pick_story_cell(copies) == 1


def test_pick_story_cell_raises_when_nothing_qualifies():
    with pytest.raises(ValueError):
        pick_story_cell({0: np.array([3, 3, 3, 3]), 1: np.array([5, 0])})


def test_plot_explain_writes_a_png(tmp_path):
    RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.1, 0.1, 1.0] + [0.0] * 12, np.float32)
    xs, ys = np.arange(6) * 0.02 + 0.03, np.arange(8) * 0.02 + 0.03
    z = np.tile(np.arange(6) % 3, (8, 1)).astype(float)
    c = {"raw": RAW, "xs": xs, "ys": ys, "left": 2, "truth": z, "smooth": z * 0.8, "v": z * 0 + 1.2}
    story = {"x": float(xs[3]), "y": float(ys[4]), "own": 2, "copies_xy": np.stack([xs[3] + np.linspace(-0.01, 0.01, 8), ys[4] + np.linspace(0.01, -0.01, 8)], 1),
             "copies_k": np.array([2, 0, 1, 2, 0, 2, 1, 1]), "iy": 4, "ix": 3}
    out = tmp_path / "e.png"
    plot_explain(c, story, str(out))
    assert out.exists() and out.stat().st_size > 5000
