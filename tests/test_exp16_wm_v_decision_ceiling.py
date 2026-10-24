"""Tests for exp16_wm.v_decision_ceiling: hole filling, ε-copy value sources, metrics per origin balls-left."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_decision_ceiling import (fill_missing, metrics_by_left, next_value_sources, perturbed_value_sources, pooled_diff,
                                         regenerate_perturbed, to_grid)


def _set() -> tuple[dict[str, np.ndarray], np.ndarray]:
    # 4 origins × 3 candidates: origins 0, 1 have 3 balls left (candidates all succeed, later balls differ), origins 2, 3 have 2 left
    t = {"made": np.array([[0.9, 0.8, 0.7], [0.6, 0.9, 0.5], [0.9, 0.8, 0.7], [0.7, 0.7, 0.9]]),
         "succ": np.array([[True, True, True], [True, True, False], [True, True, True], [True, True, True]]),
         "balls": np.ones((4, 3), dtype=int),
         "k_next": np.array([[2, 0, 1], [1, 2, -1], [1, 0, 0], [0, 0, 1]]),
         "cleared": np.zeros((4, 3), bool),
         "valid": np.ones((4, 3), bool),
         "left_next": np.array([[2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 1, 1]])}
    return t, np.array([3, 3, 2, 2])


def test_fill_missing_uses_group_mean_then_overall_mean():
    v = np.array([1.0, 3.0, np.nan, 10.0, np.nan])
    g = np.array([0, 0, 0, 1, 2])
    np.testing.assert_allclose(fill_missing(v, g), [1.0, 3.0, 2.0, 10.0, 14.0 / 3])     # group 2 has no finite value → overall mean (1+3+10)/3


def test_fill_missing_all_nan_is_an_error():
    with pytest.raises(ValueError):
        fill_missing(np.full(3, np.nan), np.zeros(3, int))


def test_to_grid_row_major():
    use = np.array([[True, False], [True, True]])
    np.testing.assert_array_equal(to_grid(use, np.array([5.0, 6.0, 7.0])), [[5, 0], [6, 7]])


def test_sources_mean_ignores_unplaced_copies_and_fills_holes():
    t, _ = _set()
    use = t["valid"] & t["succ"] & ~t["cleared"]                                       # 11 rankable next positions
    kp = np.tile(t["k_next"][use][:, None], (1, 4)).astype(float)                      # every copy reproduces the truth ...
    kp[0, :] = [4, 6, -1, -1]                                                          # ... except cell 0: mean of the two placed copies
    kp[1, :] = -1                                                                      # cell 1: nothing placed → mean of its group (left_next 2)
    s = next_value_sources(t, kp.astype(int))
    assert s["ceiling"][0, 0] == 5.0
    others = [s["ceiling"][0, 0], s["ceiling"][0, 2], s["ceiling"][1, 0], s["ceiling"][1, 1]]   # the other left_next-2 cells
    assert s["ceiling"][0, 1] == pytest.approx(np.mean([5.0] + others[1:]))
    assert len(s["single"]) == 4 and s["single"][0].shape == (4, 3)
    assert s["ceiling"][1, 2] == 0.0                                                   # a failed shot has no next position → 0 in the grid


def test_sources_wrong_row_count_is_an_error():
    t, _ = _set()
    with pytest.raises(ValueError):
        next_value_sources(t, np.zeros((5, 2), int))


def test_perfect_predictor_per_group_and_empty_group_is_nan():
    t, left = _set()
    nv = np.where(t["succ"], t["k_next"], 0).astype(float)
    r = metrics_by_left(t, left, nv)
    assert set(r) == {"all", "2", "3"}
    assert r["all"]["pair_acc"] == 1.0 and r["3"]["pair_acc"] == 1.0 and r["2"]["pair_acc"] == 1.0
    ts = {k: v[2:3] for k, v in t.items()}
    ts["k_next"] = np.ones((1, 3), int)                                                # all later balls tied → no pair to order
    assert np.isnan(metrics_by_left(ts, np.array([2]), nv[2:3])["2"]["pair_acc"])


def test_list_of_values_is_averaged():
    t, left = _set()
    good = np.where(t["succ"], t["k_next"], 0).astype(float)
    bad = -good
    r = metrics_by_left(t, left, [good, bad])
    assert r["all"]["pair_acc"] == pytest.approx(0.5)                                  # perfect and reversed average to ½


def test_pooled_diff():
    by = {"2": {"pair_acc": 0.8}, "3": {"pair_acc": 0.7}, "5": {"pair_acc": 0.6}, "6": {"pair_acc": 0.5}}
    assert pooled_diff(by) == pytest.approx(0.2)
    assert np.isnan(pooled_diff({"2": {"pair_acc": 0.8}}))


def test_regenerate_perturbed_is_reproducible_moves_only_the_cue_by_eps_and_keeps_unplaced_originals():
    from exp16_wm.networks import TABLE_L, TABLE_W
    raw = np.array([0.5, 0.5, 0.2, 0.8, 0.0, 0.8, 0.2, 0.0] + [0.0] * 12, np.float32)    # cue + two balls on the table, 12 pocket values (20 = 2 + 3·2 + 12)
    obs = np.stack([raw, raw])
    a, pa = regenerate_perturbed(obs, np.array([7, 9]), 0.01, 4, 0)
    b, pb = regenerate_perturbed(obs, np.array([7, 9]), 0.01, 4, 0)
    np.testing.assert_array_equal(a, b)                                                 # same key → same copies
    assert pa.all() and a.shape == (2, 4, 20)
    assert not np.allclose(a[0], a[1])                                                  # the rng key is the flat index → different positions
    d = np.hypot((a[0, :, 0] - 0.5) * TABLE_W, (a[0, :, 1] - 0.5) * TABLE_L)
    np.testing.assert_allclose(d, 0.01, atol=1e-5)                                      # moved by exactly eps
    np.testing.assert_array_equal(a[0, :, 2:], np.tile(raw[2:], (4, 1)))                # nothing but the cue ball changes


def test_regenerate_perturbed_unplaceable_copy_keeps_original_and_is_marked():
    raw = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.8, 0.2, 0.0] + [0.0] * 12, np.float32)    # a ball on the cue ball: no direction is free
    a, p = regenerate_perturbed(raw[None], np.array([0]), 0.01, 2, 0)
    assert not p.any()
    np.testing.assert_array_equal(a[0, 0], raw)


def test_perturbed_value_sources_applies_value_fn_to_each_copy_and_blanks_unplaced():
    t, _ = _set()
    N, K = t["made"].shape
    use = t["valid"] & t["succ"] & ~t["cleared"]
    M = int(use.sum())
    t["next_obs"] = np.zeros((N, K, 2), np.float32)
    obs_p = np.zeros((M, 3, 2), np.float32); obs_p[:, 1, 0] = 2.0; obs_p[:, 2, 0] = 4.0       # copy c carries the value 0, 2, 4
    placed = np.ones((M, 3), bool); placed[0, 2] = False                                       # cell 0 lost its third copy
    s = perturbed_value_sources(t, obs_p, placed, lambda tc: tc["next_obs"][:, :, 0])
    assert s["ceiling"][0, 0] == pytest.approx(1.0)                                            # mean of copies 0 and 2 → (0 + 2) / 2
    assert s["ceiling"][0, 1] == pytest.approx(2.0)                                            # mean of 0, 2, 4
    assert [g[0, 1] for g in s["single"]] == [0.0, 2.0, 4.0]
    assert s["single"][2][0, 0] == pytest.approx(4.0)                                          # the hole is filled with the balls-left group's mean of copy 2
