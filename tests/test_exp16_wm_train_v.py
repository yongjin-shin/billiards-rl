"""Tests for exp16_wm.train_v: V2 labels, data split, only the V modules learn."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.train_v import V_MODULES, load_v_data, metrics, targets, v_logits

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
OBS = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0, *POCKETS], np.float32)


def test_targets_are_fractions_clipped_to_one():
    np.testing.assert_allclose(targets(np.array([0, 2, 3, 4]), np.array([3, 4, 3, 3])), [0.0, 0.5, 1.0, 1.0])


def test_load_v_data_splits_whole_games_and_drops_empty_tables(tmp_path):
    n = 40
    np.savez(tmp_path / "a.npz", obs=np.tile(OBS, (n, 1)), k=np.ones(n, int), left=np.r_[np.zeros(4, int), np.full(n - 4, 3)],
             game=np.repeat(np.arange(10), 4), branch=np.zeros(n, bool))
    d = load_v_data([str(tmp_path / "a.npz")], val_frac=0.2)
    assert len(d["k"]) == n - 4 and (d["left"] > 0).all()
    assert d["val"].sum() == 8                                      # 2 of the 10 games, whole
    with pytest.raises(FileNotFoundError):
        load_v_data([])


def test_metrics_beat_the_balls_left_baseline_when_the_prediction_is_right():
    k = np.array([3, 0, 2, 1, 1, 0]); left = np.array([3, 3, 2, 2, 1, 1])
    m = metrics(k.astype(float), k, left)
    assert m["auc_clear"] == pytest.approx(1.0) and m["mae_balls"] == 0.0
    assert m["auc_clear"] > m["auc_clear_left_baseline"]


def test_only_v_modules_get_gradients():
    from exp16_wm.gnn_policy import PairGNNPolicy
    m = PairGNNPolicy(hidden=8, made2_head=True, v_head=True, qv_encoder=True)
    for name, prm in m.named_parameters():
        prm.requires_grad = name.split(".")[0] in V_MODULES
    from exp16_wm.obs_transform import relative_obs
    v_logits(m, torch.as_tensor(relative_obs(OBS[None]))).sum().backward()
    got = {n.split(".")[0] for n, prm in m.named_parameters() if prm.grad is not None and prm.grad.abs().sum() > 0}
    assert got and got <= set(V_MODULES)


def test_horizon_caps_the_target_and_the_prediction_scale():
    from exp16_wm.train_v import cap
    k, left = np.array([0, 1, 3, 5]), np.array([6, 6, 6, 1])
    np.testing.assert_array_equal(cap(left, 2), [2, 2, 2, 1])
    np.testing.assert_allclose(targets(k, left, 2), [0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(targets(k, left, 0), [0.0, 1 / 6, 0.5, 1.0])


def test_within_left_spearman_is_zero_signal_for_the_balls_left_baseline_and_one_for_a_perfect_v():
    left = np.repeat([3, 6], 20)
    k = np.r_[np.arange(20) % 4, np.arange(20) % 7]
    m = metrics(k.astype(float), k, left)
    assert m["spearman_within_left"] == pytest.approx(1.0)
    assert np.isnan(metrics(left.astype(float), k, left)["spearman_within_left"])   # constant within each group: no signal
