"""Tests for exp16_wm.val_set (fixed validation set V0): grid, calibration, ranking metrics, candidate scores."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import relative_obs
from exp16_wm.val_set import ABS7, ANG31, ece, grid_actions, grid_scores, made_metrics
from exp16_wm.wide_scan import wrap_deg

REF = PairGNNPolicy(hidden=8)
SC, BI = REF.act_scale.numpy().astype(np.float64), REF.act_bias.numpy().astype(np.float64)


def test_grid_is_217_candidates_around_the_centre_at_absolute_speeds():
    a = grid_actions(12.0, SC, BI)
    assert a.shape == (217, 2)
    d = wrap_deg(np.degrees(a[:, 0]) - 12.0)
    np.testing.assert_allclose(np.unique(np.round(d, 3)), np.round(ANG31, 3), atol=1e-3)
    assert set(np.round(a[:, 1], 3)) == set(ABS7)


def test_ece_zero_when_calibrated_and_positive_when_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200000); y = rng.uniform(size=200000) < p
    assert ece(p, y) < 0.01
    assert ece(np.full(1000, 0.9), np.zeros(1000, bool)) == pytest.approx(0.9)


def test_made_metrics_rank_only_groups_with_both_outcomes():
    ok = np.array([[True, False, False], [False, True, False], [True, True, True], [False, False, False]])
    sc = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 9.0], [1.0, 1.0, 1.0]])
    m = made_metrics(sc, ok)
    assert m["n_groups"] == 2 and m["group_top1"] == pytest.approx(0.5)          # right in group 0, wrong in group 1
    assert 0.0 <= m["ece"] <= 1.0


def test_grid_scores_are_the_made_logits_the_policy_uses():
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True).eval()
    rng = np.random.default_rng(1)
    obs = np.zeros((3, 23), np.float32)
    obs[:, :11] = np.c_[rng.uniform(0.1, 0.9, (3, 2)), np.tile([0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0], (3, 1))]
    obs[:, 11:] = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
    v0 = {"obs": obs, "grid_state": np.array([0, 2]), "grid_pair": np.array([3, 7]),
          "grid_action": np.stack([grid_actions(5.0, SC, BI), grid_actions(-40.0, SC, BI)]), "grid_ok": np.zeros((2, 217), bool)}
    s = grid_scores(m, v0)
    with torch.no_grad():
        rel = torch.as_tensor(relative_obs(obs)); o = m.forward_full(rel)
        for g, (st, j) in enumerate(zip(v0["grid_state"], v0["grid_pair"])):
            a = (torch.as_tensor(v0["grid_action"][g, 10]) - m.act_bias) / m.act_scale
            z = m.made2_logits({k: v[st:st + 1] for k, v in o.items()}, a[None])[0, j]
            assert s[g, 10] == pytest.approx(float(z), abs=1e-5)


def test_env_actions_runs_without_grad_and_matches_act():
    from exp16_wm.val_set import env_actions
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)          # train mode: params require grad
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = "all_pairs", (-0.1, 0.0, 0.1), (1.0,), (1.0, 2.0)
    obs = np.zeros((2, 23), np.float32); obs[:, :11] = [0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0]
    obs[:, 11:] = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
    a = env_actions(m, obs)
    np.testing.assert_allclose(a, m.act(relative_obs(obs)), atol=1e-6)

