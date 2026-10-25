"""Tests for exp16_wm.goal_gnn (A0: the pair structure of the pocket GNN policy with arbitrary goal points)."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.goal_gnn import GoalPairGNN, MADE_TOL, auc, build_pairs, propose_actions, refine_actions, train


def _model(h=16, layers=2):
    m = GoalPairGNN(h, layers); m.eval(); return m


def test_any_number_of_goals_share_one_ball_embedding():
    m = _model()
    ball = torch.tensor([[0.3, 0.5], [0.6, 1.4]]); goals = torch.rand(2, 6, 2)
    e = m.encode(ball, goals)
    assert e.shape == (2, 6, 16) and m.propose(e).shape == (2, 6, 3)
    one = m.encode(ball, goals[:, :1])
    assert torch.allclose(one[:, 0], e[:, 0], atol=1e-5)                       # a goal's embedding does not depend on the other goals
    assert m.encode(ball, goals[:, :3]).shape == (2, 3, 16)


def test_made2_input_difference_is_zero_for_the_proposal_itself():
    m = _model()
    e = torch.randn(4, 16); prop = torch.randn(4, 3)
    same = m.made2_logit(e, prop, prop)
    shifted = prop.clone(); shifted[:, 2] += 1.0
    assert same.shape == (4,) and not torch.allclose(same, m.made2_logit(e, shifted, prop))


def _data(n_starts=2, per=400, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform([0.2, 0.3], [0.8, 1.7], (n_starts, 2)).astype(np.float32)
    n = n_starts * per
    return {"starts": starts, "sidx": np.repeat(np.arange(n_starts), per), "ok": np.ones(n, bool),
            "end": rng.uniform([0.05, 0.05], [0.9, 1.9], (n, 2)).astype(np.float32), "phi": rng.uniform(0, 360, n).astype(np.float32),
            "v": rng.uniform(0.05, 0.5, n).astype(np.float32)}


def test_build_pairs_labels_follow_the_end_positions():
    data = _data()
    rows = build_pairs(data, np.array([0, 1]), 600, seed=1)
    n = len(rows["label"])
    assert n == 600 and all(len(v) == n for v in rows.values())
    pool_end = {s: data["end"][data["sidx"] == s] for s in (0, 1)}
    for k in range(0, n, 37):                                                     # recompute a few labels from the recorded shots
        s = 0 if k < 300 else 1
        j_end = pool_end[s][np.argmin(np.abs(data["phi"][data["sidx"] == s] - rows["phi_j"][k]) + np.abs(data["v"][data["sidx"] == s] - rows["v_j"][k]))]
        assert bool(rows["label"][k]) == bool(np.linalg.norm(j_end - rows["goal"][k]) <= MADE_TOL)
    assert rows["label"].any() and not rows["label"].all()                        # both classes present


def test_auc_perfect_random_and_ties():
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    rng = np.random.default_rng(0)
    assert abs(auc(rng.random(4000), rng.random(4000) < 0.4) - 0.5) < 0.03
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_refine_plays_the_action_made2_scores_highest():
    class Peaked(GoalPairGNN):
        def made2_logit(self, e, action, prop):
            ang = torch.rad2deg(torch.atan2(action[..., 1], action[..., 0]) - torch.atan2(prop[..., 1], prop[..., 0]))
            ang = (ang + 180) % 360 - 180
            return -((ang - 0.1) / 0.02) ** 2 - ((action[..., 2] - prop[..., 2]) / 0.005) ** 2        # best: +0.1 deg, same speed
    m = Peaked(16, 2); m.eval()
    start = np.array([[0.3, 0.5], [0.6, 1.2]], np.float32); goal = np.array([[0.7, 1.5], [0.2, 0.3]], np.float32)
    pphi, pv = propose_actions(m, start, goal)
    rphi, rv = refine_actions(m, start, goal)
    assert np.allclose(((rphi - pphi + 180) % 360) - 180, 0.1, atol=1e-3) and np.allclose(rv / pv, 1.0, atol=2e-3)


def test_training_lowers_both_losses_on_a_learnable_toy():
    rng = np.random.default_rng(3); n = 3000
    start = rng.uniform([0.1, 0.1], [0.8, 1.8], (n, 2)).astype(np.float32); goal = rng.uniform([0.1, 0.1], [0.8, 1.8], (n, 2)).astype(np.float32)
    d = goal - start
    phi = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0; v = np.sqrt(np.linalg.norm(d, axis=1) / 7.76) + 0.05
    phi_j = (phi + rng.choice([0.0, 0.05, 3.0], n)) % 360.0
    rows = {"start": start, "goal": goal, "phi_i": phi.astype(np.float32), "v_i": v.astype(np.float32), "phi_j": phi_j.astype(np.float32),
            "v_j": v.astype(np.float32), "label": np.abs(((phi_j - phi + 180) % 360) - 180) < 0.1}
    logs: list[str] = []
    model = train(rows, epochs=6, batch=256, lr=2e-3, hidden=32, enc_layers=2, made_weight=5.0, seed=0, log=logs.append)
    pick = lambda line, key: float(line.split(key)[1].split()[0])
    assert pick(logs[-1], "proposal loss ") < pick(logs[0], "proposal loss ") and pick(logs[-1], "made2 bce ") < pick(logs[0], "made2 bce ")
    ph, vv = propose_actions(model.eval(), start[:8], goal[:8])
    assert ph.shape == (8,) and (vv > 0).all()
