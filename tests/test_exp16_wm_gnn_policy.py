"""
tests/test_exp16_wm_gnn_policy.py

Unit tests for exp16_wm/gnn_policy.py — (ball, pocket)-pair GNN policy (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from exp16_wm.gnn_policy import PairGNNPolicy, N_PAIRS
from exp16_wm.obs_transform import relative_obs, REL_DIM
from exp16_wm.bc import train_bc, TransformedPolicy, evaluate_policy, val_metrics, dim_weights
from exp16_wm.teacher_data import CLEAR_ENV

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel(balls, cue=(0.5, 0.5)):
    raw = np.array(list(cue) + [v for b in balls for v in b] + POCKETS, dtype=np.float32)
    return relative_obs(raw)


REL = _rel([(0.6, 0.7, 0), (0.2, 0.3, 0), (0, 0, 1)])


class TestForward:
    def test_shapes_and_pocketed_mask(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32)
        means, logits, valid = m(torch.as_tensor(np.stack([REL, REL])))
        assert means.shape == (2, N_PAIRS, 2) and logits.shape == (2, N_PAIRS)
        assert valid[0, :12].all() and not valid[0, 12:].any()       # ball 3 pocketed → its 6 pairs masked
        assert (means.abs() <= 1).all()
        assert (logits[0, 12:] < -1e8).all()

    def test_ball_order_does_not_change_action(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32).eval()
        perm = _rel([(0, 0, 1), (0.2, 0.3, 0), (0.6, 0.7, 0)])     # same table, balls listed in another order
        a1 = m.act(REL)
        a2 = m.act(perm)
        np.testing.assert_allclose(a1, a2, atol=1e-5)

    def test_act_shapes_and_bounds(self):
        m = PairGNNPolicy(hidden=16)
        a = m.act(REL)
        assert a.shape == (2,) and -np.pi <= a[0] <= np.pi and 0.5 <= a[1] <= 8.0
        assert m.act(np.stack([REL, REL])).shape == (2, 2)


class TestLoss:
    def test_finite_with_and_without_candidates(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16)
        rel = torch.as_tensor(np.stack([REL, REL]))
        good = torch.full((2, 3, 2), float("nan"))
        good[0, 0] = torch.tensor([0.1, 0.2])                       # row 1 has no candidate → teacher target
        teacher = torch.zeros(2, 2)
        l = m.train_loss(rel, teacher, good)
        l.backward()
        assert torch.isfinite(l) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

    def test_learns_constant_target(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        n = 128
        raws = []
        for _ in range(n):
            balls = [(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0) for _ in range(3)]
            raws.append(_rel(balls))
        data = {"obs": np.stack(raws), "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
                "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
        tr = np.arange(n) < 96
        m = PairGNNPolicy(hidden=32)
        # physical-unit weights as in the CLI — in raw [-1, 1] units the angle error barely registers
        fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=60, batch=32, lr=3e-3,
                       patience=60, seed=0, log_every=0, select="loss", w=dim_weights(0.2, 0.5))
        assert fit["best_val_loss"] < fit["history"][0][2] * 0.25
        vm = val_metrics(m, data, ~tr)
        assert vm["angle_err_teacher_median_deg"] < 1.0


class TestEnv:
    def test_runs_in_env_through_relative_transform(self):
        pol = TransformedPolicy(PairGNNPolicy(hidden=16), "relative")
        out = evaluate_policy(pol, seed=0, n_games=1, env_kw={**CLEAR_ENV, "max_steps": 2})
        assert 1 <= out["mean_shots"] <= 2
        assert REL.shape == (REL_DIM,)


class TestPairSupervision:
    def _batch(self, n=6):
        rels = np.stack([REL] * n)
        good = torch.full((n, 3, 2), float("nan"))
        good[:, 0] = torch.tensor([0.05, 0.2])
        good[:, 1] = torch.tensor([-0.03, 0.1])
        return torch.as_tensor(rels), good

    def test_pair_loss_is_finite_and_has_gradients(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16)
        rel, good = self._batch()
        pairs = torch.full((6, 3), -1, dtype=torch.long)
        pairs[:, 0], pairs[:, 1] = 4, 9
        loss = m.train_loss(rel, good[:, 0], good, pairs=pairs)
        loss.backward()
        assert torch.isfinite(loss)
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)

    def test_regression_targets_the_labelled_pairs_own_output(self):
        # label asks pair 4 for (0.5, 0.5) and pair 9 for (-0.5, -0.5): after fitting, those outputs move there
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32)
        rel, _ = self._batch(8)
        good = torch.zeros(8, 2, 2)
        good[:, 0], good[:, 1] = torch.tensor([0.5, 0.5]), torch.tensor([-0.5, -0.5])
        pairs = torch.zeros(8, 2, dtype=torch.long)
        pairs[:, 0], pairs[:, 1] = 4, 9
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(300):
            loss = m.train_loss(rel, good[:, 0], good, pairs=pairs)
            opt.zero_grad(); loss.backward(); opt.step()
        means, _, _ = m(rel)
        assert torch.allclose(means[0, 4], torch.tensor([0.5, 0.5]), atol=0.05)
        assert torch.allclose(means[0, 9], torch.tensor([-0.5, -0.5]), atol=0.05)

    def test_learns_fixed_pair_selection(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32)
        rng = np.random.default_rng(0)
        rels = np.stack([_rel([(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0) for _ in range(3)]) for _ in range(64)])
        rel = torch.as_tensor(rels)
        pairs = torch.zeros(64, 1, dtype=torch.long) + 7                     # always pair 7
        good = torch.zeros(64, 1, 2)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(200):
            loss = m.train_loss(rel, good[:, 0], good, pairs=pairs)
            opt.zero_grad(); loss.backward(); opt.step()
        _, logits, _ = m(rel)
        assert (logits.argmax(dim=1) == 7).float().mean() > 0.95

    def test_rows_without_labelled_pair_fall_back_to_wta(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16)
        rel, good = self._batch(4)
        pairs = torch.tensor([[4, -1, -1], [-1, -1, -1], [9, 4, -1], [-1, -1, -1]])
        loss = m.train_loss(rel, good[:, 0], good, pairs=pairs)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(m.train_loss(rel, good[:, 0], good, pairs=torch.full((4, 3), -1)))   # all rows fall back
