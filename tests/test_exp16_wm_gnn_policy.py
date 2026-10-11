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

from exp16_wm.gnn_policy import PairGNNPolicy, N_PAIRS, action_diff, wrap_angle_unit
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


class TestAngleWrap:
    def test_wrap_angle_unit(self):
        x = torch.tensor([0.0, 0.5, 1.5, -1.5, 1.998, -1.998, 2.0])
        w = wrap_angle_unit(x)
        assert torch.allclose(w, torch.tensor([0.0, 0.5, -0.5, 0.5, -0.002, 0.002, 0.0]), atol=1e-5)
        assert (wrap_angle_unit(torch.linspace(-5, 5, 101)).abs() <= 1.0 + 1e-6).all()

    def test_action_diff_takes_the_short_way_only_for_the_angle(self):
        a, b = torch.tensor([[0.999, 0.3]]), torch.tensor([[-0.999, 0.1]])
        d = action_diff(a, b)
        assert abs(float(d[0, 0]) - (-0.002)) < 1e-5 and abs(float(d[0, 1]) - 0.2) < 1e-6

    def test_pair_loss_regression_term_is_small_just_across_the_seam(self):
        # label at +179.5° (unit 0.9972). Fit one pair's output to -179.5° (1° across the seam): the
        # regression term must be ≈ (1°/0.2°)² = 25, not (359°/0.2°)² ≈ 3e6.
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16)
        m.sel_weight = 0.0
        rel = torch.as_tensor(np.stack([REL] * 2))
        good = torch.zeros(2, 1, 2)
        good[:, 0] = torch.tensor([0.9972, 0.0])
        pairs = torch.full((2, 1), 4, dtype=torch.long)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        target = torch.tensor([-0.9972, 0.0])
        for _ in range(300):                                   # drive the pair-4 output to the other side of the seam
            means, _, _ = m(rel)
            l = ((means[:, 4] - target) ** 2).sum()
            opt.zero_grad(); l.backward(); opt.step()
        from exp16_wm.bc import dim_weights
        loss = float(m.train_loss(rel, good[:, 0], good, w=dim_weights(0.2, 0.5), pairs=pairs))
        assert loss < 100.0


class TestFactorized:
    def _extras(self, B, pair_made_slots=(7,), made_slot=7, scratch_slots=()):
        pm = torch.zeros(B, 18, dtype=torch.int64)
        pn = torch.zeros(B, 18, dtype=torch.int64)
        ps = torch.zeros(B, 18, dtype=torch.int64)
        for sl in range(12):                                       # balls 1-2 were tried (ball 3 pocketed in REL)
            pn[:, sl] = 20
        for sl in pair_made_slots:
            pm[:, sl] = 12
        for sl in scratch_slots:
            ps[:, sl] = 20
        gm = torch.zeros(B, 2, 18); gm[:, 0, made_slot] = 1.0
        return {"good_made": gm, "good_scratch": torch.zeros(B, 2), "pair_n": pn, "pair_made": pm, "pair_scratch": ps}

    def test_joint_logprob_is_a_distribution_over_the_live_pairs(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, factorized=True)
        means, logits, valid = m(torch.as_tensor(np.stack([REL, REL])))
        p = torch.exp(logits).sum(dim=1)
        assert torch.allclose(p, torch.ones(2), atol=1e-5)
        assert (logits[:, 12:] < -1e8).all()                       # ball 3 is pocketed in REL

    def test_loss_needs_extras_and_is_differentiable(self):
        import pytest
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, factorized=True)
        rel = torch.as_tensor(np.stack([REL] * 4))
        good = torch.zeros(4, 3, 2); good[:, 1:] = float("nan")
        with pytest.raises(ValueError):
            m.train_loss(rel, good[:, 0], good)
        loss = m.train_loss(rel, good[:, 0], good, extras=self._extras(4))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

    def test_learns_verified_pair_selection_and_action_for_that_pair(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32, factorized=True)
        rel = torch.as_tensor(np.stack([REL] * 16))
        good = torch.full((16, 2, 2), float("nan")); good[:, 0] = torch.tensor([0.3, -0.5])
        ex = self._extras(16, pair_made_slots=(7,), made_slot=7)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(300):
            loss = m.train_loss(rel, good[:, 0], good, extras=ex)
            opt.zero_grad(); loss.backward(); opt.step()
        means, logits, _ = m(rel)
        assert int(logits.argmax(dim=1)[0]) == 7
        assert torch.allclose(means[0, 7], torch.tensor([0.3, -0.5]), atol=0.05)

    def test_target_mass_follows_made_counts_across_pockets(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32, factorized=True)
        rel = torch.as_tensor(np.stack([REL] * 16))
        good = torch.full((16, 2, 2), float("nan")); good[:, 0] = torch.tensor([0.0, 0.0])
        ex = self._extras(16, pair_made_slots=(6, 8), made_slot=6)
        ex["pair_made"][:, 6], ex["pair_made"][:, 8] = 15, 5          # ball 2: pocket 0 three times as good as pocket 2
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(400):
            loss = m.train_loss(rel, good[:, 0], good, extras=ex)
            opt.zero_grad(); loss.backward(); opt.step()
        o = m.forward_full(rel[:1])
        p = torch.softmax(o["pocket_logits"][0, 1], dim=0)             # ball 2's pockets
        assert abs(float(p[0]) - 0.75) < 0.05 and abs(float(p[2]) - 0.25) < 0.05

    def test_scratch_head_learns_and_penalty_steers_the_choice(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32, factorized=True)
        rel = torch.as_tensor(np.stack([REL] * 16))
        good = torch.full((16, 2, 2), float("nan")); good[:, 0] = torch.tensor([0.0, 0.0])
        ex = self._extras(16, pair_made_slots=(7, 8), made_slot=7, scratch_slots=(7,))
        ex["pair_made"][:, 7], ex["pair_made"][:, 8] = 12, 11          # slot 7 slightly preferred, but it scratches
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(400):
            loss = m.train_loss(rel, good[:, 0], good, extras=ex)
            opt.zero_grad(); loss.backward(); opt.step()
        o = m.forward_full(rel[:1])
        assert float(torch.sigmoid(o["scratch_logit"][0, 7])) > 0.9 and float(torch.sigmoid(o["scratch_logit"][0, 8])) < 0.1
        m.scratch_lambda = 0.0
        a0 = m.predict_unit(rel[:1])
        m.scratch_lambda = 5.0
        picked = int(torch.argmax(o["logits"][0] - 5.0 * torch.sigmoid(o["scratch_logit"][0])))
        assert picked == 8 and a0.shape == (1, 2)


class TestFactTerms:
    def test_terms_switch_the_loss_pieces_on_and_off(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, factorized=True)
        rel = torch.as_tensor(np.stack([REL] * 4))
        good = torch.zeros(4, 3, 2); good[:, 1:] = float("nan")
        h = TestFactorized()._extras(4)
        vals = {}
        for terms in (("reg",), ("sel",), ("scratch",), ("reg", "sel", "scratch")):
            m.fact_terms = terms
            vals[terms] = float(m.train_loss(rel, good[:, 0], good, extras=h).detach())
        assert vals[("reg",)] > 0 and vals[("sel",)] > 0 and vals[("scratch",)] > 0
        parts = vals[("reg",)] + vals[("sel",)] + vals[("scratch",)]
        assert abs(vals[("reg", "sel", "scratch")] - parts) < 1e-3 * max(1.0, parts)


class TestModularLoss:
    """--sel-target / --reg-target / --scratch-head: the factorized experiment's changes switched on one at a time."""

    def _setup(self, B=4, factorized=False, scratch=False, mod=("label", "angle", False)):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, factorized=factorized, scratch_head=scratch)
        m.mod = mod
        rel = torch.as_tensor(np.stack([REL] * B))
        good = torch.full((B, 2, 2), float("nan")); good[:, 0] = torch.tensor([0.3, -0.5])
        pairs = torch.full((B, 2), -1, dtype=torch.long); pairs[:, 0] = 7
        return m, rel, good, pairs, TestFactorized()._extras(B, pair_made_slots=(7, 8), made_slot=7)

    def test_label_angle_global_equals_pair_loss(self):
        m, rel, good, pairs, ex = self._setup()
        pairs[:, 1] = 9; good[:, 1] = torch.tensor([0.1, 0.2])
        w = torch.tensor([3.0, 2.0])
        mod = m.train_loss(rel, good[:, 0], good, w=w, pairs=pairs, extras=ex)
        m.mod = None
        ref = m.train_loss(rel, good[:, 0], good, w=w, pairs=pairs)
        assert torch.allclose(mod, ref, rtol=1e-5)

    def test_every_combination_is_finite_and_differentiable(self):
        for fact in (False, True):
            for sel in ("label", "made"):
                for reg in ("angle", "made"):
                    for sc in (False, True):
                        m, rel, good, pairs, ex = self._setup(factorized=fact, scratch=sc, mod=(sel, reg, sc))
                        loss = m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)
                        loss.backward()
                        assert torch.isfinite(loss), (fact, sel, reg, sc)
                        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

    def test_rows_without_targets_give_zero_loss_not_nan(self):
        m, rel, good, pairs, ex = self._setup(mod=("made", "made", False))
        ex["pair_made"][:] = 0
        ex["good_made"][:] = 0
        loss = m.train_loss(rel, good[:, 0], good, pairs=torch.full((4, 2), -1), extras=ex)
        assert float(loss) == 0.0

    def test_missing_inputs_raise(self):
        m, rel, good, pairs, ex = self._setup(mod=("label", "made", False))
        with pytest.raises(ValueError):
            m.train_loss(rel, good[:, 0], good, pairs=None, extras=ex)
        with pytest.raises(ValueError):
            m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=None)
        m.mod = ("label", "angle", True)                       # scratch term without a scratch head
        with pytest.raises(ValueError):
            m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)

    def test_made_regression_trains_the_made_pair_not_the_angle_matched_one(self):
        m, rel, good, pairs, ex = self._setup(B=16, mod=("label", "made", False))
        ex["good_made"][:] = 0; ex["good_made"][:, 0, 8] = 1.0     # label action really made pair 8, angle matching said 7
        m.sel_weight = 0.0
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(300):
            loss = m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)
            opt.zero_grad(); loss.backward(); opt.step()
        means, _, _ = m(rel[:1])
        assert torch.allclose(means[0, 8], torch.tensor([0.3, -0.5]), atol=0.05)
        assert not torch.allclose(means[0, 7], torch.tensor([0.3, -0.5]), atol=0.05)

    def test_made_selection_matches_made_proportions_in_both_structures(self):
        for fact in (False, True):
            m, rel, good, pairs, ex = self._setup(B=16, factorized=fact, mod=("made", "angle", False))
            ex["pair_made"][:, 7], ex["pair_made"][:, 8] = 15, 5
            opt = torch.optim.Adam(m.parameters(), lr=5e-3)
            for _ in range(400):
                loss = m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)
                opt.zero_grad(); loss.backward(); opt.step()
            _, logits, _ = m(rel[:1])
            p = torch.softmax(logits[0], dim=0)
            assert abs(float(p[7]) - 0.75) < 0.05 and abs(float(p[8]) - 0.25) < 0.05, fact

    def test_scratch_head_in_global_structure(self):
        m, rel, _, _, _ = self._setup(scratch=True)
        o = m.forward_full(rel)
        assert o["scratch_logit"].shape == (4, N_PAIRS)
        assert "scratch_logit" not in PairGNNPolicy(hidden=16).forward_full(rel)

    def test_made_same_drops_combination_shot_targets(self):
        # candidate 0 aimed at pair 7 (ball 2) but made pair 2 (ball 1) and pair 8 (ball 2): only pair 8 is a target
        m, rel, good, pairs, ex = self._setup(mod=("label", "made_same", False))
        ex["good_made"][:] = 0; ex["good_made"][:, 0, 2] = 1.0; ex["good_made"][:, 0, 8] = 1.0
        m.sel_weight = 0.0
        same = m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)
        ex["good_made"][:, 0, 2] = 0.0
        m.mod = ("label", "made", False)
        only8 = m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)
        assert torch.allclose(same, only8)

    def test_made_same_without_angle_pair_has_no_target(self):
        m, rel, good, pairs, ex = self._setup(mod=("label", "made_same", False))
        m.sel_weight = 0.0
        loss = m.train_loss(rel, good[:, 0], good, pairs=torch.full((4, 2), -1), extras=ex)
        assert float(loss) == 0.0

    def test_made_head_learns_the_pair_made_rate(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=32, made_head=True)
        m.mod = ("label", "angle", False)
        rel = torch.as_tensor(np.stack([REL] * 16))
        good = torch.full((16, 2, 2), float("nan")); good[:, 0] = torch.tensor([0.3, -0.5])
        pairs = torch.full((16, 2), -1, dtype=torch.long); pairs[:, 0] = 7
        ex = TestFactorized()._extras(16, pair_made_slots=(7,), made_slot=7)
        ex["pair_made"][:, 7], ex["pair_made"][:, 8] = 15, 5          # out of 20 tries → 0.75, 0.25
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(400):
            loss = m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)
            opt.zero_grad(); loss.backward(); opt.step()
        p = torch.sigmoid(m.forward_full(rel[:1])["made_logit"][0])
        assert abs(float(p[7]) - 0.75) < 0.05 and abs(float(p[8]) - 0.25) < 0.05 and float(p[0]) < 0.05

    def test_made_head_adds_to_the_loss_and_needs_extras(self):
        m, rel, good, pairs, ex = self._setup()
        base = float(m.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex))
        torch.manual_seed(0)
        mh = PairGNNPolicy(hidden=16, made_head=True); mh.mod = ("label", "angle", False)
        assert float(mh.train_loss(rel, good[:, 0], good, pairs=pairs, extras=ex)) > base
        with pytest.raises(ValueError):
            mh.train_loss(rel, good[:, 0], good, pairs=pairs, extras=None)
        assert "made_logit" not in m.forward_full(rel)


class TestWtaFallbackSwitch:
    def test_without_fallback_unlabelled_rows_are_left_out(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16)
        rel, good = TestPairSupervision()._batch(4)
        pairs = torch.tensor([[4, -1, -1], [-1, -1, -1], [9, 4, -1], [-1, -1, -1]])
        m.wta_fallback = False
        loss = m.train_loss(rel, good[:, 0], good, pairs=pairs)
        lab = torch.tensor([0, 2])
        assert torch.allclose(loss, m._pair_loss(rel[lab], good[lab], pairs[lab], None))
        m.wta_fallback = True
        assert not torch.allclose(m.train_loss(rel, good[:, 0], good, pairs=pairs), loss)

    def test_without_fallback_no_labelled_rows_gives_zero(self):
        m = PairGNNPolicy(hidden=16)
        m.wta_fallback = False
        rel, good = TestPairSupervision()._batch(4)
        assert float(m.train_loss(rel, good[:, 0], good, pairs=torch.full((4, 3), -1))) == 0.0


class TestMade2Head:
    def _m(self):
        torch.manual_seed(0)
        return PairGNNPolicy(hidden=16, made2_head=True), torch.as_tensor(np.stack([REL] * 3))

    def test_logits_shapes_and_pocketed_pairs_masked(self):
        m, rel = self._m()
        o = m.forward_full(rel)
        z1 = m.made2_logits(o, torch.zeros(3, 2))
        z2 = m.made2_logits(o, torch.zeros(3, N_PAIRS, 2))
        assert z1.shape == z2.shape == (3, N_PAIRS) and torch.allclose(z1, z2)
        assert (z1[:, 12:] < -1e8).all()                     # ball 3 pocketed in REL

    def test_loss_finite_with_gradient_and_learns_an_action_window(self):
        # pair 7 is made only within ±0.1° of its own proposal: the head must resolve 0.1°
        m, _ = self._m()
        rel = torch.as_tensor(np.stack([REL] * 64))
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(0)
        for _ in range(1200):
            with torch.no_grad():
                prop = m.forward_full(rel)["means"][:, 7]
            off = (torch.rand(64, generator=g) - 0.5) * (1.0 / 180.0)            # ±0.5°
            a = prop.clone(); a[:, 0] += off
            made = torch.zeros(64, N_PAIRS, dtype=torch.bool)
            made[:, 7] = off.abs() < 0.1 / 180.0
            loss = m.made2_loss(rel, a, made, torch.zeros(64, dtype=torch.bool))
            opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss)
        o = m.forward_full(rel[:2])
        prop = o["means"][:, 7].detach()
        a = prop.clone(); a[1, 0] += 0.3 / 180.0
        p = torch.sigmoid(m.made2_logits(o, a))
        assert float(p[0, 7]) > 0.7 and float(p[1, 7]) < 0.3 and float(p[0, 3]) < 0.1

    def test_refinement_grid_keeps_the_proposal_and_scales_speed_in_mps(self):
        m, _ = self._m()
        a = torch.tensor([[0.1, 0.0]])
        c = m._unit_offsets(a)
        assert c.shape == (1, len(m.refine_deg) * len(m.refine_speed), 2)
        assert (c - a).abs().sum(-1).min() < 1e-6
        v = c[0, :, 1] * m.act_scale[1] + m.act_bias[1]
        v0 = a[0, 1] * m.act_scale[1] + m.act_bias[1]
        assert torch.allclose(torch.unique(torch.round(v / v0 * 100) / 100), torch.tensor([0.8, 1.0, 1.25]))

    def test_absolute_speed_grid_ignores_the_proposals_speed(self):
        m, _ = self._m()
        m.refine_speed_abs = (1.0, 2.0, 3.0)
        c = m._unit_offsets(torch.tensor([[0.1, 0.0], [0.1, 0.5]]))
        v = c[..., 1] * m.act_scale[1] + m.act_bias[1]
        assert torch.allclose(torch.unique(torch.round(v * 100) / 100), torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(c[0], c[1])

    def test_infer_modes(self, monkeypatch):
        m, rel = self._m()
        o = m.forward_full(rel)
        sel = int(o["logits"][0].argmax())
        other = 3 if sel != 3 else 4
        fake = torch.full((3, N_PAIRS), -20.0); fake[:, other] = 5.0          # verifier: only `other` can be made
        monkeypatch.setattr(m, "made2_logits", lambda o, a: fake.masked_fill(~o["valid"], -1e9))
        m.infer = "verify"
        assert torch.allclose(m.predict_unit(rel)[0], o["means"][0, other])
        m.infer = "select"
        assert torch.allclose(m.predict_unit(rel)[0], o["means"][0, sel])
        with pytest.raises(ValueError):
            p = PairGNNPolicy(hidden=16); p.infer = "refine"; p.predict_unit(rel)

    def test_refine_returns_the_candidate_the_head_scores_highest(self):
        m, rel = self._m()
        target = m.predict_unit(rel)[:, 0] + 0.1 / 180.0                       # +0.1° from the proposal
        class Head(torch.nn.Module):
            def forward(self, x):                                              # x = [e (16), a (2), (a − a_k)·scale (2)]
                return -((x[..., 16] - target[:, None]) ** 2)[..., None]
        m.head_made2 = Head()
        m.infer = "refine"
        out = m.predict_unit(rel)
        assert torch.allclose(out[:, 0], target, atol=1e-6)

    def test_checkpoints_with_or_without_made2_scale_load(self):
        m, _ = self._m()
        sd = m.state_dict()
        assert "made2_scale" not in sd
        PairGNNPolicy(hidden=16, made2_head=True).load_state_dict({**sd, "made2_scale": torch.ones(2)})
        old = {k: v for k, v in PairGNNPolicy(hidden=16).state_dict().items()}
        PairGNNPolicy(hidden=16).load_state_dict(old)


class TestGeomAngle:
    def _two(self, move_other):
        # cue at origin, ball 0 at (0.5, 0), ball 1 elsewhere, ball 2 pocketed
        def rel(b1):
            r = REL.copy()
            r[0:3] = (0.5, 0.0, 0.0); r[3:6] = (b1[0], b1[1], 0.0); r[6:9] = (0.0, 0.0, 1.0)
            return r
        return torch.as_tensor(np.stack([rel((0.3, 0.6)), rel(move_other)]))

    def test_angle_ignores_the_other_ball_but_speed_and_selection_do_not(self):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, geom_angle=True)
        o = m.forward_full(self._two((0.9, -0.5)))
        assert torch.allclose(o["means"][0, :6, 0], o["means"][1, :6, 0], atol=1e-6)        # ball 0's pairs: same angle
        assert not torch.allclose(o["means"][0, :6, 1], o["means"][1, :6, 1], atol=1e-6)    # speed sees the neighbour
        assert not torch.allclose(o["logits"][0, :6], o["logits"][1, :6], atol=1e-6)
        base = PairGNNPolicy(hidden=16)
        ob = base.forward_full(self._two((0.9, -0.5)))
        assert not torch.allclose(ob["means"][0, :6, 0], ob["means"][1, :6, 0], atol=1e-6)  # without the flag it does

    def test_default_model_is_unchanged_and_geom_model_trains(self):
        torch.manual_seed(0); a = PairGNNPolicy(hidden=16)
        torch.manual_seed(0); b = PairGNNPolicy(hidden=16, geom_angle=False)
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, geom_angle=True)
        rel = torch.as_tensor(np.stack([REL] * 4))
        good = torch.zeros(4, 2, 2); good[:, 0] = torch.tensor([0.3, -0.5])
        pairs = torch.full((4, 2), -1, dtype=torch.long); pairs[:, 0] = 7
        loss = m.train_loss(rel, good[:, 0], good, pairs=pairs)
        loss.backward()
        assert torch.isfinite(loss) and m.head_ang.weight.grad.abs().sum() > 0
        assert m.predict_unit(rel).shape == (4, 2)
