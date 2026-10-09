"""
tests/test_exp16_wm_bc.py

Unit tests for exp16_wm/bc.py — behaviour cloning of a search-planner teacher (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from exp16_wm.bc import (to_unit, bc_loss, dim_weights, TransformedPolicy, build_split, within_rate, split_by_game, angle_err_deg, nearest_good_angle_err_deg,
                         load_teacher_data, make_agent, train_bc, evaluate_policy)
from exp16_wm.train import ACT_LOW, ACT_HIGH
from exp16_wm.teacher_data import collect, save, CLEAR_ENV


class TestUnits:
    def test_to_unit_bounds(self):
        np.testing.assert_allclose(to_unit(ACT_LOW), [-1, -1])
        np.testing.assert_allclose(to_unit(ACT_HIGH), [1, 1])
        np.testing.assert_allclose(to_unit((ACT_LOW + ACT_HIGH) / 2), [0, 0], atol=1e-7)

    def test_to_unit_keeps_nan_padding(self):
        g = np.full((1, 3, 2), np.nan, dtype=np.float32)
        assert np.isnan(to_unit(g)).all()


class TestLoss:
    def test_teacher_loss_zero_at_target(self):
        t = torch.tensor([[0.1, -0.3]])
        assert float(bc_loss(t.clone(), t, torch.full((1, 2, 2), float("nan")), "teacher")) == 0.0

    def test_nearest_good_picks_closest_candidate(self):
        pred = torch.tensor([[0.0, 0.0]])
        good = torch.tensor([[[0.5, 0.0], [0.1, 0.0], [float("nan"), float("nan")]]])
        teacher = torch.tensor([[0.5, 0.0]])
        assert float(bc_loss(pred, teacher, good, "nearest_good")) == pytest.approx(0.01)
        assert float(bc_loss(pred, teacher, good, "teacher")) == pytest.approx(0.25)

    def test_nearest_good_falls_back_and_grad_is_finite(self):
        # edge case: a row with no pocketing candidate uses the teacher target, no inf/NaN gradient
        pred = torch.tensor([[0.0, 0.0], [0.2, 0.2]], requires_grad=True)
        good = torch.tensor([[[0.1, 0.0], [float("nan"), float("nan")]],
                             [[float("nan"), float("nan")], [float("nan"), float("nan")]]])
        teacher = torch.tensor([[0.3, 0.0], [0.0, 0.0]])
        l = bc_loss(pred, teacher, good, "nearest_good")
        l.backward()
        assert float(l.detach()) == pytest.approx((0.01 + 0.08) / 2)
        assert torch.isfinite(pred.grad).all()

    def test_dim_weights_physical_units(self):
        w = dim_weights(angle_scale_deg=0.2, speed_scale=0.5)
        # 1 unit of angle = 180°, 1 unit of speed = 3.75 m/s
        assert float(w[0]) == pytest.approx(900.0) and float(w[1]) == pytest.approx(7.5)

    def test_weighting_makes_nearest_mean_nearest_in_angle(self):
        # candidate A: same speed, 0.5° off; candidate B: exact angle, 2 m/s off.
        a_off = np.radians(0.5) / np.pi
        good = torch.tensor([[[a_off, 0.0], [0.0, 2.0 / 3.75]]])
        pred, teacher = torch.zeros(1, 2), torch.zeros(1, 2)
        raw = float(bc_loss(pred, teacher, good, "nearest_good"))
        weighted = float(bc_loss(pred, teacher, good, "nearest_good", dim_weights(0.2, 0.5)))
        assert raw == pytest.approx(a_off ** 2)             # unweighted: angle error looks tiny
        assert weighted == pytest.approx(min(2.5 ** 2, 4.0 ** 2))  # weighted: 0.5°→2.5, 2 m/s→4

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            bc_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 1, 2), "l1")


class TestSplitAndMetrics:
    def test_split_keeps_games_whole(self):
        game = np.repeat(np.arange(20), 3)
        tr, va = split_by_game(game, 0.2, seed=0)
        assert not set(game[tr]) & set(game[va])
        assert len(set(game[va])) == 4

    def test_single_game_goes_to_train(self):
        tr, va = split_by_game(np.zeros(5, dtype=int), 0.2, seed=0)
        assert tr.all() and not va.any()

    def test_angle_error_wraps(self):
        assert angle_err_deg(np.array([np.pi - 0.01]), np.array([-np.pi + 0.01]))[0] == pytest.approx(
            np.degrees(0.02), abs=1e-6)

    def test_nearest_good_angle_nan_when_none(self):
        good = np.full((2, 2, 2), np.nan, dtype=np.float32)
        good[0, 1] = [0.01, 3.0]
        e = nearest_good_angle_err_deg(np.array([0.0, 0.0]), good)
        assert e[0] == pytest.approx(np.degrees(0.01), rel=1e-4) and np.isnan(e[1])


class TestPipeline:
    def test_load_makes_game_ids_unique(self, tmp_path):
        for s in (11, 12):
            d = collect(seed=s, n_games=2, teacher="ghost", env_kw=CLEAR_ENV, max_good=4, log_every=0)
            save(str(tmp_path / f"g{s}.npz"), d, seed=s, teacher="ghost", env_kw=CLEAR_ENV)
        data = load_teacher_data([str(tmp_path / "g11.npz"), str(tmp_path / "g12.npz")])
        assert len(np.unique(data["game"])) == 4
        assert data["obs"].shape[0] == data["action"].shape[0] == data["good_actions"].shape[0]

    def test_train_reduces_loss_on_constant_target(self):
        rng = np.random.default_rng(0)
        n = 256
        data = {"obs": rng.uniform(0, 1, (n, 23)).astype(np.float32),
                "action": np.tile(np.array([[0.3, 4.0]], np.float32), (n, 1)),
                "good_actions": np.full((n, 2, 2), np.nan, np.float32)}
        tr = np.arange(n) < 200
        torch.manual_seed(0)              # init must not depend on RNG use by earlier tests
        agent = make_agent(3)
        fit = train_bc(agent, data, tr, ~tr, "teacher", epochs=40, batch=64, lr=1e-3,
                       patience=40, seed=0, log_every=0)
        first_val = fit["history"][0][2]
        assert fit["best_val_loss"] < first_val * 0.25

    def test_make_agent_hidden_sizes(self):
        small, big = make_agent(3), make_agent(3, [64, 64, 64])
        n = lambda a: sum(p.numel() for p in a.actor.parameters())
        assert n(small) != n(big)
        assert big.actor(torch.zeros(4, 23))[0].shape == (4, 2)

    def test_relative_policy_runs_in_env(self):
        from exp16_wm.obs_transform import REL_DIM
        pol = TransformedPolicy(make_agent(3, obs_dim=REL_DIM), "relative")
        out = evaluate_policy(pol, seed=0, n_games=1, env_kw={**CLEAR_ENV, "max_steps": 2})
        assert 1 <= out["mean_shots"] <= 2

    def test_evaluate_policy_smoke(self):
        out = evaluate_policy(make_agent(3), seed=0, n_games=1,
                              env_kw={**CLEAR_ENV, "max_steps": 3})
        assert 0.0 <= out["clear_rate"] <= 100.0 and 1 <= out["mean_shots"] <= 3


class TestBuildSplit:
    def _data(self, n_games=20, shots=3):
        n = n_games * shots
        return {"obs": np.random.default_rng(0).uniform(0, 1, (n, 23)).astype(np.float32),
                "action": np.zeros((n, 2), np.float32), "good_actions": np.full((n, 2, 2), np.nan, np.float32),
                "n_good": np.zeros(n, int), "pocketed": np.ones(n, int), "shot": np.tile(np.arange(shots), n_games),
                "game": np.repeat(np.arange(n_games), shots)}

    def test_augment_only_train_and_no_leak(self):
        d = self._data()
        plain, tr0, va0 = build_split(d, 0.2, seed=0)
        aug, tr, va = build_split(d, 0.2, seed=0, augment="flip4")
        assert tr.sum() == 4 * tr0.sum() and va.sum() == va0.sum()
        assert not set(aug["game"][tr]) & set(aug["game"][va])

    def test_data_frac_keeps_whole_games_and_full_val(self):
        d = self._data()
        _, tr_full, va_full = build_split(d, 0.2, seed=0)
        sub, tr, va = build_split(d, 0.2, seed=0, data_frac=0.25)
        assert va.sum() == va_full.sum()
        assert tr.sum() == 4 * 3                       # 16 train games × 0.25 = 4 games × 3 shots
        assert len(set(sub["game"][tr])) == 4

    def test_unknown_augment_raises(self):
        with pytest.raises(ValueError):
            build_split(self._data(), 0.2, seed=0, augment="rot90")


class TestWithinRate:
    NAN = float("nan")

    def test_exact_and_off_by_one_degree(self):
        one_deg = np.radians(1.0) / np.pi
        good = torch.tensor([[[0.1, 0.0]], [[0.1, 0.0]]])
        pred = torch.tensor([[0.1, 0.5], [0.1 + one_deg, 0.0]])
        assert within_rate(pred, good, 0.2) == 0.5

    def test_wraps_around_pi(self):
        eps = np.radians(0.05) / np.pi
        good = torch.tensor([[[1.0 - eps, 0.0]]])
        pred = torch.tensor([[-1.0 + eps, 0.0]])      # 0.1° apart across ±π
        assert within_rate(pred, good, 0.2) == 1.0

    def test_rows_without_candidates_ignored_and_all_empty_is_nan(self):
        good = torch.tensor([[[0.0, 0.0]], [[self.NAN, self.NAN]]])
        assert within_rate(torch.zeros(2, 2), good, 0.2) == 1.0
        assert np.isnan(within_rate(torch.zeros(1, 2), good[1:], 0.2))

    def test_select_within_runs_and_unknown_raises(self):
        rng = np.random.default_rng(0)
        n = 64
        data = {"obs": rng.uniform(0, 1, (n, 23)).astype(np.float32),
                "action": np.tile(np.array([[0.3, 4.0]], np.float32), (n, 1)),
                "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
        tr = np.arange(n) < 48
        torch.manual_seed(0)
        fit = train_bc(make_agent(3), data, tr, ~tr, "nearest_good", epochs=5, batch=16, lr=1e-3,
                       patience=5, seed=0, log_every=0, select="within")
        assert fit["select"] == "within" and len(fit["history"][0]) == 4
        with pytest.raises(ValueError):
            train_bc(make_agent(3), data, tr, ~tr, "teacher", epochs=1, batch=16, lr=1e-3,
                     patience=1, seed=0, log_every=0, select="median")


class TestLrSchedule:
    def _toy(self):
        rng = np.random.default_rng(0)
        n = 64
        data = {"obs": rng.uniform(0, 1, (n, 23)).astype(np.float32),
                "action": np.tile(np.array([[0.3, 4.0]], np.float32), (n, 1)),
                "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
        return data, np.arange(n) < 48

    def test_cosine_anneals_to_the_floor_and_const_stays(self):
        data, tr = self._toy()
        torch.manual_seed(0)
        fit = train_bc(make_agent(3), data, tr, ~tr, "nearest_good", epochs=10, batch=16, lr=1e-3,
                       patience=10, seed=0, log_every=0, lr_schedule="cosine", lr_final_frac=0.01)
        assert fit["final_lr"] == pytest.approx(1e-5, rel=1e-3)
        torch.manual_seed(0)
        fit = train_bc(make_agent(3), data, tr, ~tr, "nearest_good", epochs=10, batch=16, lr=1e-3,
                       patience=10, seed=0, log_every=0, lr_schedule="const")
        assert fit["final_lr"] == pytest.approx(1e-3)

    def test_unknown_schedule_raises_and_one_epoch_is_fine(self):
        data, tr = self._toy()
        with pytest.raises(ValueError):
            train_bc(make_agent(3), data, tr, ~tr, "teacher", epochs=1, batch=16, lr=1e-3,
                     patience=1, seed=0, log_every=0, lr_schedule="step")
        fit = train_bc(make_agent(3), data, tr, ~tr, "teacher", epochs=1, batch=16, lr=1e-3,
                       patience=1, seed=0, log_every=0, lr_schedule="cosine")
        assert fit["epochs_run"] == 1


class TestPairData:
    def test_use_pairs_requires_good_pairs(self):
        data = {"obs": np.zeros((4, 23), np.float32), "action": np.zeros((4, 2), np.float32),
                "good_actions": np.zeros((4, 1, 2), np.float32)}
        tr = np.array([1, 1, 1, 0], bool)
        with pytest.raises(ValueError):
            train_bc(make_agent(3), data, tr, ~tr, "nearest_good", epochs=1, batch=2, lr=1e-3,
                     patience=1, seed=0, log_every=0, use_pairs=True)

    def test_build_split_carries_pairs_only_without_augment(self):
        n = 40
        d = {"obs": np.random.default_rng(0).uniform(0, 1, (n, 23)).astype(np.float32),
             "action": np.zeros((n, 2), np.float32), "good_actions": np.full((n, 2, 2), np.nan, np.float32),
             "n_good": np.zeros(n, int), "pocketed": np.ones(n, int), "shot": np.zeros(n, int),
             "game": np.arange(n) // 2, "good_pairs": np.full((n, 2), 3, np.int8)}
        out, tr, va = build_split(d, 0.2, seed=0)
        assert "good_pairs" in out and len(out["good_pairs"]) == n
        out, tr, va = build_split(d, 0.2, seed=0, augment="flip4")
        assert "good_pairs" not in out


class TestBcLossWrap:
    def test_nearest_good_and_teacher_losses_see_the_seam_as_continuous(self):
        good = torch.tensor([[[0.9972, 0.0]]])
        teacher = torch.tensor([[0.9972, 0.0]])
        across = torch.tensor([[-0.9972, 0.0]])                  # 1° away across ±180°
        w = dim_weights(0.2, 0.5)
        for loss in ("nearest_good", "teacher"):
            l = float(bc_loss(across, teacher, good, loss, w))
            assert l < 40.0                                       # ≈ (1°/0.2°)² = 25, not (359°/0.2°)² ≈ 3e6
