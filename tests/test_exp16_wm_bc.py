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

from exp16_wm.bc import (seed_everything, to_unit, bc_loss, dim_weights, TransformedPolicy, build_split, within_rate, split_by_game, angle_err_deg, nearest_good_angle_err_deg,
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


class TestSeeding:
    def test_same_seed_gives_identical_initial_weights_and_different_seed_does_not(self):
        def weights(seed):
            seed_everything(seed)
            return torch.cat([p.flatten() for p in make_agent(3).actor.parameters()])
        assert torch.equal(weights(0), weights(0))
        assert not torch.equal(weights(0), weights(1))

    def test_gnn_init_is_seeded_too(self):
        from exp16_wm.gnn_policy import PairGNNPolicy
        def weights(seed):
            seed_everything(seed)
            return torch.cat([p.flatten() for p in PairGNNPolicy(16).parameters()])
        assert torch.equal(weights(3), weights(3))


class TestAngleWrapToggle:
    def test_toggle_switches_between_wrapped_and_raw_difference(self):
        import exp16_wm.gnn_policy as gp
        a, b = torch.tensor([[0.999, 0.0]]), torch.tensor([[-0.999, 0.0]])
        try:
            gp.ANGLE_WRAP = True
            assert abs(float(gp.action_diff(a, b)[0, 0])) < 0.01
            gp.ANGLE_WRAP = False
            assert abs(float(gp.action_diff(a, b)[0, 0])) > 1.9
        finally:
            gp.ANGLE_WRAP = True


class TestFactorizedPlumbing:
    def _data(self, n=40):
        rng = np.random.default_rng(0)
        d = {"obs": rng.uniform(0, 1, (n, 21)).astype(np.float32), "action": np.zeros((n, 2), np.float32),
             "good_actions": np.zeros((n, 2, 2), np.float32)}
        d["good_made"] = np.zeros((n, 2, 18), bool); d["good_made"][:, 0, 4] = True
        d["good_scratch"] = np.zeros((n, 2), bool)
        d["pair_n"] = np.full((n, 18), 20, np.int8); d["pair_made"] = np.zeros((n, 18), np.int8); d["pair_made"][:, 4] = 10
        d["pair_scratch"] = np.zeros((n, 18), np.int8)
        return d

    def test_missing_keys_raise_and_train_runs_with_them(self):
        from exp16_wm.gnn_policy import PairGNNPolicy
        d = self._data()
        tr = np.arange(40) < 30
        bad = {k: v for k, v in d.items() if k != "pair_made"}
        with pytest.raises(ValueError):
            train_bc(PairGNNPolicy(16, factorized=True), bad, tr, ~tr, "nearest_good", epochs=1, batch=8, lr=1e-3,
                     patience=1, seed=0, log_every=0, selection="factorized")
        fit = train_bc(PairGNNPolicy(16, factorized=True), d, tr, ~tr, "nearest_good", epochs=2, batch=8, lr=1e-3,
                       patience=2, seed=0, log_every=0, selection="factorized")
        assert fit["epochs_run"] == 2

    def test_unknown_selection_raises(self):
        d = self._data()
        tr = np.arange(40) < 30
        with pytest.raises(ValueError):
            train_bc(make_agent(3), d, tr, ~tr, "teacher", epochs=1, batch=8, lr=1e-3, patience=1, seed=0,
                     log_every=0, selection="local")


def test_oracle_pair_metrics_measure_the_made_pairs_output():
    from exp16_wm.bc import oracle_pair_metrics, to_unit
    from exp16_wm.gnn_policy import PairGNNPolicy
    from exp16_wm.train import ACT_LOW, ACT_HIGH
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16)
    N = 5
    obs = np.random.default_rng(0).uniform(0.1, 0.9, (N, 21)).astype(np.float32)
    obs[:, 2::3][:, :3] = 0.0
    with torch.no_grad():
        means = m.forward_full(torch.as_tensor(obs))["means"].numpy()
    unit = means[:, 4].copy(); unit[:, 0] += 0.1 / 180.0               # label = pair 4's output + 0.1°
    good = np.full((N, 3, 2), np.nan, dtype=np.float32)
    good[:, 0] = unit * (ACT_HIGH - ACT_LOW) / 2.0 + (ACT_HIGH + ACT_LOW) / 2.0
    gm = np.zeros((N, 2, 18), dtype=np.float32); gm[:, 0, 4] = 1
    data = {"obs": obs, "good_actions": good, "good_made": gm, "good_scratch": np.zeros((N, 2), dtype=np.float32),
            "pair_made": np.zeros((N, 18), dtype=np.int64)}
    out = oracle_pair_metrics(m, data, np.ones(N, dtype=bool))
    assert abs(out["oracle_angle_err_median_deg"] - 0.1) < 1e-3 and out["oracle_angle_within_0.2deg"] == 1.0
    assert out["top1_pair_made_rate"] == 0.0
    assert oracle_pair_metrics(m, {"obs": obs}, np.ones(N, dtype=bool)) == {}


def test_append_train_only_keeps_val_and_unique_games():
    from exp16_wm.bc import append_train_only
    data = {"obs": np.zeros((5, 3), np.float32), "game": np.array([0, 0, 1, 2, 2]), "shot": np.arange(5)}
    train, val = np.array([1, 1, 0, 1, 1], bool), np.array([0, 0, 1, 0, 0], bool)
    extra = {"obs": np.ones((3, 3), np.float32), "game": np.array([0, 1, 2]), "shot": np.zeros(3, np.int64), "other": np.zeros(3)}
    out, tr, va = append_train_only(data, train, val, extra)
    assert len(out["obs"]) == 8 and set(out) == set(data)
    assert np.array_equal(va, np.r_[val, [False] * 3]) and tr[-3:].all()
    assert set(out["game"][-3:]).isdisjoint(set(data["game"]))
    with pytest.raises(ValueError):
        append_train_only(data, train, val, {"obs": extra["obs"]})


def test_split_rows_maps_original_rows_to_build_split_order():
    from exp16_wm.bc import split_rows
    rng = np.random.default_rng(0)
    n = 40
    data = {k: np.zeros((n,) + s, np.float32) for k, s in (("action", (2,)), ("good_actions", (2, 2)))}
    data.update(obs=rng.normal(size=(n, 23)).astype(np.float32), n_good=np.zeros(n, np.int32), pocketed=np.zeros(n, np.int32),
                shot=np.zeros(n, np.int32), game=np.repeat(np.arange(10), 4))
    comb, tr, va = build_split(dict(data), 0.2, 3)
    new = split_rows(data["game"], 0.2, 3)
    assert np.array_equal(comb["obs"][new], data["obs"])


def test_load_sims_aligns_rows_across_files(tmp_path):
    from exp16_wm.bc import load_sims
    labels, sims = [], []
    for i, n in enumerate((3, 2)):
        obs = np.full((n, 23), i, np.float32) + np.arange(n)[:, None]
        lp, sp = tmp_path / "lab" / f"f_shard{i}.npz", tmp_path / "sims" / f"f_shard{i}.npz"
        lp.parent.mkdir(exist_ok=True); sp.parent.mkdir(exist_ok=True)
        np.savez(lp, obs=obs)
        np.savez(sp, obs=obs, sim_state=np.array([0, n - 1]), sim_action=np.zeros((2, 2), np.float32),
                 sim_made=np.array([1, 2], np.int32), sim_scratch_replay=np.array([False, True]))
        labels.append(str(lp)); sims.append(str(sp))
    s = load_sims(sims, labels)
    assert np.array_equal(s["row"], [0, 2, 3, 4]) and s["made"].tolist() == [1, 2, 1, 2]
    np.savez(tmp_path / "sims" / "f_shard1.npz", obs=np.zeros((2, 23), np.float32), sim_state=np.array([0]),
             sim_action=np.zeros((1, 2), np.float32), sim_made=np.array([1], np.int32), sim_scratch_replay=np.array([False]))
    with pytest.raises(ValueError):
        load_sims(sims, labels)                                              # states differ from the label file
