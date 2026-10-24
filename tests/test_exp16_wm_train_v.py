"""Tests for exp16_wm.train_v: V2 labels, data split, only the V modules learn."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.train_v import V_MODULES, hazard_nll, hazard_targets, hazard_value, load_smooth_k, load_v_data, metrics, targets, v_logits

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


def test_subsample_games_keeps_whole_games_and_all_with_frac_one():
    from exp16_wm.train_v import subsample_games
    gid = np.repeat(np.arange(10), 3); rows = np.arange(30)
    np.testing.assert_array_equal(subsample_games(gid, rows, 1.0, 0), rows)
    half = subsample_games(gid, rows, 0.5, 0)
    assert len(half) == 15 and all((gid[half] == g).sum() == 3 for g in np.unique(gid[half]))
    assert len(subsample_games(gid, rows, 0.01, 0)) == 3                        # at least one game


def test_load_v_data_game_ids_differ_across_files(tmp_path):
    for f in ("a", "b"):
        np.savez(tmp_path / f"{f}.npz", obs=np.tile(OBS, (4, 1)), k=np.ones(4, int), left=np.full(4, 3),
                 game=np.array([0, 0, 1, 1]), branch=np.zeros(4, bool))
    d = load_v_data([str(tmp_path / "a.npz"), str(tmp_path / "b.npz")], val_frac=0.0)
    assert len(np.unique(d["gid"])) == 4


def test_widen_qv_copies_the_made_side_and_gives_a_wider_v():
    from exp16_wm import eval_by_balls as E
    from exp16_wm.obs_transform import relative_obs
    from exp16_wm.train_v import widen_qv
    a = {"gnn_hidden": 8, "gnn_layers": 2, "made2_head": True, "q_weight": 1.0, "v_weight": 1.0, "qv_encoder": True}
    m = E.policy_from_args(a)
    new, a2 = widen_qv(m, a, 16)
    assert a2["qv_hidden"] == 16 and a2["q_weight"] == 0 and not hasattr(new, "head_q")
    for k, v in m.state_dict().items():
        if not k.startswith(("qv_", "head_v", "head_q")):
            assert torch.equal(new.state_dict()[k], v)
    assert new.qv_pair[-1].out_features == 16
    assert v_logits(new, torch.as_tensor(relative_obs(np.tile(OBS, (2, 1))))).shape == (2,)
    assert E.policy_from_args(a2).state_dict().keys() == new.state_dict().keys()    # load_run rebuilds the same structure


def test_qv_hidden_needs_qv_encoder_and_no_q_head():
    from exp16_wm.gnn_policy import PairGNNPolicy
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=8, v_head=True, qv_hidden=16)
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=8, made2_head=True, q_head=True, v_head=True, qv_encoder=True, qv_hidden=16)
    m = PairGNNPolicy(hidden=8, v_head=True, qv_encoder=True, qv_hidden=0)          # 0 = same width: old checkpoints load
    assert m.qv_pair[-1].out_features == 8


def _v_model(**kw):
    from exp16_wm.gnn_policy import PairGNNPolicy
    torch.manual_seed(0)
    return PairGNNPolicy(hidden=8, v_head=True, qv_encoder=True, **kw).eval()


def test_ff_fourier_features_layout_and_zero_is_identity():
    m = _v_model()
    x = torch.tensor([[0.5, -1.0]])
    assert torch.equal(m._ff(x, 0), x)                                  # k = 0: the plain coordinates
    f = m._ff(x, 3)
    assert f.shape == (1, 2 + 4 * 3)
    assert torch.equal(f[:, :2], x)
    xs = (x[..., None] * (2.0 ** torch.arange(3) * np.pi / 4.0)).flatten(-2)
    torch.testing.assert_close(f[:, 2:8], torch.sin(xs)); torch.testing.assert_close(f[:, 8:], torch.cos(xs))


def test_qv_fourier_none_or_equal_gives_the_same_model_as_before():
    rel = torch.as_tensor(__import__("exp16_wm.obs_transform", fromlist=["relative_obs"]).relative_obs(np.tile(OBS, (3, 1))))
    a, b = _v_model(), _v_model(qv_fourier=0)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert torch.equal(v_logits(a, rel), v_logits(b, rel))             # fourier = 0 on both sides → the old V, bit for bit


def test_qv_fourier_widens_only_the_v_encoder_input():
    from exp16_wm.obs_transform import relative_obs
    old, new = _v_model(), _v_model(qv_fourier=4)
    assert new.bb[0].in_features == old.bb[0].in_features and new.pair[0].in_features == old.pair[0].in_features    # made side unchanged
    assert new.qv_bb[0].in_features == old.qv_bb[0].in_features + 2 * 4 * 4          # ball_f and diff each get 4·k extra
    assert new.qv_pair[0].in_features == old.qv_pair[0].in_features + 2 * 4 * 4
    rel = torch.as_tensor(relative_obs(np.tile(OBS, (2, 1))))
    assert v_logits(new, rel).shape == (2,) and torch.isfinite(v_logits(new, rel)).all()
    # the made encoder is not affected by qv_fourier: same weights seed → same made-side embeddings
    assert torch.equal(new._encode_full(rel)[1], _v_model()._encode_full(rel)[1])


def test_qv_fourier_errors():
    from exp16_wm.gnn_policy import PairGNNPolicy
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=8, v_head=True, qv_fourier=3)                          # needs qv_encoder
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=8, v_head=True, qv_encoder=True, qv_fourier=-1)
    with pytest.raises(ValueError):
        _v_model(qv_fourier=3).copy_encoder_to_qv()                                 # other input width than the made encoder
    _v_model().copy_encoder_to_qv()                                                  # same width: still fine


def test_widen_qv_with_fourier_is_rebuilt_by_load_run_and_keeps_the_made_side():
    from exp16_wm import eval_by_balls as E
    from exp16_wm.train_v import widen_qv
    a = {"gnn_hidden": 8, "gnn_layers": 2, "made2_head": True, "q_weight": 1.0, "v_weight": 1.0, "qv_encoder": True}
    m = E.policy_from_args(a)
    new, a2 = widen_qv(m, a, 8, 6)
    assert a2["qv_fourier"] == 6 and new.qv_fourier == 6 and new.fourier == 0
    for k, v in m.state_dict().items():
        if not k.startswith(("qv_", "head_v", "head_q")):
            assert torch.equal(new.state_dict()[k], v)
    rebuilt = E.policy_from_args(a2)
    assert rebuilt.state_dict().keys() == new.state_dict().keys()
    assert all(rebuilt.state_dict()[k].shape == v.shape for k, v in new.state_dict().items())     # load_run can load the saved weights
    assert E.policy_from_args(a).qv_fourier == 0                                    # runs without the key are the old model


def test_hazard_targets_reached_and_success_rows():
    reached, success = hazard_targets(np.array([2, 0, 3, 4]), np.array([5, 3, 3, 4]), 6)
    np.testing.assert_array_equal(reached, [[1, 1, 1, 0, 0, 0],       # k = 2, 5 left: shots 1-3 played (the 3rd missed)
                                            [1, 0, 0, 0, 0, 0],       # k = 0: only the first shot, a miss
                                            [1, 1, 1, 0, 0, 0],       # k = left = 3: all three went in
                                            [1, 1, 1, 1, 0, 0]])
    np.testing.assert_array_equal(success, [[1, 1, 0, 0, 0, 0], [0] * 6, [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    with pytest.raises(ValueError):
        hazard_targets(np.array([1]), np.array([7]), 6)                    # more balls left than steps


def test_hazard_value_is_the_survival_sum_and_respects_left():
    p = torch.tensor([[0.9, 0.8, 0.7, 0.5, 0.5, 0.5]])
    z = torch.log(p / (1 - p))
    assert hazard_value(z, torch.tensor([3])).item() == pytest.approx(0.9 + 0.72 + 0.504, abs=1e-5)
    assert hazard_value(z, torch.tensor([2])).item() == pytest.approx(0.9 + 0.72, abs=1e-5)    # shots beyond the balls left do not count
    assert hazard_value(torch.full((1, 6), -20.0), torch.tensor([6])).item() < 1e-6              # the first shot never goes in → 0
    assert hazard_value(torch.full((1, 6), 20.0), torch.tensor([4])).item() == pytest.approx(4.0, abs=1e-4)


def test_hazard_nll_counts_only_played_shots():
    z = torch.tensor([[0.3, -0.2, 1.0, 5.0, 5.0, 5.0]])
    reached, success = (torch.as_tensor(x) for x in hazard_targets(np.array([2]), np.array([5]), 6))
    expect = sum(torch.nn.functional.binary_cross_entropy_with_logits(z[0, j], success[0, j]) for j in range(3))
    assert hazard_nll(z, reached, success).item() == pytest.approx(expect.item(), abs=1e-6)
    z2 = z.clone(); z2[0, 3:] = -50.0                                       # shots that were not played change nothing
    assert hazard_nll(z2, reached, success).item() == pytest.approx(expect.item(), abs=1e-6)
    zz = torch.cat([z, z]); r2, s2 = reached.repeat(2, 1), success.repeat(2, 1)
    assert hazard_nll(zz, r2, s2, "sum").item() == pytest.approx(2 * expect.item(), abs=1e-5)


def test_hazard_head_shape_and_scalar_head_unchanged():
    from exp16_wm.gnn_policy import PairGNNPolicy
    from exp16_wm.obs_transform import relative_obs
    rel = torch.as_tensor(relative_obs(np.tile(OBS, (3, 1))))
    hz = _v_model(v_steps=6)
    _, e, valid, _ = hz._encode_full(rel, qv=True)
    assert hz.v_hazard_logits({"e_qv": e, "valid": valid}).shape == (3, 6)
    with pytest.raises(ValueError):
        hz.v_logit({"e_qv": e, "valid": valid})                            # a hazard head has no single value logit
    sc = _v_model()
    assert sc.head_v[-1].out_features == 1 and hz.head_v[-1].out_features == 6
    assert {k: v.shape for k, v in sc.state_dict().items() if not k.startswith("head_v.4")} == \
           {k: v.shape for k, v in hz.state_dict().items() if not k.startswith("head_v.4")}      # only the last Linear differs
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=8, v_head=False, v_steps=6)
    with pytest.raises(ValueError):
        sc.v_hazard_logits({"e_qv": e, "valid": valid})


def test_predict_balls_hazard_matches_the_manual_survival_sum_and_p1():
    from exp16_wm.obs_transform import relative_obs
    from exp16_wm.train_v import predict_balls, predict_first_step, v_hazard_logits
    hz = _v_model(v_steps=6)
    obs = np.tile(OBS, (4, 1)); left = np.array([3, 2, 1, 3])
    got = predict_balls(hz, obs, left)
    z = v_hazard_logits(hz, torch.as_tensor(relative_obs(obs))).detach()
    np.testing.assert_allclose(got, hazard_value(z, torch.as_tensor(left)).numpy(), atol=1e-6)
    assert ((got >= 0) & (got <= left + 1e-6)).all()
    np.testing.assert_allclose(predict_first_step(hz, obs), torch.sigmoid(z[:, 0]).numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        predict_balls(hz, obs, left, horizon=2)


def test_widen_qv_with_hazard_head_is_rebuilt_by_load_run():
    from exp16_wm import eval_by_balls as E
    from exp16_wm.train_v import widen_qv
    a = {"gnn_hidden": 8, "gnn_layers": 2, "made2_head": True, "q_weight": 1.0, "v_weight": 1.0, "qv_encoder": True}
    m = E.policy_from_args(a)
    new, a2 = widen_qv(m, a, 8, None, 6)
    assert a2["v_steps"] == 6 and new.v_steps == 6 and new.head_v[-1].out_features == 6
    rebuilt = E.policy_from_args(a2)
    assert all(rebuilt.state_dict()[k].shape == v.shape for k, v in new.state_dict().items())
    assert E.policy_from_args(a).v_steps == 0                               # runs without the key are the single value head


def test_targets_clear_is_one_only_when_every_countable_ball_goes_in():
    k, left = np.array([5, 4, 0, 3, 2]), np.array([5, 5, 5, 3, 1])
    np.testing.assert_array_equal(targets(k, left, mode="clear"), [1, 0, 0, 1, 1])
    np.testing.assert_array_equal(targets(k, left, horizon=2, mode="clear"), [1, 1, 0, 1, 1])     # only the next 2 balls count: k ≥ min(left, 2)
    np.testing.assert_allclose(targets(k, left), [1.0, 0.8, 0.0, 1.0, 1.0])                        # the default is unchanged
    with pytest.raises(ValueError):
        targets(k, left, mode="soft")


def _smooth_setup(tmp_path):
    n = 6
    obs = np.tile(OBS, (n, 1)); k = np.array([3, 0, 2, 1, 3, 2]); left = np.array([3, 3, 0, 3, 3, 3])      # row 2 has no balls left → dropped
    np.savez(tmp_path / "a.npz", obs=obs, k=k, left=left, game=np.arange(n), branch=np.zeros(n, bool))
    d = load_v_data([str(tmp_path / "a.npz")], val_frac=0.0)
    ok = np.nonzero(left > 0)[0]                                                                              # what v_perturb keeps
    kp = np.array([[[3, 1, -1, -1]], [[0, 0, 0, 2]], [[2, 2, 2, 2]], [[-1, -1, -1, -1]], [[3, 3, 3, 3]]])      # (5, 1, 4 copies); row 3 has none placed
    (tmp_path / "s").mkdir()
    np.savez(tmp_path / "s" / "a.npz", rows=ok, k=k[ok], left=left[ok], k_pert=kp, eps_mm=np.array([10.0]))
    return d, str(tmp_path / "a.npz"), str(tmp_path / "s")


def test_load_smooth_k_means_over_placed_copies_and_falls_back_to_the_exact_k(tmp_path):
    d, f, sdir = _smooth_setup(tmp_path)
    got = load_smooth_k([f], sdir, d)
    np.testing.assert_allclose(got, [2.0, 0.5, 2.0, 3.0, 3.0])                       # (3+1)/2, (0+0+0+2)/4, 2, no copy placed → its exact k = 3, 3
    assert len(got) == len(d["k"]) == 5


def test_load_smooth_k_rejects_labels_of_other_data(tmp_path):
    d, f, sdir = _smooth_setup(tmp_path)
    d2 = {**d, "k": d["k"] + 1}
    with pytest.raises(ValueError):
        load_smooth_k([f], sdir, d2)                                                  # k differs
    with pytest.raises(ValueError):
        load_smooth_k([f], sdir, {k: v[:-1] for k, v in d.items()})                   # a different number of rows
    np.testing.assert_allclose(targets(load_smooth_k([f], sdir, d), d["left"]), [2 / 3, 0.5 / 3, 2 / 3, 1.0, 1.0], atol=1e-6)    # soft target in [0, 1]
