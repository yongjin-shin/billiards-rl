"""Tests for exp16_wm/frozen_verifier.py."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.frozen_verifier import AdaptiveVerifier, FrozenVerifier, fit_adaptive, fit_verifier, pair_features, rank_metrics
from exp16_wm.gnn_policy import PairGNNPolicy

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
SCALE = torch.tensor([1.0, 1.0])


def _rel(n, seed=0):
    from exp16_wm.obs_transform import relative_obs
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        balls = [(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), 0) for _ in range(3)]
        raw = np.array([0.5, 0.2, *[v for b in balls for v in b], *POCKETS], dtype=np.float32)
        rows.append(relative_obs(raw))
    return torch.as_tensor(np.stack(rows))


def test_pair_features_concatenate_embeddings_and_take_proposals_from_the_anchor():
    torch.manual_seed(0)
    a, b = PairGNNPolicy(hidden=16, made2_head=True), PairGNNPolicy(hidden=16)
    rel = _rel(7)
    one = pair_features(a, [a], rel, chunk=3)
    two = pair_features(a, [a, b], rel, chunk=3)
    assert one["e"].shape == (7, 18, 16) and two["e"].shape == (7, 18, 32)
    assert torch.allclose(two["e"][..., :16], one["e"]) and torch.allclose(two["means"], a.forward_full(rel)["means"])
    assert torch.equal(two["valid"], a.forward_full(rel)["valid"])
    only_b = pair_features(a, [b], rel)                                             # anchor only supplies means / valid
    assert torch.allclose(only_b["e"], b.forward_full(rel)["e"]) and torch.allclose(only_b["means"], one["means"])


def test_verifier_masks_invalid_pairs_and_candidate_scores_match_the_per_pair_logits():
    torch.manual_seed(1)
    rows = {"e": torch.randn(4, 18, 8), "means": torch.rand(4, 18, 2) * 2 - 1, "valid": torch.rand(4, 18) > 0.3}
    v = FrozenVerifier(8, SCALE, hidden=16)
    a = torch.rand(4, 2) * 2 - 1
    z = v.logits(rows, a)
    assert z.shape == (4, 18) and (z[~rows["valid"]] < -1e8).all() and (z[rows["valid"]] > -1e8).all()
    k = torch.tensor([0, 3, 7, 17])
    cs = v.candidate_scores(rows, k, a[:, None, :])                                  # one candidate per state = the same action
    assert cs.shape == (4, 1)
    live = rows["valid"][torch.arange(4), k]                                         # candidate_scores does not mask; compare on live pairs
    assert live.any() and torch.allclose(cs[live, 0], z[torch.arange(4), k][live], atol=1e-5)


def test_fit_verifier_learns_a_feature_that_decides_the_outcome_and_restores_the_best_epoch():
    rng = np.random.default_rng(0)
    S, F_ = 200, 6
    e = torch.randn(S, 18, F_)
    rows = {"e": e, "means": torch.zeros(S, 18, 2), "valid": torch.ones(S, 18, dtype=torch.bool)}
    sid = rng.integers(0, S, size=1500)
    a = (rng.random((1500, 2)) * 2 - 1).astype(np.float32)
    made = (e[sid, :, 0] > 0).numpy()                                                # the first feature decides every pair's outcome
    scratch = np.zeros(1500, bool)
    v, info = fit_verifier(rows, sid, a, made, scratch, SCALE, seed=0, epochs=25, lr=3e-3, batch=128)
    assert info["train"][0] > 0.4 and min(info["holdout"]) < 0.25                    # from ~0.69 chance to well below
    assert info["holdout"][info["best_epoch"]] == min(info["holdout"])
    # two head seeds share the held-out states: only the initialisation / batch order differ
    _, info2 = fit_verifier(rows, sid, a, made, scratch, SCALE, seed=1, epochs=2, lr=3e-3, batch=128)
    assert len(info2["holdout"]) == 2


def test_fit_verifier_never_trains_on_held_out_states_and_scratch_counts_as_a_failure():
    S = 40
    rows = {"e": torch.randn(S, 18, 4), "means": torch.zeros(S, 18, 2), "valid": torch.ones(S, 18, dtype=torch.bool)}
    sid = np.repeat(np.arange(S), 5)
    a = np.zeros((len(sid), 2), np.float32)
    made = np.ones((len(sid), 18), bool)
    all_scratch = np.ones(len(sid), bool)                                            # made everything but scratched → target all 0
    v, info = fit_verifier(rows, sid, a, made, all_scratch, SCALE, seed=0, epochs=30, lr=1e-2, batch=64)
    assert min(info["holdout"]) < 0.1                                               # learns "never succeeds"
    with torch.no_grad():
        assert torch.sigmoid(v.logits({k: x[:3] for k, x in rows.items()}, torch.zeros(3, 2))).max() < 0.2


def test_rank_metrics_success_at_k_ties_and_states_without_a_success():
    ok = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], bool)
    scores = np.array([[0.1, 0.2, 0.9, 0.0], [0.0, 0.5, 0.4, 0.3], [1, 2, 3, 4]], float)
    m = rank_metrics(scores, ok, ks=(1, 2, 3, 4))
    # state 0: success ranked 1st; state 1: success is the worst-scored → rank 4; state 2: no success at all
    assert m["success_at_k"] == {1: pytest.approx(1 / 3), 2: pytest.approx(1 / 3), 3: pytest.approx(1 / 3), 4: pytest.approx(2 / 3)}
    assert m["grid_has_success"] == pytest.approx(2 / 3)
    perfect = rank_metrics(ok.astype(float), ok, ks=(1,))
    assert perfect["success_at_k"][1] == pytest.approx(2 / 3) and perfect["state_auc"] == pytest.approx(1.0)   # state 2 (no outcome mix) is skipped in the AUC
    flat = rank_metrics(np.zeros((3, 4)), ok, ks=(1, 4))                              # all tied → table order: state 0 rank 3, state 1 rank 1
    assert flat["success_at_k"][1] == pytest.approx(1 / 3) and flat["success_at_k"][4] == pytest.approx(2 / 3)


def test_simulate_state_returns_one_flag_per_candidate():
    from exp16_wm import frozen_verifier as fv
    from simulator import BilliardsEnv
    from exp16_wm.teacher_data import CLEAR_ENV
    fv._init_worker()
    env = BilliardsEnv(**CLEAR_ENV)
    obs, _ = env.reset(seed=0)
    acts = np.array([[0.0, 3.0], [1.0, 1.0], [-1.0, 5.0]], np.float32)
    flags = fv._simulate_state((np.asarray(obs, np.float32), acts))
    assert flags.shape == (3,) and flags.dtype == bool


def test_fit_adaptive_trains_a_private_encoder_copy_and_leaves_the_policy_alone():
    import copy
    torch.manual_seed(0)
    policy = PairGNNPolicy(hidden=16, made2_head=True)
    before = {k: x.clone() for k, x in policy.state_dict().items()}
    rel = _rel(60, seed=3)
    anchor = {k: v for k, v in policy.forward_full(rel).items() if k in ("means", "valid")}
    rng = np.random.default_rng(0)
    sid = rng.integers(0, 60, size=600)
    a = (rng.random((600, 2)) * 2 - 1).astype(np.float32)
    made = np.repeat((rel[sid][:, 3] > 0)[:, None].numpy(), 18, axis=1)             # outcome decided by an input coordinate of the state
    enc = copy.deepcopy(policy)
    enc_before = {k: x.clone() for k, x in enc.state_dict().items()}
    v, info = fit_adaptive(enc, rel, anchor, sid, a, made, np.zeros(600, bool), policy.made2_scale, seed=0, epochs=6, lr_enc=3e-3, lr_head=3e-3, batch=128)
    assert isinstance(v, AdaptiveVerifier) and len(info["holdout"]) == 6 and info["holdout"][info["best_epoch"]] == min(info["holdout"])
    assert any(not torch.equal(enc_before[k], x) for k, x in v.enc.state_dict().items() if k.startswith(("bb.", "ball.", "pair.")))   # the copy adapted
    assert all(torch.equal(before[k], x) for k, x in policy.state_dict().items())                                                   # the policy did not
    assert min(info["holdout"]) < info["train"][0]                                                                                   # it learned something
    k = torch.zeros(5, dtype=torch.long)
    sc = v.candidate_scores(rel[:5], {n: x[:5] for n, x in anchor.items()}, k, torch.zeros(5, 4, 2), chunk=2)
    assert sc.shape == (5, 4)


def test_aim_input_lets_the_head_rank_candidates_that_the_features_alone_cannot():
    """Uninformative features, outcome = candidate angle within 0.2° of the (hidden) aim: only a head that gets the aim difference can learn it."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    S = 150
    units = (np.pi, 0.0)                                                              # angle unit 1 = π rad, no bias
    aim = torch.as_tensor(rng.uniform(-0.5, 0.5, size=(S, 18))).float()
    rows = {"e": torch.randn(S, 18, 4), "means": torch.zeros(S, 18, 2), "valid": torch.ones(S, 18, dtype=torch.bool)}
    sid = rng.integers(0, S, size=2500)
    a = np.zeros((2500, 2), np.float32)
    a[:, 0] = (aim[sid, 0].numpy() + rng.uniform(-0.002, 0.002, size=2500)) / np.pi  # candidates scattered ±0.002 rad ≈ ±0.11° around pair 0's aim
    off = np.degrees(a[:, 0] * np.pi - aim[sid, 0].numpy())
    made = np.zeros((2500, 18), bool); made[:, 0] = np.abs(off) < 0.05               # success window ±0.05° around the aim
    scratch = np.zeros(2500, bool)
    with_aim = fit_verifier({**rows, "aim": aim}, sid, a, made, scratch, SCALE, seed=0, epochs=30, lr=3e-3, batch=128, aim_units=units)[1]
    without = fit_verifier(rows, sid, a, made, scratch, SCALE, seed=0, epochs=30, lr=3e-3, batch=128)[1]
    assert min(with_aim["holdout"]) < 0.8 * min(without["holdout"])                  # the aim difference is what makes the outcome predictable
    with pytest.raises(ValueError):
        fit_verifier({**rows, "aim": aim}, sid, a, made, scratch, SCALE, epochs=1)    # aim rows without units


def test_candidate_scores_with_aim_match_the_per_pair_logits():
    torch.manual_seed(2)
    rows = {"e": torch.randn(4, 18, 6), "means": torch.rand(4, 18, 2) * 2 - 1, "valid": torch.ones(4, 18, dtype=torch.bool), "aim": torch.rand(4, 18) * 2 - 1}
    v = FrozenVerifier(6, SCALE, hidden=16, aim_units=(np.pi, 0.1))
    a = torch.rand(4, 2) * 2 - 1
    k = torch.tensor([2, 5, 11, 17])
    assert torch.allclose(v.candidate_scores(rows, k, a[:, None, :])[:, 0], v.logits(rows, a)[torch.arange(4), k], atol=1e-5)
    assert FrozenVerifier(6, SCALE, hidden=16).head[0].in_features == 10 and v.head[0].in_features == 11
