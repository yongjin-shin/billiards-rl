"""Tests for exp16_wm/verifier_distill.py (pure helpers and the training core; the simulator side is checked by the run itself)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.verifier_distill import (NEVER, encode_cache, finetune_head, first_success_lazy, first_success_rank, holdout_states, made2_bce,
                                       sample_candidates, state_auc, success_at_k)
from test_exp16_wm_speed_head import _rel


def test_sample_candidates_takes_n_top_from_the_pool_and_the_rest_from_outside_it():
    scores = np.arange(217, dtype=float)                                  # candidate 216 is the best, 0 the worst
    rng = np.random.default_rng(0)
    idx = sample_candidates(scores, n_top=12, top_pool=20, n_uniform=12, rng=rng)
    assert len(idx) == 24 and len(set(idx.tolist())) == 24
    assert (idx[:12] >= 217 - 20).all() and (idx[12:] < 217 - 20).all()


def test_sample_candidates_handles_small_pools_and_is_reproducible():
    idx = sample_candidates(np.array([3.0, 1.0, 2.0]), n_top=12, top_pool=2, n_uniform=12, rng=np.random.default_rng(1))
    assert sorted(idx.tolist()) == [0, 1, 2]                              # 2 from the pool (all of it), 1 from the rest (all of it)
    a = sample_candidates(np.random.default_rng(5).normal(size=217), 12, 20, 12, np.random.default_rng(3))
    b = sample_candidates(np.random.default_rng(5).normal(size=217), 12, 20, 12, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_first_success_rank_and_success_at_k():
    assert first_success_rank([False, False, True, True]) == 3
    assert first_success_rank([True]) == 1 and first_success_rank([False, False]) == NEVER and first_success_rank([]) == NEVER
    s = success_at_k(np.array([1, 1, 3, 20, NEVER]), ks=(1, 3, 20, 217))
    assert s == {1: 0.4, 3: 0.6, 20: 0.8, 217: 0.8}


def test_state_auc_is_per_state_and_skips_states_without_both_classes():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.6])
    labels = np.array([True, False, False, True, True, True])
    state = np.array([0, 0, 1, 1, 2, 2])
    assert state_auc(scores, labels, state) == pytest.approx((1.0 + 0.0) / 2)      # state 0 perfect, state 1 reversed, state 2 has no negative
    assert np.isnan(state_auc(np.array([0.1, 0.2]), np.array([True, True]), np.array([0, 0])))
    assert state_auc(np.array([0.5, 0.5]), np.array([True, False]), np.array([0, 0])) == 0.5   # ties count half


def test_holdout_states_hold_out_whole_states_and_keep_both_sides():
    m = holdout_states(100, 0.1, np.random.default_rng(0))
    assert m.sum() == 10 and len(m) == 100
    assert holdout_states(2, 0.9, np.random.default_rng(0)).sum() == 1             # never everything
    assert holdout_states(1, 0.5, np.random.default_rng(0)).sum() == 0


def _toy(n_states=24, per=16, seed=0):
    torch.manual_seed(seed)
    m = PairGNNPolicy(hidden=16, made2_head=True).eval()
    rel = torch.cat([_rel(1, seed=i) for i in range(n_states)])
    cache = encode_cache(m, rel)
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n_states), per)
    a = rng.uniform(-0.01, 0.01, size=(len(sid), 2)).astype(np.float32)             # offsets around 0 in [-1, 1] units
    # the made outcome of pair 0..17 depends on how far the action is from a state-specific optimum: learnable from (a, e)
    opt = rng.uniform(-0.004, 0.004, size=(n_states, 1)).astype(np.float32)
    made = np.zeros((len(sid), 18), bool)
    made[:, 0] = np.abs(a[:, 0] - opt[sid, 0]) < 0.003
    return m, cache, sid, a, made, np.zeros(len(sid), bool)


def test_finetune_trains_only_the_made_head_and_lowers_the_holdout_loss():
    m, cache, sid, a, made, scr = _toy()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    res = finetune_head(m, cache, sid, a, made, scr, epochs=40, lr=5e-3, batch=64, holdout=0.25, seed=0)
    after = m.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and all(k.startswith("head_made2.") for k in changed)           # nothing else moved
    assert res["base_holdout"] > min(res["holdout"]) and res["train"][-1] < res["train"][0]
    assert 0 <= res["best_epoch"] < 40


def test_finetune_restores_the_best_epoch_and_never_trains_on_held_out_states():
    m, cache, sid, a, made, scr = _toy()
    res = finetune_head(m, cache, sid, a, made, scr, epochs=15, lr=5e-3, batch=64, holdout=0.25, seed=1)
    sid_t = torch.as_tensor(sid); hold = holdout_states(len(cache["e"]), 0.25, np.random.default_rng(1))[sid]
    with torch.no_grad():
        ho = float(made2_bce(m, cache, sid_t[hold], torch.as_tensor(a)[hold], torch.as_tensor(made)[hold], torch.as_tensor(scr)[hold]))
    assert ho == pytest.approx(min(res["holdout"]), abs=1e-5)                       # the restored weights are the best held-out epoch
    assert hold.sum() > 0 and (~hold).sum() > 0


def test_made2_bce_matches_the_models_own_loss():
    m, cache, sid, a, made, scr = _toy(n_states=6, per=4)
    rel = torch.cat([_rel(1, seed=i) for i in range(6)])
    expected = m.made2_loss(rel[sid], torch.as_tensor(a), torch.as_tensor(made), torch.as_tensor(scr))
    got = made2_bce(m, cache, torch.as_tensor(sid), torch.as_tensor(a), torch.as_tensor(made), torch.as_tensor(scr))
    assert torch.allclose(got, expected, atol=1e-5)


def test_scratch_candidates_count_as_negatives():
    m, cache, sid, a, made, scr = _toy(n_states=6, per=4)
    made[:, :] = True
    all_pos = made2_bce(m, cache, torch.as_tensor(sid), torch.as_tensor(a), torch.as_tensor(made), torch.as_tensor(scr))
    scr2 = np.ones(len(sid), bool)                                                  # the cue ball went in → target 0 for every pair
    all_neg = made2_bce(m, cache, torch.as_tensor(sid), torch.as_tensor(a), torch.as_tensor(made), torch.as_tensor(scr2))
    assert float(all_pos) != pytest.approx(float(all_neg))


def test_first_success_lazy_stops_at_the_first_success_and_counts_checks():
    calls = []
    def ok(x):
        calls.append(x); return x >= 3
    assert first_success_lazy([0, 1, 3, 5, 7], ok) == (3, 3) and calls == [0, 1, 3]        # 5 and 7 were never checked
    assert first_success_lazy([0, 1], lambda x: False) == (NEVER, 2)
    assert first_success_lazy([], lambda x: True) == (NEVER, 0)
    assert first_success_lazy(iter([9]), lambda x: True) == (1, 1)                          # works on a generator too
