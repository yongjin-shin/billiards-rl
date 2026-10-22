"""Tests for the expert-iteration heads: Q (search scores) and V (game outcome), and Q-based inference."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.bc import dim_weights, train_bc
from exp16_wm.gnn_policy import PairGNNPolicy
from tests.test_exp16_wm_teacher_free import _made_only_setup, _rel


def _model(**kw) -> PairGNNPolicy:
    torch.manual_seed(0)
    return PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True, q_head=True, v_head=True, **kw)


def test_q_and_v_shapes_and_q_needs_made2():
    m, rel = _model().eval(), torch.as_tensor(_rel(4))
    with torch.no_grad():
        o = m.forward_full(rel)
        q = m.q_logits(o, torch.zeros(4, 2), torch.tensor([0, 3, 7, 17]))
        v = m.v_logit(o)
    assert q.shape == (4,) and v.shape == (4,)
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, q_head=True)


def test_all_pairs_q_chooses_by_the_q_head():
    m, rel = _model().eval(), torch.as_tensor(_rel(5))
    m.refine_deg, m.refine_speed, m.refine_speed_abs = (-0.1, 0.0, 0.1), (1.0,), (1.0, 2.0)
    with torch.no_grad():
        for p in m.head_q.parameters():
            p.normal_()
        m.infer = "all_pairs"; a_made = m.predict_unit(rel)
        m.infer = "all_pairs_q"; a_q = m.predict_unit(rel)
        for p in m.head_q.parameters():                       # a copy of the made head → same choice
            p.zero_()
        m.head_q.load_state_dict(m.head_made2.state_dict()); a_same = m.predict_unit(rel)
    assert not torch.equal(a_made, a_q)
    torch.testing.assert_close(a_same, a_made)


def test_training_with_q_and_v_targets_moves_both_heads():
    data, tr, sims = _made_only_setup()
    n = len(data["obs"])
    data["v_target"] = np.where(np.arange(n) % 3 == 0, np.nan, (np.arange(n) % 2).astype(np.float32))        # some rows have no target
    rng = np.random.default_rng(0)
    q = {"row": np.repeat(np.arange(n), 2), "a_unit": (rng.random((2 * n, 2)) * 2 - 1).astype(np.float32),
         "pair": rng.integers(0, 18, 2 * n), "score": rng.random(2 * n).astype(np.float32)}
    m = _model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    train_bc(m, data, tr, ~tr, "nearest_good", epochs=2, batch=8, lr=3e-3, patience=10, seed=0, log_every=0, select="within",
             w=dim_weights(0.2, 0.5), sims=sims, sims_batch=64, made_only=True, q_data=q, q_weight=1.0, v_weight=1.0, q_batch=32)
    changed = lambda pre: any(not torch.equal(before[k], m.state_dict()[k]) for k in before if k.startswith(pre))
    assert changed("head_q") and changed("head_v") and changed("head_made2")
    with pytest.raises(ValueError):
        d2 = {k: v for k, v in data.items() if k != "v_target"}
        train_bc(_model(), d2, tr, ~tr, "nearest_good", epochs=1, batch=8, lr=3e-3, patience=10, seed=0, log_every=0, select="within",
                 w=dim_weights(0.2, 0.5), sims=sims, sims_batch=64, made_only=True, v_weight=1.0)


def test_load_encoder_with_heads_continues_a_model_and_leaves_new_heads_fresh():
    torch.manual_seed(1)
    old = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    new = _model()
    q_before = {k: v.clone() for k, v in new.state_dict().items() if k.startswith("head_q")}
    keys = new.load_encoder(old.state_dict(), include_heads=True)
    assert any(k.startswith("head_made2") for k in keys)
    for k, v in old.state_dict().items():
        if k.startswith(("head_made2", "pair.", "ball.")):
            torch.testing.assert_close(new.state_dict()[k], v)
    for k, v in q_before.items():
        torch.testing.assert_close(new.state_dict()[k], v)                    # the new Q head is untouched
    keys2 = _model().load_encoder(old.state_dict())
    assert not any(k.startswith("head_") for k in keys2)                     # default: encoder (+ aux) only, as before


def test_mixed_label_files_fill_v_target_and_game_state_flags(tmp_path):
    from exp16_wm.bc import load_q_targets, load_teacher_data
    base = dict(obs=np.zeros((2, 23), np.float32), action=np.zeros((2, 2), np.float32), good_actions=np.zeros((2, 1, 2), np.float32),
                n_good=np.ones(2, np.int32), pocketed=np.zeros(2, np.int32), shot=np.zeros(2, np.int32), game=np.zeros(2, np.int32))
    np.savez(tmp_path / "a.npz", **base)
    np.savez(tmp_path / "b.npz", **base, v_target=np.array([1.0, np.nan], np.float32), is_game_state=np.array([True, False]),
             q_state=np.array([1], np.int32), q_action=np.zeros((1, 2), np.float32), q_pair=np.array([4], np.int16), q_score=np.array([0.5], np.float32))
    paths = [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]
    d = load_teacher_data(paths)
    assert np.isnan(d["v_target"][:2]).all() and d["v_target"][2] == 1.0 and d["is_game_state"].tolist() == [True, True, True, False]
    q = load_q_targets(paths)
    assert q["row"].tolist() == [3] and q["pair"].tolist() == [4]                # offset past the ordinary file


def test_build_split_keeps_the_expert_iteration_columns():
    from exp16_wm.bc import build_split
    n = 20
    data = {"obs": np.zeros((n, 23), np.float32), "action": np.zeros((n, 2), np.float32), "good_actions": np.zeros((n, 1, 2), np.float32),
            "n_good": np.ones(n, np.int32), "pocketed": np.zeros(n, np.int32), "shot": np.zeros(n, np.int32), "game": np.arange(n) // 2,
            "is_game_state": np.arange(n) % 2 == 0, "v_target": np.arange(n, dtype=np.float32)}
    d, tr, va = build_split(data, 0.2, 0, 1.0, "none")
    assert "is_game_state" in d and "v_target" in d and d["is_game_state"].sum() == 10


def test_made_topk_q_only_picks_among_the_made_heads_best():
    m, rel = _model().eval(), torch.as_tensor(_rel(4))
    m.refine_deg, m.refine_speed, m.refine_speed_abs = (-0.1, 0.0, 0.1), (1.0,), (1.0, 2.0)
    with torch.no_grad():
        for p in m.head_q.parameters():
            p.normal_()
        m.infer = "all_pairs"; a_made = m.predict_unit(rel)
        m.infer, m.q_topk = "all_pairs_qk", 1; a_k1 = m.predict_unit(rel)          # K = 1: the made head's own choice
        m.q_topk = 5; a_k5 = m.predict_unit(rel)
        o = m.forward_full(rel)
    torch.testing.assert_close(a_k1, a_made)
    assert a_k5.shape == (4, 2)



def test_qv_detach_keeps_q_and_v_gradients_out_of_the_encoder():
    rel = torch.as_tensor(_rel(4))
    for detach, expect_grad in ((False, True), (True, False)):
        m = _model(); m.qv_detach = detach
        (m.q_loss(rel, torch.zeros(4, 2), torch.tensor([0, 1, 2, 3]), torch.rand(4)) + m.v_loss(rel, torch.ones(4))).backward()
        enc = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith(("pair.", "ball.")) and p.grad is not None)
        heads = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith(("head_q", "head_v")) and p.grad is not None)
        assert heads > 0 and (enc > 0) == expect_grad


def test_q_negatives_are_missed_candidates_of_known_pairs():
    from exp16_wm.bc import q_negatives
    from exp16_wm.sim_identity import to_bitmask
    made = np.zeros((5, 18), bool); made[0, 2] = True; made[1, 7] = True; made[3, 4] = True
    sims = {"row": np.arange(5), "a_unit": np.zeros((5, 2), np.float32), "made": to_bitmask(made),
            "scratch": np.array([0, 0, 0, 1, 0], bool), "pair": np.array([2, 3, 5, 4, -1])}
    neg = q_negatives(sims, 18, 10, 0)
    assert neg["row"].tolist() == [1, 2, 3] and (neg["score"] == 0).all()     # 0 made its pair; 3 scratched; 4 unknown pair
    assert len(q_negatives(sims, 18, 2, 0)["row"]) == 2


def test_separate_qv_encoder_trains_only_itself_and_the_qv_heads():
    m, rel = _model(qv_encoder=True), torch.as_tensor(_rel(4))
    m.copy_encoder_to_qv()
    for n, p in m.named_parameters():
        if n.startswith("qv_pair."):
            torch.testing.assert_close(p, dict(m.named_parameters())[n[3:]])        # starts as a copy of the made encoder
    (m.q_loss(rel, torch.zeros(4, 2), torch.tensor([0, 1, 2, 3]), torch.rand(4)) + m.v_loss(rel, torch.ones(4))).backward()
    g = lambda pre: sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith(pre) and p.grad is not None)
    assert g("qv_pair.") > 0 and g("head_q") > 0 and g("head_v") > 0
    assert g("pair.") == 0 and g("ball.") == 0 and g("head_made2") == 0                  # made encoder and made head untouched
    m.eval(); m.refine_deg, m.refine_speed, m.refine_speed_abs = (-0.1, 0.0, 0.1), (1.0,), (1.0, 2.0)
    m.infer = "all_pairs_qk"
    with torch.no_grad():
        assert m.predict_unit(rel).shape == (4, 2)
