"""Tests for the learned-aim cascade: PairGNNPolicy(aim_encoder=...) and exp16_wm.bc.load_aim_encoder."""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.bc import load_aim_encoder
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import relative_obs

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        balls = [(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), 0) for _ in range(3)]
        rows.append(relative_obs(np.array([0.5, 0.2, *[v for b in balls for v in b], *POCKETS], dtype=np.float32)))
    return torch.as_tensor(np.stack(rows))


def _aim_model(seed: int = 0) -> PairGNNPolicy:
    torch.manual_seed(seed)
    return PairGNNPolicy(hidden=16, enc_layers=2, aux_dim=1)


def test_the_made2_aim_comes_from_the_aim_encoder_when_given_and_from_the_formula_otherwise():
    rel = _rel(5)
    enc = _aim_model()
    torch.manual_seed(1)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=enc)
    m.set_aim_stats(3.0, 40.0)
    with torch.no_grad():
        z = enc.aux_head(enc._encode(rel)[0])[..., 0]
        want = torch.deg2rad(z * 40.0 + 3.0)
        assert torch.allclose(m.forward_full(rel)["aim"], want, atol=1e-6)
        plain = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True)
        assert torch.allclose(plain.forward_full(rel)["aim"], PairGNNPolicy._geometric_aim(rel))
        assert not torch.allclose(m.forward_full(rel)["aim"], PairGNNPolicy._geometric_aim(rel))


def test_the_aim_encoder_is_frozen_during_training_and_stays_in_eval_mode():
    rel = _rel(16, seed=2)
    enc = _aim_model()
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    torch.manual_seed(1)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=enc)
    m.train()
    assert m.training and not m.aim_encoder.training
    opt = torch.optim.Adam(m.params(), lr=1e-2)
    a = torch.rand(16, 2) * 2 - 1
    made = torch.rand(16, 18) > 0.5
    for _ in range(3):
        loss = m.made2_loss(rel, a, made, torch.zeros(16, dtype=torch.bool))
        opt.zero_grad(); loss.backward(); opt.step()
    assert all(p.grad is None for p in m.aim_encoder.parameters())
    assert all(torch.equal(before[k], v) for k, v in m.aim_encoder.state_dict().items())
    assert m.head_made2[0].weight.grad is not None and m.pair[0].weight.grad is not None   # the policy itself learns


def test_cascade_checkpoint_round_trips_with_the_aim_weights_and_stats(tmp_path):
    rel = _rel(4, seed=3)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=_aim_model(0))
    m.set_aim_stats(-2.0, 35.0)
    m.save(str(tmp_path / "p.pt"))
    ck = torch.load(str(tmp_path / "p.pt"))
    assert ck["made2_aim_learned"] and any(k.startswith("aim_encoder.") for k in ck["gnn"]) and "aim_mean" in ck["gnn"]
    m2 = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=_aim_model(5))     # other weights before loading
    m2.load_state_dict(ck["gnn"])
    with torch.no_grad():
        assert torch.allclose(m2.forward_full(rel)["aim"], m.forward_full(rel)["aim"], atol=1e-6)
    assert float(m2.aim_std) == 35.0


def test_aim_encoder_needs_made2_geom_and_a_one_dimensional_aux_head():
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, made2_head=True, aim_encoder=_aim_model())                          # no made2_geom
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=PairGNNPolicy(hidden=16, aux_dim=4))


def test_load_aim_encoder_prefers_the_last_epoch_and_rejects_other_runs(tmp_path):
    best, last = _aim_model(0), _aim_model(1)
    run = tmp_path / "aimrun"; run.mkdir()
    json.dump({"args": {"aux_only": True, "geom_aux_set": "aim", "gnn_hidden": 16, "gnn_layers": 2}}, open(run / "results.json", "w"))
    best.save(str(run / "bc_model.pt")); last.save(str(run / "bc_model_last.pt"))
    got = load_aim_encoder(str(run))
    assert all(torch.equal(got.state_dict()[k], v) for k, v in last.state_dict().items()) and not got.training
    os.remove(run / "bc_model_last.pt")
    assert all(torch.equal(load_aim_encoder(str(run)).state_dict()[k], v) for k, v in best.state_dict().items())
    other = tmp_path / "policy"; other.mkdir()
    json.dump({"args": {"aux_only": False, "geom_aux_set": "four", "gnn_hidden": 16}}, open(other / "results.json", "w"))
    with pytest.raises(ValueError):
        load_aim_encoder(str(other))


# ── several aim encoders (ensemble) ──

def _const_aim_model(deg: float, seed: int = 0) -> PairGNNPolicy:
    """An aim model whose prediction is the constant `deg` (aux head last layer zeroed, bias = deg) with mean 0 / std 1 stats."""
    m = _aim_model(seed)
    with torch.no_grad():
        m.aux_head[2].weight.zero_(); m.aux_head[2].bias.fill_(deg)
    return m


def test_ensemble_aim_is_the_circular_mean_and_handles_the_180_degree_seam():
    rel = _rel(3, seed=4)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=[_const_aim_model(179.0, 0), _const_aim_model(-179.0, 1)])
    m.set_aim_stats(0.0, 1.0)
    with torch.no_grad():
        aim = torch.rad2deg(m.forward_full(rel)["aim"])
    assert torch.allclose(torch.abs(aim), torch.full_like(aim, 180.0), atol=1e-3)      # an arithmetic mean would give 0°
    m2 = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=[_const_aim_model(10.0, 0), _const_aim_model(20.0, 1), _const_aim_model(30.0, 2)])
    m2.set_aim_stats(0.0, 1.0)
    with torch.no_grad():
        assert torch.allclose(torch.rad2deg(m2.forward_full(rel)["aim"]), torch.full((3, 18), 20.0), atol=1e-3)
    assert m.aim_encoder is None and len(m.aim_ensemble) == 2


def test_ensemble_members_are_frozen_and_round_trip(tmp_path):
    rel = _rel(8, seed=5)
    encs = [_aim_model(0), _aim_model(1)]
    before = [{k: v.clone() for k, v in e.state_dict().items()} for e in encs]
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=encs)
    m.set_aim_stats(1.0, 30.0)
    m.train()
    assert all(not e.training for e in m.aim_ensemble)
    opt = torch.optim.Adam(m.params(), lr=1e-2)
    loss = m.made2_loss(rel, torch.rand(8, 2) * 2 - 1, torch.rand(8, 18) > 0.5, torch.zeros(8, dtype=torch.bool))
    opt.zero_grad(); loss.backward(); opt.step()
    assert all(torch.equal(before[i][k], v) for i, e in enumerate(m.aim_ensemble) for k, v in e.state_dict().items())
    m.save(str(tmp_path / "p.pt")); ck = torch.load(str(tmp_path / "p.pt"))
    assert ck["made2_aim_n"] == 2 and any(k.startswith("aim_ensemble.1.") for k in ck["gnn"])
    m2 = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=[_aim_model(7), _aim_model(8)])
    m2.load_state_dict(ck["gnn"])
    with torch.no_grad():
        assert torch.allclose(m2.forward_full(rel)["aim"], m.forward_full(rel)["aim"], atol=1e-6)


def test_eval_load_run_rebuilds_a_cascade_with_several_aim_encoders(tmp_path):
    from exp16_wm import eval_by_balls as E
    dirs = []
    for i in range(2):
        d = tmp_path / f"aim{i}"; d.mkdir(); dirs.append(str(d))
        json.dump({"args": {"aux_only": True, "geom_aux_set": "aim", "gnn_hidden": 16, "gnn_layers": 2}}, open(d / "results.json", "w"))
        _aim_model(i).save(str(d / "bc_model_last.pt"))
    pol = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aim_encoder=[load_aim_encoder(x) for x in dirs])
    pol.set_aim_stats(-0.5, 38.0)
    run = tmp_path / "policy"; run.mkdir()
    pol.save(str(run / "bc_model.pt"))
    json.dump({"args": {"gnn_hidden": 16, "selection": "global", "made2_head": True, "made2_geom": True, "gnn_layers": 2,
                        "made2_aim_from": ["x", "y"], "made2_aim_dirs": dirs}}, open(run / "results.json", "w"))
    got = E.load_run(str(run))
    rel = _rel(4, seed=6)
    with torch.no_grad():
        assert torch.allclose(got.forward_full(rel)["aim"], pol.forward_full(rel)["aim"], atol=1e-6)
    assert len(got.aim_ensemble) == 2 and float(got.aim_std) == 38.0


# ── aim-centred refine / select (inference options) ──

def _geo_policy() -> PairGNNPolicy:
    torch.manual_seed(3)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True)          # exact aim (no aim encoder) is enough for these tests
    m.refine_deg = (-0.1, 0.0, 0.1); m.refine_speed_abs = (1.0, 2.0)
    return m


def test_refine_candidates_are_centred_on_the_proposal_the_aim_or_both():
    m = _geo_policy()
    a = torch.tensor([[0.1, -0.2], [-0.3, 0.4]]); aim_u = torch.tensor([0.12, -0.31])
    m.refine_center = "proposal"
    assert torch.equal(m._refine_candidates(a, None), m._unit_offsets(a))
    m.refine_center = "aim"
    c = m._refine_candidates(a, aim_u)
    assert c.shape == (2, 6, 2) and torch.allclose(c[:, :, 0].mean(1), aim_u, atol=1e-6)     # symmetric grid around the aim
    assert torch.allclose(c[..., 1], m._unit_offsets(a)[..., 1])                              # same absolute speeds
    m.refine_center = "both"
    cb = m._refine_candidates(a, aim_u)
    assert cb.shape == (2, 12, 2) and torch.equal(cb[:, :6], m._unit_offsets(a)) and torch.allclose(cb[:, 6:], c)
    m.refine_center = "nonsense"
    with pytest.raises(ValueError):
        m._refine_candidates(a, aim_u)


def test_select_angle_aim_returns_the_aim_with_the_proposal_speed_and_default_is_unchanged():
    m = _geo_policy(); rel = _rel(5, seed=8)
    m.infer = "select"
    with torch.no_grad():
        base = m.predict_unit(rel)
        o = m.forward_full(rel); k = o["logits"].argmax(1); r = torch.arange(5)
        m.select_angle = "aim"
        got = m.predict_unit(rel)
    want_angle = PairGNNPolicy._geometric_aim(rel)[r, k] / np.pi
    assert torch.allclose(got[:, 0], want_angle.clamp(-1, 1), atol=1e-6) and torch.allclose(got[:, 1], base[:, 1])
    assert torch.allclose(base, o["means"][r, k])                                             # default: the proposal


def test_verify_refine_with_an_aim_centred_grid_picks_a_candidate_of_that_grid():
    m = _geo_policy(); rel = _rel(4, seed=9)
    m.infer, m.refine_center = "verify_refine", "aim"
    with torch.no_grad():
        act = m.predict_unit(rel)
        o = m.forward_full(rel)
        sc = (torch.log_softmax(o["logits"], 1) + torch.nn.functional.logsigmoid(m.made2_logits(o, o["means"])))
        k = sc.argmax(1); r = torch.arange(4)
        cand = m._refine_candidates(o["means"][r, k], m._aim_unit(o, r, k))
    assert all(any(torch.allclose(act[i], cand[i, j], atol=1e-6) for j in range(cand.shape[1])) for i in range(4))
    m2 = PairGNNPolicy(hidden=16, made2_head=True); m2.infer, m2.refine_center = "verify_refine", "aim"   # no aim available
    with pytest.raises(ValueError):
        m2.predict_unit(rel)


def test_verify_point_aim_scores_each_pair_at_its_aim_with_its_proposal_speed():
    m = _geo_policy(); rel = _rel(6, seed=10)
    m.infer = "verify"
    with torch.no_grad():
        base = m.predict_unit(rel)
        o = m.forward_full(rel)
        assert torch.equal(m._verify_points(o), o["means"])                                  # default unchanged
        m.verify_point = "aim"
        pts = m._verify_points(o)
        assert torch.allclose(pts[..., 0], (PairGNNPolicy._geometric_aim(rel) / np.pi).clamp(-1, 1), atol=1e-6)
        assert torch.equal(pts[..., 1], o["means"][..., 1])
        got = m.predict_unit(rel)
        sc = torch.log_softmax(o["logits"], 1) + torch.nn.functional.logsigmoid(m.made2_logits(o, pts))
        k = sc.argmax(1)
    assert torch.allclose(got, o["means"][torch.arange(6), k])                                # verify picks with the aim points, plays the proposal
    m.verify_point = "x"
    with pytest.raises(ValueError):
        m.predict_unit(rel)


def test_all_pairs_picks_the_best_made_logit_over_every_live_pair_grid():
    m = _geo_policy(); rel = _rel(5, seed=11)
    rel = rel.clone(); rel[0, 3:6] = torch.tensor([0.0, 0.0, 1.0])                          # state 0: ball 1 pocketed → its 6 pairs are invalid
    with torch.no_grad():
        o = m.forward_full(rel)
        d = o["aim"] - m.act_bias[0]; aim_u = (torch.atan2(torch.sin(d), torch.cos(d)) / m.act_scale[0]).clamp(-1, 1)
        for mode in ("all_pairs", "all_pairs_prior"):
            m.infer = mode
            got = m.predict_unit(rel)
            for b in range(5):                                                             # brute force over (pair, candidate)
                best, best_a = -float("inf"), None
                for j in range(18):
                    if not o["valid"][b, j]:
                        continue
                    cand = m._unit_offsets(torch.stack([aim_u[b, j], o["means"][b, j, 1]])[None])
                    z = D_made_logit(m, o, b, j, cand)
                    sc = z if mode == "all_pairs" else torch.log_softmax(o["logits"][b], 0)[j] + torch.nn.functional.logsigmoid(z)
                    if sc.max() > best:
                        best, best_a = sc.max(), cand[0, sc.argmax()]
                assert torch.allclose(got[b], best_a, atol=1e-6), (mode, b)
    m.infer = "all_pairs"
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, made2_head=True).__class__.predict_unit(_noaim(), rel)


def D_made_logit(m, o, b, j, cand):
    """Made logits of pair j for candidates cand (1, C, 2), built independently from the policy's feature layout (incl. the aim term)."""
    from exp16_wm.gnn_policy import action_diff
    C = cand.shape[1]
    feats = [o["e"][b, j][None, None].expand(1, C, -1), cand,
             action_diff(cand, o["means"][b, j][None, None].expand(1, C, -1)) * m.made2_scale,
             m._geom_feature(o["aim"][b, j][None, None].expand(1, C), cand[..., 0])]
    return m.head_made2(torch.cat(feats, dim=-1)).squeeze(-1)[0]


def _noaim():
    m = PairGNNPolicy(hidden=16, made2_head=True); m.infer = "all_pairs"; return m


def test_diagnose_made_logit_includes_the_aim_term_and_matches_the_refine_path():
    from exp16_wm.diagnose_action import made_logit
    m = _geo_policy(); rel = _rel(3, seed=12)
    with torch.no_grad():
        o = m.forward_full(rel); k = torch.tensor([0, 7, 12]); r = torch.arange(3)
        cand = m._unit_offsets(o["means"][r, k])
        z = made_logit(m, o, k, cand)
        want = torch.stack([D_made_logit(m, o, b, int(k[b]), cand[b:b + 1]) for b in range(3)])
    assert torch.allclose(z, want, atol=1e-6)
