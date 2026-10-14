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
