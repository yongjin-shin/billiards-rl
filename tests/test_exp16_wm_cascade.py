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
