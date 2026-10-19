"""Tests for the end-to-end self aim (roadmap EE1 / EE2): the made2 head reads this network's own aux-head aim."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.bc import dim_weights, train_bc
from exp16_wm.geometry_aux import aim_targets, standardize
from exp16_wm.gnn_policy import PairGNNPolicy
from tests.test_exp16_wm_teacher_free import _made_only_setup, _rel


def _model(mode: str, seed: int = 0) -> PairGNNPolicy:
    torch.manual_seed(seed)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True, aux_dim=1, self_aim=mode)
    m.set_aim_stats(10.0, 50.0)
    return m


def test_self_aim_is_the_own_aux_head_in_degrees():
    m, rel = _model("grad").eval(), torch.as_tensor(_rel(5))
    with torch.no_grad():
        o = m.forward_full(rel)
        e, _ = m._encode(rel)
        expect = torch.deg2rad(m.aux_head(e)[..., 0] * 50.0 + 10.0)
    torch.testing.assert_close(o["aim"], expect)


def _made_grad_on_aux_head(mode: str) -> float:
    m, rel = _model(mode), torch.as_tensor(_rel(6))
    a = torch.zeros(6, 2)
    made = torch.zeros(6, 18, dtype=torch.bool); made[:, 0] = True
    m.made2_loss(rel, a, made, torch.zeros(6, dtype=torch.bool)).backward()
    return sum(float(p.grad.abs().sum()) for p in m.aux_head.parameters() if p.grad is not None)


def test_grad_mode_lets_the_made_loss_reach_the_aim_head_and_detach_stops_it():
    assert _made_grad_on_aux_head("grad") > 0
    assert _made_grad_on_aux_head("detach") == 0


def test_self_aim_needs_an_aim_target_head_and_no_frozen_encoder():
    for kw in (dict(made2_geom=False, aux_dim=1), dict(made2_geom=True, aux_dim=4), dict(made2_geom=True, aux_dim=0)):
        with pytest.raises(ValueError):
            PairGNNPolicy(hidden=16, made2_head=True, self_aim="grad", **kw)
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aux_dim=1, self_aim="sometimes")
    enc = PairGNNPolicy(hidden=16, aux_dim=1)
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, aux_dim=1, self_aim="grad", aim_encoder=enc)
    assert PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True).self_aim == ""          # off by default: old checkpoints unchanged


def test_made_only_with_aux_trains_aim_and_made_head_together():
    data, tr, sims = _made_only_setup()
    aux, mean, std = standardize(*aim_targets(data["obs"]), tr)
    m = _model("grad"); m.set_aim_stats(float(mean[0]), float(std[0]))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=3, batch=8, lr=3e-3, patience=10, seed=0, log_every=0, select="within",
                   w=dim_weights(0.2, 0.5), sims=sims, sims_batch=64, made_only=True, aux=aux, aux_weight=1.0)
    after = m.state_dict()
    changed = lambda pre: any(not torch.equal(before[k], after[k]) for k in before if k.startswith(pre))
    assert changed("aux_head") and changed("head_made2") and changed("pair.")
    assert not changed("head_a") and not changed("head_s")
    assert len(fit["aux_history"]) == fit["epochs_run"] and np.isfinite(fit["best_val_loss"])
    torch.testing.assert_close(m.aim_mean, torch.tensor([float(mean[0])]))                      # stats survive (buffers in the checkpoint)


def test_self_aim_checkpoint_roundtrip(tmp_path):
    m = _model("detach").eval()
    torch.save({"gnn": m.state_dict()}, tmp_path / "m.pt")
    n = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True, aux_dim=1, self_aim="detach").eval()
    n.load_state_dict(torch.load(tmp_path / "m.pt")["gnn"])
    rel = torch.as_tensor(_rel(3))
    with torch.no_grad():
        torch.testing.assert_close(n.forward_full(rel)["aim"], m.forward_full(rel)["aim"])


def test_lambda_zero_trains_the_aim_head_through_the_made_loss_alone():
    from exp16_wm.bc import SELF_AIM_NO_TARGET_STATS
    data, tr, sims = _made_only_setup()
    m = _model("grad"); m.set_aim_stats(*SELF_AIM_NO_TARGET_STATS)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=3, batch=8, lr=3e-3, patience=10, seed=0, log_every=0, select="within",
                   w=dim_weights(0.2, 0.5), sims=sims, sims_batch=64, made_only=True)            # no aux targets, no aux weight
    changed = lambda pre: any(not torch.equal(before[k], m.state_dict()[k]) for k in before if k.startswith(pre))
    assert changed("aux_head") and changed("head_made2")
    assert fit["aux_history"] == []


def test_load_run_rebuilds_a_lambda_zero_self_aim_model(tmp_path):
    import json
    from exp16_wm.eval_by_balls import load_run
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True, aux_dim=1, self_aim="grad", enc_layers=2).eval()
    m.set_aim_stats(0.0, 180.0)
    torch.save({"gnn": m.state_dict()}, tmp_path / "bc_model.pt")
    args = {"gnn_hidden": 16, "made2_head": True, "made2_geom": True, "made2_no_proposal": True, "made2_self_aim": "grad",
            "geom_aux_weight": 0.0, "geom_aux_set": "aim", "gnn_layers": 2}
    (tmp_path / "results.json").write_text(json.dumps({"args": args}))
    n = load_run(str(tmp_path))
    rel = torch.as_tensor(_rel(3))
    with torch.no_grad():
        torch.testing.assert_close(n.forward_full(rel)["aim"], m.forward_full(rel)["aim"])

