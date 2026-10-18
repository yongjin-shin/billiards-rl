"""Tests for exp16_wm.made_ensemble: mean made2 logit over runs with the same candidates."""

import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.made_ensemble import MadeEnsemble
from exp16_wm.obs_transform import relative_obs

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel(n: int, seed: int = 0) -> torch.Tensor:
    """n tables; on every other one ball 3 is already pocketed (its 6 pairs are dead)."""
    rng = np.random.default_rng(seed)
    rows = [relative_obs(np.array([0.5, 0.2, *[v for b in range(3) for v in (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
                                                                              float(b == 2 and i % 2 == 1))], *POCKETS],
                                  dtype=np.float32)) for i in range(n)]
    return torch.as_tensor(np.stack(rows))


def _model(seed: int) -> PairGNNPolicy:
    torch.manual_seed(seed)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True).eval()
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = "all_pairs", (-0.1, 0.0, 0.1), (1.0,), (1.0, 2.0)
    return m


def test_one_member_is_the_model_itself():
    m, rel = _model(0), _rel(6)
    with torch.no_grad():
        np.testing.assert_allclose(MadeEnsemble([m]).predict_unit(rel).numpy(), m.predict_unit(rel).numpy(), atol=1e-6)


def test_identical_members_change_nothing():
    m, rel = _model(0), _rel(6)
    with torch.no_grad():
        np.testing.assert_allclose(MadeEnsemble([m, copy.deepcopy(m), copy.deepcopy(m)]).predict_unit(rel).numpy(),
                                   m.predict_unit(rel).numpy(), atol=1e-6)


def test_scores_are_the_mean_of_member_logits_and_the_action_their_argmax():
    a, b, rel = _model(0), _model(1), _rel(4)
    za, ca = MadeEnsemble([a]).scores(rel)
    zb, cb = MadeEnsemble([b]).scores(rel)
    z, c = MadeEnsemble([a, b]).scores(rel)
    torch.testing.assert_close(c, ca); torch.testing.assert_close(c, cb)                   # same candidates for every member
    live = torch.isfinite(z)
    torch.testing.assert_close(z[live], ((za + zb) / 2)[live])
    k = z.reshape(len(rel), -1).argmax(1)
    torch.testing.assert_close(MadeEnsemble([a, b]).predict_unit(rel), c.reshape(len(rel), -1, 2)[torch.arange(len(rel)), k])
    assert (~live).any()                                                                   # dead pairs stay masked


def test_act_returns_env_actions_for_one_or_many_observations():
    e, rel = MadeEnsemble([_model(0), _model(1)]), _rel(3).numpy()
    one, many = e.act(rel[0]), e.act(rel)
    assert one.shape == (2,) and many.shape == (3, 2)
    np.testing.assert_allclose(one, many[0], atol=1e-6)


def test_members_must_share_candidates_and_aim():
    with pytest.raises(ValueError):
        MadeEnsemble([])
    a, b = _model(0), _model(1)
    b.refine_speed_abs = (1.0, 3.0)
    with pytest.raises(ValueError):
        MadeEnsemble([a, b])
    c = _model(2)
    c._geometric_aim = lambda rel: torch.zeros(len(rel), 18)                                  # a different aim source
    with pytest.raises(ValueError):
        MadeEnsemble([a, c]).scores(_rel(2))
    d = PairGNNPolicy(hidden=16, made2_head=True).eval()
    d.refine_deg, d.refine_speed_abs = a.refine_deg, a.refine_speed_abs
    with pytest.raises(ValueError):
        MadeEnsemble([a, d])                                                               # no aim → no all_pairs candidates


def _shifted(seed: int, shift_rad: float) -> PairGNNPolicy:
    m = _model(seed)
    base = m._geometric_aim
    m._geometric_aim = lambda rel: base(rel) + shift_rad                                       # a member with its own aim
    return m


def test_mean_aim_builds_candidates_around_the_circular_mean_and_keeps_each_members_own_aim():
    a, b, rel = _shifted(0, +0.001), _shifted(1, -0.003), _rel(4)
    with pytest.raises(ValueError):
        MadeEnsemble([a, b]).scores(rel)                                                    # shared mode refuses different aims
    z, c = MadeEnsemble([a, b], aim="mean").scores(rel)
    ref = _model(0)
    centre = ref._geometric_aim(rel) - 0.001                                                # circular mean of +0.001 and −0.003
    d = centre - ref.act_bias[0]
    centre_u = torch.atan2(torch.sin(d), torch.cos(d)) / ref.act_scale[0]
    torch.testing.assert_close(c[:, :, 0, 0], (centre_u + (-0.1) / 180.0).clamp(-1, 1), atol=1e-5, rtol=0)
    with torch.no_grad():                                                                   # member a scores those candidates with its own aim
        oa = a.forward_full(rel); j = 0; C = c.shape[2]
        za = a.head_made2(a._made2_features(oa["e"][:, j][:, None].expand(-1, C, -1), c[:, j], oa["means"][:, j][:, None].expand(-1, C, -1),
                                            oa["aim"][:, j][:, None].expand(-1, C))).squeeze(-1)
        ob = b.forward_full(rel)
        zb = b.head_made2(b._made2_features(ob["e"][:, j][:, None].expand(-1, C, -1), c[:, j], ob["means"][:, j][:, None].expand(-1, C, -1),
                                            ob["aim"][:, j][:, None].expand(-1, C))).squeeze(-1)
    torch.testing.assert_close(z[:, j], (za + zb) / 2)


def test_mean_aim_with_one_member_is_the_model_itself():
    m, rel = _model(3), _rel(5)
    with torch.no_grad():
        np.testing.assert_allclose(MadeEnsemble([m], aim="mean").predict_unit(rel).numpy(), m.predict_unit(rel).numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        MadeEnsemble([m], aim="median")

