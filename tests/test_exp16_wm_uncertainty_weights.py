"""Tests for uncertainty weighting (Kendall et al. 2018) of the pair loss in exp16_wm/gnn_policy.py."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import LOGVAR_RANGE, PairGNNPolicy, action_diff
from test_exp16_wm_speed_head import W, _data, _model


def _kw(**kw) -> PairGNNPolicy:
    return _model(uncertainty_weights=True, **kw)


def _terms(m: PairGNNPolicy, rel, g, pairs):
    """The three unweighted loss terms (angle, speed, selection × sel_weight) the way _pair_loss_kendall builds them."""
    means, logits, _ = m(rel)
    B, K = pairs.shape
    has = pairs >= 0; pidx = pairs.clamp(min=0)
    diff = action_diff(means.gather(1, pidx[..., None].expand(B, K, 2)), torch.nan_to_num(g)) * W
    n = has.sum(1).clamp(min=1)
    ce = -(torch.log_softmax(logits, 1).gather(1, pidx) * has).sum(1) / n
    return (((diff[..., 0] ** 2 * has).sum(1) / n).mean(), ((diff[..., 1] ** 2 * has).sum(1) / n).mean(), (m.sel_weight * ce).mean())


def test_default_model_has_no_learned_weights():
    m = _model()
    assert m.uncertainty_weights is False and not hasattr(m, "log_var") and "log_var" not in m.state_dict()


def test_at_zero_log_variance_the_loss_equals_the_plain_pair_loss():
    rel, g, pairs = _data()
    plain, kendall = _model(), _kw()
    assert kendall.log_var.shape == (3,) and torch.equal(kendall.log_var, torch.zeros(3))
    assert torch.allclose(kendall._pair_loss(rel, g, pairs, W), plain._pair_loss(rel, g, pairs, W), atol=1e-4, rtol=1e-5)


def test_loss_is_the_weighted_sum_of_the_three_terms_plus_log_variances():
    rel, g, pairs = _data()
    m = _kw()
    with torch.no_grad():
        m.log_var.copy_(torch.tensor([0.5, -1.0, 2.0]))
    a, v, s = _terms(m, rel, g, pairs)
    s_ = m.log_var.detach()
    expected = (torch.exp(-s_[0]) * a + s_[0]) + (torch.exp(-s_[1]) * v + s_[1]) + (torch.exp(-s_[2]) * s + s_[2])
    assert torch.allclose(m._pair_loss(rel, g, pairs, W), expected, rtol=1e-5)


def test_gradient_pushes_the_log_variance_up_for_a_large_loss_and_down_for_a_small_one():
    rel, g, pairs = _data()
    m = _kw()
    a, v, s = _terms(m, rel, g, pairs)
    m.zero_grad(); m._pair_loss(rel, g, pairs, W).backward()
    grad = m.log_var.grad
    assert torch.allclose(grad, 1.0 - torch.stack([a, v, s]).detach(), rtol=1e-4, atol=1e-5)     # d/ds [e^{-s} L + s] at s = 0
    assert (grad[torch.stack([a, v, s]).detach() > 1.0] < 0).all()                                 # L > 1 → descent raises s (weight falls)


def test_log_variance_is_clamped_to_the_range_and_stops_receiving_gradient_outside_it():
    rel, g, pairs = _data()
    m = _kw()
    with torch.no_grad():
        m.log_var.copy_(torch.tensor([10.0, -10.0, 0.0]))
    a, v, s = _terms(m, rel, g, pairs)
    r = LOGVAR_RANGE
    expected = (np.exp(-r) * a + r) + (np.exp(r) * v - r) + (s + 0.0)
    assert torch.allclose(m._pair_loss(rel, g, pairs, W), expected.to(torch.float32), rtol=1e-5)
    m.zero_grad(); m._pair_loss(rel, g, pairs, W).backward()
    assert m.log_var.grad[0].item() == 0.0 and m.log_var.grad[1].item() == 0.0 and m.log_var.grad[2].item() != 0.0


def test_minimising_over_the_log_variance_alone_gives_log_of_each_loss():
    rel, g, pairs = _data()
    m = _kw()
    a, v, s = [t.detach() for t in _terms(m, rel, g, pairs)]
    opt = torch.optim.Adam([m.log_var], lr=0.05)
    for _ in range(600):
        opt.zero_grad(); m._pair_loss(rel, g, pairs, W).backward(); opt.step()
    target = torch.log(torch.stack([a, v, s])).clamp(-LOGVAR_RANGE, LOGVAR_RANGE)                # σ² = L at the optimum
    assert torch.allclose(m.log_var.detach().clamp(-LOGVAR_RANGE, LOGVAR_RANGE), target, atol=0.1)


def test_unsupported_combinations_raise():
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, uncertainty_weights=True, speed_loss="l1")
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, uncertainty_weights=True, speed_bins=21)
    rel, g, pairs = _data()
    m = _kw()                                                          # wta_fallback is True by default
    with pytest.raises(ValueError):
        m.train_loss(rel, g[:, 0], g, W, pairs=pairs)
    m.wta_fallback = False
    m.train_loss(rel, g[:, 0], g, W, pairs=pairs)
    with pytest.raises(ValueError):
        m.train_loss(rel, g[:, 0], g, W, pairs=None)


def test_state_dict_round_trip_keeps_the_learned_weights():
    m = _kw()
    with torch.no_grad():
        m.log_var.copy_(torch.tensor([1.0, -0.5, 0.25]))
    m2 = PairGNNPolicy(hidden=16, uncertainty_weights=True)
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.log_var, m.log_var)
    assert m.log_var in list(m.params())                              # the optimizer sees it
