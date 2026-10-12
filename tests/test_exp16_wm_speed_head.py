"""Tests for the speed options of exp16_wm/gnn_policy.py: speed_loss="l1" and the categorical speed head (speed_bins)."""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import (PairGNNPolicy, action_diff, bin_center_ms, speed_bin_centers, speed_bin_edges,
                                  speed_to_bin)
from exp16_wm.obs_transform import relative_obs

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
W = torch.tensor([900.0, 7.5])           # dim_weights(0.2°, 0.5 m/s)


def _rel(n: int = 6, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        balls = [v for _ in range(3) for v in (*rng.uniform(0.2, 0.8, 2), 0.0)]
        rows.append(relative_obs(np.array([0.5, 0.5, *balls, *POCKETS], dtype=np.float32)))
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def _data(n: int = 6):
    """rel, label actions (n, 2, 2) in [-1, 1] units (second candidate unlabelled), pairs (n, 2)."""
    rel = _rel(n)
    g = torch.zeros(n, 2, 2)
    g[:, 0, 0] = torch.linspace(-0.3, 0.3, n)
    g[:, 0, 1] = torch.tensor([(1.0 - 4.25) / 3.75, (2.0 - 4.25) / 3.75, (1.0 - 4.25) / 3.75, (3.5 - 4.25) / 3.75, (1.0 - 4.25) / 3.75, (6.0 - 4.25) / 3.75][:n])
    g[:, 1] = float("nan")
    pairs = torch.full((n, 2), -1, dtype=torch.long)
    pairs[:, 0] = torch.arange(n) % 18
    return rel, g, pairs


def _model(**kw) -> PairGNNPolicy:
    torch.manual_seed(0)
    return PairGNNPolicy(hidden=16, **kw)


# ── bin helpers ───────────────────────────────────────

def test_speed_bin_centres_are_log_spaced_and_put_1_and_2_ms_on_centres():
    c = speed_bin_centers(21)
    assert len(c) == 21 and c[0] == pytest.approx(0.5) and c[-1] == pytest.approx(8.0)
    assert c[5] == pytest.approx(1.0, rel=1e-5) and c[10] == pytest.approx(2.0, rel=1e-5) and c[15] == pytest.approx(4.0, rel=1e-5)
    r = c[1:] / c[:-1]
    assert torch.allclose(r, r[0].expand_as(r), rtol=1e-5)
    e = speed_bin_edges(21)
    assert len(e) == 20 and bool((e[1:] > e[:-1]).all()) and bool(((e > c[:-1]) & (e < c[1:])).all())


def test_speed_to_bin_is_robust_around_the_dominant_label_speed_and_clamps():
    v = torch.tensor([0.9999999, 1.0, 1.0000001, 0.5, 8.0, 0.1, 20.0])
    assert speed_to_bin(v, 21).tolist() == [5, 5, 5, 0, 20, 0, 20]
    idx = torch.arange(21)
    assert torch.equal(speed_to_bin(bin_center_ms(idx, 21), 21), idx)          # centre → its own bin, for every bin


# ── defaults are untouched ────────────────────────────

def test_default_model_has_no_speed_head_and_the_pair_loss_is_the_old_formula():
    m = _model()
    assert m.speed_loss == "mse" and m.speed_bins == 0 and not hasattr(m, "head_speed_bins")
    rel, g, pairs = _data()
    means, logits, _ = m(rel)
    pidx = pairs.clamp(min=0); has = pairs >= 0
    mm = means.gather(1, pidx[..., None].expand(6, 2, 2))
    d = ((action_diff(mm, torch.nan_to_num(g)) * W) ** 2).sum(-1)
    n = has.sum(1).clamp(min=1)
    expected = ((d * has).sum(1) / n + m.sel_weight * (-(F.log_softmax(logits, 1).gather(1, pidx) * has).sum(1) / n)).mean()
    assert torch.allclose(m._pair_loss(rel, g, pairs, W), expected, atol=1e-6)
    assert torch.allclose(m.train_loss(rel, g[:, 0], g, W, pairs=pairs), expected, atol=1e-6)


# ── L1 ────────────────────────────────────────────────

def test_l1_speed_term_is_absolute_and_leaves_the_angle_term_squared():
    m = _model(speed_loss="l1")
    rel, g, pairs = _data()
    o = m.forward_full(rel)
    pidx = pairs.clamp(min=0); has = pairs >= 0
    mm = o["means"].gather(1, pidx[..., None].expand(6, 2, 2))
    diff = action_diff(mm, torch.nan_to_num(g)) * W
    n = has.sum(1).clamp(min=1)
    reg = (((diff[..., 0] ** 2 + diff[..., 1].abs()) * has).sum(1) / n)
    ce = -(F.log_softmax(o["logits"], 1).gather(1, pidx) * has).sum(1) / n
    assert torch.allclose(m._pair_loss(rel, g, pairs, W), (reg + m.sel_weight * ce).mean(), atol=1e-5)


def test_l1_gradient_on_speed_does_not_grow_with_the_error():
    m = _model(speed_loss="l1")
    rel, g, pairs = _data(2)
    gs = []
    for far in (0.1, 0.8):                                         # same pair, speed label near vs far from the model's output
        g2 = g.clone()
        g2[:, 0, 1] = (m.forward_full(rel)["means"][torch.arange(2), pairs[:, 0], 1].detach() + far).clamp(-1, 1)
        m.zero_grad()
        m._pair_loss(rel, g2, pairs, torch.tensor([0.0, 7.5])).backward()      # angle weight 0: only speed (+ selection) contributes
        gs.append(m.head_a.weight.grad[1].abs().sum().item())
    assert gs[0] > 0                                               # the speed term does produce a gradient
    assert gs[1] == pytest.approx(gs[0], rel=0.5)                  # squared error would make the far case roughly 8x larger


# ── categorical speed head ────────────────────────────

def test_bins_head_outputs_bin_centres_as_the_proposal_speed():
    m = _model(speed_bins=21).eval()
    o = m.forward_full(_rel())
    assert o["speed_logits"].shape == (6, 18, 21) and o["means"].shape == (6, 18, 2)
    ms = o["means"][..., 1] * m.act_scale[1] + m.act_bias[1]
    assert torch.allclose(ms[..., None], speed_bin_centers(21).expand(6, 18, 21).clone().gather(-1, o["speed_logits"].argmax(-1, keepdim=True)), atol=1e-5)


def test_bins_loss_is_angle_regression_plus_selection_plus_speed_cross_entropy():
    m = _model(speed_bins=21)
    rel, g, pairs = _data()
    o = m.forward_full(rel)
    pidx = pairs.clamp(min=0); has = pairs >= 0
    mm = o["means"].gather(1, pidx[..., None].expand(6, 2, 2))
    gg = torch.nan_to_num(g)
    n = has.sum(1).clamp(min=1)
    reg = ((action_diff(mm, gg)[..., 0] * W[0]) ** 2 * has).sum(1) / n
    ce = -(F.log_softmax(o["logits"], 1).gather(1, pidx) * has).sum(1) / n
    tb = speed_to_bin(gg[..., 1] * m.act_scale[1] + m.act_bias[1], 21)
    sp = o["speed_logits"].gather(1, pidx[..., None].expand(6, 2, 21))
    ce_sp = F.cross_entropy(sp.reshape(-1, 21), tb.reshape(-1), reduction="none").reshape(6, 2)
    expected = (reg + m.sel_weight * ce + m.sel_weight * (ce_sp * has).sum(1) / n).mean()
    assert torch.allclose(m._pair_loss(rel, g, pairs, W), expected, atol=1e-5)


def test_bins_speed_is_trained_only_through_the_bin_head():
    m = _model(speed_bins=21)
    rel, g, pairs = _data()
    m.zero_grad()
    m._pair_loss(rel, g, pairs, W).backward()
    assert m.head_a.weight.grad[1].abs().sum().item() == 0.0          # head_a's speed output is unused
    assert m.head_a.weight.grad[0].abs().sum().item() > 0.0           # its angle output is trained
    assert m.head_speed_bins.weight.grad.abs().sum().item() > 0.0


def test_bins_head_learns_the_two_modes_instead_of_their_mean():
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, speed_bins=21)
    rel = _rel(4)
    g = torch.zeros(4, 2, 2); g[:, 1] = float("nan")
    g[:, 0, 1] = torch.tensor([(1.0 - 4.25) / 3.75, (1.0 - 4.25) / 3.75, (3.0 - 4.25) / 3.75, (3.0 - 4.25) / 3.75])
    pairs = torch.full((4, 2), -1, dtype=torch.long); pairs[:, 0] = torch.tensor([0, 1, 2, 3])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad(); m._pair_loss(rel, g, pairs, W).backward(); opt.step()
    m.eval()
    ms = m.forward_full(rel)["means"][torch.arange(4), pairs[:, 0], 1] * m.act_scale[1] + m.act_bias[1]
    assert torch.allclose(ms, torch.tensor([1.0, 1.0, 3.0, 3.0]), rtol=0.1)       # the bin centres, not the 2.0 average


def test_unlabelled_candidates_do_not_enter_the_speed_cross_entropy():
    m = _model(speed_bins=21)
    rel, g, pairs = _data()
    a = m._pair_loss(rel, g, pairs, W)
    g2 = g.clone(); g2[:, 1] = 0.7                                  # change the unlabelled candidate's action
    assert torch.allclose(a, m._pair_loss(rel, g2, pairs, W), atol=1e-6)


def test_speed_sampling_follows_the_softmax_and_argmax_is_deterministic():
    m = _model(speed_bins=21).eval()
    rel = _rel(2)
    with torch.no_grad():
        m.head_speed_bins.weight.zero_(); m.head_speed_bins.bias.copy_(torch.where(torch.arange(21) == 7, 50.0, 0.0))   # all mass on bin 7
        a = m.forward_full(rel)["means"][..., 1]
        m.speed_sample = True
        b = m.forward_full(rel)["means"][..., 1]
        assert torch.equal(a, b)                                          # a peaked softmax samples its mode
        m.head_speed_bins.bias.zero_()                                    # uniform softmax: samples spread over bins
        draws = torch.stack([m.forward_full(rel)["means"][..., 1] for _ in range(20)])
    assert len(torch.unique(draws)) > 5
    m.speed_sample = False


# ── guards ────────────────────────────────────────────

def test_invalid_options_and_unsupported_loss_paths_raise():
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, speed_loss="huber")
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, speed_bins=1)
    rel, g, pairs = _data()
    m = _model(speed_bins=21)                                              # wta_fallback is True by default
    with pytest.raises(ValueError):
        m.train_loss(rel, g[:, 0], g, W, pairs=pairs)
    m.wta_fallback = False
    m.train_loss(rel, g[:, 0], g, W, pairs=pairs)                          # supported combination
    with pytest.raises(ValueError):
        m.train_loss(rel, g[:, 0], g, W, pairs=None)
    m.mod = ("label", "angle", False)
    with pytest.raises(ValueError):
        m.train_loss(rel, g[:, 0], g, W, pairs=pairs)
    f = PairGNNPolicy(hidden=16, factorized=True, speed_loss="l1"); f.wta_fallback = False
    with pytest.raises(ValueError):
        f.train_loss(rel, g[:, 0], g, W, pairs=pairs, extras={})


def test_state_dict_round_trip_keeps_the_speed_head():
    m = _model(speed_bins=21).eval()
    m2 = PairGNNPolicy(hidden=16, speed_bins=21).eval()
    m2.load_state_dict(m.state_dict())
    rel = _rel(3)
    assert torch.equal(m.forward_full(rel)["means"], m2.forward_full(rel)["means"])
    assert "head_speed_bins.weight" in m.state_dict()
    assert "head_speed_bins.weight" not in _model().state_dict()
