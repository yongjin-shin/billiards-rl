"""Tests for exp16_wm/diagnose_action.py (pure helpers and the model-side refine helpers; the env-based measurements are checked by the run itself)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.diagnose_action import (configure, grad_refine, gradient_toward_window, laplace_perturb, made_logit, refine_at,
                                       scan_baselines, signed_angle_diff_deg, summarize)
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import relative_obs

OFFSETS = np.round(np.arange(-0.6, 0.6 + 1e-9, 0.05), 3)
POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel_batch(n: int = 4) -> torch.Tensor:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        balls = [v for _ in range(3) for v in (*rng.uniform(0.2, 0.8, 2), 0.0)]
        rows.append(relative_obs(np.array([0.5, 0.5, *balls, *POCKETS], dtype=np.float32)))
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def _model() -> PairGNNPolicy:
    torch.manual_seed(0)
    return PairGNNPolicy(hidden=16, made2_head=True).eval()


# ── signed_angle_diff_deg ─────────────────────────────

def test_signed_angle_diff_has_a_sign_and_wraps():
    d = signed_angle_diff_deg(np.array([0.1, 0.0, 0.999]), np.array([0.0, 0.1, -0.999]))
    assert d[0] == pytest.approx(18.0) and d[1] == pytest.approx(-18.0)
    assert d[2] == pytest.approx(-0.36, abs=1e-6)                # across the ±180° seam the short way


# ── laplace_perturb ───────────────────────────────────

def test_laplace_zero_scale_is_identity_and_draws_nothing():
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    assert laplace_perturb(0.25, 0.0, rng) == 0.25
    assert rng.bit_generator.state == state


def test_laplace_scale_is_the_mean_absolute_deviation_in_degrees():
    rng = np.random.default_rng(0)
    dev = np.array([laplace_perturb(0.0, 0.1, rng) for _ in range(20000)]) * 180.0
    assert np.abs(dev).mean() == pytest.approx(0.1, rel=0.05)    # E|Laplace(b)| = b
    assert dev.mean() == pytest.approx(0.0, abs=0.01)


# ── scan_baselines ────────────────────────────────────

def _window(lo: float, hi: float) -> np.ndarray:
    return (OFFSETS >= lo - 1e-9) & (OFFSETS <= hi + 1e-9)


def test_scan_baselines_head_that_repairs_a_missing_proposal():
    truth = np.stack([_window(-0.1, 0.2), _window(-0.1, 0.2)])           # window of 7 of 25 offsets
    prob = np.where(truth, 0.9, 0.1)                                      # perfect head
    r = scan_baselines(prob, truth, OFFSETS, np.array([0.0, 0.5]))        # state 0 proposal inside, state 1 outside
    assert r["n_partial"] == 2 and r["head_hit"] == 1.0
    assert r["proposal_hit"] == 0.5 and r["n_proposal_miss"] == 1
    assert r["head_hit_when_proposal_misses"] == 1.0 and r["head_hit_when_proposal_hits"] == 1.0
    assert r["random_offset"] == pytest.approx(7 / 25) and r["label_offset_hit"] == 1.0


def test_scan_baselines_head_that_only_echoes_the_proposal():
    truth = np.stack([_window(-0.1, 0.2)] * 2)
    prop = np.array([0.0, 0.5])
    prob = np.stack([np.exp(-np.abs(OFFSETS - p) * 20) for p in prop])    # peaked at the proposal
    r = scan_baselines(prob, truth, OFFSETS, prop)
    assert r["head_hit"] == r["proposal_hit"] == 0.5
    assert r["head_hit_when_proposal_misses"] == 0.0 and r["head_hit_when_proposal_hits"] == 1.0


def test_scan_baselines_clips_a_proposal_outside_the_scan_and_handles_no_partial_state():
    truth = np.stack([_window(0.4, 0.6)])
    r = scan_baselines(np.full((1, 25), 0.5), truth, OFFSETS, np.array([3.0]))   # far right → clipped to +0.6 (inside)
    assert r["proposal_hit"] == 1.0
    empty = scan_baselines(np.full((2, 25), 0.5), np.stack([np.ones(25, bool), np.zeros(25, bool)]), OFFSETS, np.zeros(2))
    assert empty["n_partial"] == 0 and np.isnan(empty["head_hit"]) and empty["n_proposal_miss"] == 0


# ── gradient_toward_window ────────────────────────────

def test_gradient_toward_window_counts_only_offsets_outside_it():
    truth = _window(-0.1, 0.1)
    grad = np.sign(0.0 - OFFSETS)                                         # ascent direction toward the centre everywhere
    ok, dist = gradient_toward_window(grad, truth, OFFSETS)
    assert len(ok) == len(dist) == int((~truth).sum()) and ok.all()
    assert dist.min() == pytest.approx(0.05) and dist.max() == pytest.approx(0.5)
    ok_bad, _ = gradient_toward_window(-grad, truth, OFFSETS)
    assert not ok_bad.any()


def test_gradient_toward_window_empty_without_a_partial_window():
    for truth in (np.ones(25, bool), np.zeros(25, bool)):
        ok, dist = gradient_toward_window(np.ones(25), truth, OFFSETS)
        assert len(ok) == 0 and len(dist) == 0


# ── model-side helpers ────────────────────────────────

def test_refine_at_matches_predict_unit_refine():
    m, x = _model(), _rel_batch()
    configure(m, "refine", (-0.2, -0.1, 0.0, 0.1, 0.2), (0.8, 1.2))
    with torch.no_grad():
        o = m.forward_full(x)
        k = o["logits"].argmax(1)
        assert torch.allclose(refine_at(m, o, k), m.predict_unit(x), atol=1e-6)


def test_made_logit_shape_and_refine_picks_the_best_candidate():
    m, x = _model(), _rel_batch()
    configure(m, "refine", (-0.1, 0.0, 0.1), (1.0, 2.0))
    with torch.no_grad():
        o = m.forward_full(x)
        k = torch.zeros(len(x), dtype=torch.long)
        cand = m._unit_offsets(o["means"][torch.arange(len(x)), k])
        z = made_logit(m, o, k, cand)
        assert z.shape == (len(x), 6)
        assert torch.equal(refine_at(m, o, k), cand[torch.arange(len(x)), z.argmax(1)])


def test_grad_refine_never_lowers_the_logit_and_keeps_speed():
    m, x = _model(), _rel_batch()
    with torch.no_grad():
        o = m.forward_full(x)
    k = torch.arange(len(x)) % 3
    a = o["means"][torch.arange(len(x)), k]
    g = grad_refine(m, o, k)
    with torch.no_grad():
        z0 = made_logit(m, o, k, a[:, None])[:, 0]
        z1 = made_logit(m, o, k, g[:, None])[:, 0]
    assert (z1 >= z0 - 1e-6).all()
    assert torch.equal(g[:, 1], a[:, 1])
    assert ((g[:, 0] - a[:, 0]).abs() * 180.0 <= 12 * 0.04 + 1e-4).all()      # at most steps × step_deg away


def test_grad_refine_with_zero_steps_returns_the_proposal():
    m, x = _model(), _rel_batch(2)
    with torch.no_grad():
        o = m.forward_full(x)
    k = torch.zeros(2, dtype=torch.long)
    assert torch.equal(grad_refine(m, o, k, steps=0), o["means"][torch.arange(2), k])


# ── summarize ─────────────────────────────────────────

def test_summarize_pools_seeds_by_balls_left_and_ignores_unlabelled_states():
    def split(nl, lab, ok_val):
        n = len(nl)
        ok = {k: np.full(n, ok_val, bool) for k in ("sel", "lab", "vr", "labref", "lab_grid", "lab_grad", "lab_grad_small")}
        return {"nl": np.array(nl), "lab": np.array(lab), "ok": ok, "sel_in_good": np.ones(n, bool),
                "ver_in_good": np.ones(n, bool), "sel_is_label": np.zeros(n, bool)}
    truth = np.stack([_window(-0.1, 0.2)]); scan = {"P": np.where(truth, 0.9, 0.1), "Y": truth, "prop": np.array([0.0]),
                                                    "gok": np.array([True, False]), "gdist": np.array([0.1, 0.5])}
    s = summarize("x", [split([1, 2, 3, 3], [True, True, True, False], True), split([1, 2], [True, True], False)], [scan, scan], None)
    assert s["n_by_balls"] == {"1": 2, "2": 2, "3": 1}                    # the unlabelled 3-ball state is not counted
    assert s["split"]["select_clean_policy_pair"]["1"] == 0.5 and s["split"]["select_clean_policy_pair"]["3"] == 1.0
    assert s["split"]["select_clean_policy_pair"]["all"] == pytest.approx(3 / 5)
    assert s["split"]["select_pair_is_label"]["all"] == 0.0
    assert s["scan"]["head_hit"] == 1.0 and s["gradient_toward_window"]["all"] == 0.5
    assert "sample" not in s
