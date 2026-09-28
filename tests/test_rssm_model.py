"""
tests/test_rssm_model.py

Unit tests for RSSMModel (world_model/rssm_model.py).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from world_model.rssm_model import (
    RSSMModel, EventStep, RSSMOutput,
    H_DIM, NODE_DIM, EDGE_DIM, N_TYPE,
    EVENT_BALL_BALL, EVENT_POCKET,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _model() -> RSSMModel:
    return RSSMModel(h_dim=H_DIM, hidden=[64, 64])   # small for speed


def _n_balls(n: int = 4) -> int:
    return n


def _node(type_id: int = 0) -> torch.Tensor:
    """(NODE_DIM,) = random pos/vel/avel + type one-hot."""
    feat = torch.randn(NODE_DIM - N_TYPE)
    oh   = torch.zeros(N_TYPE)
    oh[type_id] = 1.0
    return torch.cat([feat, oh])


def _edge() -> torch.Tensor:
    return torch.randn(EDGE_DIM)


def _normal() -> torch.Tensor:
    v = torch.randn(2)
    return v / v.norm()


def _ball_ball_ev(i: int, j: int) -> EventStep:
    return EventStep(
        event_type=EVENT_BALL_BALL,
        ball_i=i, ball_j=j,
        node_i=_node(0), node_j=_node(0),
        edge=_edge(),
        normal=_normal(),
    )


def _single_ev(i: int, event_type: int = 1) -> EventStep:
    return EventStep(
        event_type=event_type,
        ball_i=i, ball_j=None,
        node_i=_node(event_type), node_j=None,
        edge=None,
        normal=_normal(),
    )


# ── TestRSSMInit ──────────────────────────────────────────────────────────────

class TestRSSMInit:
    def test_init_hidden_shape(self):
        model = _model()
        h = model.init_hidden(4)
        assert isinstance(h, list) and len(h) == 4
        assert all(t.shape == (H_DIM,) for t in h)

    def test_init_hidden_is_zero(self):
        """h starts at zero — cross-event history begins empty."""
        model = _model()
        h = model.init_hidden(4)
        assert all(torch.all(t == 0) for t in h), "initial h should be all zeros"


# ── TestRSSMStep ──────────────────────────────────────────────────────────────

class TestRSSMStep:
    def test_step_ball_ball_output_shapes(self):
        model = _model()
        h = model.init_hidden(4)
        h_new, d_i, d_j, t_i, t_j = model.step_ball_ball(
            h, 0, 1, _node(), _node(), _edge()
        )
        assert isinstance(h_new, list) and len(h_new) == 4
        assert all(t.shape == (H_DIM,) for t in h_new)
        assert d_i.shape == (5,)
        assert d_j.shape == (5,)
        assert t_i.shape == (N_TYPE,)
        assert t_j.shape == (N_TYPE,)

    def test_step_ball_ball_updates_only_involved(self):
        model = _model()
        # snapshot uninvolved balls before the step
        h = model.init_hidden(4)
        snap2, snap3 = h[2].clone(), h[3].clone()
        h_new, *_ = model.step_ball_ball(h, 0, 1, _node(), _node(), _edge())

        # involved balls changed (non-zero after update from all-zero start)
        assert not torch.allclose(h_new[0], torch.zeros(H_DIM)), "h[i] should change"
        assert not torch.allclose(h_new[1], torch.zeros(H_DIM)), "h[j] should change"

        # uninvolved balls unchanged (compare against pre-step snapshots)
        assert torch.allclose(h_new[2], snap2), "h[2] should not change"
        assert torch.allclose(h_new[3], snap3), "h[3] should not change"

    def test_step_ball_ball_directional(self):
        """Reversing the edge direction should produce different results."""
        model = _model()
        e = _edge()
        n1, n2 = _node(), _node()

        h_fwd = model.init_hidden(4)
        _, d_i_fwd, _, _, _ = model.step_ball_ball(h_fwd, 0, 1, n1, n2,  e)

        h_rev = model.init_hidden(4)
        _, d_i_rev, _, _, _ = model.step_ball_ball(h_rev, 0, 1, n1, n2, -e)

        assert not torch.allclose(d_i_fwd, d_i_rev), \
            "flipping edge direction should change output (directional GNN)"

    def test_step_single_output_shapes(self):
        model = _model()
        h = model.init_hidden(4)
        h_new, d_i, t_i = model.step_single(h, 2, _node(1), _normal())

        assert isinstance(h_new, list) and len(h_new) == 4
        assert all(t.shape == (H_DIM,) for t in h_new)
        assert d_i.shape == (5,)
        assert t_i.shape == (N_TYPE,)

    def test_step_single_updates_only_involved(self):
        model = _model()
        h = model.init_hidden(4)
        snap0, snap1, snap3 = h[0].clone(), h[1].clone(), h[3].clone()
        h_new, *_ = model.step_single(h, 2, _node(1), _normal())

        assert not torch.allclose(h_new[2], torch.zeros(H_DIM)), "h[i] should change"
        assert torch.allclose(h_new[0], snap0), "h[0] should not change"
        assert torch.allclose(h_new[1], snap1), "h[1] should not change"
        assert torch.allclose(h_new[3], snap3), "h[3] should not change"


# ── TestRSSMAggregateAndForward ───────────────────────────────────────────────

class TestRSSMAggregateAndForward:
    def test_aggregate_q_shape_n2(self):
        model = _model()
        h = model.init_hidden(2)
        Q = model.aggregate_q(h)
        assert Q.shape == torch.Size([]), f"Q should be scalar, got {Q.shape}"

    def test_aggregate_q_shape_n8(self):
        """N=8 should produce the same output shape as N=2 (N-independence)."""
        model = _model()
        h2 = model.init_hidden(2)
        h8 = model.init_hidden(8)
        Q2 = model.aggregate_q(h2)
        Q8 = model.aggregate_q(h8)
        assert Q2.shape == Q8.shape, \
            f"Q shape should be N-independent: n=2→{Q2.shape}, n=8→{Q8.shape}"

    def test_forward_full_shot(self):
        """5-event shot: 2 ball_ball + 2 cushion + 1 pocket."""
        model  = _model()
        n_balls = 4
        events = [
            _ball_ball_ev(0, 1),
            _ball_ball_ev(1, 2),
            _single_ev(0, event_type=1),    # cue_linear
            _single_ev(2, event_type=5),    # tgt_linear
            _single_ev(3, event_type=EVENT_POCKET),
        ]

        out = model.forward(n_balls, events)

        assert isinstance(out, RSSMOutput)
        assert len(out.event_outputs) == 5

        # ball_ball events have both delta_j / type_j
        assert out.event_outputs[0].delta_j is not None
        assert out.event_outputs[0].type_j  is not None

        # single events have None for j
        assert out.event_outputs[2].delta_j is None
        assert out.event_outputs[2].type_j  is None

        # Q and h_final shapes
        assert out.Q.shape == torch.Size([])
        assert isinstance(out.h_final, list) and len(out.h_final) == n_balls
        assert all(t.shape == (H_DIM,) for t in out.h_final)

    def test_forward_gradients_flow(self):
        """Backprop through the full shot should not error."""
        model  = _model()
        events = [_ball_ball_ev(0, 1), _single_ev(0, 1)]
        out    = model.forward(2, events)
        loss   = out.Q + sum(e.delta_i.sum() for e in out.event_outputs)
        loss.backward()   # should not raise


# ── TestAutogradSafety ────────────────────────────────────────────────────────

class TestAutogradSafety:
    """
    Verify that list-based h does not need h.clone().

    Old (Tensor) h required clone() because h[i]=x is an in-place tensor op
    and would corrupt the autograd version counter.

    New (list) h uses Python __setitem__, which only swaps a pointer —
    no Tensor is modified in-place, so autograd is unaffected.
    """

    def test_no_inplace_autograd_error(self):
        """
        3 sequential events on the same balls must not raise
        "RuntimeError: in-place operation on tensor used for gradient".
        """
        model = _model()
        h = model.init_hidden(4)
        events = [
            _ball_ball_ev(0, 1),
            _ball_ball_ev(1, 2),
            _ball_ball_ev(0, 2),   # ball 0 appears in events 1 and 3
        ]
        out  = model.forward(4, events)
        loss = out.Q + sum(e.delta_i.sum() for e in out.event_outputs)
        loss.backward()   # must not raise

    def test_old_tensor_not_mutated_by_list_assignment(self):
        """
        After h[i] = new_tensor, the tensor that h[i] used to point to
        must be unchanged.  If h were a Tensor and we did h[i]=x in-place,
        the original data at row i would be overwritten.
        """
        model = _model()
        h = model.init_hidden(4)

        # Capture the actual tensor object (not a copy) before the step
        old_h0 = h[0]
        old_val = old_h0.detach().clone()   # numeric snapshot

        model.step_ball_ball(h, 0, 1, _node(), _node(), _edge())

        # The list now points to a new tensor at index 0
        assert not torch.allclose(h[0], old_val), "h[0] should have been replaced"
        # The original tensor object is untouched
        assert torch.allclose(old_h0, old_val), \
            "list assignment must not mutate the tensor that h[0] previously pointed to"

    def test_gradient_chains_through_sequential_events_same_ball(self):
        """
        Ball 0 is involved in event 1, then again in event 2.
        Gradient from event-2 loss must flow back through event-1's h update.

        If list assignment broke the autograd graph, h[0]-after-event-1
        would have no gradient (the chain would be cut).
        """
        model = _model()
        h = model.init_hidden(3)

        # Event 1: ball 0 ↔ ball 1
        h, _, _, _, _ = model.step_ball_ball(h, 0, 1, _node(), _node(), _edge())
        h0_mid = h[0]
        h0_mid.retain_grad()   # keep grad for this non-leaf tensor

        # Event 2: ball 0 ↔ ball 2 — consumes h0_mid
        h, d0_ev2, _, _, _ = model.step_ball_ball(h, 0, 2, _node(), _node(), _edge())

        d0_ev2.sum().backward()

        assert h0_mid.grad is not None, \
            "gradient must propagate back to h[0] after event 1 (chain intact)"
        assert h0_mid.grad.abs().sum() > 0, \
            "gradient through h[0] (event-1 output) must be non-zero"

    def test_uninvolved_ball_h_has_no_gradient(self):
        """
        Ball 2 is never involved in any event.
        Its latent h[2] should receive no gradient.

        Init zeros have requires_grad=False, so we inject a requires_grad
        sentinel to make the gradient check possible.
        """
        model = _model()
        h = model.init_hidden(3)

        # Inject a differentiable sentinel for ball 2 (the uninvolved ball)
        sentinel = torch.zeros(H_DIM, requires_grad=True)
        h[2] = sentinel

        h, d0, _, _, _ = model.step_ball_ball(h, 0, 1, _node(), _node(), _edge())
        d0.sum().backward()

        # sentinel (= h[2]) never entered any computation — grad must be None
        assert sentinel.grad is None, \
            "uninvolved ball must not receive any gradient"

    def test_msg_mlp_grad_nonzero_from_early_event(self):
        """
        Use only the first event's delta as the loss.
        msg_mlp must still get a non-zero gradient (gradient flows from event 1).
        """
        model = _model()
        events = [_ball_ball_ev(0, 1), _ball_ball_ev(1, 2), _ball_ball_ev(2, 3)]
        out = model.forward(4, events)

        # Loss comes only from the first event
        loss = out.event_outputs[0].delta_i.sum()
        loss.backward()

        w_grad = model.msg_mlp[0].weight.grad
        assert w_grad is not None
        assert w_grad.abs().sum() > 0, \
            "msg_mlp must receive gradient from event-1 loss"

    def test_list_h_is_mutated_in_place(self):
        """
        step_ball_ball returns the *same* list object (not a copy).
        Callers rely on this: h, ... = model.step_ball_ball(h, ...) is idiomatic.
        """
        model = _model()
        h = model.init_hidden(4)
        h_id = id(h)
        h_ret, *_ = model.step_ball_ball(h, 0, 1, _node(), _node(), _edge())
        assert id(h_ret) == h_id, "step_ball_ball must return the same list object"
