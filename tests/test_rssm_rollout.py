"""
tests/test_rssm_rollout.py

Tests for RolloutEngine (world_model/rssm_rollout.py).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import pooltool.constants as const

from world_model.rssm_model import RSSMModel, H_DIM, EVENT_BALL_BALL, EVENT_POCKET
from world_model.event_detector import EventResult
from world_model.ball_motion import FrictionParams
from world_model.rssm_rollout import (
    RolloutEngine, RolloutResult,
    rvw_to_phys_feat, make_node, make_edge, advance_balls,
)

PARAMS = FrictionParams(u_s=0.2, u_sp=0.0127, u_r=0.01, R=0.028575, m=0.170097, g=9.81)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _rvw(px=0.5, py=0.5, vx=0.5, vy=0.0) -> np.ndarray:
    return np.array([[px, py, 0.0], [vx, vy, 0.0], [0.0, 0.0, 0.0]])


def _balls_dict() -> dict:
    return {
        "cue": (_rvw(0.3, 0.5, 0.8, 0.0), const.sliding),
        "1":   (_rvw(0.7, 0.5, 0.0, 0.0), const.stationary),
    }


def _mock_agent(ball_id: str):
    return SimpleNamespace(id=ball_id, agent_type="ball")


def _mock_result(event_type: int, ball_ids: list[str], time: float = 0.05) -> EventResult:
    """Create a fake EventResult for unit tests."""
    raw = SimpleNamespace(agents=[_mock_agent(bid) for bid in ball_ids])
    r   = EventResult.__new__(EventResult)
    r.event_type     = event_type
    r.time           = time
    r.contact_normal = np.array([1.0, 0.0], dtype=np.float32)
    r.raw_event      = raw
    return r


def _stop_result() -> EventResult:
    r = EventResult.__new__(EventResult)
    r.event_type     = -1
    r.time           = 0.0
    r.contact_normal = np.array([0.0, 0.0], dtype=np.float32)
    r.raw_event      = SimpleNamespace(agents=[])
    return r


class MockDetector:
    """Returns a scripted sequence of EventResults, then stop."""
    def __init__(self, results: list[EventResult]):
        self._results = list(results)
        self._idx = 0

    def next_collision(self, balls):
        if self._idx >= len(self._results):
            return _stop_result()
        r = self._results[self._idx]
        self._idx += 1
        return r


def _model() -> RSSMModel:
    return RSSMModel(h_dim=H_DIM, hidden=[64, 64])


def _engine(mock_results: list[EventResult]) -> RolloutEngine:
    return RolloutEngine(
        model      = _model(),
        detector   = MockDetector(mock_results),
        ball_ids   = ["cue", "1"],
        params     = PARAMS,
        max_events = 10,
    )


# ── TestHelpers ───────────────────────────────────────────────────────────────

class TestHelpers:
    def test_rvw_to_phys_feat_shape(self):
        feat = rvw_to_phys_feat(_rvw())
        assert feat.shape == (7,)

    def test_make_node_shape_and_type_onehot(self):
        node = make_node(_rvw(), event_type=1)
        assert node.shape == (14,)
        # type_onehot starts at index 7
        assert node[7 + 1].item() == pytest.approx(1.0)
        # other type slots are zero
        for k in range(7):
            if k != 1:
                assert node[7 + k].item() == pytest.approx(0.0)

    def test_make_edge_shape(self):
        edge = make_edge(_rvw(0.3, 0.5), _rvw(0.7, 0.5), np.array([1.0, 0.0]))
        assert edge.shape == (9,)

    def test_advance_balls_moves_ball(self):
        balls = {"cue": (_rvw(0.3, 0.5, 0.8, 0.0), const.sliding)}
        updated = advance_balls(balls, dt=0.1, params=PARAMS)
        pos_before = balls["cue"][0][0, 0]
        pos_after  = updated["cue"][0][0, 0]
        assert pos_after > pos_before, "ball with positive velocity should move forward"


# ── TestRolloutEngineUnit ─────────────────────────────────────────────────────

class TestRolloutEngineUnit:
    def test_run_result_type(self):
        engine = _engine([_mock_result(EVENT_BALL_BALL, ["cue", "1"])])
        result = engine.run(_balls_dict())
        assert isinstance(result, RolloutResult)

    def test_run_event_count(self):
        """2 mock events → 2 event_steps."""
        engine = _engine([
            _mock_result(EVENT_BALL_BALL, ["cue", "1"]),
            _mock_result(1, ["cue"]),   # cue_linear
        ])
        result = engine.run(_balls_dict())
        assert len(result.event_steps) == 2
        assert len(result.event_outputs) == 2

    def test_run_q_is_scalar(self):
        engine = _engine([_mock_result(EVENT_BALL_BALL, ["cue", "1"])])
        result = engine.run(_balls_dict())
        assert result.Q.shape == torch.Size([])

    def test_run_h_final_shape(self):
        """h_final is a list of per-ball (H_DIM,) tensors, not a stacked tensor."""
        engine = _engine([_mock_result(EVENT_BALL_BALL, ["cue", "1"])])
        result = engine.run(_balls_dict())
        assert isinstance(result.h_final, list)
        assert len(result.h_final) == 2
        assert all(h.shape == (H_DIM,) for h in result.h_final)


# ── TestRolloutEngineIntegration ──────────────────────────────────────────────

class TestRolloutEngineIntegration:
    def test_integration_real_shot(self):
        """Full pipeline with real BilliardsEnv + EventDetector."""
        import pooltool as pt
        from simulator import BilliardsEnv

        env = BilliardsEnv(n_balls=1)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        sys = env.system

        # Post-strike state
        stick_ev  = sys.events[1]
        cue_agent = next(a for a in stick_ev.agents if getattr(a, "agent_type", "") == "ball")
        cue_rvw   = np.array([
            cue_agent.initial.state.rvw[0],
            list(cue_agent.final.vel),
            list(cue_agent.final.avel),
        ], dtype=np.float64)
        tgt_rvw        = sys.balls["1"].history[0].rvw.copy()
        tgt_rvw[1]     = 0.0
        tgt_rvw[2]     = 0.0

        balls_dict = {
            "cue": (cue_rvw, const.sliding),
            "1":   (tgt_rvw, const.stationary),
        }

        from world_model.event_detector import EventDetector
        detector = EventDetector(sys.table, sys.cue, PARAMS)
        engine   = RolloutEngine(
            model      = _model(),
            detector   = detector,
            ball_ids   = ["cue", "1"],
            params     = PARAMS,
            max_events = 30,
        )

        result = engine.run(balls_dict)

        assert len(result.event_steps) >= 1,       "should have at least one event"
        assert result.Q.shape == torch.Size([]),   "Q should be scalar"
        assert isinstance(result.h_final, list) and len(result.h_final) == 2, \
            "h_final should be a list of 2 per-ball tensors"
        assert all(h.shape == (H_DIM,) for h in result.h_final), "h_final wrong shape"

        env.close()
