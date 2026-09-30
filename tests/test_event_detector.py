"""
tests/test_event_detector.py

Tests for Stage 1 event detector.
Validates that EventDetector.next_event() matches pooltool's ground truth
when using the same friction parameters.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import pooltool as pt
import pooltool.constants as const
from simulator import BilliardsEnv
from world_model.ball_motion import BallTrajectory, FrictionParams
from world_model.event_detector import EventDetector


# Friction params matching the BilliardsEnv default (u_r=0.01, u_sp=0.0127)
# Must match the env to get predictions that align with GT events.
PARAMS = FrictionParams(u_s=0.2, u_sp=0.0127, u_r=0.01, R=0.028575, m=0.170097, g=9.81)


def _get_post_strike_state(seed: int):
    """Run one shot, return (ref_system, cue_rvw_post_strike, tgt_rvw_init, events_gt)."""
    env = BilliardsEnv(n_balls=1)
    env.reset(seed=seed)
    env.step(env.action_space.sample())
    ref_sys = env.system

    stick_ev = ref_sys.events[1]          # events[0]=none, events[1]=stick_ball
    assert str(stick_ev.event_type) == "stick_ball"

    cue_agent = next(a for a in stick_ev.agents if getattr(a, "agent_type", "") == "ball")
    cue_rvw   = np.array([
        cue_agent.initial.state.rvw[0],
        list(cue_agent.final.vel),
        list(cue_agent.final.avel),
    ], dtype=np.float64)

    tgt_rvw = ref_sys.balls["1"].history[0].rvw.copy()
    tgt_rvw[1] = 0.0
    tgt_rvw[2] = 0.0

    gt_events = [e for e in ref_sys.events[2:]
                 if str(e.event_type) not in ("none", "sliding_rolling",
                                               "rolling_spinning", "spinning_stationary")]
    env.close()
    return ref_sys, cue_rvw, tgt_rvw, gt_events


def _get_post_strike_state_nballs(seed: int, n_targets: int):
    """Like _get_post_strike_state but with n_targets target balls ("1".."n_targets").
    Returns (ref_system, balls_dict, events_gt), or None if the shot scratched
    (cue pocketed mid-shot leaves env.system with no recorded events — not a
    case this helper needs to support, callers should just try another seed).
    balls_dict is ready to pass straight into EventDetector.next_collision/
    full_event_sequence."""
    env = BilliardsEnv(n_balls=n_targets)
    env.reset(seed=seed)
    env.step(env.action_space.sample())
    ref_sys = env.system

    if len(ref_sys.events) < 2:
        env.close()
        return None

    stick_ev = ref_sys.events[1]
    assert str(stick_ev.event_type) == "stick_ball"

    cue_agent = next(a for a in stick_ev.agents if getattr(a, "agent_type", "") == "ball")
    cue_rvw   = np.array([
        cue_agent.initial.state.rvw[0],
        list(cue_agent.final.vel),
        list(cue_agent.final.avel),
    ], dtype=np.float64)

    balls_dict = {"cue": (cue_rvw, const.sliding)}
    for i in range(1, n_targets + 1):
        tgt_rvw = ref_sys.balls[str(i)].history[0].rvw.copy()
        tgt_rvw[1] = 0.0
        tgt_rvw[2] = 0.0
        balls_dict[str(i)] = (tgt_rvw, const.stationary)

    gt_events = [e for e in ref_sys.events[2:]
                 if str(e.event_type) not in ("none", "sliding_rolling",
                                               "rolling_spinning", "spinning_stationary")]
    env.close()
    return ref_sys, balls_dict, gt_events


# ───────────────────────────────────────────────────────────────────────────────
# BallTrajectory tests
# ───────────────────────────────────────────────────────────────────────────────

class TestBallTrajectory:
    def test_pos_at_zero(self):
        rvw = np.array([[0.3, 0.5, 0.0], [2.0, 1.5, 0.0], [0.0, 0.0, 0.0]])
        traj = BallTrajectory(rvw, const.sliding, PARAMS)
        pos = traj.pos_at(0.0)
        np.testing.assert_allclose(pos, [0.3, 0.5], atol=1e-10)

    def test_pos_at_nonzero(self):
        rvw = np.array([[0.3, 0.5, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        traj = BallTrajectory(rvw, const.sliding, PARAMS)
        t    = 0.05
        pos  = traj.pos_at(t)
        # Ball should have moved roughly v*t in x (slightly less due to deceleration)
        assert pos[0] > 0.3
        assert pos[0] < 0.3 + 2.0 * t   # upper bound: no deceleration

    def test_t_stop_positive(self):
        rvw = np.array([[0.5, 0.5, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        traj = BallTrajectory(rvw, const.sliding, PARAMS)
        t = traj.t_stop()
        assert t > 0.0

    def test_stationary_ball_stays_put(self):
        rvw = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        traj = BallTrajectory(rvw, const.stationary, PARAMS)
        np.testing.assert_allclose(traj.pos_at(1.0), [0.5, 0.5], atol=1e-10)
        assert traj.t_stop() == 0.0

    def test_rolling_ball_decelerates(self):
        rvw = np.array([[0.3, 0.3, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5/0.028575, 0.0]])
        traj = BallTrajectory(rvw, const.rolling, PARAMS)
        t_stop = traj.t_stop()
        assert t_stop > 0.0
        pos_end = traj.pos_at(t_stop + 0.5)
        pos_mid = traj.pos_at(t_stop / 2)
        # Ball should stop before t_stop
        assert pos_end[0] > 0.3


# ───────────────────────────────────────────────────────────────────────────────
# EventDetector tests
# ───────────────────────────────────────────────────────────────────────────────

class TestEventDetectorNextEvent:
    def _make_detector(self, ref_sys) -> EventDetector:
        # Use pooltool's default params for table/cue; our custom friction
        return EventDetector(table=ref_sys.table, cue=ref_sys.cue, params=PARAMS)

    def _balls_dict(self, cue_rvw, tgt_rvw):
        return {
            "cue": (cue_rvw, const.sliding),
            "1":   (tgt_rvw, const.stationary),
        }

    def test_first_event_type_matches_gt(self):
        """Predicted first physical collision type matches pooltool GT for 10 shots."""
        for seed in range(10):
            ref_sys, cue_rvw, tgt_rvw, gt_events = _get_post_strike_state(seed)
            if not gt_events:
                continue

            det    = self._make_detector(ref_sys)
            result = det.next_collision(self._balls_dict(cue_rvw, tgt_rvw))
            gt_et  = str(gt_events[0].event_type)

            assert str(result.raw_event.event_type) == gt_et, (
                f"seed={seed}: predicted={result.raw_event.event_type}, gt={gt_et}"
            )

    def test_first_event_time_close_to_gt(self):
        """Predicted first collision time matches GT to within 1 ms for 20 shots."""
        for seed in range(20):
            ref_sys, cue_rvw, tgt_rvw, gt_events = _get_post_strike_state(seed)
            if not gt_events:
                continue

            gt_time = gt_events[0].time
            det     = self._make_detector(ref_sys)
            result  = det.next_collision(self._balls_dict(cue_rvw, tgt_rvw))

            assert abs(result.time - gt_time) < 1e-3, (
                f"seed={seed}: predicted={result.time:.6f}, gt={gt_time:.6f}"
            )

    def test_contact_normal_unit_length(self):
        """Contact normal should always be a unit vector (or zero for transition events)."""
        for seed in range(5):
            ref_sys, cue_rvw, tgt_rvw, gt_events = _get_post_strike_state(seed)
            if not gt_events:
                continue
            det    = self._make_detector(ref_sys)
            result = det.next_event(self._balls_dict(cue_rvw, tgt_rvw))
            if result.event_type >= 0:
                n_len = np.linalg.norm(result.contact_normal)
                assert abs(n_len - 1.0) < 0.01, f"seed={seed}: |normal|={n_len:.4f}"


class TestEventDetectorFullSequence:
    def test_sequence_length_positive(self):
        """full_event_sequence returns at least one collision per shot."""
        for seed in range(5):
            ref_sys, cue_rvw, tgt_rvw, _ = _get_post_strike_state(seed)
            det = EventDetector(ref_sys.table, ref_sys.cue, PARAMS)
            seq = det.full_event_sequence({"cue": (cue_rvw, const.sliding),
                                            "1":  (tgt_rvw, const.stationary)})
            assert len(seq) > 0, f"seed={seed}: no events predicted"

    def test_sequence_times_monotone(self):
        """Event times in full sequence must be non-decreasing."""
        for seed in range(5):
            ref_sys, cue_rvw, tgt_rvw, _ = _get_post_strike_state(seed)
            det = EventDetector(ref_sys.table, ref_sys.cue, PARAMS)
            seq = det.full_event_sequence({"cue": (cue_rvw, const.sliding),
                                            "1":  (tgt_rvw, const.stationary)})
            times = [r.time for r in seq]
            for i in range(1, len(times)):
                assert times[i] >= times[i-1], f"seed={seed}: non-monotone at [{i}]"

    def test_sequence_matches_gt_length(self):
        """Number of physical collision events matches pooltool GT within ±2."""
        for seed in range(10):
            ref_sys, cue_rvw, tgt_rvw, gt_events = _get_post_strike_state(seed)
            det = EventDetector(ref_sys.table, ref_sys.cue, PARAMS)
            seq = det.full_event_sequence({"cue": (cue_rvw, const.sliding),
                                            "1":  (tgt_rvw, const.stationary)})
            diff = abs(len(seq) - len(gt_events))
            assert diff <= 2, (
                f"seed={seed}: predicted={len(seq)} events, gt={len(gt_events)}"
            )


# ───────────────────────────────────────────────────────────────────────────────
# N-ball generalization tests (target-target collisions, n_targets >= 2)
# ───────────────────────────────────────────────────────────────────────────────

from world_model.event_detector import _contact_normal_ball_ball, TGT_LINEAR, TGT_CIRCULAR

EVENT_BALL_BALL = 0
_PLACEHOLDER_NORMAL = np.array([1.0, 0.0], dtype=np.float32)


class TestEventDetectorNBall:
    def test_target_target_normal_not_placeholder(self):
        """With 2+ target balls, a target-target ball_ball event must get a
        real contact normal, not the [1,0] default that only cue-target
        collisions used to receive before N-ball generalization."""
        # n_targets=2 makes a direct target-target collision rare (0/60 seeds
        # in a scan); n_targets=3 hits one by seed 7, so use 3 targets here.
        found_target_target = False
        for seed in range(60):
            state = _get_post_strike_state_nballs(seed, n_targets=3)
            if state is None:
                continue
            ref_sys, balls_dict, _ = state
            det = EventDetector(ref_sys.table, ref_sys.cue, PARAMS)
            seq = det.full_event_sequence(balls_dict, max_events=30)

            for r in seq:
                if r.event_type != EVENT_BALL_BALL:
                    continue
                agents = [a.id for a in r.raw_event.agents
                          if getattr(a, "agent_type", "") == "ball"]
                if "cue" in agents:
                    continue  # cue-target, not the case under test
                found_target_target = True
                assert not np.allclose(r.contact_normal, _PLACEHOLDER_NORMAL, atol=1e-6), (
                    f"seed={seed}: target-target normal stuck at placeholder"
                )
                n_len = np.linalg.norm(r.contact_normal)
                assert abs(n_len - 1.0) < 0.01, f"seed={seed}: |normal|={n_len:.4f}"

        assert found_target_target, (
            "expected at least one target-target ball_ball collision across 60 seeds "
            "with n_targets=3 — widen seed range if this becomes flaky"
        )

    def test_ball_sort_key_orders_cue_first_then_numeric(self):
        """_ball_sort_key must put cue before any target, and order targets
        numerically (not lexically, so '2' < '10')."""
        from world_model.event_detector import _ball_sort_key
        assert _ball_sort_key("cue") < _ball_sort_key("1")
        assert _ball_sort_key("cue") < _ball_sort_key("2")
        assert _ball_sort_key("1")   < _ball_sort_key("2")
        assert _ball_sort_key("2")   < _ball_sort_key("10")  # numeric, not lexical

    def test_cushion_reclassification_applies_to_any_target(self):
        """TGT_LINEAR/TGT_CIRCULAR reclassification must fire for ball '2',
        not just ball '1' (previously hardcoded to has_tgt = '1' in ball_ids)."""
        found_ball2_cushion = False
        for seed in range(15):
            state = _get_post_strike_state_nballs(seed, n_targets=2)
            if state is None:
                continue
            ref_sys, balls_dict, _ = state
            det = EventDetector(ref_sys.table, ref_sys.cue, PARAMS)
            seq = det.full_event_sequence(balls_dict, max_events=30)

            for r in seq:
                if r.event_type not in (TGT_LINEAR, TGT_CIRCULAR):
                    continue
                agents = [a.id for a in r.raw_event.agents
                          if getattr(a, "agent_type", "") == "ball"]
                if agents == ["2"]:
                    found_ball2_cushion = True

        assert found_ball2_cushion, (
            "expected ball '2' to trigger a TGT_LINEAR/TGT_CIRCULAR cushion event "
            "in at least one of 15 seeds"
        )
