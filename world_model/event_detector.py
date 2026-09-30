"""
world_model/event_detector.py

Stage 1 event detector — given post-collision ball states, predict the
next physical event (type, time, contact normal) using pooltool's own
collision solvers with configurable friction parameters.

The only learnable knobs are (u_s, u_r, u_sp); all geometry is exact.

Usage:
    detector = EventDetector(table, params=FrictionParams(u_s=0.2, ...))
    result   = detector.next_event(balls_dict)
    # result.event_type, result.time, result.contact_normal
"""

from __future__ import annotations

import numpy as np
import pooltool as pt
import pooltool.constants as const
from pooltool.objects.ball.datatypes import BallState as PtBallState
from pooltool.objects.ball.params import BallParams
from pooltool.evolution.event_based.simulate import get_next_event
from pooltool.evolution.event_based.cache import CollisionCache, TransitionCache
from pooltool.system.datatypes import System

from world_model.ball_motion import FrictionParams, DEFAULT_FRICTION


COLL_TYPE = {
    "stick_ball":            -1,   # treated as transition; rollout starts post-strike
    "ball_ball":             0,
    "ball_linear_cushion":   1,
    "ball_circular_cushion": 2,
    "ball_pocket":           3,
    "sliding_rolling":       -1,   # transition (not a collision)
    "rolling_spinning":      -1,
    "spinning_stationary":   -1,
    "none":                  -1,
}

TGT_LINEAR  = 5
TGT_CIRCULAR = 6


def _contact_normal_lcushion(lx: float, ly: float) -> np.ndarray:
    n = np.array([lx, ly], dtype=np.float32)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)


def _contact_normal_ccushion(cx: float, cy: float, ball_pos: np.ndarray) -> np.ndarray:
    d = ball_pos[:2] - np.array([cx, cy])
    norm = np.linalg.norm(d)
    return (d / norm).astype(np.float32) if norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)


def _contact_normal_pocket(px: float, py: float, ball_pos: np.ndarray) -> np.ndarray:
    d = np.array([px, py]) - ball_pos[:2]
    norm = np.linalg.norm(d)
    return (d / norm).astype(np.float32) if norm > 1e-9 else np.array([0.0, -1.0], dtype=np.float32)


def _contact_normal_ball_ball(pos_cue: np.ndarray, pos_tgt: np.ndarray) -> np.ndarray:
    d = pos_tgt[:2] - pos_cue[:2]
    norm = np.linalg.norm(d)
    return (d / norm).astype(np.float32) if norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)


def _ball_sort_key(ball_id: str):
    """cue always sorts first; other balls sort by numeric id. Keeps the
    historical cue->target normal direction for cue-target collisions while
    giving target-target pairs a well-defined (lower id -> higher id) order."""
    return (0, "") if ball_id == "cue" else (1, int(ball_id))


def _ball_pos_at(ball: pt.Ball, t: float, params: "FrictionParams") -> np.ndarray:
    """Ball center 3-D position at time t using our friction params."""
    import pooltool.physics as ph
    rvw, _ = ph.evolve_ball_motion(
        ball.state.s, ball.state.rvw,
        R=params.R, m=params.m, u_s=params.u_s, u_sp=params.u_sp, u_r=params.u_r, g=params.g,
        t=t,
    )
    return rvw[0]


class EventResult:
    __slots__ = ("event_type", "time", "contact_normal", "raw_event")

    def __init__(self, event_type: int, time: float, contact_normal: np.ndarray, raw_event=None):
        self.event_type     = event_type
        self.time           = time
        self.contact_normal = contact_normal
        self.raw_event      = raw_event

    def __repr__(self):
        return (f"EventResult(type={self.event_type}, t={self.time:.6f}, "
                f"normal={self.contact_normal})")


class EventDetector:
    """
    Parameters
    ----------
    table  : pooltool Table object (provides cushion/pocket geometry)
    cue    : pooltool Cue object   (needed to construct System)
    params : FrictionParams        (learnable μ_s, μ_r, μ_sp)
    """

    def __init__(
        self,
        table,
        cue,
        params: FrictionParams = DEFAULT_FRICTION,
    ):
        self.table  = table
        self.cue    = cue
        self.params = params
        self._ball_params = BallParams(
            m=params.m,
            R=params.R,
            u_s=params.u_s,
            u_r=params.u_r,
            u_sp_proportionality=params.u_sp / params.R,
            g=params.g,
        )

    # ------------------------------------------------------------------

    def _make_pt_ball(self, ball_id: str, rvw: np.ndarray, state: int) -> pt.Ball:
        return pt.Ball(
            id=ball_id,
            state=PtBallState(rvw=np.array(rvw, dtype=np.float64), s=state),
            params=self._ball_params,
        )

    def next_collision(
        self,
        balls: dict[str, tuple[np.ndarray, int]],
        _t_offset: float = 0.0,
        _depth: int = 0,
        max_transitions: int = 100,
    ) -> EventResult:
        """Like next_event but skips transition events (sliding_rolling etc.).
        Returns EventResult with time measured from the original t_offset=0.
        Returns event_type=-1 if max_transitions exceeded (all balls stopped).
        """
        if _depth >= max_transitions:
            return EventResult(-1, _t_offset, np.array([0.0, 0.0], dtype=np.float32))

        result = self.next_event(balls)
        if result.event_type != -1:
            return EventResult(
                result.event_type,
                _t_offset + result.time,
                result.contact_normal,
                result.raw_event,
            )

        # Advance ball states to the transition time and recurse
        import pooltool.physics as ph
        t_trans = result.time
        updated_balls: dict[str, tuple[np.ndarray, int]] = {}
        for bid, (rvw, s) in balls.items():
            rvw_t, s_t = ph.evolve_ball_motion(
                s, np.array(rvw, dtype=np.float64),
                R=self.params.R, m=self.params.m,
                u_s=self.params.u_s, u_sp=self.params.u_sp,
                u_r=self.params.u_r, g=self.params.g,
                t=t_trans,
            )
            updated_balls[bid] = (rvw_t, s_t)
        return self.next_collision(
            updated_balls,
            _t_offset=_t_offset + t_trans,
            _depth=_depth + 1,
            max_transitions=max_transitions,
        )

    def _classify_event(self, ev, pos_fn) -> tuple[int, np.ndarray]:
        """
        Shared classification logic for next_event() / full_event_sequence().

        pos_fn(ball_id) -> (3,) rvw-style position array at collision time.
        Returns (raw_type, contact_normal). Assumes `ev` is already known to
        be a real collision (caller has filtered out transition events).

        N-ball generalization: ball_ball events sort their two participants
        with cue first (else by numeric id) so the same code path handles
        cue-target AND target-target collisions with the correct normal.
        Single-ball events (cushion/pocket) key off "is this the cue or some
        target ball" — with N>1 targets there's no single "the target", so
        the TGT_LINEAR/TGT_CIRCULAR reclassification only needs `not has_cue`.
        """
        et = str(ev.event_type)
        raw_type = COLL_TYPE[et]

        ball_ids  = {a.id for a in ev.agents if getattr(a, "agent_type", "") == "ball"}
        other_ids = {(getattr(a, "agent_type", ""), a.id)
                     for a in ev.agents if getattr(a, "agent_type", "") != "ball"}

        normal = np.array([1.0, 0.0], dtype=np.float32)

        if et == "ball_ball":
            id_lo, id_hi = sorted(ball_ids, key=_ball_sort_key)
            normal = _contact_normal_ball_ball(pos_fn(id_lo), pos_fn(id_hi))
            return raw_type, normal

        main_id  = next(iter(ball_ids))
        has_cue  = main_id == "cue"
        main_pos = pos_fn(main_id)

        if et == "ball_linear_cushion":
            for atype, aid in other_ids:
                if atype == "linear_cushion_segment":
                    cush = self.table.cushion_segments.linear[aid]
                    normal = _contact_normal_lcushion(cush.lx, cush.ly)
                    break
            if not has_cue:
                raw_type = TGT_LINEAR

        elif et == "ball_circular_cushion":
            for atype, aid in other_ids:
                if atype == "circular_cushion_segment":
                    cush = self.table.cushion_segments.circular[aid]
                    normal = _contact_normal_ccushion(cush.a, cush.b, main_pos)
                    break
            if not has_cue:
                raw_type = TGT_CIRCULAR

        elif et == "ball_pocket":
            for atype, aid in other_ids:
                if atype == "pocket":
                    pock = self.table.pockets[aid]
                    normal = _contact_normal_pocket(pock.a, pock.b, main_pos)
                    break

        return raw_type, normal

    def next_event(
        self,
        balls: dict[str, tuple[np.ndarray, int]],
    ) -> EventResult:
        """
        Parameters
        ----------
        balls : { ball_id: (rvw, pooltool_state) }
            e.g. {'cue': (rvw_cue, const.sliding), '1': (rvw_tgt, const.stationary)}

        Returns
        -------
        EventResult  with .event_type (int 0-6), .time (float), .contact_normal
        """
        pt_balls = {bid: self._make_pt_ball(bid, rvw, s) for bid, (rvw, s) in balls.items()}
        system   = System(cue=self.cue, table=self.table, balls=pt_balls, t=0.0)

        ev = get_next_event(system)
        et = str(ev.event_type)

        if et not in COLL_TYPE or COLL_TYPE[et] == -1:
            return EventResult(-1, ev.time, np.array([0.0, 0.0], dtype=np.float32), ev)

        def pos_fn(bid: str) -> np.ndarray:
            return _ball_pos_at(system.balls[bid], ev.time, self.params)

        raw_type, normal = self._classify_event(ev, pos_fn)
        return EventResult(raw_type, ev.time, normal, ev)

    # ------------------------------------------------------------------

    def full_event_sequence(
        self,
        balls: dict[str, tuple[np.ndarray, int]],
        max_events: int = 30,
    ) -> list[EventResult]:
        """
        Simulate the complete event sequence until all balls stop.
        Skips transition events; returns only physical collisions.
        """
        from pooltool.evolution.event_based.simulate import simulate as pt_simulate
        from pooltool.evolution.event_based.config import INCLUDED_EVENTS
        from pooltool.events.datatypes import EventType

        pt_balls = {bid: self._make_pt_ball(bid, rvw, s) for bid, (rvw, s) in balls.items()}
        system   = System(cue=self.cue, table=self.table, balls=pt_balls, t=0.0)

        pt_simulate(system, inplace=True, max_events=max_events)

        results = []
        for ev in system.events:
            et = str(ev.event_type)
            if et not in COLL_TYPE or COLL_TYPE[et] == -1:
                continue

            ball_ids = {a.id for a in ev.agents if getattr(a, "agent_type", "") == "ball"}
            # A pocketed ball is removed from system.balls by the time the
            # full sim finishes, so its post-hoc position is unavailable.
            if not ball_ids.issubset(system.balls.keys()):
                continue

            def pos_fn(bid: str) -> np.ndarray:
                return system.balls[bid].state.rvw[0]

            raw_type, normal = self._classify_event(ev, pos_fn)
            results.append(EventResult(raw_type, ev.time, normal, ev))

        return results
