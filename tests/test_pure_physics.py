"""
tests/test_pure_physics.py

Differential tests: pure_physics.py vs pooltool (oracle).

Layer 1 — ball_motion: evolve_ball_motion, t_stop
Layer 2 — collision times: ball_ball, linear_cushion, circular_cushion, pocket
Layer 3 — end-to-end event sequence match
"""

import numpy as np
import pytest
import pooltool as pt
import pooltool.physics as ph
import pooltool.constants as const
from pooltool.evolution.event_based import solve as pt_solve

from world_model.pure_physics import (
    evolve_ball_motion, t_stop,
    ball_ball_collision_time,
    ball_linear_cushion_time,
    ball_circular_cushion_time,
    ball_pocket_time,
    SLIDING, ROLLING, SPINNING, STATIONARY,
)
from world_model.ball_motion import DEFAULT_FRICTION as P
from world_model.event_detector import EventDetector

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rvw(px=0.4, py=0.5, vx=0.5, vy=0.3, wx=0.1, wy=0.2, wz=2.0) -> np.ndarray:
    r = np.zeros((3, 3), dtype=np.float64)
    r[0] = [px, py, 0.0]
    r[1] = [vx, vy, 0.0]
    r[2] = [wx, wy, wz]
    return r


def _pt_evolve(state, rvw, t):
    return ph.evolve_ball_motion(
        state, rvw, R=P.R, m=P.m, u_s=P.u_s, u_sp=P.u_sp, u_r=P.u_r, g=P.g, t=t
    )


def _our_evolve(state, rvw, t):
    return evolve_ball_motion(
        state, rvw, R=P.R, m=P.m, u_s=P.u_s, u_sp=P.u_sp, u_r=P.u_r, g=P.g, t=t
    )


# ── Layer 1: ball_motion ──────────────────────────────────────────────────────

@pytest.mark.parametrize("dt", [0.0, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0])
def test_evolve_sliding(dt):
    rvw = _rvw()
    exp_rvw, exp_s = _pt_evolve(const.sliding, rvw.copy(), dt)
    act_rvw, act_s = _our_evolve(SLIDING, rvw.copy(), dt)
    np.testing.assert_allclose(act_rvw, exp_rvw, atol=1e-8, err_msg=f"dt={dt}")
    assert act_s == exp_s


@pytest.mark.parametrize("dt", [0.0, 0.01, 0.05, 0.1, 0.5])
def test_evolve_rolling(dt):
    rvw = np.zeros((3, 3), dtype=np.float64)
    rvw[0] = [0.3, 0.4, 0.0]
    rvw[1] = [0.2, 0.1, 0.0]
    rvw[2] = [0.0, 0.0, 1.0]   # some spin
    exp_rvw, exp_s = _pt_evolve(const.rolling, rvw.copy(), dt)
    act_rvw, act_s = _our_evolve(ROLLING, rvw.copy(), dt)
    np.testing.assert_allclose(act_rvw, exp_rvw, atol=1e-8, err_msg=f"dt={dt}")
    assert act_s == exp_s


@pytest.mark.parametrize("dt", [0.0, 0.01, 0.05, 0.1])
def test_evolve_spinning(dt):
    rvw = np.zeros((3, 3), dtype=np.float64)
    rvw[0] = [0.3, 0.4, 0.0]
    rvw[2, 2] = 3.0   # only z-spin
    exp_rvw, exp_s = _pt_evolve(const.spinning, rvw.copy(), dt)
    act_rvw, act_s = _our_evolve(SPINNING, rvw.copy(), dt)
    np.testing.assert_allclose(act_rvw, exp_rvw, atol=1e-8, err_msg=f"dt={dt}")
    assert act_s == exp_s


def test_evolve_stationary():
    rvw = _rvw()
    exp_rvw, exp_s = _pt_evolve(const.stationary, rvw.copy(), 1.0)
    act_rvw, act_s = _our_evolve(STATIONARY, rvw.copy(), 1.0)
    np.testing.assert_allclose(act_rvw, exp_rvw, atol=1e-10)
    assert act_s == exp_s


@pytest.mark.parametrize("dt", [0.3, 1.0, 5.0])
def test_evolve_sliding_transitions_through_rolling(dt):
    """Long dt: sliding → rolling → spinning → stationary chain."""
    rvw = _rvw(vx=0.3, vy=0.2)
    exp_rvw, exp_s = _pt_evolve(const.sliding, rvw.copy(), dt)
    act_rvw, act_s = _our_evolve(SLIDING, rvw.copy(), dt)
    np.testing.assert_allclose(act_rvw, exp_rvw, atol=1e-7, err_msg=f"dt={dt}")
    assert act_s == exp_s


def test_t_stop_sliding():
    from world_model.ball_motion import BallTrajectory
    rvw = _rvw()
    expected = BallTrajectory(rvw, const.sliding, P).t_stop()
    actual   = t_stop(rvw, SLIDING, P.R, P.u_s, P.u_sp, P.u_r, P.g)
    assert abs(actual - expected) < 1e-8


def test_t_stop_rolling():
    from world_model.ball_motion import BallTrajectory
    rvw = np.zeros((3, 3), dtype=np.float64)
    rvw[0] = [0.3, 0.4, 0.0]
    rvw[1] = [0.2, 0.1, 0.0]
    expected = BallTrajectory(rvw, const.rolling, P).t_stop()
    actual   = t_stop(rvw, ROLLING, P.R, P.u_s, P.u_sp, P.u_r, P.g)
    assert abs(actual - expected) < 1e-8


# ── Layer 2: collision times ──────────────────────────────────────────────────

def _bb_oracle(rvw1, rvw2, s1=SLIDING, s2=STATIONARY):
    """Oracle: deprecated pooltool solver — same np.roots algorithm as ours."""
    return pt_solve.ball_ball_collision_time(
        rvw1, rvw2, s1, s2,
        P.u_s, P.u_s, P.m, P.m, P.g, P.g, P.R,
    )


def _bb_ours(rvw1, rvw2, s1=SLIDING, s2=STATIONARY):
    return ball_ball_collision_time(
        rvw1, rvw2, s1, s2,
        P.u_s, P.u_s, P.m, P.m, P.g, P.g, P.R,
    )


class TestBallBallTime:
    """
    Oracle: pt_solve.ball_ball_collision_time (deprecated pooltool helper).
    Uses the same np.roots algorithm as our implementation — fair 1-to-1 comparison.
    Both treat the trajectory as a single-state parabola, so spurious roots must be
    filtered via physical validation (done in our implementation only).
    """

    def test_head_on(self):
        # Very close balls, high speed → collision well within sliding phase
        rvw1 = _rvw(px=0.3, py=0.5, vx=2.0, vy=0.0, wx=0.0, wy=0.0, wz=0.0)
        rvw2 = np.zeros((3, 3)); rvw2[0] = [0.36, 0.5, 0.0]
        t_oracle = _bb_oracle(rvw1, rvw2)
        t_ours   = _bb_ours(rvw1, rvw2)
        assert t_oracle < np.inf, "oracle found no collision — adjust scenario"
        assert abs(t_ours - t_oracle) < 1e-6

    def test_glancing(self):
        rvw1 = _rvw(px=0.3, py=0.5, vx=2.0, vy=0.05, wx=0.0, wy=0.0, wz=0.0)
        rvw2 = np.zeros((3, 3)); rvw2[0] = [0.36, 0.512, 0.0]
        t_oracle = _bb_oracle(rvw1, rvw2)
        t_ours   = _bb_ours(rvw1, rvw2)
        assert t_oracle < np.inf, "oracle found no collision — adjust scenario"
        assert abs(t_ours - t_oracle) < 1e-6

    def test_diverging_returns_inf(self):
        # Ball moving away from stationary target — must not find spurious root
        rvw1 = _rvw(px=0.3, py=0.5, vx=-0.5, vy=0.0, wx=0.0, wy=0.0, wz=0.0)
        rvw2 = np.zeros((3, 3)); rvw2[0] = [0.7, 0.5, 0.0]
        assert _bb_ours(rvw1, rvw2) == np.inf

    def test_both_moving_toward_each_other(self):
        rvw1 = _rvw(px=0.2, py=0.5, vx=2.0, vy=0.0, wx=0.0, wy=0.0, wz=0.0)
        rvw2 = _rvw(px=0.5, py=0.5, vx=-2.0, vy=0.0, wx=0.0, wy=0.0, wz=0.0)
        t_oracle = _bb_oracle(rvw1, rvw2, SLIDING, SLIDING)
        t_ours   = _bb_ours(rvw1, rvw2, SLIDING, SLIDING)
        assert t_oracle < np.inf
        assert abs(t_ours - t_oracle) < 1e-6

    def test_intersecting_returns_inf(self):
        rvw1 = _rvw(px=0.5, py=0.5, vx=0.1, vy=0.0)
        rvw2 = np.zeros((3, 3)); rvw2[0] = [0.5 + P.R * 0.5, 0.5, 0.0]
        assert _bb_ours(rvw1, rvw2) == np.inf


class TestLinearCushionTime:
    def _setup(self):
        table = pt.Table.default()
        cush  = list(table.cushion_segments.linear.values())[0]
        return cush

    def _oracle(self, rvw, s, cush):
        return pt_solve.ball_linear_cushion_collision_time(
            rvw=rvw, s=s,
            lx=cush.lx, ly=cush.ly, l0=cush.l0,
            p1=cush.p1, p2=cush.p2, direction=cush.direction,
            mu=P.u_s, m=P.m, g=P.g, R=P.R,
        )

    def _ours(self, rvw, s, cush):
        return ball_linear_cushion_time(
            rvw=rvw, s=s,
            lx=cush.lx, ly=cush.ly, l0=cush.l0,
            p1=cush.p1[:2], p2=cush.p2[:2],
            direction=cush.direction,
            mu=P.u_s, m=P.m, g=P.g, R=P.R,
        )

    def test_ball_heading_toward_cushion(self):
        cush = self._setup()
        # lx=1, ly=0, l0=0, direction=1 → left wall at x=0
        # High speed + close to wall → guaranteed collision within sliding phase
        rvw = _rvw(px=0.15, py=0.5, vx=-2.0, vy=0.0, wx=0.0, wy=0.0, wz=0.0)
        t_exp = self._oracle(rvw, const.sliding, cush)
        t_act = self._ours(rvw, SLIDING, cush)
        assert t_exp < np.inf, "oracle found no collision — adjust scenario"
        assert abs(t_act - t_exp) < 1e-6, f"exp={t_exp}, act={t_act}"

    def test_ball_moving_away_returns_inf(self):
        # Ball moving right — physically cannot reach the left wall.
        # The deprecated oracle has a known spurious-root bug here (returns ~2.2s).
        # Our implementation correctly rejects roots beyond t_stop → returns inf.
        cush = self._setup()
        rvw  = _rvw(px=0.4, py=0.5, vx=2.0, vy=0.0, wx=0.0, wy=0.0, wz=0.0)
        assert self._ours(rvw, SLIDING, cush) == np.inf

    def test_stationary_returns_inf(self):
        cush = self._setup()
        rvw = _rvw()
        assert self._ours(rvw, STATIONARY, cush) == np.inf


class TestCircularCushionTime:
    def _setup(self):
        table = pt.Table.default()
        cush  = list(table.cushion_segments.circular.values())[0]
        return cush

    def _oracle(self, rvw, s, cush):
        return pt_solve.ball_circular_cushion_collision_coeffs(
            rvw=rvw, s=s, a=cush.a, b=cush.b, r=cush.radius,
            mu=P.u_s, m=P.m, g=P.g, R=P.R,
        )

    def _ours(self, rvw, s, cush):
        return ball_circular_cushion_time(
            rvw=rvw, s=s, a=cush.a, b=cush.b, r=cush.radius,
            mu=P.u_s, m=P.m, g=P.g, R=P.R,
        )

    def test_coefficients_match(self):
        """Check quartic coefficients match (oracle returns coeffs, not time)."""
        cush = self._setup()
        rvw  = _rvw(px=0.2, py=0.15, vx=-0.3, vy=-0.3)
        A, B, C, D, E = self._oracle(rvw, const.sliding, cush)

        from world_model.pure_physics import _traj_coeffs, _min_positive_real_root
        ax, ay, bx, by = _traj_coeffs(rvw, SLIDING, P.R, P.u_s, P.g)
        cx, cy = rvw[0, 0], rvw[0, 1]
        a_c, b_c = cush.a, cush.b
        r_c = cush.radius

        our_A = 0.5 * (ax**2 + ay**2)
        our_B = ax * bx + ay * by
        our_C = ax * (cx - a_c) + ay * (cy - b_c) + 0.5 * (bx**2 + by**2)
        our_D = bx * (cx - a_c) + by * (cy - b_c)
        our_E = 0.5 * (a_c**2 + b_c**2 + cx**2 + cy**2 - (r_c + P.R)**2) - (cx * a_c + cy * b_c)

        np.testing.assert_allclose([our_A, our_B, our_C, our_D, our_E],
                                    [A, B, C, D, E], atol=1e-10)

    def test_stationary_returns_inf(self):
        cush = self._setup()
        rvw  = _rvw()
        assert self._ours(rvw, STATIONARY, cush) == np.inf


class TestPocketTime:
    def _setup(self):
        table = pt.Table.default()
        pock  = list(table.pockets.values())[0]
        return pock

    def _oracle(self, rvw, s, pock):
        return pt_solve.ball_pocket_collision_coeffs(
            rvw=rvw, s=s, a=pock.a, b=pock.b, r=pock.radius,
            mu=P.u_s, m=P.m, g=P.g, R=P.R,
        )

    def _ours(self, rvw, s, pock):
        return ball_pocket_time(
            rvw=rvw, s=s, a=pock.a, b=pock.b, r=pock.radius,
            mu=P.u_s, m=P.m, g=P.g, R=P.R,
        )

    def test_coefficients_match(self):
        pock = self._setup()
        rvw  = _rvw(px=0.1, py=0.1, vx=-0.3, vy=-0.3)
        A, B, C, D, E = self._oracle(rvw, const.sliding, pock)

        from world_model.pure_physics import _traj_coeffs
        ax, ay, bx, by = _traj_coeffs(rvw, SLIDING, P.R, P.u_s, P.g)
        cx, cy = rvw[0, 0], rvw[0, 1]
        a_p, b_p, r_p = pock.a, pock.b, pock.radius

        our_A = 0.5 * (ax**2 + ay**2)
        our_B = ax * bx + ay * by
        our_C = ax * (cx - a_p) + ay * (cy - b_p) + 0.5 * (bx**2 + by**2)
        our_D = bx * (cx - a_p) + by * (cy - b_p)
        our_E = 0.5 * (a_p**2 + b_p**2 + cx**2 + cy**2 - r_p**2) - (cx * a_p + cy * b_p)

        np.testing.assert_allclose([our_A, our_B, our_C, our_D, our_E],
                                    [A, B, C, D, E], atol=1e-10)

    def test_stationary_returns_inf(self):
        pock = self._setup()
        rvw  = _rvw()
        assert self._ours(rvw, STATIONARY, pock) == np.inf


# ── Layer 3: end-to-end event sequence ───────────────────────────────────────

class TestEventSequence:
    """Compare full shot event sequences between EventDetector (pooltool) and
    a pure_physics reimplementation would produce the same ordering."""

    def _run_pooltool_sequence(self, rvw_cue, rvw_tgt):
        """Ground-truth event sequence via EventDetector.next_collision loop.
        Avoids pooltool's solve_quartics numba bug (only triggered by full_event_sequence)."""
        import pooltool.constants as pc
        from world_model.ball_motion import DEFAULT_FRICTION as P
        from world_model.event_detector import EventDetector
        from world_model.rssm_rollout import advance_balls

        table = pt.Table.default()
        cue   = pt.Cue.default()
        det   = EventDetector(table, cue)

        balls   = {"cue": (rvw_cue.copy(), pc.sliding), "1": (rvw_tgt.copy(), pc.stationary)}
        results = []
        for _ in range(20):
            ev = det.next_collision(balls)
            if ev.event_type == -1:
                break
            balls = advance_balls(balls, ev.time, P)
            results.append(ev)
        return results

    def test_event_sequence_nonempty(self):
        """Sanity: a direct shot produces events."""
        rvw_cue = _rvw(px=0.3, py=0.5, vx=0.8, vy=0.0)
        rvw_tgt = np.zeros((3, 3)); rvw_tgt[0] = [0.7, 0.5, 0.0]
        seq = self._run_pooltool_sequence(rvw_cue, rvw_tgt)
        assert len(seq) > 0

    @pytest.mark.parametrize("gap", [0.07, 0.10, 0.15, 0.20])
    def test_ball_ball_time_consistent_with_sequence(self, gap):
        """
        Direct shot: cue aimed straight at target, various distances.
        First event must be ball-ball. Compare our time vs pooltool sequence.
        Small non-zero spin avoids a pooltool numba bug with all-zero rvw.
        """
        rvw_cue = _rvw(px=0.3, py=0.5, vx=2.0, vy=0.0, wx=0.0, wy=0.0, wz=0.1)
        rvw_tgt = np.zeros((3, 3)); rvw_tgt[0] = [0.3 + gap, 0.5, 0.0]

        seq = self._run_pooltool_sequence(rvw_cue, rvw_tgt)
        bb_events = [e for e in seq if e.event_type == 0]
        assert bb_events, f"No ball-ball event for gap={gap}"

        t_oracle = bb_events[0].time
        t_ours   = ball_ball_collision_time(
            rvw_cue, rvw_tgt, SLIDING, STATIONARY,
            P.u_s, P.u_s, P.m, P.m, P.g, P.g, P.R,
        )
        assert abs(t_ours - t_oracle) < 1e-4, f"oracle={t_oracle:.6f}, ours={t_ours:.6f}"
