"""
world_model/ball_motion.py

Thin wrapper around pooltool ball motion physics.
Provides a stateful BallTrajectory that can be queried at arbitrary times.
"""

import numpy as np
import pooltool.physics as ph
import pooltool.ptmath as ptmath
import pooltool.constants as const
from dataclasses import dataclass

# Default parameters matching pooltool's BallParams.default() (== simulator.py balls,
# which are created via pt.Ball.create() with no override).
# u_sp = u_sp_proportionality * R = 0.444444... * 0.028575 = 0.0127
DEFAULT_PARAMS = dict(R=0.028575, m=0.170097, u_s=0.2, u_sp=0.0127, u_r=0.01, g=9.81)


@dataclass
class FrictionParams:
    u_s:  float = 0.2
    u_sp: float = 0.0127
    u_r:  float = 0.01
    R:    float = 0.028575
    m:    float = 0.170097
    g:    float = 9.81

    def as_dict(self) -> dict:
        return dict(R=self.R, m=self.m, u_s=self.u_s, u_sp=self.u_sp, u_r=self.u_r, g=self.g)


DEFAULT_FRICTION = FrictionParams()


class BallTrajectory:
    """Wraps evolve_ball_motion to provide rvw(t) queries."""

    def __init__(
        self,
        rvw:    np.ndarray,          # (3, 3) pos/vel/avel in table frame
        state:  int,                 # const.sliding / rolling / spinning / stationary
        params: FrictionParams = DEFAULT_FRICTION,
    ):
        self.rvw0   = np.array(rvw,  dtype=np.float64)
        self.state0 = state
        self.p      = params

    # ------------------------------------------------------------------

    def rvw_at(self, t: float) -> tuple[np.ndarray, int]:
        """(rvw, new_state) at time t after this snapshot."""
        return ph.evolve_ball_motion(self.state0, self.rvw0, **self.p.as_dict(), t=t)

    def pos_at(self, t: float) -> np.ndarray:
        """2-D ball centre position at time t."""
        rvw_t, _ = self.rvw_at(t)
        return rvw_t[0, :2]

    def t_stop(self) -> float:
        """Time until ball comes to rest (all states exhausted)."""
        p   = self.p
        rvw = self.rvw0
        s   = self.state0
        t   = 0.0

        if s == const.sliding:
            dt = ptmath.get_slide_time(rvw, p.R, p.u_s, p.g)
            rvw = ph.evolve_slide_state(rvw, p.R, p.m, p.u_s, p.u_sp, p.g, dt)
            s   = const.rolling
            t  += dt

        if s == const.rolling:
            dt = ptmath.get_roll_time(rvw, p.u_r, p.g)
            rvw = ph.evolve_roll_state(rvw, p.R, p.u_r, p.u_sp, p.g, dt)
            s   = const.spinning
            t  += dt

        if s == const.spinning:
            dt = ptmath.get_spin_time(rvw, p.R, p.u_sp, p.g)
            t  += dt

        return t
