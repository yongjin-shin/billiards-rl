"""
simulator.py — BilliardsEnv: gymnasium wrapper around pooltool physics engine.

Architecture:
  pooltool  →  physics engine (ball physics, table geometry, pocket detection)
  BilliardsEnv  →  gym wrapper  (obs / action / reward / episode logic)

Usage (sanity test):
    python simulator.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pooltool as pt


# =============================================================================
# Environment — v2
# =============================================================================

class BilliardsEnv(gym.Env):
    """
    Single-shot billiards environment.

    Observation (16-dim, all normalized [0, 1]):
      [cue_x, cue_y, target_x, target_y,          ← ball positions
       p0x,p0y, p1x,p1y, ..., p5x,p5y]            ← 6 pocket positions (fixed)

    Action (2-dim continuous):
      [delta_angle ∈ [-π, π],  speed ∈ [0.5, 8.0]]
       delta_angle = 0  → aim directly at target ball
       delta_angle ≠ 0  → cut shot (offset from cue→target line)

    Reward:
      +1.0  target ball pocketed  (binary — avoids local optima)

    Episode: single shot (horizon = 1)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.table        = pt.Table.default()
        self.table_length = self.table.l        # metres
        self.table_width  = self.table.w
        self._table_diag  = np.sqrt(self.table_length**2 + self.table_width**2)

        self.action_space = spaces.Box(
            low  = np.array([-np.pi, 0.5], dtype=np.float32),
            high = np.array([ np.pi, 8.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32
        )

        # Pocket positions — fixed for the lifetime of the env
        self._pocket_obs     = self._build_pocket_obs()          # (12,) normalized
        self._pocket_centers = self._pocket_obs.reshape(6, 2) * \
                               np.array([self.table_width, self.table_length])  # (6,2) metres

        self.system = None

    # -------------------------------------------------------------------------
    def _build_pocket_obs(self):
        """Get real pocket (x,y) from pooltool Table; fall back to standard layout."""
        try:
            pts = []
            for pocket in self.table.pockets.values():
                if hasattr(pocket, "center"):
                    xy = np.asarray(pocket.center[:2], dtype=float)
                elif hasattr(pocket, "a"):
                    xy = np.asarray(pocket.a[:2], dtype=float)
                else:
                    raise AttributeError("unknown pocket geometry attr")
                pts.extend([
                    float(np.clip(xy[0] / self.table_width,  0.0, 1.0)),
                    float(np.clip(xy[1] / self.table_length, 0.0, 1.0)),
                ])
            if len(pts) == 12:
                return np.array(pts, dtype=np.float32)
        except Exception:
            pass
        # Fallback: standard 6-pocket layout
        return np.array([
            0.0, 0.0,   # bottom-left corner
            1.0, 0.0,   # bottom-right corner
            0.0, 0.5,   # side-left  (mid-length)
            1.0, 0.5,   # side-right (mid-length)
            0.0, 1.0,   # top-left corner
            1.0, 1.0,   # top-right corner
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Cue ball: lower half; target: upper half (avoids trivial shots)
        cue_xy    = self.np_random.uniform([0.2, 0.2], [0.8, 0.4]).tolist()
        target_xy = self.np_random.uniform([0.2, 0.6], [0.8, 0.9]).tolist()

        cue_xy    = [cue_xy[0]    * self.table_width,  cue_xy[1]    * self.table_length]
        target_xy = [target_xy[0] * self.table_width,  target_xy[1] * self.table_length]

        cue_ball    = pt.Ball.create("cue", xy=cue_xy)
        target_ball = pt.Ball.create("8",   xy=target_xy)

        self.system = pt.System(
            table=self.table,
            balls={"cue": cue_ball, "8": target_ball},
            cue=pt.Cue.default(),
        )
        return self._get_obs(), {}

    # -------------------------------------------------------------------------
    def step(self, action):
        delta_angle = float(action[0])
        speed       = float(action[1])

        # Convert relative angle → absolute world angle
        cue_pos    = self.system.balls["cue"].state.rvw[0, :2]
        target_pos = self.system.balls["8"].state.rvw[0, :2]
        phi_direct = np.arctan2(target_pos[1] - cue_pos[1],
                                target_pos[0] - cue_pos[0])
        phi_deg = np.degrees(phi_direct + delta_angle)

        self.system.strike(phi=phi_deg, V0=speed, cue_ball_id="cue")
        pt.simulate(self.system, inplace=True)

        pocketed = (self.system.balls["8"].state.s == pt.constants.pocketed)
        reward   = 1.0 if pocketed else 0.0

        return self._get_obs(), reward, True, False, {"pocketed": pocketed}

    # -------------------------------------------------------------------------
    def _get_obs(self):
        def norm(ball_id, dim):
            val   = self.system.balls[ball_id].state.rvw[0, dim]
            scale = self.table_width if dim == 0 else self.table_length
            return float(np.clip(val / scale, 0.0, 1.0))

        ball_obs = np.array([
            norm("cue", 0), norm("cue", 1),
            norm("8",   0), norm("8",   1),
        ], dtype=np.float32)

        return np.concatenate([ball_obs, self._pocket_obs])


# =============================================================================
# Sanity tests — run directly to verify the simulator works
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("BilliardsEnv — Simulator Sanity Test")
    print("=" * 50)

    env = BilliardsEnv()

    # --- Spaces ---
    print(f"\n[1] Spaces")
    print(f"  obs  : {env.observation_space}")
    print(f"  act  : {env.action_space}")

    # --- Table geometry ---
    print(f"\n[2] Table geometry")
    print(f"  size : {env.table_width:.4f} m × {env.table_length:.4f} m")
    print(f"  pockets ({len(env._pocket_centers)}):")
    for i, pc in enumerate(env._pocket_centers):
        print(f"    [{i}] ({pc[0]:.3f}, {pc[1]:.3f})")

    # --- Single episode ---
    print(f"\n[3] Single episode")
    obs, _ = env.reset()
    print(f"  obs  : {obs}")
    action = env.action_space.sample()
    obs2, reward, terminated, _, info = env.step(action)
    print(f"  action  : delta={np.degrees(action[0]):.1f}°  speed={action[1]:.2f} m/s")
    print(f"  reward  : {reward}  pocketed={info['pocketed']}")
    print(f"  terminated : {terminated}")

    # --- Trajectory ---
    print(f"\n[4] Trajectory (ball.history.states)")
    for ball_id in ("cue", "8"):
        states = env.system.balls[ball_id].history.states
        if states:
            print(f"  {ball_id:3s}: {len(states)} states  "
                  f"t=[{states[0].t:.3f}→{states[-1].t:.3f}]s  "
                  f"pos0={states[0].rvw[0,:2].round(3)}")
        else:
            print(f"  {ball_id}: no history")

    # --- Random agent benchmark ---
    print(f"\n[5] Random agent  (1,000 episodes)")
    n, pocketed = 1000, 0
    for i in range(n):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        pocketed += int(info["pocketed"])
        if (i + 1) % 200 == 0:
            sys.stdout.write(f"\r  {i+1}/{n}  ({pocketed} pocketed)")
            sys.stdout.flush()
    print(f"\r  Random pocket rate: {pocketed/n*100:.1f}%  ({pocketed}/{n})")
    print("\nAll tests passed ✓")
