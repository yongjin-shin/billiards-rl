"""
simulator.py — BilliardsEnv: gymnasium wrapper around pooltool physics engine.

Architecture:
  pooltool  →  physics engine (ball physics, table geometry, pocket detection)
  BilliardsEnv  →  gym wrapper  (obs / action / reward / episode logic)

BilliardsEnv(n_balls=1)  →  single-shot env  (Phase 0, backward-compatible)
BilliardsEnv(n_balls=3)  →  multi-ball env   (Phase 1a)

Usage (sanity test):
    python simulator.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pooltool as pt


# =============================================================================
# Environment
# =============================================================================

class BilliardsEnv(gym.Env):
    """
    Billiards environment. Parameterised by n_balls.

    --- n_balls=1  (single-shot, Phase 0) ---
    Observation (16-dim, all normalized [0, 1]):
      [cue_x, cue_y, ball_x, ball_y, p0x,p0y, ..., p5x,p5y]
    Reward : +1.0 if ball pocketed, else 0.0
    Episode: single shot (horizon = 1)

    --- n_balls=3  (multi-ball, Phase 1a) ---
    Observation (23-dim base, all normalized [0, 1]):
      [cue_x, cue_y,
       b1x, b1y, b1_pocketed,
       b2x, b2y, b2_pocketed,
       b3x, b3y, b3_pocketed,
       p0x,p0y, ..., p5x,p5y]
    If shots_taken=True → 24-dim: append shots_taken/max_steps ∈ (0, 1]
    Reward : +1.0 per ball pocketed
             ·  -step_penalty×i per step if progressive_penalty else -step_penalty (flat)
             ·  -0.5 for scratch
             ·  -trunc_penalty when episode truncated (step limit reached)
             ·  +clear_bonus/steps_used when all balls cleared (terminated)
    Episode ends : all balls pocketed  OR  step >= max_steps

    Action (2-dim continuous, same for both):
      abs_angle=False (default):
        [delta_angle ∈ [-π, π],  speed ∈ [0.5, 8.0]]
        delta_angle = 0  →  aim at nearest unpocketed ball
        delta_angle ≠ 0  →  cut/offset from that direction
      abs_angle=True (Exp-12):
        [phi ∈ [0, 2π],  speed ∈ [0.5, 8.0]]
        phi = absolute cue angle in table coordinates (0 = +x axis)
        nearest-ball inductive bias 제거 → agent가 공 순서를 스스로 선택
    """

    metadata = {"render_modes": []}

    MIN_BALL_DIST = 0.12   # metres — minimum distance between any two balls at reset

    def __init__(self, n_balls: int = 1, max_steps: int = 5,
                 step_penalty: float = 0.01, trunc_penalty: float = 0.0,
                 progressive_penalty: bool = False,
                 clear_bonus: float = 0.0,
                 shots_taken: bool = False,
                 abs_angle: bool = False,
                 legacy_placement: bool = False):
        super().__init__()
        assert n_balls >= 1, "n_balls must be >= 1"

        self.n_balls             = n_balls
        self.max_steps           = max_steps
        self.step_penalty        = step_penalty        # base penalty per step
        self.trunc_penalty       = trunc_penalty       # extra penalty when truncated
        self.progressive_penalty = progressive_penalty # if True: step i costs step_penalty × i
        self.clear_bonus         = clear_bonus         # +clear_bonus/steps_used on termination
        self.shots_taken         = shots_taken         # if True: append shots_taken/max_steps to obs
        self.abs_angle           = abs_angle           # if True: action[0] = absolute phi [0, 2π]
        # legacy_placement=True: original Exp-01 ranges (n_balls=1 only)
        #   cue y∈[0.2,0.4], target y∈[0.6,0.9] — always upper/lower separated
        # legacy_placement=False (default): current unified ranges
        #   cue y∈[0.15,0.40], target y∈[0.30,0.85] — wider, harder
        self.legacy_placement    = legacy_placement and (n_balls == 1)

        # Ball IDs: "1", "2", "3", ...
        self._ball_ids = [str(i + 1) for i in range(n_balls)]

        self.table        = pt.Table.default()
        self.table_length = self.table.l
        self.table_width  = self.table.w

        if abs_angle:
            # Absolute table angle — agent freely chooses any direction
            self.action_space = spaces.Box(
                low  = np.array([0.0,      0.5], dtype=np.float32),
                high = np.array([2*np.pi,  8.0], dtype=np.float32),
            )
        else:
            # Delta offset from nearest-unpocketed-ball direction
            self.action_space = spaces.Box(
                low  = np.array([-np.pi, 0.5], dtype=np.float32),
                high = np.array([ np.pi, 8.0], dtype=np.float32),
            )

        # obs dim: 2(cue) + n_balls*2(pos) + n_balls*(0 or 1)(flag) + 12(pockets)
        # n_balls=1: no pocketed flag needed (horizon=1 → episode always ends)
        # shots_taken=True: +1 dim (shots_taken/max_steps ∈ (0,1])
        obs_dim = 2 + n_balls * (2 if n_balls == 1 else 3) + 12 + (1 if shots_taken else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._pocket_obs     = self._build_pocket_obs()
        self._pocket_centers = self._pocket_obs.reshape(6, 2) * \
                               np.array([self.table_width, self.table_length])

        self.system      = None
        self._step_count = 0
        self._pocketed   = {}

    # -------------------------------------------------------------------------
    def _build_pocket_obs(self):
        try:
            pts = []
            for pocket in self.table.pockets.values():
                if hasattr(pocket, "center"):
                    xy = np.asarray(pocket.center[:2], dtype=float)
                elif hasattr(pocket, "a"):
                    xy = np.asarray(pocket.a[:2], dtype=float)
                else:
                    raise AttributeError
                pts.extend([
                    float(np.clip(xy[0] / self.table_width,  0.0, 1.0)),
                    float(np.clip(xy[1] / self.table_length, 0.0, 1.0)),
                ])
            if len(pts) == 12:
                return np.array(pts, dtype=np.float32)
        except Exception:
            pass
        return np.array([
            0.0, 0.0,  1.0, 0.0,
            0.0, 0.5,  1.0, 0.5,
            0.0, 1.0,  1.0, 1.0,
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._pocketed   = {bid: False for bid in self._ball_ids}

        placed = []   # (x, y) in metres — collision tracking

        def sample_pos(low_norm, high_norm):
            """Sample a position that doesn't overlap with already-placed balls."""
            for _ in range(300):
                xy = self.np_random.uniform(low_norm, high_norm)
                xy_m = [xy[0] * self.table_width, xy[1] * self.table_length]
                if all(np.linalg.norm(np.array(xy_m) - np.array(p)) > self.MIN_BALL_DIST
                       for p in placed):
                    placed.append(xy_m)
                    return xy_m
            # Fallback: last sampled (very unlikely to be needed)
            placed.append(xy_m)
            return xy_m

        # Cue ball — lower portion
        if self.legacy_placement:
            # Original Exp-01 ranges: cue bottom, target top — always separated
            cue_xy   = sample_pos([0.20, 0.20], [0.80, 0.40])
            ball_xys = [sample_pos([0.20, 0.60], [0.80, 0.90])
                        for _ in self._ball_ids]
        else:
            cue_xy   = sample_pos([0.15, 0.15], [0.85, 0.40])
            # Target balls — upper portion (spread across table)
            ball_xys = [sample_pos([0.15, 0.30], [0.85, 0.85])
                        for _ in self._ball_ids]

        balls = {"cue": pt.Ball.create("cue", xy=cue_xy)}
        for bid, bxy in zip(self._ball_ids, ball_xys):
            balls[bid] = pt.Ball.create(bid, xy=bxy)

        self.system = pt.System(
            table=self.table,
            balls=balls,
            cue=pt.Cue.default(),
        )
        return self._get_obs(), {}

    # -------------------------------------------------------------------------
    def step(self, action):
        speed = float(action[1])

        if self.abs_angle:
            # Exp-12: action[0] is absolute angle in [0, 2π]
            phi_deg = np.degrees(float(action[0]))
        else:
            # Default: action[0] is offset from nearest-ball direction
            delta_angle = float(action[0])
            cue_pos = self.system.balls["cue"].state.rvw[0, :2]
            ref_pos = self._nearest_unpocketed_pos(cue_pos)
            if ref_pos is None:
                ref_pos = cue_pos + np.array([1.0, 0.0])
            phi_direct = np.arctan2(ref_pos[1] - cue_pos[1], ref_pos[0] - cue_pos[0])
            phi_deg    = np.degrees(phi_direct + delta_angle)

        self.system.strike(phi=phi_deg, V0=speed, cue_ball_id="cue")
        pt.simulate(self.system, inplace=True)
        self._step_count += 1

        # ── Scratch (cue ball pocketed) ───────────────────────────────────────
        scratch = (self.system.balls["cue"].state.s == pt.constants.pocketed)

        # ── Newly pocketed target balls ───────────────────────────────────────
        # Guard with `in` check: pooltool may drop a ball from system.balls
        # on subsequent simulate() calls once it was pocketed.
        newly_pocketed = 0
        for bid in self._ball_ids:
            if not self._pocketed[bid]:
                ball_gone = bid not in self.system.balls
                ball_pocketed = (not ball_gone and
                                 self.system.balls[bid].state.s == pt.constants.pocketed)
                if ball_gone or ball_pocketed:
                    self._pocketed[bid] = True
                    newly_pocketed += 1

        # ── Termination ───────────────────────────────────────────────────────
        if self.n_balls == 1:
            # Backward-compatible: always terminate after one shot
            terminated = True
            truncated  = False
        else:
            terminated = all(self._pocketed.values())
            truncated  = (not terminated) and (self._step_count >= self.max_steps)

        # ── Reward ────────────────────────────────────────────────────────────
        # Progressive penalty: step i costs step_penalty × i (later steps more expensive)
        # Flat penalty: constant step_penalty every step
        _step_pen = (self.step_penalty * self._step_count
                     if self.progressive_penalty else self.step_penalty)
        reward = float(newly_pocketed) - _step_pen
        if scratch:
            reward -= 0.5
        if truncated:
            reward -= self.trunc_penalty
        # Clear bonus: reward faster clears — scales as 1/steps_used so fewer steps = bigger bonus
        if terminated and self.n_balls > 1 and self.clear_bonus > 0.0:
            reward += self.clear_bonus / self._step_count

        # ── Ball-in-hand after scratch (multi-ball only) ──────────────────────
        if scratch and not terminated and not truncated:
            self._respawn_cue()

        # ── Info ──────────────────────────────────────────────────────────────
        if self.n_balls == 1:
            info = {"pocketed": bool(newly_pocketed)}
        else:
            _cb_earned = (self.clear_bonus / self._step_count
                          if terminated and self.clear_bonus > 0.0 else 0.0)
            info = {
                "pocketed_this_step": newly_pocketed,
                "total_pocketed"    : sum(self._pocketed.values()),
                "remaining"         : sum(1 for v in self._pocketed.values() if not v),
                "scratch"           : scratch,
                "clear_bonus_earned": _cb_earned,
            }

        return self._get_obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    def _nearest_unpocketed_pos(self, cue_pos: np.ndarray):
        """Return position of nearest unpocketed target ball, or None if all gone."""
        best_dist, best_pos = float("inf"), None
        for bid in self._ball_ids:
            if not self._pocketed[bid] and bid in self.system.balls:
                bpos = self.system.balls[bid].state.rvw[0, :2]
                d = np.linalg.norm(bpos - cue_pos)
                if d < best_dist:
                    best_dist, best_pos = d, bpos
        return best_pos

    # -------------------------------------------------------------------------
    def _respawn_cue(self):
        """Ball-in-hand: place cue ball at a random valid position after scratch."""
        active_positions = [
            self.system.balls[bid].state.rvw[0, :2].tolist()
            for bid in self._ball_ids if not self._pocketed[bid]
        ]

        xy_m = [self.table_width * 0.5, self.table_length * 0.25]  # safe fallback
        for _ in range(300):
            xy = self.np_random.uniform([0.10, 0.10], [0.90, 0.90])
            candidate = [xy[0] * self.table_width, xy[1] * self.table_length]
            if all(np.linalg.norm(np.array(candidate) - np.array(p)) > self.MIN_BALL_DIST
                   for p in active_positions):
                xy_m = candidate
                break

        balls = {"cue": pt.Ball.create("cue", xy=xy_m)}
        for bid in self._ball_ids:
            if not self._pocketed[bid]:
                bpos = self.system.balls[bid].state.rvw[0, :2].tolist()
                balls[bid] = pt.Ball.create(bid, xy=bpos)

        self.system = pt.System(
            table=self.table,
            balls=balls,
            cue=pt.Cue.default(),
        )

    # -------------------------------------------------------------------------
    def _get_obs(self):
        def norm_pos(ball_id):
            if ball_id not in self.system.balls:
                return 0.0, 0.0
            ball = self.system.balls[ball_id]
            if ball.state.s == pt.constants.pocketed:
                return 0.0, 0.0
            x = float(np.clip(ball.state.rvw[0, 0] / self.table_width,  0.0, 1.0))
            y = float(np.clip(ball.state.rvw[0, 1] / self.table_length, 0.0, 1.0))
            return x, y

        cue_x, cue_y = norm_pos("cue")
        obs = [cue_x, cue_y]

        if self.n_balls == 1:
            bx, by = norm_pos(self._ball_ids[0])
            obs.extend([bx, by])
        else:
            for bid in self._ball_ids:
                # Use self._pocketed as source of truth — don't access system.balls for
                # pocketed balls since pooltool may remove them from the dict.
                if self._pocketed[bid]:
                    obs.extend([0.0, 0.0, 1.0])
                else:
                    bx, by = norm_pos(bid)
                    obs.extend([bx, by, 0.0])

        obs.extend(self._pocket_obs.tolist())
        if self.shots_taken:
            obs.append(self._step_count / self.max_steps)
        return np.array(obs, dtype=np.float32)


# =============================================================================
# Sanity tests
# =============================================================================

if __name__ == "__main__":
    import sys

    # ── Single-ball (backward compat) ─────────────────────────────────────────
    print("=" * 55)
    print("BilliardsEnv(n_balls=1) — backward-compat test")
    print("=" * 55)

    env1 = BilliardsEnv(n_balls=1)
    print(f"\n[1] obs shape : {env1.observation_space.shape}  (expected 16)")
    print(f"    act shape : {env1.action_space.shape}  (expected 2)")
    assert env1.observation_space.shape == (16,)

    obs, _ = env1.reset(seed=0)
    _, r, terminated, truncated, info = env1.step(env1.action_space.sample())
    assert terminated, "n_balls=1 must always terminate after 1 step"
    assert "pocketed" in info
    print(f"[2] Single episode OK  reward={r:.2f}  pocketed={info['pocketed']}")

    n, hits = 500, 0
    for _ in range(n):
        env1.reset()
        _, _, _, _, info = env1.step(env1.action_space.sample())
        hits += int(info["pocketed"])
    print(f"[3] Random pocket rate (n=1): {hits/n*100:.1f}%")

    # ── Multi-ball (Phase 1a) ─────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("BilliardsEnv(n_balls=3) — Phase 1a test")
    print("=" * 55)

    env3 = BilliardsEnv(n_balls=3, max_steps=5)
    print(f"\n[1] obs shape : {env3.observation_space.shape}  (expected 23)")
    print(f"    act shape : {env3.action_space.shape}  (expected 2)")
    assert env3.observation_space.shape == (23,)

    obs, _ = env3.reset(seed=42)
    print(f"[2] Reset obs : {obs}")
    print(f"    Flags (pocketed): {obs[4]:.0f} {obs[7]:.0f} {obs[10]:.0f}  (all 0 expected)")
    assert obs[4] == 0.0 and obs[7] == 0.0 and obs[10] == 0.0

    # Full episode with random agent
    print(f"\n[3] Random episode (max_steps={env3.max_steps})")
    obs, _ = env3.reset(seed=1)
    ep_reward, steps = 0.0, 0
    while True:
        obs, r, terminated, truncated, info = env3.step(env3.action_space.sample())
        ep_reward += r
        steps += 1
        print(f"    step {steps:2d}  r={r:+.2f}  pocketed={info['total_pocketed']}/3"
              f"  scratch={info['scratch']}  done={terminated or truncated}")
        if terminated or truncated:
            break
    print(f"    Episode done — total_reward={ep_reward:.2f}  steps={steps}")
    print(f"    terminated={terminated}  truncated={truncated}")

    # Stats over many episodes
    print(f"\n[4] Random agent stats (500 episodes)")
    total_pocketed, clears, ep_steps_list = 0, 0, []
    for _ in range(500):
        env3.reset()
        steps = 0
        while True:
            _, _, term, trunc, info = env3.step(env3.action_space.sample())
            steps += 1
            if term or trunc:
                total_pocketed += info["total_pocketed"]
                clears += int(info["total_pocketed"] == 3)
                ep_steps_list.append(steps)
                break
    print(f"    Avg pocketed / episode : {total_pocketed/500:.2f} / 3")
    print(f"    Clear rate (all 3)     : {clears/500*100:.1f}%")
    print(f"    Avg episode length     : {sum(ep_steps_list)/len(ep_steps_list):.1f} steps")

    print("\nAll tests passed ✓")
