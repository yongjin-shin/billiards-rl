"""
exp16_wm/planner.py — pooltool search-planner reference for the Exp-16 clear-speed task.

Not an agent: before every shot it tries candidate strikes on a copy of the real
pooltool system and plays the best one. pooltool is deterministic, so the number
it reaches means "at least this much is achievable" on the same task and eval as
the RL runs (no training, greedy one-shot lookahead — no position play).

Per shot:
  1. ghost-ball aim for every remaining ball × pocket (cut angle ≤ max_cut_deg)
  2. × aim offsets × speeds → candidate actions (delta_angle, speed)
  3. simulate each on env.system.copy() with the env's own aim rule; score by
     immediate reward (balls pocketed − 0.5·scratch), tie-break: remaining balls
     closer to a pocket
  4. if nothing pockets, retry with uniform-random actions (banks/kisses by chance)
  5. play the chosen action through env.step — same physics/reward/termination as RL

Usage:
    python -m exp16_wm.planner --seeds 0 1 2 3 --n-games 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv, play_shot
from exp16_wm.train import summarize_final_eval

SCRATCH_COST = 0.5   # BilliardsEnv's multi-ball scratch penalty

# (n_pocketed, scratch, min distance of remaining balls to a pocket)
ShotResult = tuple[int, bool, float]


# ──────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────

def wrap_angle(x: float) -> float:
    """Wrap an angle in radians to [-π, π)."""
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def ghost_ball_aim(cue_xy: np.ndarray, ball_xy: np.ndarray, pocket_xy: np.ndarray,
                   radius: float) -> tuple[float, float] | None:
    """
    Cue direction that sends ball_xy toward pocket_xy (straight-line ghost ball).

    Returns (phi_rad, cut_rad) — phi is the absolute strike direction, cut the
    angle between the cue's travel and the object ball's path — or None when the
    geometry is degenerate (ball on the pocket, cue on the ghost position).
    """
    to_pocket = pocket_xy - ball_xy
    d_pocket = float(np.linalg.norm(to_pocket))
    if d_pocket < 1e-9:
        return None
    u = to_pocket / d_pocket
    aim = (ball_xy - 2.0 * radius * u) - cue_xy
    d_aim = float(np.linalg.norm(aim))
    if d_aim < 1e-9:
        return None
    phi = float(np.arctan2(aim[1], aim[0]))
    cut = float(np.arccos(np.clip(np.dot(aim / d_aim, u), -1.0, 1.0)))
    return phi, cut


# ──────────────────────────────────────────────
# Env aim rule (mirrors BilliardsEnv.step)
# ──────────────────────────────────────────────

def aim_reference(env: BilliardsEnv) -> float:
    """phi_direct that env.step adds delta_angle to: direction to the nearest unpocketed ball."""
    cue_pos = env.system.balls["cue"].state.rvw[0, :2]
    ref_pos = env._nearest_unpocketed_pos(cue_pos)
    if ref_pos is None:
        ref_pos = cue_pos + np.array([1.0, 0.0])
    return float(np.arctan2(ref_pos[1] - cue_pos[1], ref_pos[0] - cue_pos[0]))


def action_from_phi(phi_rad: float, phi_ref: float, speed: float) -> np.ndarray:
    """Absolute strike direction → env action (delta_angle, speed)."""
    return np.array([wrap_angle(phi_rad - phi_ref), speed], dtype=np.float32)


def strike_phi_deg(phi_ref: float, action: np.ndarray) -> float:
    """The exact phi (degrees) env.step will strike with for this action."""
    return float(np.degrees(phi_ref + float(action[0])))


# ──────────────────────────────────────────────
# Candidate search
# ──────────────────────────────────────────────

def simulate_candidate(env: BilliardsEnv, phi_ref: float, action: np.ndarray) -> ShotResult:
    """Play `action` on a copy of the env's system; the env itself is untouched."""
    sys_c = play_shot(env.system, strike_phi_deg(phi_ref, action), float(action[1]))

    scratch = bool(sys_c.balls["cue"].state.s == pt.constants.pocketed)
    n_pocketed, remaining = 0, []
    for bid in env._ball_ids:
        if env._pocketed[bid]:
            continue
        if bid not in sys_c.balls or sys_c.balls[bid].state.s == pt.constants.pocketed:
            n_pocketed += 1
        else:
            remaining.append(sys_c.balls[bid].state.rvw[0, :2])
    min_dist = 0.0
    if remaining:
        min_dist = float(min(np.min(np.linalg.norm(env._pocket_centers - p, axis=1))
                             for p in remaining))
    return n_pocketed, scratch, min_dist


def candidate_score(n_pocketed: int, scratch: bool) -> float:
    """Immediate-reward part that differs between candidates (the −1 shot cost is shared)."""
    return n_pocketed - SCRATCH_COST * float(scratch)


def pick_best(results: list[ShotResult]) -> int:
    """Index of the best candidate: highest immediate reward, then remaining balls nearest a pocket."""
    if not results:
        raise ValueError("pick_best needs at least one candidate")
    return max(range(len(results)),
               key=lambda i: (candidate_score(results[i][0], results[i][1]), -results[i][2]))


def ghost_aims(env: BilliardsEnv, max_cut_deg: float) -> list[tuple[str, int, float]]:
    """(ball id, pocket index, absolute ghost-ball phi in rad) for every remaining ball × pocket within max_cut."""
    cue_xy = env.system.balls["cue"].state.rvw[0, :2]
    radius = env.system.balls["cue"].params.R
    max_cut = np.radians(max_cut_deg)
    out = []
    for bid in env._ball_ids:
        if env._pocketed[bid] or bid not in env.system.balls:
            continue
        ball_xy = env.system.balls[bid].state.rvw[0, :2]
        for j, pocket_xy in enumerate(env._pocket_centers):
            aim = ghost_ball_aim(cue_xy, ball_xy, pocket_xy, radius)
            if aim is not None and aim[1] <= max_cut:
                out.append((bid, j, aim[0]))
    return out


def ghost_candidates(env: BilliardsEnv, phi_ref: float, offsets_deg: list[float],
                     speeds: list[float], max_cut_deg: float) -> list[np.ndarray]:
    """Ghost-ball aims for every remaining ball × pocket, × offsets × speeds."""
    return [action_from_phi(phi + np.radians(off), phi_ref, v)
            for _, _, phi in ghost_aims(env, max_cut_deg) for off in offsets_deg for v in speeds]


def plan_shot(env: BilliardsEnv, rng: np.random.Generator, offsets_deg: list[float],
              speeds: list[float], max_cut_deg: float, n_random: int) -> tuple[np.ndarray, dict]:
    """Choose this shot's action. Returns (action, meta)."""
    phi_ref = aim_reference(env)
    cands = ghost_candidates(env, phi_ref, offsets_deg, speeds, max_cut_deg)
    results = [simulate_candidate(env, phi_ref, a) for a in cands]
    n_ghost = len(cands)

    if not results or results[pick_best(results)][0] == 0:
        low, high = env.action_space.low, env.action_space.high
        rand = [a.astype(np.float32) for a in rng.uniform(low, high, size=(n_random, 2))]
        cands += rand
        results += [simulate_candidate(env, phi_ref, a) for a in rand]

    best = pick_best(results)
    meta = {"n_candidates": len(cands), "stage": "ghost" if best < n_ghost else "random",
            "predicted": results[best], "candidates": cands, "results": results}
    return cands[best], meta


# ──────────────────────────────────────────────
# Evaluation (same protocol as exp16_wm/train.py final eval)
# ──────────────────────────────────────────────

def play_game(env: BilliardsEnv, rng: np.random.Generator, **plan_kw) -> dict:
    """Play one game from the env's current (already reset) state."""
    done = False
    shots, ep_r, pocketed_at5, n_pocketed = 0, 0.0, 0, 0
    successes, mismatches, random_shots, plan_sec = 0, 0, 0, 0.0
    while not done:
        t0 = time.perf_counter()
        action, meta = plan_shot(env, rng, **plan_kw)
        plan_sec += time.perf_counter() - t0
        _, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        shots += 1
        ep_r  += r
        n_pocketed = info["total_pocketed"]
        if shots <= 5:
            pocketed_at5 = n_pocketed
        successes    += int(info["pocketed_this_step"] > 0)
        mismatches   += int(info["pocketed_this_step"] != meta["predicted"][0])
        random_shots += int(meta["stage"] == "random")
    return {"shots": shots, "reward": ep_r, "pocketed": n_pocketed, "pocketed_at5": pocketed_at5,
            "successes": successes, "mismatches": mismatches, "random_shots": random_shots,
            "plan_sec": plan_sec}


def evaluate_seed(seed: int, n_games: int, env_kw: dict, plan_kw: dict,
                  log_every: int = 50) -> dict:
    env = BilliardsEnv(**env_kw)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    games = []
    for g in range(n_games):
        env.reset()
        games.append(play_game(env, rng, **plan_kw))
        if log_every and (g + 1) % log_every == 0:
            sh = np.mean([x["shots"] for x in games])
            cl = np.mean([x["pocketed"] == env.n_balls for x in games])
            print(f"  seed {seed}  game {g + 1:>4}/{n_games}  clear {cl*100:5.1f}%  "
                  f"shots/game {sh:.2f}", flush=True)
    env.close()

    n_balls = env_kw["n_balls"]
    speed = summarize_final_eval([x["shots"] for x in games], [x["pocketed"] for x in games],
                                 [x["pocketed_at5"] for x in games], n_balls)
    total_shots = sum(x["shots"] for x in games)
    return {
        "seed":                seed,
        "n_games":             n_games,
        "pocket_rate":         round(sum(x["pocketed"] for x in games) / n_games / n_balls * 100, 2),
        "clear_rate":          round(float(np.mean([x["pocketed"] == n_balls for x in games])) * 100, 2),
        "mean_shots":          round(speed["mean_shots"], 3),
        "mean_shots_cleared":  (round(speed["mean_shots_cleared"], 3)
                                if speed["mean_shots_cleared"] is not None else None),
        "pocket_rate_at5":     round(speed["pocket_rate_at5"] * 100, 2),
        "final_mean_reward":   round(float(np.mean([x["reward"] for x in games])), 4),
        "shot_success_rate":   round(sum(x["successes"] for x in games) / total_shots * 100, 2),
        "random_stage_shots":  sum(x["random_shots"] for x in games),
        "prediction_mismatches": sum(x["mismatches"] for x in games),
        "plan_sec_per_shot":   round(sum(x["plan_sec"] for x in games) / total_shots, 3),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",         type=int,   nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-games",       type=int,   default=500)
    p.add_argument("--n-balls",       type=int,   default=3)
    p.add_argument("--max-steps",     type=int,   default=30)
    p.add_argument("--step-penalty",  type=float, default=1.0)
    p.add_argument("--trunc-penalty", type=float, default=0.0)
    p.add_argument("--offsets-deg",   type=float, nargs="+", default=[-0.4, -0.2, 0.0, 0.2, 0.4])
    p.add_argument("--speeds",        type=float, nargs="+", default=[1.5, 3.0, 5.0])
    p.add_argument("--max-cut-deg",   type=float, default=80.0)
    p.add_argument("--n-random",      type=int,   default=200)
    p.add_argument("--out",           type=str,   default="logs/planner_clear.json")
    return p.parse_args()


def main():
    args = parse_args()
    env_kw  = {"n_balls": args.n_balls, "max_steps": args.max_steps,
               "step_penalty": args.step_penalty, "trunc_penalty": args.trunc_penalty}
    plan_kw = {"offsets_deg": args.offsets_deg, "speeds": args.speeds,
               "max_cut_deg": args.max_cut_deg, "n_random": args.n_random}
    print(f"planner  env={env_kw}  plan={plan_kw}  seeds={args.seeds}  n_games={args.n_games}")

    per_seed = []
    for s in args.seeds:
        res = evaluate_seed(s, args.n_games, env_kw, plan_kw)
        per_seed.append(res)
        print(json.dumps(res), flush=True)

    keys = ["pocket_rate", "clear_rate", "mean_shots", "mean_shots_cleared", "pocket_rate_at5",
            "final_mean_reward", "shot_success_rate", "plan_sec_per_shot"]
    summary = {k: {"mean": round(float(np.mean([r[k] for r in per_seed if r[k] is not None])), 3),
                   "std":  round(float(np.std([r[k] for r in per_seed if r[k] is not None], ddof=1)), 3)
                           if len(per_seed) > 1 else None}
               for k in keys}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"env": env_kw, "plan": plan_kw, "per_seed": per_seed, "summary": summary}, f, indent=2)
    print(f"summary: {json.dumps(summary)}")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
