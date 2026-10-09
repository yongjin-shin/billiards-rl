"""
exp16_wm/teacher_data.py — imitation-learning data from a search-planner teacher.

Plays the clear-speed task with exp16_wm/planner.py and records, for every shot:
  obs            the env observation the policy would see before the shot
  action         the teacher's chosen action (delta_angle, speed)
  good_actions   every candidate the simulation showed pockets ≥1 ball, best first
                 (NaN-padded to max_good) — for later multimodal / set losses
  n_good         how many candidates pocketed (may exceed max_good)

Teachers differ in where candidates come from and which one is chosen:
  ghost   ghost-ball aims × offsets × speeds (geometry), random fallback
  random  uniform-random actions only (no human knowledge)
  robust  ghost candidates find the pocketing (ball, pocket) pairs; the top pairs are
          then scanned densely (angle × speed) and the shot is the centre of the
          widest pocketing angle window — the most forgiving shot, not just any
          shot that goes in. Other scanned pairs' best shots are stored as good_actions.

Usage:
    python -m exp16_wm.teacher_data --teacher ghost --seed 1000 --n-games 7500 \
        --out logs/teacher_data/ghost_s1000.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.planner import (plan_shot, candidate_score, ShotResult, ghost_aims, aim_reference,
                              action_from_phi, simulate_candidate)

TEACHERS: dict[str, dict] = {
    "ghost":  {"offsets_deg": [-0.4, -0.2, 0.0, 0.2, 0.4], "speeds": [1.5, 3.0, 5.0],
               "max_cut_deg": 80.0, "n_random": 200},
    "random": {"offsets_deg": [0.0], "speeds": [3.0], "max_cut_deg": -1.0, "n_random": 200},
}
TEACHERS["robust"] = {**TEACHERS["ghost"], "scan_offsets_deg": [round(-0.5 + 0.04 * i, 4) for i in range(26)],
                      "scan_speeds": [1.0, 2.0, 3.0, 4.0, 5.0], "top_pairs": 2}
CLEAR_ENV = {"n_balls": 3, "max_steps": 30, "step_penalty": 1.0, "trunc_penalty": 0.0}


def widest_window(hits: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    """
    Widest contiguous run of True in hits (aligned with evenly spaced offsets, deg).
    Returns (width_deg, centre_deg); width counts grid cells × spacing. (0, nan) if none.
    """
    hits = np.asarray(hits, dtype=bool)
    if not hits.any():
        return 0.0, float("nan")
    step = float(offsets[1] - offsets[0]) if len(offsets) > 1 else 0.0
    best_len, best_lo, i = 0, 0, 0
    while i < len(hits):
        if hits[i]:
            j = i
            while j + 1 < len(hits) and hits[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len, best_lo = j - i + 1, i
            i = j + 1
        else:
            i += 1
    return best_len * step, float((offsets[best_lo] + offsets[best_lo + best_len - 1]) / 2)


def robust_shot(env: BilliardsEnv, rng: np.random.Generator, plan_kw: dict,
                max_good: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    The most forgiving pocketing shot. A shot counts as good if it pockets ≥1 ball without scratching.
    Returns (action, good_actions (max_good, 2), n_good, window width deg). Falls back to the plain
    planner's choice (width 0) when no ghost-ball pair pockets.
    """
    phi_ref = aim_reference(env)
    aims = ghost_aims(env, plan_kw["max_cut_deg"])
    counts = []
    for _, _, phi in aims:
        n = 0
        for off in plan_kw["offsets_deg"]:
            for v in plan_kw["speeds"]:
                r = simulate_candidate(env, phi_ref, action_from_phi(phi + np.radians(off), phi_ref, v))
                n += int(r[0] > 0 and not r[1])
        counts.append(n)
    top = [k for k in np.argsort(counts)[::-1][:plan_kw["top_pairs"]] if counts[k] > 0]
    offs = np.asarray(plan_kw["scan_offsets_deg"], dtype=np.float64)
    per_pair = []                                   # (width, action) of each scanned pair's best shot
    for k in top:
        phi = aims[k][2]
        best = (0.0, None)
        for v in plan_kw["scan_speeds"]:
            hits = []
            for o in offs:
                r = simulate_candidate(env, phi_ref, action_from_phi(phi + np.radians(o), phi_ref, v))
                hits.append(r[0] > 0 and not r[1])
            width, centre = widest_window(np.array(hits), offs)
            if width > best[0]:
                best = (width, action_from_phi(phi + np.radians(centre), phi_ref, v))
        if best[1] is not None:
            per_pair.append(best)
    good = np.full((max_good, 2), np.nan, dtype=np.float32)
    if not per_pair:
        ghost_kw = {k: plan_kw[k] for k in ["offsets_deg", "speeds", "max_cut_deg", "n_random"]}
        action, meta = plan_shot(env, rng, **ghost_kw)
        g, n_good = pack_good_actions(meta["candidates"], meta["results"], max_good)
        return action, g, n_good, 0.0
    per_pair.sort(key=lambda wa: wa[0], reverse=True)
    for i, (_, a) in enumerate(per_pair[:max_good]):
        good[i] = a
    return per_pair[0][1], good, len(per_pair), per_pair[0][0]


def pack_good_actions(cands: list[np.ndarray], results: list[ShotResult],
                      max_good: int) -> tuple[np.ndarray, int]:
    """
    Pocketing candidates sorted best-first with the planner's own ranking
    (immediate reward, then remaining balls nearer a pocket), NaN-padded.
    Returns (good (max_good, 2), n_good) where n_good counts all pocketing candidates.
    """
    good = [(a, r) for a, r in zip(cands, results) if r[0] > 0]
    good.sort(key=lambda ar: (candidate_score(ar[1][0], ar[1][1]), -ar[1][2]), reverse=True)
    out = np.full((max_good, 2), np.nan, dtype=np.float32)
    for i, (a, _) in enumerate(good[:max_good]):
        out[i] = a
    return out, len(good)


def collect(seed: int, n_games: int, teacher: str, env_kw: dict, max_good: int = 32,
            log_every: int = 500, checkpoint: str | None = None) -> dict[str, np.ndarray]:
    """Play n_games with the teacher and return per-shot arrays."""
    plan_kw = TEACHERS[teacher]
    env = BilliardsEnv(**env_kw)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rec: dict[str, list] = {k: [] for k in
                            ["obs", "action", "good_actions", "n_good", "pocketed", "game", "shot", "win_width"]}
    t0 = time.time()
    for g in range(n_games):
        obs, _ = env.reset()
        done, k = False, 0
        while not done:
            if teacher == "robust":
                action, good, n_good, width = robust_shot(env, rng, plan_kw, max_good)
            else:
                action, meta = plan_shot(env, rng, **plan_kw)
                good, n_good = pack_good_actions(meta["candidates"], meta["results"], max_good)
                width = float("nan")
            next_obs, _, terminated, truncated, info = env.step(action)
            rec["obs"].append(obs)
            rec["action"].append(action)
            rec["good_actions"].append(good)
            rec["n_good"].append(n_good)
            rec["pocketed"].append(info["pocketed_this_step"])
            rec["game"].append(g)
            rec["shot"].append(k)
            rec["win_width"].append(width)
            obs, done, k = next_obs, terminated or truncated, k + 1
        if log_every and (g + 1) % log_every == 0:
            n = len(rec["shot"])
            print(f"  seed {seed}  game {g + 1:>6}/{n_games}  shots {n:>7}  "
                  f"shots/game {n / (g + 1):.2f}  {time.time() - t0:.0f}s", flush=True)
            if checkpoint:
                save(checkpoint, to_arrays(rec), seed, teacher, env_kw)
    env.close()
    return to_arrays(rec)


def to_arrays(rec: dict[str, list]) -> dict[str, np.ndarray]:
    if not rec["obs"]:
        return {"obs": np.zeros((0, 0), np.float32), "action": np.zeros((0, 2), np.float32),
                "good_actions": np.zeros((0, 0, 2), np.float32),
                "win_width": np.zeros(0, np.float32),
                **{k: np.zeros(0, np.int32) for k in ["n_good", "pocketed", "game", "shot"]}}
    return {
        "obs":          np.asarray(rec["obs"], dtype=np.float32),
        "action":       np.asarray(rec["action"], dtype=np.float32),
        "good_actions": np.asarray(rec["good_actions"], dtype=np.float32),
        "win_width":    np.asarray(rec.get("win_width", [np.nan] * len(rec["obs"])), dtype=np.float32),
        **{k: np.asarray(rec[k], dtype=np.int32) for k in ["n_good", "pocketed", "game", "shot"]},
    }


def save(path: str, data: dict[str, np.ndarray], seed: int, teacher: str, env_kw: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = json.dumps({"seed": seed, "teacher": teacher, "plan": TEACHERS[teacher], "env": env_kw})
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(meta), **data)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher",  type=str, default="ghost", choices=list(TEACHERS))
    p.add_argument("--seed",     type=int, default=1000)
    p.add_argument("--n-games",  type=int, default=7500)
    p.add_argument("--max-good", type=int, default=32)
    p.add_argument("--out",      type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"teacher={args.teacher}  seed={args.seed}  n_games={args.n_games}  env={CLEAR_ENV}", flush=True)
    data = collect(args.seed, args.n_games, args.teacher, CLEAR_ENV, args.max_good,
                   checkpoint=args.out)
    save(args.out, data, args.seed, args.teacher, CLEAR_ENV)
    n = len(data["shot"])
    print(f"done: {n} shots, {args.n_games} games, shot success "
          f"{np.mean(data['pocketed'] > 0) * 100:.1f}%  → {args.out}", flush=True)


if __name__ == "__main__":
    main()
