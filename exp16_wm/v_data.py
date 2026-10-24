"""
exp16_wm/v_data.py — V training data with a ball-count reward: greedy games plus branch positions (Exp-16, V2).

Reward (user decision 2026-10-24): +1 per ball pocketed before the first miss or scratch (a miss ends the turn, G1). The 0/1 run-out
signal was too weak — "5 in, then a miss" and "1 in, then a miss" looked the same.
V(s) = the number of balls the policy (PRm greedy) pockets from s before its first miss. The simulator and the policy are both
deterministic, so one play-out is the exact value of that policy (Monte Carlo, no noise).

Positions:
  game   — every position of a greedy game (label from the game itself: balls of the consecutive successes from that shot on);
  branch — at each position, with probability branch_prob, one shot from the made head's top-k other than the greedy one is played
           in a private env, and greedy plays on from the position it leaves until the first miss: the "had another shot been
           played" positions that look-ahead asks V about, which greedy games alone never visit.

Usage:
    python -m exp16_wm.v_data --runs "<run glob>" --n-balls 6 --seed 3000 --games 2000 --out logs/teacher_data/v2/s3000.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def future_balls(ok: np.ndarray, balls: np.ndarray) -> np.ndarray:
    """(T,) balls pocketed from shot t on before the first failed shot (a failed shot t gives 0)."""
    out, run = np.zeros(len(ok), np.int64), 0
    for t in range(len(ok) - 1, -1, -1):
        run = run + int(balls[t]) if ok[t] else 0
        out[t] = run
    return out


def pocketed_count(raw: np.ndarray) -> int:
    from exp16_wm.obs_transform import n_balls_raw
    n = n_balls_raw(len(raw))
    return int(sum(raw[2 + 3 * i + 2] > 0.5 for i in range(n)))


def rollout(sim, act, raw: np.ndarray, max_shots: int) -> int:
    """Balls the policy act(raw) → env action pockets from raw before its first miss / scratch (sim as in lookahead.make_sim)."""
    total, obs = 0, raw
    for _ in range(max_shots):
        nxt, made, scratch, cleared = sim(obs, act(obs))[:4]
        if not made or scratch:
            break
        total += pocketed_count(nxt) - pocketed_count(obs)
        if cleared:
            break
        obs = nxt
    return total


def branch_action(probs: np.ndarray, acts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray | None:
    """A random shot among the k most likely candidates other than the most likely one (None with fewer than 2 candidates)."""
    order = np.argsort(-probs)[:k]
    if len(order) < 2:
        return None
    return acts[order[rng.integers(1, len(order))]]


@torch.no_grad()
def candidates(scorer, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(made probabilities (M,), env actions (M, 2)) of every live candidate of the position."""
    from exp16_wm.obs_transform import relative_obs
    z, cand = scorer.scores(torch.as_tensor(relative_obs(raw[None])))
    z, cand = z[0].reshape(-1), cand[0].reshape(-1, 2)
    live = torch.isfinite(z)
    return torch.sigmoid(z[live]).numpy(), (cand[live] * scorer.act_scale + scorer.act_bias).numpy()


def generate(scorer, n_balls: int, seed: int, games: int, branch_prob: float, topk: int) -> dict[str, np.ndarray]:
    from simulator import BilliardsEnv
    from exp16_wm.bc import TransformedPolicy
    from exp16_wm.lookahead import make_sim
    from exp16_wm.teacher_data import clear_env_kw
    pol = TransformedPolicy(scorer, "relative")
    act = lambda o: pol.act(o, deterministic=True)
    env, sim, rng = BilliardsEnv(**clear_env_kw(n_balls)), make_sim(n_balls), np.random.default_rng(seed)
    env.reset(seed=seed)
    obs_l, k_l, left_l, game_l, br_l = [], [], [], [], []
    for g in range(games):
        obs, _ = env.reset()
        done, st, ok, balls, lefts = False, [], [], [], []
        while not done:
            raw = np.asarray(obs, np.float32)
            left = n_balls - pocketed_count(raw)
            if rng.random() < branch_prob:
                a = branch_action(*candidates(scorer, raw), topk, rng)
                if a is not None:
                    nxt, made, scratch, cleared = sim(raw, a)[:4]
                    left_n = n_balls - pocketed_count(nxt)
                    if not cleared and left_n > 0:
                        obs_l.append(nxt); k_l.append(rollout(sim, act, nxt, left_n)); left_l.append(left_n)
                        game_l.append(g); br_l.append(True)
            st.append(raw)
            obs, _, term, trunc, info = env.step(act(raw))
            done = term or trunc
            n = int(info["pocketed_this_step"])
            ok.append(n > 0 and not info.get("scratch", False)); balls.append(n)
            lefts.append(left)
        k = future_balls(np.array(ok), np.array(balls))
        obs_l += st; k_l += list(np.minimum(k, lefts)); left_l += lefts; game_l += [g] * len(st); br_l += [False] * len(st)
    env.close()
    return {"obs": np.stack(obs_l).astype(np.float32), "k": np.array(k_l, np.int64), "left": np.array(left_l, np.int64),
            "game": np.array(game_l, np.int64), "branch": np.array(br_l, bool)}


def main() -> None:
    from exp16_wm.code_version import code_version
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.paired_eval import find_run
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--n-balls", type=int, default=6)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--games", type=int, default=1000)
    p.add_argument("--branch-prob", type=float, default=0.5); p.add_argument("--topk", type=int, default=10)
    p.add_argument("--threads", type=int, default=1); p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    d = generate(load_ensemble([find_run(g) for g in args.runs], "mean"), args.n_balls, args.seed, args.games, args.branch_prob, args.topk)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **d, meta=str({**vars(args), "git_commit": code_version(), "seconds": round(time.time() - t0, 1)}))
    g = ~d["branch"]
    print(f"saved {args.out}: {len(d['k'])} positions ({d['branch'].sum()} branch), mean balls {d['k'].mean():.2f} of {d['left'].mean():.2f} left, "
          f"cleared from game positions {np.mean(d['k'][g] == d['left'][g]):.1%} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
