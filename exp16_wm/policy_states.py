"""
exp16_wm/policy_states.py — states the policy itself plays (Exp-16, roadmap T5).

Play games with a trained policy and save every shot's state in the label-file format (obs, game, shot, action, good_actions,
n_good, pocketed, meta) so the teacher-free pipeline (exp16_wm.aim_sims → exp16_wm.bc --made-only) can train on the policy's
own state distribution instead of the teacher's. The stored action is the policy's; --made-only does not use it.

Usage:
    python -m exp16_wm.policy_states --run "logs/experiments_teacherfree/*_s0_*" --infer all_pairs --games 5200 --eval-seed 100 \
        --out logs/teacher_data/policy_t2_s0
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_shards(records: dict[str, np.ndarray], out_dir: str, n_shards: int, meta: dict) -> list[str]:
    """records: obs (N, 23), game (N,), shot (N,), action (N, 2), pocketed (N,) → n_shards label-format files split by whole games."""
    os.makedirs(out_dir, exist_ok=True)
    games = np.unique(records["game"])
    paths = []
    for k, part in enumerate(np.array_split(games, n_shards)):
        m = np.isin(records["game"], part)
        out = os.path.join(out_dir, f"policy_shard{k}.npz")
        act = records["action"][m].astype(np.float32)
        np.savez(out, meta=json.dumps({**meta, "shard": k}), obs=records["obs"][m].astype(np.float32),
                 game=(records["game"][m] - part.min()).astype(np.int32), shot=records["shot"][m].astype(np.int32),
                 action=act, good_actions=act[:, None, :], n_good=np.ones(m.sum(), np.int32), pocketed=records["pocketed"][m].astype(np.int32))
        paths.append(out)
    return paths


def play(policy, n_games: int, eval_seed: int, env_kw: dict) -> dict[str, np.ndarray]:
    """Every shot of n_games games (same protocol as bc.evaluate_policy: reset(seed) once, then n_games resets)."""
    from simulator import BilliardsEnv
    env = BilliardsEnv(**env_kw)
    env.reset(seed=eval_seed)
    obs_l, game_l, shot_l, act_l, pk_l = [], [], [], [], []
    for g in range(n_games):
        obs, _ = env.reset()
        done, k = False, 0
        while not done:
            a = policy.act(obs, deterministic=True)
            obs_l.append(np.asarray(obs, np.float32)); game_l.append(g); shot_l.append(k); act_l.append(np.asarray(a, np.float32))
            obs, r, term, trunc, info = env.step(a)
            pk_l.append(int(info["pocketed_this_step"]))
            done, k = term or trunc, k + 1
    env.close()
    return {"obs": np.stack(obs_l), "game": np.array(game_l), "shot": np.array(shot_l), "action": np.stack(act_l), "pocketed": np.array(pk_l)}


def main() -> None:
    from exp16_wm import eval_by_balls as E
    from exp16_wm.bc import TransformedPolicy
    from exp16_wm.teacher_data import CLEAR_ENV
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--infer", default="all_pairs")
    p.add_argument("--games", type=int, default=5200); p.add_argument("--eval-seed", type=int, default=100)
    p.add_argument("--shards", type=int, default=8); p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    found = glob.glob(args.run)
    if len(found) != 1:
        raise SystemExit(f"expected one run dir, found {found}")
    m = E.load_run(found[0])
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = args.infer, E.ANG31, (1.0,), E.ABS_SPEEDS
    rec = play(TransformedPolicy(m, "relative"), args.games, args.eval_seed, CLEAR_ENV)
    meta = {"teacher": "policy_" + os.path.basename(found[0])[:40].replace(":", "_"), "run": found[0], "infer": args.infer,
            "games": args.games, "eval_seed": args.eval_seed}
    paths = write_shards(rec, args.out, args.shards, meta)
    print(f"saved {len(rec['obs'])} states of {args.games} games in {len(paths)} shards → {args.out}; "
          f"shot success {np.mean(rec['pocketed'] > 0):.1%}, shots/game {len(rec['obs']) / args.games:.3f}", flush=True)


if __name__ == "__main__":
    main()
