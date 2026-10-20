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


ABS7 = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)


class ExplorePolicy:
    """With probability p an exploratory shot: a random live pair, its (learned) aim ± 0.3° uniformly, one of 7 absolute speeds at random;
    otherwise the base policy's action. `model` must give an aim (made2_geom)."""

    def __init__(self, base, model, p: float, seed: int = 0):
        self.base, self.m, self.p = base, model, p
        self.rng = np.random.default_rng(seed)
        self.scale, self.bias = model.act_scale.numpy().astype(np.float64), model.act_bias.numpy().astype(np.float64)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.rng.random() >= self.p:
            return self.base.act(obs, deterministic=deterministic)
        from exp16_wm.obs_transform import relative_obs
        with torch.no_grad():
            o = self.m.forward_full(torch.as_tensor(relative_obs(np.asarray(obs, np.float32)))[None])
        live = np.flatnonzero(o["valid"][0].numpy())
        j = int(self.rng.choice(live))
        d = float(o["aim"][0, j]) - self.bias[0]
        u = np.clip(np.arctan2(np.sin(d), np.cos(d)) / self.scale[0] + self.rng.uniform(-0.3, 0.3) / 180.0, -1.0, 1.0)
        v = (float(self.rng.choice(ABS7)) - self.bias[1]) / self.scale[1]
        return np.array([u * self.scale[0] + self.bias[0], v * self.scale[1] + self.bias[1]], np.float32)


def random_layouts(n: int, pockets: np.ndarray, table: tuple[float, float], seed: int = 0, lo: float = 0.05, hi: float = 0.95,
                   min_gap_m: float = 0.062) -> dict[str, np.ndarray]:
    """n raw observations (23,) without games: 1/2/3 balls left with equal probability, cue and live balls uniform over the table
    (normalised lo..hi) and at least min_gap_m apart (2 ball radii + margin), the others pocketed. Same record keys as play()."""
    rng = np.random.default_rng(seed)
    W, L = table
    obs = np.zeros((n, 23), np.float32)
    for i in range(n):
        n_live = int(rng.integers(1, 4))
        live = np.sort(rng.choice(3, size=n_live, replace=False))
        pts = []
        while len(pts) < 1 + n_live:
            xy = rng.uniform(lo, hi, size=2)
            if all(np.hypot((xy[0] - q[0]) * W, (xy[1] - q[1]) * L) >= min_gap_m for q in pts):
                pts.append(xy)
        obs[i, 0:2] = pts[0]
        for b in range(3):
            if b in live:
                obs[i, 2 + 3 * b:4 + 3 * b] = pts[1 + list(live).index(b)]
            else:
                obs[i, 4 + 3 * b] = 1.0
        obs[i, 11:23] = pockets
    return {"obs": obs, "game": np.arange(n), "shot": np.zeros(n, np.int64), "action": np.zeros((n, 2), np.float32), "pocketed": np.zeros(n, np.int64)}


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
    p.add_argument("--run", default=None, help="policy run dir (not needed with --random-layouts)"); p.add_argument("--infer", default="all_pairs")
    p.add_argument("--explore", type=float, default=0.0, help="probability of an exploratory shot (random live pair, aim ± 0.3°, random speed)")
    p.add_argument("--random-layouts", type=int, default=0, help="> 0: no games, this many random table layouts instead")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, default=5200); p.add_argument("--eval-seed", type=int, default=100)
    p.add_argument("--shards", type=int, default=8); p.add_argument("--out", required=True)
    p.add_argument("--n-balls", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.random_layouts:
        from simulator import BilliardsEnv
        env = BilliardsEnv(**CLEAR_ENV); o0, _ = env.reset(seed=0)
        rec = random_layouts(args.random_layouts, np.asarray(o0[11:23], np.float32), (env.table_width, env.table_length), seed=args.seed)
        env.close()
        meta = {"teacher": "random_layouts", "n": args.random_layouts, "seed": args.seed}
        paths = write_shards(rec, args.out, args.shards, meta)
        print(f"saved {len(rec['obs'])} random layouts in {len(paths)} shards → {args.out}", flush=True)
        return
    found = glob.glob(args.run)
    if len(found) != 1:
        raise SystemExit(f"expected one run dir, found {found}")
    m = E.load_run(found[0])
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = args.infer, E.ANG31, (1.0,), E.ABS_SPEEDS
    pol = TransformedPolicy(m, "relative")
    if args.explore > 0:
        pol = ExplorePolicy(pol, m, args.explore, seed=args.seed)
    from exp16_wm.teacher_data import clear_env_kw
    rec = play(pol, args.games, args.eval_seed, clear_env_kw(args.n_balls))
    meta = {"teacher": "policy_" + os.path.basename(found[0])[:40].replace(":", "_"), "run": found[0], "infer": args.infer,
            "games": args.games, "eval_seed": args.eval_seed, "explore": args.explore, "n_balls": args.n_balls}
    paths = write_shards(rec, args.out, args.shards, meta)
    print(f"saved {len(rec['obs'])} states of {args.games} games in {len(paths)} shards → {args.out}; "
          f"shot success {np.mean(rec['pocketed'] > 0):.1%}, shots/game {len(rec['obs']) / args.games:.3f}", flush=True)


if __name__ == "__main__":
    main()
