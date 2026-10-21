"""
exp16_wm/search_data.py — games played with simulator search, recorded for expert iteration (Exp-16, roadmap "탐색 결과로 학습").

A trained model plus look-ahead search (exp16_wm.lookahead, mode sim) plays games; every decision records
  - every shot the search simulated (game states and the positions inside the search tree): state, action, which (ball, pocket)
    went in, scratch → more data for the made head, on exactly the candidates the model weighs;
  - the top level's scored shots: action, its pair, the search score (this shot × value of the position after it) → Q head targets;
  - for the real game states: did the rest of the game go without a miss → V head targets.

Output (one shard per process / eval seed), readable by exp16_wm.bc:
  <out>/labels/search_s<seed>.npz : obs (states: game states first, then search-tree states), game, shot, action, good_actions, n_good,
                                    pocketed, meta, is_game_state, v_target (NaN for tree states), q_state, q_action, q_pair, q_score
  <out>/sims/search_s<seed>.npz   : obs, game, sim_state, sim_action, sim_made (bitmask), sim_scratch_replay

Usage:
    python -m exp16_wm.search_data --runs "<run glob>" --n-balls 6 --widths 10 5 3 --games 50 --eval-seed 1000 --out logs/teacher_data/search_g6
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Recorder:
    """Collects the look-ahead's simulations and top-level scores, keyed by state (dedup on the exact observation)."""

    def __init__(self):
        self.index: dict[bytes, int] = {}
        self.states: list[np.ndarray] = []
        self.game_of: list[int] = []
        self.sims: list[tuple[int, np.ndarray, np.ndarray, bool]] = []
        self.q: list[tuple[int, np.ndarray, int, float]] = []
        self.game = 0

    def state_id(self, raw: np.ndarray) -> int:
        key = np.asarray(raw, np.float32).tobytes()
        if key not in self.index:
            self.index[key] = len(self.states); self.states.append(np.asarray(raw, np.float32)); self.game_of.append(self.game)
        return self.index[key]

    def __call__(self, kind: str, **d) -> None:
        sid = self.state_id(d["raw"])
        if kind == "sim":
            mask = d["made_mask"]
            self.sims.append((sid, d["action"], np.zeros(0, bool) if mask is None else np.asarray(mask, bool), bool(d["scratch"])))
        elif kind == "top":
            for a, j, sc in zip(d["actions"], d["pairs"], d["scores"]):
                self.q.append((sid, np.asarray(a, np.float32), int(j), float(sc)))


def runout_from_here(succ: list[bool]) -> np.ndarray:
    """Per shot of one game: 1 if this shot and every later one pocketed (the rest of the rack went without a miss)."""
    out = np.zeros(len(succ), np.float32)
    ok = True
    for i in range(len(succ) - 1, -1, -1):
        ok = ok and succ[i]; out[i] = float(ok)
    return out


def play_and_record(policy, rec: Recorder, n_games: int, eval_seed: int, env_kw: dict) -> dict[str, np.ndarray]:
    """Play n_games with the search policy; returns the real game states' (id, game, shot, action, pocketed, v_target)."""
    from simulator import BilliardsEnv
    env = BilliardsEnv(**env_kw); env.reset(seed=eval_seed)
    ids, games, shots, acts, pk, vt = [], [], [], [], [], []
    for g in range(n_games):
        rec.game = g
        obs, _ = env.reset(); done, k, succ = False, 0, []
        while not done:
            sid = rec.state_id(obs)
            a = policy.act(obs)
            obs2, _, term, trunc, info = env.step(a)
            n = int(info["pocketed_this_step"]); s = n > 0 and not info.get("scratch", False)
            ids.append(sid); games.append(g); shots.append(k); acts.append(np.asarray(a, np.float32)); pk.append(n); succ.append(s)
            obs, done, k = obs2, term or trunc, k + 1
        vt.extend(runout_from_here(succ))
    env.close()
    return {"id": np.array(ids), "game": np.array(games), "shot": np.array(shots), "action": np.stack(acts), "pocketed": np.array(pk),
            "v_target": np.array(vt, np.float32)}


def write_shard(rec: Recorder, played: dict[str, np.ndarray], out: str, name: str, meta: dict) -> tuple[str, str]:
    """Reorder states (game states first, in play order, then tree states) and write the label + sims files."""
    from exp16_wm.sim_identity import to_bitmask
    n_pairs = (len(rec.states[0]) - 14) // 3 * 6
    game_ids = list(dict.fromkeys(played["id"].tolist()))
    tree_ids = [i for i in range(len(rec.states)) if i not in set(game_ids)]
    order = game_ids + tree_ids
    new = {old: k for k, old in enumerate(order)}
    N = len(order); G = len(game_ids)
    obs = np.stack([rec.states[i] for i in order])
    game = np.array([rec.game_of[i] for i in order], np.int32)
    first = {sid: k for k, sid in enumerate(played["id"].tolist())}          # a game state visited twice keeps its first visit
    shot = np.zeros(N, np.int32); action = np.zeros((N, 2), np.float32); pocketed = np.zeros(N, np.int32)
    v_target = np.full(N, np.nan, np.float32)
    for sid, k in first.items():
        j = new[sid]; shot[j], action[j], pocketed[j], v_target[j] = played["shot"][k], played["action"][k], played["pocketed"][k], played["v_target"][k]
    is_game = np.zeros(N, bool); is_game[:G] = True
    q = rec.q
    os.makedirs(os.path.join(out, "labels"), exist_ok=True); os.makedirs(os.path.join(out, "sims"), exist_ok=True)
    lp = os.path.join(out, "labels", f"{name}.npz"); sp = os.path.join(out, "sims", f"{name}.npz")
    np.savez(lp, meta=json.dumps(meta), obs=obs, game=game, shot=shot, action=action, good_actions=action[:, None, :], n_good=np.ones(N, np.int32),
             pocketed=pocketed, is_game_state=is_game, v_target=v_target,
             q_state=np.array([new[s] for s, _, _, _ in q], np.int32), q_action=np.stack([a for _, a, _, _ in q]) if q else np.zeros((0, 2), np.float32),
             q_pair=np.array([j for _, _, j, _ in q], np.int16), q_score=np.array([sc for _, _, _, sc in q], np.float32))
    made = np.stack([m if len(m) else np.zeros(n_pairs, bool) for _, _, m, _ in rec.sims]) if rec.sims else np.zeros((0, n_pairs), bool)
    np.savez(sp, obs=obs, game=game, sim_state=np.array([new[s] for s, _, _, _ in rec.sims], np.int32),
             sim_action=np.stack([a for _, a, _, _ in rec.sims]) if rec.sims else np.zeros((0, 2), np.float32),
             sim_made=to_bitmask(made), sim_scratch_replay=np.array([sc for _, _, _, sc in rec.sims], bool))
    return lp, sp


def main() -> None:
    from exp16_wm.lookahead import LookaheadPolicy, make_sim
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.paired_eval import find_run
    from exp16_wm.teacher_data import clear_env_kw
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--aim", default="mean", choices=["shared", "mean"])
    p.add_argument("--n-balls", type=int, default=6); p.add_argument("--widths", type=int, nargs="+", default=[10, 5, 3])
    p.add_argument("--games", type=int, default=50); p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=1); p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    name = f"search_s{args.eval_seed}"
    if os.path.exists(os.path.join(args.out, "labels", f"{name}.npz")):
        print(f"{name}: already done, skipped"); return
    runs = [find_run(g) for g in args.runs]
    rec = Recorder()
    pol = LookaheadPolicy(load_ensemble(runs, args.aim), make_sim(args.n_balls), mode="sim", widths=tuple(args.widths), recorder=rec)
    t0 = time.time()
    played = play_and_record(pol, rec, args.games, args.eval_seed, clear_env_kw(args.n_balls))
    meta = {"teacher": "search", "runs": runs, "widths": args.widths, "n_balls": args.n_balls, "games": args.games, "eval_seed": args.eval_seed}
    lp, sp = write_shard(rec, played, args.out, name, meta)
    succ = played["pocketed"] > 0
    print(f"{name}: {args.games} games, {len(played['id'])} game states + {len(rec.states) - len(set(played['id'].tolist()))} tree states, "
          f"{len(rec.sims)} simulations, {len(rec.q)} Q targets, shot success {succ.mean():.1%}, run-out {np.mean([played['v_target'][played['shot'] == 0]]):.1%} "
          f"({time.time() - t0:.0f}s) → {lp}", flush=True)


if __name__ == "__main__":
    main()
