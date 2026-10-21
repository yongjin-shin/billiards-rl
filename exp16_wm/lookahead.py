"""
exp16_wm/lookahead.py — one-shot look-ahead: choose the shot that goes in AND leaves the easiest next shot (Exp-16, G1).

The made head only asks "does this shot go in". With a miss ending the turn (G1), the next position matters too. For the K shots the
made head likes best, the simulator plays each one from the current table (a separate env, so the game's own RNG is untouched) and
gives the position after it; the same model then scores every candidate of that position, and the best one is the next-shot chance.

  score(a) = p_now(a) × p_next(s'(a)),   p_next = max over the next position's candidates of P(made), 1 if the rack is cleared

p_now is the model's own probability (mode "model") or the simulated outcome itself, 1 = pocketed without a scratch (mode "sim":
the K simulations then also verify the shot — planning with the simulator, a budget of K shots per decision).

Usage:
    python -m exp16_wm.lookahead --runs "<run glob>" ... --n-balls 6 --k 10 --mode model --out logs/eval_paired/mac/LA/x.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LookaheadPolicy:
    """act(raw obs) → env action. scorer: MadeEnsemble (one member = a single model); sim(raw obs, env action) →
    (next raw obs, pocketed something, scratch, rack cleared)."""

    def __init__(self, scorer, sim, k: int = 10, mode: str = "model"):
        if mode not in ("model", "sim"):
            raise ValueError(f"mode must be 'model' or 'sim', got {mode!r}")
        self.scorer, self.sim, self.k, self.mode = scorer, sim, k, mode
        self.sc, self.bi = scorer.act_scale, scorer.act_bias

    @torch.no_grad()
    def _candidates(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(probabilities (M,), env actions (M, 2)) of every live candidate."""
        from exp16_wm.obs_transform import relative_obs
        z, cand = self.scorer.scores(torch.as_tensor(relative_obs(raw[None])))
        z, cand = z[0].reshape(-1), cand[0].reshape(-1, 2)
        live = torch.isfinite(z)
        return torch.sigmoid(z[live]).numpy(), (cand[live] * self.sc + self.bi).numpy()

    def next_chance(self, raw_next: np.ndarray) -> float:
        p, _ = self._candidates(raw_next)
        return float(p.max()) if len(p) else 0.0

    def act(self, raw_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raw = np.asarray(raw_obs, np.float32)
        p, acts = self._candidates(raw)
        if len(p) == 0:
            return np.array([0.0, 1.0], np.float32)
        top = np.argsort(-p)[:self.k]
        best, best_a = -1.0, acts[top[0]]
        for i in top:
            nxt, made, scratch, cleared = self.sim(raw, acts[i])
            ok = made and not scratch
            if self.mode == "sim":                                   # the simulated outcome verifies the shot
                score = 0.0 if not ok else (1.0 if cleared else self.next_chance(nxt))
            else:                                                    # the model's chance now × how easy the position after it is
                score = float(p[i]) * (1.0 if cleared else self.next_chance(nxt))
            if score > best:
                best, best_a = score, acts[i]
        return np.asarray(best_a, np.float32)


def make_sim(n_balls: int):
    """A private env for look-ahead simulations: (raw obs, env action) → (next raw obs, pocketed, scratch, cleared)."""
    from simulator import BilliardsEnv
    from exp16_wm.relabel import restore_state
    from exp16_wm.teacher_data import clear_env_kw
    env = BilliardsEnv(**clear_env_kw(n_balls)); env.reset(seed=12345)

    def sim(raw: np.ndarray, action: np.ndarray):
        restore_state(env, raw)
        nxt, _, term, _, info = env.step(np.asarray(action, np.float32))
        return np.asarray(nxt, np.float32), int(info["pocketed_this_step"]) > 0, bool(info.get("scratch", False)), bool(term and all(env._pocketed.values()))
    return sim


def main() -> None:
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.paired_eval import EVAL_SEEDS, GAMES_PER_SEED, _line, find_run, machine_info, play_paired, save, summarize
    from exp16_wm.teacher_data import clear_env_kw
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--aim", default="mean", choices=["shared", "mean"])
    p.add_argument("--n-balls", type=int, default=6); p.add_argument("--k", type=int, default=10)
    p.add_argument("--mode", default="model", choices=["model", "sim"]); p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS)); p.add_argument("--games-per-seed", type=int, default=GAMES_PER_SEED)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    runs = [find_run(g) for g in args.runs]
    pol = LookaheadPolicy(load_ensemble(runs, args.aim), make_sim(args.n_balls), args.k, args.mode)
    t0 = time.time()
    res = play_paired(pol, tuple(args.seeds), args.games_per_seed, clear_env_kw(args.n_balls))
    meta = {"runs": runs, "infer": f"lookahead_{args.mode}_k{args.k}", "n_balls": args.n_balls, "eval_seeds": args.seeds,
            "games_per_seed": args.games_per_seed, "seconds": round(time.time() - t0, 1), "machine": machine_info()}
    save(args.out, res, meta)
    print(_line(os.path.basename(args.out), summarize(res)), f"  ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
