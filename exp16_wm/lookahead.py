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

    def __init__(self, scorer, sim, k: int = 10, mode: str = "model", widths: tuple[int, ...] | None = None):
        """widths = shots simulated per level: (10,) looks one shot ahead (= k), (10, 5) two shots ahead, …"""
        if mode not in ("model", "sim"):
            raise ValueError(f"mode must be 'model' or 'sim', got {mode!r}")
        self.widths = tuple(widths) if widths else (k,)
        if not self.widths or min(self.widths) < 1:
            raise ValueError(f"widths must be positive, got {self.widths}")
        self.scorer, self.sim, self.k, self.mode = scorer, sim, self.widths[0], mode
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

    def _best(self, raw: np.ndarray, widths: tuple[int, ...]) -> tuple[float, np.ndarray | None]:
        """(value, best env action) of a position: with no widths left, the best shot's own chance (the position's ease);
        otherwise over its top widths[0] shots, shot chance × value of the position after it (one level deeper)."""
        p, acts = self._candidates(raw)
        if len(p) == 0:
            return 0.0, None
        if not widths:
            i = int(p.argmax()); return float(p[i]), acts[i]
        best, best_a = -1.0, None
        for i in np.argsort(-p)[:widths[0]]:
            nxt, made, scratch, cleared = self.sim(raw, acts[i])
            ok = made and not scratch
            now = float(ok) if self.mode == "sim" else float(p[i])      # sim: the simulated outcome verifies the shot
            if now == 0.0:
                score = 0.0
            elif cleared:
                score = now
            else:                                                     # the same depth for every shot (a shallower value for some
                score = now * self._best(nxt, widths[1:])[0]          # shots would favour them: fewer probabilities multiplied)
            if score > best:
                best, best_a = score, acts[i]
        return best, best_a

    def act(self, raw_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        _, a = self._best(np.asarray(raw_obs, np.float32), self.widths)
        return np.array([0.0, 1.0], np.float32) if a is None else np.asarray(a, np.float32)


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
    p.add_argument("--widths", type=int, nargs="+", default=None, help="shots simulated per level, e.g. 10 5 = two shots ahead")
    p.add_argument("--mode", default="model", choices=["model", "sim"]); p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS)); p.add_argument("--games-per-seed", type=int, default=GAMES_PER_SEED)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    runs = [find_run(g) for g in args.runs]
    pol = LookaheadPolicy(load_ensemble(runs, args.aim), make_sim(args.n_balls), args.k, args.mode, args.widths)
    t0 = time.time()
    res = play_paired(pol, tuple(args.seeds), args.games_per_seed, clear_env_kw(args.n_balls))
    meta = {"runs": runs, "infer": f"lookahead_{args.mode}_w{'-'.join(map(str, pol.widths))}", "n_balls": args.n_balls, "eval_seeds": args.seeds,
            "games_per_seed": args.games_per_seed, "seconds": round(time.time() - t0, 1), "machine": machine_info()}
    save(args.out, res, meta)
    print(_line(os.path.basename(args.out), summarize(res)), f"  ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
