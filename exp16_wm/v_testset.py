"""
exp16_wm/v_testset.py — a V test that asks what look-ahead asks: among the positions the candidate shots of one table leave, which
is better? (Exp-16, V5)

For every origin position: the made head's top-k shots are played; a shot that pockets a ball without a scratch leaves a next
position, from which greedy plays on until its first miss. A candidate's true value = balls this shot + balls greedy pockets after
it (0 if the shot fails; a shot that clears the rack is worth its balls). The next positions come from the simulator, so this
measures V as if the cue-ball landing spot were predicted perfectly.

Metrics (evaluate_testset):
  pair_acc  — over pairs of successful, non-clearing candidates of the same origin whose later balls differ: the share where the
              higher-valued next position (by the predictor) is the one greedy pocketed more from;
  pick      — mean true value of the candidate chosen by score = made × (1 + value(next)) (lookahead --objective balls);
  best / greedy / random — the same for the best candidate, made alone, a uniformly random candidate.

Usage:
    python -m exp16_wm.v_testset --runs "<run glob>" --data logs/teacher_data/v2/diag_s950.npz --workers 8 \
        --out logs/teacher_data/v5/testset_s950.npz
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def candidate_values(succ: np.ndarray, balls: np.ndarray, k_next: np.ndarray) -> np.ndarray:
    """True value of each candidate: balls this shot + balls after it if the shot succeeded, else 0 (any shape)."""
    return np.where(succ, balls + np.maximum(k_next, 0), 0).astype(float)


def pair_accuracy(pred: np.ndarray, truth: np.ndarray, use: np.ndarray) -> float:
    """pred, truth, use (N, K): within each row, over pairs (i, j) both in use with truth_i ≠ truth_j, the share where pred orders
    them like truth (pred ties count ½). NaN without such pairs."""
    hit, n = 0.0, 0
    for p, t, u in zip(pred, truth, use):
        idx = np.nonzero(u)[0]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if t[i] == t[j]:
                    continue
                d = (p[i] - p[j]) * np.sign(t[i] - t[j])
                hit += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
                n += 1
    return hit / n if n else float("nan")


def pick_value(score: np.ndarray, value: np.ndarray, valid: np.ndarray) -> float:
    """Mean over rows of the true value of the argmax-score valid candidate (rows without a valid candidate are skipped)."""
    s = np.where(valid, score, -np.inf)
    rows = valid.any(axis=1)
    return float(value[rows, s[rows].argmax(axis=1)].mean())


def evaluate_testset(t: dict[str, np.ndarray], next_value: np.ndarray) -> dict[str, float]:
    """t: the saved test set (made, succ, balls, k_next, cleared, valid; (N, K)); next_value (N, K): the predictor's value of each
    candidate's next position in balls (ignored where the shot failed; a cleared rack counts 0)."""
    value = candidate_values(t["succ"], t["balls"], t["k_next"])
    nv = np.where(t["cleared"], 0.0, next_value)
    use = t["valid"] & t["succ"] & ~t["cleared"]
    rnd = np.where(t["valid"], value, np.nan)
    return {"pair_acc": pair_accuracy(nv, t["k_next"], use),
            "pick": pick_value(t["made"] * (1.0 + nv), value, t["valid"]),
            "best": pick_value(value, value, t["valid"]),
            "greedy": pick_value(t["made"], value, t["valid"]),
            "random": float(np.nanmean(np.nanmean(rnd[t["valid"].any(axis=1)], axis=1)))}


def _worker(job: tuple) -> dict[str, np.ndarray]:
    raws, lefts, runs, n_balls, topk = job
    import torch
    torch.set_num_threads(1)
    from exp16_wm.bc import TransformedPolicy
    from exp16_wm.lookahead import make_sim
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.v_data import candidates, pocketed_count, rollout
    scorer = load_ensemble(runs, "mean")
    pol = TransformedPolicy(scorer, "relative")
    act = lambda o: pol.act(o, deterministic=True)
    sim = make_sim(n_balls)
    N, K, D = len(raws), topk, raws.shape[1]
    out = {"made": np.zeros((N, K)), "succ": np.zeros((N, K), bool), "balls": np.zeros((N, K), np.int64),
           "k_next": np.full((N, K), -1, np.int64), "cleared": np.zeros((N, K), bool), "valid": np.zeros((N, K), bool),
           "next_obs": np.zeros((N, K, D), np.float32), "left_next": np.zeros((N, K), np.int64)}
    for i, (raw, left) in enumerate(zip(raws, lefts)):
        probs, acts = candidates(scorer, raw)
        for c, j in enumerate(np.argsort(-probs)[:K]):
            nxt, made, scratch, cleared = sim(raw, acts[j])[:4]
            ok = bool(made and not scratch)
            n_in = pocketed_count(nxt) - pocketed_count(raw)
            left_n = int(left) - n_in
            out["made"][i, c], out["valid"][i, c], out["succ"][i, c] = probs[j], True, ok
            out["balls"][i, c], out["next_obs"][i, c], out["left_next"][i, c] = n_in, nxt, left_n
            out["cleared"][i, c] = ok and (cleared or left_n == 0)
            if ok and not out["cleared"][i, c]:
                out["k_next"][i, c] = rollout(sim, act, nxt, left_n)
    return out


def next_values(t: dict[str, np.ndarray], run: str) -> np.ndarray:
    """(N, K) a V run's value (balls) of every candidate's next position (its own horizon), 0 where there is no next position."""
    from exp16_wm import eval_by_balls as E
    from exp16_wm.lookahead import _run_v_horizon
    from exp16_wm.train_v import predict_balls
    N, K, D = t["next_obs"].shape
    use = (t["valid"] & t["succ"] & ~t["cleared"]).reshape(-1)
    out = np.zeros(N * K)
    out[use] = predict_balls(E.load_run(run), t["next_obs"].reshape(-1, D)[use], t["left_next"].reshape(-1)[use], _run_v_horizon(run))
    return out.reshape(N, K)


def first_step_values(t: dict[str, np.ndarray], run: str) -> np.ndarray | None:
    """(N, K) a hazard V run's P(the first shot from each candidate's next position goes in), 0 where there is no next position;
    None for a run without a hazard head."""
    from exp16_wm import eval_by_balls as E
    from exp16_wm.train_v import predict_first_step
    m = E.load_run(run)
    if not getattr(m, "v_steps", 0):
        return None
    N, K, D = t["next_obs"].shape
    use = (t["valid"] & t["succ"] & ~t["cleared"]).reshape(-1)
    out = np.zeros(N * K)
    out[use] = predict_first_step(m, t["next_obs"].reshape(-1, D)[use])
    return out.reshape(N, K)


def made_max_values(t: dict[str, np.ndarray], runs: list[str]) -> np.ndarray:
    """(N, K) baseline: the made head's highest probability on each next position (how easy the next shot looks)."""
    import torch
    from exp16_wm.made_ensemble import load_ensemble
    from exp16_wm.obs_transform import relative_obs
    ens = load_ensemble(runs, "mean")
    N, K, D = t["next_obs"].shape
    use = (t["valid"] & t["succ"] & ~t["cleared"]).reshape(-1)
    rel = torch.as_tensor(relative_obs(t["next_obs"].reshape(-1, D)[use]))
    with torch.no_grad():
        z = torch.cat([ens.scores(rel[i:i + 1024])[0].reshape(len(rel[i:i + 1024]), -1).max(1).values for i in range(0, len(rel), 1024)])
    out = np.zeros(N * K)
    out[use] = torch.sigmoid(z).numpy()
    return out.reshape(N, K)


def score_main(args: argparse.Namespace) -> None:
    """Score V runs (and the made-max baseline) on a saved test set."""
    from exp16_wm.paired_eval import find_run
    z = np.load(args.score, allow_pickle=True)
    t = {k: z[k] for k in ("made", "succ", "balls", "k_next", "cleared", "valid", "next_obs", "left_next")}
    rows = {"made max (next)": made_max_values(t, [find_run(g) for g in args.runs])}
    for r in args.v_runs:
        rows[r] = next_values(t, find_run(r))
    for name, nv in rows.items():
        m = evaluate_testset(t, nv)
        print(f"{name}: " + "  ".join(f"{n} {v:.3f}" for n, v in m.items()), flush=True)


def main() -> None:
    from multiprocessing import Pool
    from exp16_wm.code_version import code_version
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.paired_eval import find_run
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--data", default="")
    p.add_argument("--topk", type=int, default=10); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N origins only (smoke test)"); p.add_argument("--out", default="")
    p.add_argument("--score", default="", help="score --v-runs on this saved test set instead of building one")
    p.add_argument("--v-runs", nargs="*", default=[])
    args = p.parse_args()
    if args.score:
        return score_main(args)
    z = np.load(args.data, allow_pickle=True)
    rows = np.nonzero(z["left"] > 0)[0]
    if args.limit:
        rows = rows[:args.limit]
    obs, left = z["obs"][rows], z["left"][rows]
    runs = [find_run(g) for g in args.runs]
    chunks = [c for c in np.array_split(np.arange(len(rows)), max(1, args.workers * 4)) if len(c)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(_worker, [(obs[c], left[c], runs, n_balls_raw(obs.shape[1]), args.topk) for c in chunks])
    t = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, rows=rows, obs=obs, left=left, **t,
             meta=str({**vars(args), "git_commit": code_version(), "seconds": round(time.time() - t0, 1)}))
    use = t["valid"] & t["succ"] & ~t["cleared"]
    print(f"saved {args.out}: {len(rows)} origins, {t['valid'].sum()} candidates, {t['succ'][t['valid']].mean():.1%} succeeded, "
          f"{use.sum()} next positions to rank ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
