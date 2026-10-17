"""
exp16_wm/paired_eval.py — paired game evaluation (Exp-16): every model plays the same games, per-game results are saved.

The same protocol for every model (eval seeds 1-4 × 500 games: reset(seed) once, then one reset per game) and the SE of a difference
is the SE of the per-game difference. Saves shots per game, shot success and balls per shot, together with the run, the inference
mode and the machine (Mac and cloud results stay distinguishable). Moved from the /tmp helper scripts (paired_games4.py,
tf_eval.py) unchanged, so the cloud server runs exactly the evaluation behind every Exp-16 number so far.

How paired it really is: the first layout of each eval seed is the same for every model, but a scratch draws the ball-in-hand
position from the env's RNG, so after the first scratch one model has and the other has not, the rest of that seed's layouts
differ. Measured on the seed-0 teacher-free runs: per-game shots of two models correlate about 0.05, so the paired SE is about
the unpaired SE. The SEs are still right (computed from the per-game differences); the pairing just buys almost nothing.

Usage:
    python -m exp16_wm.paired_eval games --run "logs/experiments_teacherfree_T5/*_s0_*" --infer all_pairs --out logs/eval_paired/T5_s0.npz
    python -m exp16_wm.paired_eval compare logs/eval_paired/T5_s0.npz logs/eval_paired/T5r1_s0.npz logs/eval_paired/T5r2_s0.npz
    python -m exp16_wm.paired_eval clean --run "logs/experiments_teacherfree_T5/*_s0_*" --infer all_pairs
"""

import argparse
import glob
import json
import os
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_SEEDS = (1, 2, 3, 4)
GAMES_PER_SEED = 500


class Policy(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray: ...


def play_paired(policy: Policy, eval_seeds: tuple[int, ...] = EVAL_SEEDS, games_per_seed: int = GAMES_PER_SEED,
                env_kw: dict | None = None) -> dict[str, np.ndarray]:
    """Play games_per_seed games for every eval seed → shots (G,) per game, succ / balls (S,) per shot, game (S,) the shot's game."""
    from simulator import BilliardsEnv
    from exp16_wm.teacher_data import CLEAR_ENV
    shots, succ, balls, game = [], [], [], []
    for eval_seed in eval_seeds:
        env = BilliardsEnv(**(CLEAR_ENV if env_kw is None else env_kw))
        env.reset(seed=eval_seed)
        for _ in range(games_per_seed):
            obs, _ = env.reset()
            done, k = False, 0
            while not done:
                obs, _, term, trunc, info = env.step(policy.act(obs, deterministic=True))
                done, k = term or trunc, k + 1
                n = int(info["pocketed_this_step"])
                succ.append(n > 0); balls.append(n); game.append(len(shots))
            shots.append(k)
        env.close()
    return {"shots": np.array(shots, np.int64), "succ": np.array(succ, bool), "balls": np.array(balls, np.int64),
            "game": np.array(game, np.int64)}


def summarize(res: dict[str, np.ndarray]) -> dict[str, float]:
    """games, shots/game with its SE, shot success, balls per successful shot (NaN without a successful shot)."""
    s, ok, b = np.asarray(res["shots"], float), np.asarray(res["succ"], bool), np.asarray(res["balls"], float)
    return {"games": int(len(s)), "shots": float(s.mean()), "shots_se": float(s.std() / np.sqrt(len(s))),
            "shot_success": float(ok.mean()) if len(ok) else float("nan"),
            "balls_per_success": float(b[ok].mean()) if ok.any() else float("nan")}


def paired_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and SE of the per-game difference a − b (same eval seeds and game order; see the module note on how paired that is)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"not the same games: {a.shape} vs {b.shape}")
    d = a - b
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def _version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "missing"


def machine_info() -> dict[str, Any]:
    """Where a result was produced: OS / CPU and the versions that can change simulation or inference numbers."""
    return {"node": platform.node(), "system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "cpus": os.cpu_count(), **{p: _version(p) for p in ("pooltool-billiards", "panda3d", "numba", "numpy", "torch")}}


def find_run(pattern: str) -> str:
    """The one run dir matching pattern (anything else is an error: a silent pick would evaluate the wrong model)."""
    found = sorted(glob.glob(pattern))
    if len(found) != 1:
        raise ValueError(f"{pattern}: expected one run dir, found {len(found)}: {found}")
    return found[0]


def load_policy(run_dir: str, infer: str) -> Policy:
    """The run's model with the inference used in every Exp-16 evaluation (31 angles ±0.3°, 7 absolute speeds)."""
    from exp16_wm import eval_by_balls as E
    from exp16_wm.bc import TransformedPolicy
    m = E.load_run(run_dir)
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = infer, E.ANG31, (1.0,), E.ABS_SPEEDS
    transform = json.load(open(os.path.join(run_dir, "results.json")))["args"].get("obs_transform", "relative")
    return TransformedPolicy(m, transform)


def save(path: str, res: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, meta=json.dumps(meta), **res)


def load(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-game results and their meta ({} for the older /tmp files, which stored only shots / succ / balls)."""
    z = np.load(path)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return {k: z[k] for k in z.files if k != "meta"}, meta


def val_clean(run_dir: str, infer: str) -> tuple[float, int]:
    """Made head on the teacher's labelled val states of the run's split: share of chosen actions that pocket without a scratch."""
    import torch
    from simulator import BilliardsEnv
    from exp16_wm import diagnose_action as D, eval_by_balls as E
    from exp16_wm.bc import load_teacher_data
    from exp16_wm.outcomes import pocket_index_map
    from exp16_wm.teacher_data import CLEAR_ENV
    seed = json.load(open(os.path.join(run_dir, "results.json")))["args"].get("seed", 0)
    va = D.val_setup(load_teacher_data(sorted(glob.glob(E.DATA))), seed, split="val")
    idx = np.where(va["lab"])[0]
    obs, rel = va["obs"][idx], va["rel"][idx]
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    pm = pocket_index_map(env)
    m = E.load_run(run_dir)
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = infer, E.ANG31, (1.0,), E.ABS_SPEEDS
    with torch.no_grad():
        acts = np.concatenate([D.to_env(m, m.predict_unit(rel[lo:lo + 200])) for lo in range(0, len(rel), 200)])
    return float(D.clean_flags(env, pm, obs, acts).mean()), int(len(idx))


def _line(name: str, s: dict[str, float]) -> str:
    return (f"{name:28s} games {s['games']}  shots/game {s['shots']:.3f} ± {s['shots_se']:.3f} (SE)  "
            f"shot success {s['shot_success']:.1%}  balls per successful shot {s['balls_per_success']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games", help="play the paired games with one model and save per-game results")
    g.add_argument("--run", required=True, help="run dir glob (must match exactly one)"); g.add_argument("--infer", default="all_pairs")
    g.add_argument("--out", required=True); g.add_argument("--threads", type=int, default=2)
    g.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS)); g.add_argument("--games-per-seed", type=int, default=GAMES_PER_SEED)
    c = sub.add_parser("compare", help="summaries and per-game paired differences against the first file")
    c.add_argument("files", nargs="+")
    v = sub.add_parser("clean", help="made head val clean rate on the teacher's labelled val states")
    v.add_argument("--run", required=True); v.add_argument("--infer", default="all_pairs"); v.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    if args.cmd == "games":
        import torch
        torch.set_num_threads(args.threads)
        run = find_run(args.run)
        t0 = time.time()
        res = play_paired(load_policy(run, args.infer), tuple(args.seeds), args.games_per_seed)
        meta = {"run": run, "infer": args.infer, "eval_seeds": args.seeds, "games_per_seed": args.games_per_seed,
                "threads": args.threads, "seconds": round(time.time() - t0, 1), "machine": machine_info()}
        save(args.out, res, meta)
        print(_line(os.path.basename(args.out), summarize(res)), f"  ({meta['seconds']:.0f}s) → {args.out}", flush=True)
    elif args.cmd == "compare":
        ref, _ = load(args.files[0])
        print(_line(os.path.basename(args.files[0]), summarize(ref)))
        for f in args.files[1:]:
            res, _ = load(f)
            d, se = paired_diff(res["shots"], ref["shots"])
            print(_line(os.path.basename(f), summarize(res)), f"  − first {d:+.3f} ± {se:.3f}")
    else:
        import torch
        torch.set_num_threads(args.threads)
        run = find_run(args.run)
        rate, n = val_clean(run, args.infer)
        print(f"{os.path.basename(run)[:60]}: val clean (teacher labelled val states, {n}) {rate:.1%}", flush=True)


if __name__ == "__main__":
    main()
