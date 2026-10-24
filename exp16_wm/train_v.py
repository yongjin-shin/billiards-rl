"""
exp16_wm/train_v.py — train only the V head (and its own Q / V encoder) on ball-count labels (Exp-16, V2).

The made head, the aim and the made encoder stay exactly as they were (their weights are frozen and copied), so the policy's actions
do not change; only the position value is learned. Label: y = balls pocketed before the first miss / balls left, soft BCE on
sigmoid(V logit); V in balls = balls left × sigmoid (exp16_wm.lookahead objective "balls").
Validation: 10% of the games (per data file) — the final model is the epoch with the lowest validation loss. The output is a run dir
(results.json + bc_model.pt) that exp16_wm.eval_by_balls.load_run and the look-ahead tools read like any other run.

Usage:
    python -m exp16_wm.train_v --run "<PRm run dir glob>" --data "logs/teacher_data/v2/s30*.npz" --eval-data logs/teacher_data/v2/diag_s950.npz \
        --out logs/experiments_v2/v2_fresh
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V_MODULES = ("qv_bb", "qv_ball", "qv_pair", "head_v")


def load_v_data(paths: list[str], val_frac: float = 0.1, seed: int = 0) -> dict[str, np.ndarray]:
    """Concatenated v_data files with a per-file, per-game validation split: obs, k, left, branch, val (bool)."""
    if not paths:
        raise FileNotFoundError("no V data files")
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("obs", "k", "left", "branch", "val")}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        games = np.unique(z["game"])
        val_games = set(rng.choice(games, size=max(1, int(round(val_frac * len(games)))), replace=False).tolist()) if val_frac > 0 else set()
        for k in ("obs", "k", "left", "branch"):
            out[k].append(z[k])
        out["val"].append(np.isin(z["game"], list(val_games)))
    d = {k: np.concatenate(v) for k, v in out.items()}
    keep = d["left"] > 0
    return {k: v[keep] for k, v in d.items()}


def cap(left: np.ndarray, horizon: int = 0) -> np.ndarray:
    """The most balls V can count: the balls left, or at most `horizon` of them (V3: the next H shots only; 0 = no limit)."""
    left = np.asarray(left)
    return np.minimum(left, horizon) if horizon > 0 else left


def targets(k: np.ndarray, left: np.ndarray, horizon: int = 0) -> np.ndarray:
    """Fraction of the countable balls (cap) that go in before the first miss, in [0, 1]."""
    c = np.maximum(1, cap(left, horizon)).astype(np.float32)
    return np.clip(np.minimum(np.asarray(k, np.float32), c) / c, 0.0, 1.0)


def v_logits(model, rel: torch.Tensor) -> torch.Tensor:
    """(B,) V logits from the Q / V encoder only (the made encoder is not run)."""
    _, e, valid, _ = model._encode_full(rel, qv=True)
    return model.v_logit({"e_qv": e, "valid": valid})


@torch.no_grad()
def predict_balls(model, obs: np.ndarray, left: np.ndarray, horizon: int = 0, chunk: int = 2048) -> np.ndarray:
    from exp16_wm.obs_transform import relative_obs
    rel = torch.as_tensor(relative_obs(obs))
    z = torch.cat([v_logits(model, rel[i:i + chunk]) for i in range(0, len(rel), chunk)])
    return torch.sigmoid(z).numpy() * cap(left, horizon)


def metrics(pred_balls: np.ndarray, k: np.ndarray, left: np.ndarray, horizon: int = 0) -> dict[str, float]:
    """With k capped at the horizon: AUC for "every countable ball goes in", Spearman with the balls pocketed, the same Spearman within
    each number of balls left (weighted mean — telling apart positions with as many balls left: what look-ahead needs; the
    balls-left baseline is 0 there), mean absolute error (balls); baseline = −balls left."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    c = cap(left, horizon)
    k = np.minimum(k, c)
    within, wsum = 0.0, 0
    for n in np.unique(left):
        g = left == n
        if g.sum() > 10 and np.std(k[g]) > 0 and np.std(pred_balls[g]) > 0:
            within += spearmanr(pred_balls[g], k[g]).correlation * g.sum(); wsum += g.sum()
    clear = k >= c
    two = 0 < clear.mean() < 1
    return {"auc_clear": float(roc_auc_score(clear, pred_balls / np.maximum(1, c))) if two else float("nan"),
            "auc_clear_left_baseline": float(roc_auc_score(clear, -left)) if two else float("nan"),
            "spearman_balls": float(spearmanr(pred_balls, k).correlation),
            "spearman_left_baseline": float(spearmanr(left, k).correlation),
            "spearman_within_left": float(within / wsum) if wsum else float("nan"),
            "mae_balls": float(np.mean(np.abs(pred_balls - k)))}


def main() -> None:
    from exp16_wm import eval_by_balls as E
    from exp16_wm.code_version import code_version
    from exp16_wm.obs_transform import relative_obs
    from exp16_wm.paired_eval import find_run
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--data", required=True); p.add_argument("--eval-data", default="")
    p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=60); p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=0); p.add_argument("--threads", type=int, default=8)
    p.add_argument("--init", default="made", choices=["made", "keep"],
                   help="made: the Q / V encoder starts as a copy of the made encoder and the V head fresh; keep: PRm's own Q / V weights")
    p.add_argument("--no-branch", action="store_true", help="game positions only (ablation)")
    p.add_argument("--horizon", type=int, default=0, help="V3: count only the next H balls (0 = until the first miss, V2)")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    run = find_run(args.run)
    model = E.load_run(run)
    if not getattr(model, "qv_encoder", False):
        raise ValueError(f"{run} has no Q / V encoder (trained without --qv-encoder)")
    if args.init == "made":
        model.copy_encoder_to_qv()
        for mod in model.head_v.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    for name, prm in model.named_parameters():
        prm.requires_grad = name.split(".")[0] in V_MODULES
    params = [q for q in model.parameters() if q.requires_grad]
    d = load_v_data(sorted(glob.glob(args.data)), seed=args.seed)
    if args.no_branch:
        d = {k: v[~d["branch"]] for k, v in d.items()}
    rel, y = torch.as_tensor(relative_obs(d["obs"])), torch.as_tensor(targets(d["k"], d["left"], args.horizon))
    tr, va = np.nonzero(~d["val"])[0], np.nonzero(d["val"])[0]
    print(f"{len(tr)} train / {len(va)} val positions ({d['branch'].mean():.0%} branch), mean target {y.mean():.3f}", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best, best_state, hist, t0 = float("inf"), None, [], time.time()
    for ep in range(args.epochs):
        model.train()
        for i in range(0, len(tr), args.batch):
            b = torch.as_tensor(tr[rng.integers(0, len(tr), args.batch)])
            loss = F.binary_cross_entropy_with_logits(v_logits(model, rel[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(F.binary_cross_entropy_with_logits(v_logits(model, rel[va[i:i + 4096]]), y[va[i:i + 4096]], reduction="sum").item()
                     for i in range(0, len(va), 4096)) / max(1, len(va))
        hist.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep}: val loss {vl:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state); model.eval()
    H = args.horizon
    res = {"val": metrics(predict_balls(model, d["obs"][va], d["left"][va], H), d["k"][va], d["left"][va], H)}
    if args.eval_data:
        z = np.load(args.eval_data, allow_pickle=True)
        ok = z["left"] > 0
        res["eval"] = metrics(predict_balls(model, z["obs"][ok], z["left"][ok], H), z["k"][ok], z["left"][ok], H)
    os.makedirs(args.out, exist_ok=True)
    a = json.load(open(os.path.join(run, "results.json")))
    a["args"].update(v_weight=1.0, qv_encoder=True, v_horizon=args.horizon)
    a["v2"] = {**vars(args), "from_run": run, "best_val_loss": best, "val_history": hist, "metrics": res, "git_commit": code_version()}
    json.dump(a, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    torch.save({"gnn": model.state_dict()}, os.path.join(args.out, "bc_model.pt"))
    for k, m in res.items():
        print(f"{k}: " + "  ".join(f"{n} {v:.3f}" for n, v in m.items()), flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
