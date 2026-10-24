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
    out = {k: [] for k in ("obs", "k", "left", "branch", "val", "gid")}
    for f, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        games = np.unique(z["game"])
        val_games = set(rng.choice(games, size=max(1, int(round(val_frac * len(games)))), replace=False).tolist()) if val_frac > 0 else set()
        for k in ("obs", "k", "left", "branch"):
            out[k].append(z[k])
        out["val"].append(np.isin(z["game"], list(val_games)))
        out["gid"].append(f * 10_000_000 + z["game"].astype(np.int64))      # a game id unique across files
    d = {k: np.concatenate(v) for k, v in out.items()}
    keep = d["left"] > 0
    return {k: v[keep] for k, v in d.items()}


def load_smooth_k(paths: list[str], smooth_dir: str, d: dict[str, np.ndarray], eps_mm: float = 0.0) -> np.ndarray:
    """(N,) the mean k over the ε-moved copies of each position (exp16_wm.v_perturb output `<smooth_dir>/<data file name>`), aligned with the
    rows load_v_data returned for `paths`. eps_mm picks the radius when a file holds several (0 = the file must hold exactly one). A
    position none of whose copies could be placed keeps its exact k. Raises when a file's k or balls left differ from d's (the labels
    belong to other data) or the radius is not in the file."""
    ks, lefts, means = [], [], []
    for p in paths:
        z = np.load(os.path.join(smooth_dir, os.path.basename(p)))
        eps = np.atleast_1d(z["eps_mm"])
        hits = np.arange(len(eps)) if eps_mm == 0 else np.nonzero(np.isclose(eps, eps_mm))[0]
        if len(hits) != 1:
            raise ValueError(f"{p}: radii in the label file are {eps.tolist()} mm; pick one with eps_mm (asked {eps_mm})")
        kp = z["k_pert"][:, int(hits[0]), :]
        placed = kp >= 0
        n = placed.sum(axis=1)
        means.append(np.where(n > 0, np.where(placed, kp, 0).sum(axis=1) / np.maximum(n, 1), np.nan))
        ks.append(z["k"]); lefts.append(z["left"])
    k, left, mean = np.concatenate(ks), np.concatenate(lefts), np.concatenate(means)
    if len(k) != len(d["k"]) or not np.array_equal(k, d["k"]) or not np.array_equal(left, d["left"]):
        raise ValueError(f"the smoothed labels in {smooth_dir} do not match the V data rows (k / left differ or the counts differ)")
    return np.where(np.isfinite(mean), mean, d["k"].astype(float))


def cap(left: np.ndarray, horizon: int = 0) -> np.ndarray:
    """The most balls V can count: the balls left, or at most `horizon` of them (V3: the next H shots only; 0 = no limit)."""
    left = np.asarray(left)
    return np.minimum(left, horizon) if horizon > 0 else left


def targets(k: np.ndarray, left: np.ndarray, horizon: int = 0, mode: str = "frac") -> np.ndarray:
    """mode "frac": the fraction of the countable balls (cap) that go in before the first miss, in [0, 1];
    mode "clear" (V6-D): 1 if every countable ball goes in (k ≥ cap), else 0."""
    c = np.maximum(1, cap(left, horizon)).astype(np.float32)
    frac = np.clip(np.minimum(np.asarray(k, np.float32), c) / c, 0.0, 1.0)
    if mode == "frac":
        return frac
    if mode == "clear":
        return (frac >= 1.0).astype(np.float32)
    raise ValueError(f"mode must be frac or clear, got {mode!r}")


def v_logits(model, rel: torch.Tensor) -> torch.Tensor:
    """(B,) V logits from the Q / V encoder only (the made encoder is not run)."""
    _, e, valid, _ = model._encode_full(rel, qv=True)
    return model.v_logit({"e_qv": e, "valid": valid})


def v_hazard_logits(model, rel: torch.Tensor) -> torch.Tensor:
    """(B, S) hazard logits of a model with a hazard V head (V6-C): shot j goes in given that the shots before it did."""
    _, e, valid, _ = model._encode_full(rel, qv=True)
    return model.v_hazard_logits({"e_qv": e, "valid": valid})


def hazard_targets(k: np.ndarray, left: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """(reached, success), each (N, steps) float32: shot j (1-based) is played if j ≤ left and k ≥ j − 1, and succeeded if k ≥ j."""
    k, left = np.asarray(k), np.asarray(left)
    if left.max() > steps:
        raise ValueError(f"v_steps {steps} is smaller than the most balls left, {int(left.max())}")
    j = np.arange(1, steps + 1)[None, :]
    reached = (j <= left[:, None]) & (k[:, None] >= j - 1)
    return reached.astype(np.float32), (reached & (k[:, None] >= j)).astype(np.float32)


def hazard_nll(logits: torch.Tensor, reached: torch.Tensor, success: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Negative log-likelihood of each position's shot sequence: the sum over the shots that were played of BCE(logit, success).
    reduction: mean over positions, or their sum."""
    per = (F.binary_cross_entropy_with_logits(logits, success, reduction="none") * reached).sum(dim=1)
    return per.mean() if reduction == "mean" else per.sum()


def hazard_value(logits: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    """(B,) expected balls = Σ_{j ≤ left} Π_{i ≤ j} sigmoid(logit_i) (the chance of getting at least j balls, added up)."""
    surv = torch.cumprod(torch.sigmoid(logits), dim=1)
    j = torch.arange(1, logits.shape[1] + 1, device=logits.device)[None, :]
    return (surv * (j <= left[:, None]).to(surv.dtype)).sum(dim=1)


@torch.no_grad()
def predict_balls(model, obs: np.ndarray, left: np.ndarray, horizon: int = 0, chunk: int = 2048) -> np.ndarray:
    from exp16_wm.obs_transform import relative_obs
    rel = torch.as_tensor(relative_obs(obs))
    if getattr(model, "v_steps", 0):
        if horizon:
            raise ValueError("a hazard V head does not take a horizon")
        z = torch.cat([v_hazard_logits(model, rel[i:i + chunk]) for i in range(0, len(rel), chunk)])
        return hazard_value(z, torch.as_tensor(np.asarray(left))).numpy()
    z = torch.cat([v_logits(model, rel[i:i + chunk]) for i in range(0, len(rel), chunk)])
    return torch.sigmoid(z).numpy() * cap(left, horizon)


@torch.no_grad()
def predict_first_step(model, obs: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """(N,) P(the first shot goes in) of a model with a hazard V head."""
    from exp16_wm.obs_transform import relative_obs
    rel = torch.as_tensor(relative_obs(obs))
    return torch.sigmoid(torch.cat([v_hazard_logits(model, rel[i:i + chunk])[:, 0] for i in range(0, len(rel), chunk)])).numpy()


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


def subsample_games(gid: np.ndarray, rows: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """The rows of a random frac of the games among rows (whole games; frac 1 keeps all)."""
    if frac >= 1.0:
        return rows
    games = np.unique(gid[rows])
    keep = np.random.default_rng(seed).choice(games, size=max(1, int(round(frac * len(games)))), replace=False)
    return rows[np.isin(gid[rows], keep)]


def widen_qv(model, run_args: dict, qv_hidden: int, qv_fourier: int | None = None, v_steps: int = 0):
    """A copy of model whose Q / V encoder and V head have width qv_hidden, qv_fourier Fourier frequencies (None = the run's) and, with
    v_steps > 0, a hazard head over that many shots, freshly initialised; every other weight is copied. The Q head is dropped (it reads
    made2 features of the made width)."""
    from exp16_wm import eval_by_balls as E
    a = {**run_args, "qv_encoder": True, "qv_hidden": qv_hidden, "q_weight": 0, "v_weight": 1.0}
    if qv_fourier is not None:
        a["qv_fourier"] = qv_fourier
    if v_steps:
        a["v_steps"] = v_steps
    new = E.policy_from_args(a)
    keep = {k: v for k, v in model.state_dict().items() if not k.startswith(("qv_", "head_v", "head_q"))}
    res = new.load_state_dict(keep, strict=False)
    if res.unexpected_keys or any(not k.startswith(("qv_", "head_v")) for k in res.missing_keys):
        raise RuntimeError(f"widen_qv: unexpected {res.unexpected_keys}, missing {res.missing_keys}")
    new.qv_detach = model.qv_detach
    return new, a


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
    p.add_argument("--data-frac", type=float, default=1.0, help="V5: train on this fraction of the training games (val unchanged)")
    p.add_argument("--qv-hidden", type=int, default=0, help="V5: width of a fresh Q / V encoder and V head (0 = keep the run's)")
    p.add_argument("--smooth-labels", default="",
                   help="V6-E: a folder of v_perturb outputs (one per data file name); the label is the mean k of the ε-moved copies instead of the exact k")
    p.add_argument("--smooth-eps-mm", type=float, default=0.0, help="V6-E: which radius of the label files to use (0 = they hold exactly one)")
    p.add_argument("--target", default="frac", choices=["frac", "clear"],
                   help="V6-D: the label — frac = balls in a row / balls left (0 ~ 1), clear = 1 if every ball goes in else 0")
    p.add_argument("--v-steps", type=int, default=0,
                   help="V6-C: a hazard V head over this many shots (fresh Q / V encoder and head; 0 = the single value head)")
    p.add_argument("--qv-fourier", type=int, default=-1,
                   help="V6: Fourier frequencies of a fresh Q / V encoder's input coordinates (−1 = keep the run's; 0 = a fresh one without)")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    run = find_run(args.run)
    model = E.load_run(run)
    if not getattr(model, "qv_encoder", False):
        raise ValueError(f"{run} has no Q / V encoder (trained without --qv-encoder)")
    run_args = json.load(open(os.path.join(run, "results.json")))["args"]
    new_width, new_fourier = bool(args.qv_hidden and args.qv_hidden != model.qv_hidden), args.qv_fourier >= 0
    if args.v_steps and args.horizon:
        raise ValueError("--v-steps (hazard head) does not take --horizon")
    if new_width or new_fourier or args.v_steps:
        model, run_args = widen_qv(model, run_args, args.qv_hidden or model.qv_hidden, args.qv_fourier if new_fourier else None, args.v_steps)
        if args.init == "made":
            raise ValueError("--init made copies the made encoder, which has another width / input / head than --qv-hidden / --qv-fourier / --v-steps")
    elif args.init == "made":
        model.copy_encoder_to_qv()
        for mod in model.head_v.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    for name, prm in model.named_parameters():
        prm.requires_grad = name.split(".")[0] in V_MODULES
    params = [q for q in model.parameters() if q.requires_grad]
    data_paths = sorted(glob.glob(args.data))
    d = load_v_data(data_paths, seed=args.seed)
    if args.smooth_labels:
        if args.v_steps or args.target != "frac":
            raise ValueError("--smooth-labels works with the single value head and --target frac only")
        d["k_label"] = load_smooth_k(data_paths, args.smooth_labels, d, args.smooth_eps_mm)
    else:
        d["k_label"] = d["k"]
    if args.no_branch:
        d = {k: v[~d["branch"]] for k, v in d.items()}
    rel, y = torch.as_tensor(relative_obs(d["obs"])), torch.as_tensor(targets(d["k_label"], d["left"], args.horizon, args.target))
    S = args.v_steps
    if S:
        reached, success = (torch.as_tensor(x) for x in hazard_targets(d["k"], d["left"], S))

    def batch_loss(rows: torch.Tensor, reduction: str) -> torch.Tensor:
        if S:
            return hazard_nll(v_hazard_logits(model, rel[rows]), reached[rows], success[rows], reduction)
        return F.binary_cross_entropy_with_logits(v_logits(model, rel[rows]), y[rows], reduction=reduction)
    tr, va = np.nonzero(~d["val"])[0], np.nonzero(d["val"])[0]
    n_full = len(tr)
    tr = subsample_games(d["gid"], tr, args.data_frac, args.seed)
    print(f"{len(tr)} train / {len(va)} val positions ({d['branch'].mean():.0%} branch), mean target {y.mean():.3f}", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best, best_state, hist, tr_hist, t0 = float("inf"), None, [], [], time.time()
    steps = -(-n_full // args.batch)                     # steps per epoch of the full data, also with --data-frac (same compute)
    for ep in range(args.epochs):
        model.train()
        tl, tn = 0.0, 0
        for i in range(steps):
            b = torch.as_tensor(tr[rng.integers(0, len(tr), args.batch)])
            loss = batch_loss(b, "mean")
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); tn += 1
        tr_hist.append(tl / max(1, tn))
        model.eval()
        with torch.no_grad():
            vl = sum(batch_loss(torch.as_tensor(va[i:i + 4096]), "sum").item() for i in range(0, len(va), 4096)) / max(1, len(va))
        hist.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep}: train loss {tr_hist[-1]:.4f}  val loss {vl:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state); model.eval()
    H = args.horizon
    res = {"val": metrics(predict_balls(model, d["obs"][va], d["left"][va], H), d["k"][va], d["left"][va], H)}
    if args.eval_data:
        z = np.load(args.eval_data, allow_pickle=True)
        ok = z["left"] > 0
        res["eval"] = metrics(predict_balls(model, z["obs"][ok], z["left"][ok], H), z["k"][ok], z["left"][ok], H)
    os.makedirs(args.out, exist_ok=True)
    a = json.load(open(os.path.join(run, "results.json")))
    a["args"] = {**run_args, "v_weight": 1.0, "qv_encoder": True, "v_horizon": args.horizon}
    a["v2"] = {**vars(args), "from_run": run, "best_val_loss": best, "val_history": hist, "train_history": tr_hist,
               "n_train": int(len(tr)), "metrics": res, "git_commit": code_version()}
    json.dump(a, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    torch.save({"gnn": model.state_dict()}, os.path.join(args.out, "bc_model.pt"))
    for k, m in res.items():
        print(f"{k}: " + "  ".join(f"{n} {v:.3f}" for n, v in m.items()), flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
