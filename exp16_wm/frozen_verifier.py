"""
exp16_wm/frozen_verifier.py — does a made head rank candidates better when the frozen pair features carry geometry? (Exp-16)

A made head P(pair ij is made | features of ij, action a, a − proposal) is trained from scratch on simulator outcomes
(logs/vdistill/s{seed}.npz) on top of FROZEN pair embeddings. The only thing that changes between variants is which
embeddings it reads: the control encoder's, an encoder trained on the geometry look-ahead alone (--aux-only), or both.

  table  simulate all 217 candidates (31 angles × 7 absolute speeds around the control proposal) of the chosen pair of every
         labelled val state once → ok[state, candidate]; every head is then scored offline against this table
  fit    train the heads (variants × head seeds) and report success@k / within-state AUC from the table

Usage:
    python -m exp16_wm.frozen_verifier table --run "logs/experiments/*_L4_m2w1_*_s{seed}_*2026-10-11@2312" --seed 0 --out logs/frozen_verifier/table_s0.npz
    python -m exp16_wm.frozen_verifier fit   --run "<control>" --aux-run "logs/experiments_auxonly/*_s{seed}_*" --seed 0 \
        --table logs/frozen_verifier/table_s0.npz --data logs/vdistill/s0.npz --out logs/frozen_verifier/fit_s0.json
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import PairGNNPolicy, _NEG, _mlp, action_diff
from exp16_wm.verifier_distill import KS, NEVER, first_success_rank, holdout_states, state_auc, success_at_k

N_CAND = 217


# ──────────────────────────────────────────────
# Features and head (no simulator)
# ──────────────────────────────────────────────

@torch.no_grad()
def pair_features(anchor: PairGNNPolicy, feature_models: list[PairGNNPolicy], rel: torch.Tensor, chunk: int = 500) -> dict[str, torch.Tensor]:
    """
    Frozen inputs of the verifier for every state: e (S, 18, ΣF) = the pair embeddings of `feature_models` concatenated,
    means (S, 18, 2) = the anchor's proposals, valid (S, 18) = the anchor's live-pair mask.
    """
    parts = []
    for lo in range(0, len(rel), chunk):
        x = rel[lo:lo + chunk]
        outs: dict[int, dict] = {}
        for m in [anchor, *feature_models]:
            if id(m) not in outs:
                outs[id(m)] = m.forward_full(x)
        parts.append({"e": torch.cat([outs[id(m)]["e"] for m in feature_models], dim=-1),
                      "means": outs[id(anchor)]["means"], "valid": outs[id(anchor)]["valid"]})
    return {k: torch.cat([p[k] for p in parts]) for k in ("e", "means", "valid")}


class FrozenVerifier(nn.Module):
    """MLP [features, a, (a − proposal) × scale] → logit that action a pockets pair ij cleanly (same input layout as head_made2)."""

    def __init__(self, feat_dim: int, scale: torch.Tensor, hidden: int = 128):
        super().__init__()
        self.head = _mlp(feat_dim + 4, hidden, 1)
        self.register_buffer("scale", scale.clone().float())

    def logits(self, rows: dict[str, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        """rows: e (B, 18, F), means (B, 18, 2), valid (B, 18); a (B, 2) one action for all pairs or (B, 18, 2) → (B, 18), invalid masked."""
        e, means = rows["e"], rows["means"]
        a = a[:, None, :].expand_as(means) if a.dim() == 2 else a
        z = self.head(torch.cat([e, a, action_diff(a, means) * self.scale], dim=-1)).squeeze(-1)
        return z.masked_fill(~rows["valid"], _NEG)

    def candidate_scores(self, rows: dict[str, torch.Tensor], k: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """rows as above, k (B,) the chosen pair, cand (B, C, 2) candidate actions → (B, C) logits of pair k."""
        b = torch.arange(len(k))
        e_k, mean_k = rows["e"][b, k], rows["means"][b, k]
        C = cand.shape[1]
        feats = torch.cat([e_k[:, None].expand(-1, C, -1), cand, action_diff(cand, mean_k[:, None].expand(-1, C, -1)) * self.scale], dim=-1)
        return self.head(feats).squeeze(-1)


def fit_verifier(rows: dict[str, torch.Tensor], sid: np.ndarray, a: np.ndarray, made: np.ndarray, scratch: np.ndarray, scale: torch.Tensor,
                 seed: int = 0, epochs: int = 40, lr: float = 1e-3, batch: int = 512, holdout: float = 0.1, hold_seed: int = 1234) -> tuple[FrozenVerifier, dict]:
    """
    Train a FrozenVerifier from scratch on simulated candidates: sid[i] = index into rows of the state of simulation i, a[i] its action
    (unit), made[i] (18,) bool pockets, scratch[i] bool. BCE over the live pairs with target made & ~scratch. Held-out states (same mask for
    every `seed`: hold_seed) are never trained on; the epoch with the lowest held-out BCE is restored.
    Returns (verifier, {"train": [...], "holdout": [...], "best_epoch": int}).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = FrozenVerifier(rows["e"].shape[-1], scale)
    hold = holdout_states(len(rows["e"]), holdout, np.random.default_rng(hold_seed))[sid]
    tr_i, ho_i = np.flatnonzero(~hold), np.flatnonzero(hold)
    sid_t, a_t = torch.as_tensor(sid), torch.as_tensor(a).float()
    made_t, scr_t = torch.as_tensor(made), torch.as_tensor(scratch)
    opt = torch.optim.Adam(v.parameters(), lr=lr)

    def bce(idx: np.ndarray) -> torch.Tensor:
        r = {k: x[sid_t[idx]] for k, x in rows.items()}
        target = (made_t[idx] & ~scr_t[idx][:, None]).float()
        return F.binary_cross_entropy_with_logits(v.logits(r, a_t[idx])[r["valid"]], target[r["valid"]])

    hist_tr, hist_ho, best, best_ep, best_state = [], [], float("inf"), -1, None
    for ep in range(epochs):
        perm = rng.permutation(tr_i)
        tot = 0.0
        v.train()
        for lo in range(0, len(perm), batch):
            b = perm[lo:lo + batch]
            loss = bce(b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        v.eval()
        with torch.no_grad():
            h = float(bce(ho_i)) if len(ho_i) else float("nan")
        hist_tr.append(tot / max(len(perm), 1)); hist_ho.append(h)
        if h < best:
            best, best_ep, best_state = h, ep, {k: x.clone() for k, x in v.state_dict().items()}
    if best_state is not None:
        v.load_state_dict(best_state)
    v.eval()
    return v, {"train": hist_tr, "holdout": hist_ho, "best_epoch": best_ep}


def rank_metrics(scores: np.ndarray, ok: np.ndarray, ks: tuple = KS) -> dict:
    """
    scores, ok (S, C): a head's score of every candidate and whether the simulator says it succeeds. success@k = share of states whose best-scored
    successful candidate is within the head's top k (ties keep the table order); state_auc = mean within-state AUC (states with both outcomes).
    """
    scores, ok = np.asarray(scores, float), np.asarray(ok, bool)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.array([first_success_rank(ok[i][order[i]]) for i in range(len(ok))])
    return {"success_at_k": success_at_k(ranks, ks), "state_auc": state_auc(scores.reshape(-1), ok.reshape(-1), np.repeat(np.arange(len(ok)), ok.shape[1])),
            "grid_has_success": float(ok.any(1).mean())}


# ──────────────────────────────────────────────
# Simulator side
# ──────────────────────────────────────────────

_ENV = None


def _init_worker() -> None:
    global _ENV
    from simulator import BilliardsEnv
    from exp16_wm.outcomes import pocket_index_map
    from exp16_wm.teacher_data import CLEAR_ENV
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    _ENV = (env, pocket_index_map(env))


def _simulate_state(task: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(obs (23,), env actions (C, 2)) → (C,) bool: some ball pocketed and no scratch."""
    from exp16_wm.outcomes import made_slots
    obs, acts = task
    env, pm = _ENV
    out = np.zeros(len(acts), bool)
    for c, a in enumerate(acts):
        made, scr = made_slots(env, obs, np.asarray(a, np.float32), pm)
        out[c] = bool(made.any() and not scr)
    return out


def table(args) -> None:
    from exp16_wm import diagnose_action as D
    from exp16_wm import eval_by_balls as E
    from exp16_wm.bc import load_teacher_data
    raw = load_teacher_data(sorted(glob.glob(E.DATA)))
    found = glob.glob(args.run.format(seed=args.seed))
    if len(found) != 1:
        raise SystemExit(f"expected one run dir, found {found}")
    m = E.load_run(found[0])
    D.configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
    v = D.val_setup(raw, args.seed, split="val")
    idx = np.where(v["lab"])[0]
    if args.max_states:
        idx = idx[:args.max_states]
    K, CAND, ACTS = [], [], []
    with torch.no_grad():
        for lo in range(0, len(idx), 300):
            ii = idx[lo:lo + 300]
            o = m.forward_full(v["rel"][ii])
            score = (F.log_softmax(o["logits"], 1) + F.logsigmoid(m.made2_logits(o, o["means"]))).masked_fill(~o["valid"], -1e30)
            k = score.argmax(1)                                              # the pair verify would pick
            cand = m._unit_offsets(o["means"][torch.arange(len(ii)), k])    # (B, 217, 2) unit actions around its proposal
            K.append(k); CAND.append(cand)
            ACTS += [D.to_env(m, cand[r]) for r in range(len(ii))]
    K, CAND = torch.cat(K).numpy(), torch.cat(CAND).numpy().astype(np.float32)
    tasks = [(v["obs"][j], ACTS[r].astype(np.float32)) for r, j in enumerate(idx)]
    t0, ok = time.time(), []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap(_simulate_state, tasks, chunksize=8)):
            ok.append(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} states, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, state=idx, k=K, cand=CAND, ok=np.array(ok), seed=args.seed)
    ok = np.array(ok)
    print(f"saved {args.out}: {len(idx)} states × {ok.shape[1]} candidates, grid has a success in {ok.any(1).mean():.1%}, "
          f"candidate success rate {ok.mean():.1%} ({time.time() - t0:.0f}s)", flush=True)


# ──────────────────────────────────────────────
# Fit and evaluate
# ──────────────────────────────────────────────

VARIANTS = {"A": ("control",), "B": ("aux",), "C": ("control", "aux")}


def fit(args) -> None:
    from exp16_wm import diagnose_action as D
    from exp16_wm import eval_by_balls as E
    from exp16_wm.bc import load_teacher_data
    from exp16_wm.sim_identity import from_bitmask
    raw = load_teacher_data(sorted(glob.glob(E.DATA)))
    models = {}
    for name, pat in (("control", args.run), ("aux", args.aux_run)):
        found = glob.glob(pat.format(seed=args.seed))
        if len(found) != 1:
            raise SystemExit(f"{name}: expected one run dir, found {found}")
        models[name] = E.load_run(found[0])
    ctrl = models["control"]
    d, tab = np.load(args.data), np.load(args.table)
    tr, va = D.val_setup(raw, args.seed, split="train"), D.val_setup(raw, args.seed, split="val")
    uniq, sid = np.unique(d["state"], return_inverse=True)
    made = from_bitmask(d["made"]).astype(bool)
    k, cand, ok = torch.as_tensor(tab["k"]).long(), torch.as_tensor(tab["cand"]), tab["ok"]
    out: dict = {"seed": args.seed, "n_val_states": int(len(ok)), "n_sims": int(len(sid)), "variants": {}}
    # reference: the control's own jointly trained head on the same table
    with torch.no_grad():
        o = ctrl.forward_full(va["rel"][tab["state"]])
        ref = D.made_logit(ctrl, o, k, cand).numpy()
    out["original"] = rank_metrics(ref, ok)
    print(f"[original] " + "  ".join(f"@{kk} {100 * x:.1f}%" for kk, x in out["original"]["success_at_k"].items()) + f"  AUC {out['original']['state_auc']:.3f}"
          f"  (a success exists in the grid for {out['original']['grid_has_success']:.1%})", flush=True)
    for name in args.variants:
        feat_models = [models[x] for x in VARIANTS[name]]
        rows_tr = pair_features(ctrl, feat_models, tr["rel"][uniq])
        rows_va = pair_features(ctrl, feat_models, va["rel"][tab["state"]])
        res = []
        for hs in args.head_seeds:
            ver, info = fit_verifier(rows_tr, sid, d["a"], made, d["scratch"].astype(bool), ctrl.made2_scale, seed=hs, epochs=args.epochs, lr=args.lr)
            with torch.no_grad():
                s = ver.candidate_scores(rows_va, k, cand).numpy()
            met = rank_metrics(s, ok)
            res.append({"head_seed": hs, "best_epoch": info["best_epoch"], "holdout_bce": min(info["holdout"]), **met})
            print(f"[{name} seed {hs}] holdout BCE {min(info['holdout']):.4f} @{info['best_epoch']}  " +
                  "  ".join(f"@{kk} {100 * x:.1f}%" for kk, x in met["success_at_k"].items()) + f"  AUC {met['state_auc']:.3f}", flush=True)
        out["variants"][name] = res
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    print("\nsuccess@k (mean of head seeds; min–max)")
    print(f"{'':10s}" + "".join(f"{'@' + str(kk):>14s}" for kk in (1, 2, 3, 5, 10, 20)) + f"{'AUC':>14s}")
    row = lambda lab, vals: print(f"{lab:10s}" + "".join(f"{100 * np.mean(c):6.1f} ({100 * min(c):.1f}-{100 * max(c):.1f})".rjust(14) if len(c) > 1 else f"{100 * c[0]:6.1f}".rjust(14) for c in vals[:-1])
                                  + (f"{np.mean(vals[-1]):.3f}".rjust(14)))
    row("original", [[out["original"]["success_at_k"][kk]] for kk in (1, 2, 3, 5, 10, 20)] + [[out["original"]["state_auc"]]])
    for name, res in out["variants"].items():
        row(name, [[r["success_at_k"][kk] for r in res] for kk in (1, 2, 3, 5, 10, 20)] + [[r["state_auc"] for r in res]])


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("table", "fit"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True, help="control run dir glob with {seed}")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out", default=None)
        if name == "table":
            s.add_argument("--workers", type=int, default=4); s.add_argument("--max-states", type=int, default=0)
        else:
            s.add_argument("--aux-run", required=True, help="aux-only run dir glob with {seed}")
            s.add_argument("--table", required=True); s.add_argument("--data", required=True)
            s.add_argument("--variants", nargs="+", default=["A", "B", "C"], choices=list(VARIANTS))
            s.add_argument("--head-seeds", type=int, nargs="+", default=[0, 1, 2])
            s.add_argument("--epochs", type=int, default=40); s.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    torch.set_num_threads(4)
    {"table": table, "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    main()
