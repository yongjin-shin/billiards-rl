"""
exp16_wm/verifier_distill.py — search → distill, stage 1 (Exp-16): train the made head on simulator outcomes of the
candidates it is asked to rank.

At inference the verify+refine step ranks the 217 candidates (angle ±0.3° × 7 absolute speeds) around the pair head's
proposal with the made head, but the made head was trained on coarse planner candidates. Here the simulator labels
(positives and negatives) of those very candidates are used.

  gen    for every train state, simulate 24 of the 217 candidates of the label pair (half from the made head's top,
         half uniform) and record made (all 18 pairs) and scratch
  ft     fine-tune head_made2 only (encoder, pair head and selection head frozen) with the made2 BCE
  eval   success@k (candidates in made-head order, simulator picks the first success), games, and within-state AUC

Usage:
    python -m exp16_wm.verifier_distill gen  --run "logs/experiments/*_L4_m2w1_*_s{seed}_*2026-10-11@2312" --seed 0 --out logs/vdistill/s0.npz
    python -m exp16_wm.verifier_distill ft   --run ... --seed 0 --data logs/vdistill/s0.npz --out logs/vdistill/s0_head.pt
    python -m exp16_wm.verifier_distill eval --run ... --seed 0 --head logs/vdistill/s0_head.pt
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm import diagnose_action as D
from exp16_wm import eval_by_balls as E
from exp16_wm.bc import TransformedPolicy, evaluate_policy, load_teacher_data
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import TRANSFORMS
from exp16_wm.outcomes import made_slots, pocket_index_map
from exp16_wm.sim_identity import from_bitmask, to_bitmask
from exp16_wm.teacher_data import CLEAR_ENV

KS = (1, 2, 3, 5, 10, 20, 50, 217)
NEVER = 10 ** 6


# ──────────────────────────────────────────────
# Pure helpers (tested)
# ──────────────────────────────────────────────

def sample_candidates(scores: np.ndarray, n_top: int, top_pool: int, n_uniform: int, rng: np.random.Generator) -> np.ndarray:
    """Candidate indices: n_top drawn from the top_pool best-scored ones, n_uniform drawn uniformly from all the others (no repeats)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    pool, rest = order[:top_pool], order[top_pool:]
    top = rng.choice(pool, size=min(n_top, len(pool)), replace=False)
    uni = rng.choice(rest, size=min(n_uniform, len(rest)), replace=False) if len(rest) else np.zeros(0, dtype=order.dtype)
    return np.concatenate([top, uni])


def first_success_rank(ok_in_rank_order: np.ndarray) -> int:
    """1-based rank of the first success in a boolean vector ordered best-first; NEVER if there is none."""
    idx = np.flatnonzero(np.asarray(ok_in_rank_order, dtype=bool))
    return int(idx[0]) + 1 if len(idx) else NEVER


def first_success_lazy(items, ok) -> tuple[int, int]:
    """(1-based rank of the first item with ok(item) true, NEVER if none; number of ok() calls). Stops at the first success."""
    n = 0
    for i, it in enumerate(items):
        n += 1
        if ok(it):
            return i + 1, n
    return NEVER, n


def success_at_k(ranks: np.ndarray, ks: tuple = KS) -> dict[int, float]:
    """Share of states whose first success has rank ≤ k."""
    r = np.asarray(ranks)
    return {k: float(np.mean(r <= k)) for k in ks}


def state_auc(scores: np.ndarray, labels: np.ndarray, state: np.ndarray) -> float:
    """Mean over states (that have both a success and a failure) of the AUC of the scores within the state; nan if no such state."""
    scores, labels, state = np.asarray(scores, float), np.asarray(labels, bool), np.asarray(state)
    aucs = []
    for s in np.unique(state):
        m = state == s
        pos, neg = scores[m & labels], scores[m & ~labels]
        if len(pos) and len(neg):
            aucs.append((np.mean(pos[:, None] > neg[None]) + 0.5 * np.mean(pos[:, None] == neg[None])))
    return float(np.mean(aucs)) if aucs else float("nan")


def holdout_states(n_states: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask over states 0 … n_states − 1: True = held out (at least one state each side when n_states ≥ 2)."""
    n_hold = int(np.clip(round(frac * n_states), 1, max(n_states - 1, 1))) if n_states > 1 else 0
    mask = np.zeros(n_states, bool)
    mask[rng.choice(n_states, size=n_hold, replace=False)] = True
    return mask


# ──────────────────────────────────────────────
# Training core (no simulator)
# ──────────────────────────────────────────────

@torch.no_grad()
def encode_cache(m: PairGNNPolicy, rel: torch.Tensor, chunk: int = 500) -> dict[str, torch.Tensor]:
    """The frozen features the made head reads, for every state: pair embeddings e, proposals means, valid mask."""
    parts = [m.forward_full(rel[lo:lo + chunk]) for lo in range(0, len(rel), chunk)]
    return {k: torch.cat([p[k] for p in parts]) for k in ("e", "means", "valid")}


def made2_bce(m: PairGNNPolicy, cache: dict, sid: torch.Tensor, a: torch.Tensor, made: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    """Same loss as PairGNNPolicy.made2_loss, but on cached features: BCE over the live pairs, target = made & ~scratch."""
    o = {k: v[sid] for k, v in cache.items()}
    z = m.made2_logits(o, a)
    target = (made & ~scratch[:, None]).float()
    return F.binary_cross_entropy_with_logits(z[o["valid"]], target[o["valid"]])


def finetune_head(m: PairGNNPolicy, cache: dict, sid: np.ndarray, a: np.ndarray, made: np.ndarray, scratch: np.ndarray, epochs: int = 30,
                  lr: float = 3e-4, batch: int = 512, holdout: float = 0.1, seed: int = 0) -> dict:
    """
    Train head_made2 only (everything else frozen). Rows of the held-out states are never trained on; the epoch with the lowest
    held-out BCE is restored. Returns {"train": [...], "holdout": [...], "best_epoch": int, "base_holdout": bce before training}.
    """
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.head_made2.parameters():
        p.requires_grad_(True)
    rng = np.random.default_rng(seed)
    hold = holdout_states(len(cache["e"]), holdout, rng)[sid]
    tr_i, ho_i = np.flatnonzero(~hold), np.flatnonzero(hold)
    t = lambda x: torch.as_tensor(x)
    sid_t, a_t, made_t, scr_t = t(sid), t(a).float(), t(made), t(scratch)
    opt = torch.optim.Adam(m.head_made2.parameters(), lr=lr)

    def ho_loss() -> float:
        with torch.no_grad():
            return float(made2_bce(m, cache, sid_t[ho_i], a_t[ho_i], made_t[ho_i], scr_t[ho_i])) if len(ho_i) else float("nan")
    base = ho_loss()
    best, best_state, hist_tr, hist_ho = float("inf"), None, [], []
    for ep in range(epochs):
        perm = rng.permutation(tr_i); tot = 0.0
        m.head_made2.train()
        for lo in range(0, len(perm), batch):
            b = perm[lo:lo + batch]
            loss = made2_bce(m, cache, sid_t[b], a_t[b], made_t[b], scr_t[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        m.head_made2.eval()
        h = ho_loss(); hist_tr.append(tot / max(len(perm), 1)); hist_ho.append(h)
        if h < best:
            best, best_ep, best_state = h, ep, {k: v.clone() for k, v in m.head_made2.state_dict().items()}
    if best_state is not None:
        m.head_made2.load_state_dict(best_state)
    return {"train": hist_tr, "holdout": hist_ho, "best_epoch": int(best_ep) if best_state is not None else -1, "base_holdout": base}


# ──────────────────────────────────────────────
# Simulator side
# ──────────────────────────────────────────────

def _setup(args):
    raw = load_teacher_data(sorted(glob.glob(E.DATA)))
    found = glob.glob(args.run.format(seed=args.seed))
    if len(found) != 1:
        raise SystemExit(f"expected one run dir, found {found}")
    m = E.load_run(found[0])
    D.configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    return raw, m, env, pocket_index_map(env), found[0]


def gen(args) -> None:
    raw, m, env, pm, _ = _setup(args)
    v = D.val_setup(raw, args.seed, split="train")
    idx = np.where(v["lab"])[0]
    if args.max_states:
        idx = idx[:args.max_states]
    rng = np.random.default_rng(args.seed)
    S, A, MADE, SCR = [], [], [], []
    t0 = time.time()
    for lo in range(0, len(idx), 300):
        ii = idx[lo:lo + 300]
        with torch.no_grad():
            o = m.forward_full(v["rel"][ii]); p0 = v["p0"][ii]
            cand = m._unit_offsets(o["means"][torch.arange(len(ii)), p0])
            z = D.made_logit(m, o, p0, cand).numpy()
        env_c = (cand * m.act_scale + m.act_bias).numpy()
        for r, j in enumerate(ii):
            for c in sample_candidates(z[r], args.n_top, args.top_pool, args.n_uniform, rng):
                made, scr = made_slots(env, v["obs"][j], env_c[r, c].astype(np.float32), pm)
                S.append(j); A.append(cand[r, c].numpy()); MADE.append(made); SCR.append(scr)
        print(f"  {lo + len(ii)}/{len(idx)} states, {len(S)} simulations, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, state=np.array(S), a=np.array(A, np.float32), made=to_bitmask(np.array(MADE)), scratch=np.array(SCR), seed=args.seed)
    print(f"saved {args.out}: {len(S)} simulations over {len(set(S))} states", flush=True)


def ft(args) -> None:
    raw, m, _, _, _ = _setup(args)
    d = np.load(args.data)
    v = D.val_setup(raw, args.seed, split="train")
    uniq, sid = np.unique(d["state"], return_inverse=True)
    cache = encode_cache(m, v["rel"][uniq])
    made = from_bitmask(d["made"]).astype(bool)
    res = finetune_head(m, cache, sid, d["a"], made, d["scratch"].astype(bool), epochs=args.epochs, lr=args.lr, seed=args.seed)
    torch.save(m.head_made2.state_dict(), args.out)
    json.dump({**res, "n_sims": int(len(sid)), "n_states": int(len(uniq))}, open(args.out + ".json", "w"), indent=1)
    print(f"holdout BCE {res['base_holdout']:.4f} → {min(res['holdout']):.4f} at epoch {res['best_epoch']} ({len(sid)} simulations, {len(uniq)} states)", flush=True)


def _ranked(m: PairGNNPolicy, rel1: torch.Tensor):
    """The verify-chosen pair and the candidate env actions of that pair in made-head order."""
    with torch.no_grad():
        o = m.forward_full(rel1)
        score = (F.log_softmax(o["logits"], 1) + F.logsigmoid(m.made2_logits(o, o["means"])))[0]
        k = int([int(q) for q in score.argsort(descending=True) if bool(o["valid"][0][q])][0])
        c = m._unit_offsets(o["means"][:, k]); z = D.made_logit(m, o, torch.tensor([k]), c)[0]
        return k, D.to_env(m, c[0][z.argsort(descending=True)])


def eval_model(args) -> None:
    raw, m, env, pm, run = _setup(args)
    old_head = {k: v.clone() for k, v in m.head_made2.state_dict().items()}
    v = D.val_setup(raw, args.seed, split="val")
    idx = np.where(v["lab"])[0]
    if args.max_val_states:
        idx = idx[:args.max_val_states]
    heads = {"original": old_head}
    if args.head:
        heads["distilled"] = torch.load(args.head)
    if args.only:
        heads = {args.only: heads[args.only]}
    clean = lambda obs, a: bool((lambda x: x[0].any() and not x[1])(made_slots(env, obs, np.asarray(a, np.float32), pm)))
    out = {}
    # candidate sample with labels on val states (12 from the original head's top-20, 12 uniform) for BCE / within-state AUC
    rng = np.random.default_rng(1000 + args.seed)
    sub = np.sort(rng.choice(idx, size=min(args.n_auc_states, len(idx)), replace=False))
    m.head_made2.load_state_dict(old_head)
    with torch.no_grad():
        o = m.forward_full(v["rel"][sub]); p0 = v["p0"][sub]
        cand = m._unit_offsets(o["means"][torch.arange(len(sub)), p0]); z0 = D.made_logit(m, o, p0, cand).numpy()
    env_c = (cand * m.act_scale + m.act_bias).numpy()
    pairs_idx, lab = [], []
    for r, j in enumerate(sub):
        for c in sample_candidates(z0[r], 12, 20, 12, rng):
            made, scr = made_slots(env, v["obs"][j], env_c[r, c].astype(np.float32), pm)
            pairs_idx.append((r, c)); lab.append(bool(made[int(v["p0"][j])] and not scr))
    rr, cc = np.array(pairs_idx).T; lab = np.array(lab)
    for name, sd in heads.items():
        m.head_made2.load_state_dict(sd)
        with torch.no_grad():
            o = m.forward_full(v["rel"][sub]); zz = D.made_logit(m, o, v["p0"][sub], m._unit_offsets(o["means"][torch.arange(len(sub)), v["p0"][sub]])).numpy()
        s = zz[rr, cc]
        bce = float(F.binary_cross_entropy_with_logits(torch.as_tensor(s), torch.as_tensor(lab).float()))
        out[name] = {"bce": bce, "state_auc": state_auc(s, lab, rr), "positive_rate": float(lab.mean()), "n_candidates": int(len(lab))}
        # success@k on all labelled val states (verify-chosen pair, candidates in this head's order)
        ranks = []
        for j in idx:
            _, c = _ranked(m, v["rel"][j:j + 1])
            ranks.append(first_success_lazy(c, lambda a, j=j: clean(v["obs"][j], a))[0])
        out[name]["success_at_k"] = success_at_k(np.array(ranks))
        if args.games:
            m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = "verify_refine", E.ANG31, (1.0,), E.ABS_SPEEDS
            ev = evaluate_policy(TransformedPolicy(m, "relative"), 0, args.games, CLEAR_ENV)
            out[name]["games"] = {"mean_shots": ev["mean_shots"], "shot_success_rate": ev["shot_success_rate"]}
        print(f"[{name}] BCE {out[name]['bce']:.4f}  state-AUC {out[name]['state_auc']:.3f}  " + "  ".join(f"@{k} {100*x:.1f}%" for k, x in out[name]["success_at_k"].items())
              + (f"  games {out[name]['games']['mean_shots']:.3f}" if args.games else ""), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "ft", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True, help='run dir glob with {seed}')
        s.add_argument("--seed", type=int, default=0)
        if name == "gen":
            s.add_argument("--out", required=True); s.add_argument("--n-top", type=int, default=12); s.add_argument("--top-pool", type=int, default=20)
            s.add_argument("--n-uniform", type=int, default=12); s.add_argument("--max-states", type=int, default=0)
        if name == "ft":
            s.add_argument("--data", required=True); s.add_argument("--out", required=True); s.add_argument("--epochs", type=int, default=30)
            s.add_argument("--lr", type=float, default=3e-4)
        if name == "eval":
            s.add_argument("--head", default=None); s.add_argument("--only", choices=["original", "distilled"], default=None, help="evaluate just this head"); s.add_argument("--games", type=int, default=500); s.add_argument("--n-auc-states", type=int, default=600)
            s.add_argument("--max-val-states", type=int, default=0, help="limit the val states of success@k (0 = all with a label; for smoke tests)")
            s.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(2)
    {"gen": gen, "ft": ft, "eval": eval_model}[args.cmd](args)


if __name__ == "__main__":
    main()
