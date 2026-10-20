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

    def __init__(self, feat_dim: int, scale: torch.Tensor, hidden: int = 128, aim_units: tuple[float, float] | None = None):
        """aim_units = (act_scale[0], act_bias[0]) of the policy: if given, rows also carry "aim" (B, 18) rad and the head gets one more input,
        wrap(action angle − aim) in units of 0.2° (the made2_geom feature; a diagnostic, see the noisy-aim test)."""
        super().__init__()
        self.use_aim = aim_units is not None
        self.head = _mlp(feat_dim + 4 + int(self.use_aim), hidden, 1)
        self.register_buffer("scale", scale.clone().float())
        self.register_buffer("aim_units", torch.tensor(aim_units if self.use_aim else (0.0, 0.0), dtype=torch.float32))

    def _aim_feature(self, aim: torch.Tensor, a_angle_unit: torch.Tensor) -> torch.Tensor:
        d = a_angle_unit * self.aim_units[0] + self.aim_units[1] - aim
        return (torch.rad2deg(torch.atan2(torch.sin(d), torch.cos(d))) / 0.2)[..., None]

    def logits(self, rows: dict[str, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        """rows: e (B, 18, F), means (B, 18, 2), valid (B, 18); a (B, 2) one action for all pairs or (B, 18, 2) → (B, 18), invalid masked."""
        e, means = rows["e"], rows["means"].detach()                 # the proposal is an anchor, not something to train through
        a = a[:, None, :].expand_as(means) if a.dim() == 2 else a
        feats = [e, a, action_diff(a, means) * self.scale]
        if self.use_aim:
            feats.append(self._aim_feature(rows["aim"], a[..., 0]))
        z = self.head(torch.cat(feats, dim=-1)).squeeze(-1)
        return z.masked_fill(~rows["valid"], _NEG)

    def candidate_scores(self, rows: dict[str, torch.Tensor], k: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """rows as above, k (B,) the chosen pair, cand (B, C, 2) candidate actions → (B, C) logits of pair k."""
        b = torch.arange(len(k))
        e_k, mean_k = rows["e"][b, k], rows["means"][b, k].detach()
        C = cand.shape[1]
        feats = [e_k[:, None].expand(-1, C, -1), cand, action_diff(cand, mean_k[:, None].expand(-1, C, -1)) * self.scale]
        if self.use_aim:
            feats.append(self._aim_feature(rows["aim"][b, k][:, None].expand(-1, C), cand[..., 0]))
        return self.head(torch.cat(feats, dim=-1)).squeeze(-1)


def fit_verifier(rows: dict[str, torch.Tensor], sid: np.ndarray, a: np.ndarray, made: np.ndarray, scratch: np.ndarray, scale: torch.Tensor,
                 seed: int = 0, epochs: int = 40, lr: float = 1e-3, batch: int = 512, holdout: float = 0.1, hold_seed: int = 1234,
                 aim_units: tuple[float, float] | None = None) -> tuple[FrozenVerifier, dict]:
    """
    Train a FrozenVerifier from scratch on simulated candidates: sid[i] = index into rows of the state of simulation i, a[i] its action
    (unit), made[i] (18,) bool pockets, scratch[i] bool. BCE over the live pairs with target made & ~scratch. Held-out states (same mask for
    every `seed`: hold_seed) are never trained on; the epoch with the lowest held-out BCE is restored.
    Returns (verifier, {"train": [...], "holdout": [...], "best_epoch": int}).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if ("aim" in rows) != (aim_units is not None):
        raise ValueError("rows['aim'] and aim_units go together")
    v = FrozenVerifier(rows["e"].shape[-1], scale, aim_units=aim_units)
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


class AdaptiveVerifier(nn.Module):
    """
    A verifier with its own encoder: `enc` (a PairGNNPolicy, normally a deep copy so the policy itself is untouched) gives the pair
    embeddings from the relative observation, a FrozenVerifier head reads them. The proposals and live-pair mask stay the policy's (anchor).
    """

    def __init__(self, enc: PairGNNPolicy, scale: torch.Tensor, hidden: int = 128):
        super().__init__()
        self.enc = enc
        self.head = FrozenVerifier(enc.hidden, scale, hidden)

    def rows(self, rel: torch.Tensor, anchor: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"e": self.enc.forward_full(rel)["e"], "means": anchor["means"], "valid": anchor["valid"]}

    def candidate_scores(self, rel: torch.Tensor, anchor: dict[str, torch.Tensor], k: torch.Tensor, cand: torch.Tensor, chunk: int = 300) -> torch.Tensor:
        parts = []
        for lo in range(0, len(rel), chunk):
            sl = slice(lo, lo + chunk)
            parts.append(self.head.candidate_scores(self.rows(rel[sl], {n: x[sl] for n, x in anchor.items()}), k[sl], cand[sl]))
        return torch.cat(parts)


def fit_adaptive(enc: PairGNNPolicy, rel: torch.Tensor, anchor: dict[str, torch.Tensor], sid: np.ndarray, a: np.ndarray, made: np.ndarray,
                 scratch: np.ndarray, scale: torch.Tensor, seed: int = 0, epochs: int = 15, lr_enc: float = 3e-4, lr_head: float = 1e-3,
                 batch: int = 512, holdout: float = 0.1, hold_seed: int = 1234, log=None) -> tuple[AdaptiveVerifier, dict]:
    """
    fit_verifier with the encoder trained too: rel (S, 21) the relative observations of the states, anchor the policy's means / valid for them,
    sid / a / made / scratch the simulations as in fit_verifier. `enc` is trained IN PLACE (pass a copy). Same held-out states as fit_verifier
    (same hold_seed and number of states); the epoch with the lowest held-out BCE is restored.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = AdaptiveVerifier(enc, scale)
    hold = holdout_states(len(rel), holdout, np.random.default_rng(hold_seed))[sid]
    tr_i, ho_i = np.flatnonzero(~hold), np.flatnonzero(hold)
    sid_t, a_t = torch.as_tensor(sid), torch.as_tensor(a).float()
    made_t, scr_t = torch.as_tensor(made), torch.as_tensor(scratch)
    opt = torch.optim.Adam([{"params": enc.parameters(), "lr": lr_enc}, {"params": v.head.parameters(), "lr": lr_head}])

    def bce(idx: np.ndarray) -> torch.Tensor:
        st = sid_t[idx]
        r = v.rows(rel[st], {k: x[st] for k, x in anchor.items()})
        target = (made_t[idx] & ~scr_t[idx][:, None]).float()
        return F.binary_cross_entropy_with_logits(v.head.logits(r, a_t[idx])[r["valid"]], target[r["valid"]])

    hist_tr, hist_ho, best, best_ep, best_state = [], [], float("inf"), -1, None
    for ep in range(epochs):
        t0 = time.time()
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
            h = float(sum(bce(ho_i[lo:lo + 2048]) * len(ho_i[lo:lo + 2048]) for lo in range(0, len(ho_i), 2048)) / max(len(ho_i), 1)) if len(ho_i) else float("nan")
        hist_tr.append(tot / max(len(perm), 1)); hist_ho.append(h)
        if log:
            log(f"    epoch {ep:>2}  train {hist_tr[-1]:.4f}  holdout {h:.4f}  ({time.time() - t0:.0f}s)")
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


def load_sims(paths: list[str]) -> dict[str, np.ndarray]:
    """Concatenate simulation files (gen / verifier_distill gen format): state, a (unit), made (N, 18) bool, scratch (N,) bool."""
    from exp16_wm.sim_identity import from_bitmask
    parts = [np.load(q) for q in paths]
    return {"state": np.concatenate([d["state"] for d in parts]), "a": np.concatenate([d["a"] for d in parts]).astype(np.float32),
            "made": np.concatenate([from_bitmask(d["made"]).astype(bool) for d in parts]),
            "scratch": np.concatenate([d["scratch"] for d in parts]).astype(bool)}


def group_top1(scores: np.ndarray, ok: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Top-1 success (best-scored candidate succeeds) within each boolean state group; nan for an empty group."""
    hit = np.asarray(ok, bool)[np.arange(len(ok)), np.asarray(scores).argmax(1)]
    return {g: (float(hit[m].mean()) if m.any() else float("nan")) for g, m in groups.items()}


def state_groups(ctrl: PairGNNPolicy, rel: torch.Tensor, label_pair: np.ndarray, label_angle: np.ndarray, k: np.ndarray,
                 grid_half_deg: float = 0.31) -> dict[str, np.ndarray]:
    """
    The three groups of the failure analysis: verify picked the teacher's pair and the teacher's angle is inside the candidate grid
    around the proposal ("same_near"), same pair but the teacher's angle is outside ("same_far"), or another pair ("other_pair").
    """
    with torch.no_grad():
        prop = ctrl.forward_full(rel)["means"][torch.arange(len(k)), torch.as_tensor(k)][:, 0]
    prop_ang = (prop * ctrl.act_scale[0] + ctrl.act_bias[0]).numpy()
    off = np.degrees(np.arctan2(np.sin(label_angle - prop_ang), np.cos(label_angle - prop_ang)))
    same, near = np.asarray(k) == np.asarray(label_pair), np.abs(off) <= grid_half_deg
    return {"same_near": same & near, "same_far": same & ~near, "other_pair": ~same}


# ──────────────────────────────────────────────
# Simulator side
# ──────────────────────────────────────────────

_ENV = None


_ENVS: dict[int, tuple] = {}


def _env_for(obs: np.ndarray) -> tuple:
    """(env, pocket map) for the number of balls of this raw observation (created once per worker)."""
    from simulator import BilliardsEnv
    from exp16_wm.obs_transform import n_balls_raw
    from exp16_wm.outcomes import pocket_index_map
    from exp16_wm.teacher_data import clear_env_kw
    n = n_balls_raw(len(obs))
    if n not in _ENVS:
        env = BilliardsEnv(**clear_env_kw(n)); env.reset(seed=0)
        _ENVS[n] = (env, pocket_index_map(env))
    return _ENVS[n]


def _init_worker() -> None:
    global _ENV
    from simulator import BilliardsEnv
    from exp16_wm.outcomes import pocket_index_map
    from exp16_wm.teacher_data import CLEAR_ENV
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    _ENV = (env, pocket_index_map(env))
    _ENVS[CLEAR_ENV["n_balls"]] = _ENV


def _simulate_state(task: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(obs (23,), env actions (C, 2)) → (C,) bool: some ball pocketed and no scratch."""
    from exp16_wm.outcomes import made_slots
    obs, acts = task
    env, pm = _env_for(obs)
    out = np.zeros(len(acts), bool)
    for c, a in enumerate(acts):
        made, scr = made_slots(env, obs, np.asarray(a, np.float32), pm)
        out[c] = bool(made.any() and not scr)
    return out


def _simulate_full(task: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(obs (23,), env actions (C, 2)) → (made (C, 18) bool, scratch (C,) bool)."""
    from exp16_wm.outcomes import made_slots
    obs, acts = task
    env, pm = _env_for(obs)
    made, scr = np.zeros((len(acts), len(env._ball_ids) * 6), bool), np.zeros(len(acts), bool)
    for c, a in enumerate(acts):
        mm, sc = made_slots(env, obs, np.asarray(a, np.float32), pm)
        made[c], scr[c] = mm, bool(sc)
    return made, scr


def gen(args) -> None:
    """Training candidates around the pair the policy's verify step picks (on-policy pair) or the teacher's pair, same sampling as verifier_distill gen."""
    from exp16_wm import diagnose_action as D
    from exp16_wm import eval_by_balls as E
    from exp16_wm.bc import load_teacher_data
    from exp16_wm.sim_identity import to_bitmask
    from exp16_wm.verifier_distill import sample_candidates
    raw = load_teacher_data(sorted(glob.glob(E.DATA)))
    found = glob.glob(args.run.format(seed=args.seed))
    if len(found) != 1:
        raise SystemExit(f"expected one run dir, found {found}")
    m = E.load_run(found[0])
    D.configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
    v = D.val_setup(raw, args.seed, split="train")
    idx = np.where(v["lab"])[0]
    if args.max_states:
        idx = idx[:args.max_states]
    rng = np.random.default_rng(args.seed + 500)
    S, A, K, tasks = [], [], [], []
    with torch.no_grad():
        for lo in range(0, len(idx), 300):
            ii = idx[lo:lo + 300]
            o = m.forward_full(v["rel"][ii])
            if args.pair == "verify":
                score = (F.log_softmax(o["logits"], 1) + F.logsigmoid(m.made2_logits(o, o["means"]))).masked_fill(~o["valid"], -1e30)
                k = score.argmax(1)
            else:
                k = v["p0"][ii]
            cand = m._unit_offsets(o["means"][torch.arange(len(ii)), k])
            z = D.made_logit(m, o, k, cand).numpy()
            for r, j in enumerate(ii):
                cs = sample_candidates(z[r], args.n_top, args.top_pool, args.n_uniform, rng)
                S += [j] * len(cs); A.append(cand[r, cs].numpy()); K += [int(k[r])] * len(cs)
                tasks.append((v["obs"][j], D.to_env(m, cand[r, cs])))
    t0, MADE, SCR = time.time(), [], []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for i, (mm, sc) in enumerate(pool.imap(_simulate_full, tasks, chunksize=16)):
            MADE.append(mm); SCR.append(sc)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(tasks)} states, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, state=np.array(S), a=np.concatenate(A).astype(np.float32), made=to_bitmask(np.concatenate(MADE)),
             scratch=np.concatenate(SCR), pair=np.array(K), seed=args.seed)
    print(f"saved {args.out}: {len(S)} simulations over {len(idx)} states, pair = {args.pair} "
          f"(differs from the teacher's in {np.mean(np.array(K)[::args.n_top + args.n_uniform] != v['p0'][idx].numpy()):.1%}) ({time.time() - t0:.0f}s)", flush=True)


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
    d, tab = load_sims(args.data), np.load(args.table)
    tr, va = D.val_setup(raw, args.seed, split="train"), D.val_setup(raw, args.seed, split="val")
    uniq, sid = np.unique(d["state"], return_inverse=True)
    made = d["made"]
    k, cand, ok = torch.as_tensor(tab["k"]).long(), torch.as_tensor(tab["cand"]), tab["ok"]
    groups = state_groups(ctrl, va["rel"][tab["state"]], va["p0"][tab["state"]].numpy(), np.asarray(va["label_act"])[tab["state"], 0], tab["k"])
    out: dict = {"seed": args.seed, "data": args.data, "n_val_states": int(len(ok)), "n_sims": int(len(sid)), "variants": {},
                 "group_share": {g: float(m.mean()) for g, m in groups.items()}}
    # reference: the control's own jointly trained head on the same table
    with torch.no_grad():
        o = ctrl.forward_full(va["rel"][tab["state"]])
        ref = D.made_logit(ctrl, o, k, cand).numpy()
    out["original"] = {**rank_metrics(ref, ok), "group_top1": group_top1(ref, ok, groups)}
    print(f"[original] " + "  ".join(f"@{kk} {100 * x:.1f}%" for kk, x in out["original"]["success_at_k"].items()) + f"  AUC {out['original']['state_auc']:.3f}"
          f"  (a success exists in the grid for {out['original']['grid_has_success']:.1%})", flush=True)
    aim_units = (float(ctrl.act_scale[0]), float(ctrl.act_bias[0]))
    if args.aim_noise_deg:
        with torch.no_grad():                                               # how far the policy's own proposal is from the exact aim
            o_va, r_ = ctrl.forward_full(va["rel"][tab["state"]]), torch.arange(len(k))
            d_ = o_va["means"][r_, k, 0] * aim_units[0] + aim_units[1] - ctrl._geometric_aim(va["rel"][tab["state"]])[r_, k]
            err = torch.rad2deg(torch.atan2(torch.sin(d_), torch.cos(d_))).abs().numpy()
        out["pair_head_aim_error_deg"] = {"median": float(np.median(err)), "p25": float(np.percentile(err, 25)), "p75": float(np.percentile(err, 75)), "p90": float(np.percentile(err, 90))}
        print(f"pair head |proposal − exact aim| on the val pairs: median {np.median(err):.3f}°  p25 {np.percentile(err, 25):.3f}  p75 {np.percentile(err, 75):.3f}  p90 {np.percentile(err, 90):.3f}", flush=True)
    jobs = [(n, None) for n in args.variants] + [(f"AIM{sg:g}", sg) for sg in args.aim_noise_deg]
    for name, sigma in jobs:
        feat_models = [models["control"]] if sigma is not None else [models[x] for x in VARIANTS[name]]
        rows_tr = pair_features(ctrl, feat_models, tr["rel"][uniq])
        rows_va = pair_features(ctrl, feat_models, va["rel"][tab["state"]])
        if sigma is not None:                                               # exact aim + noise fixed per (state, pair), independent between train and eval states
            for rows, rel_, gseed in ((rows_tr, tr["rel"][uniq], 11), (rows_va, va["rel"][tab["state"]], 22)):
                g = torch.Generator().manual_seed(gseed)
                rows["aim"] = ctrl._geometric_aim(rel_) + torch.randn(rows["valid"].shape, generator=g) * float(np.radians(sigma))
        res = []
        for hs in args.head_seeds:
            ver, info = fit_verifier(rows_tr, sid, d["a"], made, d["scratch"].astype(bool), ctrl.made2_scale, seed=hs, epochs=args.epochs, lr=args.lr,
                                     aim_units=aim_units if sigma is not None else None)
            with torch.no_grad():
                s = ver.candidate_scores(rows_va, k, cand).numpy()
            met = {**rank_metrics(s, ok), "group_top1": group_top1(s, ok, groups)}
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
    print("\ntop-1 by state group (share): " + "  ".join(f"{g} {100 * sh:.1f}%" for g, sh in out["group_share"].items()))
    print(f"{'original':10s}" + "".join(f"{100 * out['original']['group_top1'][g]:12.1f}" for g in groups))
    for name, res in out["variants"].items():
        print(f"{name:10s}" + "".join(f"{100 * np.mean([r['group_top1'][g] for r in res]):12.1f}" for g in groups))


def fit_adaptive_cmd(args) -> None:
    import copy
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
    policy_before = {k: x.clone() for k, x in ctrl.state_dict().items()}
    d, tab = load_sims(args.data), np.load(args.table)
    tr, va = D.val_setup(raw, args.seed, split="train"), D.val_setup(raw, args.seed, split="val")
    uniq, sid = np.unique(d["state"], return_inverse=True)
    made = d["made"]
    k, cand, ok = torch.as_tensor(tab["k"]).long(), torch.as_tensor(tab["cand"]), tab["ok"]
    rel_tr, rel_va = tr["rel"][uniq], va["rel"][tab["state"]]
    anchor_tr, anchor_va = pair_features(ctrl, [ctrl], rel_tr), pair_features(ctrl, [ctrl], rel_va)
    anchor_tr, anchor_va = {n: anchor_tr[n] for n in ("means", "valid")}, {n: anchor_va[n] for n in ("means", "valid")}
    out: dict = {"seed": args.seed, "variants": {}}
    for name in args.variants:
        res = []
        for hs in args.head_seeds:
            enc = copy.deepcopy(models["aux"] if name == "V2" else ctrl)
            lr_enc = args.lr_enc
            if name == "V3":                                                 # random initialisation of the same architecture
                torch.manual_seed(100 + hs)
                for mod in enc.modules():
                    if hasattr(mod, "reset_parameters"):
                        mod.reset_parameters()
                lr_enc = args.lr_scratch
            print(f"[{name} seed {hs}]", flush=True)
            ver, info = fit_adaptive(enc, rel_tr, anchor_tr, sid, d["a"], made, d["scratch"].astype(bool), ctrl.made2_scale, seed=hs,
                                     epochs=args.epochs, lr_enc=lr_enc, log=lambda m: print(m, flush=True))
            with torch.no_grad():
                sc = ver.candidate_scores(rel_va, anchor_va, k, cand).numpy()
            met = rank_metrics(sc, ok)
            res.append({"head_seed": hs, "best_epoch": info["best_epoch"], "holdout_bce": min(info["holdout"]), "train_bce": info["train"][info["best_epoch"]], **met})
            print(f"[{name} seed {hs}] holdout BCE {min(info['holdout']):.4f} (train {info['train'][info['best_epoch']]:.4f}) @{info['best_epoch']}  " +
                  "  ".join(f"@{kk} {100 * x:.1f}%" for kk, x in met["success_at_k"].items()) + f"  AUC {met['state_auc']:.3f}", flush=True)
        out["variants"][name] = res
    same = all(torch.equal(policy_before[n], x) for n, x in ctrl.state_dict().items())
    out["policy_unchanged"] = bool(same)
    print(f"\npolicy weights unchanged: {same}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    cols = (1, 2, 3, 5, 10, 20)
    print("\nsuccess@k (mean of head seeds; min–max)")
    print(f"{'':6s}" + "".join(f"{'@' + str(c):>16s}" for c in cols) + f"{'AUC':>8s}{'holdout BCE':>13s}")
    for name, res in out["variants"].items():
        cell = lambda c: f"{100 * np.mean(c):5.1f} ({100 * min(c):.1f}-{100 * max(c):.1f})".rjust(16)
        print(f"{name:6s}" + "".join(cell([r["success_at_k"][c] for r in res]) for c in cols)
              + f"{np.mean([r['state_auc'] for r in res]):8.3f}{np.mean([r['holdout_bce'] for r in res]):13.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("table", "gen", "fit", "fit-adaptive"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True, help="control run dir glob with {seed}")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out", default=None); s.add_argument("--threads", type=int, default=4)
        if name in ("table", "gen"):
            s.add_argument("--workers", type=int, default=4); s.add_argument("--max-states", type=int, default=0)
            if name == "gen":
                s.add_argument("--pair", choices=["verify", "label"], default="verify")
                s.add_argument("--n-top", type=int, default=12); s.add_argument("--top-pool", type=int, default=20); s.add_argument("--n-uniform", type=int, default=12)
        else:
            s.add_argument("--aux-run", required=True, help="aux-only run dir glob with {seed}")
            s.add_argument("--table", required=True); s.add_argument("--data", required=True, nargs="+", help="one or more simulation files (concatenated)")
            s.add_argument("--head-seeds", type=int, nargs="+", default=[0, 1, 2])
            if name == "fit":
                s.add_argument("--variants", nargs="*", default=["A", "B", "C"], choices=list(VARIANTS))
                s.add_argument("--aim-noise-deg", type=float, nargs="*", default=[], help="diagnostic: also give the head wrap(a − (exact aim + N(0, σ²))); one variant per σ")
                s.add_argument("--epochs", type=int, default=40); s.add_argument("--lr", type=float, default=1e-3)
            else:
                s.add_argument("--variants", nargs="+", default=["V1", "V2", "V3"], choices=["V1", "V2", "V3"])
                s.add_argument("--epochs", type=int, default=15)
                s.add_argument("--lr-enc", type=float, default=3e-4); s.add_argument("--lr-scratch", type=float, default=1e-3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    {"table": table, "gen": gen, "fit": fit, "fit-adaptive": fit_adaptive_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
