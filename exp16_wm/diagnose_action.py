"""
exp16_wm/diagnose_action.py — where the remaining shot failures of a BC GNN come from (Exp-16).

Three measurements per run (really played in the simulator, val states of the run's own split):

  split   selection vs regression: the policy's pair against the label pair (cand 0) with the model's own action,
          for `select` and for verify+refine (31 angles × 7 absolute speeds); plus angle-only refinement of the
          label pair (made-head grid vs gradient ascent on the made head) at the proposal's speed
  scan    label action moved ±0.6° (0.05° steps), really played: how often the made head's best offset is in the
          true success window, against baselines (random offset, the pair head's own proposal, offset 0), whether
          the head repairs proposals that miss the window, and whether ∂logit/∂angle points to the window
  sample  label pair, proposal's speed, angle = proposal + Laplace(b): success of one sample vs the best of N
          (the simulator picks), i.e. what sampling from a distribution head would give

Two more parts (`--parts oracle visited`; default `--parts base` = the three above):

  oracle  at the label pair, swap the pair head's angle and/or speed for the label's and play: is the precision
          failure the angle, the speed, or both together? Plus simulator-picked best over a speed grid / angle grid /
          both (an upper bound for refine with a perfect verifier). Judged by `oracle_row` (experiments.md table)
  visited roll the policy out for games and record each shot's balls left and previous shot's outcome: are the
          states the policy reaches harder than the teacher's val states? Judged by `visited_flags`. Also, for shots that
          follow a miss or a scratch: did the object balls stay where they were (cue ball aside)? `retry_row`
  vroracle the oracle swap again, but on the action the policy really plays (verify+refine, made head picks from the grid):
          swap the refined angle / speed for the label's, and split the failures into pair selection, angle, speed, rest
  speedmap  at the label angle, play a speed grid: where does the pair head's speed fall relative to the set of
          speeds that pocket (a gap between two modes, outside a single interval, or nothing works)? `speed_row`

Usage:
    python -m exp16_wm.diagnose_action --runs L4="logs/experiments/*_L4_m2w1_*_s{seed}_*" --seeds 0 1 2 \
        --out logs/diagnose_action/L4.json
    python -m exp16_wm.diagnose_action --runs ... --parts oracle visited --out logs/diagnose_action/L4_oracle_visited.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm import eval_by_balls as E
from exp16_wm.bc import build_split, load_teacher_data, to_unit
from exp16_wm.gnn_policy import PairGNNPolicy, action_diff
from exp16_wm.obs_transform import TRANSFORMS
from exp16_wm.outcomes import pocket_index_map
from exp16_wm.teacher_data import CLEAR_ENV

CHUNK = 300
SAMPLE_SCALES_DEG = (0.0, 0.05, 0.1, 0.2)                 # Laplace scale b of the angle noise (0 = the proposal itself)
DIST_BINS = ((0.0, 0.15), (0.15, 0.30), (0.30, 1.0))      # gradient check: distance of an offset from the window edge
PREV_KINDS = ("first", "after_hit", "after_miss", "after_scratch")   # how the previous shot of the game ended
GRID_MODES = {"speed7": "speed", "angle31": "angle", "both217": "both"}
SPEED_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)   # m/s
SPEED_KINDS = ("empty", "gap", "below", "above")


# ──────────────────────────────────────────────
# Pure helpers (tested)
# ──────────────────────────────────────────────

def signed_angle_diff_deg(a_unit: np.ndarray, b_unit: np.ndarray) -> np.ndarray:
    """Signed (a − b) of delta_angle in [-1, 1] units (1 = π rad), wrapped to (−180°, 180°], in degrees."""
    d = (np.asarray(a_unit, dtype=np.float64) - np.asarray(b_unit, dtype=np.float64)) * np.pi
    return np.degrees(np.arctan2(np.sin(d), np.cos(d)))


def laplace_perturb(angle_unit: float, scale_deg: float, rng: np.random.Generator) -> float:
    """angle + Laplace(0, scale_deg degrees), angle in [-1, 1] units (1 unit = 180°). scale 0 → unchanged, no draw."""
    if scale_deg <= 0:
        return float(angle_unit)
    return float(angle_unit) + float(rng.laplace(0.0, scale_deg)) / 180.0


def scan_baselines(prob: np.ndarray, truth: np.ndarray, offsets: np.ndarray, proposal_deg: np.ndarray) -> dict[str, float]:
    """
    prob, truth (S, K): the made head's P(made) and the real outcome at K angle offsets (deg) of S states;
    proposal_deg (S,): the pair head's own angle relative to the scanned centre (deg).
    Over the states whose window is only part of the scan:
      head_hit            the head's best offset really succeeds
      random_offset       a random offset succeeds (= mean window fraction)
      proposal_hit        the offset nearest the pair head's own proposal succeeds (no made head at all)
      label_offset_hit    offset 0 succeeds
      head_hit_when_proposal_misses / head_hit_when_proposal_hits   does the head repair a missed proposal / keep a good one
    """
    partial = truth.any(1) & ~truth.all(1)
    n = int(partial.sum())
    nan = float("nan")
    if n == 0:
        return {"n_partial": 0, "head_hit": nan, "random_offset": nan, "proposal_hit": nan, "label_offset_hit": nan,
                "n_proposal_miss": 0, "head_hit_when_proposal_misses": nan, "head_hit_when_proposal_hits": nan}
    rows = np.arange(len(truth))
    head = truth[rows, prob.argmax(1)]
    near = np.abs(offsets[None, :] - np.clip(proposal_deg, offsets.min(), offsets.max())[:, None]).argmin(1)
    prop = truth[rows, near]
    zero = truth[:, int(np.argmin(np.abs(offsets)))]
    miss, hit = partial & ~prop, partial & prop
    return {"n_partial": n, "head_hit": float(head[partial].mean()), "random_offset": float(truth[partial].mean(1).mean()),
            "proposal_hit": float(prop[partial].mean()), "label_offset_hit": float(zero[partial].mean()),
            "n_proposal_miss": int(miss.sum()),
            "head_hit_when_proposal_misses": float(head[miss].mean()) if miss.any() else nan,
            "head_hit_when_proposal_hits": float(head[hit].mean()) if hit.any() else nan}


def gradient_toward_window(grad: np.ndarray, truth: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One state: grad (K,) = ∂logit/∂angle at each offset, truth (K,) bool = real outcome.
    → (ok, dist) over the offsets outside the window: ok = the gradient points toward the window centre,
    dist = distance (deg) of the offset from the nearest window edge. Empty unless the window is only part of the scan.
    """
    if not (truth.any() and not truth.all()):
        return np.zeros(0, bool), np.zeros(0)
    inside = offsets[truth]
    lo, hi, centre = inside.min(), inside.max(), inside.mean()
    out = ~truth
    dist = np.where(offsets < lo, lo - offsets, offsets - hi)[out]
    ok = (np.sign(grad) == np.sign(centre - offsets))[out]
    return ok, dist


def pct(x: np.ndarray) -> str:
    return f"{100 * float(np.mean(x)):.1f}%" if len(x) else "n/a"


def substitute_action(model_act: np.ndarray, label_act: np.ndarray, angle: str = "model", speed: str = "model") -> np.ndarray:
    """(N, 2) env actions [delta_angle, speed]: the angle / speed column comes from "model" or "label" (inputs untouched)."""
    for name, v in (("angle", angle), ("speed", speed)):
        if v not in ("model", "label"):
            raise ValueError(f"{name} must be 'model' or 'label', got {v!r}")
    model_act, label_act = np.asarray(model_act), np.asarray(label_act)
    out = model_act.copy()
    if angle == "label":
        out[:, 0] = label_act[:, 0]
    if speed == "label":
        out[:, 1] = label_act[:, 1]
    return out


def candidate_grid(action: np.ndarray, deg: tuple, speeds: tuple, mode: str) -> np.ndarray:
    """
    One env action [angle rad, speed] → (K, 2) candidates around it: "speed" = the absolute speeds at its angle,
    "angle" = its angle ± deg (degrees) at its speed, "both" = every angle × every speed.
    """
    a = np.asarray(action, dtype=np.float64)
    angles = a[0] + np.radians(np.asarray(deg, dtype=np.float64))
    if mode == "speed":
        cand = [(a[0], sp) for sp in speeds]
    elif mode == "angle":
        cand = [(t, a[1]) for t in angles]
    elif mode == "both":
        cand = [(t, sp) for t in angles for sp in speeds]
    else:
        raise ValueError(f"mode must be speed / angle / both, got {mode!r}")
    return np.asarray(cand, dtype=np.float32)


def oracle_row(m: float, ls: float, la: float, l: float) -> str:
    """
    Row of the oracle-swap judgement table (experiments.md 2026-10-12), rates in [0, 1]:
    m model angle + model speed, ls model angle + label speed, la label angle + model speed, l label angle + label speed.
      A  l < 97%                                  the label action is not reproduced (harness problem)
      B  r_a + r_s < 0.5 (l − m)                  angle and speed must be right together
      C  r_a ≥ 2 r_s and r_a − r_s ≥ 2%p         angle error dominates
      D  r_s ≥ 2 r_a and r_s − r_a ≥ 2%p         speed error dominates
      E  otherwise                                both contribute
    """
    r_a, r_s = la - m, ls - m
    if l < 0.97:
        return "A"
    if r_a + r_s < 0.5 * (l - m):
        return "B"
    if r_a >= 2 * r_s and r_a - r_s >= 0.02:
        return "C"
    if r_s >= 2 * r_a and r_s - r_a >= 0.02:
        return "D"
    return "E"


def previous_outcome(pocketed: bool, scratch: bool) -> str:
    """How a shot ended, as the label of the next shot's state (scratch wins over a pocket)."""
    return "after_scratch" if scratch else ("after_hit" if pocketed else "after_miss")


def shot_groups(n_left: np.ndarray, prev: np.ndarray, clean: np.ndarray) -> dict:
    """Clean rate and counts of shots by balls left and by how the previous shot ended; empty groups are nan."""
    nan = float("nan")
    n = len(clean)
    rate = lambda mask: float(clean[mask].mean()) if mask.any() else nan
    by_balls = {b: {"n": int((n_left == b).sum()), "rate": rate(n_left == b)} for b in (1, 2, 3)}
    by_prev = {k: {"n": int((prev == k).sum()), "rate": rate(prev == k)} for k in PREV_KINDS}
    normal = np.isin(prev, ("first", "after_hit"))
    return {"by_balls": by_balls, "mix": {b: (by_balls[b]["n"] / n if n else nan) for b in (1, 2, 3)}, "by_prev": by_prev,
            "normal": rate(normal), "after_fail": rate(~normal), "n_after_fail": int((~normal).sum())}


def visited_flags(game: dict, val: dict, normal: float, after_fail: float) -> dict[str, bool]:
    """
    Rows of the visited-states judgement table (experiments.md 2026-10-12); game / val: balls left → clean rate.
      A  every balls-left group within 5%p of val and after-failure states within 3%p of normal ones
      B  after-failure states ≥ 5%p lower than first-shot / after-hit states
      C  games ≥ 5%p lower than val in every group      D  games ≥ 5%p higher than val in every group
    Several rows can hold at once. Groups or comparisons with nan are not evaluated.
    """
    diffs = [game[b] - val[b] for b in (1, 2, 3) if np.isfinite(game[b]) and np.isfinite(val[b])]
    gap = normal - after_fail if np.isfinite(normal) and np.isfinite(after_fail) else float("nan")
    return {"A": bool(diffs) and all(abs(d) <= 0.05 for d in diffs) and (not np.isfinite(gap) or gap <= 0.03),
            "B": bool(np.isfinite(gap) and gap >= 0.05),
            "C": bool(diffs) and all(d <= -0.05 for d in diffs),
            "D": bool(diffs) and all(d >= 0.05 for d in diffs)}


def state_displacement(prev_obs: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """
    Between two raw (23,) observations (cue xy, 3 × (x, y, pocketed flag), pockets), in obs units:
    (how far the cue ball moved, the largest move of an object ball that is unpocketed in both; nan if there is none).
    """
    p, c = np.asarray(prev_obs, dtype=np.float64), np.asarray(obs, dtype=np.float64)
    cue = float(np.hypot(*(c[:2] - p[:2])))
    pb, cb = p[2:11].reshape(3, 3), c[2:11].reshape(3, 3)
    alive = (pb[:, 2] <= 0.5) & (cb[:, 2] <= 0.5)
    if not alive.any():
        return cue, float("nan")
    return cue, float(np.hypot(*(cb[alive, :2] - pb[alive, :2]).T).max())


def retry_groups(prev: np.ndarray, obj_move: np.ndarray, cue_move: np.ndarray, clean: np.ndarray, tol: float = 1e-3) -> dict:
    """
    Shots that follow a miss or a scratch, split by whether the object balls stayed (move < tol) or moved.
    Shots with no object ball left in both states (nan) are counted separately.
    """
    nan = float("nan")
    after = np.isin(prev, ("after_miss", "after_scratch"))
    stayed = after & (obj_move < tol)
    moved = after & (obj_move >= tol)
    n_after = int(after.sum())
    rate = lambda m: float(clean[m].mean()) if m.any() else nan
    q = lambda x, f: float(np.nanpercentile(x, f)) if np.isfinite(x).any() else nan
    return {"n_after": n_after, "n_stayed": int(stayed.sum()), "n_moved": int(moved.sum()), "n_nan": int((after & np.isnan(obj_move)).sum()),
            "share_stayed": (float(stayed.sum()) / n_after if n_after else nan), "rate_stayed": rate(stayed), "rate_moved": rate(moved),
            "obj_move_quantiles": {str(f): q(obj_move[after], f) for f in (10, 50, 90)},
            "cue_move_quantiles": {str(f): q(cue_move[after], f) for f in (10, 50, 90)}}


def retry_row(g: dict) -> str:
    """
    Row of the TODO 22 judgement table (experiments.md 2026-10-12), g = retry_groups(...):
      D  fewer than 30 shots with the object balls staying   not enough to judge
      A  stayed share ≥ 50% and their success ≥ 5%p below the moved group   repeated failure on the same target
      C  stayed share ≥ 50% but no such gap                  no repeated-failure tendency
      B  stayed share < 50%                                  mostly new layouts
    """
    if g["n_stayed"] < 30:
        return "D"
    if g["share_stayed"] < 0.5:
        return "B"
    if np.isfinite(g["rate_moved"]) and g["rate_moved"] - g["rate_stayed"] >= 0.05:
        return "A"
    return "C"


def n_intervals(ok: np.ndarray) -> int:
    """Number of separate runs of True in a boolean vector ordered by speed."""
    ok = np.asarray(ok, dtype=bool)
    return int((ok & ~np.concatenate([[False], ok[:-1]])).sum())


def speed_failure_kind(speed: float, speeds: np.ndarray, ok: np.ndarray) -> str:
    """
    Where a failing speed lies relative to the speeds that pocket (speeds ascending, ok = pocketed at that speed):
    empty (none does), below / above (outside all of them), gap (between the slowest and fastest success, but failing).
    """
    ok = np.asarray(ok, dtype=bool)
    if not ok.any():
        return "empty"
    lo, hi = np.asarray(speeds)[ok].min(), np.asarray(speeds)[ok].max()
    return "below" if speed < lo else ("above" if speed > hi else "gap")


def speed_row(shares: dict, n_fail: int) -> str:
    """
    Row of the speed-map judgement table (experiments.md 2026-10-12); shares = fraction of the failing states per kind.
      E  fewer than 30 failing states     C  empty ≥ 30%     A  gap ≥ 40%     B  below + above ≥ 60%     D  otherwise
    """
    if n_fail < 30:
        return "E"
    if shares["empty"] >= 0.30:
        return "C"
    if shares["gap"] >= 0.40:
        return "A"
    if shares["below"] + shares["above"] >= 0.60:
        return "B"
    return "D"


# ──────────────────────────────────────────────
# Model-side helpers
# ──────────────────────────────────────────────

def made_logit(m: PairGNNPolicy, o: dict, k: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """cand (B, C, 2) candidate actions for pair k[b] → (B, C) made2 logits (the inputs predict_unit's refine uses)."""
    rows = torch.arange(len(k)); C = cand.shape[1]
    a = o["means"][rows, k]
    e_k = o["e"][rows, k][:, None].expand(-1, C, -1)
    feats = [e_k, cand, action_diff(cand, a[:, None].expand(-1, C, -1)) * m.made2_scale]
    if m.made2_geom:                                                              # the aim term (exact or learned aim)
        feats.append(m._geom_feature(o["aim"][rows, k][:, None].expand(-1, C), cand[..., 0]))
    return m.head_made2(torch.cat(feats, -1)).squeeze(-1)


def refine_at(m: PairGNNPolicy, o: dict, k: torch.Tensor) -> torch.Tensor:
    """The refine step of predict_unit (current m.refine_* grid) for a given pair k instead of the selected one."""
    rows = torch.arange(len(k))
    cand = m._unit_offsets(o["means"][rows, k])
    return cand[rows, made_logit(m, o, k, cand).argmax(1)]


def grad_refine(m: PairGNNPolicy, o: dict, k: torch.Tensor, steps: int = 12, step_deg: float = 0.04) -> torch.Tensor:
    """
    Angle-only gradient ascent on the made head from the pair head's proposal (speed fixed): steps of step_deg along
    the sign of ∂logit/∂angle; the best iterate seen (never worse than the start) is returned.
    """
    rows = torch.arange(len(k)); a = o["means"][rows, k]; e_k = o["e"][rows, k]

    def logit(ang: torch.Tensor) -> torch.Tensor:
        act = torch.stack([ang, a[:, 1]], -1)
        return m.head_made2(torch.cat([e_k, act, action_diff(act, a) * m.made2_scale], -1)).squeeze(-1)

    ang = a[:, 0].clone()
    with torch.no_grad():
        best = logit(ang)
    best_ang = ang.clone()
    for _ in range(steps):
        ang = ang.detach().requires_grad_(True)
        g, = torch.autograd.grad(logit(ang).sum(), ang)
        ang = (ang + torch.sign(g) * step_deg / 180.0).detach()
        with torch.no_grad():
            z = logit(ang)
        better = z > best
        best, best_ang = torch.where(better, z, best), torch.where(better, ang, best_ang)
    return torch.stack([best_ang, a[:, 1]], -1)


def configure(m: PairGNNPolicy, mode: str, deg: tuple, speeds: tuple | None) -> None:
    """Set the inference mode and refine grid (proposal speed × 1 when speeds is None)."""
    m.infer, m.refine_deg, m.refine_speed_abs, m.refine_speed = mode, deg, speeds, (1.0,)


def to_env(m: PairGNNPolicy, u: torch.Tensor) -> np.ndarray:
    return (u * m.act_scale + m.act_bias).numpy()


def clean_flags(env, pm, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """Really play acts[i] on obs[i]: a pocket and no scratch (the 'clean' definition of eval_by_balls)."""
    return np.array([(lambda x: x[0].any() and not x[1])(E.play(env, pm, obs[i], acts[i])) for i in range(len(obs))])


def val_setup(raw: dict, seed: int, split: str = "val") -> dict:
    """Val (or train) states of the run's split with the label pair (candidate 0) and its action."""
    data, tr, va = build_split(dict(raw), 0.1, seed, 1.0, "none")
    vi = np.where(va if split == "val" else tr)[0]
    obs = data["obs"][vi]
    g = data["good_pairs"][vi]
    k0 = np.argmax(g >= 0, 1)
    return {"obs": obs, "rel": torch.as_tensor(TRANSFORMS["relative"](obs)), "nl": E.balls_left(obs), "good": g,
            "p0": torch.as_tensor(g[np.arange(len(vi)), k0].clip(min=0).astype(np.int64)), "lab": (g >= 0).any(1),
            "label_act": data["good_actions"][vi][np.arange(len(vi)), k0]}


# ──────────────────────────────────────────────
# Measurements
# ──────────────────────────────────────────────

def measure_split(m: PairGNNPolicy, v: dict, env, pm) -> dict:
    rel, p0, good = v["rel"], v["p0"], v["good"]
    acts: dict[str, list] = {n: [] for n in ("sel", "lab", "vr", "labref", "lab_grid", "lab_grad", "lab_grad_small")}
    ks_sel, ks_ver = [], []
    for lo in range(0, len(rel), CHUNK):
        x, p = rel[lo:lo + CHUNK], p0[lo:lo + CHUNK]
        rows = torch.arange(len(x))
        with torch.no_grad():
            o = m.forward_full(x)
            ks = o["logits"].argmax(1)
            kv = (F.log_softmax(o["logits"], 1) + F.logsigmoid(m.made2_logits(o, o["means"]))).argmax(1)
            ks_sel.append(ks.numpy()); ks_ver.append(kv.numpy())
            acts["sel"].append(to_env(m, o["means"][rows, ks]))
            acts["lab"].append(to_env(m, o["means"][rows, p]))
            configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
            acts["vr"].append(to_env(m, m.predict_unit(x)))
            acts["labref"].append(to_env(m, refine_at(m, o, p)))
            configure(m, "verify_refine", E.ANG31, None)
            acts["lab_grid"].append(to_env(m, refine_at(m, o, p)))
        acts["lab_grad"].append(to_env(m, grad_refine(m, o, p)))
        acts["lab_grad_small"].append(to_env(m, grad_refine(m, o, p, steps=24, step_deg=0.01)))
    ks_sel, ks_ver = np.concatenate(ks_sel), np.concatenate(ks_ver)
    ok = {n: clean_flags(env, pm, v["obs"], np.concatenate(a)) for n, a in acts.items()}
    in_good = lambda k: np.array([(good[i] == k[i]).any() for i in range(len(k))])
    return {"nl": v["nl"], "lab": v["lab"], "ok": ok, "sel_in_good": in_good(ks_sel), "ver_in_good": in_good(ks_ver),
            "sel_is_label": ks_sel == p0.numpy()}


def measure_scan(m: PairGNNPolicy, v: dict, seed: int, env, pm, n_scan: int) -> dict:
    off = E.OFFSETS
    pick = np.random.default_rng(seed).choice(np.where(v["lab"])[0], size=min(n_scan, int(v["lab"].sum())), replace=False)
    P, Y, prop, gok, gdist = [], [], [], [], []
    for j in pick:
        pair, la = int(v["p0"][j]), v["label_act"][j]
        truth = E.scan_truth(env, pm, v["obs"][j], pair, la)
        off_u = torch.as_tensor(np.stack([to_unit(np.array([la[0] + np.radians(d), la[1]], np.float32)) for d in off]).astype(np.float32))
        with torch.no_grad():
            o = m.forward_full(v["rel"][j:j + 1].expand(len(off), -1))
        x = off_u.clone().requires_grad_(True)
        z = m.made2_logits(o, x)[:, pair]
        grad, = torch.autograd.grad(z.sum(), x)
        P.append(torch.sigmoid(z).detach().numpy()); Y.append(truth)
        ok, dist = gradient_toward_window(grad[:, 0].numpy(), truth, off)
        gok.append(ok); gdist.append(dist)
        prop.append(float(signed_angle_diff_deg(np.array([o["means"][0, pair, 0].item()]), np.array([to_unit(la.astype(np.float32))[0]]))[0]))
    return {"P": np.stack(P), "Y": np.stack(Y), "prop": np.array(prop), "gok": np.concatenate(gok), "gdist": np.concatenate(gdist)}


def measure_sample(m: PairGNNPolicy, v: dict, seed: int, env, pm, n_states: int, n_draws: int) -> dict:
    pick = np.random.default_rng(100 + seed).choice(np.where(v["lab"])[0], size=min(n_states, int(v["lab"].sum())), replace=False)
    with torch.no_grad():
        o = m.forward_full(v["rel"][pick])
        a = o["means"][torch.arange(len(pick)), v["p0"][pick]]
    rng = np.random.default_rng(seed)
    out: dict = {"two_left": v["nl"][pick] == 2}
    for b in SAMPLE_SCALES_DEG:
        single, best = [], []
        for i, j in enumerate(pick):
            ok = []
            for _ in range(1 if b == 0 else n_draws):
                u = a[i].clone()
                u[0] = laplace_perturb(u[0].item(), b, rng)
                ok.append(bool(clean_flags(env, pm, v["obs"][j:j + 1], to_env(m, u[None]))[0]))
            single.append(np.mean(ok)); best.append(any(ok))
        out[b] = {"single": np.array(single), "best": np.array(best)}
    return out


def any_clean(env, pm, obs: np.ndarray, cands: np.ndarray) -> bool:
    """Does any candidate action pocket cleanly from this state? Stops at the first success."""
    return any(bool((lambda x: x[0].any() and not x[1])(E.play(env, pm, obs, c))) for c in cands)


def measure_oracle(m: PairGNNPolicy, v: dict, seed: int, env, pm, n_grid: int) -> dict:
    rows = torch.arange(len(v["rel"]))
    with torch.no_grad():
        o = m.forward_full(v["rel"])
        model = to_env(m, o["means"][rows, v["p0"]])                      # the pair head's own action at the label pair
    label = np.asarray(v["label_act"], dtype=np.float32)
    acts = {"M": model, "Ls": substitute_action(model, label, "model", "label"),
            "La": substitute_action(model, label, "label", "model"), "L": label}
    ok = {k: clean_flags(env, pm, v["obs"], a) for k, a in acts.items()}
    pick = np.random.default_rng(200 + seed).choice(np.where(v["lab"])[0], size=min(n_grid, int(v["lab"].sum())), replace=False)
    grid = {name: np.array([any_clean(env, pm, v["obs"][j], candidate_grid(model[j], E.ANG31, E.ABS_SPEEDS, mode)) for j in pick])
            for name, mode in GRID_MODES.items()}
    configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
    with torch.no_grad():
        ref = to_env(m, refine_at(m, m.forward_full(v["rel"][pick]), v["p0"][pick]))
    grid["M"] = ok["M"][pick]
    grid["verify_refine_label_pair"] = clean_flags(env, pm, v["obs"][pick], ref)
    return {"lab": v["lab"], "nl": v["nl"], "ok": ok, "grid": grid}


def rollout_shots(m: PairGNNPolicy, n_games: int, env_kw: dict, seed: int = 0) -> dict:
    """
    Play n_games with the policy's current inference mode (same protocol as bc.evaluate_policy: reset(seed) once, then
    one reset per game) and record per shot the balls left, how the previous shot ended and whether it was clean.
    """
    env = BilliardsEnv(**env_kw)
    env.reset(seed=seed)
    n_left, prev, clean, cue_move, obj_move = [], [], [], [], []
    for _ in range(n_games):
        obs, _ = env.reset()
        last, last_obs, done = "first", None, False
        while not done:
            n_left.append(int(E.balls_left(obs[None])[0]))
            prev.append(last)
            cm, om = (float("nan"), float("nan")) if last_obs is None else state_displacement(last_obs, obs)
            cue_move.append(cm); obj_move.append(om)
            last_obs = obs
            obs, _, term, trunc, info = env.step(m.act(TRANSFORMS["relative"](obs)))
            hit, scr = info["pocketed_this_step"] > 0, bool(info["scratch"])
            clean.append(bool(hit and not scr))
            last, done = previous_outcome(hit, scr), term or trunc
    env.close()
    return {"n_left": np.array(n_left), "prev": np.array(prev), "clean": np.array(clean),
            "cue_move": np.array(cue_move), "obj_move": np.array(obj_move)}


def measure_visited(m: PairGNNPolicy, v: dict, env, pm, n_games: int) -> dict:
    configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
    with torch.no_grad():
        acts = np.concatenate([to_env(m, m.predict_unit(v["rel"][lo:lo + CHUNK])) for lo in range(0, len(v["rel"]), CHUNK)])
    return {"val": {"nl": v["nl"], "lab": v["lab"], "clean": clean_flags(env, pm, v["obs"], acts)},
            "games": rollout_shots(m, n_games, CLEAR_ENV)}


def speed_map(v: dict, env, pm, idx: np.ndarray) -> np.ndarray:
    """(len(idx), K) did [label angle, speed] pocket cleanly, for each speed of SPEED_GRID (depends on the state only)."""
    return np.array([[bool((lambda x: x[0].any() and not x[1])(E.play(env, pm, v["obs"][j], np.array([v["label_act"][j][0], sp], np.float32))))
                      for sp in SPEED_GRID] for j in idx])


def measure_speedmap(m: PairGNNPolicy, v: dict, seed: int, env, pm, cache: dict, n_states: int = 0) -> dict:
    idx = np.where(v["lab"])[0]
    if n_states:
        idx = np.sort(np.random.default_rng(300 + seed).choice(idx, size=min(n_states, len(idx)), replace=False))
    if (seed, n_states) not in cache:                                  # the map does not depend on the model: reuse it
        cache[(seed, n_states)] = speed_map(v, env, pm, idx)
    flags = cache[(seed, n_states)]
    with torch.no_grad():
        o = m.forward_full(v["rel"][idx])
        a = to_env(m, o["means"][torch.arange(len(idx)), v["p0"][idx]])        # the pair head's own action at the label pair
    model_speed = a[:, 1].astype(np.float64)
    acts = np.stack([np.asarray(v["label_act"][idx], dtype=np.float32)[:, 0], model_speed.astype(np.float32)], 1)
    return {"nl": v["nl"][idx], "flags": flags, "model_speed": model_speed, "model_ok": clean_flags(env, pm, v["obs"][idx], acts)}


def measure_vr_oracle(m: PairGNNPolicy, v: dict, env, pm) -> dict:
    """Oracle swap on the verify+refine action: the label pair's refined action R, and the policy's own played action P."""
    configure(m, "verify_refine", E.ANG31, E.ABS_SPEEDS)
    refined, played = [], []
    for lo in range(0, len(v["rel"]), CHUNK):
        x = v["rel"][lo:lo + CHUNK]
        with torch.no_grad():
            o = m.forward_full(x)
            refined.append(to_env(m, refine_at(m, o, v["p0"][lo:lo + CHUNK])))
            played.append(to_env(m, m.predict_unit(x)))
    ref, label = np.concatenate(refined), np.asarray(v["label_act"], dtype=np.float32)
    acts = {"R": ref, "Rs": substitute_action(ref, label, "model", "label"), "La": substitute_action(ref, label, "label", "model"),
            "L": label, "P": np.concatenate(played)}
    return {"lab": v["lab"], "nl": v["nl"], "ok": {k: clean_flags(env, pm, v["obs"], a) for k, a in acts.items()}}


def failure_budget(p: float, r: float, rs: float, ra: float, l: float) -> dict[str, float]:
    """
    Where the failures of the played action go, in rate (0-1): p = policy's played action, r = label pair + refine,
    rs / ra = r with the label's speed / angle, l = the label action.
      selection = r − p     angle = ra − r     speed = rs − r     other = (l − r) − angle − speed     never_fixed = 1 − l
    The parts sum to 1 − p when the two swaps are measured on the same states.
    """
    angle, speed = ra - r, rs - r
    return {"selection": r - p, "angle": angle, "speed": speed, "other": (l - r) - angle - speed, "never_fixed": 1.0 - l}


# ──────────────────────────────────────────────
# Pooling and printing
# ──────────────────────────────────────────────

def summarize_vr_oracle(rs: list) -> dict:
    lab = np.concatenate([r["lab"] for r in rs]); nl = np.concatenate([r["nl"] for r in rs])
    ok = {k: np.concatenate([r["ok"][k] for r in rs]) for k in rs[0]["ok"]}
    rate = lambda a, mask: float(a[mask].mean()) if mask.any() else float("nan")
    groups = {"all": lab, **{str(b): lab & (nl == b) for b in (1, 2, 3)}}
    out: dict = {"n": int(lab.sum()), "clean": {k: {g: rate(a, mk) for g, mk in groups.items()} for k, a in ok.items()}}
    out["budget"] = {g: failure_budget(out["clean"]["P"][g], out["clean"]["R"][g], out["clean"]["Rs"][g], out["clean"]["La"][g], out["clean"]["L"][g])
                     for g in groups}
    c = out["clean"]
    out["row"] = oracle_row(c["R"]["all"], c["Rs"]["all"], c["La"]["all"], c["L"]["all"])
    b = out["budget"]["all"]
    out["selection_first"] = bool(b["selection"] > max(b["angle"], b["speed"]))
    return out


def print_vr_oracle(label: str, s: dict) -> None:
    f = "%-60s %8s %8s %8s %8s"
    print(f"\n===== {label} oracle swap on the verify+refine action (val states with a label: {s['n']})")
    print(f % ("clean (real shot, pocket, no scratch)", "1 left", "2 left", "3 left", "all"))
    names = {"P": "P  policy's pair + refine (what is played in games)", "R": "R  label pair + refine", "Rs": "Rs refined angle + label speed",
             "La": "La label angle + refined speed", "L": "L  label action (teacher)"}
    for k, name in names.items():
        c = s["clean"][k]
        print(f % (name, *[f"{100 * c[g]:.1f}%" for g in ("1", "2", "3", "all")]))
    print(f % ("failure budget (of the failures of P, in %p of all states):", "1 left", "2 left", "3 left", "all"))
    for k, name in (("selection", "  pair selection (R − P)"), ("angle", "  angle, at the label pair (La − R)"), ("speed", "  speed, at the label pair (Rs − R)"),
                    ("other", "  both together / other ((L − R) − angle − speed)"), ("never_fixed", "  not fixed even by the label action (1 − L)")):
        print(f % (name, *[f"{100 * s['budget'][g][k]:+.1f}" for g in ("1", "2", "3", "all")]))
    b = s["budget"]["all"]
    print(f"  total failures of P {100 * (1 - s['clean']['P']['all']):.1f}%p = " + " + ".join(f"{k} {100 * b[k]:.1f}" for k in b)
          + f"   → judgement row {s['row']}" + ("; pair selection is the largest part" if s["selection_first"] else ""))


def summarize_speedmap(rs: list) -> dict:
    flags = np.concatenate([r["flags"] for r in rs]); ms = np.concatenate([r["model_speed"] for r in rs])
    ok = np.concatenate([r["model_ok"] for r in rs]); nl = np.concatenate([r["nl"] for r in rs])
    speeds = np.asarray(SPEED_GRID)
    ni = np.array([n_intervals(f) for f in flags])
    nonempty = flags.any(1)
    ratio = np.array([speeds[f].max() / speeds[f].min() for f in flags[nonempty]]) if nonempty.any() else np.array([])
    fail = np.where(~ok)[0]
    kinds = [speed_failure_kind(ms[i], speeds, flags[i]) for i in fail]
    shares = {k: (sum(x == k for x in kinds) / len(kinds) if kinds else float("nan")) for k in SPEED_KINDS}
    q = lambda x, f: float(np.percentile(x, f)) if len(x) else float("nan")
    return {"n": int(len(ok)), "model_ok": float(ok.mean()), "n_fail": int(len(fail)), "fail_shares": shares,
            "map": {"share_empty": float((~nonempty).mean()), "share_multi_interval": float((ni >= 2).mean()),
                    "share_single_interval": float((ni == 1).mean()), "range_ratio_quantiles": {str(f): q(ratio, f) for f in (10, 50, 90)}},
            "fail_by_balls": {str(b): int(((nl == b) & ~ok).sum()) for b in (1, 2, 3)},
            "model_speed_quantiles_failing": {str(f): q(ms[fail], f) for f in (10, 50, 90)}, "row": speed_row(shares, len(fail))}


def print_speedmap(label: str, s: dict) -> None:
    m, sh = s["map"], s["fail_shares"]
    print(f"\n===== {label} speed window map at the label angle (val states with a label: {s['n']}; grid of {len(SPEED_GRID)} speeds, {SPEED_GRID[0]}-{SPEED_GRID[-1]} m/s)")
    print(f"  map: no speed pockets {100 * m['share_empty']:.1f}% | one interval {100 * m['share_single_interval']:.1f}% | two or more intervals {100 * m['share_multi_interval']:.1f}% | "
          f"fastest/slowest successful speed p10/p50/p90 = " + " / ".join(f"{m['range_ratio_quantiles'][k]:.1f}x" for k in ('10', '50', '90')))
    print(f"  the pair head's speed at the label angle pockets in {100 * s['model_ok']:.1f}% ({s['n_fail']} failing states; by balls left "
          + "/".join(str(s['fail_by_balls'][b]) for b in ('1', '2', '3')) + ")")
    print("  where the failing speed lies: " + " | ".join(f"{k} {100 * sh[k]:.1f}%" for k in SPEED_KINDS) +
          "   failing model speed p10/p50/p90 = " + " / ".join(f"{s['model_speed_quantiles_failing'][k]:.2f}" for k in ('10', '50', '90')) + f" m/s   → judgement row {s['row']}")


def summarize_oracle(rs: list) -> dict:
    lab = np.concatenate([r["lab"] for r in rs]); nl = np.concatenate([r["nl"] for r in rs])
    out: dict = {"n": int(lab.sum()), "clean": {}, "grid": {}}
    for k in rs[0]["ok"]:
        arr = np.concatenate([r["ok"][k] for r in rs])
        out["clean"][k] = {**{str(b): x for b, x in E.by_balls(arr[lab], nl[lab]).items()}, "all": float(arr[lab].mean())}
    out["grid_n"] = int(sum(len(r["grid"]["M"]) for r in rs))
    for k in rs[0]["grid"]:
        out["grid"][k] = float(np.concatenate([r["grid"][k] for r in rs]).mean())
    c = out["clean"]
    out["row"] = oracle_row(c["M"]["all"], c["Ls"]["all"], c["La"]["all"], c["L"]["all"])
    return out


def summarize_visited(rs: list) -> dict:
    cat = lambda f: np.concatenate([f(r) for r in rs])
    games = shot_groups(cat(lambda r: r["games"]["n_left"]), cat(lambda r: r["games"]["prev"]), cat(lambda r: r["games"]["clean"]))
    nl, lab, clean = cat(lambda r: r["val"]["nl"]), cat(lambda r: r["val"]["lab"]), cat(lambda r: r["val"]["clean"])
    val = {b: (float(clean[lab & (nl == b)].mean()) if (lab & (nl == b)).any() else float("nan")) for b in (1, 2, 3)}
    game = {b: games["by_balls"][b]["rate"] for b in (1, 2, 3)}
    retry = retry_groups(cat(lambda r: r["games"]["prev"]), cat(lambda r: r["games"]["obj_move"]),
                         cat(lambda r: r["games"]["cue_move"]), cat(lambda r: r["games"]["clean"]))
    retry["row"] = retry_row(retry)
    return {"games": games, "val": {"by_balls": val, "all": float(clean[lab].mean()), "n": int(lab.sum())},
            "flags": visited_flags(game, val, games["normal"], games["after_fail"]), "retry": retry}

def summarize(label: str, split: list, scan: list, sample: list | None) -> dict:
    nl = np.concatenate([r["nl"] for r in split]); lab = np.concatenate([r["lab"] for r in split])
    cat = lambda f: np.concatenate([f(r) for r in split])
    ok = {n: cat(lambda r: r["ok"][n]) for n in split[0]["ok"]}

    def rows(arr: np.ndarray) -> dict:
        return {**{str(k): v for k, v in E.by_balls(arr[lab], nl[lab]).items()}, "all": float(arr[lab].mean())}
    out: dict = {"n_by_balls": {str(b): int(((nl == b) & lab).sum()) for b in (1, 2, 3)},
                 "split": {"select_pair_in_good": rows(cat(lambda r: r["sel_in_good"])),
                           "select_pair_is_label": rows(cat(lambda r: r["sel_is_label"])),
                           "select_clean_policy_pair": rows(ok["sel"]), "select_clean_label_pair": rows(ok["lab"]),
                           "verify_pair_in_good": rows(cat(lambda r: r["ver_in_good"])),
                           "verify_refine_clean_policy_pair": rows(ok["vr"]), "verify_refine_clean_label_pair": rows(ok["labref"]),
                           "angle_only_proposal": rows(ok["lab"]), "angle_only_grid": rows(ok["lab_grid"]),
                           "angle_only_gradient": rows(ok["lab_grad"]), "angle_only_gradient_small_step": rows(ok["lab_grad_small"])}}
    out["scan"] = scan_baselines(np.concatenate([r["P"] for r in scan]), np.concatenate([r["Y"] for r in scan]),
                                 E.OFFSETS, np.concatenate([r["prop"] for r in scan]))
    gok, gd = np.concatenate([r["gok"] for r in scan]), np.concatenate([r["gdist"] for r in scan])
    out["gradient_toward_window"] = {f"{lo:.2f}-{hi:.2f}": float(gok[(gd >= lo) & (gd < hi)].mean()) if ((gd >= lo) & (gd < hi)).any() else float("nan")
                                     for lo, hi in DIST_BINS} | {"all": float(gok.mean())}
    if sample:
        two = np.concatenate([r["two_left"] for r in sample])
        out["sample"] = {str(b): {"single": float(np.concatenate([r[b]["single"] for r in sample]).mean()),
                                  "single_two_left": float(np.concatenate([r[b]["single"] for r in sample])[two].mean()),
                                  "best_of_n": float(np.concatenate([r[b]["best"] for r in sample]).mean())} for b in SAMPLE_SCALES_DEG}
    return out


def print_summary(label: str, s: dict) -> None:
    n = s["n_by_balls"]
    print(f"\n===== {label} (val states with a label pair; n by balls left {n['1']}/{n['2']}/{n['3']})")
    f = "%-58s %8s %8s %8s %8s"
    print(f % ("split (clean = real shot, pocket, no scratch)", "1 left", "2 left", "3 left", "all"))
    names = {"select_pair_in_good": "select: policy's pair is one of the label's good pairs",
             "select_pair_is_label": "select: policy's pair == label pair (cand 0)",
             "select_clean_policy_pair": "select: clean, policy's pair",
             "select_clean_label_pair": "select: clean, label pair + model action",
             "verify_pair_in_good": "verify: policy's pair is one of the label's good pairs",
             "verify_refine_clean_policy_pair": "verify+refine 31x7: clean, policy's pair",
             "verify_refine_clean_label_pair": "verify+refine 31x7: clean, label pair + refine",
             "angle_only_proposal": "angle only, label pair, proposal as is",
             "angle_only_grid": "angle only: made-head grid +-0.3deg x31",
             "angle_only_gradient": "angle only: gradient ascent (12 x 0.04deg)",
             "angle_only_gradient_small_step": "angle only: gradient ascent (24 x 0.01deg)"}
    for k, label_ in names.items():
        r = s["split"][k]
        print(f % (label_, *[f"{100 * r[c]:.1f}%" for c in ("1", "2", "3", "all")]))
    c = s["scan"]
    print(f"scan (n partial states {c['n_partial']}): made head argmax in window {100 * c['head_hit']:.1f}% | random offset {100 * c['random_offset']:.1f}% | "
          f"nearest the pair head's proposal {100 * c['proposal_hit']:.1f}% | offset 0 {100 * c['label_offset_hit']:.1f}%")
    print(f"  proposal misses the window in {c['n_proposal_miss']} states: head lands inside there {100 * c['head_hit_when_proposal_misses']:.1f}%; "
          f"proposal inside: head stays inside {100 * c['head_hit_when_proposal_hits']:.1f}%")
    print("  gradient points toward the window (offsets outside it): " + " | ".join((f"{k}deg from edge " if k != "all" else "all ") + f"{100 * v:.1f}%" for k, v in s["gradient_toward_window"].items()))
    if "sample" in s:
        print("sample (label pair, proposal's speed): angle = proposal + Laplace(b)      1 sample   1 sample (2 left)   best of N (simulator picks)")
        for b, r in s["sample"].items():
            print(f"  b = {float(b):.2f} deg {'(proposal as is)' if float(b) == 0 else '':18s}{100 * r['single']:>20.1f}%{100 * r['single_two_left']:>17.1f}%{100 * r['best_of_n']:>22.1f}%")


def print_oracle(label: str, s: dict) -> None:
    f = "%-62s %8s %8s %8s %8s"
    print(f"\n===== {label} oracle swap (label pair, pair head's own action; val states with a label: {s['n']})")
    print(f % ("clean (real shot, pocket, no scratch)", "1 left", "2 left", "3 left", "all"))
    names = {"M": "M  model angle + model speed", "Ls": "Ls model angle + label speed", "La": "La label angle + model speed",
             "L": "L  label angle + label speed (teacher action)"}
    for k, name in names.items():
        c = s["clean"][k]
        print(f % (name, *[f"{100 * c[b]:.1f}%" for b in ("1", "2", "3", "all")]))
    c = s["clean"]
    print(f"  r_a = La − M = {100 * (c['La']['all'] - c['M']['all']):+.1f}%p   r_s = Ls − M = {100 * (c['Ls']['all'] - c['M']['all']):+.1f}%p   "
          f"L − M = {100 * (c['L']['all'] - c['M']['all']):+.1f}%p   → judgement row {s['row']}")
    print(f"  simulator-picked best over a grid around the model's action (subsample of {s['grid_n']} states):")
    g = s["grid"]
    print(f"    model action as is {100 * g['M']:.1f}% | best of 7 speeds {100 * g['speed7']:.1f}% | best of 31 angles {100 * g['angle31']:.1f}% | "
          f"best of 31 angles × 7 speeds {100 * g['both217']:.1f}% | made-head verify+refine (same label pair) {100 * g['verify_refine_label_pair']:.1f}%")
    print(f"    (best of 217) − (verify+refine) = {100 * (g['both217'] - g['verify_refine_label_pair']):+.1f}%p")


def print_visited(label: str, s: dict) -> None:
    g, v = s["games"], s["val"]
    print(f"\n===== {label} policy-visited states vs teacher val states (clean = pocket, no scratch; verify+refine 31x7)")
    f = "%-46s %9s %9s %9s %9s"
    print(f % ("", "1 left", "2 left", "3 left", "all"))
    gall = sum(g["by_balls"][b]["rate"] * g["by_balls"][b]["n"] for b in (1, 2, 3) if g["by_balls"][b]["n"]) / sum(g["by_balls"][b]["n"] for b in (1, 2, 3))
    print(f % ("games: clean rate", *[f"{100 * g['by_balls'][b]['rate']:.1f}%" for b in (1, 2, 3)], f"{100 * gall:.1f}%"))
    print(f % ("games: shots", *[str(g["by_balls"][b]["n"]) for b in (1, 2, 3)], str(sum(g["by_balls"][b]["n"] for b in (1, 2, 3)))))
    print(f % ("games: share of shots", *[f"{100 * g['mix'][b]:.1f}%" for b in (1, 2, 3)], ""))
    print(f % ("val (teacher states): clean rate", *[f"{100 * v['by_balls'][b]:.1f}%" for b in (1, 2, 3)], f"{100 * v['all']:.1f}%"))
    print("  by how the previous shot ended (games):  " + " | ".join(f"{k} {100 * d['rate']:.1f}% (n {d['n']})" for k, d in g["by_prev"].items()))
    print(f"  first shot + after hit {100 * g['normal']:.1f}%  vs  after miss/scratch {100 * g['after_fail']:.1f}% (n {g['n_after_fail']})")
    print("  judgement rows that hold: " + (", ".join(k for k, x in s["flags"].items() if x) or "none"))
    r = s["retry"]
    print(f"  shots after a miss or scratch: {r['n_after']}  (object balls stayed: {r['n_stayed']} = {100 * r['share_stayed']:.1f}%, moved: {r['n_moved']}, none left: {r['n_nan']})")
    print(f"    clean rate: object balls stayed {100 * r['rate_stayed']:.1f}%  |  moved {100 * r['rate_moved']:.1f}%")
    print("    object-ball move (p10/p50/p90): " + " / ".join(f"{r['obj_move_quantiles'][k]:.4f}" for k in ('10', '50', '90')) +
          "   cue-ball move (p10/p50/p90): " + " / ".join(f"{r['cue_move_quantiles'][k]:.3f}" for k in ('10', '50', '90')) + f"   → TODO 22 judgement row {r['row']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='label=glob with {seed}, e.g. L4="logs/experiments/*_L4_m2w1_*_s{seed}_*"')
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-scan", type=int, default=150, help="val states per seed for the scan")
    p.add_argument("--sample-states", type=int, default=600, help="val states per seed for the sampling test (0 = skip)")
    p.add_argument("--sample-draws", type=int, default=6)
    p.add_argument("--parts", nargs="+", choices=["base", "oracle", "visited", "speedmap", "vroracle"], default=["base"],
                   help="base = split + scan + sample; oracle = swap angle/speed for the label's; visited = games vs val states; "
                        "speedmap = speed grid at the label angle; vroracle = oracle swap on the verify+refine action")
    p.add_argument("--speedmap-states", type=int, default=0, help="val states per seed for the speed map (0 = all with a label)")
    p.add_argument("--speed-sample", action="store_true", help="models with a categorical speed head: sample the speed bin from its softmax instead of taking the most probable one")
    p.add_argument("--oracle-states", type=int, default=300, help="val states per seed for the grid oracle")
    p.add_argument("--visited-games", type=int, default=500, help="games per seed")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    raw = load_teacher_data(sorted(glob.glob(E.DATA)))
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pm = pocket_index_map(env)
    result, map_cache = {}, {}
    for label, pat in dict(a.split("=", 1) for a in args.runs).items():
        split, scan, sample, oracle, visited, speedmap, vror = [], [], [], [], [], [], []
        for sd in args.seeds:
            found = glob.glob(pat.format(seed=sd))
            if len(found) != 1:
                raise SystemExit(f"{label}: expected one run dir for seed {sd}, found {found}")
            m = E.load_run(found[0])
            if not m.has_made2 or m.made2_geom:
                raise SystemExit(f"{label}: needs a made2 head without the geometry input")
            if args.speed_sample:
                if not m.speed_bins:
                    raise SystemExit(f"{label}: --speed-sample needs a model trained with --speed-bins")
                m.speed_sample = True
            v = val_setup(raw, sd)
            if "base" in args.parts:
                split.append(measure_split(m, v, env, pm))
                scan.append(measure_scan(m, v, sd, env, pm, args.n_scan))
                if args.sample_states:
                    sample.append(measure_sample(m, v, sd, env, pm, args.sample_states, args.sample_draws))
            if "oracle" in args.parts:
                oracle.append(measure_oracle(m, v, sd, env, pm, args.oracle_states))
            if "visited" in args.parts:
                visited.append(measure_visited(m, v, env, pm, args.visited_games))
            if "speedmap" in args.parts:
                speedmap.append(measure_speedmap(m, v, sd, env, pm, map_cache, args.speedmap_states))
            if "vroracle" in args.parts:
                vror.append(measure_vr_oracle(m, v, env, pm))
            print(f"{label} seed {sd} done", flush=True)
        result[label] = {}
        if split:
            result[label] = summarize(label, split, scan, sample or None)
            print_summary(label, result[label])
        if oracle:
            result[label]["oracle"] = summarize_oracle(oracle)
            print_oracle(label, result[label]["oracle"])
        if visited:
            result[label]["visited"] = summarize_visited(visited)
            print_visited(label, result[label]["visited"])
        if speedmap:
            result[label]["speedmap"] = summarize_speedmap(speedmap)
            print_speedmap(label, result[label]["speedmap"])
        if vror:
            result[label]["vroracle"] = summarize_vr_oracle(vror)
            print_vr_oracle(label, result[label]["vroracle"])
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
