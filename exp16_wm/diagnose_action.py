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

Usage:
    python -m exp16_wm.diagnose_action --runs L4="logs/experiments/*_L4_m2w1_*_s{seed}_*" --seeds 0 1 2 \
        --out logs/diagnose_action/L4.json
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


# ──────────────────────────────────────────────
# Model-side helpers
# ──────────────────────────────────────────────

def made_logit(m: PairGNNPolicy, o: dict, k: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """cand (B, C, 2) candidate actions for pair k[b] → (B, C) made2 logits (the inputs predict_unit's refine uses)."""
    rows = torch.arange(len(k)); C = cand.shape[1]
    a = o["means"][rows, k]
    e_k = o["e"][rows, k][:, None].expand(-1, C, -1)
    feats = [e_k, cand, action_diff(cand, a[:, None].expand(-1, C, -1)) * m.made2_scale]
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


def val_setup(raw: dict, seed: int) -> dict:
    """Val states of the run's split with the label pair (candidate 0) and its action."""
    data, _, va = build_split(dict(raw), 0.1, seed, 1.0, "none")
    vi = np.where(va)[0]
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


# ──────────────────────────────────────────────
# Pooling and printing
# ──────────────────────────────────────────────

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='label=glob with {seed}, e.g. L4="logs/experiments/*_L4_m2w1_*_s{seed}_*"')
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-scan", type=int, default=150, help="val states per seed for the scan")
    p.add_argument("--sample-states", type=int, default=600, help="val states per seed for the sampling test (0 = skip)")
    p.add_argument("--sample-draws", type=int, default=6)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    raw = load_teacher_data(sorted(glob.glob(E.DATA)))
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pm = pocket_index_map(env)
    result = {}
    for label, pat in dict(a.split("=", 1) for a in args.runs).items():
        split, scan, sample = [], [], []
        for sd in args.seeds:
            found = glob.glob(pat.format(seed=sd))
            if len(found) != 1:
                raise SystemExit(f"{label}: expected one run dir for seed {sd}, found {found}")
            m = E.load_run(found[0])
            if not m.has_made2 or m.made2_geom:
                raise SystemExit(f"{label}: needs a made2 head without the geometry input")
            v = val_setup(raw, sd)
            split.append(measure_split(m, v, env, pm))
            scan.append(measure_scan(m, v, sd, env, pm, args.n_scan))
            if args.sample_states:
                sample.append(measure_sample(m, v, sd, env, pm, args.sample_states, args.sample_draws))
            print(f"{label} seed {sd} done", flush=True)
        result[label] = summarize(label, split, scan, sample or None)
        print_summary(label, result[label])
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
