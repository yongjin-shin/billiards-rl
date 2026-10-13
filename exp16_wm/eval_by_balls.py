"""
exp16_wm/eval_by_balls.py — evaluate BC GNN runs by the number of balls left (Exp-16).

The overall mean hides that states with two balls left are the weak spot, so every
metric is split by balls left (1 / 2 / 3). Per run (rebuilt from its results.json args):

  angle   pair head angle error vs the label of its labelled pair (median, within 0.2°)
  scan    made2 head only: the label action's angle is moved ±0.6° (0.05° steps) at the label
          speed and really played; AUC / Brier of P(made) vs the true outcome, and how often
          the head's best offset lies inside the true success window (needs a made2 head)
  clean   really playing the policy's action on the val states: clean pocket rate per
          inference mode (select; with a made2 head also verify+refine on the old and on the
          dense angle × absolute speed grid)
  games   (optional) policy alone, 500 games, shots/game

Usage:
    python -m exp16_wm.eval_by_balls --runs ctrl="logs/experiments/*_m2w1_nowrap_a0.2_v0.5_s{seed}_*" \
        --seeds 0 1 2 --out logs/eval_by_balls/ctrl.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.bc import TransformedPolicy, build_split, evaluate_policy, load_teacher_data, to_unit
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import TRANSFORMS
from exp16_wm.outcomes import made_slots, pocket_index_map
from exp16_wm.teacher_data import CLEAR_ENV

DATA = "logs/teacher_data/relabel_s1000/robust/*.npz"
OFFSETS = np.round(np.arange(-0.6, 0.6 + 1e-9, 0.05), 3)                      # scan, degrees
ANG9 = (-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3)
ANG31 = tuple(np.round(np.arange(-0.3, 0.3 + 1e-9, 0.02), 4))
ABS_SPEEDS = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
INFER_CONFIGS = {                                                             # name → (mode, angles, speed ×, absolute speeds)
    "select": ("select", ANG9, (1.0,), None),
    "verify+refine old grid": ("verify_refine", ANG9, (0.8, 1.0, 1.25), None),
    "verify+refine 31 angles x 7 abs speeds": ("verify_refine", ANG31, (1.0,), ABS_SPEEDS)}


# ──────────────────────────────────────────────
# Pure helpers (tested)
# ──────────────────────────────────────────────

def balls_left(obs: np.ndarray) -> np.ndarray:
    """Raw observations (N, 23) → number of unpocketed balls (N,)."""
    return 3 - (np.asarray(obs)[:, 2:11].reshape(-1, 3, 3)[:, :, 2] > 0.5).sum(1)


def by_balls(values: np.ndarray, n_left: np.ndarray, fn=np.mean) -> dict[int, float]:
    """fn(values) for each balls-left group 1 / 2 / 3 (nan if the group is empty)."""
    return {n: (float(fn(values[n_left == n])) if np.any(n_left == n) else float("nan")) for n in (1, 2, 3)}


def wrapped_angle_error_deg(a_unit: np.ndarray, b_unit: np.ndarray) -> np.ndarray:
    """|a − b| of delta_angle in [-1, 1] units (1 = π rad), wrapped, in degrees."""
    d = (np.asarray(a_unit) - np.asarray(b_unit)) * np.pi
    return np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))


def scan_metrics(prob: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """
    prob, truth (S, K): the head's P(made) and the real outcome at K offsets of S states.
    AUC and Brier over all points; argmax_in_window over the states whose window is only part of the
    scan (some offsets succeed, some fail) = how often the head's best offset really succeeds.
    """
    from sklearn.metrics import roc_auc_score
    p, y = prob.reshape(-1), truth.reshape(-1).astype(bool)
    partial = truth.any(1) & ~truth.all(1)
    hit = truth[np.arange(len(truth)), prob.argmax(1)][partial]
    return {"auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "brier": float(np.mean((p - y) ** 2)), "argmax_in_window": float(hit.mean()) if len(hit) else float("nan"),
            "n_states_partial": int(partial.sum()), "success_rate": float(y.mean())}


# ──────────────────────────────────────────────
# Models and measurements
# ──────────────────────────────────────────────

def load_run(run_dir: str) -> PairGNNPolicy:
    """Rebuild the policy from the run's results.json args and its bc_model.pt."""
    a = json.load(open(os.path.join(run_dir, "results.json")))["args"]
    m = PairGNNPolicy(a["gnn_hidden"], factorized=a.get("selection") == "factorized",
                      scratch_head=bool(a.get("scratch_head")), made_head=bool(a.get("made_head")),
                      made2_head=bool(a.get("made2_head")), geom_angle=bool(a.get("geom_angle")),
                      made2_geom=bool(a.get("made2_geom")), enc_layers=a.get("gnn_layers", 2),
                      speed_loss=a.get("speed_loss", "mse"), speed_bins=int(a.get("speed_bins", 0)),
                      uncertainty_weights=bool(a.get("uncertainty_weights", False)),
                      aux_dim=(1 if a.get("geom_aux_set") == "aim" else 4) if float(a.get("geom_aux_weight", 0) or 0) > 0 else 0)
    m.load_state_dict(torch.load(os.path.join(run_dir, "bc_model.pt"))["gnn"])
    return m.eval()


def play(env, pm, obs: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, bool]:
    made, scratch = made_slots(env, obs, action, pm)
    return made, bool(scratch)


def scan_truth(env, pm, obs: np.ndarray, pair: int, label: np.ndarray) -> np.ndarray:
    """(K,) did the label action, moved by each offset (deg), pocket `pair` cleanly (no scratch)?"""
    out = []
    for d in OFFSETS:
        made, scr = play(env, pm, obs, np.array([label[0] + np.radians(d), label[1]], np.float32))
        out.append(bool(made[pair] and not scr))
    return np.array(out)


def evaluate_seed(runs: dict[str, str], seed: int, raw: dict, env, pm, n_scan: int, games: int) -> dict:
    """runs: label → run dir for this seed. Returns label → metrics (per-state arrays are kept for pooling)."""
    data, tr, va = build_split(dict(raw), 0.1, seed, 1.0, "none")
    vi = np.where(va)[0]
    obs = data["obs"][vi]
    rel = TRANSFORMS["relative"](obs)
    nl = balls_left(obs)
    g = data["good_pairs"][vi]
    k0 = np.argmax(g >= 0, 1)
    p0 = g[np.arange(len(vi)), k0].clip(min=0)
    labelled = (g >= 0).any(1)
    label_u = to_unit(data["good_actions"][vi][np.arange(len(vi)), k0])
    label_act = data["good_actions"][vi][np.arange(len(vi)), k0]
    pick = np.random.default_rng(seed).choice(np.where(labelled)[0], size=min(n_scan, int(labelled.sum())), replace=False)
    truth = np.stack([scan_truth(env, pm, obs[j], int(p0[j]), label_act[j]) for j in pick]) if n_scan else None
    off_u = np.stack([[to_unit(np.array([label_act[j][0] + np.radians(d), label_act[j][1]], np.float32)) for d in OFFSETS] for j in pick]).astype(np.float32) if n_scan else None
    res = {}
    for label, run_dir in runs.items():
        m = load_run(run_dir)
        r: dict = {"n_left": nl}
        with torch.no_grad():
            means = m.forward_full(torch.as_tensor(rel))["means"].numpy()
        r["angle_err"] = wrapped_angle_error_deg(means[np.arange(len(vi)), p0, 0], label_u[:, 0])
        r["labelled"] = labelled
        if m.has_made2 and n_scan:
            prob = []
            for s, j in enumerate(pick):
                with torch.no_grad():
                    o = m.forward_full(torch.as_tensor(rel[j:j + 1]).expand(len(OFFSETS), -1))
                    prob.append(torch.sigmoid(m.made2_logits(o, torch.as_tensor(off_u[s])))[:, int(p0[j])].numpy())
            r["scan"] = scan_metrics(np.stack(prob), truth)
            r["_scan_prob"] = np.stack(prob)
        r["clean"] = {}
        for name, (mode, ang, mul, ab) in INFER_CONFIGS.items():
            if mode != "select" and not m.has_made2:
                continue
            m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = mode, ang, mul, ab
            acts = np.concatenate([m.act(rel[lo:lo + 300]) for lo in range(0, len(rel), 300)])
            r["clean"][name] = np.array([(lambda x: x[0].any() and not x[1])(play(env, pm, obs[i], acts[i])) for i in range(len(vi))])
        if games:
            m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = (("verify_refine", ANG31, (1.0,), ABS_SPEEDS) if m.has_made2 else ("select", ANG9, (1.0,), None))
            ev = evaluate_policy(TransformedPolicy(m, "relative"), 0, games, CLEAR_ENV)
            r["games"] = {"mean_shots": ev["mean_shots"], "shot_success_rate": ev["shot_success_rate"]}
        res[label] = r
    res["_truth"] = truth
    return res


def summarize(per_seed: list[dict]) -> dict:
    """Pool the seeds: metrics per balls-left group."""
    out: dict = {}
    for label in [k for k in per_seed[0] if not k.startswith("_")]:
        rs = [s[label] for s in per_seed]
        nl = np.concatenate([r["n_left"] for r in rs])
        lab = np.concatenate([r["labelled"] for r in rs])
        err = np.concatenate([r["angle_err"] for r in rs])
        d = {"angle_err_median_deg": by_balls(err[lab], nl[lab], np.median), "angle_within_0.2deg": by_balls((err < 0.2)[lab], nl[lab]),
             "angle_err_median_deg_all": float(np.median(err[lab])), "angle_err_median_by_seed": [float(np.median(r["angle_err"][r["labelled"]])) for r in rs]}
        d["clean"] = {}
        for name in rs[0]["clean"]:
            ok = np.concatenate([r["clean"][name] for r in rs])
            d["clean"][name] = {**by_balls(ok, nl), "all": float(ok.mean()), "by_seed": [float(r["clean"][name].mean()) for r in rs]}
        if "scan" in rs[0]:
            P = np.concatenate([r["_scan_prob"] for r in rs]); Y = np.concatenate([s["_truth"] for s in per_seed])
            d["scan"] = scan_metrics(P, Y)
            d["scan_argmax_by_seed"] = [r["scan"]["argmax_in_window"] for r in rs]
        if "games" in rs[0]:
            sh = [r["games"]["mean_shots"] for r in rs]
            d["games"] = {"shots": sh, "mean": float(np.mean(sh)), "std": float(np.std(sh, ddof=1)) if len(sh) > 1 else float("nan"),
                          "success": float(np.mean([r["games"]["shot_success_rate"] for r in rs]))}
        out[label] = d
    return out


def print_summary(summary: dict) -> None:
    for label, d in summary.items():
        print(f"\n== {label}")
        print(f"pair head angle error median: all {d['angle_err_median_deg_all']:.3f}°  by balls left "
              + " / ".join(f"{v:.3f}°" for v in d["angle_err_median_deg"].values()) + f"   by seed {np.round(d['angle_err_median_by_seed'], 3).tolist()}")
        print("pair head within 0.2°: " + " / ".join(f"{v:.1%}" for v in d["angle_within_0.2deg"].values()))
        if "scan" in d:
            s = d["scan"]
            print(f"made head scan: AUC {s['auc']:.3f}  Brier {s['brier']:.3f}  argmax in window {s['argmax_in_window']:.1%} "
                  f"(n {s['n_states_partial']})  by seed {np.round(d['scan_argmax_by_seed'], 3).tolist()}")
        for name, c in d["clean"].items():
            print(f"clean [{name}]: " + " / ".join(f"{c[n]:.1%}" for n in (1, 2, 3)) + f"  all {c['all']:.1%}  by seed {np.round(c['by_seed'], 3).tolist()}")
        if "games" in d:
            g = d["games"]
            print(f"games: shots {np.round(g['shots'], 2).tolist()} mean {g['mean']:.2f} ± {g['std']:.2f}  success {g['success']:.1f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='label=glob with {seed}, e.g. ctrl="logs/experiments/*_m2w1_*_s{seed}_*"')
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-scan", type=int, default=150, help="val states per seed for the made-head scan (0 = skip)")
    p.add_argument("--games", type=int, default=0, help="also play this many games policy alone (0 = skip)")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    raw = load_teacher_data(sorted(glob.glob(DATA)))
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pm = pocket_index_map(env)
    specs = dict(a.split("=", 1) for a in args.runs)
    per_seed = []
    for sd in args.seeds:
        runs = {}
        for label, pat in specs.items():
            found = glob.glob(pat.format(seed=sd))
            if len(found) != 1:
                raise SystemExit(f"{label}: expected one run dir for seed {sd}, found {found}")
            runs[label] = found[0]
        per_seed.append(evaluate_seed(runs, sd, raw, env, pm, args.n_scan, args.games))
        print(f"seed {sd} done", flush=True)
    summary = summarize(per_seed)
    print_summary(summary)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
