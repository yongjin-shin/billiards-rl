"""
exp16_wm/label_gap.py — how far is the trained policy's action from the teacher's label?

For a finished BC run (exp16_wm/bc.py), reload its model and the exact validation states
(same game-level split) and compare the policy's (angle, speed) with the label, per
dimension, and in the simulator:

  angle / speed error vs the teacher's chosen action, and vs the nearest stored good
  candidate (the one nearest_good trains toward — in the loss's physical units)
  pocket rate of: the teacher's action, the policy's action, and two hybrids
    label angle + policy speed   → how much the speed error alone costs
    policy angle + label speed   → how much the angle error alone costs

Usage:
    python -m exp16_wm.label_gap --runs <run_dir> [<run_dir> ...] [--limit 800]
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
from exp16_wm.bc import load_teacher_data, build_split, dim_weights, to_unit, as_model
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import TRANSFORMS
from exp16_wm.planner import wrap_angle, aim_reference, simulate_candidate
from exp16_wm.relabel import restore_state
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.train import ACT_LOW, ACT_HIGH


def unit_to_env(u: np.ndarray) -> np.ndarray:
    return u * (ACT_HIGH - ACT_LOW) / 2.0 + (ACT_HIGH + ACT_LOW) / 2.0


def nearest_good_idx(pred_env: np.ndarray, good_env: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Per row, index of the stored good candidate nearest to the prediction in the loss's
    weighted unit space; -1 when a row has no candidate. pred (N,2), good (N,K,2) NaN-padded.
    """
    pu, gu = to_unit(pred_env), to_unit(good_env)
    d = (((pu[:, None, :] - np.nan_to_num(gu)) * w) ** 2).sum(-1)
    d = np.where(np.isnan(good_env[..., 0]), np.inf, d)
    idx = d.argmin(axis=1)
    return np.where(np.isfinite(d.min(axis=1)), idx, -1)


def err_stats(angle_err_deg: np.ndarray, speed_err: np.ndarray, speed_label: np.ndarray) -> dict:
    """Median / p90 and tolerance rates of the angle (deg) and speed (m/s, relative) errors."""
    rel = speed_err / np.maximum(speed_label, 1e-6)
    return {
        "angle_median_deg": float(np.median(angle_err_deg)),
        "angle_p90_deg": float(np.percentile(angle_err_deg, 90)),
        "angle_within_0.2": float(np.mean(angle_err_deg < 0.2)),
        "angle_within_1.0": float(np.mean(angle_err_deg < 1.0)),
        "speed_median_ms": float(np.median(speed_err)),
        "speed_p90_ms": float(np.percentile(speed_err, 90)),
        "speed_rel_median": float(np.median(rel)),
        "speed_within_20pct": float(np.mean(rel < 0.2)),
    }


def clean_pocket(env: BilliardsEnv, raw_obs: np.ndarray, action: np.ndarray) -> bool:
    restore_state(env, raw_obs)
    n, scratch, _ = simulate_candidate(env, aim_reference(env), action.astype(np.float32))
    return bool(n >= 1 and not scratch)


def pick_rows(mask: np.ndarray, limit: int, split: str, seed: int = 0) -> np.ndarray:
    """
    Row indices of a boolean split mask. With a limit: the first N for val (as before),
    a seeded random sample for train (the first N train rows are only the first games).
    """
    idx = np.where(mask)[0]
    if not limit or limit >= len(idx):
        return idx
    if split == "train":
        return np.sort(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
    return idx[:limit]


def analyse_run(run_dir: str, limit: int = 0, split: str = "val") -> dict:
    res = json.load(open(os.path.join(run_dir, "results.json")))
    a = res["args"]
    assert a["model"] == "gnn", "label_gap currently reloads the GNN policy only"
    paths = sorted({p for pat in a["data"] for p in glob.glob(pat)})
    assert split in ("val", "train")
    data, train, val = build_split(load_teacher_data(paths), a["val_frac"], a["seed"], a.get("data_frac", 1.0), "none")
    idx = pick_rows(val if split == "val" else train, limit, split)
    raw = data["obs"][idx]
    model = PairGNNPolicy(a["gnn_hidden"], a["wta_eps"], a["sel_weight"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "bc_model.pt"))["gnn"])
    model.eval()
    with torch.no_grad():
        pu = as_model(model, "nearest_good").predict_unit(torch.as_tensor(TRANSFORMS[a["obs_transform"]](raw))).numpy()
    pred = unit_to_env(pu)
    label, good = data["action"][idx], data["good_actions"][idx]

    def stats(target: np.ndarray, rows: np.ndarray) -> dict:
        ang = np.degrees(np.abs([wrap_angle(float(p - t)) for p, t in zip(pred[rows, 0], target[rows, 0])]))
        return err_stats(ang, np.abs(pred[rows, 1] - target[rows, 1]), target[rows, 1])

    allrows = np.arange(len(idx))
    out = {"run": os.path.basename(run_dir), "label_set": a["data"][0].split("/")[-2], "split": split, "n_val": int(len(idx)),
           "vs_teacher_action": stats(label, allrows)}
    ng = nearest_good_idx(pred, good, dim_weights(0.2, 0.5).numpy())
    has = ng >= 0
    near = np.where(has[:, None], good[allrows, np.maximum(ng, 0)], label)
    out["vs_nearest_good"] = stats(near, allrows)
    out["frac_rows_with_candidates"] = float(has.mean())
    out["speed_median"] = {"label": float(np.median(label[:, 1])), "policy": float(np.median(pred[:, 1]))}

    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    hy_speed = np.stack([label[:, 0], pred[:, 1]], axis=1)      # label angle + policy speed
    hy_angle = np.stack([pred[:, 0], label[:, 1]], axis=1)      # policy angle + label speed
    pocket = {"teacher": [], "policy": [], "label_angle_policy_speed": [], "policy_angle_label_speed": []}
    for i, r in enumerate(raw):
        pocket["teacher"].append(clean_pocket(env, r, label[i]))
        pocket["policy"].append(clean_pocket(env, r, pred[i]))
        pocket["label_angle_policy_speed"].append(clean_pocket(env, r, hy_speed[i]))
        pocket["policy_angle_label_speed"].append(clean_pocket(env, r, hy_angle[i]))
    env.close()
    out["pocket_rate"] = {k: float(np.mean(v)) for k, v in pocket.items()}
    return out


def print_report(results: list[dict]) -> None:
    for r in results:
        print(f"\n== {r['run']}  (labels: {r['label_set']}, {r['n_val']} {r.get('split', 'val')} states)")
        for name in ("vs_teacher_action", "vs_nearest_good"):
            s = r[name]
            print(f"  {name:<18} angle: median {s['angle_median_deg']:.3f}°  p90 {s['angle_p90_deg']:.2f}°  "
                  f"<0.2° {s['angle_within_0.2']*100:4.1f}%  <1° {s['angle_within_1.0']*100:4.1f}%  |  "
                  f"speed: median {s['speed_median_ms']:.2f} m/s ({s['speed_rel_median']*100:.0f}%)  "
                  f"p90 {s['speed_p90_ms']:.2f}  within 20% {s['speed_within_20pct']*100:4.1f}%")
        sp = r["speed_median"]; pr = r["pocket_rate"]
        print(f"  speed median: label {sp['label']:.2f}  policy {sp['policy']:.2f} m/s   (rows with candidates {r['frac_rows_with_candidates']*100:.0f}%)")
        print(f"  clean-pocket rate in the simulator: teacher {pr['teacher']*100:.1f}%  policy {pr['policy']*100:.1f}%  |  "
              f"label angle + policy speed {pr['label_angle_policy_speed']*100:.1f}%  |  "
              f"policy angle + label speed {pr['policy_angle_label_speed']*100:.1f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, nargs="+", required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "train"])
    p.add_argument("--limit", type=int, default=0, help="only N states (val: the first N; train: a seeded sample)")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    results = [analyse_run(r, args.limit, args.split) for r in args.runs]
    print_report(results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
