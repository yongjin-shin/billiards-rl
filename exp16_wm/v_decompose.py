"""
exp16_wm/v_decompose.py — V6-B: where does a V's same-origin order accuracy come from?

No training, no simulation: reads the V5 test set (exp16_wm.v_testset) and the V5-c perturbation result and, for every pair of candidates
of the same origin whose later ball counts (k_next) differ, records how each predictor orders it. Pairs are split into
  Z  one candidate's next position misses at once (k = 0), the other does not   → "does the next shot go in"
  P  both have k ≥ 1                                                           → "how long does the chain run"
and by the true gap |Δk| = 1 / 2 / ≥ 3. A predictor's credit for a pair is 1 if it orders it like the truth, ½ for a tie, else 0.

Usage:
    python -m exp16_wm.v_decompose --testset logs/teacher_data/v5/testset_s950.npz --ceiling logs/teacher_data/v5/order_ceiling_s950.npz \
        --runs "<PRm run dir glob>" --v-runs logs/experiments_v2/v6a_ctrl logs/experiments_v2/v5_A --out logs/teacher_data/v5/decompose_s950.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUCKETS = ("all", "Z", "P", "gap1", "gap2", "gap3+")


def pair_table(t: dict[str, np.ndarray], preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """t: the saved test set (k_next, succ, cleared, valid; (N, K)); preds: name → (N, K) predicted value of each candidate's next position.
    One record per same-origin pair of rankable candidates with different k_next: kmin, kmax, and per predictor name the credit
    ("credit:name" ∈ {0, ½, 1}) and the sign of its ordering relative to the truth ("sign:name": +1 right, −1 wrong, 0 tie)."""
    use = t["valid"] & t["succ"] & ~t["cleared"]
    rec: dict[str, list[float]] = {"kmin": [], "kmax": []}
    for n in preds:
        rec[f"sign:{n}"] = []
    for r in range(len(use)):
        idx = np.nonzero(use[r])[0]
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        a, b = idx[a], idx[b]
        k = t["k_next"][r]
        keep = k[a] != k[b]
        a, b = a[keep], b[keep]
        hi, lo = np.where(k[a] > k[b], a, b), np.where(k[a] > k[b], b, a)
        rec["kmin"].extend(k[lo]); rec["kmax"].extend(k[hi])
        for n, p in preds.items():
            rec[f"sign:{n}"].extend(np.sign(p[r][hi] - p[r][lo]))
    out = {k: np.asarray(v) for k, v in rec.items()}
    for n in preds:
        out[f"credit:{n}"] = (out[f"sign:{n}"] + 1.0) / 2.0              # +1 → 1, tie → ½, −1 → 0
    out["gap"] = out["kmax"] - out["kmin"]
    return out


def bucket_masks(rec: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    gap = rec["gap"]
    return {"all": np.ones(len(gap), bool), "Z": rec["kmin"] == 0, "P": rec["kmin"] >= 1,
            "gap1": gap == 1, "gap2": gap == 2, "gap3+": gap >= 3}


def summarize(rec: dict[str, np.ndarray], names: list[str], ref: str) -> dict[str, dict[str, float]]:
    """Per bucket: n pairs, each predictor's mean credit, and for every non-ref predictor its agreement with ref (share of pairs, both untied,
    ordered the same way) and its accuracy on the pairs where the two disagree."""
    out: dict[str, dict[str, float]] = {}
    for b, m in bucket_masks(rec).items():
        row: dict[str, float] = {"n": float(m.sum())}
        for n in names:
            row[n] = float(rec[f"credit:{n}"][m].mean()) if m.any() else float("nan")
        for n in names:
            if n == ref:
                continue
            both = m & (rec[f"sign:{n}"] != 0) & (rec[f"sign:{ref}"] != 0)
            same = rec[f"sign:{n}"] == rec[f"sign:{ref}"]
            row[f"agree:{n}"] = float(same[both].mean()) if both.any() else float("nan")
            dis = both & ~same
            row[f"acc_when_disagree:{n}"] = float((rec[f"sign:{n}"][dis] > 0).mean()) if dis.any() else float("nan")
        out[b] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", required=True); p.add_argument("--ceiling", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="made run globs (for the made-max baseline)")
    p.add_argument("--v-runs", nargs="+", required=True); p.add_argument("--out", default=""); p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from exp16_wm.paired_eval import find_run
    from exp16_wm.v_decision_ceiling import ROW_KEYS, next_value_sources
    from exp16_wm.v_testset import made_max_values, next_values
    z = np.load(args.testset, allow_pickle=True)
    t = {k: z[k] for k in (*ROW_KEYS, "next_obs", "left_next")}
    kp = np.load(args.ceiling, allow_pickle=True)["k_pert"]
    preds = {"made max": made_max_values(t, [find_run(g) for g in args.runs])}
    for r in args.v_runs:
        preds[os.path.basename(r.rstrip("/"))] = next_values(t, find_run(r))
    preds["ceiling (1cm, 8 copies)"] = next_value_sources(t, kp[:, 0, :])["ceiling"]
    rec = pair_table(t, preds)
    res = summarize(rec, list(preds), "made max")
    names = list(preds)
    for b, row in res.items():
        print(f"\n== {b}: {int(row['n'])} pairs")
        for n in names:
            extra = "" if n == "made max" or n.startswith("ceiling") else f"   agree with made max {row[f'agree:{n}']:.3f}   acc when they disagree {row[f'acc_when_disagree:{n}']:.3f}"
            print(f"  {n[:30]:30s} {row[n]:.3f}{extra}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"testset": args.testset, "ceiling": args.ceiling, "v_runs": args.v_runs, "result": res}, open(args.out, "w"), indent=1)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
