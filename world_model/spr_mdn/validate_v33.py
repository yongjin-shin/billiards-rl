"""
world_model/spr_mdn/validate_v33.py — V33 세 가지 추론 방법 비교

1. segment_gt   : GT seg_type으로 segment별 rollout   (학습 프로토콜, upper bound)
2. chain_oracle : 60스텝 rollout, GT bounce 타이밍/타입으로 z 재초기화
3. chain_pred   : 60스텝 rollout, TypeHead 예측으로 z 재초기화 (실제 inference)

Usage:
    python world_model/spr_mdn/validate_v33.py \
        --ckpt world_model/results/spr_mdn_v33_segment \
        --data-dir world_model/data_fixeddt
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.train_v33_segment import V33Model
from world_model.spr_mdn.spr_dataset import SPRDataset, SegmentDataset, make_balanced_val_eps
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H

STEPS      = 60
TYPE_NAMES = ["cue_strike", "bb", "lin_cush", "circ_cush", "pocket"]


def pos_err_cm(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(T, 14) → (T,) 평균 위치 오차 [cm]"""
    ce = np.sqrt(((pred[:,0]-gt[:,0])*TABLE_W)**2 + ((pred[:,1]-gt[:,1])*TABLE_H)**2)
    te = np.sqrt(((pred[:,7]-gt[:,7])*TABLE_W)**2 + ((pred[:,8]-gt[:,8])*TABLE_H)**2)
    return (ce + te) / 2 * 100


# ── 1. Segment GT eval ────────────────────────────────────────────────────────
def eval_segment_gt(model: V33Model, episodes: list, device: str) -> dict:
    model.eval()
    ds = SegmentDataset(episodes, min_len=2, augment=False)
    errs_all, errs_by_type = [], {i: [] for i in range(5)}

    with torch.no_grad():
        for seg_s, _, seg_type_start, seg_len in ds:
            L  = int(seg_len)
            st = seg_type_start.unsqueeze(0).to(device)
            z  = model.encoder(seg_s[0:1].to(device), st)
            sc = seg_s[0:1].to(device)
            preds = []
            for _ in range(L):
                z = model.transition(z, sc); sc = model.mu_head(z)
                preds.append(sc[0].cpu().numpy())
            pr = np.stack(preds)
            gt = seg_s[1:L+1].numpy()
            e  = pos_err_cm(pr, gt).mean()
            errs_all.append(e)
            errs_by_type[int(seg_type_start)].append(e)

    def _m(l): return float(np.mean(l)) if l else float("nan")
    return {"mean": _m(errs_all),
            **{TYPE_NAMES[i]: _m(errs_by_type[i]) for i in range(5)}}


# ── 2. Chain Oracle eval ──────────────────────────────────────────────────────
def eval_chain_oracle(model: V33Model, episodes: list, device: str) -> dict:
    """GT bounce 타이밍/타입으로 z 재초기화 — chaining 이론 상한."""
    model.eval()
    valid = [(ep_s, ep_f, ep_t) for ep_s, ep_f, ep_t, _, _ in episodes
             if len(ep_s) >= STEPS + 1]
    errs_all, errs_bb, errs_nbb = [], [], []

    with torch.no_grad():
        for ep_s, ep_f, ep_t in valid:
            s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
            st = torch.zeros(1, dtype=torch.long, device=device)
            z  = model.encoder(s0, st)
            sc = s0
            preds = []
            for t in range(STEPS):
                z     = model.transition(z, sc)
                s_hat = model.mu_head(z)
                preds.append(s_hat[0].cpu().numpy())
                sc = s_hat
                # GT bounce → z 재초기화
                if ep_f[t]:
                    gt_type = torch.tensor([int(ep_t[t])], dtype=torch.long, device=device)
                    z = model.encoder(sc, gt_type)

            pr = np.stack(preds)
            gt = ep_s[1:STEPS+1]
            e  = pos_err_cm(pr, gt)
            errs_all.append(e.mean())
            has_bb = bool(np.any(ep_f[:STEPS] & (ep_t[:STEPS] == 1)))
            (errs_bb if has_bb else errs_nbb).append(e.mean())

    def _m(l): return float(np.mean(l)) if l else float("nan")
    return {"mean": _m(errs_all), "bb": _m(errs_bb), "nbb": _m(errs_nbb)}


# ── 3. Chain Pred eval ────────────────────────────────────────────────────────
def eval_chain_pred(model: V33Model, episodes: list, device: str) -> dict:
    """TypeHead 예측으로 z 재초기화 — 실제 inference."""
    model.eval()
    valid = [(ep_s, ep_f, ep_t) for ep_s, ep_f, ep_t, _, _ in episodes
             if len(ep_s) >= STEPS + 1]
    errs_all, errs_bb, errs_nbb = [], [], []

    # TypeHead 정확도 추적
    tp_bounce, fp_bounce, fn_bounce = 0, 0, 0

    with torch.no_grad():
        for ep_s, ep_f, ep_t in valid:
            s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
            st = torch.zeros(1, dtype=torch.long, device=device)
            z  = model.encoder(s0, st)
            sc = s0
            preds = []
            for t in range(STEPS):
                z       = model.transition(z, sc)
                s_hat   = model.mu_head(z)
                t_logit = model.type_head(z)
                type_pred = int(t_logit.argmax(-1).item())
                preds.append(s_hat[0].cpu().numpy())
                sc = s_hat

                is_gt_bounce   = bool(ep_f[t]) if t < len(ep_f) else False
                is_pred_bounce = type_pred != 0

                if is_gt_bounce and is_pred_bounce:
                    tp_bounce += 1
                elif not is_gt_bounce and is_pred_bounce:
                    fp_bounce += 1
                elif is_gt_bounce and not is_pred_bounce:
                    fn_bounce += 1

                if is_pred_bounce:
                    gt_type_tensor = torch.tensor([type_pred], dtype=torch.long, device=device)
                    z = model.encoder(sc, gt_type_tensor)

            pr = np.stack(preds)
            gt = ep_s[1:STEPS+1]
            e  = pos_err_cm(pr, gt)
            errs_all.append(e.mean())
            has_bb = bool(np.any(ep_f[:STEPS] & (ep_t[:STEPS] == 1)))
            (errs_bb if has_bb else errs_nbb).append(e.mean())

    total_gt = tp_bounce + fn_bounce
    total_pred = tp_bounce + fp_bounce
    prec = tp_bounce / total_pred if total_pred > 0 else 0.0
    rec  = tp_bounce / total_gt   if total_gt   > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec)  if prec+rec   > 0 else 0.0

    def _m(l): return float(np.mean(l)) if l else float("nan")
    return {
        "mean": _m(errs_all), "bb": _m(errs_bb), "nbb": _m(errs_nbb),
        "bounce_precision": prec, "bounce_recall": rec, "bounce_f1": f1,
        "tp": tp_bounce, "fp": fp_bounce, "fn": fn_bounce,
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default="world_model/results/spr_mdn_v33_segment")
    p.add_argument("--data-dir",  default="world_model/data_fixeddt")
    p.add_argument("--n-each",    type=int, default=250)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(Path(args.ckpt) / "best.pt", map_location=device, weights_only=False)
    model = V33Model().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Model: ep{ckpt['epoch']}  saved_err={ckpt['mean_err']:.2f}cm\n")

    dataset = SPRDataset(args.data_dir)
    rng     = np.random.default_rng(0)
    perm    = rng.permutation(len(dataset.episodes))
    n_val   = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all  = [dataset.episodes[i] for i in perm[:n_val]]
    balanced_val = make_balanced_val_eps(val_eps_all, n_each=args.n_each, seed=0)

    print("Running eval_segment_gt  ...")
    r_seg = eval_segment_gt(model, balanced_val, device)

    print("Running eval_chain_oracle ...")
    r_ora = eval_chain_oracle(model, balanced_val, device)

    print("Running eval_chain_pred  ...")
    r_prd = eval_chain_pred(model, balanced_val, device)

    # ── 결과 출력 ───────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print(f"{'Method':<20} {'mean':>8} {'bb':>8} {'nbb':>8}")
    print("-"*62)
    print(f"{'segment_gt':<20} {r_seg['mean']:>7.2f}cm  {'N/A':>6}   {'N/A':>6}")
    print(f"{'chain_oracle':<20} {r_ora['mean']:>7.2f}cm  {r_ora['bb']:>6.2f}cm  {r_ora['nbb']:>6.2f}cm")
    print(f"{'chain_pred':<20} {r_prd['mean']:>7.2f}cm  {r_prd['bb']:>6.2f}cm  {r_prd['nbb']:>6.2f}cm")
    print("="*62)

    print(f"\nSegment GT — per type:")
    for t in TYPE_NAMES:
        print(f"  {t:12s}: {r_seg[t]:.2f}cm")

    print(f"\nTypeHead bounce detection (over {STEPS}-step rollouts):")
    print(f"  TP={r_prd['tp']}  FP={r_prd['fp']}  FN={r_prd['fn']}")
    print(f"  Precision={r_prd['bounce_precision']:.3f}  "
          f"Recall={r_prd['bounce_recall']:.3f}  "
          f"F1={r_prd['bounce_f1']:.3f}")

    degradation = r_prd["mean"] - r_seg["mean"]
    oracle_gap  = r_ora["mean"] - r_seg["mean"]
    pred_gap    = r_prd["mean"] - r_ora["mean"]
    print(f"\nGap analysis:")
    print(f"  oracle  vs segment_gt : +{oracle_gap:.2f}cm  (error accumulation between bounces)")
    print(f"  pred    vs oracle     : +{pred_gap:.2f}cm   (TypeHead mis-detection cost)")
    print(f"  pred    vs segment_gt : +{degradation:.2f}cm  (total chaining degradation)")


if __name__ == "__main__":
    main()
