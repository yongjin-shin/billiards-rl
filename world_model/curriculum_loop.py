"""
world_model/curriculum_loop.py — Curriculum Learning Pipeline

Stage 0: SAC 데이터, first_ball_ball 필터 → 직접 타격 케이스만 학습
Stage 1: SAC + random, any_ball_ball 필터 → bank shot 포함
Stage 2: Stage1 + 신규 생성 데이터 (min_ball_ball=1)

Usage:
    python world_model/curriculum_loop.py \\
        --sac-models path1 path2 path3 \\
        --data-dir world_model/data_v3 \\
        --out-dir world_model/results/curriculum_<ts>
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
from copy import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.generate_data_v3 import (
    generate, passes_filter, BALL_BALL_IDX, COLLISION_IDXS, EVENT_DIM_V3,
    MAX_AVEL, MAX_SPEED_V3,
)
from world_model.wm_predictor import EVENT_TYPES, TABLE_W, TABLE_H, MAX_EVENTS
from world_model.markov_predictor import (
    MarkovEncoder, MarkovTransition,
    S_TYPE_OH, transition_loss, encoder_loss,
)
from world_model.train_markov import (
    load_data, TransitionDataset, EncoderDataset,
    train_transition, train_encoder, COLLISION_TYPES,
)

# ── 커리큘럼 단계 정의 ─────────────────────────────────────────────────────────

# (name, filter_mode, trans_epochs, enc_epochs, max_per_class)
STAGES = [
    ("stage0_direct",  "first_ball_ball", 200, 100, 10000),
    ("stage1_bankshot", "any_ball_ball",   75,  40,  None),
    ("stage2_expanded", "any_ball_ball",   50,  25,  None),
]

ADVANCE_THRESHOLD_DEFAULT = 0.72   # ball_ball accuracy


# ── 에피소드 필터 ──────────────────────────────────────────────────────────────

def filter_episodes(data: dict, mode: str) -> np.ndarray:
    """
    mode별로 에피소드 인덱스 마스크 반환.
      'first_ball_ball' : 첫 충돌 이벤트 = ball_ball
      'any_ball_ball'   : 에피소드 내 ball_ball ≥ 1
      'all'             : 전체
    """
    events  = data["events"]   # (N, T, 24)
    lengths = data["lengths"]  # (N,)
    N = len(lengths)

    if mode == "all":
        return np.ones(N, dtype=bool)

    all_types = events[:, :, 14:24].argmax(axis=2)  # (N, T)
    is_coll   = np.isin(all_types, list(COLLISION_IDXS))

    mask = np.zeros(N, dtype=bool)
    for i, L in enumerate(lengths):
        coll_ts = np.where(is_coll[i, :int(L)])[0]
        if len(coll_ts) == 0:
            continue
        if mode == "first_ball_ball":
            if all_types[i, coll_ts[0]] == BALL_BALL_IDX:
                mask[i] = True
        elif mode == "any_ball_ball":
            if BALL_BALL_IDX in all_types[i, coll_ts]:
                mask[i] = True
    return mask


# ── 모델 평가 ─────────────────────────────────────────────────────────────────

def evaluate_per_class_acc(ckpt_dir: Path, data: dict,
                           cfg: dict, n_eval: int = 3000,
                           device: str = "cpu") -> dict:
    """transition_best.pt 로드 → per-class accuracy 계산."""
    trans = MarkovTransition(tuple(cfg["trans_hidden"]), cfg["embed_dim"]).to(device)
    ck = torch.load(ckpt_dir / "transition_best.pt", weights_only=False)
    trans.load_state_dict(ck["state"])
    trans.eval()

    events  = data["events"]
    lengths = data["lengths"]
    all_types = events[:, :, 14:24].argmax(axis=2)
    is_coll   = np.isin(all_types, list(COLLISION_TYPES))

    N = len(lengths)
    eval_idx = np.random.default_rng(0).choice(N, min(n_eval, N), replace=False)

    correct = {c: 0 for c in [2, 3, 4, 5]}
    total   = {c: 0 for c in [2, 3, 4, 5]}

    with torch.no_grad():
        for i in eval_idx:
            L = int(lengths[i])
            coll_ts = np.where(is_coll[i, :L])[0]
            for k in range(len(coll_ts) - 1):
                t, t1 = coll_ts[k], coll_ts[k + 1]
                s = torch.from_numpy(events[i, t]).float().unsqueeze(0).to(device)
                logits, _, _ = trans(s)
                pred = logits.argmax(-1).item()
                gt   = int(all_types[i, t1])
                total[gt]   = total.get(gt, 0) + 1
                correct[gt] = correct.get(gt, 0) + (pred == gt)

    acc = {}
    for cls in [2, 3, 4, 5]:
        n = total[cls]
        acc[EVENT_TYPES[cls]] = correct[cls] / n if n > 0 else 0.0
    return acc


# ── Stage 2용 신규 데이터 생성 ────────────────────────────────────────────────

def generate_new_data(sac_models: list[str], out_dir: Path,
                      n_episodes: int, seed: int = 99) -> dict:
    """SAC 모델들로 min_ball_ball=1 필터 데이터 생성 → npz 저장."""
    from simulator import BilliardsEnv
    from stable_baselines3 import SAC

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else []

    rng = np.random.default_rng(seed)
    n_per_model = (n_episodes + len(sac_models) - 1) // len(sac_models)

    all_arrays: dict = {}
    total_saved = 0

    for model_path in sac_models:
        print(f"  Generating {n_per_model} ep from {Path(model_path).parent.parent.name}")
        sb3 = SAC.load(model_path)
        env = BilliardsEnv(n_balls=1)
        policy_fn = lambda obs: sb3.predict(obs, deterministic=True)[0]

        d = generate(env, policy_fn, n_per_model, rng,
                     min_ball_ball=1, first_ball_ball=False)
        env.close()

        for k, v in d.items():
            all_arrays.setdefault(k, []).append(v)

        total_saved += n_per_model

    merged = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"active_v3_{ts}.npz"
    np.savez_compressed(out_dir / fname, **merged)

    pocket_rate = merged["pocketed"].mean() * 100
    meta.append({
        "file": fname, "tag": "active_v3",
        "n_episodes": total_saved,
        "pocket_rate": round(pocket_rate, 2),
        "avg_length": round(float(merged["lengths"].mean()), 2),
        "created_at": ts, "format": "v3", "event_dim": EVENT_DIM_V3,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {total_saved} episodes → {out_dir / fname}")
    return merged


# ── Stage 학습 ────────────────────────────────────────────────────────────────

def run_stage(stage_idx: int, stage_data: dict, args: SimpleNamespace,
              device: str, prev_ckpt: Path | None) -> Path:
    """하나의 커리큘럼 단계 학습."""
    name, filter_mode, trans_ep, enc_ep, max_per_class = STAGES[stage_idx]
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 에피소드 필터 적용 ─────────────────────────────────────────────────────
    ep_mask = filter_episodes(stage_data, filter_mode)
    n_total = len(ep_mask)
    n_keep  = int(ep_mask.sum())
    print(f"  Episodes: {n_keep}/{n_total}  ({n_keep/n_total*100:.1f}%)")
    if n_keep == 0:
        raise RuntimeError(f"Stage {stage_idx}: no episodes after filter '{filter_mode}'")

    filtered = {k: v[ep_mask] for k, v in stage_data.items()}

    # ── train/val split ────────────────────────────────────────────────────────
    rng   = np.random.default_rng(args.seed)
    perm  = rng.permutation(n_keep)
    n_val = max(1, int(n_keep * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # ── stage args (epoch 수 override) ────────────────────────────────────────
    st_args = copy(args)
    st_args.trans_epochs = trans_ep
    st_args.enc_epochs   = enc_ep
    st_args.out_dir      = str(out_dir)

    # ── config 저장 ───────────────────────────────────────────────────────────
    cfg = vars(args).copy()
    cfg.update({"stage": stage_idx, "filter": filter_mode,
                "trans_epochs": trans_ep, "enc_epochs": enc_ep,
                "n_filtered": n_keep})
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Transition 학습 ───────────────────────────────────────────────────────
    print(f"\n  ── Transition (epochs={trans_ep}) ──")
    tr_ds  = TransitionDataset(filtered, train_idx, augment=True,
                               max_per_class=max_per_class)
    val_ds = TransitionDataset(filtered, val_idx,   augment=False)
    print(f"  Pairs: train={len(tr_ds)}  val={len(val_ds)}")

    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            num_workers=0, pin_memory=False)

    trans = MarkovTransition(tuple(args.trans_hidden), args.embed_dim).to(device)
    if prev_ckpt is not None:
        ck = torch.load(prev_ckpt / "transition_best.pt", weights_only=False)
        trans.load_state_dict(ck["state"])
        print(f"  [Trans] resumed from {prev_ckpt}")
    train_transition(trans, tr_loader, val_loader, st_args, device,
                     class_weights=tr_ds.class_weights)

    # ── Encoder 학습 ──────────────────────────────────────────────────────────
    print(f"\n  ── Encoder (epochs={enc_ep}) ──")
    enc_tr  = EncoderDataset(filtered, train_idx)
    enc_val = EncoderDataset(filtered, val_idx)
    print(f"  Samples: train={len(enc_tr)}  val={len(enc_val)}")

    enc_tr_loader  = DataLoader(enc_tr,  batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)
    enc_val_loader = DataLoader(enc_val, batch_size=args.batch_size * 2,
                                num_workers=0, pin_memory=False)

    enc = MarkovEncoder(tuple(args.enc_hidden)).to(device)
    if prev_ckpt is not None:
        ck = torch.load(prev_ckpt / "encoder_best.pt", weights_only=False)
        enc.load_state_dict(ck["state"])
        print(f"  [Enc]   resumed from {prev_ckpt}")
    train_encoder(enc, enc_tr_loader, enc_val_loader, st_args, device)

    return out_dir


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], csv_path: Path):
    header = ["stage", "filter", "n_episodes",
              "ball_ball", "linear_cushion", "circular_cushion", "ball_pocket"]
    print(f"\n{'='*70}")
    print(f"{'Stage':10s} {'Filter':18s} {'N':>7s}  "
          f"{'ball_ball':>9s} {'lin_cush':>9s} {'circ_cush':>9s} {'pocket':>9s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:10s} {r['filter']:18s} {r['n_episodes']:>7d}  "
              f"{r.get('ball_ball', 0):>9.3f} "
              f"{r.get('ball_linear_cushion', 0):>9.3f} "
              f"{r.get('ball_circular_cushion', 0):>9.3f} "
              f"{r.get('ball_pocket', 0):>9.3f}")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"\nCSV → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",          type=str, default="world_model/data_v3")
    p.add_argument("--new-data-dir",      type=str, default="world_model/data_v3_cur")
    p.add_argument("--sac-models",        nargs="*", default=[])
    p.add_argument("--out-dir",           type=str, default=None)
    p.add_argument("--max-stage",         type=int, default=2,
                   help="최대 진행 stage (0=Stage0만, 1=0+1, 2=전체)")
    p.add_argument("--advance-threshold", type=float, default=ADVANCE_THRESHOLD_DEFAULT)
    p.add_argument("--new-episodes",      type=int, default=15000,
                   help="Stage2 신규 생성 에피소드 수")
    # 학습 하이퍼파라미터
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--val-frac",     type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--trans-hidden", nargs="+",  type=int, default=[256, 512, 256])
    p.add_argument("--enc-hidden",   nargs="+",  type=int, default=[128, 256, 256])
    p.add_argument("--embed-dim",    type=int,   default=32)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--n-eval",       type=int,   default=3000)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        args.out_dir = f"world_model/results/curriculum_{ts}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── 기존 데이터 로드 ──────────────────────────────────────────────────────
    print("Loading base data...")
    base_data = load_data(args.data_dir, tags=["random_v3", "sac_v3"])
    print(f"  Total: {len(base_data['obs'])} episodes\n")

    # 고정 holdout (stage간 비교용 — stage별 분할과 별도)
    N = len(base_data["obs"])
    rng = np.random.default_rng(args.seed + 9999)
    holdout_idx    = rng.choice(N, max(500, int(N * 0.05)), replace=False)
    non_holdout    = np.setdiff1d(np.arange(N), holdout_idx)
    holdout_data   = {k: v[holdout_idx]   for k, v in base_data.items()}
    pool_data      = {k: v[non_holdout]   for k, v in base_data.items()}

    cfg = {"trans_hidden": args.trans_hidden, "enc_hidden": args.enc_hidden,
           "embed_dim": args.embed_dim}

    results   = []
    prev_ckpt = None

    for stage_idx in range(min(args.max_stage + 1, len(STAGES))):
        name, filter_mode, trans_ep, enc_ep, max_per_class = STAGES[stage_idx]

        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}: {name}  [filter={filter_mode}]")

        # Stage 2: 신규 데이터 생성 및 병합
        if stage_idx == 2:
            new_data_dir = Path(args.new_data_dir)
            new_data_path = list(new_data_dir.glob("active_v3_*.npz"))

            if not new_data_path:
                if not args.sac_models:
                    print("  ⚠ No SAC models provided and no cached active data. Skipping Stage 2.")
                    break
                print(f"\n  Generating {args.new_episodes} new episodes...")
                new_d = generate_new_data(args.sac_models, new_data_dir,
                                          args.new_episodes, seed=args.seed + 42)
            else:
                print(f"  Loading cached active data: {[p.name for p in new_data_path]}")
                new_d = load_data(str(new_data_dir), tags=["active_v3"])

            # pool_data에 병합
            stage_pool = {k: np.concatenate([pool_data[k], new_d[k]], axis=0)
                          for k in pool_data}
        else:
            stage_pool = pool_data

        # ── Stage 학습 ────────────────────────────────────────────────────────
        ckpt = run_stage(stage_idx, stage_pool, args, device, prev_ckpt)

        # ── 평가 ──────────────────────────────────────────────────────────────
        acc = evaluate_per_class_acc(ckpt, holdout_data, cfg,
                                     n_eval=args.n_eval, device=device)
        n_ep = int(filter_episodes(stage_pool, filter_mode).sum())

        result = {"name": name, "filter": filter_mode, "n_episodes": n_ep, **acc}
        results.append(result)

        print(f"\n  Per-class accuracy (holdout):")
        for cls_name, a in acc.items():
            print(f"    {cls_name:28s} {a:.3f}")

        bb_acc = acc.get("ball_ball", 0.0)
        if bb_acc >= args.advance_threshold:
            print(f"\n  ball_ball acc {bb_acc:.3f} >= {args.advance_threshold} → advance!")
        else:
            print(f"\n  ball_ball acc {bb_acc:.3f} < {args.advance_threshold}")

        prev_ckpt = ckpt

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    print_summary(results, Path(args.out_dir) / "results.csv")
    print(f"\nDone. Outputs → {args.out_dir}")
    print(f"Final ckpt    → {prev_ckpt}")


if __name__ == "__main__":
    main()
