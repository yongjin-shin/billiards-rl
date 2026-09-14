"""
world_model/run_grid.py — 하이퍼파라미터 그리드 서치 + 결과 테이블 추출

Grid:
    decoder      : mlp, lstm
    z_dim        : 8, 16
    beta         : 0.1, 0.3, 0.5
    pos_weight   : 1, 5, 10

총 2 × 2 × 3 × 3 = 36 실험

Usage:
    python world_model/run_grid.py
    python world_model/run_grid.py --jobs 4   # 최대 병렬 수
    python world_model/run_grid.py --summary-only  # 학습 건너뛰고 결과만 재집계
"""

import os
import sys
import json
import argparse
import subprocess
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ── 그리드 정의 ───────────────────────────────────────────────────────────────
DECODERS    = ["mlp", "lstm"]
Z_DIMS      = [8, 16]
BETAS       = [0.1, 0.3, 0.5]
POS_WEIGHTS = [1, 5, 10]

# 공통 하이퍼파라미터
EPOCHS        = 100
BATCH_SIZE    = 256
LR            = 3e-4
HIDDEN_ENC    = 64
HIDDEN_DEC    = 128
# LSTM 전용
KL_ANNEAL     = 50
SS_ANNEAL     = 80
SS_MIN_RATIO  = 0.0

DATA_DIR  = "world_model/data_abs"
CKPT_DIR  = "world_model/checkpoints"
RESULT_CSV = "world_model/grid_results.csv"


def config_name(decoder, z, beta, pw):
    return f"{decoder}_z{z}_b{beta}_pw{pw}"


def train_one(cfg):
    """단일 실험 실행 → (cfg, best_val_loss, history_path) 반환"""
    decoder, z, beta, pw = cfg
    name = config_name(decoder, z, beta, pw)

    cmd = [
        sys.executable, "world_model/train_vae.py",
        "--data",       DATA_DIR,
        "--decoder",    decoder,
        "--z-dim",      str(z),
        "--beta",       str(beta),
        "--pos-weight", str(pw),
        "--epochs",     str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr",         str(LR),
        "--hidden-enc", str(HIDDEN_ENC),
        "--hidden-dec", str(HIDDEN_DEC),
    ]
    if decoder == "lstm":
        cmd += [
            "--kl-anneal-epochs", str(KL_ANNEAL),
            "--ss-anneal-epochs", str(SS_ANNEAL),
            "--ss-min-ratio",     str(SS_MIN_RATIO),
        ]

    log_path = f"/tmp/grid_{name}.log"
    t0 = datetime.now()
    with open(log_path, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (datetime.now() - t0).total_seconds()

    if ret.returncode != 0:
        print(f"  [FAIL] {name}  ({elapsed:.0f}s)")
        return cfg, None, None, elapsed

    # 가장 최근 ckpt 찾기 (타임스탬프 기준)
    candidates = [
        f for f in os.listdir(CKPT_DIR)
        if f.startswith(f"vae_{decoder}_z{z}_") and f.endswith(".pt")
    ]
    if not candidates:
        return cfg, None, None, elapsed

    latest = sorted(candidates)[-1]
    ckpt_path = os.path.join(CKPT_DIR, latest)
    hist_path = ckpt_path.replace(".pt", "_history.json")

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    val_loss = ckpt["val_loss"]
    best_ep  = ckpt["epoch"]

    # history에서 세부 정보 추출
    pos_l = type_l = kl_l = None
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        best_row = min(h, key=lambda x: x["val"])
        pos_l  = best_row.get("pos")
        type_l = best_row.get("type")
        kl_l   = best_row.get("kl")

    print(f"  [OK]   {name:<30} val={val_loss:.4f}  ep={best_ep}  ({elapsed:.0f}s)")
    return cfg, val_loss, dict(pos=pos_l, type=type_l, kl=kl_l, ep=best_ep,
                               ckpt=latest, elapsed=elapsed), elapsed


def collect_results():
    """이미 학습된 모든 ckpt에서 결과 집계"""
    import torch

    all_cfgs = list(itertools.product(DECODERS, Z_DIMS, BETAS, POS_WEIGHTS))
    rows = []

    for cfg in all_cfgs:
        decoder, z, beta, pw = cfg
        # 해당 설정의 ckpt 찾기
        candidates = [
            f for f in os.listdir(CKPT_DIR)
            if f.startswith(f"vae_{decoder}_z{z}_") and f.endswith(".pt")
        ]
        if not candidates:
            continue

        for ckpt_file in sorted(candidates):
            ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
            hist_path = ckpt_path.replace(".pt", "_history.json")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                args = ckpt.get("args", {})
                # args의 beta, pos_weight 확인
                if abs(args.get("beta", -1) - beta) > 1e-6:
                    continue
                if abs(args.get("pos_weight", -1) - pw) > 1e-6:
                    continue
            except Exception:
                continue

            val_loss = ckpt["val_loss"]
            best_ep  = ckpt["epoch"]

            pos_l = type_l = kl_l = None
            if os.path.exists(hist_path):
                h = json.load(open(hist_path))
                best_row = min(h, key=lambda x: x["val"])
                pos_l  = best_row.get("pos")
                type_l = best_row.get("type")
                kl_l   = best_row.get("kl")
                final_kl = h[-1].get("kl")
            else:
                final_kl = None

            rows.append(dict(
                decoder=decoder, z_dim=z, beta=beta, pos_weight=pw,
                val_loss=val_loss, best_ep=best_ep,
                pos_loss=pos_l, type_loss=type_l, kl=kl_l,
                final_kl=final_kl,
                ckpt=ckpt_file,
            ))

    # (decoder, z, beta, pw) 당 best val_loss 1개만 유지
    seen = {}
    for r in rows:
        key = (r["decoder"], r["z_dim"], r["beta"], r["pos_weight"])
        if key not in seen or r["val_loss"] < seen[key]["val_loss"]:
            seen[key] = r
    return list(seen.values())


def print_table(rows):
    rows_sorted = sorted(rows, key=lambda r: r["val_loss"])
    print("\n" + "=" * 95)
    print(f"{'decoder':<6} {'z':>3} {'beta':>5} {'pw':>4} | "
          f"{'val_loss':>9} {'pos':>7} {'type':>7} {'kl':>7} {'fKL':>7} | "
          f"{'ep':>4}  ckpt")
    print("-" * 95)
    for r in rows_sorted:
        print(f"{r['decoder']:<6} {r['z_dim']:>3} {r['beta']:>5} {r['pos_weight']:>4} | "
              f"{r['val_loss']:>9.4f} "
              f"{r['pos_loss'] or 0:>7.4f} {r['type_loss'] or 0:>7.4f} "
              f"{r['kl'] or 0:>7.4f} {r['final_kl'] or 0:>7.4f} | "
              f"{r['best_ep']:>4}  {r['ckpt']}")
    print("=" * 95)


def save_csv(rows):
    import csv
    keys = ["decoder", "z_dim", "beta", "pos_weight",
            "val_loss", "pos_loss", "type_loss", "kl", "final_kl", "best_ep", "ckpt"]
    with open(RESULT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["val_loss"]))
    print(f"\nCSV saved → {RESULT_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",         type=int, default=3,
                        help="최대 병렬 학습 수 (CPU 부하 고려)")
    parser.add_argument("--summary-only", action="store_true",
                        help="학습 건너뛰고 기존 ckpt에서 결과만 집계")
    args = parser.parse_args()

    all_cfgs = list(itertools.product(DECODERS, Z_DIMS, BETAS, POS_WEIGHTS))
    print(f"Grid search: {len(all_cfgs)} experiments  (jobs={args.jobs})")
    print(f"  decoder={DECODERS}  z={Z_DIMS}  beta={BETAS}  pw={POS_WEIGHTS}")

    if not args.summary_only:
        print(f"\n{'='*60}")
        print(f"Training {len(all_cfgs)} models ...")
        print(f"{'='*60}")
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(train_one, cfg): cfg for cfg in all_cfgs}
            for fut in as_completed(futs):
                cfg, val, info, elapsed = fut.result()

    print("\nCollecting results ...")
    rows = collect_results()
    print(f"  Found {len(rows)} completed experiments")
    print_table(rows)
    save_csv(rows)


if __name__ == "__main__":
    main()
