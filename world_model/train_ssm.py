"""
world_model/train_ssm.py — Deterministic SSM 학습 (v16: 5-class unified head)

18-stage curriculum:
  Phase 1 (P1a-P3b): state MSE only, rollout × ss 커리큘럼
  Phase 2 (C1a-C3b): 5-class CE 추가 (w_type=1.0), collision clip oversampling
  Phase 3 (R1-R6):   BERT masking 커리큘럼 (max_mask=0.2→0.8) → pure inference fine-tuning

Stage 전환: L_state plateau (patience 에포크 동안 min_delta 미만 개선)

Usage:
    python world_model/train_ssm.py --data-dir world_model/data_fixeddt
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.ssm_model import SSMWorldModel, ssm_rollout_loss, LATENT_DIM
from world_model.train_fixeddt import augment_state
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H
from log_utils import Logger


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SSMDataset:
    """
    data_fixeddt 포맷 전체 로드.  episodes 는 (ep_s, ep_f, ep_t, n_cush) 튜플 리스트.
    n_cush: 에피소드 내 쿠션 충돌 횟수 (linear=2 OR circular=3, v16 remap 기준).
    rollout_steps 필터는 make_loaders()에서 수행.

    v16 type remap: coll_types + 1  →  -1(no_coll)→0, 0(bb)→1, 1(lin)→2, 2(circ)→3, 3(pkt)→4
    """

    MIN_LEN = 17  # rollout_steps=16 기준 최소 에피소드 길이

    def __init__(self, data_dir: str, logger: "Logger | None" = None):
        from world_model.val_cache import (save_dataset_cache, load_dataset_cache,
                                           dataset_cache_exists)

        meta_path = Path(data_dir) / "metadata.json"
        assert meta_path.exists(), f"metadata.json not found in {data_dir}"
        self._log = logger.log if logger is not None else print

        cache_path = Path(data_dir) / "episodes_cache.npz"  # pkl suffix로 변환됨
        if dataset_cache_exists(cache_path):
            self._log(f"SSMDataset: loading from cache ({cache_path})")
            self.episodes = load_dataset_cache(cache_path)
            self._log(f"SSMDataset: {len(self.episodes):,} episodes  (from cache)")
            return

        meta = json.load(open(meta_path))
        self.episodes: list = []
        n_coll_per_type = np.zeros(5, dtype=np.int64)  # 0=no_coll 1-4=types

        for entry in meta:
            fpath = Path(data_dir) / entry["file"]
            d = np.load(fpath)
            states     = d["states"]
            coll_flags = d["coll_flags"]
            coll_types = d["coll_types"]
            lengths    = d["lengths"]

            for i, L in enumerate(lengths):
                L = int(L)
                if L < self.MIN_LEN:
                    continue
                ep_s = states[i, :L].astype(np.float32)
                ep_f = coll_flags[i, :L-1].astype(bool)
                # v16 remap: +1  (-1→0, 0→1, 1→2, 2→3, 3→4)
                ep_t = coll_types[i, :L-1].astype(np.int64) + 1
                n_cush = int((ep_f & ((ep_t == 2) | (ep_t == 3))).sum())
                self.episodes.append((ep_s, ep_f, ep_t, n_cush))
                for c in range(1, 5):
                    n_coll_per_type[c] += (ep_t[ep_f] == c).sum()

        n_cushs = np.array([ep[3] for ep in self.episodes])
        self._log(f"SSMDataset: {len(self.episodes):,} episodes  (min_len={self.MIN_LEN})")
        self._log(f"  충돌 타입 분포 (v16 remap):")
        for i, name in enumerate(["no_coll", "ball_ball", "linear", "circular", "pocket"]):
            self._log(f"    {name}: {n_coll_per_type[i]:,}")
        self._log(f"  쿠션 횟수 분포:")
        for k in [0, 1, 2, 3, 4, 5]:
            cnt = (n_cushs <= k).sum()
            self._log(f"    n_cush≤{k}: {cnt:,} ({100*cnt/len(n_cushs):.1f}%)")

        self._log(f"  캐시 저장 중: {cache_path}")
        save_dataset_cache(self.episodes, cache_path)
        self._log(f"  캐시 완료")


class EpisodeSubset(Dataset):
    """rollout_steps 길이 chunk 랜덤 샘플."""

    def __init__(self, episodes: list, rollout_steps: int, augment: bool):
        self.episodes     = episodes
        self.rollout_steps = rollout_steps
        self.augment      = augment

    def __len__(self):
        return len(self.episodes) * 4

    def __getitem__(self, idx):
        ep_s, ep_f, ep_t, _ = self.episodes[idx % len(self.episodes)]
        seq_s = torch.from_numpy(ep_s[:self.rollout_steps + 1])
        seq_f = torch.from_numpy(ep_f[:self.rollout_steps])
        seq_t = torch.from_numpy(ep_t[:self.rollout_steps])
        if self.augment:
            flip_lr = torch.rand(1).item() < 0.5
            flip_tb = torch.rand(1).item() < 0.5
            if flip_lr or flip_tb:
                seq_s = torch.stack([
                    augment_state(seq_s[i], flip_lr, flip_tb)
                    for i in range(seq_s.shape[0])
                ])
        return seq_s, seq_f, seq_t


class _CapDataset(Dataset):
    """Dataset wrapper that caps __len__ without changing sampling."""

    def __init__(self, ds: Dataset, max_items: int):
        self.ds = ds
        self._max = min(max_items, len(ds))

    def __len__(self):
        return self._max

    def __getitem__(self, idx):
        return self.ds[idx % len(self.ds)]


class CollisionClipDataset(Dataset):
    """
    충돌 이벤트(ep_f=True)를 앵커로 하는 short clip dataset.

    각 충돌 스텝 t_c에서 max(0, t_c - pre_steps)를 clip 시작점으로 잡고
    rollout_steps 길이만큼 잘라냄. 모델이 충돌 직전 상태에서 충돌을
    반드시 예측해야 하므로 collision recall 학습에 집중됨.

    max_items: dataset __len__ 상한 (DataLoader 크기 비율 조절용).
    """

    def __init__(self, episodes: list, rollout_steps: int, pre_steps: int,
                 augment: bool, max_items: int | None = None):
        self.rollout_steps = rollout_steps
        self.augment       = augment
        self.clips: list   = []

        for ep_s, ep_f, ep_t, _ in episodes:
            coll_times = np.where(ep_f)[0]
            for t_c in coll_times:
                t_start = max(0, t_c - pre_steps)
                if len(ep_s) - t_start >= rollout_steps + 1:
                    self.clips.append((
                        ep_s[t_start:t_start + rollout_steps + 1],
                        ep_f[t_start:t_start + rollout_steps],
                        ep_t[t_start:t_start + rollout_steps],
                    ))

        self._max = max_items if max_items is not None else len(self.clips) * 4

    def __len__(self):
        return self._max

    def __getitem__(self, idx):
        ep_s, ep_f, ep_t = self.clips[idx % len(self.clips)]
        seq_s = torch.from_numpy(ep_s.copy())
        seq_f = torch.from_numpy(ep_f.copy())
        seq_t = torch.from_numpy(ep_t.copy())
        if self.augment:
            flip_lr = torch.rand(1).item() < 0.5
            flip_tb = torch.rand(1).item() < 0.5
            if flip_lr or flip_tb:
                seq_s = torch.stack([
                    augment_state(seq_s[i], flip_lr, flip_tb)
                    for i in range(seq_s.shape[0])
                ])
        return seq_s, seq_f, seq_t


# ──────────────────────────────────────────────────────────────────────────────
# Rollout 오차 평가
# ──────────────────────────────────────────────────────────────────────────────

def make_balanced_val_eps(episodes: list, n_each: int = 250,
                          seed: int = 0) -> list:
    """
    has_bb(ball-ball 충돌 있음) : no_bb = 50:50 비율로 val 에피소드 구성.
    v16 remap: ball_ball = type 1.
    항상 동일한 seed로 고정 → 학습 내내 동일한 val set 사용.
    """
    has_bb = [ep for ep in episodes if np.any(ep[1] & (ep[2] == 1))]
    no_bb  = [ep for ep in episodes if not np.any(ep[1] & (ep[2] == 1))]
    rng    = np.random.default_rng(seed)
    n      = min(n_each, len(has_bb), len(no_bb))
    idx_bb  = rng.choice(len(has_bb),  n, replace=False)
    idx_no  = rng.choice(len(no_bb),   n, replace=False)
    return [has_bb[i] for i in idx_bb] + [no_bb[i] for i in idx_no]


def evaluate_rollout_error(model: SSMWorldModel, episodes: list, device: str,
                           n_eval: int = 200, rollout_steps: int = 60) -> dict:
    """
    50:50 balanced val — has_bb vs no_bb 분리 평가 + collision recall.
    v16: argmax(type_logit) != 0 → collision 예측.

    반환:
      mean_err          전체 평균 (cm)
      mean_err_has_bb   ball-ball 충돌 있는 에피소드만
      mean_err_no_bb    ball-ball 충돌 없는 에피소드만
      coll_recall       충돌 스텝 감지 recall
      coll_precision    충돌 스텝 감지 precision
      0.5s / 1.0s / 2.0s / 3.0s  시간별 오차 (cm)
    """
    model.eval()
    checkpoint_steps = {
        "0.5s": int(0.5 / DT),
        "1.0s": int(1.0 / DT),
        "2.0s": int(2.0 / DT),
        "3.0s": int(3.0 / DT),
    }

    errs_all, errs_bb, errs_no = [], [], []
    cp_errors = {k: [] for k in checkpoint_steps}
    coll_gt_sum, coll_tp_sum, coll_pred_sum = 0, 0, 0

    rng = np.random.default_rng(0)
    ep_indices = rng.choice(len(episodes), min(n_eval, len(episodes)), replace=False)

    with torch.no_grad():
        for ep_idx in ep_indices:
            ep_s, ep_f, ep_t, _ = episodes[ep_idx]
            T = min(rollout_steps, len(ep_s) - 1)
            if T < 1:
                continue
            s0 = torch.from_numpy(ep_s[0:1]).float().to(device)
            s_hat, type_logit = model(s0, n_steps=T)
            pr = s_hat[0].cpu().numpy()
            # v16: collision predicted when argmax != 0
            pc = (type_logit[0].argmax(-1).cpu().numpy() != 0)

            gt = ep_s[1:T+1]
            pr_pos = pr[1:T+1]
            cue_err = np.sqrt(((pr_pos[:, 0]-gt[:, 0])*TABLE_W)**2 +
                               ((pr_pos[:, 1]-gt[:, 1])*TABLE_H)**2)
            tgt_err = np.sqrt(((pr_pos[:, 7]-gt[:, 7])*TABLE_W)**2 +
                               ((pr_pos[:, 8]-gt[:, 8])*TABLE_H)**2)
            step_err = (cue_err + tgt_err) / 2 * 100
            me = step_err.mean()

            # v16 remap: ball_ball = type 1
            has_bb = bool(np.any(ep_f & (ep_t == 1)))
            errs_all.append(me)
            (errs_bb if has_bb else errs_no).append(me)

            for label, t in checkpoint_steps.items():
                if t <= T:
                    cp_errors[label].append(step_err[t - 1])

            gt_flags = ep_f[:T]
            coll_gt_sum   += int(gt_flags.sum())
            coll_tp_sum   += int((gt_flags & pc[:len(gt_flags)]).sum())
            coll_pred_sum += int(pc[:len(gt_flags)].sum())

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")

    recall    = coll_tp_sum / coll_gt_sum   if coll_gt_sum   > 0 else 0.0
    precision = coll_tp_sum / coll_pred_sum if coll_pred_sum > 0 else 0.0

    result = {
        "mean_err":        _m(errs_all),
        "mean_err_has_bb": _m(errs_bb),
        "mean_err_no_bb":  _m(errs_no),
        "coll_recall":     recall,
        "coll_precision":  precision,
    }
    for k, v in cp_errors.items():
        result[k] = _m(v)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum Scheduler (v16: 18-stage, 7-tuple, plateau-only)
# ──────────────────────────────────────────────────────────────────────────────

class CurriculumScheduler3D:
    """
    18-stage curriculum with plateau-only stage transition.

    7-tuple: (max_cush, rollout_T, w_type, label, clip_pre, coll_ratio, ss_ratio)
    ss_ratio < 0 → BERT masking with max_mask = abs(ss_ratio)
    """

    STAGES = [
        # ── Phase 1: Position MSE only, rollout × ss 커리큘럼 ─────────────────
        (None, 16, 0.0, "P1a[T=16,ss=1.0]",  0, 0.00,  1.0),
        (None, 16, 0.0, "P1b[T=16,ss=0.0]",  0, 0.00,  0.0),
        (None, 32, 0.0, "P2a[T=32,ss=1.0]",  0, 0.00,  1.0),
        (None, 32, 0.0, "P2b[T=32,ss=0.0]",  0, 0.00,  0.0),
        (None, 60, 0.0, "P3a[T=60,ss=1.0]",  0, 0.00,  1.0),
        (None, 60, 0.0, "P3b[T=60,ss=0.0]",  0, 0.00,  0.0),
        # ── Phase 2: 5-class CE, collision difficulty × ss ────────────────────
        (None,  8, 1.0, "C1a[clip=8,ss=1.0]",   2, 0.90,  1.0),
        (None,  8, 1.0, "C1b[clip=8,ss=0.0]",   2, 0.90,  0.0),
        (None, 20, 1.0, "C2a[clip=20,ss=1.0]",  5, 0.70,  1.0),
        (None, 20, 1.0, "C2b[clip=20,ss=0.0]",  5, 0.70,  0.0),
        (None, 60, 1.0, "C3a[full,ss=1.0]",    15, 0.30,  1.0),
        (None, 60, 1.0, "C3b[full,ss=0.0]",    15, 0.30,  0.0),
        # ── Phase 3: BERT masking 커리큘럼 → pure inference fine-tuning ───────
        (None, 60, 1.0, "R1[bert=0.2]",  0, 0.00, -0.2),
        (None, 60, 1.0, "R2[bert=0.4]",  0, 0.00, -0.4),
        (None, 60, 1.0, "R3[bert=0.6]",  0, 0.00, -0.6),
        (None, 60, 1.0, "R4[bert=0.8]",  0, 0.00, -0.8),
        (None, 60, 1.0, "R5[inf]",       0, 0.00,  0.0),
        (None, 60, 1.0, "R6[inf]",       0, 0.00,  0.0),
    ]

    def __init__(self, patience: int = 10, ramp_epochs: int = 5,
                 min_delta: float = 0.0005):
        self.patience    = patience
        self.ramp_epochs = ramp_epochs
        self.min_delta   = min_delta
        self.stage       = 0
        self.best_loss   = float("inf")
        self.stall       = 0
        self.stage_epoch = 0

    def step(self, loss_state: float, epoch: int) -> bool:
        """plateau 감지 시 stage 전환."""
        if loss_state < self.best_loss - self.min_delta:
            self.best_loss = loss_state
            self.stall = 0
        else:
            self.stall += 1

        if self.stage >= len(self.STAGES) - 1:
            return False

        if self.stall >= self.patience:
            self.stage += 1
            self.stage_epoch = epoch
            self.stall = 0
            self.best_loss = float("inf")
            return True
        return False

    def weights(self, epoch: int) -> float:
        """현재 stage의 w_type (ramp 적용)."""
        w_type_t = self.STAGES[self.stage][2]
        progress = min(1.0, (epoch - self.stage_epoch) / max(1, self.ramp_epochs))
        return w_type_t * progress

    @property
    def rollout_steps(self) -> int:
        return self.STAGES[self.stage][1]

    @property
    def max_cushion(self):
        return self.STAGES[self.stage][0]

    @property
    def clip_pre(self) -> int:
        return self.STAGES[self.stage][4]

    @property
    def coll_ratio(self) -> float:
        return self.STAGES[self.stage][5]

    @property
    def ss_ratio(self) -> float:
        return self.STAGES[self.stage][6]

    @property
    def mask_mode(self) -> str:
        return "bert" if self.STAGES[self.stage][6] < 0 else "inference"

    def status(self, epoch: int) -> str:
        wt = self.weights(epoch)
        label = self.STAGES[self.stage][3]
        return (f"{label}  stall={self.stall}/{self.patience}"
                f"  w_type={wt:.2f}  mask={self.mask_mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def make_loaders(train_eps_all, balanced_val_eps, rollout_steps, max_cushion,
                 clip_pre, coll_ratio, batch_size, logger):
    """
    Train loader: coll_ratio 비율로 collision clip + normal episode 혼합.
    Val loader: balanced_val_eps (50:50 has_bb/no_bb) 고정 사용, 항상 T=60.

    coll_ratio=0.0 → CollisionClipDataset 없이 EpisodeSubset만.
    """
    from torch.utils.data import ConcatDataset

    min_len = rollout_steps + 1
    train_eps = [ep for ep in train_eps_all
                 if len(ep[0]) >= min_len and
                 (max_cushion is None or ep[3] <= max_cushion)]

    normal_ds = EpisodeSubset(train_eps, rollout_steps, augment=True)
    n_normal  = len(normal_ds)

    if coll_ratio > 0.0:
        MAX_EPOCH_ITEMS = 60_000
        n_coll_target = int(MAX_EPOCH_ITEMS * coll_ratio)
        n_normal_cap  = MAX_EPOCH_ITEMS - n_coll_target
        normal_ds = _CapDataset(normal_ds, n_normal_cap)
        coll_ds = CollisionClipDataset(
            train_eps, rollout_steps, pre_steps=clip_pre,
            augment=True, max_items=n_coll_target,
        )
        train_ds = ConcatDataset([coll_ds, normal_ds])
        logger.log(f"  Train: {len(train_eps):,} eps  "
                   f"coll_clips={len(coll_ds):,}  normal={len(normal_ds):,}"
                   f"  ratio={coll_ratio:.0%}:{1-coll_ratio:.0%}"
                   f"  rollout={rollout_steps}  pre={clip_pre}")
    else:
        train_ds = normal_ds
        logger.log(f"  Train: {len(train_eps):,} eps  "
                   f"normal={n_normal:,}  rollout={rollout_steps}")

    VAL_T  = 60
    val_ds = EpisodeSubset(
        [ep for ep in balanced_val_eps if len(ep[0]) >= VAL_T + 1],
        VAL_T, augment=False,
    )
    logger.log(f"  Val (50:50 balanced): {len(val_ds.episodes):,} eps  T={VAL_T}")

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    vl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return tl, vl


def train(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.log(f"Device: {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    dataset = SSMDataset(args.data_dir, logger=logger)

    rng_split = np.random.default_rng(0)
    perm = rng_split.permutation(len(dataset.episodes))
    n_val = max(200, int(len(dataset.episodes) * 0.1))
    val_eps_all   = [dataset.episodes[i] for i in perm[:n_val]]
    train_eps_all = [dataset.episodes[i] for i in perm[n_val:]]

    # 50:50 balanced val set — 학습 내내 고정
    balanced_val_eps = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    n_bb  = sum(1 for ep in balanced_val_eps if np.any(ep[1] & (ep[2] == 1)))
    n_nbb = len(balanced_val_eps) - n_bb
    logger.log(f"Balanced val: {n_bb} has_bb + {n_nbb} no_bb = {len(balanced_val_eps)} total")

    # ── 모델 / 옵티마이저 ────────────────────────────────────────────────────
    model = SSMWorldModel(args.latent_dim).to(device)
    if args.ckpt:
        ckpt_data = torch.load(args.ckpt, map_location=device, weights_only=False)
        saved = ckpt_data["state"]
        cur   = model.state_dict()
        # shape가 맞는 key만 로드 (transition/type_head 구조 변경 시 자동 skip)
        filtered = {k: v for k, v in saved.items()
                    if k in cur and v.shape == cur[k].shape}
        skipped  = [k for k in saved if k not in filtered]
        cur.update(filtered)
        model.load_state_dict(cur)
        logger.log(f"Fine-tuning from: {args.ckpt}  (skipped: {skipped})")
    logger.log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # v16 fixed class weights: no_coll(0)=1.0, types(1-4)=4.1  (sqrt inverse-freq)
    CLASS_WEIGHTS = torch.tensor([1.0, 4.1, 4.1, 4.1, 4.1], device=device)

    # Kendall uncertainty weights [state, type]
    log_sigma = torch.nn.Parameter(torch.zeros(2, device=device))

    opt   = torch.optim.Adam(list(model.parameters()) + [log_sigma], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── 커리큘럼 초기화 ──────────────────────────────────────────────────────
    curriculum = CurriculumScheduler3D(
        patience=args.curriculum_patience,
        ramp_epochs=args.curriculum_ramp,
        min_delta=args.curriculum_delta,
    )
    train_loader, val_loader = make_loaders(
        train_eps_all, balanced_val_eps,
        curriculum.rollout_steps, curriculum.max_cushion,
        curriculum.clip_pre, curriculum.coll_ratio,
        args.batch_size, logger,
    )

    best_mean_err = float("inf")

    for epoch in range(1, args.epochs + 1):
        cur_rollout = curriculum.rollout_steps
        cur_w       = curriculum.weights(epoch)   # w_type (float)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_losses = []
        cur_ss = curriculum.ss_ratio
        for seq_s, seq_f, seq_t in train_loader:
            seq_s = seq_s.to(device)
            seq_f = seq_f.to(device)
            seq_t = seq_t.to(device)

            # gt_ar: (B, T, 19) = [physical GT(14); type one-hot(5)]
            type_onehot = F.one_hot(seq_t.long(), num_classes=5).float()  # (B, T, 5)
            gt_ar = torch.cat([seq_s[:, :cur_rollout], type_onehot], dim=-1)  # (B, T, 19)

            s_hat, type_logit = model(
                seq_s[:, 0], cur_rollout,
                gt_states=gt_ar, ss_ratio=cur_ss,
            )
            loss, _ = ssm_rollout_loss(
                s_hat, seq_s, type_logit, seq_t,
                class_weights=CLASS_WEIGHTS,
                log_sigma=log_sigma if cur_w > 0 else None,
                w_type=cur_w,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + [log_sigma], 1.0)
            opt.step()
            tr_losses.append(loss.item())

        sched.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_details = [], []
        type_correct, type_total = 0, 0

        with torch.no_grad():
            for seq_s, seq_f, seq_t in val_loader:
                seq_s = seq_s.to(device)
                seq_f = seq_f.to(device)
                seq_t = seq_t.to(device)

                val_T = seq_s.shape[1] - 1   # val_loader는 항상 T=60
                # val은 항상 pure inference (no gt_states)
                s_hat, type_logit = model(seq_s[:, 0], val_T)
                loss, detail = ssm_rollout_loss(
                    s_hat, seq_s, type_logit, seq_t,
                    class_weights=CLASS_WEIGHTS,
                    log_sigma=log_sigma if cur_w > 0 else None,
                    w_type=cur_w,
                )
                val_losses.append(loss.item())
                val_details.append(detail)

                # type_acc: 전 스텝 (not just collision steps)
                pred = type_logit.argmax(-1)  # (B, T)
                type_correct += (pred == seq_t).sum().item()
                type_total   += pred.numel()

        val_loss   = np.mean(val_losses)
        d          = {k: np.mean([x[k] for x in val_details]) for k in val_details[0]}
        type_acc   = type_correct / type_total if type_total > 0 else 0.0
        loss_state = d['loss_cue'] + d['loss_tgt']

        # ── Rollout 오차 + balanced eval (항상 T=60) ─────────────────────────
        rerr     = evaluate_rollout_error(model, balanced_val_eps, device,
                                          n_eval=len(balanced_val_eps),
                                          rollout_steps=60)
        mean_err    = rerr["mean_err"]
        coll_recall = rerr["coll_recall"]
        coll_prec   = rerr["coll_precision"]

        # ── Curriculum step ───────────────────────────────────────────────────
        advanced = curriculum.step(loss_state, epoch)
        if advanced:
            label = curriculum.STAGES[curriculum.stage][3]
            logger.log(f"  *** Curriculum → Stage {curriculum.stage}: {label} ***")
            train_loader, val_loader = make_loaders(
                train_eps_all, balanced_val_eps,
                curriculum.rollout_steps, curriculum.max_cushion,
                curriculum.clip_pre, curriculum.coll_ratio,
                args.batch_size, logger,
            )
            best_mean_err = float("inf")
            remaining = args.epochs - epoch + 1
            for pg in opt.param_groups:
                pg['lr'] = args.lr * 0.3
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, remaining), eta_min=args.lr * 0.01)

        # ── 로그 ──────────────────────────────────────────────────────────────
        lc  = d['loss_cue']
        lt  = d['loss_tgt']
        lty = d['loss_type']

        kendall_str = ""
        if cur_w > 0:
            s = log_sigma.clamp(-6, 6).detach()
            kw = torch.exp(-s)
            kendall_str = f"  kw=[{kw[0].item():.2f},{kw[1].item():.2f}]"

        cp_keys = [k for k in ["0.5s", "1.0s", "2.0s", "3.0s"]
                   if not np.isnan(rerr.get(k, float("nan")))]
        cp_str  = " | ".join(f"{k}={rerr[k]:.1f}cm" for k in cp_keys) \
                  if (epoch % 10 == 0 or epoch == args.epochs) else ""

        log_line = (
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  [{curriculum.status(epoch)}]"
            f"  tr={np.mean(tr_losses):.4f}  val={val_loss:.4f}"
            f"  | L_cue={lc:.4f}  L_tgt={lt:.4f}  L_type={lty:.4f}"
            f"  type_acc={type_acc:.3f}"
            f"  recall={coll_recall:.3f}  prec={coll_prec:.3f}"
            f"  err={mean_err:.1f}cm"
            f"  (bb={rerr['mean_err_has_bb']:.1f}/nbb={rerr['mean_err_no_bb']:.1f})"
            f"{kendall_str}"
        )
        if cp_str:
            log_line += f"  [{cp_str}]"
        logger.log(log_line)

        # ── Best model 저장 (mean_err 기준) ──────────────────────────────────
        if mean_err < best_mean_err:
            best_mean_err = mean_err
            torch.save({
                "state":              model.state_dict(),
                "log_sigma":          log_sigma.detach().cpu(),
                "epoch":              epoch,
                "mean_err":           mean_err,
                "val_loss":           val_loss,
                "latent_dim":         args.latent_dim,
                "rollout_steps":      cur_rollout,
                "curriculum_stage":   curriculum.stage,
            }, out_dir / "best.pt")

    torch.save({"state": model.state_dict(), "epoch": args.epochs,
                "latent_dim": args.latent_dim}, out_dir / "final.pt")

    cfg = {
        "latent_dim": args.latent_dim,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "best_mean_err_cm": best_mean_err,
        "curriculum_stages": CurriculumScheduler3D.STAGES,
    }
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    logger.log(f"\nSaved → {out_dir}  best_mean_err={best_mean_err:.1f}cm")
    logger.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",      default="world_model/data_fixeddt")
    p.add_argument("--out-dir",       default=None)
    p.add_argument("--latent-dim",    type=int,   default=LATENT_DIM)
    p.add_argument("--epochs",        type=int,   default=600)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--ckpt",          default=None,
                   help="fine-tune 시작점 checkpoint (state dict 포함된 .pt)")
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--curriculum-patience", type=int,   default=15,
                   help="L_state plateau 판정 에포크 수")
    p.add_argument("--curriculum-ramp",     type=int,   default=8,
                   help="stage 진입 후 weight ramp 기간 (에포크)")
    p.add_argument("--curriculum-delta",    type=float, default=0.0005,
                   help="plateau 판정 최소 개선량")
    args = p.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"world_model/results/ssm_3dcur_{ts}"

    train(args)


if __name__ == "__main__":
    main()
