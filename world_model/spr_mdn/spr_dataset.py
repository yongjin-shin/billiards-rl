"""
world_model/spr_mdn/spr_dataset.py — SPRDataset with action loading

SSMDataset와 달리 actions(m=2)를 npz에서 읽어 에피소드 5-tuple로 저장.
episodes: (ep_s, ep_f, ep_t, n_cush, ep_a) where ep_a is (m,) float32.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class SPRDataset:
    """
    data_fixeddt 포맷 전체 로드.
    episodes는 (ep_s, ep_f, ep_t, n_cush, ep_a) 튜플 리스트.
      ep_s:   (L, 14) float32 — 상태 시퀀스
      ep_f:   (L-1,) bool    — 충돌 플래그
      ep_t:   (L-1,) int64   — 충돌 타입 (0=none, 1=bb, 2=linear, 3=circular, 4=pocket)
      n_cush: int            — 쿠션 횟수
      ep_a:   (2,) float32   — 타격 파라미터 (angle, speed)
    """
    MIN_LEN = 5

    def __init__(self, data_dir: str, logger=None):
        self._log = logger.log if logger is not None else print
        meta_path = Path(data_dir) / "metadata.json"
        assert meta_path.exists(), f"metadata.json not found in {data_dir}"

        meta = json.load(open(meta_path))
        self.episodes: list = []

        for entry in meta:
            fpath = Path(data_dir) / entry["file"]
            d = np.load(fpath)
            states     = d["states"]
            coll_flags = d["coll_flags"]
            coll_types = d["coll_types"]
            lengths    = d["lengths"]
            actions    = d["actions"]   # (n_ep, 2) float32

            for i, L in enumerate(lengths):
                L = int(L)
                if L < self.MIN_LEN:
                    continue
                ep_s = states[i, :L].astype(np.float32)
                ep_f = coll_flags[i, :L - 1].astype(bool)
                ep_t = coll_types[i, :L - 1].astype(np.int64) + 1   # -1→0, 0→1, …
                n_cush = int((ep_f & ((ep_t == 2) | (ep_t == 3))).sum())
                ep_a = actions[i].astype(np.float32)                 # (2,)
                self.episodes.append((ep_s, ep_f, ep_t, n_cush, ep_a))

        self._log(f"SPRDataset: {len(self.episodes):,} episodes  (min_len={self.MIN_LEN})")


class SPREpisodeSubset(Dataset):
    """
    episodes (5-tuple) 리스트에서 고정 rollout_steps 슬라이스를 반환.
    반환: (seq_s, seq_f, seq_t, seq_a)
      seq_s: (rollout_steps+1, 14)
      seq_f: (rollout_steps,)
      seq_t: (rollout_steps,)
      seq_a: (2,)              — 에피소드 전체에서 동일한 타격 파라미터
    """

    def __init__(self, episodes: list, rollout_steps: int, augment: bool = False,
                 flip_tb: bool = False):
        self.episodes      = episodes
        self.rollout_steps = rollout_steps
        self.augment       = augment
        self.flip_tb       = flip_tb

    def __len__(self) -> int:
        return len(self.episodes) * 4

    def __getitem__(self, idx: int):
        ep_s, ep_f, ep_t, _, ep_a = self.episodes[idx % len(self.episodes)]
        seq_s = torch.from_numpy(ep_s[:self.rollout_steps + 1].copy())
        seq_f = torch.from_numpy(ep_f[:self.rollout_steps].copy())
        seq_t = torch.from_numpy(ep_t[:self.rollout_steps].copy())
        seq_a = torch.from_numpy(ep_a.copy())

        if self.augment:
            do_lr = torch.rand(1).item() < 0.5
            do_tb = self.flip_tb and (torch.rand(1).item() < 0.5)
            if do_lr or do_tb:
                seq_s = seq_s.clone()
                seq_a = seq_a.clone()
            if do_lr:
                seq_s[:, 0] = 1.0 - seq_s[:, 0]   # cue_x → 1-x
                seq_s[:, 7] = 1.0 - seq_s[:, 7]   # tgt_x → 1-x
                seq_s[:, 2] = -seq_s[:, 2]          # cue_vx → -vx
                seq_s[:, 9] = -seq_s[:, 9]          # tgt_vx → -vx
                seq_a[0] = -seq_a[0]                # angle → -angle
            if do_tb:
                seq_s[:, 1]  = 1.0 - seq_s[:, 1]  # cue_y → 1-y
                seq_s[:, 8]  = 1.0 - seq_s[:, 8]  # tgt_y → 1-y
                seq_s[:, 3]  = -seq_s[:, 3]         # cue_vy → -vy
                seq_s[:, 10] = -seq_s[:, 10]        # tgt_vy → -vy
                seq_s[:, 5]  = -seq_s[:, 5]         # cue_wy → -wy
                seq_s[:, 12] = -seq_s[:, 12]        # tgt_wy → -wy
                seq_a[0] = -seq_a[0]                # angle → -angle (LR+TB 시 상쇄)

        return seq_s, seq_f, seq_t, seq_a


class _CapDataset(Dataset):
    """에포크당 최대 항목 수를 제한하는 래퍼."""

    def __init__(self, dataset: Dataset, max_items: int):
        self.dataset   = dataset
        self.max_items = max_items

    def __len__(self) -> int:
        return min(len(self.dataset), self.max_items)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def make_balanced_val_eps(episodes: list, n_each: int = 250, seed: int = 0) -> list:
    """has_bb / no_bb 균형 val 셋 반환."""
    rng = np.random.default_rng(seed)
    has_bb = [ep for ep in episodes if np.any(ep[1] & (ep[2] == 1))]
    no_bb  = [ep for ep in episodes if not np.any(ep[1] & (ep[2] == 1))]
    n_bb  = min(n_each, len(has_bb))
    n_nbb = min(n_each, len(no_bb))
    sel_bb  = [has_bb[i] for i in rng.choice(len(has_bb), n_bb,  replace=False)]
    sel_nbb = [no_bb[i]  for i in rng.choice(len(no_bb),  n_nbb, replace=False)]
    return sel_bb + sel_nbb
