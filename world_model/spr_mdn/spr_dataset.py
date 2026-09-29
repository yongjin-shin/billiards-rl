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


N_COLL_TYPES = 7   # 0=cue_strike, 1=bb, 2=lin, 3=circ, 4=pocket, 5=slide_roll, 6=roll_stop
COLL_TYPE_NAMES = ["cue_strike", "bb", "lin_cush", "circ_cush", "pocket", "slide_roll", "roll_stop"]


class SPRDataset:
    """
    data_fixeddt 포맷 전체 로드.
    episodes는 (ep_s, ep_f, ep_t, n_cush, ep_a) 튜플 리스트.
      ep_s:   (L, 14) float32 — 상태 시퀀스
      ep_f:   (L-1,) bool    — 충돌 플래그
      ep_t:   (L-1,) int64   — 충돌 타입 (+1 offset: 0=none, 1=bb, 2=lin, 3=circ, 4=pocket, 5=slide_roll, 6=roll_stop)
      n_cush: int            — 쿠션 횟수 (lin+circ)
      ep_a:   (2,) float32   — 타격 파라미터 (angle, speed)

    coll_balls: (ep_b,) int8 bitmask — 1=cue, 2=target, 3=both (-1=none)
                available as self.coll_balls[i] if new-format data; None otherwise.
    """
    MIN_LEN = 5

    def __init__(self, data_dir: str, logger=None):
        self._log = logger.log if logger is not None else print
        meta_path = Path(data_dir) / "metadata.json"
        assert meta_path.exists(), f"metadata.json not found in {data_dir}"

        meta = json.load(open(meta_path))
        self.episodes: list = []
        self.coll_balls: list = []   # parallel to episodes; None if old-format file

        for entry in meta:
            fpath = Path(data_dir) / entry["file"]
            d = np.load(fpath)
            states     = d["states"]
            coll_flags = d["coll_flags"]
            coll_types = d["coll_types"]
            lengths    = d["lengths"]
            actions    = d["actions"]
            has_coll_ball = "coll_ball" in d

            for i, L in enumerate(lengths):
                L = int(L)
                if L < self.MIN_LEN:
                    continue
                ep_s = states[i, :L].astype(np.float32)
                ep_f = coll_flags[i, :L - 1].astype(bool)
                ep_t = coll_types[i, :L - 1].astype(np.int64) + 1   # -1→0, 0→1, …
                n_cush = int((ep_f & ((ep_t == 2) | (ep_t == 3))).sum())
                ep_a = actions[i].astype(np.float32)
                self.episodes.append((ep_s, ep_f, ep_t, n_cush, ep_a))
                if has_coll_ball:
                    self.coll_balls.append(d["coll_ball"][i, :L - 1].astype(np.int8))
                else:
                    self.coll_balls.append(None)

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


class BounceAugDataset(Dataset):
    """
    Data augmentation: t=0 + 모든 바운스 직후(b+1) 지점을 sub-sequence 시작점으로 사용.

    t=0은 큐스트라이크 직후, ep_f[b]=True면 b+1이 바운스 직후 상태.
    remaining >= rollout_steps+1 인 시작점만 포함 (훈련 루프 변경 불필요).

    반환: (seq_s, seq_f, seq_t, seq_a) — SPREpisodeSubset 과 동일 포맷.
    """

    def __init__(self, episodes: list, rollout_steps: int, augment: bool = False):
        self.rollout_steps = rollout_steps
        self.augment       = augment
        self.episodes      = episodes

        min_remaining = rollout_steps + 1
        pairs: list = []
        for ei, (ep_s, ep_f, ep_t, _, ep_a) in enumerate(episodes):
            L = len(ep_s)
            if L >= min_remaining:
                pairs.append((ei, 0, 0))          # seg_type=0: cue_strike
            for b in np.where(ep_f)[0]:
                start = int(b) + 1
                if L - start >= min_remaining:
                    pairs.append((ei, start, int(ep_t[b])))  # seg_type = bounce type

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ei, start_t, seg_type = self.pairs[idx]
        ep_s, ep_f, ep_t, _, ep_a = self.episodes[ei]
        T = self.rollout_steps

        ei, start_t, seg_type = self.pairs[idx]
        ep_s, ep_f, ep_t, _, ep_a = self.episodes[ei]
        T = self.rollout_steps

        seq_s = torch.from_numpy(ep_s[start_t:start_t + T + 1].copy())
        seq_f = torch.from_numpy(ep_f[start_t:start_t + T].copy())
        seq_t = torch.from_numpy(ep_t[start_t:start_t + T].copy())
        seq_a = torch.from_numpy(ep_a.copy())

        if self.augment:
            if torch.rand(1).item() < 0.5:
                seq_s = seq_s.clone()
                seq_a = seq_a.clone()
                seq_s[:, 0] = 1.0 - seq_s[:, 0]
                seq_s[:, 7] = 1.0 - seq_s[:, 7]
                seq_s[:, 2] = -seq_s[:, 2]
                seq_s[:, 9] = -seq_s[:, 9]
                seq_a[0]    = -seq_a[0]

        return seq_s, seq_f, seq_t, seq_a, torch.tensor(seg_type, dtype=torch.long)


class SegmentDataset(Dataset):
    """
    bounce-to-bounce 진짜 세그먼트.
    각 샘플 = 한 bounce 직후 → 다음 bounce 직전+1 까지의 상태 시퀀스.

    반환: (seg_s, seg_t, seg_type_start, seg_len)
      seg_s:           (L+1, 14)  L = 이 세그먼트 스텝 수 (가변)
      seg_t:           (L,)  int  중간은 0, 마지막만 next_bounce_type
      seg_type_start:  int        이 세그먼트를 시작한 바운스 타입 (0=cue_strike)
      seg_len:         int        L
    """

    def __init__(self, episodes: list, min_len: int = 2,
                 max_seg_len: int = 0, max_per_type: int = 0,
                 augment: bool = False, seed: int = 0):
        self.augment  = augment
        from collections import defaultdict
        raw: dict = defaultdict(list)  # type → list of (seg_s, seg_t)

        for ep_s, ep_f, ep_t, _, ep_a in episodes:
            L = len(ep_s)
            bounce_idx = list(np.where(ep_f)[0])

            prev = -1
            for b in bounce_idx:
                seg_start = prev + 1
                seg_end   = b + 1
                seg_len   = seg_end - seg_start - 1

                if seg_len < min_len:
                    prev = b
                    continue

                # cap 긴 세그먼트
                if max_seg_len > 0 and seg_len > max_seg_len:
                    seg_end = seg_start + max_seg_len
                    seg_len = max_seg_len

                seg_s = ep_s[seg_start : seg_end + 1].copy()
                seg_t = ep_t[seg_start : seg_end].copy()
                seg_type_start = 0 if prev == -1 else int(ep_t[prev])
                raw[seg_type_start].append((seg_s, seg_t))
                prev = b

            # 마지막 세그먼트
            seg_start = prev + 1
            seg_end   = L - 1
            seg_len   = seg_end - seg_start
            if seg_len >= min_len:
                if max_seg_len > 0 and seg_len > max_seg_len:
                    seg_end = seg_start + max_seg_len
                seg_s = ep_s[seg_start : seg_end + 1].copy()
                seg_t = ep_t[seg_start : seg_end].copy()
                seg_type_start = 0 if prev == -1 else int(ep_t[prev])
                raw[seg_type_start].append((seg_s, seg_t))

        # max_per_type 다운샘플
        rng = np.random.default_rng(seed)
        self.segments: list = []
        for t, segs in raw.items():
            if max_per_type is not None and max_per_type > 0 and len(segs) > max_per_type:
                idx = rng.choice(len(segs), max_per_type, replace=False)
                segs = [segs[i] for i in idx]
            for seg_s, seg_t in segs:
                self.segments.append((seg_s, seg_t, t))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg_s, seg_t, seg_type_start = self.segments[idx]

        if self.augment and torch.rand(1).item() < 0.5:
            seg_s = seg_s.copy()
            seg_s[:, 0] = 1.0 - seg_s[:, 0]
            seg_s[:, 7] = 1.0 - seg_s[:, 7]
            seg_s[:, 2] = -seg_s[:, 2]
            seg_s[:, 9] = -seg_s[:, 9]

        return (
            torch.from_numpy(seg_s),
            torch.from_numpy(seg_t),
            torch.tensor(seg_type_start, dtype=torch.long),
            torch.tensor(len(seg_s) - 1, dtype=torch.long),
        )


def collate_seg(batch):
    """SegmentDataset용 collate: max 길이로 패딩, mask 반환."""
    seg_s_list, seg_t_list, seg_types, seg_lens = zip(*batch)
    max_L = max(int(sl) for sl in seg_lens)
    B     = len(batch)

    seg_s_pad = torch.zeros(B, max_L + 1, 14)
    seg_t_pad = torch.zeros(B, max_L, dtype=torch.long)
    mask      = torch.zeros(B, max_L, dtype=torch.bool)

    for i, (s, t, sl) in enumerate(zip(seg_s_list, seg_t_list, seg_lens)):
        L = int(sl)
        seg_s_pad[i, :L + 1] = s
        seg_t_pad[i, :L]     = t
        mask[i, :L]          = True

    return seg_s_pad, seg_t_pad, torch.stack(list(seg_types)), torch.stack(list(seg_lens)), mask


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
