"""
tests/test_rssm_batch.py

Unit tests for world_model/rssm_batch.py (Phase 0: wavefront scheduler).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from world_model.rssm_batch import iter_wavefronts, split_wavefront_by_type
from world_model.rssm_dataset import ShotData
from world_model.rssm_model import EventStep, EVENT_BALL_BALL, EVENT_POCKET


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _dummy_normal() -> torch.Tensor:
    return torch.tensor([1.0, 0.0])


def _ev(event_type: int, ball_i: int = 0, ball_j: int | None = None) -> EventStep:
    return EventStep(
        event_type=event_type, ball_i=ball_i, ball_j=ball_j,
        node_i=None, node_j=None, edge=None, normal=_dummy_normal(),
    )


def _shot(event_types: list[int], n_balls: int = 2) -> ShotData:
    """Minimal ShotData with only event_steps populated meaningfully."""
    n = len(event_types)
    # ball_ball events need a real ball_j (matches rssm_dataset.py's normal
    # case); ball_j=None is only for the untracked-second-ball edge case,
    # tested explicitly in test_ball_ball_type_with_none_ball_j_routed_to_single.
    event_steps = [
        _ev(et, ball_j=1) if et == EVENT_BALL_BALL else _ev(et)
        for et in event_types
    ]
    return ShotData(
        n_balls     = n_balls,
        event_steps = event_steps,
        gt_deltas_i = [None] * n,
        gt_deltas_j = [None] * n,
        gt_types_i  = [None] * n,
        gt_types_j  = [None] * n,
        raw_rvws_i  = [None] * n,
        raw_rvws_j  = [None] * n,
        dt_to_next  = [0.0] * n,
    )


# ── TestIterWavefronts ───────────────────────────────────────────────────────

class TestIterWavefronts:
    def test_all_events_visited_exactly_once(self):
        """Every (shot_idx, event_idx) pair must appear exactly once, in order."""
        shots = [
            _shot([EVENT_BALL_BALL, 1, 5]),          # 3 events
            _shot([1, EVENT_BALL_BALL, 1, EVENT_POCKET, 5]),  # 5 events
            _shot([1]),                               # 1 event
        ]
        visited: dict[int, list[int]] = {i: [] for i in range(len(shots))}
        for wavefront in iter_wavefronts(shots):
            for s, k in wavefront:
                visited[s].append(k)

        for s, shot in enumerate(shots):
            assert visited[s] == list(range(len(shot.event_steps))), \
                f"shot {s} events not visited in order exactly once: {visited[s]}"

    def test_active_count_nonincreasing(self):
        """Wavefront size must never increase (shots only drop out, never rejoin)."""
        shots = [
            _shot([1, 1, 1, 1, 1]),
            _shot([1, 1]),
            _shot([1, 1, 1]),
        ]
        sizes = [len(w) for w in iter_wavefronts(shots)]
        assert sizes == sorted(sizes, reverse=True), \
            f"wavefront sizes must be non-increasing, got {sizes}"
        assert sizes[0] == 3   # all shots active at start
        assert sizes[-1] == 1  # only the longest shot active at the end

    def test_empty_shot_list_yields_no_wavefronts(self):
        assert list(iter_wavefronts([])) == []

    def test_shot_with_no_events_never_appears(self):
        shots = [_shot([]), _shot([1, 1])]
        for wavefront in iter_wavefronts(shots):
            assert all(s != 0 for s, _ in wavefront), \
                "empty shot must never appear in any wavefront"

    def test_total_yielded_items_matches_total_events(self):
        shots = [_shot([1, 1, 1]), _shot([1]), _shot([1, 1])]
        total_events = sum(len(s.event_steps) for s in shots)
        total_yielded = sum(len(w) for w in iter_wavefronts(shots))
        assert total_yielded == total_events


# ── TestSplitWavefrontByType ─────────────────────────────────────────────────

class TestSplitWavefrontByType:
    def test_grouping_correct(self):
        shots = [
            _shot([EVENT_BALL_BALL, 1]),
            _shot([1, EVENT_BALL_BALL]),
            _shot([EVENT_POCKET]),
        ]
        # First wavefront: shot0->ev0(bb), shot1->ev0(single type1), shot2->ev0(pocket)
        first_wavefront = next(iter_wavefronts(shots))
        bb, single = split_wavefront_by_type(shots, first_wavefront)

        assert bb == [(0, 0)]
        assert set(single) == {(1, 0), (2, 0)}

    def test_partition_covers_all_items_exactly_once(self):
        shots = [
            _shot([EVENT_BALL_BALL, EVENT_BALL_BALL, 1, EVENT_POCKET]),
        ]
        for wavefront in iter_wavefronts(shots):
            bb, single = split_wavefront_by_type(shots, wavefront)
            assert set(bb) | set(single) == set(wavefront)
            assert set(bb) & set(single) == set()

    def test_all_ball_ball_wavefront(self):
        shots = [_shot([EVENT_BALL_BALL]), _shot([EVENT_BALL_BALL])]
        wavefront = next(iter_wavefronts(shots))
        bb, single = split_wavefront_by_type(shots, wavefront)
        assert len(bb) == 2
        assert single == []

    def test_all_single_wavefront(self):
        shots = [_shot([1]), _shot([5])]
        wavefront = next(iter_wavefronts(shots))
        bb, single = split_wavefront_by_type(shots, wavefront)
        assert bb == []
        assert len(single) == 2

    def test_ball_ball_type_with_none_ball_j_routed_to_single(self):
        """
        raw_type==0 events with an untracked second ball keep event_type ==
        EVENT_BALL_BALL but ball_j is None (see rssm_dataset.py's else-branch
        fallback) — these must be routed to the single-ball path, matching
        compute_shot_ss_loss's `ev_type == EVENT_BALL_BALL and bj is not None` guard.
        """
        shot = _shot([EVENT_BALL_BALL])
        shot.event_steps[0].ball_j = None
        wavefront = next(iter_wavefronts([shot]))
        bb, single = split_wavefront_by_type([shot], wavefront)
        assert bb == []
        assert single == [(0, 0)]
