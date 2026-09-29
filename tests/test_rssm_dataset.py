"""
tests/test_rssm_dataset.py

Tests for rssm_dataset.py (ShotData generation and RSSMDataset).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from world_model.rssm_model import N_TYPE, NODE_DIM
from world_model.rssm_dataset import (
    ShotData, RSSMDataset, generate_shot_data, collect_dataset,
)
from world_model.rssm_rollout import make_node


# ── Shared fixture ────────────────────────────────────────────────────────────

def _get_shots(n: int = 5) -> list[ShotData]:
    return collect_dataset(n_shots=n, n_balls=1, seed_start=0)


# ── TestGenerateShotData ──────────────────────────────────────────────────────

class TestGenerateShotData:
    def test_shot_data_has_events(self):
        shots = _get_shots(5)
        assert all(len(s.event_steps) >= 1 for s in shots), \
            "every shot should have at least one event"

    def test_gt_deltas_shape(self):
        shots = _get_shots(5)
        for shot in shots:
            for d in shot.gt_deltas_i:
                assert d.shape == (5,), f"gt_deltas_i shape: expected (5,), got {d.shape}"
            for d in shot.gt_deltas_j:
                if d is not None:
                    assert d.shape == (5,), f"gt_deltas_j shape: expected (5,), got {d.shape}"

    def test_event_steps_node_is_none(self):
        """node_i/node_j are not stored in the pkl (recomputed at training time from raw_rvws)."""
        shots = _get_shots(5)
        for shot in shots:
            for ev in shot.event_steps:
                assert ev.node_i is None
                assert ev.node_j is None

    def test_event_steps_node_recompute_shape(self):
        """make_node(raw_rvws, event_type) is the actual path used to rebuild nodes at train time."""
        shots = _get_shots(5)
        for shot in shots:
            for ev, raw_i, raw_j in zip(shot.event_steps, shot.raw_rvws_i, shot.raw_rvws_j):
                node_i = make_node(raw_i, ev.event_type)
                assert node_i.shape == (NODE_DIM,), \
                    f"node_i shape wrong: {node_i.shape}"
                if raw_j is not None:
                    node_j = make_node(raw_j, ev.event_type)
                    assert node_j.shape == (NODE_DIM,), \
                        f"node_j shape wrong: {node_j.shape}"

    def test_gt_types_range(self):
        shots = _get_shots(5)
        for shot in shots:
            for t in shot.gt_types_i + shot.gt_types_j:
                if t is not None:
                    assert 0 <= t < N_TYPE, \
                        f"gt_type out of range: {t} (expected 0..{N_TYPE-1})"

    def test_some_gt_types_are_none(self):
        """Terminal events (last event for each ball) should have None next type."""
        shots = _get_shots(10)
        has_none = any(t is None for shot in shots for t in shot.gt_types_i)
        assert has_none, "at least some gt_types_i should be None (terminal events)"


# ── TestCollectDataset ────────────────────────────────────────────────────────

class TestCollectDataset:
    def test_collect_dataset_length(self):
        shots = collect_dataset(n_shots=5)
        assert len(shots) == 5

    def test_collect_dataset_variety(self):
        """Different seeds should produce shots with varying event counts."""
        shots = collect_dataset(n_shots=10, seed_start=0)
        lengths = [len(s.event_steps) for s in shots]
        assert len(set(lengths)) > 1, \
            "shots should have varying event counts (data is not all identical)"


# ── TestRSSMDataset ───────────────────────────────────────────────────────────

class TestRSSMDataset:
    def test_dataset_len(self):
        shots = _get_shots(5)
        ds = RSSMDataset(shots)
        assert len(ds) == 5

    def test_dataset_getitem_type(self):
        shots = _get_shots(3)
        ds    = RSSMDataset(shots)
        assert isinstance(ds[0], ShotData)

    def test_dataset_getitem_consistency(self):
        """List lengths inside ShotData must all match."""
        shots = _get_shots(5)
        ds    = RSSMDataset(shots)
        for i in range(len(ds)):
            shot = ds[i]
            n    = len(shot.event_steps)
            assert shot.n_balls == 2, "n_balls should be 2 (cue + 1)"
            assert len(shot.gt_deltas_i) == n
            assert len(shot.gt_deltas_j) == n
            assert len(shot.gt_types_i)  == n
            assert len(shot.gt_types_j)  == n
            assert len(shot.raw_rvws_i)  == n
            assert len(shot.raw_rvws_j)  == n
            assert len(shot.dt_to_next)  == n


# ── TestScheduledSamplingFields ───────────────────────────────────────────────

class TestScheduledSamplingFields:
    def test_raw_rvws_shape(self):
        """raw_rvws_i entries must be (3, 3) numpy arrays."""
        shots = _get_shots(5)
        for shot in shots:
            for rvw in shot.raw_rvws_i:
                assert isinstance(rvw, np.ndarray), "raw_rvws_i must be ndarray"
                assert rvw.shape == (3, 3), f"expected (3,3), got {rvw.shape}"
            for rvw in shot.raw_rvws_j:
                if rvw is not None:
                    assert isinstance(rvw, np.ndarray)
                    assert rvw.shape == (3, 3), f"expected (3,3), got {rvw.shape}"

    def test_dt_to_next_non_negative(self):
        """All dt_to_next values must be >= 0."""
        shots = _get_shots(10)
        for shot in shots:
            for dt in shot.dt_to_next:
                assert dt >= 0.0, f"dt_to_next must be non-negative, got {dt}"

    def test_dt_to_next_last_is_zero(self):
        """Last event in each shot has no successor → dt == 0.0."""
        shots = _get_shots(10)
        for shot in shots:
            if shot.dt_to_next:
                assert shot.dt_to_next[-1] == 0.0, \
                    f"last dt_to_next should be 0.0, got {shot.dt_to_next[-1]}"
