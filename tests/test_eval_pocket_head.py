"""
tests/test_eval_pocket_head.py

Tests for world_model/eval_pocket_head.py:
- Regression test locking in the node/edge-reconstruction fix
  (ev.node_i/node_j/edge are always None in current pkl data; the script
  must rebuild them via make_node/make_edge instead of crashing).
- Structural correctness of the new same-shot "which ball" metric.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from world_model.eval_pocket_head import (
    collect_pocket_preds,
    collect_which_ball_preds,
    compute_which_ball_metrics,
)
from world_model.rssm_dataset import collect_dataset
from world_model.rssm_model import RSSMModel


def _device():
    return torch.device("cpu")


class TestCollectPocketPreds:
    def test_no_crash_with_none_node_edge(self):
        """ev.node_i/node_j/edge are always None in current data — this must
        not raise AttributeError('NoneType' object has no attribute 'to')."""
        shots = collect_dataset(n_shots=5, n_balls=1, seed_start=0)
        model = RSSMModel()
        device = _device()

        probs, labels, tauto, baseline = collect_pocket_preds(model, shots, device)

        assert len(probs) == len(labels) == len(tauto) == len(baseline)
        assert len(probs) > 0
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
        assert set(labels.tolist()) <= {0, 1}

    def test_empty_shots(self):
        model = RSSMModel()
        probs, labels, tauto, baseline = collect_pocket_preds(model, [], _device())
        assert len(probs) == 0
        assert len(labels) == 0


class TestCollectWhichBallPreds:
    def test_structural_correctness(self):
        shots = collect_dataset(n_shots=20, n_balls=2, seed_start=0)
        model = RSSMModel()
        device = _device()

        records = collect_which_ball_preds(model, shots, device)

        expected_n = sum(1 for s in shots if s.n_pocketed_targets() == 1)
        assert len(records) == expected_n

        for r in records:
            assert r["pocketed_idx"] in r["probs"]
            assert set(r["probs"].keys()) == set(range(1, 3))  # targets 1, 2
            for p in r["probs"].values():
                assert 0.0 <= p <= 1.0

    def test_single_target_has_no_comparison(self):
        """n_balls=1 has exactly one target ball. A pocketed single target still
        satisfies n_pocketed_targets()==1, so it's a valid record — but with
        only one candidate there's no other ball to rank against."""
        shots = collect_dataset(n_shots=10, n_balls=1, seed_start=0)
        model = RSSMModel()
        records = collect_which_ball_preds(model, shots, _device())

        for r in records:
            assert r["probs"].keys() == {1}

        m = compute_which_ball_metrics(records)
        if m["n"] > 0:
            assert m["n_pairs"] == 0
            assert m["pairwise_acc"] is None
            assert m["top1_acc"] == 1.0  # trivially "ranked highest" among 1 candidate


class TestComputeWhichBallMetrics:
    def test_known_accuracy_values(self):
        records = [
            {"pocketed_idx": 1, "probs": {1: 0.9, 2: 0.1}},  # top-1 correct, pairwise correct
            {"pocketed_idx": 2, "probs": {1: 0.8, 2: 0.2}},  # top-1 wrong, pairwise wrong
        ]
        m = compute_which_ball_metrics(records)

        assert m["n"] == 2
        assert m["n_targets"] == 2
        assert m["top1_acc"] == 0.5
        assert m["n_pairs"] == 2
        assert m["pairwise_acc"] == 0.5

    def test_all_correct(self):
        records = [
            {"pocketed_idx": 1, "probs": {1: 0.9, 2: 0.1}},
            {"pocketed_idx": 2, "probs": {1: 0.1, 2: 0.9}},
        ]
        m = compute_which_ball_metrics(records)
        assert m["top1_acc"] == 1.0
        assert m["pairwise_acc"] == 1.0

    def test_empty_records(self):
        m = compute_which_ball_metrics([])
        assert m == {"n": 0, "n_targets": 0, "top1_acc": None, "n_pairs": 0, "pairwise_acc": None}
