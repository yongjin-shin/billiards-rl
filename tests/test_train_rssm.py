"""
tests/test_train_rssm.py

Smoke tests for world_model/train_rssm.py.
Verifies the full training pipeline runs without error
and produces expected outputs.
"""

import sys
import os
import json
import random
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from world_model.train_rssm import (
    TrainConfig, train,
    compute_type_class_weights, evaluate,
    compute_shot_ss_loss, compute_batch_ss_loss,
)
from world_model.rssm_model import RSSMModel, H_DIM, EventStep, EVENT_BALL_BALL, NODE_DIM, EDGE_DIM
from world_model.rssm_dataset import ShotData, collect_dataset
from world_model.ball_motion import DEFAULT_FRICTION


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _tiny_config(out_dir: str) -> TrainConfig:
    return TrainConfig(
        n_balls       = 1,
        n_shots_train = 20,
        n_shots_val   = 10,
        seed_train    = 0,
        seed_val      = 5000,
        h_dim         = 32,
        hidden        = [64, 64],
        lr            = 1e-3,
        max_epochs    = 2,
        patience      = 10,
        eval_every    = 1,
        accum_steps   = 4,
        use_kendall   = True,
        ss_warmup     = 2,
        out_dir       = out_dir,
        device        = "cpu",
    )


# ── TestSmokeTrainLoop ────────────────────────────────────────────────────────

class TestSmokeTrainLoop:
    def test_train_completes(self):
        """Full train() call completes without error and writes result.json."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            train(cfg)
            result_path = os.path.join(tmp, "result.json")
            assert os.path.exists(result_path), "result.json not written"
            result = json.load(open(result_path))
            assert "best_val_rmse" in result
            assert "epochs_run" in result

    def test_train_writes_checkpoint(self):
        """best.pt should be written when a val improvement occurs."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            train(cfg)
            assert os.path.exists(os.path.join(tmp, "best.pt")), "best.pt not saved"
            assert os.path.exists(os.path.join(tmp, "last.pt")), "last.pt not saved"

    def test_train_val_rmse_finite(self):
        """Val RMSE should be a finite positive number after training."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert result["best_val_rmse"] < float("inf")
            assert result["best_val_rmse"] > 0.0


# ── TestClassWeights ─────────────────────────────────────────────────────────

class TestClassWeights:
    def test_weights_shape(self):
        shots = collect_dataset(n_shots=10, n_balls=1, seed_start=0)
        w = compute_type_class_weights(shots, torch.device("cpu"))
        assert w.shape == (7,), f"expected (7,), got {w.shape}"

    def test_weights_positive(self):
        shots = collect_dataset(n_shots=10, n_balls=1, seed_start=0)
        w = compute_type_class_weights(shots, torch.device("cpu"))
        assert (w > 0).all(), "all class weights should be positive"


# ── TestEvaluate ──────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_evaluate_returns_finite(self):
        shots  = collect_dataset(n_shots=5, n_balls=1, seed_start=0)
        model  = RSSMModel(h_dim=32, hidden=[64, 64])
        device = torch.device("cpu")
        rmse, per_type, acc, pocket_acc = evaluate(model, shots, device)
        assert rmse < float("inf"), "val_rmse should be finite"
        assert 0.0 <= acc <= 1.0,   "type_acc must be in [0, 1]"

    def test_evaluate_model_back_to_train(self):
        """evaluate() must leave the model in train mode."""
        shots  = collect_dataset(n_shots=3, n_balls=1, seed_start=0)
        model  = RSSMModel(h_dim=32, hidden=[64, 64])
        model.train()
        evaluate(model, shots, torch.device("cpu"))
        assert model.training, "model should be in train mode after evaluate()"


# ── TestShotSSLoss ────────────────────────────────────────────────────────────

class TestShotSSLoss:
    def _get_shot(self):
        shots = collect_dataset(n_shots=5, n_balls=1, seed_start=0)
        return shots[0]

    def test_loss_teacher_forcing(self):
        """ss_prob=1.0 (teacher forcing) should produce finite losses."""
        shot   = self._get_shot()
        model  = RSSMModel(h_dim=32, hidden=[64, 64])
        vel, tp, pk, n = compute_shot_ss_loss(
            model, shot, ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert n > 0
        assert vel.item() >= 0.0
        assert tp.item() >= 0.0
        assert pk.item() >= 0.0

    def test_loss_free_running(self):
        """ss_prob=0.0 (free running) should also produce finite losses."""
        shot  = self._get_shot()
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        vel, tp, pk, n = compute_shot_ss_loss(
            model, shot, ss_prob=0.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert n > 0
        assert torch.isfinite(vel)
        assert torch.isfinite(tp)
        assert torch.isfinite(pk)

    def test_loss_gradients_flow(self):
        """Backward through SS loss should produce non-None gradients."""
        shot  = self._get_shot()
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        vel, tp, pk, n = compute_shot_ss_loss(
            model, shot, ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        loss = (vel + 0.3 * tp + 0.5 * pk) / n
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients were computed"


# ── TestBatchSSLoss ───────────────────────────────────────────────────────────
#
# compute_batch_ss_loss batches multiple shots' event sequences via wavefront
# scheduling. Synthetic shots below pre-populate node_i/node_j/edge directly
# (fix ②/⑤ reuse path) so at ss_prob=1.0 the GT branch is always taken and
# make_node/make_edge/raw_rvws are never touched — full control without
# needing physically-consistent rvw data.

def _synth_event(event_type: int, ball_i: int = 0, ball_j: int | None = None) -> EventStep:
    return EventStep(
        event_type = event_type,
        ball_i     = ball_i,
        ball_j     = ball_j,
        node_i     = torch.randn(NODE_DIM),
        node_j     = torch.randn(NODE_DIM) if ball_j is not None else None,
        edge       = torch.randn(EDGE_DIM) if ball_j is not None else None,
        normal     = torch.tensor([1.0, 0.0]),
    )


def _synth_shot(event_types: list[int], n_balls: int = 2) -> ShotData:
    """Minimal but fully-formed ShotData usable at ss_prob=1.0 without real physics."""
    event_steps, gt_deltas_i, gt_deltas_j = [], [], []
    gt_types_i, gt_types_j = [], []
    raw_rvws_i, raw_rvws_j, dt_to_next = [], [], []
    for et in event_types:
        is_bb = (et == EVENT_BALL_BALL)
        ev = _synth_event(et, ball_i=0, ball_j=1 if is_bb else None)
        event_steps.append(ev)
        gt_deltas_i.append(torch.randn(5))
        gt_types_i.append(1)
        raw_rvws_i.append(np.zeros((3, 3)))
        if is_bb:
            gt_deltas_j.append(torch.randn(5))
            gt_types_j.append(1)
            raw_rvws_j.append(np.zeros((3, 3)))
        else:
            gt_deltas_j.append(None)
            gt_types_j.append(None)
            raw_rvws_j.append(None)
        dt_to_next.append(0.01)
    return ShotData(
        n_balls     = n_balls,
        event_steps = event_steps,
        gt_deltas_i = gt_deltas_i,
        gt_deltas_j = gt_deltas_j,
        gt_types_i  = gt_types_i,
        gt_types_j  = gt_types_j,
        raw_rvws_i  = raw_rvws_i,
        raw_rvws_j  = raw_rvws_j,
        dt_to_next  = dt_to_next,
    )


class TestBatchSSLoss:
    def test_batch_loss_size1_matches_compute_shot_ss_loss(self):
        """A batch of exactly one shot must reduce to compute_shot_ss_loss exactly."""
        shots = collect_dataset(n_shots=1, n_balls=2, seed_start=0)
        shot  = shots[0]
        model = RSSMModel(h_dim=32, hidden=[64, 64])

        vel1, tp1, pk1, n1 = compute_shot_ss_loss(
            model, shot, ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        results = compute_batch_ss_loss(
            model, [shot], ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        vel2, tp2, pk2, n2 = results[0]

        assert n1 == n2
        assert torch.allclose(vel1, vel2, atol=1e-5)
        assert torch.allclose(tp1, tp2, atol=1e-5)
        assert torch.allclose(pk1, pk2, atol=1e-5)

    @pytest.mark.parametrize("ss_prob", [1.0, 0.0])
    def test_batch_loss_exact_match_sum_of_per_shot(self, ss_prob):
        """
        At ss_prob ∈ {0.0, 1.0}, the GT-vs-predicted branch is independent of
        the drawn RNG value (see _pick_node_i/_pick_node_j: `x < 0.0` is always
        False, `x < 1.0` is always True), so wavefront interleaving across
        shots cannot change the outcome — batched and per-shot processing
        must match exactly, unlike the 0<ss_prob<1 regime.
        """
        shots = collect_dataset(n_shots=3, n_balls=1, seed_start=0)
        model = RSSMModel(h_dim=32, hidden=[64, 64])

        expected = [
            compute_shot_ss_loss(
                model, shot, ss_prob=ss_prob, params=DEFAULT_FRICTION, device=torch.device("cpu"),
            )
            for shot in shots
        ]
        actual = compute_batch_ss_loss(
            model, shots, ss_prob=ss_prob, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )

        for (ev, et, epk, en), (av, at, apk, an) in zip(expected, actual):
            assert en == an
            assert torch.allclose(ev, av, atol=1e-5)
            assert torch.allclose(et, at, atol=1e-5)
            assert torch.allclose(epk, apk, atol=1e-5)

    def test_batch_loss_intermediate_ss_prob_finite_no_crash(self):
        """0 < ss_prob < 1: exact match not expected (RNG order differs), but must not crash."""
        shots = collect_dataset(n_shots=3, n_balls=1, seed_start=0)
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        results = compute_batch_ss_loss(
            model, shots, ss_prob=0.5, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert len(results) == 3
        for vel, tp, pk, n in results:
            assert n > 0
            assert torch.isfinite(vel) and torch.isfinite(tp) and torch.isfinite(pk)

    def test_batch_loss_gradients_flow(self):
        shots = collect_dataset(n_shots=3, n_balls=1, seed_start=0)
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        results = compute_batch_ss_loss(
            model, shots, ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        loss = sum((vel + 0.3 * tp + 0.5 * pk) / n for vel, tp, pk, n in results) / len(results)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients were computed"

    def test_batch_loss_n_events_matches_sum_of_shot_lengths(self):
        shots = collect_dataset(n_shots=4, n_balls=1, seed_start=0)
        model = RSSMModel(h_dim=16, hidden=[32])
        results = compute_batch_ss_loss(
            model, shots, ss_prob=0.5, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        for shot, (_, _, _, n) in zip(shots, results):
            assert n == len(shot.event_steps)

    def test_batch_loss_heterogeneous_lengths_no_crash(self):
        """Mixing a 2-event shot and a 30-event shot in one batch must not crash."""
        short_shot = _synth_shot([EVENT_BALL_BALL, 1])
        long_shot  = _synth_shot([EVENT_BALL_BALL, 1] * 15)
        model = RSSMModel(h_dim=16, hidden=[32])
        results = compute_batch_ss_loss(
            model, [short_shot, long_shot], ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert results[0][3] == 2
        assert results[1][3] == 30

    def test_batch_loss_mixed_ball_ball_and_single_in_same_wavefront(self):
        """First wavefront: shot A's event is ball_ball, shot B's is single — both paths hit."""
        shot_a = _synth_shot([EVENT_BALL_BALL, 1])
        shot_b = _synth_shot([1, EVENT_BALL_BALL])
        model  = RSSMModel(h_dim=16, hidden=[32])
        results = compute_batch_ss_loss(
            model, [shot_a, shot_b], ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert results[0][3] == 2
        assert results[1][3] == 2
        for vel, tp, pk, n in results:
            assert torch.isfinite(vel) and torch.isfinite(tp) and torch.isfinite(pk)

    def test_batch_loss_legacy_pkl_node_i_not_none_still_respected(self, monkeypatch):
        """
        node_i/node_j/edge pre-populated (as in ②/⑤'s reuse path) must be
        reused verbatim at ss_prob=1.0 — make_node/make_edge must NOT be
        called at all, since raw_rvws are dummy zeros here (would silently
        corrupt the loss if the reuse path were broken).
        """
        import world_model.train_rssm as train_rssm_mod

        def _boom(*a, **kw):
            raise AssertionError("make_node/make_edge should not be called when node/edge are reused")

        monkeypatch.setattr(train_rssm_mod, "make_node", _boom)
        monkeypatch.setattr(train_rssm_mod, "make_edge", _boom)

        shot  = _synth_shot([EVENT_BALL_BALL, 1])
        model = RSSMModel(h_dim=16, hidden=[32])
        results = compute_batch_ss_loss(
            model, [shot], ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert results[0][3] == 2

    def test_batch_loss_no_inplace_autograd_error_same_ball_reused(self):
        """
        Same ball touched across 3 sequential ball_ball events within a shot,
        interleaved (via wavefronts) with a shorter unrelated shot — the
        list-reassignment h pattern must not trigger autograd's in-place
        version-counter error on .backward().
        """
        shot_a = _synth_shot([EVENT_BALL_BALL, EVENT_BALL_BALL, EVENT_BALL_BALL])
        shot_b = _synth_shot([1, 1])
        model  = RSSMModel(h_dim=16, hidden=[32])
        results = compute_batch_ss_loss(
            model, [shot_a, shot_b], ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        loss = sum((vel + tp + pk) / max(n, 1) for vel, tp, pk, n in results)
        loss.backward()   # must not raise
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_batch_loss_uniform_n_balls_assert(self):
        """Mixing shots with different n_balls must raise, not silently corrupt."""
        shot_1 = _synth_shot([1], n_balls=1)
        shot_2 = _synth_shot([1], n_balls=2)
        model  = RSSMModel(h_dim=16, hidden=[32])
        with pytest.raises(AssertionError):
            compute_batch_ss_loss(
                model, [shot_1, shot_2], ss_prob=1.0,
                params=DEFAULT_FRICTION, device=torch.device("cpu"),
            )

    def test_batch_loss_empty_shot_list_returns_empty(self):
        model = RSSMModel(h_dim=16, hidden=[32])
        results = compute_batch_ss_loss(
            model, [], ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert results == []


# ── TestSmokePkl ──────────────────────────────────────────────────────────────

_ROOT      = Path(__file__).parent.parent
_DATA_N1   = _ROOT / "world_model" / "data_rssm"
_DATA_N2   = _ROOT / "world_model" / "data_rssm_n2"

def _has_chunks(data_dir: Path) -> bool:
    return data_dir.is_dir() and bool(list(data_dir.glob("*_chunk*.pkl")))


def _pkl_config(out_dir: str, data_dir: Path, n_balls: int) -> TrainConfig:
    """Minimal config that loads from pre-generated pkl."""
    return TrainConfig(
        n_balls       = n_balls,
        n_shots_train = 20,
        n_shots_val   = 10,
        h_dim         = 32,
        hidden        = [64, 64],
        lr            = 1e-3,
        max_epochs    = 2,
        patience      = 10,
        eval_every    = 1,
        accum_steps   = 4,
        use_kendall   = True,
        ss_warmup     = 2,
        data_dir      = str(data_dir),
        out_dir       = out_dir,
        device        = "cpu",
    )


@pytest.mark.skipif(not _has_chunks(_DATA_N1), reason="data_rssm pkl not found")
class TestSmokePklN1:
    """Smoke tests loading real n_balls=1 pkl data (fix ⑤: node_i=None)."""

    def test_load_and_train_completes(self):
        """train() with pkl data_dir must complete and write result.json."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pkl_config(tmp, _DATA_N1, n_balls=1)
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert "best_val_rmse" in result
            assert result["best_val_rmse"] < float("inf")

    def test_node_is_none_in_loaded_shots(self):
        """Shots from new pkl must have node_i=None (fix ⑤ applied at generation)."""
        from world_model.rssm_dataset import load_dataset
        shots = load_dataset(str(_DATA_N1), max_shots=10)
        assert shots, "no shots loaded"
        for shot in shots:
            for ev in shot.event_steps:
                assert ev.node_i is None, \
                    f"node_i should be None in new pkl; got {type(ev.node_i)}"
                assert ev.node_j is None
                assert ev.edge   is None

    def test_val_rmse_finite_from_pkl(self):
        """evaluate() on pkl-loaded shots must return finite RMSE."""
        from world_model.rssm_dataset import load_dataset
        shots = load_dataset(str(_DATA_N1), max_shots=10)
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        from world_model.train_rssm import evaluate
        rmse, _, acc, _ = evaluate(model, shots, torch.device("cpu"))
        assert rmse < float("inf")
        assert 0.0 <= acc <= 1.0


@pytest.mark.skipif(not _has_chunks(_DATA_N2), reason="data_rssm_n2 pkl not found")
class TestSmokePklN2:
    """Smoke tests loading real n_balls=2 pkl data."""

    def test_load_and_train_completes(self):
        """train() with n_balls=2 pkl data_dir must complete without error."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pkl_config(tmp, _DATA_N2, n_balls=2)
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert "best_val_rmse" in result
            assert result["best_val_rmse"] < float("inf")

    def test_node_is_none_in_loaded_shots(self):
        """n_balls=2 pkl must also have node_i=None."""
        from world_model.rssm_dataset import load_dataset
        shots = load_dataset(str(_DATA_N2), max_shots=5)
        assert shots, "no shots loaded"
        for shot in shots:
            for ev in shot.event_steps:
                assert ev.node_i is None, \
                    f"node_i should be None; got {type(ev.node_i)}"

    def test_n2_shot_has_three_balls(self):
        """--n-balls 2 means 2 object balls + cue = 3 total; shot.n_balls must be 3."""
        from world_model.rssm_dataset import load_dataset
        shots = load_dataset(str(_DATA_N2), max_shots=5)
        for shot in shots:
            assert shot.n_balls == 3, \
                f"expected n_balls=3 (cue+2 obj), got {shot.n_balls}"

    def test_val_rmse_finite_from_pkl(self):
        """evaluate() on n_balls=2 pkl shots must return finite RMSE."""
        from world_model.rssm_dataset import load_dataset
        shots = load_dataset(str(_DATA_N2), max_shots=5)
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        from world_model.train_rssm import evaluate
        rmse, _, acc, _ = evaluate(model, shots, torch.device("cpu"))
        assert rmse < float("inf")
        assert 0.0 <= acc <= 1.0
