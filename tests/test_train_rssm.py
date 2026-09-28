"""
tests/test_train_rssm.py

Smoke tests for world_model/train_rssm.py.
Verifies the full training pipeline runs without error
and produces expected outputs.
"""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from world_model.train_rssm import (
    TrainConfig, train,
    compute_type_class_weights, evaluate,
    compute_shot_ss_loss,
)
from world_model.rssm_model import RSSMModel, H_DIM
from world_model.rssm_dataset import collect_dataset
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
        rmse, per_type, acc = evaluate(model, shots, device)
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
        vel, tp, n = compute_shot_ss_loss(
            model, shot, ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert n > 0
        assert vel.item() >= 0.0
        assert tp.item() >= 0.0

    def test_loss_free_running(self):
        """ss_prob=0.0 (free running) should also produce finite losses."""
        shot  = self._get_shot()
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        vel, tp, n = compute_shot_ss_loss(
            model, shot, ss_prob=0.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        assert n > 0
        assert torch.isfinite(vel)
        assert torch.isfinite(tp)

    def test_loss_gradients_flow(self):
        """Backward through SS loss should produce non-None gradients."""
        shot  = self._get_shot()
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        vel, tp, n = compute_shot_ss_loss(
            model, shot, ss_prob=1.0,
            params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        loss = (vel + 0.3 * tp) / n
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients were computed"
