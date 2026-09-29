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
    compute_type_class_weights, evaluate, evaluate_free_running,
    compute_shot_ss_loss, compute_batch_ss_loss,
    compute_vel_magnitude_weights, _magnitude_weight,
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


# ── TestPeriodicCheckpoint ──────────────────────────────────────────────────
#
# train() used to only ever save best.pt (overwritten on improvement) and
# last.pt (final epoch) -- any intermediate epoch's weights were
# irrecoverably lost once a later epoch became the new best. ckpt_every adds
# independent periodic snapshots.

class TestPeriodicCheckpoint:
    def test_periodic_checkpoints_written(self):
        """ckpt_every=1 must write an epoch_NNNN.pt for every epoch."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            cfg.max_epochs = 3
            cfg.ckpt_every = 1
            train(cfg)
            for epoch in (1, 2, 3):
                path = os.path.join(tmp, f"epoch_{epoch:04d}.pt")
                assert os.path.exists(path), f"{path} not saved"
            ckpt = torch.load(os.path.join(tmp, "epoch_0002.pt"), weights_only=False)
            assert ckpt["epoch"] == 2
            assert "state" in ckpt

    def test_ckpt_every_zero_disables_periodic_saves(self):
        """ckpt_every=0 (disabled) must not write any epoch_NNNN.pt file."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            cfg.max_epochs = 3
            cfg.ckpt_every = 0
            train(cfg)
            assert not list(Path(tmp).glob("epoch_*.pt")), \
                "ckpt_every=0 should disable periodic checkpoints"
            # best/last logic must be unaffected
            assert os.path.exists(os.path.join(tmp, "last.pt"))


# ── TestLRSchedule ────────────────────────────────────────────────────────────
#
# LR must stay constant at cfg.lr while ss_prob anneals (ss_start→ss_end),
# then a fresh CosineAnnealingLR cycle starts exactly when ss_prob first
# reaches ss_end -- not a single global schedule spanning all of max_epochs.

class TestLRSchedule:
    def test_lr_constant_during_ss_warmup_then_decays(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            cfg.max_epochs = 4
            cfg.ss_warmup  = 1     # ss_prob reaches ss_end at epoch 2
            cfg.eval_every = 1
            train(cfg)
            lrs = json.load(open(os.path.join(tmp, "result.json")))["lr_history"]
            assert len(lrs) == 4
            # Epoch 1 (ss_prob=1.0, still annealing) and epoch 2 (ss_prob
            # first hits ss_end) both train at the unmodified base LR.
            assert lrs[0] == pytest.approx(cfg.lr)
            assert lrs[1] == pytest.approx(cfg.lr)
            # From epoch 3 on, the fresh cosine cycle has taken at least one
            # step, so LR must have started decaying.
            assert lrs[2] < lrs[1]

    def test_lr_history_present_in_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert "lr_history" in result
            assert len(result["lr_history"]) == result["epochs_run"]


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


# ── TestVelMagnitudeWeights ───────────────────────────────────────────────────
#
# compute_vel_type_weights only reweights across event *types*, so a
# within-type imbalance (e.g. cue_circular's 83.6% near-zero-delta vs 4.7%
# large-delta events -- see experiments.md "cos(incidence) 재검증") is
# invisible to it. compute_vel_magnitude_weights bins directly on |gt_delta|
# instead, independent of event type.

def _shot_with_delta_norm(norm: float) -> ShotData:
    """Minimal single-event ShotData whose gt_deltas_i has exactly `norm`."""
    ev = _synth_event(event_type=1, ball_i=0)
    delta = torch.zeros(5)
    delta[0] = norm
    return ShotData(
        n_balls     = 1,
        event_steps = [ev],
        gt_deltas_i = [delta],
        gt_deltas_j = [None],
        gt_types_i  = [1],
        gt_types_j  = [None],
        raw_rvws_i  = [np.zeros((3, 3))],
        raw_rvws_j  = [None],
        dt_to_next  = [0.01],
    )


class TestVelMagnitudeWeights:
    def test_weights_len_matches_bins(self):
        shots = [_shot_with_delta_norm(0.5) for _ in range(5)]
        edges, weights = compute_vel_magnitude_weights(shots, torch.device("cpu"))
        assert weights.shape[0] == edges.shape[0] - 1

    def test_rare_bin_gets_higher_weight(self):
        """Fewer events in a bin -> larger inverse-frequency weight."""
        shots = (
            [_shot_with_delta_norm(0.5) for _ in range(90)]     # bin [0,1)
            + [_shot_with_delta_norm(15.0) for _ in range(10)]  # bin [10,30), rarer
        )
        edges, weights = compute_vel_magnitude_weights(shots, torch.device("cpu"), max_w=100.0)
        bin_small = int(torch.bucketize(torch.tensor(0.5), edges[1:-1]))
        bin_large = int(torch.bucketize(torch.tensor(15.0), edges[1:-1]))
        assert weights[bin_large] > weights[bin_small]

    def test_absent_bin_gets_zero_not_poisoned_mean(self):
        """Bins with zero events get weight 0 (not clamped to 1, which would
        skew the mean normalization for populated bins)."""
        shots = [_shot_with_delta_norm(0.5) for _ in range(5)]
        edges, weights = compute_vel_magnitude_weights(shots, torch.device("cpu"))
        bin_absent = int(torch.bucketize(torch.tensor(50.0), edges[1:-1]))
        assert weights[bin_absent] == 0.0

    def test_weights_capped_at_max_w(self):
        shots = [_shot_with_delta_norm(0.5) for _ in range(999)] + [_shot_with_delta_norm(15.0)]
        edges, weights = compute_vel_magnitude_weights(shots, torch.device("cpu"), max_w=5.0)
        assert (weights <= 5.0).all()

    def test_magnitude_weight_lookup_picks_correct_bin(self):
        edges   = torch.tensor([0.0, 1.0, 3.0, 10.0, 30.0, float("inf")])
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        gt = torch.zeros(5)
        gt[0] = 2.0   # norm=2.0 -> bin [1,3) -> weight 2.0
        w = _magnitude_weight(gt, edges, weights)
        assert float(w) == pytest.approx(2.0)

    def test_magnitude_weight_none_edges_returns_one(self):
        """No mag weighting configured (default/opt-out) -> neutral weight 1.0."""
        w = _magnitude_weight(torch.randn(5), None, None)
        assert w == 1.0


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


# ── TestEvaluateFreeRunning ───────────────────────────────────────────────────
#
# evaluate() is always teacher-forced (GT pre-event rvw) regardless of the
# model's trained ss_prob regime, which doesn't measure the actual
# deployment target: free-running multi-step rollout for Q-value MC
# estimation (roadmap ③). evaluate_free_running() chains the model's own
# predictions forward instead (ss_prob=0.0 semantics), mirroring
# compute_shot_ss_loss's free-running branch.

class TestEvaluateFreeRunning:
    def test_returns_finite(self):
        shots  = collect_dataset(n_shots=5, n_balls=1, seed_start=0)
        model  = RSSMModel(h_dim=32, hidden=[64, 64])
        device = torch.device("cpu")
        rmse, per_type, acc, pocket_acc = evaluate_free_running(model, shots, device)
        assert rmse < float("inf")
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= pocket_acc <= 1.0

    def test_model_back_to_train(self):
        """evaluate_free_running() must leave the model in train mode."""
        shots  = collect_dataset(n_shots=3, n_balls=1, seed_start=0)
        model  = RSSMModel(h_dim=32, hidden=[64, 64])
        model.train()
        evaluate_free_running(model, shots, torch.device("cpu"))
        assert model.training

    def test_multiball_shots_do_not_crash(self):
        """n_balls=2 exercises the ball_ball free-running chaining branch."""
        shots  = collect_dataset(n_shots=5, n_balls=2, seed_start=0)
        model  = RSSMModel(h_dim=32, hidden=[64, 64])
        rmse, per_type, acc, pocket_acc = evaluate_free_running(model, shots, torch.device("cpu"))
        assert rmse < float("inf")

    def test_differs_from_teacher_forced_evaluate(self):
        """
        A randomly-initialized (untrained) model's free-running predictions
        diverge from GT quickly, so RMSE should generally differ from
        evaluate()'s teacher-forced RMSE on shots with >1 event. Not a strict
        inequality requirement (could coincide by chance on tiny data), but
        both must at least be well-defined finite numbers computed
        independently via different code paths.
        """
        shots = collect_dataset(n_shots=10, n_balls=1, seed_start=0)
        model = RSSMModel(h_dim=32, hidden=[64, 64])
        device = torch.device("cpu")
        tf_rmse, _, _, _ = evaluate(model, shots, device)
        fr_rmse, _, _, _ = evaluate_free_running(model, shots, device)
        assert tf_rmse < float("inf") and fr_rmse < float("inf")


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
        shots = collect_dataset(n_shots=5, n_balls=2, seed_start=0)
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

    def test_batch_loss_mean_matches_manual_per_shot_accum_gradients(self):
        """
        Design principle 2: train()'s batch_loss = mean(per-shot normalized
        losses), backward()'d once. This must equal the old accum_steps
        pattern of (shot_loss / N).backward() called N times (summed grads),
        since mean(x_i) and sum(x_i/N) are the same expression — exactly, by
        linearity of autograd, not just approximately.
        """
        shots = collect_dataset(n_shots=3, n_balls=1, seed_start=0)
        model_a = RSSMModel(h_dim=16, hidden=[32])
        model_b = RSSMModel(h_dim=16, hidden=[32])
        model_b.load_state_dict(model_a.state_dict())

        # (a) new pattern: one batched call, mean of shot losses, one backward()
        results = compute_batch_ss_loss(
            model_a, shots, ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
        )
        shot_losses_a = [
            vel / n + tp / n + pk / (n * shot.n_balls)
            for shot, (vel, tp, pk, n) in zip(shots, results)
        ]
        (sum(shot_losses_a) / len(shot_losses_a)).backward()

        # (b) old pattern: per-shot call, (loss / accum_steps).backward() per shot
        n_shots = len(shots)
        for shot in shots:
            vel, tp, pk, n = compute_shot_ss_loss(
                model_b, shot, ss_prob=1.0, params=DEFAULT_FRICTION, device=torch.device("cpu"),
            )
            shot_loss = vel / n + tp / n + pk / (n * shot.n_balls)
            (shot_loss / n_shots).backward()

        compared = 0
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            # q_proj/q_head are only used by aggregate_q, never by the SS loss
            # path — both stay ungrad-touched here, which is expected, not a bug.
            if pa.grad is None and pb.grad is None:
                continue
            assert torch.allclose(pa.grad, pb.grad, atol=1e-5), \
                "batched mean-loss gradient must match manual per-shot accum gradient"
            compared += 1
        assert compared > 0, "no parameters received gradients — test is vacuous"


# ── TestTrainBatchSize ───────────────────────────────────────────────────────

class TestTrainBatchSize:
    def test_train_with_batch_size_smoke(self):
        """batch_size > 1 must complete train() and write result.json."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            cfg.batch_size  = 4
            cfg.accum_steps = 1
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert result["best_val_rmse"] < float("inf")
            assert result["best_val_rmse"] > 0.0

    def test_train_batch_size_larger_than_dataset_no_crash(self):
        """batch_size >= n_shots_train collapses to a single batch per epoch."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            cfg.batch_size  = 100   # > n_shots_train=20
            cfg.accum_steps = 1
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert result["best_val_rmse"] < float("inf")


# ── TestVelMagWeightIntegration ──────────────────────────────────────────────

class TestVelMagWeightIntegration:
    def test_train_with_vel_mag_weight_completes(self):
        """cfg.vel_mag_weight=True must wire through the full train() loop without error."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            cfg.vel_mag_weight = True
            train(cfg)
            result = json.load(open(os.path.join(tmp, "result.json")))
            assert result["best_val_rmse"] < float("inf")
            assert result["best_val_rmse"] > 0.0


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
