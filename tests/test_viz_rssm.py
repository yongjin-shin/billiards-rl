"""
tests/test_viz_rssm.py

Tests for world_model/viz_rssm.py's free-running rollout support:
shot_rmse(free_running=...), reconstruct(free_running=...), reconstruct_both().
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from world_model.rssm_model import RSSMModel, H_DIM
from world_model.rssm_dataset import generate_shot_data
from world_model.viz_rssm import shot_rmse, reconstruct, reconstruct_both, BALL_IDS

DEVICE = torch.device("cpu")


def _model() -> RSSMModel:
    return RSSMModel(h_dim=H_DIM, hidden=[64, 64])


def _real_shot(seed: int = 0):
    """Generate one real multi-event shot via BilliardsEnv, same as viz_rssm's
    on-the-fly generation path in main()."""
    from simulator import BilliardsEnv

    env = BilliardsEnv(n_balls=1)
    for s in range(seed, seed + 50):
        env.reset(seed=s)
        env.action_space.seed(s)   # env.reset(seed=...) alone does not seed
                                    # action_space's RNG -- without this, sample()
                                    # is nondeterministic across runs.
        env.step(env.action_space.sample())
        shot = generate_shot_data(env.system, BALL_IDS)
        if len(shot.event_steps) >= 2:
            env.close()
            return shot
    env.close()
    pytest.skip("could not find a >=2-event shot in 50 tries")


# ── shot_rmse ─────────────────────────────────────────────────────────────────

class TestShotRmse:
    def test_teacher_forced_returns_finite(self):
        shot = _real_shot()
        rmse = shot_rmse(_model(), shot, DEVICE, free_running=False)
        assert np.isfinite(rmse)
        assert rmse >= 0

    def test_free_running_returns_finite(self):
        shot = _real_shot()
        rmse = shot_rmse(_model(), shot, DEVICE, free_running=True)
        assert np.isfinite(rmse)
        assert rmse >= 0

    def test_single_event_shot_does_not_crash(self):
        """Degenerate 0/1-event shots must not crash node_i/node_j=None handling
        (fix ⑤: node_i/node_j/edge are always None in stored/generated data)."""
        from simulator import BilliardsEnv

        env = BilliardsEnv(n_balls=1)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        shot = generate_shot_data(env.system, BALL_IDS)
        env.close()
        rmse = shot_rmse(_model(), shot, DEVICE, free_running=True)
        assert np.isfinite(rmse) or rmse == float("inf")


# ── reconstruct ───────────────────────────────────────────────────────────────

class TestReconstruct:
    def test_first_touch_model_input_matches_gt_in_both_modes(self):
        """At each ball's first touch, teacher-forcing and free-running must
        feed the model the identical (GT) pre-collision state, so the very
        first predicted delta -- and hence the first rendered pred position --
        is identical between modes."""
        shot = _real_shot()
        model = _model()
        model.eval()

        frames_tf = reconstruct(model, shot, DEVICE, free_running=False)
        frames_fr = reconstruct(model, shot, DEVICE, free_running=True)

        assert len(frames_tf) == len(frames_fr) > 0
        first_tf, first_fr = frames_tf[0], frames_fr[0]
        for bidx in first_tf["pred"]:
            np.testing.assert_allclose(
                first_tf["pred"][bidx], first_fr["pred"][bidx], atol=1e-6,
            )

    def test_teacher_forcing_gt_and_pred_share_frame_keys(self):
        shot = _real_shot()
        frames = reconstruct(_model(), shot, DEVICE, free_running=False)
        assert frames
        for frame in frames:
            assert set(frame["gt"].keys()) == set(frame["pred"].keys())
            assert "is_event" in frame

    def test_free_running_diverges_after_second_touch(self):
        """With a freshly-initialised (untrained, effectively random) model,
        the free-running prediction for a ball's second touch onward feeds a
        different pre-collision state than teacher-forcing (model's own prior
        prediction vs GT), so the reconstructed pred trajectories should
        differ from some point onward for any shot with repeat contacts."""
        shot = _real_shot()
        ball_touch_counts: dict[int, int] = {}
        for ev in shot.event_steps:
            ball_touch_counts[ev.ball_i] = ball_touch_counts.get(ev.ball_i, 0) + 1
        if max(ball_touch_counts.values(), default=0) < 2:
            pytest.skip("shot has no ball with a repeat touch")

        torch.manual_seed(0)
        model = _model()
        model.eval()
        frames_tf = reconstruct(model, shot, DEVICE, free_running=False)
        frames_fr = reconstruct(model, shot, DEVICE, free_running=True)

        diverged = any(
            not np.allclose(ftf["pred"][b], ffr["pred"][b], atol=1e-6)
            for ftf, ffr in zip(frames_tf, frames_fr)
            for b in ftf["pred"]
        )
        assert diverged


# ── reconstruct_both ──────────────────────────────────────────────────────────

class TestReconstructBoth:
    def test_frame_count_matches_individual_reconstructs(self):
        shot = _real_shot()
        model = _model()
        model.eval()

        frames_tf = reconstruct(model, shot, DEVICE, free_running=False)
        frames_both = reconstruct_both(model, shot, DEVICE)
        assert len(frames_both) == len(frames_tf)

    def test_combined_frames_have_expected_keys(self):
        shot = _real_shot()
        frames = reconstruct_both(_model(), shot, DEVICE)
        assert frames
        for frame in frames:
            assert set(frame.keys()) == {"is_event", "gt", "pred_tf", "pred_fr"}
