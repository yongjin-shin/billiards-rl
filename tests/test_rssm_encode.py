"""
tests/test_rssm_encode.py

Tests for world_model/rssm_encode.py — encoding an already-simulated pooltool
shot into R-SSM per-ball latents via a frozen, pretrained checkpoint.

Uses real pooltool simulation (BilliardsEnv), no mocks — same style as
tests/test_rssm_dataset.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from simulator import BilliardsEnv
from world_model.rssm_model import H_DIM, RSSMModel
from world_model.rssm_encode import load_frozen_rssm, encode_shot_to_latent

CHECKPOINT = "world_model/results/rssm_v9_ls01_3ball/best.pt"


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frozen_model() -> RSSMModel:
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"checkpoint not found: {CHECKPOINT}")
    return load_frozen_rssm(CHECKPOINT)


def _shot_with_events(n_balls: int = 3, seed_start: int = 0, max_tries: int = 20):
    """Simulate shots until one has at least one tracked event; return (system, ball_ids)."""
    ball_ids = ["cue"] + [str(i) for i in range(1, n_balls + 1)]
    env = BilliardsEnv(n_balls=n_balls)
    try:
        for i in range(max_tries):
            env.reset(seed=seed_start + i)
            env.step(env.action_space.sample())
            if len(env.system.events) > 0:
                return env.system, ball_ids
        pytest.fail("no shot with events found within max_tries")
    finally:
        env.close()


def _shot_without_events(n_balls: int = 3, seed_start: int = 1000, max_tries: int = 50):
    """Simulate shots (min speed, aimed away from the nearest ball) until one has
    zero tracked ball events."""
    from world_model.rssm_dataset import generate_shot_data

    ball_ids = ["cue"] + [str(i) for i in range(1, n_balls + 1)]
    env = BilliardsEnv(n_balls=n_balls)
    try:
        for i in range(max_tries):
            env.reset(seed=seed_start + i)
            # offset=pi (away from nearest ball), min speed — shortest, most isolated roll.
            action = np.array([np.pi, 0.5], dtype=np.float32)
            env.step(action)
            shot = generate_shot_data(env.system, ball_ids)
            if not shot.event_steps:
                return env.system, ball_ids
        pytest.skip("could not find a zero-event shot within max_tries")
    finally:
        env.close()


# ── TestEncodeShotToLatent ───────────────────────────────────────────────────

class TestEncodeShotToLatent:
    def test_happy_path_shape_and_finite(self, frozen_model):
        system, ball_ids = _shot_with_events()
        h_final = encode_shot_to_latent(frozen_model, system, ball_ids)

        assert len(h_final) == len(ball_ids)
        for h in h_final:
            assert h.shape == (H_DIM,)
            assert torch.isfinite(h).all()

    def test_zero_event_shot_returns_zero_latent(self, frozen_model):
        system, ball_ids = _shot_without_events()
        h_final = encode_shot_to_latent(frozen_model, system, ball_ids)

        assert len(h_final) == len(ball_ids)
        for h in h_final:
            assert torch.allclose(h, torch.zeros(H_DIM))

    def test_deterministic_frozen(self, frozen_model):
        system, ball_ids = _shot_with_events()
        h1 = encode_shot_to_latent(frozen_model, system, ball_ids)
        h2 = encode_shot_to_latent(frozen_model, system, ball_ids)

        for a, b in zip(h1, h2):
            assert torch.equal(a, b)

    def test_frozen_params_require_no_grad(self, frozen_model):
        assert all(not p.requires_grad for p in frozen_model.parameters())
        assert not frozen_model.training


# ── TestSimulatorWmTargetRssm ────────────────────────────────────────────────

class TestSimulatorWmTargetRssm:
    def test_info_h_real_shape_and_finite(self):
        if not os.path.exists(CHECKPOINT):
            pytest.skip(f"checkpoint not found: {CHECKPOINT}")

        n_balls = 3
        env = BilliardsEnv(n_balls=n_balls, rssm_checkpoint=CHECKPOINT)
        env.wm_target = "rssm"
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
        finally:
            env.close()

        assert info["h_real"].shape == (n_balls + 1, H_DIM)
        assert info["traj_len"] == n_balls + 1
        assert np.isfinite(info["h_real"]).all()

    def test_wm_target_none_by_default_omits_h_real(self):
        env = BilliardsEnv(n_balls=3)
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
        finally:
            env.close()

        assert "h_real" not in info

    def test_wm_target_traj_unaffected_by_refactor(self):
        env = BilliardsEnv(n_balls=3)
        env.wm_target = "traj"
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
        finally:
            env.close()

        from simulator import TRAJ_MAX_EVENTS, TRAJ_EVENT_DIM
        assert info["h_real"].shape == (TRAJ_MAX_EVENTS, TRAJ_EVENT_DIM)
        assert isinstance(info["traj_len"], int)
