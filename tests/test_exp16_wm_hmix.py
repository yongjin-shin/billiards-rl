"""
tests/test_exp16_wm_hmix.py

Tests for Exp-16 M: h_mix_prob schedule and WMSAC real-h mixing.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp16_wm.sac import WMSAC
from exp16_wm.train import ACT_HIGH, ACT_LOW, h_mix_prob


class TestHMixProb:
    def test_linear_decay(self):
        assert h_mix_prob(0, 0.5, 1000) == 0.5
        assert h_mix_prob(500, 0.5, 1000) == pytest.approx(0.25)

    def test_zero_after_decay(self):
        assert h_mix_prob(1500, 0.5, 1000) == 0.0

    def test_disabled(self):
        assert h_mix_prob(0, 0.0, 1000) == 0.0
        assert h_mix_prob(0, 0.5, 0) == 0.0


def _agent_and_batch(B: int = 32):
    torch.manual_seed(0)
    ag = WMSAC(23, 2, ACT_LOW, ACT_HIGH, max_events=4, event_dim=8, wm_coef=0.0)
    batch = {
        "obs": np.random.rand(B, 23).astype(np.float32),
        "action": np.random.uniform(ACT_LOW, ACT_HIGH, (B, 2)).astype(np.float32),
        "reward": np.random.randn(B, 1).astype(np.float32),
        "next_obs": np.random.rand(B, 23).astype(np.float32),
        "done": np.zeros((B, 1), dtype=np.float32),
        "h_real": np.random.randn(B, 4, 8).astype(np.float32),
    }
    return ag, batch


def _m1_grad_after_critic_update(ag, batch) -> float:
    obs, action, reward, next_obs, done = ag._unpack(batch)
    ag.update_critic(obs, action, reward, next_obs, done, h_real=ag._to_tensor(batch["h_real"]))
    return sum(p.grad.abs().sum().item() for p in ag.critic.M1.parameters() if p.grad is not None)


class TestWMSACHMix:
    def test_default_is_off(self):
        ag, _ = _agent_and_batch()
        assert ag.h_mix_p == 0.0

    def test_p0_bellman_reaches_world_model(self):
        # wm_coef=0 → only the Bellman term can push gradient into M
        ag, batch = _agent_and_batch()
        assert _m1_grad_after_critic_update(ag, batch) > 0

    def test_p1_q_reads_only_real_h(self):
        # every sample uses real h → Bellman no longer depends on M (wm_coef=0)
        ag, batch = _agent_and_batch()
        ag.h_mix_p = 1.0
        assert _m1_grad_after_critic_update(ag, batch) == 0.0

    def test_metrics_report_p(self):
        ag, batch = _agent_and_batch()
        ag.h_mix_p = 0.3
        assert ag.update(batch)["h_mix_p"] == pytest.approx(0.3)
