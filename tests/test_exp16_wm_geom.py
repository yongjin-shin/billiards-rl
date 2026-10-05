"""
tests/test_exp16_wm_geom.py

Tests for Exp-16 G: geom_features / GeomCritic / GeomSAC.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp16_wm.diagnose_q import geometric_preshot_score
from exp16_wm.networks import GEOM_DIM, POCKET_CENTERS, TABLE_L, TABLE_W, GeomCritic, geom_features
from exp16_wm.sac import GeomSAC
from exp16_wm.train import ACT_HIGH, ACT_LOW, get_obs_dim

N_BALLS = 3
OBS_DIM = get_obs_dim(N_BALLS)


def make_obs(cue_m, balls_m, pocketed=(False, False, False)) -> torch.Tensor:
    obs = [cue_m[0] / TABLE_W, cue_m[1] / TABLE_L]
    for (x, y), p in zip(balls_m, pocketed):
        obs += [0.0, 0.0, 1.0] if p else [x / TABLE_W, y / TABLE_L, 0.0]
    obs += [0.0] * 12
    return torch.tensor([obs], dtype=torch.float32)


STRAIGHT_IN = dict(cue_m=(0.4, 0.4), balls_m=[(0.2, 0.2), (0.8, 1.8), (0.7, 1.6)])


class TestGeomFeatures:
    def test_straight_in_shot(self):
        # cue → nearest ball → bottom-left corner pocket on one diagonal, delta=0
        f = geom_features(make_obs(**STRAIGHT_IN), torch.tensor([[0.0, 4.0]]), N_BALLS)[0]
        assert f.shape == (GEOM_DIM,)
        assert f[0] == 1.0                       # hit
        assert f[1] == pytest.approx(1.0, abs=1e-3)  # full-ball contact
        assert f[3] == pytest.approx(0.0, abs=1e-2)  # ray goes into the pocket

    def test_aiming_away_misses(self):
        f = geom_features(make_obs(**STRAIGHT_IN), torch.tensor([[np.pi, 4.0]]), N_BALLS)[0]
        assert f[0] == 0.0
        assert f[1:5].tolist() == [0.0, 1.0, 1.0, 1.0]   # sentinels

    def test_all_balls_pocketed_edge_case(self):
        obs = make_obs((0.4, 0.4), [(0, 0)] * 3, pocketed=(True, True, True))
        f = geom_features(obs, torch.tensor([[0.0, 4.0]]), N_BALLS)
        assert torch.isfinite(f).all() and f[0, 0] == 0.0

    def test_speed_rescaled(self):
        f = geom_features(make_obs(**STRAIGHT_IN), torch.tensor([[0.0, 8.0]]), N_BALLS)[0]
        assert f[5] == pytest.approx(1.0)

    def test_differentiable_in_angle(self):
        # slightly cut shot: miss distance must respond to the aim angle
        a = torch.tensor([[0.05, 4.0]], requires_grad=True)
        f = geom_features(make_obs(**STRAIGHT_IN), a, N_BALLS)
        f[0, 3].backward()
        assert torch.isfinite(a.grad).all() and a.grad[0, 0].abs() > 0

    def test_matches_numpy_baseline(self):
        # same "miss" quantity as the D4 numpy baseline (normalized by 0.5 m)
        obs = make_obs(**STRAIGHT_IN)
        delta = 0.05
        f = geom_features(obs, torch.tensor([[delta, 4.0]]), N_BALLS)[0]
        cue = np.array(STRAIGHT_IN["cue_m"])
        ref = np.array(STRAIGHT_IN["balls_m"][0]) - cue
        phi = np.arctan2(ref[1], ref[0]) + delta
        s = geometric_preshot_score(cue, np.array([np.cos(phi), np.sin(phi)]),
                                    {str(i): np.array(b) for i, b in enumerate(STRAIGHT_IN["balls_m"])},
                                    POCKET_CENTERS.numpy())
        assert f[3].item() == pytest.approx(min(-s / 0.5, 1.0), abs=1e-4)


class TestGeomSAC:
    def test_critic_shapes(self):
        c = GeomCritic(OBS_DIM, 2, N_BALLS)
        q1, q2 = c(torch.rand(5, OBS_DIM), torch.rand(5, 2))
        assert q1.shape == q2.shape == (5, 1)

    def test_one_update_step(self):
        ag = GeomSAC(OBS_DIM, 2, ACT_LOW, ACT_HIGH, n_balls=N_BALLS)
        B = 16
        batch = {
            "obs": np.random.rand(B, OBS_DIM).astype(np.float32),
            "action": np.random.uniform(ACT_LOW, ACT_HIGH, (B, 2)).astype(np.float32),
            "reward": np.random.randn(B, 1).astype(np.float32),
            "next_obs": np.random.rand(B, OBS_DIM).astype(np.float32),
            "done": np.zeros((B, 1), dtype=np.float32),
        }
        info = ag.update(batch)
        assert np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"])
