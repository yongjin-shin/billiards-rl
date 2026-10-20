"""Tests for N-ball support (G1: 3 → 6 → 9 balls): observation sizes, model shapes, slots, bitmasks, simulation."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.geometry_aux import aim_targets
from exp16_wm.gnn_policy import PairGNNPolicy, n_balls_of
from exp16_wm.obs_transform import n_balls_raw, n_balls_rel, relative_obs
from exp16_wm.sim_identity import from_bitmask, to_bitmask
from exp16_wm.teacher_data import clear_env_kw


@pytest.fixture(scope="module", params=[3, 6, 9])
def env_obs(request):
    from simulator import BilliardsEnv
    env = BilliardsEnv(**clear_env_kw(request.param))
    obs, _ = env.reset(seed=1)
    yield request.param, env, np.asarray(obs, np.float32)
    env.close()


def test_env_kwargs_scale_the_shot_limit():
    assert clear_env_kw(3)["max_steps"] == 30 and clear_env_kw(9) == {**clear_env_kw(3), "n_balls": 9, "max_steps": 90}


def test_observation_sizes_and_ball_counts(env_obs):
    n, _, obs = env_obs
    assert obs.shape == (2 + 3 * n + 12,) and n_balls_raw(len(obs)) == n
    rel = relative_obs(obs)
    assert rel.shape == (3 * n + 12,) and n_balls_rel(len(rel)) == n
    with pytest.raises(ValueError):
        n_balls_raw(len(obs) + 1)


def test_model_and_aim_targets_have_six_pairs_per_ball(env_obs):
    n, _, obs = env_obs
    rel = relative_obs(np.stack([obs, obs]))
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True).eval()
    m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = "all_pairs", (-0.1, 0.0, 0.1), (1.0,), (1.0, 2.0)
    with torch.no_grad():
        o = m.forward_full(torch.as_tensor(rel))
        a = m.predict_unit(torch.as_tensor(rel))
    assert n_balls_of(torch.as_tensor(rel)) == n
    assert o["e"].shape[1] == 6 * n and o["valid"].shape == (2, 6 * n) and o["aim"].shape == (2, 6 * n) and a.shape == (2, 2)
    t, v = aim_targets(rel)
    assert t.shape == (2, 6 * n, 1) and v.shape == (2, 6 * n) and v.all()


def test_same_weights_play_any_number_of_balls():
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True).eval()
    pockets = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
    obs3 = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0, *pockets], np.float32)
    obs6 = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0, *[0, 0, 1] * 3, *pockets], np.float32)   # 3 extra pocketed balls
    with torch.no_grad():
        e3 = m.forward_full(torch.as_tensor(relative_obs(obs3))[None])
        e6 = m.forward_full(torch.as_tensor(relative_obs(obs6))[None])
    torch.testing.assert_close(e6["e"][:, :18], e3["e"])                       # pocketed balls change nothing for the live ones
    assert not e6["valid"][0, 18:].any()


@pytest.mark.parametrize("slots", [18, 36, 54])
def test_bitmask_roundtrip_for_up_to_nine_balls(slots):
    rng = np.random.default_rng(0)
    made = rng.random((5, slots)) < 0.2
    m = to_bitmask(made)
    assert m.dtype == (np.int32 if slots <= 31 else np.int64)
    np.testing.assert_array_equal(from_bitmask(m, slots), made)


def test_made_slots_simulates_any_rack(env_obs):
    from exp16_wm.frozen_verifier import _simulate_full
    n, _, obs = env_obs
    made, scr = _simulate_full((obs, np.array([[0.0, 2.0], [0.5, 5.0]], np.float32)))
    assert made.shape == (2, 6 * n) and scr.shape == (2,)
