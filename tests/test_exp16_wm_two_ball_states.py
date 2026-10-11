"""Tests for exp16_wm/two_ball_states.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.two_ball_states import make_states, to_relabel_input, POCKET_CLEARANCE
from exp16_wm.relabel import restore_state, KEYS
from exp16_wm.teacher_data import CLEAR_ENV


def _balls(obs):
    return obs[:, 2:11].reshape(-1, 3, 3)


@pytest.mark.parametrize("placement", ["env", "uniform"])
def test_exactly_one_pocketed_slot_and_it_is_zeroed(placement):
    obs = make_states(60, seed=0, placement=placement)
    b = _balls(obs)
    assert obs.shape == (60, 23)
    assert ((b[:, :, 2] > 0.5).sum(1) == 1).all()
    assert np.allclose(b[b[:, :, 2] > 0.5][:, :2], 0.0)
    assert np.bincount(np.argmax(b[:, :, 2], 1), minlength=3).min() > 5      # every slot gets pocketed sometimes


def test_env_placement_stays_in_the_interior_and_uniform_reaches_the_cushions():
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    W, L = env.table_width, env.table_length
    for placement, near_expected in (("env", False), ("uniform", True)):
        obs = make_states(300, seed=1, placement=placement)
        b = _balls(obs)
        live = b[b[:, :, 2] < 0.5][:, :2] * np.array([W, L])
        d_cush = np.min(np.stack([live[:, 0], W - live[:, 0], live[:, 1], L - live[:, 1]]), 0)
        assert (np.mean(d_cush < 0.1) > 0.05) == near_expected
    env.close()


def test_uniform_balls_do_not_overlap_and_stay_clear_of_pockets():
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    W, L, R = env.table_width, env.table_length, env.system.balls["cue"].params.R
    pockets = env._pocket_obs.reshape(-1, 2) * np.array([W, L])
    obs = make_states(100, seed=2, placement="uniform")
    for o in obs:
        b = o[2:11].reshape(3, 3)
        pts = np.vstack([o[:2]] + [b[k, :2] for k in range(3) if b[k, 2] < 0.5]) * np.array([W, L])
        dd = np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(len(pts), 1)]
        assert dd.min() > 2 * R
        assert np.linalg.norm(pts[:, None] - pockets[None], axis=-1).min() >= POCKET_CLEARANCE
    env.close()


@pytest.mark.parametrize("placement", ["env", "uniform"])
def test_restored_table_reproduces_the_observation(placement):
    env = BilliardsEnv(**CLEAR_ENV); env.reset(seed=0)
    for o in make_states(10, seed=3, placement=placement):
        restore_state(env, o)
        assert np.allclose(np.asarray(env._get_obs(), np.float32), o, atol=1e-6)
    env.close()


def test_relabel_input_has_the_keys_relabel_reads_and_rejects_bad_placement():
    d = to_relabel_input(make_states(4, seed=0, placement="env"))
    assert set(KEYS) <= set(d) and len(np.unique(d["game"])) == 4
    with pytest.raises(ValueError):
        make_states(2, seed=0, placement="corner")
