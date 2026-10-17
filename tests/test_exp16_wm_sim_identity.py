"""Tests for exp16_wm/sim_identity.py."""

import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.sim_identity import to_bitmask, from_bitmask, replay

SIMS = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "logs/teacher_data/relabel_s1000/sims/*.npz")))


def test_bitmask_roundtrip_and_empty():
    made = np.zeros((4, 18), bool)
    made[1, 0] = made[2, 17] = made[3, [3, 9]] = True
    m = to_bitmask(made)
    assert m.dtype == np.int32 and m[0] == 0 and m[2] == 1 << 17
    assert np.array_equal(from_bitmask(m), made)


def test_replay_is_deterministic_and_counts_object_balls():
    from exp16_wm.two_ball_states import make_states
    obs = make_states(3, seed=0, placement="env")
    st = np.repeat(np.arange(3), 4)
    act = np.tile(np.array([[0.0, 1.0], [0.002, 1.0], [-0.002, 3.0], [0.5, 2.0]], np.float32), (3, 1))
    m1, s1 = replay(obs, st, act)
    m2, s2 = replay(obs, st, act)
    assert np.array_equal(m1, m2) and np.array_equal(s1, s2)
    made = from_bitmask(m1)
    live = obs[:, 2:11].reshape(-1, 3, 3)[:, :, 2] < 0.5
    for i, s in enumerate(st):                                  # pocketed slots belong to live balls only
        assert not made[i].reshape(3, 6)[~live[s]].any()


@pytest.mark.skipif(not SIMS, reason="relabel sims not present")
def test_replay_matches_the_stored_teacher_simulations():
    z = np.load(SIMS[0])
    idx = np.arange(0, 200)
    mask, scratch = replay(z["obs"], z["sim_state"][idx], z["sim_action"][idx])
    assert np.mean(scratch == z["sim_scratch"][idx]) > 0.99
    assert np.mean(from_bitmask(mask).sum(1) == z["sim_pocketed"][idx]) > 0.99
