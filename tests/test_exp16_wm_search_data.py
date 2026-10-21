"""Tests for exp16_wm.search_data: recorded search games for expert iteration."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.search_data import Recorder, runout_from_here, write_shard

OBS = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0, 0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1], np.float32)


def test_runout_from_here_is_one_only_when_every_later_shot_pockets():
    np.testing.assert_array_equal(runout_from_here([True, False, True, True]), [0, 0, 1, 1])
    np.testing.assert_array_equal(runout_from_here([True, True]), [1, 1])
    assert len(runout_from_here([])) == 0


def test_recorder_dedups_states_and_write_shard_puts_game_states_first(tmp_path):
    rec = Recorder()
    tree = OBS.copy(); tree[0] = 0.9
    made = np.zeros(18, bool); made[3] = True
    rec("sim", raw=tree, action=np.array([0.1, 1.0], np.float32), made_mask=made, scratch=False)       # tree state seen first
    g = rec.state_id(OBS)
    rec("top", raw=OBS, actions=np.array([[0.0, 1.0], [0.2, 2.0]], np.float32), pairs=np.array([3, 5]), scores=np.array([0.7, 0.1], np.float32))
    rec("sim", raw=OBS, action=np.array([0.0, 1.0], np.float32), made_mask=made, scratch=True)
    played = {"id": np.array([g]), "game": np.array([0]), "shot": np.array([0]), "action": np.array([[0.0, 1.0]], np.float32),
              "pocketed": np.array([1]), "v_target": np.array([1.0], np.float32)}
    lp, sp = write_shard(rec, played, str(tmp_path), "t", {"teacher": "search"})
    z, s = np.load(lp), np.load(sp)
    assert np.array_equal(z["obs"][0], OBS) and z["is_game_state"].tolist() == [True, False]
    assert z["v_target"][0] == 1.0 and np.isnan(z["v_target"][1])
    assert z["q_state"].tolist() == [0, 0] and z["q_pair"].tolist() == [3, 5] and np.allclose(z["q_score"], [0.7, 0.1])
    assert sorted(s["sim_state"].tolist()) == [0, 1] and s["sim_scratch_replay"].tolist().count(True) == 1
    from exp16_wm.bc import load_sims, load_teacher_data
    d = load_teacher_data([lp]); sims = load_sims([sp], [lp])
    assert len(d["obs"]) == 2 and len(sims["row"]) == 2
