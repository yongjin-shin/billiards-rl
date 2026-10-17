"""
tests/test_exp16_wm_pair_labels.py

Unit tests for exp16_wm/pair_labels.py — which (ball, pocket) pair made each label (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulator import BilliardsEnv
from exp16_wm.pair_labels import assign_pairs, pairs_for_file
from exp16_wm.planner import aim_reference, ghost_aims, wrap_angle
from exp16_wm.teacher_data import CLEAR_ENV, collect, save


def _env(seed=61):
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=seed)
    env.reset()
    return env


class TestAssignPairs:
    def test_candidate_built_on_a_pair_gets_that_pair(self):
        env = _env()
        phi_ref = aim_reference(env)
        aims = ghost_aims(env, 80.0)
        assert aims
        good = np.full((len(aims) + 1, 2), np.nan, dtype=np.float32)
        for k, (bid, j, phi) in enumerate(aims):
            good[k] = [wrap_angle(phi + np.radians(0.3) - phi_ref), 2.0]      # 0.3° off the ghost aim
        out = assign_pairs(env, good, tol_deg=2.5)
        want = [(int(bid) - 1) * 6 + j for bid, j, _ in aims]
        # a pair whose aim is within 0.3°+ of another pair's would be ambiguous; only require the unambiguous ones
        phis = np.array([p for _, _, p in aims])
        for k, (_, _, phi) in enumerate(aims):
            others = np.degrees(np.abs([wrap_angle(float(phi - q)) for q in np.delete(phis, k)]))
            if len(others) == 0 or others.min() > 1.0:
                assert out[k] == want[k]
        assert out[-1] == -1                                                  # NaN padding stays unassigned
        env.close()

    def test_far_off_candidate_is_unassigned(self):
        env = _env()
        phi_ref = aim_reference(env)
        aims = ghost_aims(env, 80.0)
        # an angle at least 10° from every aim
        for delta in np.radians(np.arange(-170, 170, 5.0)):
            if min(abs(np.degrees(wrap_angle(float(phi_ref + delta - p)))) for _, _, p in aims) > 10:
                good = np.array([[delta, 1.0]], dtype=np.float32)
                assert assign_pairs(env, good)[0] == -1
                break
        else:
            raise AssertionError("no free angle found")
        env.close()

    def test_empty_aims_returns_all_unassigned(self):
        env = _env()
        out = assign_pairs(env, np.array([[0.1, 1.0]], dtype=np.float32), max_cut_deg=-1.0)
        assert out.tolist() == [-1]
        env.close()


class TestFile:
    def test_teacher_labels_are_assigned_and_written_back(self, tmp_path):
        data = collect(seed=62, n_games=2, teacher="ghost", env_kw=CLEAR_ENV, max_good=8, log_every=0)
        path = str(tmp_path / "s.npz")
        save(path, data, seed=62, teacher="ghost", env_kw=CLEAR_ENV)
        st = pairs_for_file(path)
        z = np.load(path)
        assert z["good_pairs"].shape == z["good_actions"].shape[:2]
        assert st["states_with_pair"] >= st["states"] * 0.9           # nearly every state's label has a pair
        # the chosen action (good[0]) belongs to a pair whose slot is a valid 0..17 index
        first = z["good_pairs"][:, 0]
        assert ((first >= -1) & (first < 18)).all()
        np.testing.assert_array_equal(z["action"], data["action"])    # other arrays preserved
