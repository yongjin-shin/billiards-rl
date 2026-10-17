"""
tests/test_exp16_wm_augment.py

Unit tests for exp16_wm/augment.py — table-symmetry augmentation (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pooltool as pt

from simulator import BilliardsEnv
from exp16_wm.augment import reflect_obs, reflect_action, augment_flip4, FLIPS
from exp16_wm.obs_transform import relative_obs
from exp16_wm.planner import aim_reference, plan_shot, strike_phi_deg
from exp16_wm.networks import TABLE_W, TABLE_L

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
OBS = np.array([0.3, 0.4, 0.6, 0.7, 0, 0.1, 0.9, 0, 0, 0, 1] + POCKETS, dtype=np.float32)


class TestReflect:
    def test_lr_flips_x_of_live_balls_only(self):
        r = reflect_obs(OBS, lr=True, tb=False)
        np.testing.assert_allclose(r[[0, 2, 5]], [0.7, 0.4, 0.9])
        np.testing.assert_allclose(r[[1, 3, 6]], OBS[[1, 3, 6]])
        np.testing.assert_array_equal(r[8:11], [0, 0, 1])          # pocketed ball untouched
        np.testing.assert_array_equal(r[11:], OBS[11:])           # pockets untouched

    def test_double_reflection_is_identity(self):
        for lr, tb in FLIPS:
            np.testing.assert_allclose(reflect_obs(reflect_obs(OBS, lr, tb), lr, tb), OBS, atol=1e-7)

    def test_action_sign_rule_and_nan_padding(self):
        a = np.array([[0.2, 3.0], [np.nan, np.nan]], dtype=np.float32)
        np.testing.assert_allclose(reflect_action(a, True, False)[0], [-0.2, 3.0])
        np.testing.assert_allclose(reflect_action(a, False, True)[0], [-0.2, 3.0])
        np.testing.assert_allclose(reflect_action(a, True, True)[0], [0.2, 3.0])
        assert np.isnan(reflect_action(a, True, False)[1]).all()

    def test_single_flip_is_mirror_in_relative_frame(self):
        # one reflection = y → −y in the action-aligned frame; both = identity (rotation is factored out)
        rel = relative_obs(OBS).reshape(-1)
        balls, pockets = rel[:9].reshape(3, 3), rel[9:].reshape(6, 2)
        r_lr = relative_obs(reflect_obs(OBS, True, False))
        np.testing.assert_allclose(r_lr[:9].reshape(3, 3)[:, 1], -balls[:, 1], atol=1e-5)
        np.testing.assert_allclose(r_lr[:9].reshape(3, 3)[:, 0], balls[:, 0], atol=1e-5)
        r_both = relative_obs(reflect_obs(OBS, True, True))
        np.testing.assert_allclose(r_both[:9], rel[:9], atol=1e-5)
        assert pockets.shape == (6, 2)


class TestAugmentDataset:
    def test_four_copies_identity_first(self):
        n = 5
        data = {"obs": np.tile(OBS, (n, 1)), "action": np.tile([[0.1, 2.0]], (n, 1)).astype(np.float32),
                "good_actions": np.full((n, 2, 2), np.nan, np.float32),
                **{k: np.arange(n) for k in ["n_good", "pocketed", "shot", "game"]}}
        out = augment_flip4(data, np.array([0, 2]))
        assert len(out["obs"]) == 8
        np.testing.assert_array_equal(out["obs"][:2], data["obs"][[0, 2]])
        np.testing.assert_allclose(out["action"][2:4, 0], [-0.1, -0.1])
        np.testing.assert_array_equal(out["game"], [0, 2] * 4)

    def test_empty_index(self):
        data = {"obs": OBS[None], "action": np.zeros((1, 2), np.float32),
                "good_actions": np.full((1, 1, 2), np.nan, np.float32),
                **{k: np.zeros(1, int) for k in ["n_good", "pocketed", "shot", "game"]}}
        assert len(augment_flip4(data, np.array([], dtype=int))["obs"]) == 0


class TestPhysics:
    def test_reflected_shot_pockets_same_balls(self):
        """pooltool check: a teacher shot and its mirror pocket the same balls."""
        env = BilliardsEnv(n_balls=3, max_steps=30, step_penalty=1.0, trunc_penalty=0.0)
        env.reset(seed=21)
        rng = np.random.default_rng(0)
        kw = dict(offsets_deg=[0.0], speeds=[3.0], max_cut_deg=80.0, n_random=20)
        for _ in range(3):
            env.reset()
            a, meta = plan_shot(env, rng, **kw)
            for lr, tb in FLIPS[1:]:
                sys_r = env.system.copy()
                for b in sys_r.balls.values():
                    x, y = b.state.rvw[0, 0], b.state.rvw[0, 1]
                    b.state.rvw[0, 0], b.state.rvw[0, 1] = (TABLE_W - x if lr else x), (TABLE_L - y if tb else y)
                cue = sys_r.balls["cue"].state.rvw[0, :2]
                ref = min((sys_r.balls[b].state.rvw[0, :2] for b in env._ball_ids),
                          key=lambda p: np.linalg.norm(p - cue))
                phi_ref = np.arctan2(ref[1] - cue[1], ref[0] - cue[0])
                ar = reflect_action(a, lr, tb)
                sys_r.strike(phi=float(np.degrees(phi_ref + float(ar[0]))), V0=float(ar[1]), cue_ball_id="cue")
                pt.simulate(sys_r, inplace=True, max_events=2000)
                n = sum(b not in sys_r.balls or sys_r.balls[b].state.s == pt.constants.pocketed
                        for b in env._ball_ids)
                assert n == meta["predicted"][0]
        env.close()
