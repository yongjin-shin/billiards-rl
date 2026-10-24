"""Tests for exp16_wm.v_perturb: moving the cue ball by ε, and the repeat / ceiling / same metrics of the moved play-outs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_perturb import perturb_cue, perturb_metrics, within_spearman

W, L, R = 0.9906, 1.9812, 0.028575
POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
OBS = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 1, *POCKETS], np.float32)   # ball 3 pocketed


def test_perturb_cue_moves_only_the_cue_ball_by_eps():
    rng = np.random.default_rng(0)
    for eps in (0.002, 0.01):
        out = perturb_cue(OBS, eps, rng, W, L, R)
        d = np.hypot((out[0] - OBS[0]) * W, (out[1] - OBS[1]) * L)
        assert abs(d - eps) < 1e-6
        np.testing.assert_array_equal(out[2:], OBS[2:])
        assert out is not OBS and OBS[0] == np.float32(0.5)       # the input is not changed


def test_perturb_cue_stays_on_the_table_and_off_other_balls():
    rng = np.random.default_rng(1)
    corner = OBS.copy(); corner[0], corner[1] = R / W, R / L        # cue ball touching two cushions
    for _ in range(50):
        out = perturb_cue(corner, 0.01, rng, W, L, R)
        x, y = out[0] * W, out[1] * L
        assert R - 1e-6 <= x <= W - R + 1e-6 and R - 1e-6 <= y <= L - R + 1e-6
    near = OBS.copy(); near[0], near[1] = (0.3 * W + 2.2 * R) / W, 0.6               # 2.2 radii from ball 1
    for _ in range(50):
        out = perturb_cue(near, 0.01, rng, W, L, R)
        assert np.hypot(out[0] * W - 0.3 * W, out[1] * L - 0.6 * L) >= 2 * R


def test_perturb_cue_ignores_pocketed_balls_and_gives_none_when_boxed_in():
    rng = np.random.default_rng(2)
    on_pocketed = OBS.copy(); on_pocketed[0], on_pocketed[1] = 0.7, 0.7            # where pocketed ball 3's stale coordinates are
    assert perturb_cue(on_pocketed, 0.01, rng, W, L, R) is not None
    assert perturb_cue(OBS, 5.0, rng, W, L, R) is None                             # every 5 m move leaves the table


def test_metrics_perfect_when_moving_changes_nothing():
    rng = np.random.default_rng(3)
    left = np.repeat([2, 4], 50)
    k = np.array([rng.integers(0, n + 1) for n in left])
    m = perturb_metrics(k, left, np.repeat(k[:, None], 4, axis=1))
    assert m["same"] == 1.0 and m["placed"] == 1.0
    assert m["ceiling"] > 0.999 and m["repeat"] > 0.999


def test_metrics_near_zero_for_unrelated_copies_and_handles_unplaced():
    rng = np.random.default_rng(4)
    left = np.repeat([3, 5], 400)
    k = np.array([rng.integers(0, n + 1) for n in left])
    kp = np.stack([[rng.integers(0, n + 1) for _ in range(8)] for n in left])
    kp[:20] = -1                                                   # 20 positions where no copy could be placed
    m = perturb_metrics(k, left, kp)
    assert abs(m["ceiling"]) < 0.1 and abs(m["repeat"]) < 0.1
    assert abs(m["placed"] - (1 - 20 / 800)) < 1e-9


def test_metrics_horizon_caps_labels():
    left = np.array([4] * 30)
    k = np.array([0, 1, 2, 3, 4] * 6)
    kp = np.repeat(np.minimum(k, 2)[:, None], 2, axis=1) + np.array([0, 0])          # copies agree only below the cap
    assert perturb_metrics(k, left, kp, horizon=2)["same"] == 1.0
    assert perturb_metrics(k, left, kp, horizon=0)["same"] < 1.0


def test_within_spearman_is_nan_without_variation():
    left = np.array([2] * 20)
    assert np.isnan(within_spearman(np.ones(20), np.arange(20.0), left))


def test_testset_order_metrics_perfect_and_reversed_copies():
    from exp16_wm.v_perturb import testset_order_metrics
    t = {"valid": np.ones((2, 3), bool), "succ": np.array([[True, True, True], [True, True, False]]),
         "cleared": np.zeros((2, 3), bool), "k_next": np.array([[3, 1, 2], [0, 4, -1]])}
    use = t["valid"] & t["succ"] & ~t["cleared"]
    k = t["k_next"][use]                                                   # row-major order of the rankable positions
    m = testset_order_metrics(t, np.repeat(k[:, None], 4, axis=1))
    assert m["order_ceiling"] == 1.0 and m["order_repeat"] == 1.0 and m["placed"] == 1.0
    rev = testset_order_metrics(t, np.repeat((10 - k)[:, None], 2, axis=1))
    assert rev["order_ceiling"] == 0.0
    part = np.repeat(k[:, None], 2, axis=1); part[0] = -1                    # one position never placed: dropped, not a crash
    assert testset_order_metrics(t, part)["placed"] == 0.8
