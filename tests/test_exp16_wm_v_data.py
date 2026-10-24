"""Tests for exp16_wm.v_data: ball-count V labels (balls before the first miss), branch shots, play-outs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_data import branch_action, future_balls, pocketed_count, rollout

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
OBS = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0, *POCKETS], np.float32)   # 3 balls on the table


def test_future_balls_counts_until_the_first_miss():
    ok = np.array([True, True, False, True, True])
    balls = np.array([1, 2, 0, 1, 1])
    np.testing.assert_array_equal(future_balls(ok, balls), [3, 2, 0, 2, 1])


def test_future_balls_empty_and_all_misses():
    assert len(future_balls(np.array([], bool), np.array([], int))) == 0
    np.testing.assert_array_equal(future_balls(np.array([False, False]), np.array([0, 0])), [0, 0])


def _pocket(raw, i):
    nxt = raw.copy(); nxt[2 + 3 * i + 2] = 1.0
    return nxt


def test_rollout_stops_at_the_first_miss_and_counts_balls():
    outcomes = iter([(1, True), (None, False)])                  # pocket ball 1, then miss
    def sim(raw, a):
        i, ok = next(outcomes)
        nxt = _pocket(raw, i) if ok else raw.copy()
        return nxt, ok, False, False
    assert rollout(sim, lambda o: np.zeros(2), OBS, max_shots=3) == 1


def test_rollout_stops_when_the_rack_is_cleared_and_on_scratch():
    def clear(raw, a):
        nxt = _pocket(raw, next(i for i in range(3) if raw[2 + 3 * i + 2] < 0.5))
        return nxt, True, False, pocketed_count(nxt) == 3
    assert rollout(clear, lambda o: np.zeros(2), OBS, max_shots=10) == 3
    scratch = lambda raw, a: (_pocket(raw, 0), True, True, False)  # pocketed but the cue ball went in: a failure
    assert rollout(scratch, lambda o: np.zeros(2), OBS, max_shots=3) == 0


def test_branch_action_never_picks_the_greedy_shot():
    probs = np.array([0.1, 0.9, 0.5, 0.3])
    acts = np.arange(8, dtype=np.float32).reshape(4, 2)
    rng = np.random.default_rng(0)
    picked = {tuple(branch_action(probs, acts, 3, rng)) for _ in range(50)}
    assert picked == {tuple(acts[2]), tuple(acts[3])}               # top-3 = [1, 2, 3] without the best (1)
    assert branch_action(np.array([0.5]), acts[:1], 3, rng) is None
