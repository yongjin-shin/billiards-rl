"""Tests for exp16_wm.v_testset: candidate values, same-origin pair accuracy, depth-1 pick value."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_testset import candidate_values, pair_accuracy, pick_value, evaluate_testset


def _set() -> dict[str, np.ndarray]:
    # 2 origins × 3 candidates; origin 0: candidates 0, 1 succeed (later balls 3 and 1), 2 fails; origin 1: 0 clears, 1 succeeds, 2 invalid
    return {"made": np.array([[0.9, 0.8, 0.7], [0.6, 0.9, 0.0]]),
            "succ": np.array([[True, True, False], [True, True, False]]),
            "balls": np.array([[1, 1, 0], [2, 1, 0]]),
            "k_next": np.array([[3, 1, -1], [-1, 0, -1]]),
            "cleared": np.array([[False, False, False], [True, False, False]]),
            "valid": np.array([[True, True, True], [True, True, False]])}


def test_candidate_values_zero_for_failed_shots_and_balls_only_when_cleared():
    t = _set()
    np.testing.assert_array_equal(candidate_values(t["succ"], t["balls"], t["k_next"]), [[4, 2, 0], [2, 1, 0]])


def test_pair_accuracy_perfect_reversed_ties_and_empty():
    truth = np.array([[3, 1, 2]]); use = np.ones((1, 3), bool)
    assert pair_accuracy(np.array([[0.9, 0.1, 0.5]]), truth, use) == 1.0
    assert pair_accuracy(np.array([[0.1, 0.9, 0.5]]), truth, use) == 0.0
    assert pair_accuracy(np.zeros((1, 3)), truth, use) == 0.5
    assert np.isnan(pair_accuracy(np.zeros((1, 3)), np.ones((1, 3)), use))        # every truth tied: no pair counts
    assert np.isnan(pair_accuracy(np.zeros((1, 3)), truth, np.zeros((1, 3), bool)))


def test_pair_accuracy_only_compares_within_a_row():
    pred = np.array([[1.0, 0.0], [0.0, 1.0]]); truth = np.array([[5, 4], [1, 0]])     # second row reversed
    assert pair_accuracy(pred, truth, np.ones((2, 2), bool)) == 0.5


def test_pick_value_skips_invalid_candidates_and_rows():
    value = np.array([[4.0, 2.0, 9.0], [1.0, 0.0, 0.0]])
    valid = np.array([[True, True, False], [False, False, False]])
    assert pick_value(np.array([[0.1, 0.2, 5.0], [1, 1, 1]]), value, valid) == 2.0


def test_evaluate_testset_on_a_hand_made_set():
    t = _set()
    good = np.array([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])               # V that knows the later balls
    m = evaluate_testset(t, good)
    assert m["pair_acc"] == 1.0
    assert m["best"] == (4 + 2) / 2
    assert m["greedy"] == (4 + 1) / 2                                    # made alone picks candidate 0, then candidate 1
    # origin 0: 0.9·(1+3) > 0.8·(1+1) → value 4; origin 1: a clearing shot scores made × 1 like lookahead (0.6) < 0.9·(1+0) → value 1
    assert m["pick"] == (4 + 1) / 2
    assert abs(m["random"] - ((4 + 2 + 0) / 3 + (2 + 1) / 2) / 2) < 1e-12


def test_evaluate_testset_bad_v_orders_wrongly():
    t = _set()
    m = evaluate_testset(t, np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]))
    assert m["pair_acc"] == 0.0
