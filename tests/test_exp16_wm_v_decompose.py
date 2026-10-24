"""Tests for exp16_wm.v_decompose: pair records, Z / P / gap buckets, agreement with a reference predictor."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_decompose import bucket_masks, clear_counts, clear_truth, pair_table, summarize


def _set() -> dict[str, np.ndarray]:
    # 2 origins × 4 candidates. origin 0: k_next 0, 2, 2, 1 (all rankable). origin 1: only candidates 0 and 3 rankable (k 3 and 1)
    return {"k_next": np.array([[0, 2, 2, 1], [3, 9, 9, 1]]),
            "succ": np.array([[True] * 4, [True, False, True, True]]),
            "cleared": np.array([[False] * 4, [False, False, True, False]]),
            "valid": np.ones((2, 4), bool)}


def test_pair_table_counts_buckets_and_skips_ties_and_unrankable_candidates():
    t = _set()
    truth = np.array([[0, 2, 2, 1], [3, 0, 0, 1]], float)
    rec = pair_table(t, {"truth": truth, "anti": -truth})
    assert len(rec["kmin"]) == 6                                   # origin 0: 5 pairs ((2, 2) tied, skipped); origin 1: one pair (3 vs 1)
    m = bucket_masks(rec)
    assert m["Z"].sum() == 3 and m["P"].sum() == 3 and m["all"].sum() == 6
    assert m["gap1"].sum() == 3 and m["gap2"].sum() == 3 and m["gap3+"].sum() == 0      # gaps: origin 0 → 2, 2, 1, 1, 1; origin 1 → 2
    assert (rec["credit:truth"] == 1).all() and (rec["credit:anti"] == 0).all()


def test_ties_count_half_and_summary_agreement_and_disagreement_accuracy():
    t = _set()
    truth = np.array([[0, 2, 2, 1], [3, 0, 0, 1]], float)
    rec = pair_table(t, {"truth": truth, "flat": np.zeros_like(truth), "anti": -truth})
    s = summarize(rec, ["truth", "flat", "anti"], "truth")
    assert s["all"]["flat"] == pytest.approx(0.5) and s["all"]["truth"] == 1.0 and s["all"]["anti"] == 0.0
    assert s["all"]["agree:anti"] == 0.0 and s["all"]["acc_when_disagree:anti"] == 0.0     # always opposite to the reference, always wrong
    assert np.isnan(s["all"]["agree:flat"])                                                 # a predictor that ties everything has no untied pair
    assert s["Z"]["n"] == 3 and np.isnan(s["gap3+"]["truth"])                               # an empty bucket is NaN, not 0


def test_origin_without_two_rankable_candidates_gives_no_pairs():
    t = {"k_next": np.array([[1, 2]]), "succ": np.array([[True, False]]), "cleared": np.zeros((1, 2), bool), "valid": np.ones((1, 2), bool)}
    rec = pair_table(t, {"p": np.zeros((1, 2))})
    assert len(rec["kmin"]) == 0 and len(rec["credit:p"]) == 0


def test_clear_truth_and_counts_turn_balls_into_whole_rack_or_not():
    t = {"k_next": np.array([[3, 2, 0]]), "left_next": np.array([[3, 3, 3]])}
    np.testing.assert_array_equal(clear_truth(t)["k_next"], [[1, 0, 0]])
    assert t["k_next"][0, 0] == 3                                                  # the input is not modified
    kp = np.array([[3, 2, -1, 3], [0, 1, 2, 2]])
    np.testing.assert_array_equal(clear_counts(kp, np.array([3, 2])), [[1, 0, -1, 1], [0, 0, 1, 1]])     # unplaced stays −1
