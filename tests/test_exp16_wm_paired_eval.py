"""Tests for exp16_wm.paired_eval: paired game evaluation (same games for every model), summaries, paired differences, files."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.paired_eval import find_run, load, machine_info, paired_diff, play_paired, save, summarize


class FixedPolicy:
    """Always the same action; records the first observation of every game (shot index 0 is the first obs after a reset)."""

    def __init__(self, action: tuple[float, float]):
        self.a = np.array(action, np.float32)
        self.seen: list[np.ndarray] = []

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.seen.append(np.array(obs, np.float32))
        return self.a


def test_summarize_counts_games_success_and_balls():
    res = {"shots": np.array([3, 5]), "succ": np.array([1, 1, 0, 1, 0, 0, 1, 1], bool), "balls": np.array([1, 2, 0, 1, 0, 0, 1, 1])}
    s = summarize(res)
    assert s["games"] == 2
    assert s["shots"] == pytest.approx(4.0)
    assert s["shots_se"] == pytest.approx(np.std([3, 5]) / np.sqrt(2))
    assert s["shot_success"] == pytest.approx(5 / 8)
    assert s["balls_per_success"] == pytest.approx(6 / 5)


def test_summarize_without_a_successful_shot_gives_nan_balls():
    s = summarize({"shots": np.array([30]), "succ": np.zeros(30, bool), "balls": np.zeros(30, int)})
    assert s["shot_success"] == 0.0
    assert np.isnan(s["balls_per_success"])


def test_paired_diff_is_the_per_game_difference():
    a, b = np.array([3, 4, 5, 3]), np.array([3, 3, 4, 4])
    d, se = paired_diff(a, b)
    assert d == pytest.approx(0.25)
    assert se == pytest.approx(np.std([0, 1, 1, -1]) / 2)
    assert paired_diff(a, a) == (0.0, 0.0)


def test_paired_diff_refuses_different_games():
    with pytest.raises(ValueError):
        paired_diff(np.array([3, 4]), np.array([3, 4, 5]))


def test_play_paired_starts_every_eval_seed_from_the_same_layout():
    p1, p2 = FixedPolicy((0.5, 2.0)), FixedPolicy((2.5, 1.0))
    r1 = play_paired(p1, eval_seeds=(1, 2), games_per_seed=2)
    r2 = play_paired(p2, eval_seeds=(1, 2), games_per_seed=2)
    starts = lambda r: np.r_[0, np.cumsum(r["shots"])[:-1]]                # index of each game's first observation
    for g in (0, 2):                                                        # first game of eval seed 1 and of eval seed 2
        np.testing.assert_array_equal(p1.seen[starts(r1)[g]], p2.seen[starts(r2)[g]])
    for r in (r1, r2):
        assert len(r["shots"]) == 4 and r["shots"].sum() == len(r["succ"]) == len(r["balls"]) == len(r["game"])
        assert (r["shots"] >= 1).all() and (r["shots"] <= 30).all()        # CLEAR_ENV max_steps 30
        np.testing.assert_array_equal(np.bincount(r["game"]), r["shots"])
        assert ((r["balls"] > 0) == r["succ"]).all()


def test_play_paired_is_deterministic():
    a = play_paired(FixedPolicy((0.5, 2.0)), eval_seeds=(2,), games_per_seed=2)
    b = play_paired(FixedPolicy((0.5, 2.0)), eval_seeds=(2,), games_per_seed=2)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_save_load_roundtrip_keeps_results_and_meta(tmp_path):
    res = {"shots": np.array([3, 4]), "succ": np.ones(7, bool), "balls": np.ones(7, int), "game": np.array([0, 0, 0, 1, 1, 1, 1])}
    path = str(tmp_path / "sub" / "x.npz")
    save(path, res, {"run": "r", "machine": machine_info()})
    back, meta = load(path)
    assert set(back) == set(res)
    for k in res:
        np.testing.assert_array_equal(back[k], res[k])
    assert meta["run"] == "r" and meta["machine"]["machine"]


def test_load_reads_the_older_files_without_meta(tmp_path):
    path = str(tmp_path / "old.npz")
    np.savez(path, shots=np.array([3]), succ=np.ones(3, bool), balls=np.ones(3, int))
    res, meta = load(path)
    assert meta == {} and list(res["shots"]) == [3]


def test_find_run_needs_exactly_one_match(tmp_path):
    (tmp_path / "run_a_s0").mkdir(); (tmp_path / "run_b_s0").mkdir()
    assert find_run(str(tmp_path / "run_a_*")).endswith("run_a_s0")
    with pytest.raises(ValueError):
        find_run(str(tmp_path / "run_*"))
    with pytest.raises(ValueError):
        find_run(str(tmp_path / "nothing_*"))


def test_machine_info_names_the_versions_that_matter():
    info = machine_info()
    for k in ("system", "machine", "python", "cpus", "pooltool-billiards", "panda3d", "numba", "numpy", "torch"):
        assert k in info
    json.dumps(info)                                                         # stored in the npz meta


def test_runout_counts_games_without_a_miss():
    from exp16_wm.paired_eval import runout_flags
    shots = np.array([3, 4, 2])
    succ = np.array([1, 1, 1,  1, 0, 1, 1,  1, 1], bool)
    np.testing.assert_array_equal(runout_flags(shots, succ), [True, False, True])
    s = summarize({"shots": shots, "succ": succ, "balls": succ.astype(int)})
    assert s["runout"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        runout_flags(np.array([3]), np.ones(2, bool))


def test_runout_treats_a_scratch_as_a_miss_when_flags_exist():
    shots = np.array([3]); succ = np.ones(3, bool); balls = np.ones(3, int)
    assert summarize({"shots": shots, "succ": succ, "balls": balls})["runout"] == 1.0              # old files: no flags
    assert summarize({"shots": shots, "succ": succ, "balls": balls, "scratch": np.array([0, 1, 0], bool)})["runout"] == 0.0

