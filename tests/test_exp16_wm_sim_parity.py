"""Tests for exp16_wm/sim_parity.py."""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pooltool as pt
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.outcomes import pocket_index_map
from exp16_wm.sim_parity import POS_CLOSE_M, compare, load, make_cases, run, simulate_case
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.two_ball_states import make_states

TINY = dict(n_states=2, n_random=3, n_scan_states=1, scan_half_deg=0.1, scan_step_deg=0.05, scan_speeds=(1.0,), seed=0)


def test_make_cases_shapes_and_determinism() -> None:
    a = make_cases(n_states=3, n_random=4, n_scan_states=2, scan_half_deg=0.1, scan_step_deg=0.05, scan_speeds=(1.0, 3.0), seed=0)
    b = make_cases(n_states=3, n_random=4, n_scan_states=2, scan_half_deg=0.1, scan_step_deg=0.05, scan_speeds=(1.0, 3.0), seed=0)
    n = 3 * 4 + 2 * 2 * 5  # random shots + (scan states x speeds x 5 angles)
    assert a["obs"].shape == (5, 23) and a["action"].shape == (n, 2) and a["state_idx"].shape == (n,) and a["group"].shape == (n,)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert (a["group"][:12] == -1).all() and sorted(set(a["group"][12:])) == [0, 1, 2, 3]
    scan_delta = a["action"][12:17, 0]
    assert np.allclose(scan_delta, np.radians([-0.1, -0.05, 0.0, 0.05, 0.1]), atol=1e-6)  # symmetric scan around the aim reference
    assert 0.5 <= a["action"][:12, 1].min() and a["action"][:12, 1].max() <= 8.0


def test_make_cases_empty() -> None:
    c = make_cases(0, 5, 0, 1.0, 0.5, (1.0,), 0)
    assert c["action"].shape == (0, 2) and c["state_idx"].shape == (0,) and len(c["obs"]) == 0


@pytest.fixture(scope="module")
def dump(tmp_path_factory: pytest.TempPathFactory) -> dict[str, np.ndarray]:
    out = str(tmp_path_factory.mktemp("parity") / "a.npz")
    run(make_cases(**TINY), out, log_every=0)
    return load(out)


def test_run_is_deterministic_and_records_machine(dump: dict[str, np.ndarray], tmp_path: Path) -> None:
    again = str(tmp_path / "b.npz")
    run(make_cases(**TINY), again, log_every=0)
    report = compare(dump, load(again))
    assert report["verdict"] == "IDENTICAL" and report["n_cases"] == 11 and report["n_scan_cases"] == 5
    assert "pooltool-billiards" in str(dump["meta"]) and (dump["n_events"] > 0).all()


def test_compare_flags_tiny_position_noise_as_close_not_identical(dump: dict[str, np.ndarray]) -> None:
    other = copy.deepcopy(dump)
    other["final_xy"] = other["final_xy"] + 1e-12
    assert compare(dump, other)["verdict"] == "IDENTICAL"  # below 1e-9 m
    other["final_xy"] = dump["final_xy"] + POS_CLOSE_M / 10
    assert compare(dump, other)["verdict"] == "CLOSE"
    other["final_xy"] = dump["final_xy"] + POS_CLOSE_M * 10
    assert compare(dump, other)["verdict"] == "DIFFERENT"


def test_compare_flags_outcome_flip_in_a_scan(dump: dict[str, np.ndarray]) -> None:
    other = copy.deepcopy(dump)
    scan_i = int(np.where(dump["group"] >= 0)[0][0])
    other["mask"][scan_i] = 1 if dump["mask"][scan_i] == 0 else 0
    r = compare(dump, other)
    assert r["verdict"] == "DIFFERENT" and r["n_outcome_differs"] == 1
    assert r["n_scan_made_flips"] == 1 and r["n_scan_groups_with_flip"] == 1
    assert r["first_mismatches"][0]["case"] == scan_i


def test_compare_flags_different_event_sequence(dump: dict[str, np.ndarray]) -> None:
    other = copy.deepcopy(dump)
    other["ev_crc"][0] += 1
    r = compare(dump, other)
    assert r["verdict"] == "DIFFERENT" and r["n_event_sequence_differs"] == 1 and r["n_outcome_differs"] == 0


def test_compare_rejects_dumps_of_different_cases(dump: dict[str, np.ndarray]) -> None:
    other = copy.deepcopy(dump)
    other["action"][0, 0] += 0.01
    with pytest.raises(ValueError, match="different cases"):
        compare(dump, other)
    shorter = {k: (v[:-1] if k in ("state_idx", "action", "group") else v) for k, v in dump.items()}
    with pytest.raises(ValueError, match="different cases"):
        compare(dump, shorter)


def test_simulate_case_records_a_pooltool_exception_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise ValueError("quartic solver failed")

    monkeypatch.setattr(pt, "simulate", boom)
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    r = simulate_case(env, pocket_index_map(env), make_states(1, 0, "env")[0], np.array([0.0, 1.0], np.float32))
    assert r["error"] == 1 and "ValueError" in r["message"] and r["n_events"] == 0 and r["mask"] == 0


def test_compare_treats_matching_errors_as_equal_and_one_sided_errors_as_different(dump: dict[str, np.ndarray]) -> None:
    both = copy.deepcopy(dump), copy.deepcopy(dump)
    for d in both:
        d["error"][3], d["mask"][3], d["n_events"][3], d["ev_crc"][3] = 1, 0, 0, 0
        d["final_xy"][3] = 0.0
    r = compare(*both)
    assert r["verdict"] == "IDENTICAL" and r["n_errors"] == [1, 1]
    one_sided = copy.deepcopy(dump)
    one_sided["error"][3] = 1
    r = compare(dump, one_sided)
    assert r["verdict"] == "DIFFERENT" and r["n_errors"] == [0, 1] and r["n_outcome_differs"] == 1
