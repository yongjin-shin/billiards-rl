"""Tests for simulator.play_shot — the guard around pooltool 0.5.0's degenerate-quartic ValueError."""

import os
import sys

import numpy as np
import pooltool as pt
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulator
from simulator import BilliardsEnv, play_shot
from exp16_wm.planner import aim_reference, strike_phi_deg
from exp16_wm.relabel import restore_state
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.two_ball_states import make_states

# Parity case 24364 (roadmap "pooltool 0.5.0 quartic solver 예외 조사"): aim exactly at the nearest ball (delta 0.0), 1.0 m/s.
# Both balls then roll the same way with exactly equal acceleration → the collision polynomial has a = b = 0.
FAILING_OBS = np.zeros(23, np.float32)
FAILING_OBS[:11] = [0.7763011, 0.25567922, 0.56265146, 0.3134699, 0.0, 0.62142193, 0.8054987, 0.0, 0.0, 0.0, 1.0]
FAILING_ACTION = np.array([0.0, 1.0], np.float32)


@pytest.fixture
def env() -> BilliardsEnv:
    e = BilliardsEnv(**CLEAR_ENV)
    e.reset(seed=0)
    return e


def _failing_phi(env: BilliardsEnv) -> float:
    restore_state(env, FAILING_OBS)
    return strike_phi_deg(aim_reference(env), FAILING_ACTION)


def _raw(env: BilliardsEnv, phi: float, v: float) -> pt.System:
    sys_c = env.system.copy()
    sys_c.strike(phi=phi, V0=v, cue_ball_id="cue")
    pt.simulate(sys_c, inplace=True, max_events=2000)
    return sys_c


def _final(system: pt.System) -> np.ndarray:
    return np.array([b.state.rvw[0, :2] for b in system.balls.values()])


@pytest.fixture
def known_bug(env: BilliardsEnv) -> float:
    """phi of the failing shot; skips the dependent tests if pooltool no longer raises on it (bug fixed upstream)."""
    phi = _failing_phi(env)
    try:
        _raw(env, phi, 1.0)
    except ValueError as err:
        assert "cannot assign slice" in str(err)
        return phi
    pytest.skip("pooltool no longer raises on the known degenerate shot")


def test_unguarded_shot_raises_and_play_shot_recovers(env: BilliardsEnv, known_bug: float) -> None:
    before = dict(simulator.QUARTIC_RETRIES)
    out = play_shot(env.system, known_bug, 1.0)
    assert simulator.QUARTIC_RETRIES["retried"] == before["retried"] + 1
    assert simulator.QUARTIC_RETRIES["recovered"] == before["recovered"] + 1
    # physically continuous: same as a shot a hair away that never triggered the bug (1e-5 deg)
    neighbour = _raw(env, known_bug + 1e-5, 1.0)
    assert np.abs(_final(out) - _final(neighbour)).max() < 1e-4


def test_play_shot_not_inplace_leaves_the_original_untouched(env: BilliardsEnv, known_bug: float) -> None:
    pos = _final(env.system).copy()
    out = play_shot(env.system, known_bug, 1.0, inplace=False)
    assert out is not env.system and np.array_equal(_final(env.system), pos) and len(env.system.events) == 0


def test_inplace_returns_same_object_normally_and_a_new_one_when_guarded(env: BilliardsEnv, known_bug: float) -> None:
    obs = make_states(1, 0, "env")[0]
    restore_state(env, obs)
    sys0 = env.system
    assert play_shot(sys0, 20.0, 2.0, inplace=True) is sys0 and len(sys0.events) > 0
    restore_state(env, FAILING_OBS)
    sys1 = env.system
    out = play_shot(sys1, known_bug, 1.0, inplace=True)
    assert out is not sys1 and len(out.events) > 0


def test_env_step_survives_the_degenerate_shot(env: BilliardsEnv, known_bug: float) -> None:
    restore_state(env, FAILING_OBS)
    _, reward, terminated, truncated, info = env.step(FAILING_ACTION)
    assert np.isfinite(reward) and isinstance(terminated, (bool, np.bool_)) and isinstance(truncated, (bool, np.bool_))
    env.step(np.array([0.3, 2.0], np.float32))  # the replaced system keeps working for the next shot


def test_normal_shots_are_bit_identical_to_raw_pooltool_and_not_counted(env: BilliardsEnv) -> None:
    before = dict(simulator.QUARTIC_RETRIES)
    rng = np.random.default_rng(0)
    obs = make_states(6, 1, "uniform")
    for o in obs:
        restore_state(env, o)
        for _ in range(3):
            phi, v = float(rng.uniform(-180, 180)), float(rng.uniform(0.5, 8.0))
            ref, got = _raw(env, phi, v), play_shot(env.system, phi, v)
            assert len(ref.events) == len(got.events)
            assert all(np.array_equal(ref.balls[k].state.rvw, got.balls[k].state.rvw) for k in ref.balls)
    assert simulator.QUARTIC_RETRIES == before


def test_other_errors_are_raised_unchanged_without_retry(env: BilliardsEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(1)
        raise ValueError("something unrelated")

    monkeypatch.setattr(pt, "simulate", boom)
    before = dict(simulator.QUARTIC_RETRIES)
    with pytest.raises(ValueError, match="unrelated"):
        play_shot(env.system, 10.0, 1.0)
    assert len(calls) == 1 and simulator.QUARTIC_RETRIES == before


def test_bug_that_survives_every_nudge_is_raised_after_all_retries(env: BilliardsEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def always(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(1)
        raise ValueError("cannot assign slice of shape (2,) from input of shape (4,)")

    monkeypatch.setattr(pt, "simulate", always)
    before = dict(simulator.QUARTIC_RETRIES)
    with pytest.raises(ValueError, match="cannot assign slice"):
        play_shot(env.system, 10.0, 1.0)
    assert len(calls) == 1 + len(simulator.QUARTIC_RETRY_NUDGES_RAD)
    assert simulator.QUARTIC_RETRIES["retried"] == before["retried"] + 1 and simulator.QUARTIC_RETRIES["recovered"] == before["recovered"]
