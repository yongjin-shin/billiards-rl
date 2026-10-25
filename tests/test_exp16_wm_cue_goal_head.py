"""Tests for exp16_wm.cue_goal_head (CT2a: goal-conditioned head trained on hindsight-relabelled cue-only shots)."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_head import (ANGLE_TOL_RAD, LOGV_TOL, Head, eval_cases, features, load_rows, loss_fn, lookup_action, play_error,
                                    predict_action, summarize, targets, train)
from exp16_wm.single_ball import make_cue_only_system, shoot


def _fake_npz(path, n_starts=6, per=200, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform([0.1, 0.1], [0.8, 1.8], (n_starts, 2))
    n = n_starts * per
    k = rng.integers(0, 5, n).astype(np.int16)
    end = rng.uniform([0.05, 0.05], [0.9, 1.9], (n, 2)).astype(np.float32)
    scratch = rng.random(n) < 0.1
    end[5] = (-1.0, 1.0)                              # off the table
    np.savez(path, starts=starts, start_idx=np.repeat(np.arange(n_starts), per), phi=rng.uniform(0, 360, n).astype(np.float32),
             v=rng.uniform(0.01, 8, n).astype(np.float32), end=end, k=k, scratch=scratch)
    return k, scratch, end


def test_features_hold_the_vector_and_its_length():
    s = np.array([[0.2, 0.4]], np.float32); g = np.array([[0.5, 0.8]], np.float32)
    f = features(s, g)
    assert f.shape == (1, 7) and np.allclose(f[0, 4:6], [0.3, 0.4]) and np.isclose(f[0, 6], 0.5)


def test_targets_unit_vector_and_log_speed():
    t = targets(np.array([0.0, 90.0, 225.0]), np.array([1.0, np.e, 0.5]))
    assert np.allclose(np.linalg.norm(t[:, :2], axis=1), 1.0, atol=1e-6) and np.allclose(t[:, 2], [0.0, 1.0, np.log(0.5)], atol=1e-6)


def test_load_rows_filters_and_keeps_holdout_out_of_training(tmp_path):
    p = str(tmp_path / "d.npz"); k, scratch, end = _fake_npz(p)
    d = load_rows(p, k_bin=0, n_holdout=2, n_train=10_000, seed=3)
    rows = d["rows"]
    assert len(set(d["holdout"]) & set(d["train_starts"])) == 0 and len(d["holdout"]) == 2
    assert np.isin(d["sidx"][rows], d["train_starts"]).all()                       # no row from a held-out start
    assert (k[rows] == 0).all() and not scratch[rows].any() and 5 not in rows       # k filter, scratch and off-table rows dropped
    again = load_rows(p, 0, 2, 10_000, 3)
    assert np.array_equal(again["rows"], rows) and np.array_equal(again["holdout"], d["holdout"])   # deterministic
    three = load_rows(p, 3, 2, 10_000, 3)
    assert (np.minimum(k[three["rows"]], 3) == 3).all()                            # "3" means 3 or more


def test_loss_is_zero_at_target_and_grows_with_error():
    tgt = torch.from_numpy(targets(np.array([30.0, 200.0]), np.array([1.0, 2.0])))
    assert float(loss_fn(tgt.clone(), tgt)) == 0.0
    off = tgt.clone(); off[:, :2] = torch.from_numpy(targets(np.array([30.0, 200.0]) + 1.0, np.array([1.0, 2.0])))[:, :2]
    small = tgt.clone(); small[:, :2] = torch.from_numpy(targets(np.array([30.0, 200.0]) + 0.1, np.array([1.0, 2.0])))[:, :2]
    assert float(loss_fn(off, tgt)) > float(loss_fn(small, tgt)) > 0.0
    slow = tgt.clone(); slow[:, 2] += 10 * LOGV_TOL
    assert float(loss_fn(slow, tgt)) > 1.0 and ANGLE_TOL_RAD > 0


def test_play_error_is_zero_for_the_action_that_made_the_shot():
    system = make_cue_only_system(0.2, 0.5)
    r = shoot(system, 33.0, 0.25)
    err = play_error(np.array([[0.2, 0.5]]), np.array([[r.x, r.y]]), np.array([33.0]), np.array([0.25]))
    assert err[0] < 1e-9
    wrong = play_error(np.array([[0.2, 0.5]]), np.array([[r.x, r.y]]), np.array([60.0]), np.array([0.25]))
    assert wrong[0] > 0.05


def test_play_error_is_inf_for_a_scratch():
    err = play_error(np.array([[0.5, 0.5]]), np.array([[0.5, 1.0]]), np.array([225.0]), np.array([2.0]))   # straight into the corner pocket
    assert np.isinf(err[0])


def test_summarize_shares():
    s = summarize(np.array([0.005, 0.015, 0.03, 0.2]))
    assert s["within_1cm"] == 0.25 and s["within_2cm"] == 0.5 and s["within_5cm"] == 0.75 and s["within_10cm"] == 0.75
    assert np.isclose(s["median_cm"], 2.25)


def test_eval_cases_are_deterministic_and_belong_to_their_start():
    starts = np.array([[0.3, 0.5], [0.6, 1.2], [0.4, 0.9]], np.float32)
    s, g = eval_cases(starts, np.array([2, 0]), 5, seed=1)
    s2, g2 = eval_cases(starts, np.array([2, 0]), 5, seed=1)
    assert s.shape == g.shape == (10, 2) and np.array_equal(g, g2)
    assert (s[:5] == starts[2]).all() and (s[5:] == starts[0]).all()


def test_training_on_a_learnable_toy_problem_lowers_the_loss():
    # toy "physics": the action is a smooth function of (start, goal); the head must fit it
    rng = np.random.default_rng(0)
    n = 4000
    start = rng.uniform([0.1, 0.1], [0.8, 1.8], (n, 2)).astype(np.float32); goal = rng.uniform([0.1, 0.1], [0.8, 1.8], (n, 2)).astype(np.float32)
    d = goal - start
    phi = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0; v = np.sqrt(np.linalg.norm(d, axis=1) / 7.76) + 0.05
    data = {"starts": start, "sidx": np.arange(n), "rows": np.arange(n), "end": goal, "phi": phi.astype(np.float32), "v": v.astype(np.float32)}
    logs: list[str] = []
    model = train(data, epochs=6, batch=256, lr=2e-3, width=64, depth=3, seed=0, log=logs.append)
    first, last = float(logs[0].split("loss ")[1].split()[0]), float(logs[-1].split("loss ")[1].split()[0])
    assert last < first
    ph, vv = predict_action(model, start[:50], goal[:50])
    assert ph.shape == (50,) and (ph >= 0).all() and (ph < 360).all() and (vv >= 1e-4).all() and (vv <= 8.0).all()


def test_lookup_baseline_replays_the_recorded_action_of_the_nearest_start():
    starts = np.array([[0.2, 0.2], [0.8, 1.8]], np.float32)
    data = {"starts": starts, "train_starts": np.array([0, 1]), "sidx": np.array([0, 0, 1, 1]), "ok": np.ones(4, bool),
            "end": np.array([[0.3, 0.3], [0.5, 0.5], [0.7, 1.7], [0.6, 1.5]], np.float32),
            "phi": np.array([10.0, 20.0, 30.0, 40.0], np.float32), "v": np.array([0.1, 0.2, 0.3, 0.4], np.float32)}
    phi, v = lookup_action(data, np.array([[0.25, 0.25]], np.float32), np.array([[0.49, 0.49]], np.float32))
    assert phi[0] == 20.0 and np.isclose(v[0], 0.2)                                  # start 0 is nearest; its shot ending nearest the goal
    phi2, _ = lookup_action(data, np.array([[0.75, 1.75]], np.float32), np.array([[0.61, 1.49]], np.float32))
    assert phi2[0] == 40.0
    assert Head().x_mean.shape == (7,)


# ---------------------------------------------------------------- CT2b-2: the wall to hit as an extra input


def test_features_with_wall_appends_a_one_hot():
    from exp16_wm.cue_walls import N_WALLS
    s = np.array([[0.2, 0.4], [0.3, 0.5]], np.float32); g = np.array([[0.5, 0.8], [0.6, 0.9]], np.float32)
    f = features(s, g, np.array([0, 6]))
    assert f.shape == (2, 7 + N_WALLS) and f[0, 7] == 1.0 and f[0, 8:].sum() == 0.0 and f[1, 7 + 6] == 1.0 and f[:, 7:].sum() == 2.0
    assert np.array_equal(f[:, :7], features(s, g))


def test_load_rows_with_walls_keeps_only_labelled_one_cushion_rows(tmp_path):
    p = str(tmp_path / "d.npz"); k, scratch, end = _fake_npz(p)
    cand = np.nonzero((k == 1) & ~scratch & (np.arange(len(k)) != 5))[0]
    labelled = cand[::2]                                                   # half of the valid k = 1 rows have a label
    w = str(tmp_path / "w.npz"); np.savez(w, rows=labelled, wall=(np.arange(len(labelled)) % 7).astype(np.int8))
    d = load_rows(p, 1, 2, 10_000, 3, walls_path=w)
    assert set(d["rows"]) <= set(labelled.tolist()) and d["wall"].shape == k.shape
    assert (d["wall"][d["rows"]] >= 0).all() and (d["wall"][~np.isin(np.arange(len(k)), labelled)] == -1).all()
    assert load_rows(p, 0, 2, 10_000, 3)["wall"] is None


def test_head_with_extra_inputs_and_training_with_walls_lowers_the_loss():
    from exp16_wm.cue_walls import N_WALLS
    assert Head(32, 2, n_extra=N_WALLS).x_mean.shape == (7 + N_WALLS,)
    rng = np.random.default_rng(1); n = 4000
    start = rng.uniform([0.1, 0.1], [0.8, 1.8], (n, 2)).astype(np.float32); goal = rng.uniform([0.1, 0.1], [0.8, 1.8], (n, 2)).astype(np.float32)
    wall = rng.integers(0, N_WALLS, n)
    phi = (np.degrees(np.arctan2(*(goal - start).T[::-1])) + 30.0 * wall) % 360.0     # the action depends on the wall, so it must be an input
    v = 0.2 + 0.05 * wall
    data = {"starts": start, "sidx": np.arange(n), "rows": np.arange(n), "end": goal, "phi": phi.astype(np.float32), "v": v.astype(np.float32), "wall": wall.astype(np.int8)}
    logs: list[str] = []
    model = train(data, epochs=6, batch=256, lr=2e-3, width=64, depth=3, seed=0, log=logs.append)
    first, last = float(logs[0].split("loss ")[1].split()[0]), float(logs[-1].split("loss ")[1].split()[0])
    assert last < first and model.x_mean.shape == (7 + N_WALLS,)
    ph, vv = predict_action(model, start[:20], goal[:20], wall[:20])
    assert ph.shape == (20,) and (vv > 0).all()


def test_feasible_cases_lists_only_walls_with_a_recorded_shot_near_the_goal():
    from exp16_wm.cue_walls import N_WALLS
    starts = np.array([[0.5, 1.0]], np.float32)
    goals = sample_starts_for_test(3, 1000 + 7919)
    end = np.concatenate([goals[0:1] + 0.005, goals[0:1] + 0.1, goals[1:2] + 0.01]).astype(np.float32)     # near goal 0 (wall 2), far from goal 0 (wall 3), near goal 1 (wall 2)
    data = {"starts": starts, "sidx": np.zeros(3, int), "ok": np.ones(3, bool), "end": end, "wall": np.array([2, 3, 2], np.int8)}
    from exp16_wm.cue_goal_head import feasible_cases
    s, g, w, gid = feasible_cases(data, np.array([0]), 3, seed=1000)
    assert set(zip(gid.tolist(), w.tolist())) == {(0, 2), (1, 2)} and (s == starts[0]).all() and len(w) == 2 and N_WALLS == 7


def sample_starts_for_test(n, seed):
    from exp16_wm.single_ball import sample_starts
    return sample_starts(n, seed).astype(np.float32)


def test_best_per_goal_takes_the_smallest_error_of_each_goal():
    from exp16_wm.cue_goal_head import best_per_goal
    err = np.array([0.05, 0.01, 0.2, 0.3, 0.04]); gid = np.array([0, 0, 1, 2, 2])
    assert best_per_goal(err, gid).tolist() == [0.01, 0.2, 0.04]


def test_lookup_with_wall_replays_a_shot_of_that_wall():
    starts = np.array([[0.2, 0.2], [0.8, 1.8]], np.float32)
    data = {"starts": starts, "train_starts": np.array([0, 1]), "sidx": np.array([0, 0, 0, 1]), "ok": np.ones(4, bool),
            "end": np.array([[0.3, 0.3], [0.31, 0.31], [0.5, 0.5], [0.7, 1.7]], np.float32), "wall": np.array([1, 2, 2, 1], np.int8),
            "phi": np.array([10.0, 20.0, 30.0, 40.0], np.float32), "v": np.array([0.1, 0.2, 0.3, 0.4], np.float32)}
    phi, v = lookup_action(data, np.array([[0.25, 0.25]], np.float32), np.array([[0.3, 0.3]], np.float32), np.array([2]))
    assert phi[0] == 20.0                                      # wall 1's shot (10.0) ends nearer the goal, but wall 2 was asked for
    phi2, _ = lookup_action(data, np.array([[0.25, 0.25]], np.float32), np.array([[0.3, 0.3]], np.float32), np.array([5]))
    assert phi2[0] == 10.0                                     # no shot of wall 5: falls back to any shot of the nearest start
