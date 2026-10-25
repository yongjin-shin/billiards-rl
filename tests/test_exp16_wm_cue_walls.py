"""Tests for exp16_wm.cue_walls (CT2b-2: which wall a one-cushion shot hit)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import TABLE_L, TABLE_W, Params, collect_start
from exp16_wm.cue_walls import (N_WALLS, WALLS, _label_task, k1_rows, label, replay, replay_seq, route_code, route_digits, rows_with_k, wall_index,
                                wall_names)


def test_wall_names_from_contact_positions():
    ids = np.array(["a", "a", "b", "c", "d", "e", "f", "7t"])
    contact = np.array([[0.5, 0.03], [0.6, 0.03], [0.03, 0.5], [0.03, 1.4], [TABLE_W - 0.03, 0.4], [TABLE_W - 0.03, 1.6], [0.5, TABLE_L - 0.03], [0.1, 0.1]])
    n = wall_names(ids, contact)
    assert n == {"a": "아래 짧은 변", "b": "왼쪽 아래", "c": "왼쪽 위", "d": "오른쪽 아래", "e": "오른쪽 위", "f": "위 짧은 변", "7t": "포켓 모서리"}
    assert set(n.values()) == set(WALLS)                                           # every wall name is produced here
    assert wall_index(ids, n).tolist() == [WALLS.index(n[str(i)]) for i in ids] and N_WALLS == len(WALLS) == 7


def test_wall_names_use_the_median_so_one_stray_contact_does_not_rename_a_wall():
    ids = np.array(["x"] * 5); contact = np.array([[0.02, 0.5]] * 4 + [[0.5, 0.02]])
    assert wall_names(ids, contact) == {"x": "왼쪽 아래"}


def test_replay_reads_the_wall_a_real_shot_hit():
    start = np.array([0.5, 0.5])
    ids, contact, end = replay(start, np.array([180.0, 270.0, 90.0]), np.array([0.4, 0.4, 0.5]))   # toward x = 0, y = 0, y = L
    assert len(ids) == 3 and contact.shape == (3, 2) and end.shape == (3, 2)
    assert contact[0, 0] < 0.05 and abs(contact[0, 1] - 0.5) < 0.02                  # left wall at y = 0.5
    assert contact[1, 1] < 0.05 and contact[2, 1] > TABLE_L - 0.05
    names = wall_names(ids, contact)
    assert [names[str(i)] for i in ids] == ["왼쪽 아래", "아래 짧은 변", "위 짧은 변"]


def test_k1_rows_filters_cushion_count_scratch_and_off_table():
    z = {"end": np.array([[0.5, 0.5], [0.5, 0.5], [-1.0, 0.5], [0.5, 0.5]]), "scratch": np.array([False, False, False, True]),
         "k": np.array([1, 2, 1, 1], np.int16)}
    assert k1_rows(z).tolist() == [0]


def test_label_tiny_collection_end_to_end(tmp_path):
    prm = Params(n0=400, children=2, max_rounds=2, max_shots=1500)
    res = [collect_start((np.array(s), 3 + i, prm)) for i, s in enumerate([(0.4, 0.7), (0.7, 1.3)])]
    sizes = [len(r["k"]) for r in res]
    p = str(tmp_path / "d.npz")
    np.savez(p, starts=np.array([r["start"] for r in res]), start_idx=np.repeat(np.arange(2), sizes),
             **{key: np.concatenate([r[key] for r in res]) for key in ("phi", "v", "end", "k", "scratch")})
    out = str(tmp_path / "w.npz")
    label(p, out, workers=1)
    w = np.load(out)
    z = {k_: v_ for k_, v_ in np.load(p).items()}
    assert np.array_equal(w["rows"], k1_rows(z)) and len(w["wall"]) == len(w["rows"]) > 0
    assert w["wall"].min() >= 0 and w["wall"].max() < N_WALLS
    assert set(json.loads(str(w["id_names"])).values()) <= set(WALLS)
    ids, contact = _label_task((z["starts"][0], z["phi"][w["rows"][:3]], z["v"][w["rows"][:3]], 1))   # same helper the pool uses
    assert ids.shape == (3, 1) and contact.shape == (3, 1, 2) and int(w["n_seq"]) == 1


# ---------------------------------------------------------------- routes of two cushions (CT2c)


def test_route_code_round_trips_and_one_cushion_is_the_wall_index():
    w = np.array([[0, 0], [3, 5], [6, 6], [2, 0]])
    code = route_code(w)
    assert code.tolist() == [0, 3 * 7 + 5, 48, 14]
    assert np.array_equal(route_digits(code, 2), w)
    assert route_code(np.array([[4], [1]])).tolist() == [4, 1] and np.array_equal(route_digits(np.array([4, 1]), 1), [[4], [1]])


def test_replay_seq_reads_both_cushions_in_order():
    # from (0.5, 0.5) toward the left wall hard enough to bounce back to the right wall
    ids, contact, end = replay_seq(np.array([0.5, 0.5]), np.array([180.0, 180.0]), np.array([0.7, 0.75]), 2)
    k2 = [i for i in range(2) if (ids[i] != "").all()]
    assert ids.shape == (2, 2) and contact.shape == (2, 2, 2) and k2
    i = k2[0]
    assert contact[i, 0, 0] < 0.05                                              # first the left wall
    assert contact[i, 1, 0] > TABLE_W - 0.05                                    # then the right wall


def test_rows_with_k_selects_exact_cushion_count():
    z = {"end": np.array([[0.5, 0.5]] * 4), "scratch": np.array([False, False, True, False]), "k": np.array([1, 2, 2, 3], np.int16)}
    assert rows_with_k(z, 2).tolist() == [1] and rows_with_k(z, 1).tolist() == [0] and k1_rows(z).tolist() == [0]


def test_label_two_cushions_with_names_from_the_one_cushion_map(tmp_path):
    prm = Params(n0=1500, children=2, max_rounds=1, max_shots=3000)
    res = [collect_start((np.array(s), 5 + i, prm)) for i, s in enumerate([(0.4, 0.7), (0.7, 1.3)])]
    sizes = [len(r["k"]) for r in res]
    p = str(tmp_path / "d.npz")
    np.savez(p, starts=np.array([r["start"] for r in res]), start_idx=np.repeat(np.arange(2), sizes),
             **{key: np.concatenate([r[key] for r in res]) for key in ("phi", "v", "end", "k", "scratch")})
    w1 = str(tmp_path / "w1.npz"); label(p, w1, workers=1)
    w2 = str(tmp_path / "w2.npz"); label(p, w2, workers=1, k=2, names_from=w1)
    out = np.load(w2); z = {key: val for key, val in np.load(p).items()}
    assert np.array_equal(out["rows"], rows_with_k(z, 2)) and int(out["n_seq"]) == 2
    assert len(out["wall"]) == len(out["rows"]) > 0 and out["wall"].min() >= 0 and out["wall"].max() < N_WALLS ** 2
    names1 = json.loads(str(np.load(w1)["id_names"])); names2 = json.loads(str(out["id_names"]))
    assert all(names2[c] == n for c, n in names1.items())                       # ids named in the one-cushion map keep their name
