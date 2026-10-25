"""Tests for exp16_wm.cue_walls (CT2b-2: which wall a one-cushion shot hit)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import TABLE_L, TABLE_W, Params, collect_start
from exp16_wm.cue_walls import N_WALLS, WALLS, _label_task, k1_rows, label, replay, wall_index, wall_names


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
    ids, contact = _label_task((z["starts"][0], z["phi"][w["rows"][:3]], z["v"][w["rows"][:3]]))   # same helper the pool uses
    assert len(ids) == 3
