"""Tests for exp16_wm.v_map_data: layout choice, free cue cells, ring copies, labels, the file format train_v reads."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.v_map_data import aggregate, build_layout, finish_layout, label_from_copies, pick_layouts, relabel, ring_copies, run_layouts, sample_cells, write_file

W, L, R = 0.9906, 1.9812, 0.028575
RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.3, 0.7, 0.0] + [0.0] * 12, np.float32)      # 2 balls on the table, cue ball anywhere; 20 = 2 + 3·2 + 12


def _v_file(tmp_path, n=60):
    obs = np.tile(RAW, (n, 1)); obs[:, 0] = np.linspace(0.1, 0.9, n)
    np.savez(tmp_path / "a.npz", obs=obs, k=np.zeros(n, int), left=np.arange(n) % 6 + 1, game=np.arange(n), branch=np.zeros(n, bool))
    return str(tmp_path / "a.npz")


def test_pick_layouts_takes_per_left_distinct_games_for_each_left_and_is_seeded(tmp_path):
    f = _v_file(tmp_path)
    obs, left, key = pick_layouts([f], 3, seed=0)
    assert len(obs) == 15 and sorted(set(left)) == [2, 3, 4, 5, 6] and all((left == e).sum() == 3 for e in range(2, 7))
    assert len(set(key)) == 15                                                         # every layout from another game
    assert np.array_equal(pick_layouts([f], 3, seed=0)[2], key) and not np.array_equal(pick_layouts([f], 3, seed=1)[2], key)
    with pytest.raises(ValueError):
        pick_layouts([f], 50, seed=0)                                                  # not enough games per left


def test_sample_cells_are_free_positions_and_jittered_off_the_grid():
    rng = np.random.default_rng(0)
    cells = sample_cells(RAW, 0.05, 0.025, rng, R, W, L)
    assert 600 < len(cells) < 800
    assert (cells[:, 0] >= R).all() and (cells[:, 0] <= W - R).all() and (cells[:, 1] >= R).all() and (cells[:, 1] <= L - R).all()
    d = np.hypot(cells[:, 0] - 0.5 * W, cells[:, 1] - 0.5 * L)
    assert (d >= 2 * R).all() and (d < 2 * R + 0.05).any()                              # ball at (0.5, 0.5) normalised (the pocketed one is ignored): cells hug it but never overlap it
    assert not np.allclose(cells[:, 0] % 0.05, (cells[:, 0] % 0.05)[0])                 # jitter: not all on one grid line


def test_ring_copies_are_exactly_the_radius_away_and_unplaceable_ones_are_marked():
    base = RAW.copy(); base[0], base[1] = 0.3, 0.3
    out, placed = ring_copies(base, (0.01, 0.03), 3, np.random.default_rng(0), W, L, R)
    assert out.shape == (6, len(base)) and placed.all()
    d = np.hypot((out[:, 0] - base[0]) * W, (out[:, 1] - base[1]) * L)
    np.testing.assert_allclose(d, [0.01] * 3 + [0.03] * 3, atol=1e-5)
    np.testing.assert_array_equal(out[:, 2:], np.tile(base[2:], (6, 1)))               # only the cue ball moves
    jam = RAW.copy(); jam[0], jam[1] = 0.5, 0.5                                          # cue ball inside the other ball: nowhere free to move
    o2, p2 = ring_copies(jam, (0.01,), 2, np.random.default_rng(0), W, L, R)
    assert not p2.any() and np.array_equal(o2[0], jam)


def test_aggregate_robust_is_the_mean_of_ring_means_worst_the_min_and_empty_rows_fall_back_to_exact():
    k_exact = np.array([4, 2, 1])
    ring_k = np.array([[[2, 4, 3], [5, 5, 5]],                                         # ring means 3.0, 5.0
                       [[0, -1, -1], [-1, -1, -1]],                                    # ring means 0.0, none
                       [[-1, -1, -1], [-1, -1, -1]]])                                  # nothing placed
    a = aggregate(k_exact, ring_k)
    np.testing.assert_allclose(a["ring_mean"], [[3.0, 5.0], [0.0, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(a["robust"], [4.0, 0.0, 1.0]); np.testing.assert_allclose(a["worst"], [3.0, 0.0, 1.0])


def test_pipeline_with_a_fake_simulator_keeps_row_order_and_writes_a_train_v_file(tmp_path):
    from exp16_wm.train_v import load_v_data
    radii = (10.0, 30.0)
    lays = [build_layout(RAW, 2, 7, 0, radii, 2, 0.1, 0.05, R, W, L), build_layout(RAW, 3, 8, 0, radii, 2, 0.1, 0.05, R, W, L)]
    fake = lambda jobs: [np.floor(o[:, 0] * 10).astype(int) for o, _ in jobs]           # "balls" = floor(10 · cue x): depends on the row only
    ks = run_layouts(lays, fake, chunk=7)
    for lay, k in zip(lays, ks):
        np.testing.assert_array_equal(k, np.floor(lay["obs"][:, 0] * 10).astype(int))
    fins = [finish_layout(lay, k, len(radii), 2) for lay, k in zip(lays, ks)]
    np.testing.assert_array_equal(fins[0]["k_exact"], np.floor(lays[0]["obs"][::int(lays[0]["per"]), 0] * 10).astype(int))
    out = str(tmp_path / "m.npz")
    n = write_file(out, np.stack([RAW, RAW]), np.array([2, 3]), np.array([7, 8]), lays, fins, radii)
    d = load_v_data([out], val_frac=0.5)
    assert len(d["k"]) == n == sum(len(l["cells"]) for l in lays) and d["k"].dtype == np.float32
    assert set(np.unique(d["gid"])) == {7, 8} and d["val"].sum() > 0 and set(np.unique(d["left"])) == {2, 3}   # split by layout (a layout is a "game")


def test_relabel_switches_the_label_between_exact_robust_worst_and_one_ring(tmp_path):
    n = 4
    np.savez(tmp_path / "v.npz", obs=np.zeros((n, 20), np.float32), k=np.zeros(n, np.float32), left=np.full(n, 3), game=np.arange(n), branch=np.zeros(n, bool),
             k_exact=np.array([3, 1, 0, 2]), ring_mean=np.array([[2.0, 4.0], [0.0, np.nan], [np.nan, np.nan], [1.0, 3.0]], np.float32),
             radii_mm=np.array([10.0, 30.0]), meta="{}")
    src = str(tmp_path / "v.npz")
    got = {}
    for t in ("exact", "robust", "worst", "ring10", "ring30"):
        relabel(src, str(tmp_path / f"{t}.npz"), t); got[t] = np.load(tmp_path / f"{t}.npz", allow_pickle=True)["k"]
    np.testing.assert_allclose(got["exact"], [3, 1, 0, 2]); np.testing.assert_allclose(got["robust"], [3.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(got["worst"], [2.0, 0.0, 0.0, 1.0]); np.testing.assert_allclose(got["ring10"], [2.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(got["ring30"], [4.0, 1.0, 0.0, 3.0])                       # row 1 had no ring-30 copy → its exact k
    with pytest.raises(ValueError):
        relabel(src, str(tmp_path / "x.npz"), "median")


def test_label_from_copies_picks_radii_and_copies_and_falls_back_to_exact():
    ring_k = np.array([[[1, 3, 5], [0, 0, 6], [2, 2, 2]],                         # position 0: radii 10, 30, 60 × copies 0, 1, 2
                       [[-1, -1, -1], [4, -1, 4], [-1, -1, -1]]])                 # position 1: only two copies of radius 30 were placed
    radii, kx = np.array([10.0, 30.0, 60.0]), np.array([7, 9])
    np.testing.assert_allclose(label_from_copies(ring_k, radii, [10.0], [0, 1, 2], kx), [3.0, 9.0])           # three copies at one distance; none placed → exact
    np.testing.assert_allclose(label_from_copies(ring_k, radii, [10.0, 30.0, 60.0], [0], kx), [1.0, 4.0])     # one copy at each of three distances
    np.testing.assert_allclose(label_from_copies(ring_k, radii, [10.0, 30.0, 60.0], [0, 1, 2], kx), [(1 + 3 + 5 + 0 + 0 + 6 + 2 + 2 + 2) / 9, 4.0])


def test_pipeline_files_keep_every_copy_and_relabel_sel_uses_them(tmp_path):
    radii = (10.0, 30.0)
    lays = [build_layout(RAW, 2, 7, 0, radii, 3, 0.1, 0.05, R, W, L)]
    ks = run_layouts(lays, lambda jobs: [np.floor(o[:, 0] * 10).astype(int) for o, _ in jobs], chunk=9)
    fins = [finish_layout(lays[0], ks[0], 2, 3)]
    out = str(tmp_path / "m.npz")
    n = write_file(out, np.stack([RAW]), np.array([2]), np.array([7]), lays, fins, radii)
    z = np.load(out, allow_pickle=True)
    assert z["ring_k"].shape == (n, 2, 3) and z["ring_k"].dtype == np.int8
    relabel(out, str(tmp_path / "o.npz"), "sel:10:0,1,2"); relabel(out, str(tmp_path / "r10.npz"), "ring10")
    np.testing.assert_allclose(np.load(tmp_path / "o.npz", allow_pickle=True)["k"], np.load(tmp_path / "r10.npz", allow_pickle=True)["k"])   # all three copies = the ring mean
    old = {k: z[k] for k in z.files if k != "ring_k"}; np.savez(tmp_path / "old.npz", **old)
    with pytest.raises(ValueError):
        relabel(str(tmp_path / "old.npz"), str(tmp_path / "x.npz"), "sel:10:0")                      # a file from the older generator has no per-copy results
