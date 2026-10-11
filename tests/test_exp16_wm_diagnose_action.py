"""Tests for exp16_wm/diagnose_action.py (pure helpers and the model-side refine helpers; the env-based measurements are checked by the run itself)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.diagnose_action import (PREV_KINDS, candidate_grid, configure, grad_refine, gradient_toward_window, laplace_perturb,
                                       made_logit, oracle_row, previous_outcome, refine_at, rollout_shots, scan_baselines,
                                       shot_groups, signed_angle_diff_deg, substitute_action, summarize, summarize_oracle,
                                       summarize_visited, visited_flags)
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import relative_obs
from exp16_wm.teacher_data import CLEAR_ENV

OFFSETS = np.round(np.arange(-0.6, 0.6 + 1e-9, 0.05), 3)
POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel_batch(n: int = 4) -> torch.Tensor:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        balls = [v for _ in range(3) for v in (*rng.uniform(0.2, 0.8, 2), 0.0)]
        rows.append(relative_obs(np.array([0.5, 0.5, *balls, *POCKETS], dtype=np.float32)))
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def _model() -> PairGNNPolicy:
    torch.manual_seed(0)
    return PairGNNPolicy(hidden=16, made2_head=True).eval()


# ── signed_angle_diff_deg ─────────────────────────────

def test_signed_angle_diff_has_a_sign_and_wraps():
    d = signed_angle_diff_deg(np.array([0.1, 0.0, 0.999]), np.array([0.0, 0.1, -0.999]))
    assert d[0] == pytest.approx(18.0) and d[1] == pytest.approx(-18.0)
    assert d[2] == pytest.approx(-0.36, abs=1e-6)                # across the ±180° seam the short way


# ── laplace_perturb ───────────────────────────────────

def test_laplace_zero_scale_is_identity_and_draws_nothing():
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    assert laplace_perturb(0.25, 0.0, rng) == 0.25
    assert rng.bit_generator.state == state


def test_laplace_scale_is_the_mean_absolute_deviation_in_degrees():
    rng = np.random.default_rng(0)
    dev = np.array([laplace_perturb(0.0, 0.1, rng) for _ in range(20000)]) * 180.0
    assert np.abs(dev).mean() == pytest.approx(0.1, rel=0.05)    # E|Laplace(b)| = b
    assert dev.mean() == pytest.approx(0.0, abs=0.01)


# ── scan_baselines ────────────────────────────────────

def _window(lo: float, hi: float) -> np.ndarray:
    return (OFFSETS >= lo - 1e-9) & (OFFSETS <= hi + 1e-9)


def test_scan_baselines_head_that_repairs_a_missing_proposal():
    truth = np.stack([_window(-0.1, 0.2), _window(-0.1, 0.2)])           # window of 7 of 25 offsets
    prob = np.where(truth, 0.9, 0.1)                                      # perfect head
    r = scan_baselines(prob, truth, OFFSETS, np.array([0.0, 0.5]))        # state 0 proposal inside, state 1 outside
    assert r["n_partial"] == 2 and r["head_hit"] == 1.0
    assert r["proposal_hit"] == 0.5 and r["n_proposal_miss"] == 1
    assert r["head_hit_when_proposal_misses"] == 1.0 and r["head_hit_when_proposal_hits"] == 1.0
    assert r["random_offset"] == pytest.approx(7 / 25) and r["label_offset_hit"] == 1.0


def test_scan_baselines_head_that_only_echoes_the_proposal():
    truth = np.stack([_window(-0.1, 0.2)] * 2)
    prop = np.array([0.0, 0.5])
    prob = np.stack([np.exp(-np.abs(OFFSETS - p) * 20) for p in prop])    # peaked at the proposal
    r = scan_baselines(prob, truth, OFFSETS, prop)
    assert r["head_hit"] == r["proposal_hit"] == 0.5
    assert r["head_hit_when_proposal_misses"] == 0.0 and r["head_hit_when_proposal_hits"] == 1.0


def test_scan_baselines_clips_a_proposal_outside_the_scan_and_handles_no_partial_state():
    truth = np.stack([_window(0.4, 0.6)])
    r = scan_baselines(np.full((1, 25), 0.5), truth, OFFSETS, np.array([3.0]))   # far right → clipped to +0.6 (inside)
    assert r["proposal_hit"] == 1.0
    empty = scan_baselines(np.full((2, 25), 0.5), np.stack([np.ones(25, bool), np.zeros(25, bool)]), OFFSETS, np.zeros(2))
    assert empty["n_partial"] == 0 and np.isnan(empty["head_hit"]) and empty["n_proposal_miss"] == 0


# ── gradient_toward_window ────────────────────────────

def test_gradient_toward_window_counts_only_offsets_outside_it():
    truth = _window(-0.1, 0.1)
    grad = np.sign(0.0 - OFFSETS)                                         # ascent direction toward the centre everywhere
    ok, dist = gradient_toward_window(grad, truth, OFFSETS)
    assert len(ok) == len(dist) == int((~truth).sum()) and ok.all()
    assert dist.min() == pytest.approx(0.05) and dist.max() == pytest.approx(0.5)
    ok_bad, _ = gradient_toward_window(-grad, truth, OFFSETS)
    assert not ok_bad.any()


def test_gradient_toward_window_empty_without_a_partial_window():
    for truth in (np.ones(25, bool), np.zeros(25, bool)):
        ok, dist = gradient_toward_window(np.ones(25), truth, OFFSETS)
        assert len(ok) == 0 and len(dist) == 0


# ── model-side helpers ────────────────────────────────

def test_refine_at_matches_predict_unit_refine():
    m, x = _model(), _rel_batch()
    configure(m, "refine", (-0.2, -0.1, 0.0, 0.1, 0.2), (0.8, 1.2))
    with torch.no_grad():
        o = m.forward_full(x)
        k = o["logits"].argmax(1)
        assert torch.allclose(refine_at(m, o, k), m.predict_unit(x), atol=1e-6)


def test_made_logit_shape_and_refine_picks_the_best_candidate():
    m, x = _model(), _rel_batch()
    configure(m, "refine", (-0.1, 0.0, 0.1), (1.0, 2.0))
    with torch.no_grad():
        o = m.forward_full(x)
        k = torch.zeros(len(x), dtype=torch.long)
        cand = m._unit_offsets(o["means"][torch.arange(len(x)), k])
        z = made_logit(m, o, k, cand)
        assert z.shape == (len(x), 6)
        assert torch.equal(refine_at(m, o, k), cand[torch.arange(len(x)), z.argmax(1)])


def test_grad_refine_never_lowers_the_logit_and_keeps_speed():
    m, x = _model(), _rel_batch()
    with torch.no_grad():
        o = m.forward_full(x)
    k = torch.arange(len(x)) % 3
    a = o["means"][torch.arange(len(x)), k]
    g = grad_refine(m, o, k)
    with torch.no_grad():
        z0 = made_logit(m, o, k, a[:, None])[:, 0]
        z1 = made_logit(m, o, k, g[:, None])[:, 0]
    assert (z1 >= z0 - 1e-6).all()
    assert torch.equal(g[:, 1], a[:, 1])
    assert ((g[:, 0] - a[:, 0]).abs() * 180.0 <= 12 * 0.04 + 1e-4).all()      # at most steps × step_deg away


def test_grad_refine_with_zero_steps_returns_the_proposal():
    m, x = _model(), _rel_batch(2)
    with torch.no_grad():
        o = m.forward_full(x)
    k = torch.zeros(2, dtype=torch.long)
    assert torch.equal(grad_refine(m, o, k, steps=0), o["means"][torch.arange(2), k])


# ── summarize ─────────────────────────────────────────

def test_summarize_pools_seeds_by_balls_left_and_ignores_unlabelled_states():
    def split(nl, lab, ok_val):
        n = len(nl)
        ok = {k: np.full(n, ok_val, bool) for k in ("sel", "lab", "vr", "labref", "lab_grid", "lab_grad", "lab_grad_small")}
        return {"nl": np.array(nl), "lab": np.array(lab), "ok": ok, "sel_in_good": np.ones(n, bool),
                "ver_in_good": np.ones(n, bool), "sel_is_label": np.zeros(n, bool)}
    truth = np.stack([_window(-0.1, 0.2)]); scan = {"P": np.where(truth, 0.9, 0.1), "Y": truth, "prop": np.array([0.0]),
                                                    "gok": np.array([True, False]), "gdist": np.array([0.1, 0.5])}
    s = summarize("x", [split([1, 2, 3, 3], [True, True, True, False], True), split([1, 2], [True, True], False)], [scan, scan], None)
    assert s["n_by_balls"] == {"1": 2, "2": 2, "3": 1}                    # the unlabelled 3-ball state is not counted
    assert s["split"]["select_clean_policy_pair"]["1"] == 0.5 and s["split"]["select_clean_policy_pair"]["3"] == 1.0
    assert s["split"]["select_clean_policy_pair"]["all"] == pytest.approx(3 / 5)
    assert s["split"]["select_pair_is_label"]["all"] == 0.0
    assert s["scan"]["head_hit"] == 1.0 and s["gradient_toward_window"]["all"] == 0.5
    assert "sample" not in s


# ── substitute_action / candidate_grid ────────────────

def test_substitute_action_takes_each_column_from_the_chosen_source_and_leaves_inputs_alone():
    model = np.array([[0.1, 1.5], [0.2, 2.5]], np.float32); label = np.array([[0.3, 1.0], [0.4, 3.0]], np.float32)
    m0, l0 = model.copy(), label.copy()
    assert np.allclose(substitute_action(model, label), model)
    assert np.allclose(substitute_action(model, label, "label", "label"), label)
    assert np.allclose(substitute_action(model, label, "model", "label"), [[0.1, 1.0], [0.2, 3.0]])
    assert np.allclose(substitute_action(model, label, "label", "model"), [[0.3, 1.5], [0.4, 2.5]])
    assert np.array_equal(model, m0) and np.array_equal(label, l0)
    with pytest.raises(ValueError):
        substitute_action(model, label, "teacher", "model")


def test_candidate_grid_modes_keep_the_other_dimension_fixed():
    a = np.array([0.5, 2.0]); deg, sp = (-0.1, 0.0, 0.1), (1.0, 2.0)
    s, g, b = candidate_grid(a, deg, sp, "speed"), candidate_grid(a, deg, sp, "angle"), candidate_grid(a, deg, sp, "both")
    assert s.shape == (2, 2) and g.shape == (3, 2) and b.shape == (6, 2)
    assert np.allclose(s[:, 0], 0.5) and np.allclose(s[:, 1], sp)
    assert np.allclose(g[:, 1], 2.0) and np.allclose(g[:, 0], 0.5 + np.radians(deg), atol=1e-6)
    expected = np.array([(0.5 + np.radians(d), t) for d in deg for t in sp])
    assert np.allclose(b, expected, atol=1e-6)                         # angle-major order
    with pytest.raises(ValueError):
        candidate_grid(a, deg, sp, "nothing")


# ── oracle_row: the pre-registered judgement table ────

def test_oracle_row_follows_the_table_in_order():
    assert oracle_row(0.80, 0.85, 0.90, 0.95) == "A"                  # the label action is not reproduced
    assert oracle_row(0.80, 0.82, 0.83, 0.998) == "B"                 # fixing either alone recovers little (3 + 2 < 0.5 · 19.8)
    assert oracle_row(0.80, 0.82, 0.95, 0.998) == "C"                 # angle dominates (15 vs 2)
    assert oracle_row(0.80, 0.95, 0.82, 0.998) == "D"                 # speed dominates
    assert oracle_row(0.80, 0.90, 0.91, 0.998) == "E"                 # both contribute about equally


def test_oracle_row_needs_a_two_point_gap_for_c_and_d():
    assert oracle_row(0.80, 0.90, 0.915, 0.998) == "E"                # ratio 1.5 < 2
    assert oracle_row(0.97, 0.97, 0.985, 0.998) == "E"                # r_a ≥ 2 r_s holds (1.5 vs 0) and B does not, but the gap is under 2%p
    assert oracle_row(0.97, 0.97, 0.995, 0.998) == "C"                # same, with a 2.5%p gap
    assert oracle_row(0.80, 0.80, 0.80, 0.80) == "A"                  # l < 0.97 wins over everything


# ── previous_outcome / shot_groups / visited_flags ────

def test_previous_outcome_scratch_wins_over_a_pocket():
    assert previous_outcome(True, False) == "after_hit" and previous_outcome(False, False) == "after_miss"
    assert previous_outcome(False, True) == "after_scratch" and previous_outcome(True, True) == "after_scratch"


def test_shot_groups_counts_rates_and_mix():
    n_left = np.array([3, 3, 2, 2, 1, 1]); prev = np.array(["first", "after_hit", "after_miss", "after_hit", "after_scratch", "after_hit"])
    clean = np.array([True, False, False, True, False, True])
    g = shot_groups(n_left, prev, clean)
    assert g["by_balls"][3] == {"n": 2, "rate": 0.5} and g["by_balls"][2]["rate"] == 0.5 and g["by_balls"][1]["n"] == 2
    assert sum(g["mix"].values()) == pytest.approx(1.0)
    assert g["by_prev"]["after_hit"] == {"n": 3, "rate": pytest.approx(2 / 3)} and g["by_prev"]["first"]["n"] == 1
    assert g["normal"] == pytest.approx(3 / 4) and g["after_fail"] == 0.0 and g["n_after_fail"] == 2


def test_shot_groups_empty_groups_are_nan():
    g = shot_groups(np.array([3, 3]), np.array(["first", "after_hit"]), np.array([True, False]))
    assert np.isnan(g["by_balls"][1]["rate"]) and g["by_balls"][1]["n"] == 0 and np.isnan(g["after_fail"]) and g["n_after_fail"] == 0
    assert np.isnan(g["by_prev"]["after_miss"]["rate"])
    e = shot_groups(np.array([], int), np.array([], str), np.array([], bool))
    assert np.isnan(e["mix"][1]) and np.isnan(e["normal"])


def test_visited_flags_rows():
    val = {1: 0.85, 2: 0.75, 3: 0.83}
    same = visited_flags({1: 0.86, 2: 0.73, 3: 0.84}, val, 0.80, 0.78)
    assert same == {"A": True, "B": False, "C": False, "D": False}
    harder = visited_flags({1: 0.86, 2: 0.73, 3: 0.84}, val, 0.80, 0.70)           # failures leave hard states
    assert harder["B"] and not harder["A"]
    lower = visited_flags({1: 0.75, 2: 0.65, 3: 0.75}, val, 0.70, 0.69)
    assert lower["C"] and not lower["A"] and not lower["D"]
    higher = visited_flags({1: 0.92, 2: 0.82, 3: 0.90}, val, 0.88, 0.87)
    assert higher["D"] and not higher["C"]


def test_visited_flags_skips_nan_groups_and_missing_failures():
    f = visited_flags({1: 0.84, 2: float("nan"), 3: 0.82}, {1: 0.85, 2: 0.75, 3: 0.83}, 0.8, float("nan"))
    assert f["A"] and not f["B"]                                       # no failure states: nothing to compare, A rests on the groups
    none = visited_flags({1: float("nan"), 2: float("nan"), 3: float("nan")}, {1: 0.8, 2: 0.8, 3: 0.8}, float("nan"), float("nan"))
    assert not any(none.values())


# ── pooling ───────────────────────────────────────────

def _oracle_seed(m_ok, ls_ok, la_ok, l_ok, lab, nl, grid):
    return {"lab": np.array(lab), "nl": np.array(nl),
            "ok": {"M": np.array(m_ok), "Ls": np.array(ls_ok), "La": np.array(la_ok), "L": np.array(l_ok)}, "grid": grid}


def test_summarize_oracle_pools_seeds_ignores_unlabelled_states_and_names_the_row():
    grid = {k: np.array([True, False]) for k in ("M", "speed7", "angle31", "both217", "verify_refine_label_pair")}
    r = _oracle_seed([1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0], [True, True, True, False], [1, 2, 3, 3], grid)
    s = summarize_oracle([r, r])
    assert s["n"] == 6 and s["clean"]["M"]["all"] == pytest.approx(1 / 3) and s["clean"]["L"]["all"] == 1.0   # the unlabelled state is dropped
    assert s["clean"]["M"]["2"] == 0.0 and s["clean"]["M"]["1"] == 1.0
    assert s["grid_n"] == 4 and s["grid"]["both217"] == 0.5 and s["row"] in "ABCDE"


def test_summarize_visited_compares_games_with_labelled_val_states():
    game = {"n_left": np.array([3, 2, 2, 1]), "prev": np.array(["first", "after_hit", "after_miss", "after_hit"]),
            "clean": np.array([True, True, False, True])}
    val = {"nl": np.array([3, 2, 1, 1]), "lab": np.array([True, True, True, False]), "clean": np.array([True, False, True, False])}
    s = summarize_visited([{"games": game, "val": val}])
    assert s["val"]["by_balls"][1] == 1.0 and s["val"]["by_balls"][2] == 0.0 and s["val"]["n"] == 3     # the unlabelled 1-ball state is dropped
    assert s["games"]["by_balls"][2]["rate"] == 0.5 and s["games"]["after_fail"] == 0.0
    assert set(s["flags"]) == {"A", "B", "C", "D"}


# ── rollout smoke test on the real env ────────────────

def test_rollout_shots_records_one_row_per_shot_with_valid_labels():
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True).eval()
    r = rollout_shots(m, 2, CLEAR_ENV)
    n = len(r["clean"])
    assert n >= 2 and len(r["n_left"]) == n and len(r["prev"]) == n
    assert set(r["prev"]) <= set(PREV_KINDS) and int((r["prev"] == "first").sum()) == 2      # one first shot per game
    assert set(r["n_left"]) <= {1, 2, 3} and (r["n_left"][r["prev"] == "first"] == 3).all()
    assert r["clean"].dtype == bool
