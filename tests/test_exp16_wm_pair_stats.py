"""
tests/test_exp16_wm_pair_stats.py

Unit tests for exp16_wm/pair_stats.py — per-pair coarse outcome statistics (2026-10-09).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulator import BilliardsEnv
from exp16_wm.outcomes import pocket_index_map, made_slots
from exp16_wm.pair_stats import pair_stats_for_state, stats_for_obs, attach, COARSE
from exp16_wm.planner import ghost_aims, aim_reference, action_from_phi
from exp16_wm.relabel import restore_state
from exp16_wm.teacher_data import collect, save, CLEAR_ENV


def _obs(n_games=1, seed=81):
    return collect(seed=seed, n_games=n_games, teacher="ghost", env_kw=CLEAR_ENV, max_good=4, log_every=0)["obs"]


class TestPairStats:
    def test_invariants_on_real_states(self):
        env = BilliardsEnv(**CLEAR_ENV)
        env.reset(seed=0)
        pmap = pocket_index_map(env)
        for o in _obs()[:3]:
            st = pair_stats_for_state(env, o, pmap)
            n_cand = len(COARSE["offsets_deg"]) * len(COARSE["speeds"])
            assert set(np.unique(st["pair_n"])) <= {0, n_cand}
            assert (st["pair_made"] <= st["pair_any"]).all()               # making this pair cleanly ⊂ cleanly pocketing something
            assert (st["pair_made"] <= st["pair_n"]).all() and (st["pair_scratch"] <= st["pair_n"]).all()
            assert (st["pair_any"] <= st["pair_n"]).all()
            restore_state(env, o)
            tried = {(int(b) - 1) * 6 + j for b, j, _ in ghost_aims(env, COARSE["max_cut_deg"])}
            assert {int(s) for s in np.where(st["pair_n"] > 0)[0]} == tried
        env.close()

    def test_matches_direct_simulation_of_one_candidate(self):
        env = BilliardsEnv(**CLEAR_ENV)
        env.reset(seed=0)
        pmap = pocket_index_map(env)
        o = _obs()[0]
        restore_state(env, o)
        bid, j, phi = ghost_aims(env, 80.0)[0]
        slot = (int(bid) - 1) * 6 + j
        phi_ref = aim_reference(env)
        direct = 0
        for off in COARSE["offsets_deg"]:
            for v in COARSE["speeds"]:
                made, scratch = made_slots(env, o, action_from_phi(phi + np.radians(off), phi_ref, v), pmap)
                direct += int(made[slot] and not scratch)
        assert pair_stats_for_state(env, o, pmap)["pair_made"][slot] == direct
        env.close()

    def test_pocketed_ball_has_no_pairs(self):
        env = BilliardsEnv(**CLEAR_ENV)
        env.reset(seed=0)
        pmap = pocket_index_map(env)
        o = _obs(2)
        late = next(x for x in o if x[4] > 0.5 or x[7] > 0.5 or x[10] > 0.5)          # a state with a pocketed ball
        st = pair_stats_for_state(env, late, pmap)
        gone = [i for i, f in enumerate((late[4], late[7], late[10])) if f > 0.5]
        for g in gone:
            assert st["pair_n"][g * 6:(g + 1) * 6].sum() == 0
        env.close()


class TestAttach:
    def test_attach_adds_arrays_and_keeps_the_rest(self, tmp_path):
        data = collect(seed=82, n_games=1, teacher="ghost", env_kw=CLEAR_ENV, max_good=4, log_every=0)
        path = str(tmp_path / "s.npz")
        save(path, data, seed=82, teacher="ghost", env_kw=CLEAR_ENV)
        st = stats_for_obs(data["obs"])
        attach(path, st)
        z = np.load(path)
        for k in ("pair_n", "pair_made", "pair_any", "pair_scratch"):
            assert z[k].shape == (len(data["obs"]), 18) and z[k].dtype == np.int8
        np.testing.assert_array_equal(z["action"], data["action"])
