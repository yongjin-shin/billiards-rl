"""Tests for exp16_wm.lookahead: one-shot look-ahead choice (shot now × best next shot)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.lookahead import LookaheadPolicy

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]
OBS = np.array([0.5, 0.2, 0.3, 0.6, 0, 0.6, 0.3, 0, 0.7, 0.7, 0, *POCKETS], np.float32)


class Scorer:
    """Fake made head: the candidate angles are 0.0, 0.1, 0.2 (unit); logits from a table keyed by the observation's cue x."""
    act_scale, act_bias = torch.ones(2), torch.zeros(2)

    def __init__(self, logits_by_cue: dict[float, list[float]]):
        self.t = logits_by_cue

    def scores(self, rel):
        cue = round(float(self._cue), 2)
        z = torch.tensor(self.t[cue], dtype=torch.float32)[None, None]          # (1, 1 pair, 3 candidates)
        cand = torch.tensor([[[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]]])[None]
        return z, cand


def _policy(now_logits, next_by_action, made_by_action, k=3, mode="model"):
    sc = Scorer({0.5: now_logits, **{round(0.1 + j / 10, 2): [next_by_action[j]] * 3 for j in range(3)}})
    def sim(raw, a):
        j = int(round(float(a[0]) * 10))
        nxt = raw.copy(); nxt[0] = 0.1 + j / 10                                  # a different position per action
        sc._cue = nxt[0]
        return nxt, made_by_action[j], False, False
    pol = LookaheadPolicy(sc, sim, k=k, mode=mode)
    orig = pol._candidates
    def cands(raw):
        sc._cue = raw[0]; return orig(raw)
    pol._candidates = cands
    return pol


def test_model_mode_trades_a_slightly_easier_shot_for_a_much_easier_next_one():
    pol = _policy(now_logits=[2.0, 1.8, -3.0], next_by_action=[-4.0, 3.0, 3.0], made_by_action=[True, True, True])
    assert pol.act(OBS)[0] == pytest.approx(0.1)                                # 0.86 × 0.95 beats 0.88 × 0.02


def test_sim_mode_only_takes_shots_that_really_go_in():
    pol = _policy(now_logits=[2.0, 1.8, 1.0], next_by_action=[3.0, 3.0, 2.0], made_by_action=[False, False, True], mode="sim")
    assert pol.act(OBS)[0] == pytest.approx(0.2)


def test_k_limits_which_shots_are_simulated():
    pol = _policy(now_logits=[2.0, 1.8, 1.0], next_by_action=[-4.0, -4.0, 5.0], made_by_action=[True, True, True], k=2)
    assert pol.act(OBS)[0] in (pytest.approx(0.0), pytest.approx(0.1))         # the third shot is never looked at
    with pytest.raises(ValueError):
        LookaheadPolicy(Scorer({}), lambda r, a: None, mode="greedy")


def test_two_levels_see_a_trap_one_level_misses():
    """Shot A leaves an easy next shot that leads nowhere (its own next position is hopeless); shot B leaves a slightly harder next
    shot whose follow-up is easy. One level prefers A, two levels prefer B."""
    import torch as _t

    class S:
        act_scale, act_bias = _t.ones(2), _t.zeros(2)
        table = {0.5: [1.0, 1.0], 0.1: [3.0, -9.0], 0.2: [1.5, -9.0], 0.3: [-9.0, -9.0], 0.4: [4.0, 4.0]}
        def scores(self, rel):
            z = _t.tensor(self.table[round(self.cue, 2)], dtype=_t.float32)[None, None]
            return z, _t.tensor([[[0.0, 1.0], [0.1, 1.0]]])[None]
    sc = S()
    nxt_of = {(0.5, 0): 0.1, (0.5, 1): 0.2, (0.1, 0): 0.3, (0.2, 0): 0.4}       # (position, shot index) → next position

    def sim(raw, a):
        j = int(round(float(a[0]) * 10)); key = (round(float(raw[0]), 2), j)
        out = raw.copy(); out[0] = nxt_of.get(key, 0.3)
        return out, True, False, False
    one = LookaheadPolicy(sc, sim, widths=(2,))
    two = LookaheadPolicy(sc, sim, widths=(2, 1))
    for pol in (one, two):
        orig = pol._candidates
        pol._candidates = (lambda o: (lambda raw: (setattr(sc, "cue", float(raw[0])), o(raw))[1]))(orig)
    assert one.act(OBS)[0] == pytest.approx(0.0) and two.act(OBS)[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        LookaheadPolicy(sc, sim, widths=(2, 0))


def test_model_mode_values_every_shot_at_the_same_depth():
    """A missed shot (simulated) must not be scored one level shallower than a made one (that would favour misses)."""
    import torch as _t

    class S:
        act_scale, act_bias = _t.ones(2), _t.zeros(2)
        table = {0.5: [2.0, 2.0], 0.1: [1.0, 1.0], 0.2: [1.0, 1.0], 0.3: [1.0, 1.0]}
        def scores(self, rel):
            z = _t.tensor(self.table[round(self.cue, 2)], dtype=_t.float32)[None, None]
            return z, _t.tensor([[[0.0, 1.0], [0.1, 1.0]]])[None]
    sc = S()
    calls = []

    def sim(raw, a):
        j = int(round(float(a[0]) * 10)); out = raw.copy(); out[0] = 0.1 if round(float(raw[0]), 2) == 0.5 and j == 0 else (0.2 if j == 1 else 0.3)
        calls.append((round(float(raw[0]), 2), j))
        return out, j == 1, False, False                                       # shot 0 misses, shot 1 goes in
    pol = LookaheadPolicy(sc, sim, widths=(2, 1))
    orig = pol._candidates
    pol._candidates = lambda raw: (setattr(sc, "cue", float(raw[0])), orig(raw))[1]
    pol.act(OBS)
    assert (0.1, 0) in calls or (0.1, 1) in calls                               # the missed shot's position is also looked into



def test_v_leaf_values_the_last_positions_with_the_v_head_and_counts_simulations():
    import torch as _t

    class M:                                                                      # a member with a V head: V = position's cue x
        def forward_full(self, rel): return {"rel": rel}
        def v_logit(self, o): return _t.logit(_t.tensor([float(self.cue)]))
    class S:
        act_scale, act_bias = _t.ones(2), _t.zeros(2)
        def __init__(self): self.models = [M()]
        def scores(self, rel):
            return _t.tensor([[[1.0, 1.0]]]), _t.tensor([[[0.0, 1.0], [0.1, 1.0]]])[None]
    sc = S()
    def sim(raw, a):
        out = raw.copy(); out[0] = 0.2 if float(a[0]) < 0.05 else 0.7              # shot 1 leaves the position V likes
        return out, True, False, False
    pol = LookaheadPolicy(sc, sim, widths=(2,), leaf="v", mode="sim")
    orig = pol._candidates
    def cands(raw):
        sc.models[0].cue = float(raw[0]); return orig(raw)
    pol._candidates = cands
    sc.models[0].cue = 0.5
    _orig_v = pol._v
    pol._v = lambda raw: (setattr(sc.models[0], "cue", float(raw[0])), _orig_v(raw))[1]
    assert pol.act(OBS)[0] == pytest.approx(0.1) and pol.n_sims == 2
    with pytest.raises(ValueError):
        LookaheadPolicy(sc, sim, leaf="mcts")


def test_cue_only_next_keeps_unpocketed_balls_where_they_were():
    from exp16_wm.lookahead import cue_only_next
    nxt = OBS.copy()
    nxt[0:2] = [0.9, 0.9]                       # cue ball landed elsewhere
    nxt[2:5] = [0.0, 0.0, 1]                    # ball 0 pocketed
    nxt[5:7] = [0.1, 0.1]                       # ball 1 was pushed — cue_only ignores it
    out = cue_only_next(OBS, nxt)
    np.testing.assert_allclose(out[0:2], [0.9, 0.9])
    np.testing.assert_allclose(out[2:5], [0.0, 0.0, 1])
    np.testing.assert_allclose(out[5:11], OBS[5:11])
    np.testing.assert_allclose(out[11:], OBS[11:])


def test_cue_only_next_with_nothing_changed_is_the_same_position():
    from exp16_wm.lookahead import cue_only_next
    np.testing.assert_allclose(cue_only_next(OBS, OBS.copy()), OBS)


def test_next_state_cue_feeds_the_approximate_position_to_the_next_level():
    seen = []
    def sim(raw, a):
        nxt = raw.copy(); nxt[0] += 0.1; nxt[5] = 0.123                       # cue moves, ball 1 is pushed
        seen.append(raw.copy())
        return nxt, True, False, False
    sc = Scorer({round(0.5 + 0.1 * d, 2): [1.0, 0.0, -1.0] for d in range(4)})
    pol = LookaheadPolicy(sc, sim, mode="sim", widths=(1, 1), next_state="cue")
    orig = pol._candidates
    pol._candidates = lambda raw, **kw: (setattr(sc, "_cue", raw[0]), orig(raw, **kw))[1]
    pol.act(OBS)
    assert len(seen) == 2 and seen[1][5] == pytest.approx(OBS[5])          # level 2 sees ball 1 unmoved
    with pytest.raises(ValueError):
        LookaheadPolicy(sc, sim, next_state="world")
