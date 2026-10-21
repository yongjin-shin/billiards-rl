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
