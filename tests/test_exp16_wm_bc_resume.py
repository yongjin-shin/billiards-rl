"""Tests for checkpoint / resume of exp16_wm.bc.train_bc (a reclaimed Spot instance continues where it stopped)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.bc import dim_weights, train_bc
from exp16_wm.geometry_aux import aim_targets, standardize
from exp16_wm.gnn_policy import PairGNNPolicy
from tests.test_exp16_wm_teacher_free import _made_only_setup

KW = dict(batch=8, lr=3e-3, patience=100, seed=0, log_every=0, select="within", w=dim_weights(0.2, 0.5), sims_batch=64,
          made_only=True, lr_schedule="cosine")


def _fit(epochs: int, ckpt: str | None, every: int = 0, stop_after: int | None = None, aux: bool = False):
    data, tr, sims = _made_only_setup()
    torch.manual_seed(5)
    extra = {}
    if aux:
        a, mean, std = standardize(*aim_targets(data["obs"]), tr)
        m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True, aux_dim=1, self_aim="grad")
        m.set_aim_stats(float(mean[0]), float(std[0])); extra = dict(aux=a, aux_weight=1.0)
    else:
        m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=epochs, sims=sims, ckpt_path=ckpt, ckpt_every=every, stop_after=stop_after,
                   **KW, **extra)
    return m, fit


@pytest.mark.parametrize("aux", [False, True])
def test_interrupted_and_resumed_run_ends_like_an_uninterrupted_one(tmp_path, aux):
    ref, fr = _fit(5, None, aux=aux)
    ck = str(tmp_path / "c.pt")
    _, part = _fit(5, ck, every=2, stop_after=3, aux=aux)                     # "reclaimed" after epoch 3 (not a multiple of 2)
    assert part["epochs_run"] == 3 and os.path.exists(ck)
    res, fit = _fit(5, ck, every=2, aux=aux)                                   # a fresh process resumes
    assert fit["epochs_run"] == 5 and fit["best_epoch"] == fr["best_epoch"]
    np.testing.assert_allclose([h[1] for h in fit["history"]], [h[1] for h in fr["history"]], rtol=1e-6)
    for k, v in ref.state_dict().items():
        torch.testing.assert_close(res.state_dict()[k], v)
    for k, v in fr["last_state"].items():                                       # the last epoch's weights too
        torch.testing.assert_close(fit["last_state"][k], v)


def test_no_checkpoint_without_every_and_a_finished_checkpoint_skips_training(tmp_path):
    ck = str(tmp_path / "c.pt")
    _fit(2, ck, every=0)
    assert not os.path.exists(ck)                                              # ckpt_every 0: nothing written
    a, fa = _fit(3, ck, every=1)
    b, fb = _fit(3, ck, every=1)                                               # already at the last epoch: no further training
    assert fb["epochs_run"] == 3
    for k, v in a.state_dict().items():
        torch.testing.assert_close(b.state_dict()[k], v)


def test_select_fn_picks_the_best_scored_epoch_only_among_scored_epochs(tmp_path):
    calls = []

    def sel(module):                                                    # pretend epoch 3 (index 3) is the best decision quality
        calls.append(len(calls)); return [0.2, 0.9, 0.5][len(calls) - 1]
    data, tr, sims = _made_only_setup()
    torch.manual_seed(5)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=6, sims=sims, select_fn=sel, select_every=2, **KW)
    assert [e for e, _ in fit["select_history"]] == [1, 3, 5] and fit["best_epoch"] == 3
    assert fit["best_score"] == pytest.approx(0.9)


def test_select_fn_survives_resume(tmp_path):
    ck = str(tmp_path / "c.pt")
    scores = iter([0.3, 0.8, 0.1, 0.4])
    data, tr, sims = _made_only_setup()
    torch.manual_seed(5)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    train_bc(m, data, tr, ~tr, "nearest_good", epochs=8, sims=sims, select_fn=lambda _: next(scores), select_every=2,
             ckpt_path=ck, ckpt_every=2, stop_after=4, **KW)
    torch.manual_seed(5)
    m2 = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    fit = train_bc(m2, data, tr, ~tr, "nearest_good", epochs=8, sims=sims, select_fn=lambda _: next(scores), select_every=2,
                   ckpt_path=ck, ckpt_every=2, **KW)
    assert [e for e, _ in fit["select_history"]] == [1, 3, 5, 7] and fit["best_epoch"] == 3   # the pre-interruption best is kept
