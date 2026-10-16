"""Tests for the teacher-free pieces: made2 head without the proposal term, made-only training (exp16_wm.bc)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.bc import dim_weights, train_bc
from exp16_wm.gnn_policy import PairGNNPolicy
from exp16_wm.obs_transform import relative_obs
from exp16_wm.sim_identity import to_bitmask

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        balls = [(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), 0) for _ in range(3)]
        rows.append(relative_obs(np.array([0.5, 0.2, *[v for b in balls for v in b], *POCKETS], dtype=np.float32)))
    return np.stack(rows)


def test_no_proposal_head_ignores_the_pair_head_proposal():
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    assert m.head_made2[0].in_features == 16 + 2 + 1
    assert PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True).head_made2[0].in_features == 16 + 4 + 1
    rel = torch.as_tensor(_rel(4))
    with torch.no_grad():
        o = m.forward_full(rel); a = torch.rand(4, 2) * 2 - 1
        z = m.made2_logits(o, a)
        o2 = dict(o); o2["means"] = o["means"] + 0.3                                          # move every proposal
        assert torch.allclose(m.made2_logits(o2, a), z)


def _made_only_setup(n=40, seed=1):
    rel = _rel(n, seed)
    tr = np.arange(n) < 30
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), 6)
    a = (rng.random((len(rows), 2)) * 2 - 1).astype(np.float32)
    made = np.zeros((len(rows), 18), bool); made[np.arange(len(rows)), rng.integers(0, 18, len(rows))] = rng.random(len(rows)) > 0.5
    sims = {"row": rows, "a_unit": a, "made": to_bitmask(made), "scratch": rng.random(len(rows)) > 0.9}
    data = {"obs": rel, "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
            "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
    return data, tr, sims


def test_made_only_trains_the_made_head_and_encoder_but_not_the_action_or_selection_heads():
    data, tr, sims = _made_only_setup()
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, made2_head=True, made2_geom=True, made2_no_proposal=True)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=4, batch=8, lr=3e-3, patience=10, seed=0, log_every=0, select="within",
                   w=dim_weights(0.2, 0.5), sims=sims, sims_batch=64, made_only=True)
    after = m.state_dict()
    changed = lambda pre: any(not torch.equal(before[k], after[k]) for k in before if k.startswith(pre))
    assert changed("head_made2") and changed("pair.") and changed("ball.")
    assert not changed("head_a") and not changed("head_s")                                   # no action / selection loss
    assert fit["select"] == "loss" and np.isfinite(fit["best_val_loss"]) and 0 <= fit["best_epoch"] < 4
    assert all(np.isnan(h[3]) for h in fit["history"])                                      # no action metric


def test_made_only_needs_sims_and_val_simulations():
    data, tr, sims = _made_only_setup()
    kw = dict(epochs=1, batch=8, lr=1e-3, patience=5, seed=0, log_every=0, w=dim_weights(0.2, 0.5))
    with pytest.raises(ValueError):
        train_bc(PairGNNPolicy(hidden=16, made2_head=True), data, tr, ~tr, "nearest_good", made_only=True, **kw)
    only_train = {k: v[sims["row"] < 30] for k, v in sims.items()}
    with pytest.raises(ValueError):
        train_bc(PairGNNPolicy(hidden=16, made2_head=True), data, tr, ~tr, "nearest_good", sims=only_train, sims_batch=16, made_only=True, **kw)


# ── exp16_wm.aim_sims (T1 generator) ──

from exp16_wm.aim_sims import ABS7, ANG31, ensemble_aim, grid_units, sample_pair_candidates  # noqa: E402


def test_grid_units_centre_on_the_aim_with_absolute_speeds():
    scale, bias = np.array([np.pi, 3.75]), np.array([0.0, 4.25])
    aim_u = np.array([[0.1, -0.2]])
    g = grid_units(aim_u, scale, bias)
    assert g.shape == (1, 2, len(ANG31) * len(ABS7), 2)
    assert np.allclose(g[..., 0].mean(-1), aim_u)                                            # symmetric angle grid
    speeds = np.unique(np.round(g[0, 0, :, 1] * scale[1] + bias[1], 6))
    assert np.allclose(speeds, ABS7)


def test_sample_pair_candidates_only_live_pairs_k_distinct_each():
    valid = np.zeros((3, 18), bool); valid[0, [0, 5]] = True; valid[2, 17] = True
    pick = sample_pair_candidates(valid, 4, 217, np.random.default_rng(0))
    assert len(pick) == 12 and set(map(tuple, pick[:, :2])) == {(0, 0), (0, 5), (2, 17)}
    for s, j in {(0, 0), (0, 5), (2, 17)}:
        c = pick[(pick[:, 0] == s) & (pick[:, 1] == j), 2]
        assert len(set(c)) == 4 and c.max() < 217
    assert sample_pair_candidates(np.zeros((2, 18), bool), 4, 217, np.random.default_rng(0)).shape == (0, 3)


def test_ensemble_aim_is_the_circular_mean_of_the_encoders():
    encs = []
    for deg, seed in ((179.0, 0), (-179.0, 1)):
        torch.manual_seed(seed)
        e = PairGNNPolicy(hidden=16, aux_dim=1)
        with torch.no_grad():
            e.aux_head[2].weight.zero_(); e.aux_head[2].bias.fill_(deg)
        encs.append(e)
    a = ensemble_aim(encs, torch.as_tensor(_rel(3)), 0.0, 1.0, chunk=2)
    assert a.shape == (3, 18) and np.allclose(np.abs(np.degrees(a)), 180.0, atol=1e-3)
