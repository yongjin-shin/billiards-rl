"""Tests for exp16_wm/geometry_aux.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.geometry_aux import N_AUX, N_PAIRS, pair_geometry, standardize

TABLE = (1.0, 2.0)                                           # width, length in metres (so normalised (x, y) → (x·1, y·2))
POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]            # (0,0) (0,.5) (0,1) (1,0) (1,.5) (1,1), normalised


def _obs(cue, balls):
    """balls: three (x, y, pocketed) normalised."""
    return np.array([*cue, *[v for b in balls for v in b], *POCKETS], dtype=np.float64)[None]


def test_shapes_and_valid_mask_follow_the_pocketed_flags():
    obs = _obs((0.5, 0.2), [(0.3, 0.5, 0), (0.7, 0.5, 1), (0.5, 0.8, 0)])
    t, v = pair_geometry(obs, TABLE)
    assert t.shape == (1, N_PAIRS, N_AUX) and v.shape == (1, N_PAIRS) and t.dtype == np.float32
    assert v[0, 0:6].all() and not v[0, 6:12].any() and v[0, 12:18].all()          # ball 1 is pocketed: its 6 pairs are invalid
    assert (t[0, 6:12] == 0).all()


def test_ball_to_cushion_is_the_nearest_rail_in_metres_and_the_same_for_all_pockets_of_a_ball():
    obs = _obs((0.5, 0.2), [(0.1, 0.5, 0), (0.5, 0.5, 0), (0.5, 0.95, 0)])        # metres: (0.1, 1.0), (0.5, 1.0), (0.5, 1.9)
    t, _ = pair_geometry(obs, TABLE)
    assert t[0, 0:6, 1] == pytest.approx(0.1)                                      # 0.1 m from the left rail
    assert t[0, 6:12, 1] == pytest.approx(0.5)                                     # 0.5 m from the side rails, 1.0 from the ends
    assert t[0, 12:18, 1] == pytest.approx(0.1)                                    # 1.9 m → 0.1 m from the far end rail


def test_pocket_entry_deviation_is_zero_on_the_pocket_axis_and_large_along_the_rail():
    # corner pocket 0 at (0, 0); table centre at (0.5, 1.0) m. A ball on the line pocket→centre enters straight on.
    on_axis = _obs((0.9, 1.8), [(0.25, 0.25, 0), (0.6, 0.9, 1), (0.6, 0.8, 1)])    # metres (0.25, 0.5) = half way to the centre
    t, _ = pair_geometry(on_axis, TABLE)
    assert t[0, 0, 0] == pytest.approx(0.0, abs=1e-4)
    along = _obs((0.9, 1.8), [(0.5, 0.01, 0), (0.6, 0.9, 1), (0.6, 0.8, 1)])       # ball (0.5, 0.02) m: along the bottom rail from pocket 0
    t2, _ = pair_geometry(along, TABLE)
    assert t2[0, 0, 0] > 60.0


def test_blockers_are_segment_distances_capped_and_ignore_pocketed_balls():
    # cue (0.5, 0.2) → ball 0 (0.5, 1.0) m is a vertical segment x = 0.5, y in [0.4, 2.0]·… in metres: cue (0.5, 0.4), ball (0.5, 1.0)
    obs = _obs((0.5, 0.2), [(0.5, 0.5, 0), (0.62, 0.35, 0), (0.1, 0.9, 1)])        # ball 1 at (0.62, 0.7) m, 0.12 m beside the cue→ball path
    t, v = pair_geometry(obs, TABLE, blocker_cap=0.6)
    assert t[0, 0, 2] == pytest.approx(0.12, abs=1e-6)                             # nearest other live ball to the cue→ball path
    assert t[0, 0, 3] <= 0.6 and v[0, 0]
    alone = _obs((0.5, 0.2), [(0.5, 0.5, 0), (0.62, 0.35, 1), (0.1, 0.9, 1)])      # the others are pocketed
    t2, _ = pair_geometry(alone, TABLE, blocker_cap=0.6)
    assert t2[0, 0, 2] == pytest.approx(0.6) and t2[0, 0, 3] == pytest.approx(0.6)  # no blockers → the cap


def test_blocker_distance_uses_the_segment_not_the_infinite_line():
    # another ball far beyond the target ball on the same line must not count as a blocker of the cue→ball segment
    obs = _obs((0.5, 0.1), [(0.5, 0.3, 0), (0.5, 0.9, 0), (0.1, 0.9, 1)])          # ball 1 is on the cue→ball line but past ball 0
    t, _ = pair_geometry(obs, TABLE, blocker_cap=3.0)
    assert t[0, 0, 2] > 1.0                                                          # distance to the segment end (≈ 1.2 m), not 0


def test_standardize_uses_valid_train_entries_only_and_masks_the_rest():
    rng = np.random.default_rng(0)
    obs = np.concatenate([_obs((0.5, 0.2), [(float(x), float(y), 0), (0.7, 0.5, 1), (0.5, 0.8, 0)]) for x, y in rng.uniform(0.1, 0.9, size=(40, 2))])
    t, v = pair_geometry(obs, TABLE)
    train = np.arange(40) < 30
    z, mean, std = standardize(t, v, train)
    assert z.shape == t.shape and mean.shape == (N_AUX,) and (std > 0).all()
    zt = z[train][v[train]]
    assert np.allclose(zt.mean(0), 0.0, atol=1e-5) and np.allclose(zt.std(0), 1.0, atol=1e-4)
    assert (z[~v] == 0).all()                                                        # invalid pairs are zero


# ── the model side: aux head, aux loss, training loop ──

import torch  # noqa: E402

from exp16_wm.bc import train_bc, dim_weights  # noqa: E402
from exp16_wm.gnn_policy import PairGNNPolicy  # noqa: E402
from exp16_wm.obs_transform import relative_obs  # noqa: E402


def _rel_rows(n, seed=0):
    rng = np.random.default_rng(seed)
    rows, raws = [], []
    for _ in range(n):
        balls = [(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), 0) for _ in range(3)]
        raw = np.array([0.5, 0.2, *[v for b in balls for v in b], *POCKETS], dtype=np.float32)
        raws.append(raw); rows.append(relative_obs(raw))
    return np.stack(raws), np.stack(rows)


def test_default_model_has_no_aux_head_and_aux_loss_refuses_to_run():
    m = PairGNNPolicy(hidden=16)
    assert m.aux_dim == 0 and not hasattr(m, "aux_head") and "aux_head.0.weight" not in m.state_dict()
    with pytest.raises(ValueError):
        m.aux_loss(torch.zeros(1, 21), torch.zeros(1, 18, 4))
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, aux_dim=-1)


def test_aux_loss_ignores_invalid_pairs_and_trains_the_encoder_and_the_aux_head():
    raws, rel = _rel_rows(8)
    rel = torch.as_tensor(rel)
    t, v = pair_geometry(raws, TABLE)
    z, _, _ = standardize(t, v, np.ones(8, bool))
    m = PairGNNPolicy(hidden=16, aux_dim=N_AUX)
    base = m.aux_loss(rel, torch.as_tensor(z))
    z2 = z.copy(); z2[~v] = 123.0                                                    # garbage in the invalid pairs must not matter
    assert torch.allclose(m.aux_loss(rel, torch.as_tensor(z2)), base)
    m.zero_grad(); base.backward()
    assert m.aux_head[0].weight.grad.abs().sum() > 0
    assert m.pair[0].weight.grad.abs().sum() > 0 and m.ball[0].weight.grad.abs().sum() > 0     # the encoder receives the signal


def test_aux_loss_can_be_minimised_and_state_dict_round_trips():
    raws, rel = _rel_rows(64, seed=1)
    rel = torch.as_tensor(rel); t, v = pair_geometry(raws, TABLE); z = torch.as_tensor(standardize(t, v, np.ones(64, bool))[0])
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=32, aux_dim=N_AUX)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = float(m.aux_loss(rel, z))
    for _ in range(150):
        opt.zero_grad(); l = m.aux_loss(rel, z); l.backward(); opt.step()
    assert float(l) < 0.5 * first
    m2 = PairGNNPolicy(hidden=32, aux_dim=N_AUX); m2.load_state_dict(m.state_dict())
    assert torch.allclose(m2.aux_loss(rel, z), m.aux_loss(rel, z))


def test_train_bc_with_aux_runs_changes_the_aux_head_and_validates_the_row_count():
    raws, rel = _rel_rows(48, seed=2)
    n = len(rel)
    t, v = pair_geometry(raws, TABLE); tr = np.arange(n) < 36
    aux, _, _ = standardize(t, v, tr)
    data = {"obs": rel, "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
            "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
    m = PairGNNPolicy(hidden=16, aux_dim=N_AUX)
    before = {k: x.clone() for k, x in m.aux_head.state_dict().items()}
    train_bc(m, data, tr, ~tr, "nearest_good", epochs=3, batch=16, lr=3e-3, patience=10, seed=0, log_every=0, select="loss",
             w=dim_weights(0.2, 0.5), aux=aux, aux_weight=5.0)
    assert any(not torch.equal(before[k], x) for k, x in m.aux_head.state_dict().items())
    with pytest.raises(ValueError):
        train_bc(PairGNNPolicy(hidden=16, aux_dim=N_AUX), data, tr, ~tr, "nearest_good", epochs=1, batch=16, lr=3e-3, patience=10, seed=0,
                 log_every=0, w=dim_weights(0.2, 0.5), aux=aux[:10], aux_weight=5.0)


def test_aux_weight_zero_leaves_the_training_loss_unchanged():
    raws, rel = _rel_rows(32, seed=3)
    n = len(rel); t, v = pair_geometry(raws, TABLE); tr = np.arange(n) < 24; aux, _, _ = standardize(t, v, tr)
    data = {"obs": rel, "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
            "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
    hist = []
    for use_aux in (False, True):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, aux_dim=N_AUX if use_aux else 0)
        fit = train_bc(m, data, tr, ~tr, "nearest_good", epochs=2, batch=16, lr=3e-3, patience=10, seed=0, log_every=0, select="loss",
                       w=dim_weights(0.2, 0.5), aux=aux if use_aux else None, aux_weight=0.0)
        hist.append(fit["history"][0][1])
    assert hist[0] == pytest.approx(hist[1], rel=1e-6)                              # weight 0 → identical training loss


def test_aux_history_logs_train_and_val_mse_per_epoch_without_changing_training():
    raws, rel = _rel_rows(40, seed=4)
    n = len(rel); t, v = pair_geometry(raws, TABLE); tr = np.arange(n) < 30; aux, _, _ = standardize(t, v, tr)
    data = {"obs": rel, "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
            "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
    fits = []
    for w_aux in (0.0, 5.0):
        torch.manual_seed(0)
        m = PairGNNPolicy(hidden=16, aux_dim=N_AUX)
        fits.append(train_bc(m, data, tr, ~tr, "nearest_good", epochs=4, batch=16, lr=3e-3, patience=10, seed=0, log_every=0,
                             select="loss", w=dim_weights(0.2, 0.5), aux=aux, aux_weight=w_aux))
    assert fits[0]["aux_history"] == []                                              # weight 0 → nothing logged
    h = np.array(fits[1]["aux_history"])
    assert h.shape == (4, 3) and (h[:, 0] == np.arange(4)).all() and np.isfinite(h[:, 1:]).all() and (h[:, 1:] > 0).all()
    assert h[-1, 1] < h[0, 1]                                                        # the train aux MSE falls over a few epochs
    # logging adds only a (detached) value: the main loss at epoch 0 minus the weighted aux term equals the weight-0 run
    assert fits[1]["history"][0][1] - 5.0 * h[0, 1] == pytest.approx(fits[0]["history"][0][1], abs=0.05)   # the term is ~5 of ~55000
