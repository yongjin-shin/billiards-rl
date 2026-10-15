"""Tests for the Fourier input features of PairGNNPolicy and the training-loss plateau stop of exp16_wm.bc.train_bc."""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.bc import dim_weights, load_aim_encoder, train_bc, train_plateau_converged, train_seed
from exp16_wm.gnn_policy import FOURIER_SCALE, PairGNNPolicy
from exp16_wm.obs_transform import relative_obs

POCKETS = [0, 0, 0, 0.5, 0, 1, 1, 0, 1, 0.5, 1, 1]


def _rel(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        balls = [(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), 0) for _ in range(3)]
        rows.append(relative_obs(np.array([0.5, 0.2, *[v for b in balls for v in b], *POCKETS], dtype=np.float32)))
    return np.stack(rows)


def test_fourier_off_keeps_the_original_encoder_shapes():
    m = PairGNNPolicy(hidden=16)
    assert m.fourier == 0 and m.bb[0].in_features == 9 and m.ball[0].in_features == 5 + 16 and m.pair[0].in_features == 16 + 6
    x = torch.randn(3, 2)
    assert torch.equal(m._ff(x), x)


def test_fourier_features_are_sin_cos_at_octaves_and_widen_every_coordinate_input():
    K = 3
    m = PairGNNPolicy(hidden=16, fourier=K)
    d2 = 2 + 4 * K
    assert m.bb[0].in_features == (d2 + 3) + d2 + 2 and m.ball[0].in_features == d2 + 3 + 16 and m.pair[0].in_features == 16 + 2 * d2 + 2
    x = torch.tensor([[0.3, -1.2]])
    f = m._ff(x)
    assert f.shape == (1, d2) and torch.allclose(f[0, :2], x[0])
    freq = (2.0 ** torch.arange(K)) * np.pi / FOURIER_SCALE
    want_sin = torch.sin(x[0][:, None] * freq).flatten()                              # order: (coordinate, frequency)
    assert torch.allclose(f[0, 2:2 + 2 * K], want_sin, atol=1e-6) and torch.allclose(f[0, 2 + 2 * K:], torch.cos(x[0][:, None] * freq).flatten(), atol=1e-6)
    o = m.forward_full(torch.as_tensor(_rel(4)))
    assert o["e"].shape == (4, 18, 16) and torch.isfinite(o["e"]).all()
    with pytest.raises(ValueError):
        PairGNNPolicy(hidden=16, fourier=-1)


def test_load_aim_encoder_rebuilds_a_fourier_model(tmp_path):
    torch.manual_seed(0)
    m = PairGNNPolicy(hidden=16, aux_dim=1, fourier=2)
    run = tmp_path / "r"; run.mkdir()
    json.dump({"args": {"aux_only": True, "geom_aux_set": "aim", "gnn_hidden": 16, "gnn_layers": 2, "input_fourier": 2}}, open(run / "results.json", "w"))
    m.save(str(run / "bc_model_last.pt"))
    got = load_aim_encoder(str(run))
    assert got.fourier == 2 and all(torch.equal(got.state_dict()[k], v) for k, v in m.state_dict().items())


def test_train_plateau_converged_rule():
    assert not train_plateau_converged([5, 4, 3], 2, 0.01)                             # fewer than 2 windows
    assert not train_plateau_converged([10, 9, 8, 7], 2, 0.01)                         # still falling: 7 < 0.99 · 9
    assert train_plateau_converged([10, 9, 8.99, 8.95], 2, 0.01)                       # 8.95 > 0.99 · 9 = 8.91 → converged
    assert not train_plateau_converged([10, 9, 8.99, 8.95], 0, 0.01)                   # off


def test_train_bc_stops_when_the_training_loss_is_flat():
    rel = _rel(20, seed=1)
    n = len(rel); tr = np.arange(n) < 15
    data = {"obs": rel, "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
            "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
    fit = train_bc(PairGNNPolicy(hidden=16), data, tr, ~tr, "nearest_good", epochs=50, batch=8, lr=0.0, patience=100, seed=0, log_every=0,
                   w=dim_weights(0.2, 0.5), train_plateau=4, train_plateau_tol=0.01)
    assert fit["epochs_run"] == 8 and "converged" in fit["stop_reason"]                # lr 0 → flat loss → stops after 2 windows
    fit2 = train_bc(PairGNNPolicy(hidden=16), data, tr, ~tr, "nearest_good", epochs=6, batch=8, lr=0.0, patience=100, seed=0, log_every=0,
                    w=dim_weights(0.2, 0.5))
    assert fit2["epochs_run"] == 6 and fit2["stop_reason"] is None                     # off by default


def test_train_seed_uses_the_init_seed_only_when_given():
    assert train_seed(3, None) == 3 and train_seed(3, 7) == 7 and train_seed(0, 0) == 0


def test_another_training_seed_on_the_same_split_gives_another_model_and_the_same_seed_repeats():
    rel = _rel(24, seed=2)
    n = len(rel); tr = np.arange(n) < 18
    data = {"obs": rel, "action": np.tile([[0.3, 4.0]], (n, 1)).astype(np.float32),
            "good_actions": np.tile(np.array([[[0.3, 4.0]]], np.float32), (n, 1, 1))}
    def fit(s):
        torch.manual_seed(s)
        m = PairGNNPolicy(hidden=16)
        train_bc(m, data, tr, ~tr, "nearest_good", epochs=2, batch=8, lr=3e-3, patience=10, seed=s, log_every=0, w=dim_weights(0.2, 0.5))
        return m.state_dict()
    a, a2, b = fit(1), fit(1), fit(2)
    assert all(torch.equal(a[k], a2[k]) for k in a)                                    # deterministic
    assert any(not torch.equal(a[k], b[k]) for k in a)                                 # a different member
