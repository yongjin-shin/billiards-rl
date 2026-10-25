"""Tests for exp16_wm.train_vmap: the masked Gaussian convolution, the map loss, the layout loader and a few learning steps on a tiny model."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.train_vmap import blur_masked, gaussian_kernel, layout_loss, load_layouts, map_loss, to_grid

W, L, R = 0.9906, 1.9812, 0.028575
RAW = np.array([0.2, 0.2, 0.5, 0.5, 0.0, 0.3, 0.7, 0.0] + [0.0] * 12, np.float32)


def test_gaussian_kernel_is_normalised_and_sigma_zero_is_the_identity():
    k = gaussian_kernel(1.2)
    assert k.shape[-1] == k.shape[-2] == 2 * int(np.ceil(3.6)) + 1 and float(k.sum()) == pytest.approx(1.0)
    assert float(k[0, 0, k.shape[-2] // 2, k.shape[-1] // 2]) == float(k.max())                   # the peak is at the centre
    assert gaussian_kernel(0.0).shape == (1, 1, 1, 1) and float(gaussian_kernel(0.0)) == 1.0


def test_to_grid_places_values_and_marks_free_cells():
    ij = torch.tensor([[0, 0], [2, 1], [3, 3]])
    g, m = to_grid(torch.tensor([5.0, 6.0, 7.0]), ij, (4, 4))
    assert g.shape == m.shape == (1, 1, 4, 4) and g[0, 0, 2, 1] == 6.0 and g[0, 0, 1, 1] == 0.0
    assert float(m.sum()) == 3.0 and m[0, 0, 3, 3] == 1.0 and m[0, 0, 0, 1] == 0.0


def test_blur_masked_keeps_constants_ignores_empty_cells_and_averages_neighbours():
    ij = torch.tensor([[i, j] for i in range(6) for j in range(6) if (i, j) != (2, 2)])           # one empty cell in the middle
    g, m = to_grid(torch.full((len(ij),), 0.7), ij, (6, 6))
    b = blur_masked(g, m, gaussian_kernel(1.0))
    assert torch.allclose(b[m.bool()], torch.tensor(0.7), atol=1e-5)                                # no dark halo around the hole or at the border
    vals = torch.zeros(len(ij)); vals[0] = 1.0
    g2, m2 = to_grid(vals, ij, (6, 6))
    b2 = blur_masked(g2, m2, gaussian_kernel(1.0))
    assert b2[0, 0, 0, 0] < 1.0 and b2[0, 0, 0, 1] > 0.0 and b2[0, 0, 5, 5] < b2[0, 0, 0, 1]      # a spike spreads to near cells and fades with distance
    ident = blur_masked(g2, m2, gaussian_kernel(0.0))
    assert torch.allclose(ident[m2.bool()], g2[m2.bool()])


def test_map_loss_prefers_the_right_map_has_gradients_and_sigma_zero_is_the_plain_bce():
    ij = torch.tensor([[i, j] for i in range(5) for j in range(5)])
    label = torch.tensor([float((i + j) % 2) * 0.8 + 0.1 for i, j in ij.tolist()])
    good = label.clone().requires_grad_(True)
    bad = (1 - label).clone()
    k = gaussian_kernel(0.0)
    assert map_loss(good, label, ij, (5, 5), k) < map_loss(bad, label, ij, (5, 5), k)
    map_loss(good, label, ij, (5, 5), k).backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()
    plain = torch.nn.functional.binary_cross_entropy(label.clamp(1e-6, 1 - 1e-6), label)
    assert float(map_loss(label, label, ij, (5, 5), k)) == pytest.approx(float(plain), abs=1e-5)
    # blurring both maps the same way makes a checkerboard disagreement much smaller than pointwise
    assert map_loss(bad, label, ij, (5, 5), gaussian_kernel(1.5)) < map_loss(bad, label, ij, (5, 5), k)


def _layout_files(tmp_path):
    from exp16_wm.v_map_data import build_layout, finish_layout, run_layouts, write_file
    lays = [build_layout(RAW, 2, 11 + i, i, (10.0, 30.0), 1, 0.1, 0.0, R, W, L) for i in range(3)]
    ks = run_layouts(lays, lambda jobs: [np.floor(o[:, 0] * 4 + o[:, 1] * 2).astype(int) % 3 for o, _ in jobs])
    fins = [finish_layout(l, k, 2, 1) for l, k in zip(lays, ks)]
    out = str(tmp_path / "m.npz")
    write_file(out, np.stack([RAW] * 3), np.array([2, 2, 2]), np.array([11, 12, 13]), lays, fins, (10.0, 30.0))
    return out, lays


def test_load_layouts_gives_one_entry_per_layout_with_aligned_arrays_and_rejects_files_without_a_grid(tmp_path):
    out, lays = _layout_files(tmp_path)
    ls = load_layouts([out])
    assert [l["gid"] for l in ls] == [11, 12, 13] and all(l["nxy"] == (10, 20) for l in ls)
    assert all(len(l["rel"]) == len(l["ij"]) == len(l["label"]) == len(lay["cells"]) for l, lay in zip(ls, lays))
    assert ls[0]["rel"].shape[1] == 18 and ls[0]["left"] == 2
    z = dict(np.load(out, allow_pickle=True)); z.pop("cell_ij")
    np.savez(tmp_path / "old.npz", **z)
    with pytest.raises(ValueError):
        load_layouts([str(tmp_path / "old.npz")])


def test_a_tiny_model_can_fit_one_layout_map_by_gradient_steps(tmp_path):
    from exp16_wm.gnn_policy import PairGNNPolicy
    out, _ = _layout_files(tmp_path)
    lay = load_layouts([out])[0]
    torch.manual_seed(0)
    model = PairGNNPolicy(hidden=8, v_head=True, qv_encoder=True)
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in ("qv_bb", "qv_ball", "qv_pair", "head_v")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    k = gaussian_kernel(1.0)
    first = layout_loss(model, lay, k).item()
    for _ in range(60):
        loss = layout_loss(model, lay, k)
        opt.zero_grad(); loss.backward(); opt.step()
    assert np.isfinite(float(loss)) and float(loss) < first                                          # learning reduces the map loss
