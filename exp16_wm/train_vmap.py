"""
exp16_wm/train_vmap.py — V8: train V on whole value maps (one sample = one layout's n × m map) with a convolution at the end.

The V model is the one exp16_wm.train_v trains (the Q / V encoder and head; the made encoder and the policy stay frozen). For every layout the
model gives a value for each free cell of the layout's cue-position grid (all in one batch); the fractions sigmoid(logit) are put on the grid and
blurred with a Gaussian (a convolution that ignores the empty cells), and so is the label map; the loss is the BCE between the two blurred maps.
With sigma 0 the blur is the identity and this is point-wise training on the same data.

Usage:
    python -m exp16_wm.train_vmap --run "<PRm run dir glob>" --data "logs/teacher_data/v8/smoke/*.npz" --sigma-cm 8 --out logs/experiments_v2/v8_map
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_layouts(paths: list[str]) -> list[dict]:
    """One dict per layout of the V8 files: gid, left, rel (C, D) relative observations of its free cells, ij (C, 2) grid cell numbers,
    label (C,) the training label (balls), nxy (nx, ny) the grid size."""
    from exp16_wm.obs_transform import relative_obs
    out = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if "cell_ij" not in z.files:
            raise ValueError(f"{p} has no grid cell numbers (cell_ij): it was not made on a regular grid")
        nxy = tuple(int(v) for v in z["grid_nxy"])
        for gid in np.unique(z["layout_id"]):
            m = z["layout_id"] == gid
            out.append({"gid": int(gid), "left": int(z["left"][m][0]), "rel": relative_obs(z["obs"][m]).astype(np.float32),
                        "ij": z["cell_ij"][m].astype(np.int64), "label": z["k"][m].astype(np.float32), "nxy": nxy})
    if not out:
        raise FileNotFoundError("no V8 files")
    return out


def gaussian_kernel(sigma_cells: float) -> torch.Tensor:
    """(1, 1, K, K) normalised Gaussian kernel (the identity for sigma 0), K = 2 · ceil(3σ) + 1."""
    if sigma_cells <= 0:
        return torch.ones(1, 1, 1, 1)
    r = int(np.ceil(3 * sigma_cells))
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-ax ** 2 / (2 * sigma_cells ** 2))
    k2 = k1[:, None] * k1[None, :]
    return (k2 / k2.sum())[None, None]


def to_grid(values: torch.Tensor, ij: torch.Tensor, nxy: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """values (C,) at grid cells ij (C, 2) → (1, 1, nx, ny) map (0 on empty cells) and the 0/1 mask of the free cells."""
    grid = torch.zeros(nxy[0], nxy[1], dtype=values.dtype)
    mask = torch.zeros(nxy[0], nxy[1], dtype=values.dtype)
    grid = grid.index_put((ij[:, 0], ij[:, 1]), values)
    mask[ij[:, 0], ij[:, 1]] = 1.0
    return grid[None, None], mask[None, None]


def blur_masked(grid: torch.Tensor, mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolution of (1, 1, nx, ny) grid with the kernel that ignores the empty cells: conv(grid · mask) / conv(mask)."""
    pad = kernel.shape[-1] // 2
    num = F.conv2d(grid * mask, kernel, padding=pad)
    den = F.conv2d(mask, kernel, padding=pad)
    return num / den.clamp_min(1e-8)


def map_loss(frac_pred: torch.Tensor, frac_label: torch.Tensor, ij: torch.Tensor, nxy: tuple[int, int], kernel: torch.Tensor) -> torch.Tensor:
    """BCE between the blurred predicted and the blurred label map, averaged over the free cells (both maps hold fractions of the balls left)."""
    gp, mask = to_grid(frac_pred, ij, nxy)
    gl, _ = to_grid(frac_label, ij, nxy)
    bp, bl = blur_masked(gp, mask, kernel), blur_masked(gl, mask, kernel)
    m = mask.bool()
    return F.binary_cross_entropy(bp[m].clamp(1e-6, 1 - 1e-6), bl[m])


def layout_loss(model, lay: dict, kernel: torch.Tensor, logits: torch.Tensor | None = None) -> torch.Tensor:
    """The loss of one layout (its cells' logits are computed here unless given)."""
    from exp16_wm.train_v import v_logits
    z = v_logits(model, torch.as_tensor(lay["rel"])) if logits is None else logits
    label = torch.as_tensor(np.clip(lay["label"] / max(1, lay["left"]), 0.0, 1.0))
    return map_loss(torch.sigmoid(z), label, torch.as_tensor(lay["ij"]), lay["nxy"], kernel)


def main() -> None:
    from exp16_wm import eval_by_balls as E
    from exp16_wm.code_version import code_version
    from exp16_wm.paired_eval import find_run
    from exp16_wm.train_v import V_MODULES, widen_qv
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sigma-cm", type=float, default=8.0, help="Gaussian blur of the maps in cm (0 = point-wise)"); p.add_argument("--step-cm", type=float, default=7.0)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--layouts-per-step", type=int, default=4); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--init", default="fresh", choices=["fresh", "keep"], help="fresh: a new Q / V encoder (like --qv-fourier 0 in train_v); keep: the run's own")
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    run = find_run(args.run)
    model = E.load_run(run)
    run_args = json.load(open(os.path.join(run, "results.json")))["args"]
    if args.init == "fresh":
        model, run_args = widen_qv(model, run_args, model.qv_hidden, 0)
    for name, prm in model.named_parameters():
        prm.requires_grad = name.split(".")[0] in V_MODULES
    params = [q for q in model.parameters() if q.requires_grad]
    lays = load_layouts(sorted(glob.glob(args.data)))
    rng = np.random.default_rng(args.seed)
    val_idx = set(rng.choice(len(lays), size=max(1, int(round(0.1 * len(lays)))), replace=False).tolist())
    tr = [l for i, l in enumerate(lays) if i not in val_idx]; va = [l for i, l in enumerate(lays) if i in val_idx]
    kernel = gaussian_kernel(args.sigma_cm / args.step_cm)
    print(f"{len(tr)} train / {len(va)} val layouts, {sum(len(l['label']) for l in tr):,} train cells, grid {lays[0]['nxy']}, blur σ {args.sigma_cm:g} cm = {args.sigma_cm / args.step_cm:.2f} cells", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    from exp16_wm.train_v import v_logits
    best, best_state, hist, tr_hist, t0 = float("inf"), None, [], [], time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        tl, tn = 0.0, 0
        for s in range(0, len(order), args.layouts_per_step):
            batch = [tr[i] for i in order[s:s + args.layouts_per_step]]
            z = v_logits(model, torch.as_tensor(np.concatenate([l["rel"] for l in batch])))
            parts = torch.split(z, [len(l["label"]) for l in batch])
            loss = torch.stack([layout_loss(model, l, kernel, pz) for l, pz in zip(batch, parts)]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); tn += 1
        tr_hist.append(tl / max(1, tn))
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([layout_loss(model, l, kernel).item() for l in va]))
        hist.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep}: train loss {tr_hist[-1]:.4f}  val loss {vl:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state); model.eval()
    os.makedirs(args.out, exist_ok=True)
    a = json.load(open(os.path.join(run, "results.json")))
    a["args"] = {**run_args, "v_weight": 1.0, "qv_encoder": True, "v_horizon": 0}
    a["v2"] = {**vars(args), "from_run": run, "best_val_loss": best, "val_history": hist, "train_history": tr_hist, "n_train": len(tr), "git_commit": code_version()}
    json.dump(a, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    torch.save({"gnn": model.state_dict()}, os.path.join(args.out, "bc_model.pt"))
    print(f"saved {args.out} (best val {best:.4f} at epoch {hist.index(best)})", flush=True)


if __name__ == "__main__":
    main()
