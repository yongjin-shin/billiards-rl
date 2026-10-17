"""
exp16_wm/obs_transform.py — relative-coordinate observation in the action's frame.

The action's delta_angle is measured from the cue → nearest-unpocketed-ball
direction (BilliardsEnv.step). This transform expresses every ball and pocket in
that same frame: cue ball at the origin, cue → nearest ball along +x. Only a
choice of coordinate frame — no billiards formulas (ghost ball etc.).

  raw obs (23) : [cue_xy, (bx, by, pocketed) × 3, pocket_xy × 6]   normalised [0, 1]
  relative (21): [(rx, ry, pocketed) × 3, (px, py) × 6]            metres, rotated
                 pocketed balls → (0, 0, 1)
"""

import numpy as np

from exp16_wm.networks import TABLE_W, TABLE_L

N_BALLS = 3
N_POCKETS = 6
RAW_DIM = 2 + 3 * N_BALLS + 2 * N_POCKETS
REL_DIM = 3 * N_BALLS + 2 * N_POCKETS
_SCALE = np.array([TABLE_W, TABLE_L])


def _split(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    assert o.shape[1] == RAW_DIM, f"expected raw obs of dim {RAW_DIM}, got {o.shape[1]}"
    cue = o[:, 0:2] * _SCALE
    balls = o[:, 2:2 + 3 * N_BALLS].reshape(-1, N_BALLS, 3)
    pockets = o[:, 2 + 3 * N_BALLS:].reshape(-1, N_POCKETS, 2) * _SCALE
    return cue, balls[..., :2] * _SCALE, balls[..., 2] > 0.5, pockets


def aim_reference_from_obs(obs: np.ndarray) -> np.ndarray:
    """phi_direct (rad) the env adds delta_angle to; 0 when every ball is pocketed. Shape (N,)."""
    cue, bpos, pocketed, _ = _split(obs)
    rel = bpos - cue[:, None]
    d = np.where(pocketed, np.inf, np.linalg.norm(rel, axis=-1))
    idx = np.argmin(d, axis=1)                    # first minimum, like the env's strict '<' loop
    ref = rel[np.arange(len(rel)), idx]
    return np.where(np.isfinite(d.min(axis=1)), np.arctan2(ref[:, 1], ref[:, 0]), 0.0)


def relative_obs(obs: np.ndarray) -> np.ndarray:
    """Raw obs (23,) or (N, 23) → relative obs (21,) or (N, 21), float32."""
    single = np.asarray(obs).ndim == 1
    cue, bpos, pocketed, pockets = _split(obs)
    phi = aim_reference_from_obs(obs)
    c, s = np.cos(-phi)[:, None], np.sin(-phi)[:, None]

    def rotate(v: np.ndarray) -> np.ndarray:          # v (N, k, 2)
        return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)

    rb = rotate(bpos - cue[:, None])
    rb[pocketed] = 0.0
    rp = rotate(pockets - cue[:, None])
    out = np.concatenate([np.concatenate([rb, pocketed[..., None].astype(np.float64)], axis=-1)
                          .reshape(len(rb), -1), rp.reshape(len(rp), -1)], axis=1).astype(np.float32)
    return out[0] if single else out


TRANSFORMS = {"raw": lambda o: np.asarray(o, dtype=np.float32), "relative": relative_obs}
OBS_DIMS = {"raw": RAW_DIM, "relative": REL_DIM}
