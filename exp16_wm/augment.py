"""
exp16_wm/augment.py — table-symmetry augmentation for imitation data.

The table and pooltool physics are symmetric under left-right (x → W − x) and
top-bottom (y → L − y) reflection. Reflecting the cue/ball positions maps a
shot to its mirror image; because delta_angle is measured from the cue →
nearest-ball direction (distances are preserved, so the nearest ball is the
same), the matching action is delta → −delta for one reflection and unchanged
for both (a 180° rotation). Speed is unchanged. Validated with pooltool before
use: same balls pocketed + same scratch in 99.2% of teacher shots, 100% of
random shots (experiments.md, 2026-10-09).

Raw obs (normalised): [cue_xy, (bx, by, pocketed) × 3, pocket_xy × 6]. The
pocket list is left as is — the reflected table has the same pockets in the
same order. Pocketed balls stay (0, 0, 1).
"""

import numpy as np

N_BALLS = 3
FLIPS: list[tuple[bool, bool]] = [(False, False), (True, False), (False, True), (True, True)]


def reflect_obs(obs: np.ndarray, lr: bool, tb: bool) -> np.ndarray:
    """Reflect raw obs (…, 23). lr: x → 1 − x, tb: y → 1 − y for the cue and unpocketed balls."""
    out = np.array(obs, dtype=np.float32, copy=True)
    cols_x = [0] + [2 + 3 * i for i in range(N_BALLS)]
    cols_y = [1] + [3 + 3 * i for i in range(N_BALLS)]
    live = np.ones(out.shape[:-1] + (N_BALLS + 1,), dtype=bool)     # cue is always on the table in obs
    live[..., 1:] = out[..., [4 + 3 * i for i in range(N_BALLS)]] < 0.5
    if lr:
        out[..., cols_x] = np.where(live, 1.0 - out[..., cols_x], out[..., cols_x])
    if tb:
        out[..., cols_y] = np.where(live, 1.0 - out[..., cols_y], out[..., cols_y])
    return out


def reflect_action(action: np.ndarray, lr: bool, tb: bool) -> np.ndarray:
    """(…, 2) actions (delta_angle, speed); NaN padding passes through."""
    out = np.array(action, dtype=np.float32, copy=True)
    if lr ^ tb:
        out[..., 0] = -out[..., 0]
    return out


def augment_flip4(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """
    The rows `idx` of a teacher dataset, each in all 4 reflections (identity first).
    Keys used: obs, action, good_actions, n_good, pocketed, shot, game.
    """
    parts = []
    for lr, tb in FLIPS:
        parts.append({
            "obs":          reflect_obs(data["obs"][idx], lr, tb),
            "action":       reflect_action(data["action"][idx], lr, tb),
            "good_actions": reflect_action(data["good_actions"][idx], lr, tb),
            **{k: data[k][idx] for k in ["n_good", "pocketed", "shot", "game"]},
        })
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
