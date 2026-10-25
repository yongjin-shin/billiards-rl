"""
exp16_wm/cue_walls.py — which walls did a one- or two-cushion shot hit, in order? (CT2b-2, CT2c)

The collected shots (exp16_wm.cue_goal_data) only count cushion events. This replays the shots with exactly one cushion event, reads the id of the
cushion and where the cue ball touched it, and names each wall from where its contacts lie: the two halves of each long side, the two short
sides, and the pocket jaws (many small cushions with ids ending in 't', one group).

Usage:
    python -m exp16_wm.cue_walls label --data logs/ct0b/collect_200.npz --out logs/ct0b/walls_k1.npz --workers 8
    python -m exp16_wm.cue_walls label --data logs/ct0b/collect_200.npz --out logs/ct0b/walls_k2.npz --k 2 --names-from logs/ct0b/walls_k1.npz

A route is the ordered walls of a shot as one integer: sum of wall_index * N_WALLS**(k - 1 - position), so one cushion = the wall index itself.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import TABLE_L, TABLE_W
from exp16_wm.single_ball import CUSHION_EVENTS, make_cue_only_system
from simulator import play_shot

WALLS = ["왼쪽 아래", "왼쪽 위", "오른쪽 아래", "오른쪽 위", "아래 짧은 변", "위 짧은 변", "포켓 모서리"]
N_WALLS = len(WALLS)


def replay(start: np.ndarray, phi: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per shot (exactly one cushion event): the id of the cushion, the cue ball position at that contact, and where it ended."""
    system = make_cue_only_system(float(start[0]), float(start[1]))
    ids, contact, end = [], np.zeros((len(phi), 2)), np.zeros((len(phi), 2))
    for i in range(len(phi)):
        out = play_shot(system, float(phi[i]), float(v[i]))
        hit = [e for e in out.events if e.event_type in CUSHION_EVENTS][0]
        ids.append(next(str(a.id) for a in hit.agents if str(a.id) != "cue"))
        contact[i] = next(a.final.state.rvw[0][:2] for a in hit.agents if str(a.id) == "cue")
        end[i] = out.balls["cue"].state.rvw[0][:2]
    return np.array(ids, dtype="U8"), contact, end


def replay_seq(start: np.ndarray, phi: np.ndarray, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per shot with exactly k cushion events: their ids (n, k) and the cue ball position at each contact (n, k, 2), and where it ended."""
    system = make_cue_only_system(float(start[0]), float(start[1]))
    ids = np.empty((len(phi), k), dtype="U8"); contact = np.zeros((len(phi), k, 2)); end = np.zeros((len(phi), 2))
    for i in range(len(phi)):
        out = play_shot(system, float(phi[i]), float(v[i]))
        hits = [e for e in out.events if e.event_type in CUSHION_EVENTS][:k]
        for j, h in enumerate(hits):
            ids[i, j] = next(str(a.id) for a in h.agents if str(a.id) != "cue")
            contact[i, j] = next(a.final.state.rvw[0][:2] for a in h.agents if str(a.id) == "cue")
        end[i] = out.balls["cue"].state.rvw[0][:2]
    return ids, contact, end


def route_code(wall_idx: np.ndarray) -> np.ndarray:
    """(n, k) wall indices -> (n,) route integers (first wall is the most significant digit)."""
    code = np.zeros(len(wall_idx), np.int64)
    for j in range(wall_idx.shape[1]):
        code = code * N_WALLS + wall_idx[:, j]
    return code


def route_digits(code: np.ndarray, n_seq: int) -> np.ndarray:
    """(n,) route integers -> (n, n_seq) wall indices."""
    code = np.asarray(code, np.int64)
    return np.stack([(code // N_WALLS ** (n_seq - 1 - j)) % N_WALLS for j in range(n_seq)], axis=1)


def wall_names(ids: np.ndarray, contact: np.ndarray) -> dict[str, str]:
    """Name each cushion id by the median of its contact positions; ids ending in 't' (pocket jaws) are one group."""
    names = {}
    for cid in np.unique(ids):
        cid = str(cid)
        if cid.endswith("t"):
            names[cid] = "포켓 모서리"; continue
        c = np.median(contact[ids == cid], axis=0)
        if c[1] < 0.2: names[cid] = "아래 짧은 변"
        elif c[1] > TABLE_L - 0.2: names[cid] = "위 짧은 변"
        else: names[cid] = ("왼쪽 " if c[0] < TABLE_W / 2 else "오른쪽 ") + ("아래" if c[1] < TABLE_L / 2 else "위")
    return names


def wall_index(ids: np.ndarray, names: dict[str, str]) -> np.ndarray:
    lut = {cid: WALLS.index(name) for cid, name in names.items()}
    return np.array([lut[str(c)] for c in ids], np.int8)


def _label_task(task: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
    start, phi, v, k = task
    ids, contact, _ = replay_seq(start, phi, v, k)
    return ids, contact


def rows_with_k(z: dict, k: int) -> np.ndarray:
    """Indices of the shots with exactly k cushions that end on the table and are not scratches (same filter as cue_goal_head.load_rows)."""
    end = z["end"]
    on_table = (end[:, 0] >= 0) & (end[:, 0] <= TABLE_W) & (end[:, 1] >= 0) & (end[:, 1] <= TABLE_L)
    return np.nonzero(on_table & ~z["scratch"] & (z["k"] == k))[0]


def k1_rows(z: dict) -> np.ndarray:
    return rows_with_k(z, 1)


def label(data_path: str, out: str, workers: int, k: int = 1, names_from: str | None = None) -> None:
    from exp16_wm.code_version import code_version
    z = {key: val for key, val in np.load(data_path).items()}
    rows = rows_with_k(z, k)
    sidx = z["start_idx"][rows]
    per_start = [rows[sidx == s] for s in range(len(z["starts"]))]
    tasks = [(z["starts"][s], z["phi"][r], z["v"][r], k) for s, r in enumerate(per_start)]
    t0, parts = time.time(), []
    if workers <= 1:
        parts = [_label_task(t) for t in tasks]
    else:
        with mp.Pool(workers) as pool:
            for n, p in enumerate(pool.imap(_label_task, tasks)):
                parts.append(p)
                if (n + 1) % 20 == 0:
                    print(f"[walls] start {n + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    ids = np.concatenate([p[0] for p in parts]); contact = np.concatenate([p[1] for p in parts])
    names = json.loads(str(np.load(names_from)["id_names"])) if names_from else {}
    new = np.array([c not in names for c in ids.ravel()]).reshape(ids.shape)   # ids the given map does not know are named from their contacts
    if new.any():
        names.update(wall_names(ids[new], contact[new]))
    flat = wall_index(ids.ravel(), names).reshape(ids.shape)
    code = route_code(flat)
    np.savez_compressed(out, rows=np.concatenate(per_start), wall=code.astype(np.int16), n_seq=k, id_names=json.dumps(names, ensure_ascii=False),
                        commit=code_version())
    print("saved", out, f"{len(code):,} shots with {k} cushions;", f"{len(np.unique(code))} different routes")
    print("first wall:", {WALLS[i]: int((flat[:, 0] == i).sum()) for i in range(N_WALLS)})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--data", required=True); lb.add_argument("--out", required=True); lb.add_argument("--workers", type=int, default=8)
    lb.add_argument("--k", type=int, default=1); lb.add_argument("--names-from", default=None)
    args = ap.parse_args()
    label(args.data, args.out, args.workers, args.k, args.names_from)


if __name__ == "__main__":
    main()
