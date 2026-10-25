"""
exp16_wm/cue_walls.py — which wall did a one-cushion shot hit? (CT2b-2)

The collected shots (exp16_wm.cue_goal_data) only count cushion events. This replays the shots with exactly one cushion event, reads the id of the
cushion and where the cue ball touched it, and names each wall from where its contacts lie: the two halves of each long side, the two short
sides, and the pocket jaws (many small cushions with ids ending in 't', one group).

Usage:
    python -m exp16_wm.cue_walls label --data logs/ct0b/collect_200.npz --out logs/ct0b/walls_k1.npz --workers 8
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


def _label_task(task: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    start, phi, v = task
    ids, contact, _ = replay(start, phi, v)
    return ids, contact


def k1_rows(z: dict) -> np.ndarray:
    """Indices of the shots with exactly one cushion that end on the table and are not scratches (same filter as cue_goal_head.load_rows)."""
    end = z["end"]
    on_table = (end[:, 0] >= 0) & (end[:, 0] <= TABLE_W) & (end[:, 1] >= 0) & (end[:, 1] <= TABLE_L)
    return np.nonzero(on_table & ~z["scratch"] & (z["k"] == 1))[0]


def label(data_path: str, out: str, workers: int) -> None:
    from exp16_wm.code_version import code_version
    z = {k: v for k, v in np.load(data_path).items()}
    rows = k1_rows(z)
    sidx = z["start_idx"][rows]
    per_start = [rows[sidx == s] for s in range(len(z["starts"]))]
    tasks = [(z["starts"][s], z["phi"][r], z["v"][r]) for s, r in enumerate(per_start)]
    t0, parts = time.time(), []
    with mp.Pool(workers) as pool:
        for n, p in enumerate(pool.imap(_label_task, tasks)):
            parts.append(p)
            if (n + 1) % 20 == 0:
                print(f"[walls] start {n + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    ids = np.concatenate([p[0] for p in parts]); contact = np.concatenate([p[1] for p in parts])
    names = wall_names(ids, contact)
    wall = wall_index(ids, names)
    np.savez_compressed(out, rows=np.concatenate(per_start), wall=wall, id_names=json.dumps(names, ensure_ascii=False), commit=code_version())
    print("saved", out, f"{len(wall):,} shots;", {WALLS[i]: int((wall == i).sum()) for i in range(N_WALLS)})
    print("id -> wall:", names)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--data", required=True); lb.add_argument("--out", required=True); lb.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    label(args.data, args.out, args.workers)


if __name__ == "__main__":
    main()
