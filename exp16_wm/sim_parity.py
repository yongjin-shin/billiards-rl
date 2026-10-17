"""
exp16_wm/sim_parity.py — does the simulator give the same shots on two machines? (Exp-16, cloud move)

The cloud server (Linux x86_64, panda3d 1.11 dev) and the Mac (arm64, panda3d 1.10) run the same pooltool 0.5.0 but not the
same binaries. The pipeline depends on 0.02° aim steps (success-window edges), so before mixing results from both machines this
plays one fixed set of (state, action) cases on each machine and compares outcomes, event sequences and final ball positions.

Cases: random shots on random tables (long, multi-collision paths) plus fine angle scans around the aim reference (the
success-window edges, the most sensitive spot). Inputs are generated from a seed and stored in the dump, so compare can
check that both machines really played the same cases.

Usage:
    python -m exp16_wm.sim_parity run --out logs/parity/mac.npz
    python -m exp16_wm.sim_parity compare logs/parity/mac.npz logs/parity/cloud.npz
"""

import argparse
import json
import os
import platform
import sys
import time
import zlib
from importlib.metadata import version
from typing import Any

import numpy as np
import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from exp16_wm.outcomes import pocket_index_map, slots_from_system
from exp16_wm.planner import aim_reference, strike_phi_deg
from exp16_wm.relabel import restore_state
from exp16_wm.sim_identity import to_bitmask
from exp16_wm.teacher_data import CLEAR_ENV
from exp16_wm.two_ball_states import make_states

EVENT_CODES = {e: i for i, e in enumerate(pt.EventType)}
INPUT_KEYS = ("obs", "state_idx", "action", "group")
RESULT_KEYS = ("mask", "scratch", "n_events", "ev_crc", "final_xy", "t_end", "error")
POS_IDENTICAL_M = 1e-9      # final positions within this → bitwise-level identical for our purposes
POS_CLOSE_M = 1e-6          # 1 micrometre; far below the 0.02° window resolution (~0.3 mm at 1 m)
MAX_EVENTS = 2000


def make_cases(n_states: int, n_random: int, n_scan_states: int, scan_half_deg: float, scan_step_deg: float,
               scan_speeds: tuple[float, ...], seed: int) -> dict[str, np.ndarray]:
    """
    obs (S, 23) float32, state_idx (N,), action (N, 2) float32 = (delta_angle rad, speed m/s), group (N,) int32.
    group = -1 for random shots; otherwise the id of one (scan state, speed) angle scan.
    """
    rng = np.random.default_rng(seed)
    obs = np.concatenate([make_states(n_states, seed, "uniform"), make_states(n_scan_states, seed + 1, "env")])
    idx: list[np.ndarray] = []
    act: list[np.ndarray] = []
    grp: list[np.ndarray] = []
    for s in range(n_states):
        delta = rng.uniform(-np.pi, np.pi, n_random)
        speed = np.exp(rng.uniform(np.log(0.5), np.log(8.0), n_random))
        idx.append(np.full(n_random, s))
        act.append(np.stack([delta, speed], axis=1))
        grp.append(np.full(n_random, -1))
    n_angles = int(round(2 * scan_half_deg / scan_step_deg)) + 1
    scan_delta = np.radians(np.linspace(-scan_half_deg, scan_half_deg, n_angles))
    gid = 0
    for s in range(n_scan_states):
        for speed in scan_speeds:
            idx.append(np.full(n_angles, n_states + s))
            act.append(np.stack([scan_delta, np.full(n_angles, speed)], axis=1))
            grp.append(np.full(n_angles, gid))
            gid += 1
    if not idx:
        return {"obs": obs, "state_idx": np.zeros(0, np.int32), "action": np.zeros((0, 2), np.float32), "group": np.zeros(0, np.int32)}
    return {"obs": obs, "state_idx": np.concatenate(idx).astype(np.int32), "action": np.concatenate(act).astype(np.float32),
            "group": np.concatenate(grp).astype(np.int32)}


def simulate_case(env: BilliardsEnv, pocket_map: dict[str, int], obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
    """
    Play one (obs, action) from rest; returns outcome bitmask, scratch, event count/checksum, final xy of every ball, end time.
    A shot that makes pooltool raise (seen once in 18k cases: numba quartic solver, deterministic) is recorded as error=1 with
    zeroed results instead of aborting the run, so compare can check that both machines fail on the same cases.
    """
    restore_state(env, obs)
    sys_c = env.system.copy()
    sys_c.strike(phi=strike_phi_deg(aim_reference(env), action), V0=float(action[1]), cue_ball_id="cue")
    try:
        pt.simulate(sys_c, inplace=True, max_events=MAX_EVENTS)
    except Exception as e:  # noqa: BLE001 — diagnostic tool: any simulator failure is a result to compare
        return {"mask": 0, "scratch": False, "n_events": 0, "ev_crc": 0, "final_xy": np.zeros((4, 2)), "t_end": 0.0, "error": 1,
                "message": f"{type(e).__name__}: {e}"}
    made, scratch = slots_from_system(env, sys_c, pocket_map)
    codes = np.array([EVENT_CODES[e.event_type] for e in sys_c.events], np.int16)
    xy = np.array([sys_c.balls[b].state.rvw[0, :2] for b in ["cue", *env._ball_ids]], np.float64)
    return {"mask": int(to_bitmask(made)), "scratch": bool(scratch), "n_events": len(codes), "ev_crc": zlib.crc32(codes.tobytes()),
            "final_xy": xy, "t_end": float(sys_c.events[-1].time) if len(sys_c.events) else 0.0, "error": 0}


def machine_info() -> dict[str, str]:
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            **{p: version(p) for p in ("pooltool-billiards", "panda3d", "numpy", "numba", "llvmlite")}}


def run(cases: dict[str, np.ndarray], out: str, log_every: int = 2000) -> dict[str, np.ndarray]:
    """Simulate every case and save inputs + results (+ machine info) to out (.npz)."""
    env = BilliardsEnv(**CLEAR_ENV)
    env.reset(seed=0)
    pmap = pocket_index_map(env)
    n = len(cases["action"])
    res = {"mask": np.zeros(n, np.int32), "scratch": np.zeros(n, bool), "n_events": np.zeros(n, np.int32),
           "ev_crc": np.zeros(n, np.uint32), "final_xy": np.zeros((n, 4, 2), np.float64), "t_end": np.zeros(n, np.float64),
           "error": np.zeros(n, np.int8)}
    t0 = time.time()
    for i in range(n):
        r = simulate_case(env, pmap, cases["obs"][cases["state_idx"][i]], cases["action"][i])
        for k in RESULT_KEYS:
            res[k][i] = r[k]
        if r["error"] and res["error"][:i].sum() == 0:
            print(f"  case {i} raised {r['message']} (further errors are only counted)", flush=True)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **cases, **res, meta=json.dumps(machine_info()))
    return res


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray], max_listed: int = 10) -> dict[str, Any]:
    """Compare two dumps. Raises ValueError if they did not play the same cases."""
    for key in INPUT_KEYS:
        if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
            raise ValueError(f"dumps played different cases ({key} differs); regenerate both with the same arguments")
    n = len(a["mask"])
    outcome_eq = (a["mask"] == b["mask"]) & (a["scratch"] == b["scratch"]) & (a["error"] == b["error"])
    seq_eq = (a["n_events"] == b["n_events"]) & (a["ev_crc"] == b["ev_crc"])
    pos = np.abs(a["final_xy"] - b["final_xy"]).max(axis=(1, 2)) if n else np.zeros(0)
    tim = np.abs(a["t_end"] - b["t_end"])
    ok = seq_eq & (a["error"] == 0) & (b["error"] == 0)
    pos_s, tim_s = pos[ok], tim[ok]  # positions are only comparable when both played the same event sequence without error
    scan = a["group"] >= 0
    scan_diff = scan & ((a["mask"] != 0) != (b["mask"] != 0))
    bad = np.where(~outcome_eq | ~seq_eq)[0]
    pos_max = float(pos_s.max()) if len(pos_s) else 0.0
    if len(bad) == 0 and pos_max <= POS_IDENTICAL_M:
        verdict = "IDENTICAL"
    elif len(bad) == 0 and pos_max <= POS_CLOSE_M:
        verdict = "CLOSE"
    else:
        verdict = "DIFFERENT"
    return {
        "verdict": verdict, "n_cases": n, "n_scan_cases": int(scan.sum()),
        "n_outcome_differs": int((~outcome_eq).sum()), "n_event_sequence_differs": int((~seq_eq).sum()),
        "n_errors": [int(a["error"].sum()), int(b["error"].sum())],
        "n_scan_made_flips": int(scan_diff.sum()), "n_scan_groups_with_flip": int(len(np.unique(a["group"][scan_diff]))),
        "final_pos_diff_m": {"max": pos_max, "median": float(np.median(pos_s)) if len(pos_s) else 0.0,
                             "p99": float(np.percentile(pos_s, 99)) if len(pos_s) else 0.0},
        "t_end_diff_max_s": float(tim_s.max()) if len(tim_s) else 0.0,
        "first_mismatches": [{"case": int(i), "group": int(a["group"][i]), "delta_deg": float(np.degrees(a["action"][i, 0])),
                              "speed": float(a["action"][i, 1]), "mask": [int(a["mask"][i]), int(b["mask"][i])],
                              "n_events": [int(a["n_events"][i]), int(b["n_events"][i])]} for i in bad[:max_listed]],
    }


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="simulate the fixed cases on this machine")
    r.add_argument("--out", required=True)
    r.add_argument("--n-states", type=int, default=300)
    r.add_argument("--n-random", type=int, default=20, help="random actions per random state")
    r.add_argument("--n-scan-states", type=int, default=20)
    r.add_argument("--scan-half-deg", type=float, default=3.0)
    r.add_argument("--scan-step-deg", type=float, default=0.02)
    r.add_argument("--scan-speeds", type=float, nargs="+", default=[1.0, 3.0])
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare", help="compare two dumps")
    c.add_argument("a")
    c.add_argument("b")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "run":
        cases = make_cases(args.n_states, args.n_random, args.n_scan_states, args.scan_half_deg, args.scan_step_deg,
                           tuple(args.scan_speeds), args.seed)
        print(f"{len(cases['action'])} cases on {machine_info()}", flush=True)
        t0 = time.time()
        run(cases, args.out)
        print(f"done in {time.time() - t0:.0f}s → {args.out}", flush=True)
        return 0
    a, b = load(args.a), load(args.b)
    print("A:", str(a["meta"]))
    print("B:", str(b["meta"]))
    report = compare(a, b)
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] != "DIFFERENT" else 1


if __name__ == "__main__":
    sys.exit(main())
