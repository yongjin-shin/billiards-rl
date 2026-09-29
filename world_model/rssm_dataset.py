"""
world_model/rssm_dataset.py

Data pipeline for R-SSM training.

ShotData : one training sample — event sequence + GT labels
generate_shot_data : parse a simulated pooltool System → ShotData
collect_dataset    : run n_shots simulations → list[ShotData]
RSSMDataset        : PyTorch Dataset wrapping list[ShotData]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from world_model.ball_motion import FrictionParams, DEFAULT_FRICTION
from world_model.event_detector import (
    COLL_TYPE, TGT_LINEAR, TGT_CIRCULAR,
    _contact_normal_ball_ball,
    _contact_normal_lcushion,
    _contact_normal_ccushion,
    _contact_normal_pocket,
)
from world_model.rssm_model import EventStep, N_TYPE


SKIP_EVENTS = frozenset({"stick_ball", "none"})


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ShotData:
    """One training sample: a single shot's event sequence with GT labels."""
    n_balls     : int
    event_steps : list[EventStep]
    gt_deltas_i : list[torch.Tensor]           # (5,) Δvel+Δavel for ball_i per event
    gt_deltas_j : list[Optional[torch.Tensor]] # (5,) for ball_ball; None for single
    gt_types_i  : list[Optional[int]]          # next event type for ball_i; None=terminal
    gt_types_j  : list[Optional[int]]          # same for ball_j
    raw_rvws_i  : list[np.ndarray]             # (3,3) pre-collision rvw for ball_i
    raw_rvws_j  : list[Optional[np.ndarray]]   # (3,3) or None for single events
    dt_to_next  : list[float]                  # seconds to next event; 0.0 for last
    will_pocket : dict = None                  # {ball_idx: bool} — pocketed in this shot?

    def __post_init__(self):
        if self.will_pocket is None:
            self._derive_will_pocket()

    def __setstate__(self, state):
        # Called by pickle on load — __post_init__ is not re-run
        self.__dict__.update(state)
        if self.will_pocket is None:
            self._derive_will_pocket()

    def _derive_will_pocket(self):
        self.will_pocket = {i: False for i in range(self.n_balls)}
        for ev in self.event_steps:
            if ev.event_type == 3:   # EVENT_POCKET
                self.will_pocket[ev.ball_i] = True

    def n_pocketed_targets(self) -> int:
        """Number of pocketed target balls (excludes ball_idx=0, the cue/scratch)."""
        return sum(1 for i in range(1, self.n_balls) if self.will_pocket.get(i, False))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_raw_type_and_normal(ev, table) -> tuple[int, np.ndarray]:
    """Return (raw_type, contact_normal) for a physical event."""
    et = str(ev.event_type)
    raw_type = COLL_TYPE.get(et, -1)
    if raw_type == -1:
        return -1, np.zeros(2, dtype=np.float32)

    ball_agents  = [a for a in ev.agents if getattr(a, "agent_type", "") == "ball"]
    other_agents = [(getattr(a, "agent_type", ""), a.id)
                    for a in ev.agents if getattr(a, "agent_type", "") != "ball"]
    has_cue = any(a.id == "cue" for a in ball_agents)
    normal  = np.array([1.0, 0.0], dtype=np.float32)

    if et == "ball_ball":
        cue_a = next((a for a in ball_agents if a.id == "cue"), None)
        obj_a = next((a for a in ball_agents if a.id != "cue"), None)
        if cue_a is not None and obj_a is not None:
            normal = _contact_normal_ball_ball(
                cue_a.initial.state.rvw[0],
                obj_a.initial.state.rvw[0],
            )

    elif et == "ball_linear_cushion":
        for atype, aid in other_agents:
            if atype == "linear_cushion_segment":
                cush   = table.cushion_segments.linear[aid]
                normal = _contact_normal_lcushion(cush.lx, cush.ly)
                break
        if not has_cue:
            raw_type = TGT_LINEAR

    elif et == "ball_circular_cushion":
        main_a = ball_agents[0] if ball_agents else None
        for atype, aid in other_agents:
            if atype == "circular_cushion_segment" and main_a is not None:
                cush   = table.cushion_segments.circular[aid]
                normal = _contact_normal_ccushion(
                    cush.a, cush.b, main_a.initial.state.rvw[0]
                )
                break
        if not has_cue:
            raw_type = TGT_CIRCULAR

    elif et == "ball_pocket":
        main_a = ball_agents[0] if ball_agents else None
        for atype, aid in other_agents:
            if atype == "pocket" and main_a is not None:
                pock   = table.pockets[aid]
                normal = _contact_normal_pocket(
                    pock.a, pock.b, main_a.initial.state.rvw[0]
                )
                break

    return raw_type, normal


def _extract_delta(ball_agent) -> torch.Tensor:
    """Compute Δvel(2)+Δavel(3) from ball agent's initial→final states."""
    pre_rvw = ball_agent.initial.state.rvw   # (3, 3)

    try:
        if hasattr(ball_agent.final, "state"):
            post_vel  = ball_agent.final.state.rvw[1].copy()
            post_avel = ball_agent.final.state.rvw[2].copy()
        else:
            post_vel  = np.array(list(ball_agent.final.vel),  dtype=np.float64)
            post_avel = np.array(list(ball_agent.final.avel), dtype=np.float64)
    except (AttributeError, TypeError):
        post_vel  = pre_rvw[1].copy()
        post_avel = pre_rvw[2].copy()

    delta_vel  = post_vel[:2]  - pre_rvw[1, :2]
    delta_avel = post_avel[:3] - pre_rvw[2, :3]
    return torch.tensor(
        np.concatenate([delta_vel, delta_avel]), dtype=torch.float32
    )


def _find_next_type(
    physical_events : list,
    event_idx       : int,
    ball_id         : str,
    table,
) -> Optional[int]:
    """Return the type of the next event in sequence that involves ball_id."""
    for j in range(event_idx + 1, len(physical_events)):
        ev_j     = physical_events[j]
        ids_j    = {a.id for a in ev_j.agents if getattr(a, "agent_type", "") == "ball"}
        if ball_id in ids_j:
            raw_type, _ = _get_raw_type_and_normal(ev_j, table)
            if raw_type != -1:
                return raw_type
    return None


# ── Core generation ───────────────────────────────────────────────────────────

def generate_shot_data(
    system   : "pooltool.System",
    ball_ids : list[str],
    params   : FrictionParams = DEFAULT_FRICTION,
) -> ShotData:
    """
    Parse a simulated pooltool System into a training ShotData.

    Parameters
    ----------
    system   : already-simulated pooltool System (BilliardsEnv.system)
    ball_ids : ordered list of tracked ball IDs, e.g. ["cue", "1", "2"]

    Returns
    -------
    ShotData  (event_steps may be empty for degenerate shots)
    """
    id_to_idx = {bid: i for i, bid in enumerate(ball_ids)}
    table     = system.table

    # Filter to physical collision events
    physical_events = [
        ev for ev in system.events
        if str(ev.event_type) not in SKIP_EVENTS
        and COLL_TYPE.get(str(ev.event_type), -1) != -1
    ]

    # ④ pre-scan: compute (raw_type, normal, ball_id_set) once per event.
    # Eliminates O(N²) _get_raw_type_and_normal calls from _find_next_type.
    _pre_types   : list[int]          = []
    _pre_normals : list               = []
    _pre_bid_sets: list[frozenset]    = []
    for _ev in physical_events:
        _rt, _nm = _get_raw_type_and_normal(_ev, table)
        _pre_types.append(_rt)
        _pre_normals.append(_nm)
        _pre_bid_sets.append(frozenset(
            a.id for a in _ev.agents if getattr(a, "agent_type", "") == "ball"
        ))

    def _next_type(ev_idx: int, ball_id: str) -> Optional[int]:
        for j in range(ev_idx + 1, len(physical_events)):
            if ball_id in _pre_bid_sets[j] and _pre_types[j] != -1:
                return _pre_types[j]
        return None

    event_steps : list[EventStep]         = []
    gt_deltas_i : list[torch.Tensor]      = []
    gt_deltas_j : list                    = []
    gt_types_i  : list                    = []
    gt_types_j  : list                    = []
    raw_rvws_i  : list                    = []
    raw_rvws_j  : list                    = []
    dt_to_next  : list[float]             = []
    event_times : list[float]             = []   # time of each tracked event

    for ev_idx, ev in enumerate(physical_events):
        raw_type = _pre_types[ev_idx]
        normal   = _pre_normals[ev_idx]
        if raw_type == -1:
            continue

        ball_agents = [a for a in ev.agents if getattr(a, "agent_type", "") == "ball"]
        tracked     = [a for a in ball_agents if a.id in id_to_idx]
        if not tracked:
            continue

        normal_t = torch.tensor(normal, dtype=torch.float32)

        if raw_type == 0 and len(tracked) >= 2:    # EVENT_BALL_BALL
            tracked_sorted = sorted(tracked, key=lambda a: id_to_idx[a.id])
            a_i, a_j       = tracked_sorted[0], tracked_sorted[1]
            rvw_i, rvw_j   = a_i.initial.state.rvw, a_j.initial.state.rvw

            # ⑤ node_i/node_j/edge stored as None — recomputed at training time
            event_steps.append(EventStep(
                event_type = raw_type,
                ball_i     = id_to_idx[a_i.id],
                ball_j     = id_to_idx[a_j.id],
                node_i     = None,
                node_j     = None,
                edge       = None,
                normal     = normal_t,
            ))
            gt_deltas_i.append(_extract_delta(a_i))
            gt_deltas_j.append(_extract_delta(a_j))
            gt_types_i.append(_next_type(ev_idx, a_i.id))
            gt_types_j.append(_next_type(ev_idx, a_j.id))
            raw_rvws_i.append(rvw_i.copy())
            raw_rvws_j.append(rvw_j.copy())
            event_times.append(float(ev.time))

        else:    # single-ball event
            a_i   = tracked[0]
            rvw_i = a_i.initial.state.rvw

            # ⑤ node_i stored as None — recomputed at training time
            event_steps.append(EventStep(
                event_type = raw_type,
                ball_i     = id_to_idx[a_i.id],
                ball_j     = None,
                node_i     = None,
                node_j     = None,
                edge       = None,
                normal     = normal_t,
            ))
            gt_deltas_i.append(_extract_delta(a_i))
            gt_deltas_j.append(None)
            gt_types_i.append(_next_type(ev_idx, a_i.id))
            gt_types_j.append(None)
            raw_rvws_i.append(rvw_i.copy())
            raw_rvws_j.append(None)
            event_times.append(float(ev.time))

    # Compute dt_to_next as time between consecutive tracked events.
    # Using event_times (not physical_events indices) ensures we skip
    # intermediate untracked events that would otherwise produce wrong dts.
    for k in range(len(event_times)):
        if k + 1 < len(event_times):
            dt_to_next.append(event_times[k + 1] - event_times[k])
        else:
            dt_to_next.append(0.0)

    return ShotData(
        n_balls     = len(ball_ids),
        event_steps = event_steps,
        gt_deltas_i = gt_deltas_i,
        gt_deltas_j = gt_deltas_j,
        gt_types_i  = gt_types_i,
        gt_types_j  = gt_types_j,
        raw_rvws_i  = raw_rvws_i,
        raw_rvws_j  = raw_rvws_j,
        dt_to_next  = dt_to_next,
    )


def collect_dataset(
    n_shots    : int,
    n_balls    : int = 1,
    seed_start : int = 0,
) -> list[ShotData]:
    """
    Simulate n_shots and collect ShotData.

    Parameters
    ----------
    n_shots    : number of shots to generate
    n_balls    : number of object balls (BilliardsEnv parameter)
    seed_start : RNG seed for first shot
    """
    from simulator import BilliardsEnv

    ball_ids = ["cue"] + [str(i) for i in range(1, n_balls + 1)]
    shots: list[ShotData] = []

    env = BilliardsEnv(n_balls=n_balls)
    try:
        for i in range(n_shots):
            env.reset(seed=seed_start + i)
            env.step(env.action_space.sample())
            shot = generate_shot_data(env.system, ball_ids)
            if shot.event_steps:
                shots.append(shot)
    finally:
        env.close()

    return shots


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(
    data_dir   : str,
    max_shots  : Optional[int] = None,
    pocket_only: bool = False,
) -> list[ShotData]:
    """
    Load ShotData from pickle chunks written by generate_rssm_data.py.

    Parameters
    ----------
    data_dir    : directory containing *_chunk*.pkl files
    max_shots   : cap on total shots loaded (None = all)
    pocket_only : if True, keep only shots with at least one pocket event (type 3)
    """
    import pickle
    from pathlib import Path

    chunk_files = sorted(Path(data_dir).glob("*_chunk*.pkl"))
    assert chunk_files, f"No chunk pkl files found in {data_dir}"

    shots: list[ShotData] = []
    for path in chunk_files:
        with open(path, "rb") as f:
            chunk: list[ShotData] = pickle.load(f)
        if pocket_only:
            chunk = [s for s in chunk
                     if any(e.event_type == 3 for e in s.event_steps)]
        shots.extend(chunk)
        if max_shots is not None and len(shots) >= max_shots:
            shots = shots[:max_shots]
            break

    return shots


class RSSMDataset(Dataset):
    """
    PyTorch Dataset over a list of ShotData.

    Use DataLoader with batch_size=1 since event counts vary per shot.
    """

    def __init__(self, shots: list[ShotData]):
        self.shots = shots

    def __len__(self) -> int:
        return len(self.shots)

    def __getitem__(self, idx: int) -> ShotData:
        return self.shots[idx]
