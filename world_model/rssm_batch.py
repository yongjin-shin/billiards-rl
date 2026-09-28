"""
world_model/rssm_batch.py

Wavefront scheduler for batched R-SSM training across multiple shots.

Different shots have independent, unrelated event sequences (no shared state
between shots), so there is no correctness requirement that shot A's k-th
event be processed "with" shot B's k-th event — only that, within a single
shot, event k+1 is processed strictly after event k (h is sequential per shot).

A "wavefront" is one iteration: the set of (shot_idx, event_idx) pairs for
all shots that still have events remaining, each pointing at that shot's
next unconsumed event. As shots finish, they drop out of later wavefronts.
"""

from __future__ import annotations

from typing import Iterator

from world_model.rssm_dataset import ShotData
from world_model.rssm_model import EVENT_BALL_BALL


WavefrontItem = tuple[int, int]   # (shot_idx, event_idx)


def iter_wavefronts(shots: list[ShotData]) -> Iterator[list[WavefrontItem]]:
    """
    Yield one wavefront per call: the list of (shot_idx, event_idx) for every
    shot that still has events remaining, pointing at its next event.

    Shots with fewer events drop out of the active set over time; the
    wavefront list shrinks monotonically until empty (loop ends).
    """
    n_events = [len(s.event_steps) for s in shots]
    ptrs     = [0] * len(shots)

    while True:
        active = [(s, ptrs[s]) for s in range(len(shots)) if ptrs[s] < n_events[s]]
        if not active:
            return
        yield active
        for s, _ in active:
            ptrs[s] += 1


def split_wavefront_by_type(
    shots  : list[ShotData],
    active : list[WavefrontItem],
) -> tuple[list[WavefrontItem], list[WavefrontItem]]:
    """Split a wavefront into (ball_ball_items, single_items) by event_type."""
    ball_ball: list[WavefrontItem] = []
    single   : list[WavefrontItem] = []
    for s, k in active:
        ev = shots[s].event_steps[k]
        if ev.event_type == EVENT_BALL_BALL:
            ball_ball.append((s, k))
        else:
            single.append((s, k))
    return ball_ball, single
