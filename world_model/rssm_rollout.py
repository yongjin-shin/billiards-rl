"""
world_model/rssm_rollout.py

RolloutEngine — bridges EventDetector (physics) and RSSMModel (learned dynamics).

Flow per event:
  1. EventDetector.next_collision()  → EventResult (type, time, normal)
  2. advance_balls()                 → balls at collision time
  3. build EventStep                 → node/edge features
  4. RSSMModel.step_*()              → predicted Δvel/Δavel, next_type
  5. _apply_delta()                  → update ball velocities
  6. repeat until no collision or max_events
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from world_model.ball_motion import FrictionParams, DEFAULT_FRICTION
from world_model.pure_physics import evolve_ball_motion, SLIDING
from world_model.event_detector import EventDetector, EventResult
from world_model.rssm_model import (
    RSSMModel, EventStep, EventOutput,
    EVENT_BALL_BALL, EVENT_POCKET, N_TYPE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rvw_to_phys_feat(rvw: np.ndarray) -> torch.Tensor:
    """(3, 3) rvw → (7,) float32: pos_xy(2) + vel_xy(2) + avel(3)"""
    pos  = rvw[0, :2]
    vel  = rvw[1, :2]
    avel = rvw[2]
    return torch.tensor(np.concatenate([pos, vel, avel]), dtype=torch.float32)


def make_node(rvw: np.ndarray, event_type: int) -> torch.Tensor:
    """(14,) = phys_feat(7) + type_onehot(7)"""
    phys = rvw_to_phys_feat(rvw)
    oh   = torch.zeros(N_TYPE)
    oh[event_type] = 1.0
    return torch.cat([phys, oh])


def make_edge(
    rvw_i  : np.ndarray,
    rvw_j  : np.ndarray,
    normal : np.ndarray,
) -> torch.Tensor:
    """(9,) = rel_pos(2) + rel_vel(2) + rel_avel(3) + normal(2)  — i→j direction"""
    rel_pos  = rvw_j[0, :2] - rvw_i[0, :2]
    rel_vel  = rvw_j[1, :2] - rvw_i[1, :2]
    rel_avel = rvw_j[2]     - rvw_i[2]
    return torch.tensor(
        np.concatenate([rel_pos, rel_vel, rel_avel, normal]),
        dtype=torch.float32,
    )


def advance_balls(
    balls  : dict[str, tuple[np.ndarray, int]],
    dt     : float,
    params : FrictionParams,
) -> dict[str, tuple[np.ndarray, int]]:
    """Advance all balls by dt using free-motion physics (no collision)."""
    updated = {}
    for bid, (rvw, s) in balls.items():
        rvw_new, s_new = evolve_ball_motion(
            s, np.array(rvw, dtype=np.float64),
            R=params.R, m=params.m,
            u_s=params.u_s, u_sp=params.u_sp,
            u_r=params.u_r, g=params.g,
            t=dt,
        )
        updated[bid] = (rvw_new, s_new)
    return updated


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    event_steps   : list[EventStep]
    event_outputs : list[EventOutput]
    Q             : torch.Tensor      # scalar
    h_final       : list              # list of (H_DIM,) tensors, one per ball


# ── RolloutEngine ─────────────────────────────────────────────────────────────

class RolloutEngine:
    """
    Parameters
    ----------
    model      : RSSMModel
    detector   : EventDetector
    ball_ids   : ordered list of ball IDs — defines index mapping
                 e.g. ["cue", "1", "2", "3"]
    params     : FrictionParams for ball advancement
    max_events : safety cap on rollout length
    """

    def __init__(
        self,
        model      : RSSMModel,
        detector   : EventDetector,
        ball_ids   : list[str],
        params     : FrictionParams = DEFAULT_FRICTION,
        max_events : int = 30,
    ):
        self.model      = model
        self.detector   = detector
        self.ball_ids   = ball_ids
        self.id_to_idx  = {bid: i for i, bid in enumerate(ball_ids)}
        self.n_balls    = len(ball_ids)
        self.params     = params
        self.max_events = max_events

    # ── Internal ──────────────────────────────────────────────────────────────

    def _involved_ball_ids(self, result: EventResult) -> list[str]:
        """Ball IDs from the event that are in our tracked set."""
        raw_ids = [
            a.id for a in result.raw_event.agents
            if getattr(a, "agent_type", "") == "ball"
        ]
        return [bid for bid in raw_ids if bid in self.id_to_idx]

    def _build_event_step(
        self,
        result       : EventResult,
        balls        : dict[str, tuple[np.ndarray, int]],
        involved_ids : list[str],
    ) -> EventStep:
        raw_type = result.event_type
        normal   = torch.tensor(result.contact_normal, dtype=torch.float32)

        if raw_type == EVENT_BALL_BALL and len(involved_ids) >= 2:
            # Order by index for consistent edge direction
            id_i, id_j = sorted(involved_ids, key=lambda b: self.id_to_idx[b])
            rvw_i, _ = balls[id_i]
            rvw_j, _ = balls[id_j]
            return EventStep(
                event_type = raw_type,
                ball_i     = self.id_to_idx[id_i],
                ball_j     = self.id_to_idx[id_j],
                node_i     = make_node(rvw_i, raw_type),
                node_j     = make_node(rvw_j, raw_type),
                edge       = make_edge(rvw_i, rvw_j, result.contact_normal),
                normal     = normal,
            )
        else:
            id_i  = involved_ids[0]
            rvw_i, _ = balls[id_i]
            return EventStep(
                event_type = raw_type,
                ball_i     = self.id_to_idx[id_i],
                ball_j     = None,
                node_i     = make_node(rvw_i, raw_type),
                node_j     = None,
                edge       = None,
                normal     = normal,
            )

    def _apply_delta(
        self,
        balls     : dict[str, tuple[np.ndarray, int]],
        ev_step   : EventStep,
        ev_output : EventOutput,
    ) -> dict[str, tuple[np.ndarray, int]]:
        """Apply predicted Δvel/Δavel; set state to sliding."""
        updated = dict(balls)

        def _apply(bid: str, delta: torch.Tensor) -> None:
            rvw, _ = updated[bid]
            rvw = rvw.copy()
            rvw[1, :2] += delta[:2].detach().numpy()
            rvw[2]     += delta[2:].detach().numpy()
            updated[bid] = (rvw, SLIDING)

        _apply(self.ball_ids[ev_step.ball_i], ev_output.delta_i)

        if ev_step.ball_j is not None and ev_output.delta_j is not None:
            _apply(self.ball_ids[ev_step.ball_j], ev_output.delta_j)

        return updated

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(
        self,
        balls_dict : dict[str, tuple[np.ndarray, int]],
        device     : torch.device | None = None,
    ) -> RolloutResult:
        """
        Run a full shot rollout.

        Parameters
        ----------
        balls_dict : {ball_id: (rvw (3,3), pooltool_state)}

        Returns
        -------
        RolloutResult
        """
        h      = self.model.init_hidden(self.n_balls, device)
        active = dict(balls_dict)
        event_steps:   list[EventStep]   = []
        event_outputs: list[EventOutput] = []

        for _ in range(self.max_events):
            if not active:
                break

            result = self.detector.next_collision(active)
            if result.event_type == -1:
                break

            active       = advance_balls(active, result.time, self.params)
            involved_ids = self._involved_ball_ids(result)

            if not involved_ids:
                break

            ev_step = self._build_event_step(result, active, involved_ids)

            if result.event_type == EVENT_BALL_BALL and ev_step.ball_j is not None:
                h, d_i, d_j, t_i, t_j = self.model.step_ball_ball(
                    h,
                    ev_step.ball_i, ev_step.ball_j,
                    ev_step.node_i, ev_step.node_j, ev_step.edge,
                )
                ev_out = EventOutput(d_i, d_j, t_i, t_j)
            else:
                h, d_i, t_i = self.model.step_single(
                    h, ev_step.ball_i, ev_step.node_i, ev_step.normal,
                )
                ev_out = EventOutput(d_i, None, t_i, None)

            active = self._apply_delta(active, ev_step, ev_out)

            if result.event_type == EVENT_POCKET:
                active.pop(involved_ids[0], None)

            event_steps.append(ev_step)
            event_outputs.append(ev_out)

        Q = self.model.aggregate_q(h)
        return RolloutResult(
            event_steps=event_steps,
            event_outputs=event_outputs,
            Q=Q,
            h_final=h,
        )
