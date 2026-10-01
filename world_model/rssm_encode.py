"""
world_model/rssm_encode.py

Encode an already-simulated pooltool shot into R-SSM per-ball latents, using a
frozen pretrained checkpoint. Used by exp16_wm's WMSAC to supervise its blind
world-model predictor with a physics-grounded target instead of a flat
(x, y, event_type) trajectory encoding.

Reuses, rather than duplicates:
  generate_shot_data()      world_model/rssm_dataset.py  — pooltool System → ShotData
  make_node() / make_edge() world_model/rssm_rollout.py   — raw rvw → GNN node/edge features
  RSSMModel.forward()       world_model/rssm_model.py     — teacher-forced shot rollout
"""

from __future__ import annotations

import torch

from world_model.rssm_dataset import generate_shot_data
from world_model.rssm_model import EventStep, RSSMModel
from world_model.rssm_rollout import make_edge, make_node


def load_frozen_rssm(checkpoint_path: str, device: str = "cpu") -> RSSMModel:
    """Load a pretrained R-SSM checkpoint in eval mode with gradients disabled."""
    model = RSSMModel()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_shot_to_latent(
    model: RSSMModel,
    system: "pooltool.System",
    ball_ids: list[str],
) -> list[torch.Tensor]:
    """
    Run a frozen R-SSM, teacher-forced, over the real events of an already-
    simulated shot. Returns the final per-ball latent (h_final).

    Shots with no tracked events (e.g. a complete miss) return the
    zero-initialised latent unchanged (model.init_hidden).
    """
    shot = generate_shot_data(system, ball_ids)

    events: list[EventStep] = []
    for k, ev in enumerate(shot.event_steps):
        rvw_i = shot.raw_rvws_i[k]
        node_i = make_node(rvw_i, ev.event_type)

        node_j = edge = None
        if ev.ball_j is not None:
            rvw_j = shot.raw_rvws_j[k]
            node_j = make_node(rvw_j, ev.event_type)
            edge = make_edge(rvw_i, rvw_j, ev.normal.numpy())

        events.append(EventStep(
            event_type = ev.event_type,
            ball_i     = ev.ball_i,
            ball_j     = ev.ball_j,
            node_i     = node_i,
            node_j     = node_j,
            edge       = edge,
            normal     = ev.normal,
        ))

    out = model.forward(len(ball_ids), events)
    return out.h_final
