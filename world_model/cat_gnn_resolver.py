"""
world_model/cat_gnn_resolver.py

GNN collision resolver — continuous latent (categorical 제거).

z = pool(h_cue ‖ h_tgt)  (128-dim continuous)
  → dec_cue/dec_tgt: Δvel, Δavel
  → pocket_head: P(target pocketed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_DIM = 9   # pos(2) + pre_vel(2) + pre_avel(3) + normal(2)
EDGE_DIM = 4   # rel_pos(2) + rel_vel(2)


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class CatGNNResolver(nn.Module):
    """
    Parameters
    ----------
    hidden   : MLP hidden dims
    node_dim : node embedding dim (D)
    z_dim    : continuous latent dim (pool projection)
    """

    def __init__(
        self,
        hidden  : list[int] = [128, 128],
        node_dim: int       = 64,
        z_dim   : int       = 64,
    ):
        super().__init__()
        self.z_dim = z_dim

        # ── GNN encoder ────────────────────────────────────────────────
        self.node_enc = _mlp(NODE_DIM,                 hidden, node_dim)
        self.msg_net  = _mlp(node_dim * 2 + EDGE_DIM,  hidden, node_dim)
        self.upd_net  = _mlp(node_dim * 2,             hidden, node_dim)

        # ── Continuous latent projection ────────────────────────────────
        # pool(h_cue ‖ h_tgt) → z
        self.pool_proj = _mlp(node_dim * 2, hidden, z_dim)

        # ── Per-ball decoders ───────────────────────────────────────────
        # [z ‖ h_ball] → Δvel(2) + Δavel(3)
        self.dec_cue = _mlp(z_dim + node_dim, hidden, 5)
        self.dec_tgt = _mlp(z_dim + node_dim, hidden, 5)

        # ── Pocket prediction head ──────────────────────────────────────
        # [z ‖ h_cue ‖ h_tgt] → logit
        self.pocket_head = _mlp(z_dim + node_dim * 2, hidden, 1)

    # ── GNN message passing ────────────────────────────────────────────

    def _encode(
        self,
        pos      : torch.Tensor,   # (B, 2, 2)
        pre_vel  : torch.Tensor,   # (B, 2, 2)
        pre_avel : torch.Tensor,   # (B, 2, 3)
        normal   : torch.Tensor,   # (B, 2)
        has_tgt  : torch.Tensor,   # (B,)
    ) -> torch.Tensor:             # (B, 2, node_dim)
        normal_exp = normal.unsqueeze(1).expand(-1, 2, -1)
        node_feat  = torch.cat([pos, pre_vel, pre_avel, normal_exp], dim=-1)  # (B,2,9)
        h = self.node_enc(node_feat)                                           # (B,2,D)

        rel_pos = pos[:, 1]     - pos[:, 0]
        rel_vel = pre_vel[:, 1] - pre_vel[:, 0]
        e01 = torch.cat([ rel_pos,  rel_vel], dim=-1)
        e10 = torch.cat([-rel_pos, -rel_vel], dim=-1)

        m01 = self.msg_net(torch.cat([h[:, 0], h[:, 1], e01], dim=-1))
        m10 = self.msg_net(torch.cat([h[:, 1], h[:, 0], e10], dim=-1))

        tgt_m = has_tgt.float().unsqueeze(-1)
        h0 = self.upd_net(torch.cat([h[:, 0], m01 * tgt_m], dim=-1))
        h1 = self.upd_net(torch.cat([h[:, 1], m10 * tgt_m], dim=-1))
        return torch.stack([h0, h1], dim=1)                                    # (B,2,D)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        pos      : torch.Tensor,
        pre_vel  : torch.Tensor,
        pre_avel : torch.Tensor,
        normal   : torch.Tensor,
        has_tgt  : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        delta_vel    (B, 2, 2)
        delta_avel   (B, 2, 3)
        pocket_logit (B,)
        """
        h    = self._encode(pos, pre_vel, pre_avel, normal, has_tgt)
        pool = torch.cat([h[:, 0], h[:, 1]], dim=-1)   # (B, 2D)
        z    = self.pool_proj(pool)                     # (B, z_dim)

        o_cue = self.dec_cue(torch.cat([z, h[:, 0]], dim=-1))   # (B, 5)
        o_tgt = self.dec_tgt(torch.cat([z, h[:, 1]], dim=-1))   # (B, 5)
        out   = torch.stack([o_cue, o_tgt], dim=1)               # (B, 2, 5)

        pocket_logit = self.pocket_head(
            torch.cat([z, h[:, 0], h[:, 1]], dim=-1)
        ).squeeze(-1)                                             # (B,)

        return out[:, :, :2], out[:, :, 2:], pocket_logit
