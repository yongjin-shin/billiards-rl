"""
world_model/gnn_resolver.py

GNN collision resolver: (pre_state, geometry) → post_state

모든 충돌 타입을 단일 모델로 처리.
contact_normal이 충돌 기하를 인코딩하므로 타입 분류 불필요.

Graph:
    Node 0 (cue ball): pos(2) + pre_vel(2) + pre_avel(3) + normal(2) = 9 dim
    Node 1 (tgt ball): pos(2) + pre_vel(2) + pre_avel(3) + normal(2) = 9 dim  [ball_ball만 active]

    Edge 0→1, 1→0: relative_pos(2) + relative_vel(2) = 4 dim

    ball_ball:   양방향 메시지 패싱, 두 노드 모두 업데이트
    others:      cue 노드만 업데이트 (tgt는 zero-mask)

Output:
    delta_vel  (2,2): post_vel  - pre_vel
    delta_avel (2,3): post_avel - pre_avel
"""

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


NODE_DIM = 9   # pos(2) + pre_vel(2) + pre_avel(3) + normal(2)
EDGE_DIM = 4   # rel_pos(2) + rel_vel(2)


class GNNResolver(nn.Module):
    """
    Args:
        hidden: MLP hidden dims (message net, update net, output net 공유)
        node_dim: node embedding dim
    """
    def __init__(self, hidden: list[int] = [128, 128], node_dim: int = 64):
        super().__init__()
        self.node_enc = _mlp(NODE_DIM, hidden, node_dim)
        self.msg_net  = _mlp(node_dim * 2 + EDGE_DIM, hidden, node_dim)
        self.upd_net  = _mlp(node_dim * 2, hidden, node_dim)
        # delta vel (2) + delta avel (3) per ball
        self.out_net  = _mlp(node_dim, hidden, 5)

    def forward(
        self,
        pos      : torch.Tensor,   # (B, 2, 2)  normalized
        pre_vel  : torch.Tensor,   # (B, 2, 2)  normalized
        pre_avel : torch.Tensor,   # (B, 2, 3)  normalized
        normal   : torch.Tensor,   # (B, 2)     contact normal (unit vec)
        has_tgt  : torch.Tensor,   # (B,)       bool — ball_ball이면 True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_vel  (B, 2, 2)
            delta_avel (B, 2, 3)
        """
        B = pos.shape[0]

        # normal을 두 노드 모두에 broadcast
        normal_exp = normal.unsqueeze(1).expand(-1, 2, -1)  # (B,2,2)

        # node features
        node_feat = torch.cat([pos, pre_vel, pre_avel, normal_exp], dim=-1)  # (B,2,9)
        h = self.node_enc(node_feat)                                          # (B,2,D)

        # edge features (0→1, 1→0)
        rel_pos = pos[:, 1] - pos[:, 0]   # (B,2)
        rel_vel = pre_vel[:, 1] - pre_vel[:, 0]  # (B,2)
        edge_01 = torch.cat([rel_pos,  rel_vel], dim=-1)   # (B,4)
        edge_10 = torch.cat([-rel_pos, -rel_vel], dim=-1)  # (B,4)

        # messages
        msg_01 = self.msg_net(torch.cat([h[:, 0], h[:, 1], edge_01], dim=-1))  # (B,D)
        msg_10 = self.msg_net(torch.cat([h[:, 1], h[:, 0], edge_10], dim=-1))  # (B,D)

        # has_tgt mask: ball_ball이 아니면 tgt 메시지 0
        tgt_mask = has_tgt.float().unsqueeze(-1)  # (B,1)
        msg_01 = msg_01 * tgt_mask  # tgt→cue 메시지: ball_ball에만 의미 있음
        msg_10 = msg_10 * tgt_mask  # cue→tgt 메시지

        # node update
        h0_new = self.upd_net(torch.cat([h[:, 0], msg_01], dim=-1))  # (B,D)
        h1_new = self.upd_net(torch.cat([h[:, 1], msg_10], dim=-1))  # (B,D)

        h_new = torch.stack([h0_new, h1_new], dim=1)  # (B,2,D)

        # output
        out = self.out_net(h_new)  # (B,2,5)
        delta_vel  = out[:, :, :2]  # (B,2,2)
        delta_avel = out[:, :, 2:]  # (B,2,3)

        return delta_vel, delta_avel
