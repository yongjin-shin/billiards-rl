"""
world_model/markov_predictor.py — Markov Transition World Model

당구 물리는 (pos, vel, avel) 완전 관측 시 순수 Markov — LSTM 불필요.

Architecture:
    MarkovEncoder   : (obs_norm 19) → MLP → first event prediction (24 dims)
    MarkovTransition: (event_state 24) → MLP → next event prediction (24 dims)

Event state (24 dims):
    [0:2]   cue_xy     normalized [0,1]
    [2:4]   cue_vel    normalized /MAX_SPEED
    [4:7]   cue_avel   normalized /MAX_AVEL
    [7:9]   tgt_xy     normalized [0,1]
    [9:11]  tgt_vel    normalized /MAX_SPEED
    [11:14] tgt_avel   normalized /MAX_AVEL
    [14:24] type one-hot (10)

MarkovTransition output:
    type_logits (10) — CrossEntropy target
    Δcue_xy    (2)   — 위치 delta
    cue_vel    (2)   — 절대값 예측
    cue_avel   (3)   — 절대값 예측
    Δtgt_xy    (2)   — 위치 delta
    tgt_vel    (2)   — 절대값 예측
    tgt_avel   (3)   — 절대값 예측

Training:
    Encoder   : GT = events[:, 0, :]  (첫 이벤트 전체)
    Transition: GT pairs (events[:, t, :], events[:, t+1, :])  — 순수 Markov supervised
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.generate_data_v3 import EVENT_DIM_V3, MAX_AVEL, MAX_SPEED_V3
from world_model.wm_predictor import (
    N_EVENT_TYPES, BALL_POCKET_IDX, MAX_EVENTS,
    TABLE_W, TABLE_H,
    encode_obs_act,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

ENC_IN_DIM = 19   # obs_norm(16) + sin/cos angle(2) + speed_norm(1)

# event_state slice indices
S_CUE_XY    = slice(0,  2)
S_CUE_VEL   = slice(2,  4)
S_CUE_AVEL  = slice(4,  7)
S_TGT_XY    = slice(7,  9)
S_TGT_VEL   = slice(9,  11)
S_TGT_AVEL  = slice(11, 14)
S_TYPE_OH   = slice(14, 24)


def _mlp(in_dim: int, hidden: tuple, out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ── MarkovEncoder ─────────────────────────────────────────────────────────────

class MarkovEncoder(nn.Module):
    """
    (obs_norm, act) → 첫 번째 이벤트 예측.

    Output heads (총 24 dims):
        event_head : type_logits (10)
        pos_head   : cue_xy(2) + tgt_xy(2)    절대 좌표
        vel_head   : cue_vel(2) + tgt_vel(2)  절대값
        avel_head  : cue_avel(3) + tgt_avel(3) 절대값
    """
    def __init__(self, hidden: tuple = (128, 256, 256)):
        super().__init__()
        self.trunk      = _mlp(ENC_IN_DIM, hidden[:-1], hidden[-1])
        self.event_head = nn.Linear(hidden[-1], N_EVENT_TYPES)
        self.pos_head   = nn.Linear(hidden[-1], 4)   # cue_xy + tgt_xy
        self.vel_head   = nn.Linear(hidden[-1], 4)   # cue_vel + tgt_vel
        self.avel_head  = nn.Linear(hidden[-1], 6)   # cue_avel + tgt_avel

    def forward(self, obs_norm: torch.Tensor, act: torch.Tensor):
        """
        obs_norm : (B, 16)
        act      : (B, 2)
        → event_logits (B, 10), pos (B, 4), vel (B, 4), avel (B, 6)
        """
        h = self.trunk(encode_obs_act(obs_norm, act))
        return (self.event_head(h),
                self.pos_head(h),
                self.vel_head(h),
                self.avel_head(h))

    @torch.no_grad()
    def predict_event(self, obs_norm: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """추론용 — event state tensor (B, 24) 반환."""
        if obs_norm.dim() == 1:
            obs_norm = obs_norm.unsqueeze(0)
            act      = act.unsqueeze(0)
        logits, pos, vel, avel = self.forward(obs_norm, act)
        type_oh = F.one_hot(logits.argmax(-1), N_EVENT_TYPES).float()
        # [cue_xy, cue_vel, cue_avel, tgt_xy, tgt_vel, tgt_avel, type_oh]
        return torch.cat([pos[:, :2], vel[:, :2], avel[:, :3],
                          pos[:, 2:], vel[:, 2:], avel[:, 3:],
                          type_oh], dim=-1)


# ── MarkovTransition ──────────────────────────────────────────────────────────

class MarkovTransition(nn.Module):
    """
    event_state_t (24) → next event prediction.

    Input 인코딩:
        type_embed(d) + cue_xy(2) + cue_vel(2) + cue_avel(3)
                      + tgt_xy(2) + tgt_vel(2) + tgt_avel(3) = d+14

    Output heads:
        event_head : type_logits (10)
        cue_head   : Δcue_xy(2) + cue_vel(2) + cue_avel(3) = 7
        tgt_head   : Δtgt_xy(2) + tgt_vel(2) + tgt_avel(3) = 7
    """
    def __init__(self, hidden: tuple = (256, 512, 256), embed_dim: int = 32):
        super().__init__()
        self.embed      = nn.Embedding(N_EVENT_TYPES, embed_dim)
        in_dim          = embed_dim + 14
        self.trunk      = _mlp(in_dim, hidden[:-1], hidden[-1])
        self.event_head = nn.Linear(hidden[-1], N_EVENT_TYPES)
        self.cue_head   = nn.Linear(hidden[-1], 7)   # Δcue_xy + cue_vel + cue_avel
        self.tgt_head   = nn.Linear(hidden[-1], 7)   # Δtgt_xy + tgt_vel + tgt_avel

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        """state (B, 24) → trunk input (B, embed_dim+14)."""
        type_idx = state[:, S_TYPE_OH].argmax(dim=-1)          # (B,)
        e        = self.embed(type_idx)                         # (B, d)
        phys     = state[:, :14]                                # (B, 14): pos+vel+avel
        return torch.cat([e, phys], dim=-1)

    def forward(self, state: torch.Tensor):
        """
        state : (B, 24)  event_state_t
        → (event_logits (B,10), cue_out (B,7), tgt_out (B,7))
        """
        h = self.trunk(self._encode(state))
        return self.event_head(h), self.cue_head(h), self.tgt_head(h)

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> torch.Tensor:
        """
        추론용 — 다음 event state (B, 24) 반환.
        위치는 delta를 현재 위치에 더해 절대좌표로 복원.
        """
        logits, cue_out, tgt_out = self.forward(state)
        type_oh  = F.one_hot(logits.argmax(-1), N_EVENT_TYPES).float()
        cue_xy   = state[:, S_CUE_XY] + cue_out[:, :2]  # 절대좌표 복원
        tgt_xy   = state[:, S_TGT_XY] + tgt_out[:, :2]
        return torch.cat([cue_xy,         cue_out[:, 2:4],  cue_out[:, 4:7],
                          tgt_xy,         tgt_out[:, 2:4],  tgt_out[:, 4:7],
                          type_oh], dim=-1)


# ── MarkovPredictor (Encoder + Transition 합체) ───────────────────────────────

class MarkovPredictor(nn.Module):
    """
    (obs_norm, act) → event sequence 예측 (autoregressive Markov).

    추론:
        event_0  = encoder(obs_norm, act)
        event_t1 = transition(event_t)  반복
    """
    def __init__(
        self,
        enc_hidden  : tuple = (128, 256, 256),
        trans_hidden: tuple = (256, 512, 256),
        embed_dim   : int   = 32,
    ):
        super().__init__()
        self.encoder    = MarkovEncoder(enc_hidden)
        self.transition = MarkovTransition(trans_hidden, embed_dim)

    @torch.no_grad()
    def predict(self, obs_norm: torch.Tensor, act: torch.Tensor,
                max_steps: int = MAX_EVENTS):
        """
        obs_norm, act : (16,) or (1, 16) tensors
        Returns:
            types   : list[int]
            cue_xys : (T, 2) tensor  normalized
            tgt_xys : (T, 2) tensor  normalized
        """
        self.eval()
        if obs_norm.dim() == 1:
            obs_norm = obs_norm.unsqueeze(0)
            act      = act.unsqueeze(0)

        state  = self.encoder.predict_event(obs_norm, act)   # (1, 24)
        types, cue_xys, tgt_xys = [], [], []

        for _ in range(max_steps):
            type_idx = state[0, S_TYPE_OH].argmax().item()
            types.append(type_idx)
            cue_xys.append(state[0, S_CUE_XY])
            tgt_xys.append(state[0, S_TGT_XY])

            if type_idx == BALL_POCKET_IDX:
                break

            state = self.transition.step(state)

        return types, torch.stack(cue_xys), torch.stack(tgt_xys)


# ── Loss ──────────────────────────────────────────────────────────────────────

def encoder_loss(
    event_logits: torch.Tensor,  # (B, 10)
    pos_pred    : torch.Tensor,  # (B, 4)  cue_xy+tgt_xy
    vel_pred    : torch.Tensor,  # (B, 4)  cue_vel+tgt_vel
    avel_pred   : torch.Tensor,  # (B, 6)  cue_avel+tgt_avel
    events_gt   : torch.Tensor,  # (B, 24) first event GT
    cue_masks   : torch.Tensor,  # (B,)
    tgt_masks   : torch.Tensor,  # (B,)
    label_smoothing: float = 0.1,
):
    """Loss for MarkovEncoder against first event GT."""
    gt_type = events_gt[:, S_TYPE_OH].argmax(dim=-1)
    ce      = F.cross_entropy(event_logits, gt_type,
                              label_smoothing=label_smoothing)

    cm = cue_masks.float().unsqueeze(1)  # (B, 1)
    tm = tgt_masks.float().unsqueeze(1)

    cue_pos_loss = (F.mse_loss(pos_pred[:, :2], events_gt[:, S_CUE_XY], reduction="none")
                    .mean(-1, keepdim=True) * cm).mean()
    tgt_pos_loss = (F.mse_loss(pos_pred[:, 2:], events_gt[:, S_TGT_XY], reduction="none")
                    .mean(-1, keepdim=True) * tm).mean()
    cue_vel_loss = (F.mse_loss(vel_pred[:, :2], events_gt[:, S_CUE_VEL], reduction="none")
                    .mean(-1, keepdim=True) * cm).mean()
    tgt_vel_loss = (F.mse_loss(vel_pred[:, 2:], events_gt[:, S_TGT_VEL], reduction="none")
                    .mean(-1, keepdim=True) * tm).mean()
    cue_av_loss  = (F.mse_loss(avel_pred[:, :3], events_gt[:, S_CUE_AVEL], reduction="none")
                    .mean(-1, keepdim=True) * cm).mean()
    tgt_av_loss  = (F.mse_loss(avel_pred[:, 3:], events_gt[:, S_TGT_AVEL], reduction="none")
                    .mean(-1, keepdim=True) * tm).mean()

    pos_loss  = (cue_pos_loss + tgt_pos_loss) / 2
    vel_loss  = (cue_vel_loss + tgt_vel_loss) / 2
    avel_loss = (cue_av_loss  + tgt_av_loss)  / 2
    total     = ce + pos_loss + vel_loss + avel_loss
    return total, ce, pos_loss, vel_loss, avel_loss


def transition_loss(
    event_logits: torch.Tensor,  # (B, 10)
    cue_out     : torch.Tensor,  # (B, 7)  Δcue_xy + cue_vel + cue_avel
    tgt_out     : torch.Tensor,  # (B, 7)  Δtgt_xy + tgt_vel + tgt_avel
    state_t     : torch.Tensor,  # (B, 24) current event GT
    state_t1    : torch.Tensor,  # (B, 24) next event GT
    cue_masks   : torch.Tensor,  # (B,)    next event cue valid
    tgt_masks   : torch.Tensor,  # (B,)    next event tgt valid
    label_smoothing: float = 0.1,
):
    """Loss for MarkovTransition: (state_t) → (state_{t+1})."""
    gt_type = state_t1[:, S_TYPE_OH].argmax(dim=-1)
    ce      = F.cross_entropy(event_logits, gt_type,
                              label_smoothing=label_smoothing)

    cm = cue_masks.float().unsqueeze(1)
    tm = tgt_masks.float().unsqueeze(1)

    # GT Δpos = pos_{t+1} - pos_t
    gt_dcue_xy = state_t1[:, S_CUE_XY] - state_t[:, S_CUE_XY]
    gt_dtgt_xy = state_t1[:, S_TGT_XY] - state_t[:, S_TGT_XY]

    cue_pos_loss = (F.mse_loss(cue_out[:, :2], gt_dcue_xy, reduction="none")
                    .mean(-1, keepdim=True) * cm).mean()
    tgt_pos_loss = (F.mse_loss(tgt_out[:, :2], gt_dtgt_xy, reduction="none")
                    .mean(-1, keepdim=True) * tm).mean()
    cue_vel_loss = (F.mse_loss(cue_out[:, 2:4], state_t1[:, S_CUE_VEL], reduction="none")
                    .mean(-1, keepdim=True) * cm).mean()
    tgt_vel_loss = (F.mse_loss(tgt_out[:, 2:4], state_t1[:, S_TGT_VEL], reduction="none")
                    .mean(-1, keepdim=True) * tm).mean()
    cue_av_loss  = (F.mse_loss(cue_out[:, 4:7], state_t1[:, S_CUE_AVEL], reduction="none")
                    .mean(-1, keepdim=True) * cm).mean()
    tgt_av_loss  = (F.mse_loss(tgt_out[:, 4:7], state_t1[:, S_TGT_AVEL], reduction="none")
                    .mean(-1, keepdim=True) * tm).mean()

    pos_loss  = (cue_pos_loss + tgt_pos_loss) / 2
    vel_loss  = (cue_vel_loss + tgt_vel_loss) / 2
    avel_loss = (cue_av_loss  + tgt_av_loss)  / 2
    total     = ce + pos_loss + vel_loss + avel_loss
    return total, ce, pos_loss, vel_loss, avel_loss
