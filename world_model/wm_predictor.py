"""
world_model/wm_predictor.py — redesigned (obs, act) → trajectory predictor

Architecture:
    Encoder : MLP(19) → (h0, c0)
              obs_norm(16) + sin(angle) + cos(angle) + speed_norm = 19

    Decoder : LSTM step-by-step, 두 개의 head
              Step_t input : [event_embed(d) | cue_xy_{t-1}(2) | tgt_xy_{t-1}(2)]
              Event head   : Linear(H → 10)     → event type logits
              Pos head     : Linear(H+d → 4)    → (cue_xy, tgt_xy)

              t=0 시작 위치: obs의 초기 cue_pos, tgt_pos 사용
              EOS          : BALL_POCKET 예측 시 생성 종료

Data format v2 (generate_data_v2.py 와 대응):
    events    : (B, MAX_EVENTS, EVENT_DIM_V2)
                [:, :, 0:2] = cue_xy  (normalized)
                [:, :, 2:4] = tgt_xy  (normalized)
                [:, :, 4: ] = one_hot type (10)
    cue_masks : (B, MAX_EVENTS)  — cue_xy 가 유효한 스텝 = 1
    tgt_masks : (B, MAX_EVENTS)  — tgt_xy 가 유효한 스텝 = 1
    lengths   : (B,)

Note: Dataset 에서 좌표를 [0,1] 로 정규화하고 넘겨줘야 함.
      TABLE_W / TABLE_H 상수는 generate_data_v2.py 와 공유.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 상수 ──────────────────────────────────────────────────────────────────────

EVENT_TYPES = [
    "none",
    "stick_ball",
    "ball_ball",
    "ball_linear_cushion",
    "ball_circular_cushion",
    "ball_pocket",
    "sliding_rolling",
    "rolling_spinning",
    "rolling_stationary",
    "spinning_stationary",
]
N_EVENT_TYPES    = len(EVENT_TYPES)        # 10
BALL_POCKET_IDX  = EVENT_TYPES.index("ball_pocket")   # 5
EVENT2IDX        = {e: i for i, e in enumerate(EVENT_TYPES)}
MAX_EVENTS       = 32
EVENT_DIM_V2     = 4 + N_EVENT_TYPES      # cue_xy + tgt_xy + one_hot = 14

TABLE_W   = 0.9906   # pooltool standard billiards table width  (m)
TABLE_H   = 1.9812   # pooltool standard billiards table height (m)
MAX_SPEED = 8.0      # act[:, 1] normalization factor


# ── Input encoding ────────────────────────────────────────────────────────────

def encode_obs_act(obs_norm: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """
    obs_norm : (B, 16)  이미 정규화된 obs  [0,1] 좌표
    act      : (B, 2)   [delta_angle, speed]  raw
    →          (B, 19)  angle → sin/cos,  speed → [0,1]
    """
    angle = act[:, 0:1]
    speed = act[:, 1:2] / MAX_SPEED
    return torch.cat([obs_norm, angle.sin(), angle.cos(), speed], dim=-1)


# ── Model ─────────────────────────────────────────────────────────────────────

class WMPredictor(nn.Module):
    """
    (obs_norm, act) → event sequence (event_types, cue_xys, tgt_xys)

    파라미터:
        enc_hidden      : Encoder MLP hidden layer sizes  e.g. (128, 128)
        lstm_hidden     : LSTM hidden size
        lstm_layers     : LSTM num_layers
        event_embed_dim : event type embedding dimension
    """
    def __init__(
        self,
        enc_hidden      : tuple = (128, 256),
        lstm_hidden     : int   = 256,
        lstm_layers     : int   = 2,
        lstm_dropout    : float = 0.1,
        event_embed_dim : int   = 32,
    ):
        super().__init__()
        self.lstm_hidden     = lstm_hidden
        self.lstm_layers     = lstm_layers
        self.event_embed_dim = event_embed_dim

        # ── Encoder ───────────────────────────────────────────────────────────
        in_dim, layers = 19, []
        for h in enc_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_h0   = nn.Linear(in_dim, lstm_hidden * lstm_layers)
        self.fc_c0   = nn.Linear(in_dim, lstm_hidden * lstm_layers)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.event_embed = nn.Embedding(N_EVENT_TYPES, event_embed_dim)

        # step input: event_embed(d) + cue_xy(2) + tgt_xy(2) + Δcue(2) + Δtgt(2) = d+8
        self.decoder   = nn.LSTM(event_embed_dim + 8, lstm_hidden,
                                 lstm_layers, batch_first=True,
                                 dropout=lstm_dropout if lstm_layers > 1 else 0.0)
        # 두 head 완전 독립: gradient 간섭 없음
        # pos_head  : hidden → Δpos (절대좌표 아닌 이동량 예측)
        # event_head: hidden → event logits
        self.pos_head   = nn.Linear(lstm_hidden, 4)
        self.event_head = nn.Linear(lstm_hidden, N_EVENT_TYPES)

    # ─────────────────────────────────────────────────────────────────────────

    def _init_hidden(self, obs_norm, act):
        ctx = self.encoder(encode_obs_act(obs_norm, act))   # (B, enc_out)
        B   = obs_norm.size(0)
        h0  = (torch.tanh(self.fc_h0(ctx))
               .view(B, self.lstm_layers, self.lstm_hidden)
               .permute(1, 0, 2).contiguous())
        c0  = (torch.tanh(self.fc_c0(ctx))
               .view(B, self.lstm_layers, self.lstm_hidden)
               .permute(1, 0, 2).contiguous())
        return h0, c0

    def _step(self, prev_type_idx, prev_cue, prev_tgt, delta_cue, delta_tgt, h, c):
        """LSTM step → hidden (B, H)"""
        e_emb    = self.event_embed(prev_type_idx)                              # (B, d)
        step_in  = torch.cat([e_emb, prev_cue, prev_tgt,
                               delta_cue, delta_tgt], dim=-1)                  # (B, d+8)
        out, (h, c) = self.decoder(step_in.unsqueeze(1), (h, c))
        return out.squeeze(1), h, c                                             # (B, H)

    def _pos(self, hidden):
        """Position head: hidden → Δpos (Δcue_xy, Δtgt_xy)  (B, 4)"""
        return self.pos_head(hidden)

    def _event(self, hidden):
        """Event head: hidden → logits  (B, K)  — pos와 독립"""
        return self.event_head(hidden)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(self, obs_norm, act, events, lengths, tf_ratio=1.0):
        """
        obs_norm : (B, 16)  정규화된 obs
        act      : (B, 2)
        events   : (B, MAX_EVENTS, 14)  v2 format, 정규화된 좌표
        lengths  : (B,)
        tf_ratio : float [0,1]  per-step teacher forcing 비율

        Returns:
            event_logits : (B, MAX_EVENTS, 10)
            pos_pred     : (B, MAX_EVENTS, 4)   — (cue_xy, tgt_xy) normalized
        """
        B      = obs_norm.size(0)
        device = obs_norm.device
        h, c   = self._init_hidden(obs_norm, act)

        gt_cue_tgt = events[:, :, :4]                          # (B, T, 4)
        gt_types   = events[:, :, 4:].argmax(dim=-1)           # (B, T)

        # t=0 이전 상태: obs의 초기 위치, delta는 0으로 시작
        prev_type  = torch.zeros(B, dtype=torch.long, device=device)   # none(0) as start
        prev_cue   = obs_norm[:, 0:2]                                   # cue_pos_0
        prev_tgt   = obs_norm[:, 2:4]                                   # tgt_pos_0
        delta_cue  = torch.zeros(B, 2, device=device)                   # Δcue_0 = 0
        delta_tgt  = torch.zeros(B, 2, device=device)                   # Δtgt_0 = 0

        all_event_logits, all_pos_pred = [], []

        for t in range(MAX_EVENTS):
            hidden, h, c = self._step(
                prev_type, prev_cue, prev_tgt, delta_cue, delta_tgt, h, c)

            use_tf = (tf_ratio == 1.0) or \
                     (tf_ratio > 0 and torch.rand(1).item() < tf_ratio)

            # pos_head: Δpos 예측, 절대좌표 복원
            delta_pred = self._pos(hidden)                               # (B, 4): Δcue, Δtgt
            abs_cue    = prev_cue + delta_pred[:, 0:2]                  # 절대좌표 복원
            abs_tgt    = prev_tgt + delta_pred[:, 2:4]
            all_pos_pred.append(delta_pred)                              # loss는 Δpos 기준

            # event head: hidden에서 독립적으로 예측
            event_logit = self._event(hidden)                            # (B, K)
            all_event_logits.append(event_logit)

            # 다음 스텝 입력: 절대좌표 기준
            if t < MAX_EVENTS - 1:
                if use_tf:
                    next_cue  = gt_cue_tgt[:, t, 0:2]
                    next_tgt  = gt_cue_tgt[:, t, 2:4]
                    prev_type = gt_types[:, t]
                else:
                    next_cue  = abs_cue.detach()
                    next_tgt  = abs_tgt.detach()
                    prev_type = event_logit.argmax(dim=-1).detach()
                delta_cue = next_cue - prev_cue
                delta_tgt = next_tgt - prev_tgt
                prev_cue  = next_cue
                prev_tgt  = next_tgt

        return (torch.stack(all_event_logits, dim=1),   # (B, T, 10)
                torch.stack(all_pos_pred,     dim=1))   # (B, T, 4)

    # ── Autoregressive inference ───────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, obs_norm, act, max_steps=MAX_EVENTS):
        """
        obs_norm, act : (16,) or (1, 16) tensors
        Returns:
            types    : list[int]           event type indices
            cue_xys  : (T, 2) tensor       normalized
            tgt_xys  : (T, 2) tensor       normalized
        """
        self.eval()
        if obs_norm.dim() == 1:
            obs_norm = obs_norm.unsqueeze(0)
            act      = act.unsqueeze(0)

        device = obs_norm.device
        h, c   = self._init_hidden(obs_norm, act)

        prev_type  = torch.zeros(1, dtype=torch.long, device=device)
        prev_cue   = obs_norm[:, 0:2]
        prev_tgt   = obs_norm[:, 2:4]
        delta_cue  = torch.zeros(1, 2, device=device)
        delta_tgt  = torch.zeros(1, 2, device=device)

        types, cue_xys, tgt_xys = [], [], []

        for _ in range(max_steps):
            hidden, h, c = self._step(
                prev_type, prev_cue, prev_tgt, delta_cue, delta_tgt, h, c)

            delta_pred  = self._pos(hidden)                      # Δpos
            abs_cue     = prev_cue + delta_pred[:, 0:2]          # 절대좌표 복원
            abs_tgt     = prev_tgt + delta_pred[:, 2:4]
            event_logit = self._event(hidden)
            pred_type   = event_logit.argmax(dim=-1)             # (1,)

            types.append(pred_type.item())
            cue_xys.append(abs_cue[0])
            tgt_xys.append(abs_tgt[0])

            if pred_type.item() == BALL_POCKET_IDX:
                break

            delta_cue = abs_cue - prev_cue
            delta_tgt = abs_tgt - prev_tgt
            prev_type = pred_type
            prev_cue  = abs_cue
            prev_tgt  = abs_tgt

        return types, torch.stack(cue_xys), torch.stack(tgt_xys)


# ── Loss ──────────────────────────────────────────────────────────────────────

def wm_loss(event_logits, pos_pred, events, cue_masks, tgt_masks, lengths,
            lambda_event=1.0, lambda_pos=1.0, label_smoothing=0.1,
            init_cue_tgt=None):
    """
    event_logits : (B, T, 10)
    pos_pred     : (B, T, 4)    — 예측 Δpos (Δcue_xy, Δtgt_xy)
    events       : (B, T, 14)
    cue_masks    : (B, T)       — 1 if cue_xy valid
    tgt_masks    : (B, T)       — 1 if tgt_xy valid
    lengths      : (B,)
    init_cue_tgt : (B, 4)       — obs에서의 초기 cue/tgt 절대좌표 (Δpos GT 계산용)
    """
    B, T   = event_logits.shape[:2]
    device = event_logits.device

    # sequence mask
    idx      = torch.arange(T, device=device).unsqueeze(0)
    len_mask = (idx < lengths.unsqueeze(1)).float()             # (B, T)

    # ── event CE (with label smoothing) ─────────────────────────────────────
    gt_types   = events[:, :, 4:].argmax(dim=-1)                # (B, T)
    ce         = F.cross_entropy(event_logits.reshape(-1, N_EVENT_TYPES),
                                 gt_types.reshape(-1),
                                 label_smoothing=label_smoothing,
                                 reduction="none").view(B, T)
    event_loss = (ce * len_mask).sum() / len_mask.sum().clamp(min=1)

    # ── GT Δpos 계산 ─────────────────────────────────────────────────────────
    # Δpos_t = pos_t - pos_{t-1},  t=0: pos_{-1} = init_cue_tgt (obs 초기 위치)
    gt_abs  = events[:, :, :4]                                      # (B, T, 4)
    if init_cue_tgt is not None:
        prev_abs = torch.cat([init_cue_tgt.unsqueeze(1),
                              gt_abs[:, :-1, :]], dim=1)            # (B, T, 4)
    else:
        prev_abs = torch.cat([gt_abs[:, :1, :],
                              gt_abs[:, :-1, :]], dim=1)            # fallback
    gt_delta = gt_abs - prev_abs                                     # (B, T, 4)

    # ── cue Δpos MSE ────────────────────────────────────────────────────────
    cue_m    = len_mask * cue_masks
    cue_loss = (F.mse_loss(pos_pred[:, :, 0:2], gt_delta[:, :, 0:2], reduction="none")
                .mean(-1) * cue_m).sum() / cue_m.sum().clamp(min=1)

    # ── tgt Δpos MSE ────────────────────────────────────────────────────────
    tgt_m    = len_mask * tgt_masks
    tgt_loss = (F.mse_loss(pos_pred[:, :, 2:4], gt_delta[:, :, 2:4], reduction="none")
                .mean(-1) * tgt_m).sum() / tgt_m.sum().clamp(min=1)

    pos_loss = (cue_loss + tgt_loss) / 2
    total    = lambda_event * event_loss + lambda_pos * pos_loss
    return total, event_loss, cue_loss, tgt_loss
