"""
world_model/predictor.py — (obs, act) → trajectory predictor

두 모델 비교:
    MLPPredictor : obs+act → MLP → (MAX_EVENTS, EVENT_DIM)   (non-autoregressive)
    LSTMPredictor: obs+act → context → LSTM AR decoder → (MAX_EVENTS, EVENT_DIM)

LSTMPredictor 학습 전략 (--strategy):
    curriculum  : AR prefix를 0 → MAX_EVENTS 까지 점진적으로 확장 (기본값)
                  앞 ar_steps 스텝은 항상 AR, 나머지는 teacher forcing
                  추론 분포와 직접 일치하는 커리큘럼
    tf_ratio    : 각 스텝마다 확률 tf_ratio 로 TF/AR 를 무작위 선택 (scheduled sampling)

Usage:
    from world_model.predictor import MLPPredictor, LSTMPredictor, predictor_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EVENT_TYPES = 10
EVENT_DIM     = 2 + N_EVENT_TYPES   # 12
MAX_EVENTS    = 32


# ── MLP Predictor ─────────────────────────────────────────────────────────────

class MLPPredictor(nn.Module):
    """
    obs + act → MLP → (MAX_EVENTS, EVENT_DIM)

    이벤트 간 순서 의존성 없이 전체 시퀀스를 한 번에 예측.
    hidden_dims: e.g. (256, 512, 256)
    """
    def __init__(
        self,
        obs_dim    : int   = 16,
        act_dim    : int   = 2,
        hidden_dims        = (256, 512, 256),
        event_dim  : int   = EVENT_DIM,
        max_events : int   = MAX_EVENTS,
    ):
        super().__init__()
        self.max_events = max_events
        self.event_dim  = event_dim

        in_dim = obs_dim + act_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, event_dim * max_events))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act, **kwargs):
        """
        obs : (B, obs_dim)
        act : (B, act_dim)
        → (B, MAX_EVENTS, EVENT_DIM)
        """
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).view(-1, self.max_events, self.event_dim)


# ── LSTM Predictor ────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    obs + act → context encoder → LSTM autoregressive decoder → (MAX_EVENTS, EVENT_DIM)

    ctx_hidden  : context MLP hidden size (obs+act → h0/c0 초기화)
    lstm_hidden : LSTM hidden size
    lstm_layers : LSTM num_layers

    학습 : teacher forcing (tf_ratio=1.0) or scheduled sampling (0 < tf_ratio < 1)
    추론 : autoregressive (tf_ratio=0.0)
    """
    def __init__(
        self,
        obs_dim    : int   = 16,
        act_dim    : int   = 2,
        ctx_hidden : int   = 128,
        lstm_hidden: int   = 256,
        lstm_layers: int   = 1,
        event_dim  : int   = EVENT_DIM,
        max_events : int   = MAX_EVENTS,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.max_events  = max_events
        self.event_dim   = event_dim

        in_dim = obs_dim + act_dim
        self.ctx_net = nn.Sequential(
            nn.Linear(in_dim, ctx_hidden),
            nn.ReLU(),
        )
        # 각 레이어의 h0, c0를 한 번에 생성
        self.fc_h0  = nn.Linear(ctx_hidden, lstm_hidden * lstm_layers)
        self.fc_c0  = nn.Linear(ctx_hidden, lstm_hidden * lstm_layers)

        self.lstm   = nn.LSTM(event_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden, event_dim)

        # 학습 가능한 시작 토큰
        self.start_token = nn.Parameter(torch.zeros(1, 1, event_dim))

    def _init_hidden(self, obs, act):
        B   = obs.size(0)
        ctx = self.ctx_net(torch.cat([obs, act], dim=-1))          # (B, ctx_hidden)
        h0  = torch.tanh(self.fc_h0(ctx))                          # (B, layers*hidden)
        c0  = torch.tanh(self.fc_c0(ctx))
        # (layers, B, hidden)
        h0  = h0.view(B, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()
        c0  = c0.view(B, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()
        return h0, c0

    def forward(self, obs, act, x_teacher=None, ar_steps=None, tf_ratio=None):
        """
        obs       : (B, obs_dim)
        act       : (B, act_dim)
        x_teacher : (B, MAX_EVENTS, EVENT_DIM) — None 이면 완전 AR inference
        ar_steps  : int  curriculum 전략. 앞 ar_steps 스텝을 AR로 생성하고 나머지는 TF.
                    0          → 완전 TF (x_teacher 필요)
                    MAX_EVENTS → 완전 AR (x_teacher 불필요)
                    None       → tf_ratio 전략으로 fallback
        tf_ratio  : float [0,1]  scheduled sampling 전략 (ar_steps=None 일 때 사용)
                    1.0 → 완전 TF,  0.0 → 완전 AR
        → (B, MAX_EVENTS, EVENT_DIM)
        """
        # x_teacher 없으면 항상 완전 AR
        if x_teacher is None:
            return self._forward_ar(obs, act)

        # curriculum 전략
        if ar_steps is not None:
            return self._forward_curriculum(obs, act, x_teacher, ar_steps)

        # tf_ratio 전략 (scheduled sampling)
        ratio = 1.0 if tf_ratio is None else tf_ratio
        return self._forward_tf_ratio(obs, act, x_teacher, ratio)

    # ── curriculum ────────────────────────────────────────────────────────────

    def _forward_curriculum(self, obs, act, x_teacher, ar_steps: int):
        """
        앞 ar_steps 스텝은 AR, 이후는 TF.

        경계 처리: AR 마지막 예측을 TF suffix 의 첫 번째 input 으로 사용
                   → teacher token 으로 전환하지 않아 분포 연속성 유지.

        step:  [0]  [1] ... [ar-1] | [ar] [ar+1] ... [L-1]
               <─── AR ──────────> | <──── TF ───────────>
               start_token 부터       마지막 AR pred 부터
        """
        B    = obs.size(0)
        h, c = self._init_hidden(obs, act)
        outputs = []

        # ── AR prefix ──
        inp = self.start_token.expand(B, 1, -1)
        for _ in range(ar_steps):
            out, (h, c) = self.lstm(inp, (h, c))
            pred = self.fc_out(out)
            outputs.append(pred)
            inp = pred.detach()

        # ── TF suffix ──
        tf_len = self.max_events - ar_steps
        if tf_len > 0:
            if ar_steps == 0:
                inp = self.start_token.expand(B, 1, -1)
            # inp: AR 마지막 예측 (or start_token) → TF suffix 의 step[ar_steps] 입력
            x_in = torch.cat([inp, x_teacher[:, ar_steps:ar_steps + tf_len - 1]], dim=1)
            out, _ = self.lstm(x_in, (h, c))
            outputs.append(self.fc_out(out))

        return torch.cat(outputs, dim=1)

    # ── scheduled sampling (legacy) ───────────────────────────────────────────

    def _forward_tf_ratio(self, obs, act, x_teacher, tf_ratio: float):
        """각 스텝마다 확률 tf_ratio 로 TF / AR 를 무작위 선택."""
        B    = obs.size(0)
        h, c = self._init_hidden(obs, act)

        if tf_ratio == 1.0:
            # fast path: 완전 TF
            start = self.start_token.expand(B, 1, -1)
            x_in  = torch.cat([start, x_teacher[:, :-1]], dim=1)
            out, _ = self.lstm(x_in, (h, c))
            return self.fc_out(out)

        outputs = []
        inp = self.start_token.expand(B, 1, -1)
        for t in range(self.max_events):
            out, (h, c) = self.lstm(inp, (h, c))
            pred = self.fc_out(out)
            outputs.append(pred)
            if t < self.max_events - 1:
                inp = x_teacher[:, t:t+1] if torch.rand(1).item() < tf_ratio \
                      else pred.detach()
        return torch.cat(outputs, dim=1)

    # ── fully AR (inference) ──────────────────────────────────────────────────

    def _forward_ar(self, obs, act):
        """완전 Autoregressive — 추론 전용."""
        B    = obs.size(0)
        h, c = self._init_hidden(obs, act)
        outputs = []
        inp = self.start_token.expand(B, 1, -1)
        for _ in range(self.max_events):
            out, (h, c) = self.lstm(inp, (h, c))
            pred = self.fc_out(out)
            outputs.append(pred)
            inp = pred
        return torch.cat(outputs, dim=1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def predictor_loss(x, x_pred, lengths, pos_weight=1.0):
    """
    x, x_pred  : (B, MAX_EVENTS, EVENT_DIM)
    lengths    : (B,)  실제 이벤트 수 (패딩 마스킹에 사용)
    pos_weight : position MSE 가중치
    Returns    : total, pos_loss, type_loss
    """
    B, L, _ = x.shape
    device  = x.device

    # 유효한 timestep mask
    idx  = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
    mask = (idx < lengths.unsqueeze(1)).float()           # (B, L)

    # ── position MSE (첫 2 dim) ──
    pred_xy  = x_pred[:, :, :2]
    true_xy  = x[:, :, :2]
    pos_loss = (F.mse_loss(pred_xy, true_xy, reduction="none").mean(-1) * mask).sum()
    pos_loss = pos_loss / mask.sum().clamp(min=1)

    # ── event type CrossEntropy ──
    true_type  = x[:, :, 2:].argmax(dim=-1)           # (B, L)
    pred_logit = x_pred[:, :, 2:]                      # (B, L, N_TYPES)
    ce = F.cross_entropy(
        pred_logit.reshape(-1, N_EVENT_TYPES),
        true_type.reshape(-1),
        reduction="none",
    )
    type_loss = (ce.view(B, L) * mask).sum() / mask.sum().clamp(min=1)

    total = pos_weight * pos_loss + type_loss
    return total, pos_loss, type_loss
