"""
world_model/model.py — LSTM encoder + LSTM decoder VAE (Exp-15)

Architecture:
    Encoder : LSTM(input=12, hidden=64) → h_final → μ, log σ² → z ∈ R^z_dim
    Decoder : LSTM — z로 초기 hidden/cell state 초기화 후 autoregressive 생성
                teacher forcing (train): [START, e0, ..., e_{L-2}] → [e0, ..., e_{L-1}]
                autoregressive (infer) : [START] → e0 → e1 → ...

    Loss:
        recon_pos  = MSE(pred_xy,   true_xy)   [mask padded positions]
        recon_type = CrossEntropy(pred_type, true_type)  [mask padded]
        kl         = -0.5 * Σ(1 + log σ² - μ² - σ²)
        total      = recon_pos + recon_type + β * kl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# generate_data.py 와 일치해야 함
N_EVENT_TYPES = 10
EVENT_DIM     = 2 + N_EVENT_TYPES   # 12
MAX_EVENTS    = 32


# ── Encoder ──────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """가변 길이 이벤트 시퀀스 → (μ, log_var)"""
    def __init__(self, input_dim=EVENT_DIM, hidden_dim=64, z_dim=16, num_layers=1):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu   = nn.Linear(hidden_dim, z_dim)
        self.fc_logv = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, lengths):
        """
        x       : (B, MAX_EVENTS, EVENT_DIM)
        lengths : (B,) 실제 이벤트 수 (0인 경우 1로 clamp)
        """
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths_clamped,
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]                      # (B, hidden_dim)
        return self.fc_mu(h), self.fc_logv(h)


# ── Decoder: MLP ─────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """z → 전체 이벤트 시퀀스 재구성 (MLP, MAX_EVENTS 고정)"""
    def __init__(self, z_dim=16, hidden_dim=128, output_dim=EVENT_DIM, max_len=MAX_EVENTS):
        super().__init__()
        self.max_len    = max_len
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * max_len),
        )

    def forward(self, z, **kwargs):
        """z : (B, z_dim) → (B, MAX_EVENTS, EVENT_DIM)"""
        return self.net(z).view(-1, self.max_len, self.output_dim)


# ── Decoder: LSTM ─────────────────────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    """z → 이벤트 시퀀스 (LSTM autoregressive decoder)

    Training  : teacher forcing — [START, e0, ..., e_{L-2}] 입력, [e0, ..., e_{L-1}] 예측
    Inference : autoregressive  — 이전 예측을 다음 입력으로 사용
    """
    def __init__(self, z_dim=16, hidden_dim=128, output_dim=EVENT_DIM, max_len=MAX_EVENTS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len    = max_len
        self.output_dim = output_dim

        # z → 초기 hidden / cell state
        self.fc_h0  = nn.Linear(z_dim, hidden_dim)
        self.fc_c0  = nn.Linear(z_dim, hidden_dim)

        # LSTM: 이전 이벤트(EVENT_DIM) → hidden → 다음 이벤트
        self.lstm   = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 학습 가능한 시작 토큰
        self.start_token = nn.Parameter(torch.zeros(1, 1, output_dim))

    def forward(self, z, x_teacher=None, tf_ratio=1.0):
        """
        z         : (B, z_dim)
        x_teacher : (B, MAX_EVENTS, EVENT_DIM) — teacher forcing 시 실제 시퀀스 제공
                    None → autoregressive inference
        tf_ratio  : float [0, 1] — teacher forcing 비율 (scheduled sampling)
                    1.0 = 완전 teacher forcing, 0.0 = 완전 AR
        Returns   : (B, MAX_EVENTS, EVENT_DIM)
        """
        B  = z.size(0)
        h0 = torch.tanh(self.fc_h0(z)).unsqueeze(0)  # (1, B, H)
        c0 = torch.tanh(self.fc_c0(z)).unsqueeze(0)  # (1, B, H)

        if x_teacher is None or tf_ratio == 0.0:
            # 완전 Autoregressive
            outputs = []
            inp  = self.start_token.expand(B, 1, -1)
            h, c = h0, c0
            for _ in range(self.max_len):
                out, (h, c) = self.lstm(inp, (h, c))
                pred = self.fc_out(out)
                outputs.append(pred)
                inp = pred
            return torch.cat(outputs, dim=1)

        elif tf_ratio == 1.0:
            # 완전 Teacher forcing (fast path)
            start = self.start_token.expand(B, 1, -1)
            x_in  = torch.cat([start, x_teacher[:, :-1]], dim=1)
            out, _ = self.lstm(x_in, (h0, c0))
            return self.fc_out(out)

        else:
            # Scheduled Sampling: 스텝마다 확률적으로 teacher / 모델 예측 선택
            outputs = []
            inp  = self.start_token.expand(B, 1, -1)
            h, c = h0, c0
            for t in range(self.max_len):
                out, (h, c) = self.lstm(inp, (h, c))
                pred = self.fc_out(out)          # (B, 1, D)
                outputs.append(pred)
                if t < self.max_len - 1:
                    if torch.rand(1).item() < tf_ratio:
                        inp = x_teacher[:, t:t+1]   # teacher token
                    else:
                        inp = pred.detach()          # 모델 자체 예측
            return torch.cat(outputs, dim=1)


# ── VAE ──────────────────────────────────────────────────────────────────────

class TrajectoryVAE(nn.Module):
    def __init__(self, z_dim=16, hidden_enc=64, hidden_dec=128, decoder_type="lstm"):
        super().__init__()
        self.encoder      = LSTMEncoder(EVENT_DIM, hidden_enc, z_dim)
        self.decoder_type = decoder_type
        if decoder_type == "mlp":
            self.decoder = MLPDecoder(z_dim, hidden_dec, EVENT_DIM, MAX_EVENTS)
        else:
            self.decoder = LSTMDecoder(z_dim, hidden_dec, EVENT_DIM, MAX_EVENTS)
        self.z_dim = z_dim

    # ── reparameterization ──
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── encode only (inference용) ──
    def encode(self, x, lengths):
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x, lengths, tf_ratio=1.0):
        """학습/검증 공용.
        MLP decoder: tf_ratio 무시 (항상 z만으로 생성)
        LSTM decoder: tf_ratio=1.0→teacher forcing, 0.0→AR, 중간→scheduled sampling
        """
        mu, logvar = self.encoder(x, lengths)
        z          = self.reparameterize(mu, logvar)
        if self.decoder_type == "mlp":
            x_recon = self.decoder(z)
        else:
            x_recon = self.decoder(z, x_teacher=x, tf_ratio=tf_ratio)
        return x_recon, mu, logvar, z

    def reconstruct(self, x, lengths):
        """AR reconstruction — 실제 생성 품질 평가용 (MLP는 forward와 동일)."""
        mu, logvar = self.encoder(x, lengths)
        z          = self.reparameterize(mu, logvar)
        if self.decoder_type == "mlp":
            x_recon = self.decoder(z)
        else:
            x_recon = self.decoder(z, x_teacher=None)
        return x_recon, mu, logvar, z


# ── Loss ─────────────────────────────────────────────────────────────────────

def vae_loss(x, x_recon, mu, logvar, lengths, beta=1.0, pos_weight=1.0):
    """
    x, x_recon  : (B, MAX_EVENTS, EVENT_DIM)
    lengths      : (B,)
    pos_weight   : position MSE 손실 가중치 (type loss 대비 상대적 중요도)
    Returns: total, recon_pos, recon_type, kl
    """
    B, L, _ = x.shape
    device   = x.device

    # 유효한 timestep mask
    mask = torch.zeros(B, L, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l.item()] = 1.0

    # ── position MSE (처음 2 dim) ──
    pred_xy  = x_recon[:, :, :2]
    true_xy  = x[:, :, :2]
    pos_loss = (F.mse_loss(pred_xy, true_xy, reduction="none").mean(-1) * mask).sum()
    pos_loss = pos_loss / mask.sum().clamp(min=1)

    # ── event type CrossEntropy (2: 이후 one-hot) ──
    true_type  = x[:, :, 2:].argmax(dim=-1)        # (B, L)
    pred_logit = x_recon[:, :, 2:]                  # (B, L, N_TYPES)
    ce = F.cross_entropy(pred_logit.reshape(-1, N_EVENT_TYPES),
                         true_type.reshape(-1), reduction="none")
    type_loss = (ce.view(B, L) * mask).sum() / mask.sum().clamp(min=1)

    # ── KL ──
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = pos_weight * pos_loss + type_loss + beta * kl
    return total, pos_loss, type_loss, kl
