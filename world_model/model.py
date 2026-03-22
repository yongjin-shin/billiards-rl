"""
world_model/model.py — LSTM encoder + VAE for billiards trajectory (Exp-15)

Architecture:
    Encoder : LSTM(input=12, hidden=64) → h_final → μ, log σ² → z ∈ R^z_dim
    Decoder : MLP(z_dim → 128 → EVENT_DIM × MAX_EVENTS) → reconstruct trajectory

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


# ── Decoder ──────────────────────────────────────────────────────────────────

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

    def forward(self, z):
        """z : (B, z_dim) → (B, MAX_EVENTS, EVENT_DIM)"""
        return self.net(z).view(-1, self.max_len, self.output_dim)


# ── VAE ──────────────────────────────────────────────────────────────────────

class TrajectoryVAE(nn.Module):
    def __init__(self, z_dim=16, hidden_enc=64, hidden_dec=128):
        super().__init__()
        self.encoder = LSTMEncoder(EVENT_DIM, hidden_enc, z_dim)
        self.decoder = MLPDecoder(z_dim, hidden_dec, EVENT_DIM, MAX_EVENTS)
        self.z_dim   = z_dim

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

    def forward(self, x, lengths):
        mu, logvar = self.encoder(x, lengths)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decoder(z)
        return x_recon, mu, logvar, z


# ── Loss ─────────────────────────────────────────────────────────────────────

def vae_loss(x, x_recon, mu, logvar, lengths, beta=1.0):
    """
    x, x_recon : (B, MAX_EVENTS, EVENT_DIM)
    lengths     : (B,)
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
    pred_logit = x_recon[:, :, 2:]                   # (B, L, N_TYPES)
    ce = F.cross_entropy(pred_logit.reshape(-1, N_EVENT_TYPES),
                         true_type.reshape(-1), reduction="none")
    type_loss = (ce.view(B, L) * mask).sum() / mask.sum().clamp(min=1)

    # ── KL ──
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = pos_loss + type_loss + beta * kl
    return total, pos_loss, type_loss, kl
