"""tests/test_spr_mdn_model.py — SPR-MDN model shape and correctness tests."""

import math
import torch
import pytest
from world_model.spr_mdn.spr_mdn_model import (
    MixtureHead, SPRMDNModel, mdn_nll, spr_rollout_loss,
    LATENT_DIM, N_COMPONENTS,
)


def test_mixture_head_shapes():
    head = MixtureHead(LATENT_DIM, N_COMPONENTS)
    z = torch.randn(4, LATENT_DIM)
    pi, mu, sigma = head(z)

    assert pi.shape    == (4, N_COMPONENTS),                    f"pi: {pi.shape}"
    assert mu.shape    == (4, N_COMPONENTS, LATENT_DIM),        f"mu: {mu.shape}"
    assert sigma.shape == (4, N_COMPONENTS, LATENT_DIM),        f"sigma: {sigma.shape}"

    assert (pi >= 0).all() and (pi <= 1).all(), "pi out of [0,1]"
    assert torch.allclose(pi.sum(-1), torch.ones(4), atol=1e-5), "pi not summing to 1"
    assert (sigma > 0).all(), "sigma must be positive"


def test_mdn_nll_scalar():
    pi    = torch.softmax(torch.randn(8, N_COMPONENTS), dim=-1)
    mu    = torch.randn(8, N_COMPONENTS, LATENT_DIM)
    sigma = torch.ones(8, N_COMPONENTS, LATENT_DIM) * 0.1
    # target = first component mean → NLL should be low
    target = mu[:, 0, :]

    nll = mdn_nll(pi, mu, sigma, target)
    assert nll.shape == (), f"expected scalar, got {nll.shape}"
    assert torch.isfinite(nll), "NLL is not finite"


def test_mdn_nll_lower_for_better_fit():
    """NLL should be lower when the target matches the mixture mean."""
    torch.manual_seed(0)
    pi    = torch.softmax(torch.ones(4, N_COMPONENTS), dim=-1)   # uniform
    mu    = torch.zeros(4, N_COMPONENTS, LATENT_DIM)
    sigma = torch.ones(4, N_COMPONENTS, LATENT_DIM)

    target_good = torch.zeros(4, LATENT_DIM)        # at the mean
    target_bad  = torch.ones(4, LATENT_DIM) * 100   # far away

    nll_good = mdn_nll(pi, mu, sigma, target_good)
    nll_bad  = mdn_nll(pi, mu, sigma, target_bad)
    assert nll_good < nll_bad, "NLL should be lower for on-mean target"


def test_forward_shapes():
    model = SPRMDNModel()
    B, T = 4, 10
    s_0   = torch.randn(B, 14)
    seq_s = torch.randn(B, T + 1, 14)

    s_hat, type_logit, z_hat_list, pi_list, mu_list, sigma_list, z_bar_list = \
        model(s_0, seq_s, T)

    assert s_hat.shape      == (B, T + 1, 14), f"s_hat: {s_hat.shape}"
    assert type_logit.shape == (B, T, 5),       f"type_logit: {type_logit.shape}"
    assert len(z_hat_list)  == T + 1
    assert len(pi_list)     == T
    assert len(z_bar_list)  == T


def test_rollout_eval_shapes():
    model = SPRMDNModel()
    B, T = 2, 60
    s_0 = torch.randn(B, 14)

    s_hat, type_logit = model.rollout_eval(s_0, T)

    assert s_hat.shape      == (B, T + 1, 14), f"s_hat: {s_hat.shape}"
    assert type_logit.shape == (B, T, 5),       f"type_logit: {type_logit.shape}"


def test_ema_update():
    model = SPRMDNModel(ema_tau=0.9)
    # Perturb encoder weights
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p))

    enc_before  = [p.data.clone() for p in model.encoder.parameters()]
    ema_before  = [p.data.clone() for p in model.ema_encoder.parameters()]

    model.update_ema()

    ema_after = [p.data.clone() for p in model.ema_encoder.parameters()]

    for eb, ea, encb in zip(ema_before, ema_after, enc_before):
        expected = 0.9 * eb + 0.1 * encb
        assert torch.allclose(ea, expected, atol=1e-5), "EMA update incorrect"


def test_ema_not_in_grad():
    model = SPRMDNModel()
    for p in model.ema_encoder.parameters():
        assert not p.requires_grad, "ema_encoder params must not require grad"


def test_spr_rollout_loss_runs():
    model = SPRMDNModel()
    log_sigma = torch.nn.Parameter(torch.zeros(2))
    B, T = 4, 10
    s_0   = torch.randn(B, 14)
    seq_s = torch.randn(B, T + 1, 14)
    seq_t = torch.zeros(B, T, dtype=torch.long)

    s_hat, type_logit, z_hat_list, pi_list, mu_list, sigma_list, z_bar_list = \
        model(s_0, seq_s, T)

    loss, detail = spr_rollout_loss(
        pi_list, mu_list, sigma_list, z_bar_list,
        s_hat, seq_s, type_logit, seq_t,
        log_sigma=log_sigma,
    )

    assert torch.isfinite(loss), f"loss not finite: {loss}"
    for k in ("loss_nll", "loss_recon", "loss_type", "kw_nll", "kw_recon"):
        assert k in detail, f"missing key: {k}"

    loss.backward()
    assert model.encoder.net[0].weight.grad is not None, "no grad on encoder"
    assert model.mixture_head.net[0].weight.grad is not None, "no grad on mixture_head"
    assert log_sigma.grad is not None, "no grad on log_sigma"
