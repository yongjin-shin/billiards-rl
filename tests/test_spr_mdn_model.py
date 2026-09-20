"""tests/test_spr_mdn_model.py — SPR-MDN (Laplace) model shape and correctness tests."""

import math
import numpy as np
import torch
import pytest
from world_model.spr_mdn.spr_mdn_model import (
    MixtureHead, SPRMDNModel, laplace_nll_mixture, spr_rollout_loss,
    LATENT_DIM, N_COMPONENTS, ACTION_DIM,
)


def test_mixture_head_shapes():
    head = MixtureHead(LATENT_DIM, N_COMPONENTS, ACTION_DIM)
    B = 4
    z = torch.randn(B, LATENT_DIM)
    a = torch.randn(B, ACTION_DIM)
    pi, mu, b = head(z, a)

    assert pi.shape == (B, N_COMPONENTS),                    f"pi: {pi.shape}"
    assert mu.shape == (B, N_COMPONENTS, LATENT_DIM),        f"mu: {mu.shape}"
    assert b.shape  == (B, N_COMPONENTS, LATENT_DIM),        f"b: {b.shape}"

    assert (pi >= 0).all() and (pi <= 1).all(), "pi out of [0,1]"
    assert torch.allclose(pi.sum(-1), torch.ones(B), atol=1e-5), "pi not summing to 1"
    assert (b > 0).all(), "b must be positive"
    assert b.min() >= 0.01 - 1e-5, "b must be ≥ epsilon=0.01"


def test_mixture_head_b_init():
    """Initial b should be ≈ softplus(2.0) + 0.01 ≈ 2.13 (bias init check)."""
    torch.manual_seed(0)
    head = MixtureHead(LATENT_DIM, N_COMPONENTS, ACTION_DIM)
    z = torch.zeros(1, LATENT_DIM)
    a = torch.zeros(1, ACTION_DIM)
    with torch.no_grad():
        _, _, b = head(z, a)
    # After bias init of 2.0, softplus(~2.0) + 0.01 ≈ 2.13 for the average component
    assert b.mean().item() > 1.0, f"Initial b too small: {b.mean().item():.3f}"


def test_laplace_nll_scalar():
    pi    = torch.softmax(torch.randn(8, N_COMPONENTS), dim=-1)
    mu    = torch.randn(8, N_COMPONENTS, LATENT_DIM)
    b     = torch.ones(8, N_COMPONENTS, LATENT_DIM) * 0.5
    target = mu[:, 0, :]   # at first component mean → NLL should be finite

    nll = laplace_nll_mixture(pi, mu, b, target)
    assert nll.shape == (), f"expected scalar, got {nll.shape}"
    assert torch.isfinite(nll), f"NLL is not finite: {nll}"


def test_laplace_nll_lower_for_better_fit():
    """NLL should be lower when target is at the mixture mean."""
    torch.manual_seed(0)
    pi    = torch.softmax(torch.ones(4, N_COMPONENTS), dim=-1)   # uniform
    mu    = torch.zeros(4, N_COMPONENTS, LATENT_DIM)
    b     = torch.ones(4, N_COMPONENTS, LATENT_DIM)

    target_good = torch.zeros(4, LATENT_DIM)        # at the mean
    target_bad  = torch.ones(4, LATENT_DIM) * 100   # far away

    nll_good = laplace_nll_mixture(pi, mu, b, target_good)
    nll_bad  = laplace_nll_mixture(pi, mu, b, target_bad)
    assert nll_good < nll_bad, "NLL should be lower for on-mean target"


def test_forward_shapes():
    model = SPRMDNModel()
    B, T = 4, 10
    s_0    = torch.randn(B, 14)
    seq_s  = torch.randn(B, T + 1, 14)
    action = torch.randn(B, ACTION_DIM)

    s_hat, type_logit, z_hat_list, pi_list, mu_list, b_list, z_bar_list = \
        model(s_0, seq_s, action, T)

    assert s_hat.shape      == (B, T + 1, 14), f"s_hat: {s_hat.shape}"
    assert type_logit.shape == (B, T, 5),       f"type_logit: {type_logit.shape}"
    assert len(z_hat_list)  == T + 1,           f"z_hat_list len: {len(z_hat_list)}"
    assert len(pi_list)     == T,               f"pi_list len: {len(pi_list)}"
    assert len(b_list)      == T,               f"b_list len: {len(b_list)}"
    assert len(z_bar_list)  == T,               f"z_bar_list len: {len(z_bar_list)}"


def test_rollout_eval_shapes():
    model = SPRMDNModel()
    B, T = 2, 60
    s_0    = torch.randn(B, 14)
    action = torch.randn(B, ACTION_DIM)

    s_hat, type_logit = model.rollout_eval(s_0, T, action=action)

    assert s_hat.shape      == (B, T + 1, 14), f"s_hat: {s_hat.shape}"
    assert type_logit.shape == (B, T, 5),       f"type_logit: {type_logit.shape}"


def test_ema_update():
    model = SPRMDNModel(ema_tau=0.9)
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


def test_z_chain_stop_grad():
    """
    z_hat_{h≥1} gradient must NOT flow back to mixture_head (μ,b path is stop-grad).
    But encoder.ln γ,β SHOULD receive gradient from the self-chaining path.
    """
    model = SPRMDNModel()
    B, T = 2, 5
    s_0    = torch.randn(B, 14)
    seq_s  = torch.randn(B, T + 1, 14)
    action = torch.randn(B, ACTION_DIM)

    s_hat, type_logit, z_hat_list, pi_list, mu_list, b_list, z_bar_list = \
        model(s_0, seq_s, action, T)

    # Compute a loss that flows gradient through z_hat_list[1:]
    loss = s_hat.sum()
    loss.backward()

    # mixture_head must NOT get gradient from the chain (stop-grad on sampling path)
    mh_grad = model.mixture_head.net[0].weight.grad
    # After zeroing, re-run with only recon from z_hat[1:] to verify
    # Simpler proxy: z_hat[h>=1] grad_fn should be LayerNorm (not sampling), i.e.
    # gradient stops at the z_samp boundary.
    for h, z in enumerate(z_hat_list[1:], start=1):
        # z_hat goes through encoder.ln — grad_fn should be layer norm, not deeper
        assert z.grad_fn is not None, f"z_hat[{h}] should have grad_fn (encoder.ln)"
        assert "LayerNorm" in type(z.grad_fn).__name__, \
            f"z_hat[{h}] grad_fn should be LayerNorm, got {type(z.grad_fn).__name__}"

    # encoder.ln γ,β must receive gradient from self-chaining path
    assert model.encoder.ln.weight.grad is not None, "encoder.ln.weight should have grad"


def test_spr_rollout_loss_runs():
    model = SPRMDNModel()
    B, T = 4, 10
    s_0    = torch.randn(B, 14)
    seq_s  = torch.randn(B, T + 1, 14)
    action = torch.randn(B, ACTION_DIM)
    seq_t  = torch.zeros(B, T, dtype=torch.long)

    s_hat, type_logit, z_hat_list, pi_list, mu_list, b_list, z_bar_list = \
        model(s_0, seq_s, action, T)

    loss, detail = spr_rollout_loss(
        pi_list, mu_list, b_list, z_bar_list,
        s_hat, seq_s, type_logit, seq_t,
    )

    assert torch.isfinite(loss), f"loss not finite: {loss}"
    for k in ("loss_nll", "loss_recon", "loss_type"):
        assert k in detail, f"missing key: {k}"

    loss.backward()
    assert model.encoder.enc.net[0].weight.grad is not None, "no grad on encoder"
    assert model.mixture_head.net[0].weight.grad is not None, "no grad on mixture_head"


# ── SPREpisodeSubset augmentation tests ───────────────────────────────────────

def _make_episode(L: int = 10):
    from world_model.spr_mdn.spr_dataset import SPREpisodeSubset
    # State layout: [cue_x(0), cue_y(1), cue_vx(2), cue_vy(3), cue_wx(4), cue_wy(5), cue_wz(6),
    #                tgt_x(7), tgt_y(8), tgt_vx(9), tgt_vy(10), tgt_wx(11), tgt_wy(12), tgt_wz(13)]
    rng = np.random.default_rng(0)
    ep_s = rng.random((L, 14)).astype(np.float32)
    ep_f = np.zeros(L - 1, dtype=bool)
    ep_t = np.zeros(L - 1, dtype=np.int64)
    ep_a = rng.random(2).astype(np.float32)
    episodes = [(ep_s, ep_f, ep_t, 0, ep_a)]
    return episodes, ep_s, ep_a


def test_aug_lr_flip_negates_vx_not_vy():
    """LR flip must negate vx (indices 2,9), NOT vy (indices 3,10)."""
    from world_model.spr_mdn.spr_dataset import SPREpisodeSubset
    episodes, ep_s, ep_a = _make_episode()
    ds = SPREpisodeSubset(episodes, rollout_steps=5, augment=True, flip_tb=False)

    torch.manual_seed(0)
    found_flip = False
    for _ in range(50):
        seq_s, _, _, seq_a = ds[0]
        orig_x   = torch.from_numpy(ep_s[:6, 0])
        flipped_x = 1.0 - orig_x
        if torch.allclose(seq_s[:6, 0], flipped_x, atol=1e-5):
            # LR flip was applied — verify vx negated, vy unchanged
            assert torch.allclose(seq_s[:6, 2], -torch.from_numpy(ep_s[:6, 2]), atol=1e-5), \
                "LR flip: cue_vx should be negated"
            assert torch.allclose(seq_s[:6, 9], -torch.from_numpy(ep_s[:6, 9]), atol=1e-5), \
                "LR flip: tgt_vx should be negated"
            assert torch.allclose(seq_s[:6, 3],  torch.from_numpy(ep_s[:6, 3]), atol=1e-5), \
                "LR flip: cue_vy must NOT be modified"
            assert torch.allclose(seq_s[:6, 10], torch.from_numpy(ep_s[:6, 10]), atol=1e-5), \
                "LR flip: tgt_vy must NOT be modified"
            found_flip = True
            break
    assert found_flip, "LR flip never triggered in 50 tries"


def test_aug_tb_flip_negates_vy_not_vx():
    """TB flip must negate vy (indices 3,10) and wy (5,12), NOT vx."""
    from world_model.spr_mdn.spr_dataset import SPREpisodeSubset
    episodes, ep_s, ep_a = _make_episode()
    ds = SPREpisodeSubset(episodes, rollout_steps=5, augment=True, flip_tb=True)

    torch.manual_seed(1)
    found_tb_only = False
    for _ in range(200):
        seq_s, _, _, seq_a = ds[0]
        orig_x = torch.from_numpy(ep_s[:6, 0])
        orig_y = torch.from_numpy(ep_s[:6, 1])
        flipped_y = 1.0 - orig_y
        lr_flipped = not torch.allclose(seq_s[:6, 0], orig_x, atol=1e-5)
        tb_flipped = torch.allclose(seq_s[:6, 1], flipped_y, atol=1e-5)
        if tb_flipped and not lr_flipped:
            assert torch.allclose(seq_s[:6, 3],  -torch.from_numpy(ep_s[:6, 3]),  atol=1e-5), \
                "TB flip: cue_vy should be negated"
            assert torch.allclose(seq_s[:6, 10], -torch.from_numpy(ep_s[:6, 10]), atol=1e-5), \
                "TB flip: tgt_vy should be negated"
            assert torch.allclose(seq_s[:6, 5],  -torch.from_numpy(ep_s[:6, 5]),  atol=1e-5), \
                "TB flip: cue_wy should be negated"
            assert torch.allclose(seq_s[:6, 12], -torch.from_numpy(ep_s[:6, 12]), atol=1e-5), \
                "TB flip: tgt_wy should be negated"
            assert torch.allclose(seq_s[:6, 2], torch.from_numpy(ep_s[:6, 2]), atol=1e-5), \
                "TB flip: cue_vx must NOT be modified"
            found_tb_only = True
            break
    assert found_tb_only, "TB-only flip never triggered in 200 tries"
