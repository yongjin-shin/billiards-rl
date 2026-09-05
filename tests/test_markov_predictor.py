"""tests/test_markov_predictor.py — MarkovEncoder / MarkovTransition unit tests."""

import pytest
import torch
import numpy as np

from world_model.markov_predictor import (
    MarkovEncoder, MarkovTransition, MarkovPredictor,
    encoder_loss, transition_loss,
    EVENT_DIM_V3, S_CUE_XY, S_TGT_XY, S_TYPE_OH,
)
from world_model.wm_predictor import N_EVENT_TYPES, MAX_EVENTS


# ── fixtures ──────────────────────────────────────────────────────────────────

B = 8   # batch size

@pytest.fixture
def obs_act():
    obs = torch.rand(B, 16)
    act = torch.rand(B, 2)
    return obs, act

@pytest.fixture
def event_state():
    """Random valid event state (B, 24)."""
    state = torch.rand(B, EVENT_DIM_V3)
    # type one-hot: pick random class
    type_oh = torch.zeros(B, N_EVENT_TYPES)
    idx     = torch.randint(0, N_EVENT_TYPES, (B,))
    type_oh.scatter_(1, idx.unsqueeze(1), 1.0)
    state[:, S_TYPE_OH] = type_oh
    return state

@pytest.fixture
def masks():
    cue_m = torch.ones(B)
    tgt_m = (torch.rand(B) > 0.5).float()
    return cue_m, tgt_m


# ── MarkovEncoder ─────────────────────────────────────────────────────────────

class TestMarkovEncoder:
    def test_forward_shape(self, obs_act):
        obs, act = obs_act
        enc = MarkovEncoder()
        logits, pos, vel, avel = enc(obs, act)
        assert logits.shape == (B, N_EVENT_TYPES)
        assert pos.shape   == (B, 4)
        assert vel.shape   == (B, 4)
        assert avel.shape  == (B, 6)

    def test_predict_event_shape(self, obs_act):
        obs, act = obs_act
        enc = MarkovEncoder()
        state = enc.predict_event(obs, act)
        assert state.shape == (B, EVENT_DIM_V3)

    def test_predict_event_1d_input(self):
        enc = MarkovEncoder()
        obs = torch.rand(16)
        act = torch.rand(2)
        state = enc.predict_event(obs, act)
        assert state.shape == (1, EVENT_DIM_V3)

    def test_encoder_loss_backward(self, obs_act, event_state, masks):
        obs, act = obs_act
        cue_m, tgt_m = masks
        enc = MarkovEncoder()
        logits, pos, vel, avel = enc(obs, act)
        loss, ce, pl, vl, al = encoder_loss(
            logits, pos, vel, avel, event_state, cue_m, tgt_m)
        assert loss.item() > 0
        loss.backward()
        for p in enc.parameters():
            assert p.grad is not None


# ── MarkovTransition ──────────────────────────────────────────────────────────

class TestMarkovTransition:
    def test_forward_shape(self, event_state):
        trans = MarkovTransition()
        logits, cue_out, tgt_out = trans(event_state)
        assert logits.shape  == (B, N_EVENT_TYPES)
        assert cue_out.shape == (B, 7)
        assert tgt_out.shape == (B, 7)

    def test_step_shape(self, event_state):
        trans = MarkovTransition()
        next_state = trans.step(event_state)
        assert next_state.shape == (B, EVENT_DIM_V3)

    def test_transition_loss_backward(self, event_state, masks):
        state_t  = event_state
        state_t1 = torch.rand(B, EVENT_DIM_V3)
        # valid one-hot for t+1
        type_oh = torch.zeros(B, N_EVENT_TYPES)
        type_oh.scatter_(1, torch.randint(0, N_EVENT_TYPES, (B,)).unsqueeze(1), 1.0)
        state_t1[:, S_TYPE_OH] = type_oh

        cue_m, tgt_m = masks
        trans = MarkovTransition()
        logits, cue_out, tgt_out = trans(state_t)
        loss, *_ = transition_loss(logits, cue_out, tgt_out,
                                   state_t, state_t1, cue_m, tgt_m)
        assert loss.item() > 0
        loss.backward()
        for p in trans.parameters():
            assert p.grad is not None


# ── MarkovPredictor (end-to-end) ──────────────────────────────────────────────

class TestMarkovPredictor:
    def test_predict_returns_valid_types(self, obs_act):
        obs, act = obs_act
        model = MarkovPredictor()
        obs1  = obs[0]
        act1  = act[0]
        types, cue_xys, tgt_xys = model.predict(obs1, act1)
        assert len(types) >= 1
        assert len(types) <= MAX_EVENTS
        assert cue_xys.shape[0] == len(types)
        assert all(0 <= t < N_EVENT_TYPES for t in types)

    def test_predict_batch_1d(self):
        model = MarkovPredictor()
        obs   = torch.rand(16)
        act   = torch.rand(2)
        types, cue_xys, tgt_xys = model.predict(obs, act)
        assert isinstance(types, list)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_tgt_mask_zero(self, event_state):
        """tgt가 전혀 없는 배치 — loss가 NaN이 되면 안 됨."""
        cue_m = torch.ones(B)
        tgt_m = torch.zeros(B)
        enc   = MarkovEncoder()
        obs, act = torch.rand(B, 16), torch.rand(B, 2)
        logits, pos, vel, avel = enc(obs, act)
        loss, *_ = encoder_loss(logits, pos, vel, avel, event_state, cue_m, tgt_m)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_single_sample(self):
        """배치 크기 1."""
        state = torch.rand(1, EVENT_DIM_V3)
        type_oh = torch.zeros(1, N_EVENT_TYPES); type_oh[0, 0] = 1.0
        state[:, S_TYPE_OH] = type_oh
        trans = MarkovTransition()
        logits, cue_out, tgt_out = trans(state)
        assert logits.shape == (1, N_EVENT_TYPES)

    def test_deterministic_inference(self, obs_act):
        """같은 입력 → 같은 출력 (eval 모드)."""
        obs, act = obs_act
        model = MarkovPredictor()
        model.eval()
        t1, c1, _ = model.predict(obs[0], act[0])
        t2, c2, _ = model.predict(obs[0], act[0])
        assert t1 == t2
        assert torch.allclose(c1, c2)
