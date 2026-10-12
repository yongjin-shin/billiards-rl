# Experiments

Billiards RL experiment records. Quick reference in the Results table, detailed observations in the Experiment Log.

---

## Results

### Phase 0 · Exp-01 algorithm benchmark

SAC / PPO / TQC, 1M steps, seeds {0, 1, 2}, legacy batch conditions.

| Algo | s0 / s1 / s2 | avg | std |
|------|-------------|-----|-----|
| **SAC** | 81.4 / 73.6 / 77.8 | **77.6%** | ±3.9pp |
| TQC | 84.6 / 80.6 / 35.6 | 66.9% | ±27pp |
| PPO | 24.0 / 30.0 / 31.6 | 28.5% | ±4.0pp |
| Random | — | ~6% | — |

> **Note on reproduction:** Retraining with current code yields ~50% (legacy batch, no-scratch) or ~42% (current batch).
> For root cause and ablation results, see [Exp-01 log](#exp-01--phase-0-single-ball-benchmark).

---

### Phase 1 · ms=5 series (Exp-02~09)

#### Exp-02~06: baseline → reward shaping

| Exp | Condition | Pocket% | Clear% | Ep Len | Note |
|-----|------|---------|--------|--------|------|
| 02 | ms=∞ (ms=15) | 98.3% | 95.8% | — | Too loose, random also 40% |
| 03 | ms=5 scratch | 60.7% | 29.4% | 4.60 | Phase 1 baseline |
| 04 | Transfer A zero-shot | 63.6% | 31.4% | — | Exceeds baseline without additional training |
| 05 | Transfer B warm-start | 61.5% | 30.4% | — | Lower than zero-shot |
| **06** | **pp=✓ scratch** | **63.9%** | **33.2%** | **4.48** | **ms=5 best** |

> Exp-06 rerun: seed=0 complete 63.1%/31.2% (original 63.9%/33.2%). seed=1/2 incomplete.

#### Exp-07~08: ep_len reduction attempts → all failed

| Exp | Changed condition | Pocket% | Clear% | Ep Len |
|-----|----------|---------|--------|--------|
| 06 | pp=✓ (baseline) | 63.9% | 33.2% | 4.48 |
| 07 SAC | cb=2.0 | 62.0% | 29.2% | 4.50 |
| 07 TQC | cb=2.0 | 49.7% | 17.6% | 4.80 |
| 08a | shots_taken + cb=2.0 | 63.7% | 30.0% | 4.60 |
| 08b | shots_taken + lr=1e-4 + gs=10 | 62.1% | 28.8% | 4.50 |

> **Conclusion:** ep_len cannot be shortened via reward shaping / obs. The task structure itself has expected value > 0 at every step.

#### Exp-09: ms × pp ablation grid

SAC, 1M, seed=42, sp=0.1, tp=1.0.

| ms | pp=✗ (pocket/clear) | pp=✓ (pocket/clear) | Δ clear | ep_len/ms |
|----|---------------------|---------------------|---------|-----------|
| 5 | 63.6% / 32.2% | 63.9% / 33.2% | +1.0pp | 88% |
| 4 | 55.1% / 17.6% | 51.2% / 15.8% | −1.8pp | 97% |
| 3 | 41.4% / 9.0% | 41.5% / 7.6% | −1.4pp | **99%** |

> **Conclusion:** pp discarded. ms=3 is the Phase 2 frontier — algorithmic improvement needed, not reward shaping.

---

### Phase 2 · ms=3 frontier (Exp-10~12)

#### Exp-10: algorithm benchmark

SAC/TQC/PPO × 3 seeds, ms=3, sp=0.1, tp=1.0, 2M steps.

| Algo | avg Pocket% | avg Clear% | Note |
|------|------------|------------|------|
| **SAC** | **41.7%** | **8.4%** | Stable (low variance across s0/s1) |
| TQC | 27.1% | 2.0% | Overconservatism |
| PPO | ~6.5% | ~0.0% | Credit assignment limitation |
| Random | ~9% | ~0% | — |

#### Exp-11: Curriculum ms=5→4→3

SAC, seed=42, 2M total (1M + 500k + 500k).

| Stage | ms | Steps | Pocket% | Clear% |
|-------|----|-------|---------|--------|
| 1 | 5 | 1M | 65.1% | 33.2% |
| 2 | 4 | 500k | 53.6% | 21.6% |
| **3** | **3** | **500k** | **43.0%** | **10.4%** |
| +ext | 2 | +500k | 28.6% | 1.6% |
| Exp-10 baseline | 3 | 2M scratch | 41.7% | 8.4% | — |

> Curriculum 2M vs scratch 2M: +1.3pp pocket / +2.0pp clear. ms=2 extension failed due to absence of learning signal.

#### Exp-12: abs_angle ❌ discarded

Replacing delta_angle with absolute angle [0, 2π].

| Seed | Steps | Pocket% | Clear% |
|------|-------|---------|--------|
| 1 | 5M | 37.8% | 6.4% |
| Exp-10 SAC | 2M | **41.7%** | **8.4%** |

> Even with 5M steps, worse than delta 2M. Missing inductive bias (delta=0 → aim directly at ball) is critical. **Keep delta_angle.**

---

## Exp-13 Results · Phase 0 Reward Shaping & Steps Scaling

### Proximity Reward ❌

| α | Pocket% | vs baseline |
|---|---------|-------------|
| 0.0 (baseline, 1M) | **50.0%** | — |
| 0.05 | 40.8% | −9.2pp |
| 0.1 | 37.8% | −12.2pp |
| 0.3 | 41.4% | −8.6pp |
| 0.5 | 42.0% | −8.0pp |

All α values fall below baseline. **Post-shot final distance is not a valid gradient signal.**
After 1~2 cushion bounces, the causal link between the ball's final position and the initial action is broken — `f(action) → chaos`.

### Steps Scaling ✅ · gradient_steps ceiling confirmed

| Steps | gs | Pocket% | Training time | Efficiency |
|-------|----|---------|---------|------|
| 1M | 1 | 50.0% | ~20 min | — |
| 2M | 1 | 56.2% | ~35 min | 6.2pp/1M |
| 5M | 1 | 65.8% | 91 min | 3.2pp/1M |
| 5M | 4 | **68.6%** | 187 min | — |

Consistent performance improvement as steps increase. However, **diminishing returns are pronounced** — efficiency drops by half.

`gradient_steps=4` yields only +2.8pp while doubling training time → low ROI.

**Conclusion: flat policy ceiling confirmed.** Structural limitation of approximating `Q(o,a) ≈ P(pocket | positions, angle)` with only binary feedback. Without knowledge of physics dynamics, adding more samples hits a wall. → **Pivoting to World Model (Exp-15).**

---

## Exp-15 · Trajectory VAE (World Model Foundation)

### Root Problem (flat policy limitation)

```
Q(o, a) ≈ P(pocketed | cue_pos, ball_pos, angle, speed)
```

This function must fully encode billiards physics:
- **Discontinuous**: in/out differs by 0.1°
- **Chaotic**: causal link with initial direction broken after multiple cushion bounces
- Binary reward alone requires enormous samples → 68.6% ceiling at 5M steps

The same reason explains proximity reward failure — post-shot final position is noise after the physics causal chain is broken.

### Formulation

```
f(O_t | a_t) = (O_{t+1}, ..., O_{t+n})    # dynamics: trajectory
g(O_{t+1}, ..., O_{t+n}) = h_t             # encoder → latent z
p(O_t, h_t) = a_t                          # policy
```

Circular dependency resolved → **Recurrent** (h from past) or **MPC/imagination** (TD-MPC / DreamerV3 direction).

### Exp-15 Implementation: Trajectory VAE

**Data**: Full trajectory extracted from pooltool `system.events` starting from `stick_ball`
(cue approach path + ball_ball collisions + target ball's post-collision path all included)
```
event = (x, y, type_one_hot_10)  → 12-dim
sequence: variable length, max 32 events
```

**Event types (10 kinds)**:
`none` / `stick_ball` / `ball_ball` / `ball_linear_cushion` /
`ball_circular_cushion` / `ball_pocket` / `sliding_rolling` /
`rolling_spinning` / `rolling_stationary` / `spinning_stationary`

**Model**:
```
Encoder: LSTM(12 → 64) → h_final → μ, log σ² → z ∈ R^z_dim
Decoder: MLP(z_dim → 128 → 12 × 32)
Loss:    MSE(pos) + CE(event_type) + β·KL
```

**Analysis**:
- t-SNE: latent structure by pocketed / n_bounces / tag (SAC vs random)
- Action correlation: z_dim vs delta_angle / speed
- Latent traversal: decode trajectory by varying each dim over ±3σ

### File Structure

```
world_model/
├── data/              # Generated trajectory datasets (.npz + metadata.json)
├── checkpoints/       # Trained VAE checkpoints
├── results/           # t-SNE, correlation, traversal images
├── generate_data.py   # Data collection with SAC/random models
├── model.py           # LSTM encoder + VAE
├── train_vae.py       # VAE training
├── visualize.py       # t-SNE + latent traversal
└── analyze.py         # Linear probe + correlation analysis
```

### Execution Order

```bash
# 1. Generate data
python world_model/generate_data.py --tag sac_5m_gs4 \
    --model logs/experiments/SAC_5000k_s42_sp0.0_tp0.0_gs4_20260322_150734/best_model/best_model \
    --n-episodes 5000
python world_model/generate_data.py --tag random --n-episodes 5000

# 2. Train VAE (compare z_dim)
python world_model/train_vae.py --z-dim 8
python world_model/train_vae.py --z-dim 16
python world_model/train_vae.py --z-dim 32

# 3. Analysis
python world_model/visualize.py --ckpt world_model/checkpoints/vae_z16_*.pt
python world_model/analyze.py   --ckpt world_model/checkpoints/vae_z16_*.pt
```

---

## Exp-16 · World Model Critic

### Root Problem (flat policy limitation revisited)

```
Q(s, a) ≈ E[r | s, a]
```

Q must implicitly learn physics, but:
- Compressing **(s, a) → scalar** destroys the physics causal chain
- `∂Q/∂a` is noisy — knows "this angle was statistically bad" but not "why it was bad"
- Back-inferring from sparse binary reward → 68.6% ceiling at 5M steps

### Architecture

```
Actor:   π_θ(s) → a                     Standard SAC, unmodified
Critic:  Q(s, a) = q( M(s, a) )         M exists only inside the critic
```

Inside the Critic:

```
s, a ──→ M ──→ ĥ ──→ q ──→ Q
          ↑           ↑
   L_WM (dense)   L_Bellman (sparse)
   MSE(ĥ, h_real)  Bellman backup
```

- `M: (s, a) → ĥ` — simple MLP. Predicts full trajectory. Explicitly learns physics via dense supervision
- `q: ĥ → Q` — value head. Trained with Bellman. Physics is handled by M, so q's problem becomes simpler

### Why Better than Conventional Q(s,a)

```
Previous: ∂Q/∂a            — compresses physics into a single scalar gradient
New:      ∂q/∂ĥ · ∂ĥ/∂a  — ∂ĥ/∂a is the physics Jacobian, M learns it accurately via dense training
```

Gradient path during actor update:
```
actor_loss = -Q + entropy = -q(M(s, π(s))) + entropy
∂/∂θ: ∂q/∂ĥ · ∂ĥ/∂a · ∂a/∂θ   ← physics gradient reaches here, no shared weights
```

### Training

```python
# Critic update (M + q simultaneously)
h_hat    = M(s, a)
L_WM      = MSE(h_hat, h_real)           # dense, all trajectory steps
L_Bellman = MSE(q(h_hat), target_Q)      # sparse, reward
L_critic  = L_Bellman + λ · L_WM
L_critic.backward()
critic_optimizer.step()

# Actor update (standard SAC)
a_pi = π_θ(s)
actor_loss = -q(M(s, a_pi)) + α·log π_θ(a_pi)
actor_loss.backward()   # new forward → new graph, no double-backward
actor_optimizer.step()
```

### SB3 Modification Scope

```
Actor                → unmodified
TrajectoryBuffer     → added h_real storage          (~60 lines)
WorldModelCritic     → inherits ContinuousCritic      (~80 lines)
WorldModelSAC        → overrides train()              (~30 lines)
simulator.py         → includes trajectory in info    (~20 lines)
─────────────────────────────────────────────────────────
Total                ~190 lines, SB3 core intact
```

### Results · Vanilla SAC Implementation Validation (2026-03-28)

VanillaSAC (custom) vs SB3 SAC — 2M steps, n_balls=3, ms=5, seeds {0,1,2,3,42}

| seed | vanilla pocket | SB3 pocket | vanilla clear | SB3 clear |
|------|---------------|-----------|--------------|----------|
| 0    | 65.9%         | 63.5%     | 32.2%        | 32.0%    |
| 1    | 63.5%         | 62.5%     | 29.0%        | 29.8%    |
| 2    | 59.0%         | 66.5%     | 24.4%        | 35.8%    |
| 3    | 60.5%         | 60.1%     | 29.2%        | 28.2%    |
| 42   | 62.1%         | 65.3%     | 30.4%        | 33.0%    |
| **mean** | **62.2%** | **63.6%** | **29.0%** | **31.8%** |
| **std**  | 2.65      | 2.48      | 2.89         | 2.93     |

**Significance test (paired t-test):**
- pocket: diff=−1.4pp, t=−0.77, **p=0.49** → not significant
- clear:  diff=−2.7pp, t=−1.21, **p=0.29** → not significant

**Conclusion: VanillaSAC ≈ SB3 SAC. Implementation validated.**

### Results · Vanilla SAC Implementation Validation — Phase 0 (2026-03-29)

VanillaSAC (custom) vs SB3 SAC — 2M steps, n_balls=1, ms=1, seeds {0,1,2,3,42}

| seed | vanilla pocket | SB3 pocket |
|------|---------------|-----------|
| 0    | 54.8%         | 49.0%     |
| 1    | 57.6%         | 58.6%     |
| 2    | 60.8%         | 50.2%     |
| 3    | 51.4%         | 57.6%     |
| 42   | 59.8%         | 61.0%     |
| **mean** | **56.9%** | **55.3%** |
| **std**  | 3.83      | 5.35      |

> random baseline: ~2.8%
> best checkpoint (peak during eval): vanilla 74.0% vs SAC 72.4%
> SAC s3 — collapse observed from best 66% to 8% late in training

**Significance test (paired t-test):**
- best pocket: diff=+1.6pp, t=0.63, **p=0.57** → not significant
- final pocket: diff=+1.6pp, t=0.54, **p=0.62** → not significant

**Conclusion: VanillaSAC ≈ SB3 SAC in Phase 0 as well. Implementation validated.**

---

## WMPredictor Experiments · Exp-16 World Model Component Development

A standalone trajectory predictor development track for the `M(s, a) → ĥ` component of the Exp-16 WM critic.
Originally planned to pretrain on SAC/random data and integrate into Exp-16 critic, but direction changed before confirming training results.

### WMPredictor v1 (2026-03-28)

`world_model/predictor.py` + `train_predictor.py`

**Comparison axes:** architecture (MLP vs LSTM) × training strategy (curriculum vs tf_ratio) × scale-up (h=256/512/1024)

- LSTM curriculum: TF ratio gradually decreasing 1→0 (over 80 epochs)
- LSTM tf_ratio: scheduled sampling approach
- scale-up: h=256 → 512 → 1024 (scaleup experiments)

### WMPredictor v2 (2026-03-29 ~ 2026-04-03)

`world_model/wm_predictor.py` + `train_wm_predictor.py` (new format)

**Changes:** Separated cue / target ball trajectory data format (data_v2/) + dual-head LSTM

```
Encoder : MLP (obs_norm, act) → h0, c0
Decoder : LSTM step input [e_emb | cue_xy | tgt_xy] (d+4)
          ├── event_head : hidden → logits (K=10)
          └── pos_head   : [hidden | event_embed] → abs cue_xy, tgt_xy
```

**Experiment scope:** lstm_hidden {256, 512} × lstm_layers {1, 2} × aug {on, off} × seeds {0,1,2,42}
**Last checkpoint:** `wmv2_enc128_256_h256_l2_emb32_s*_20260403_*`

### WMPredictor v3 (2026-09-05, architecture only — training not completed)

**Motivation:** Error accumulation when predicting abs coordinates directly in v2 + gradient interference between pos/event heads observed.

**Changes:**

| | v2 | v3 |
|---|---|---|
| pos prediction | Direct absolute coordinates | **Predict Δpos then accumulate** |
| decoder input | `[e_emb\|cue\|tgt]` d+4 | `[e_emb\|cue\|tgt\|Δcue\|Δtgt]` d+8 |
| LSTM | 1-layer | **2-layer + dropout=0.1** |
| head structure | pos_head = hidden+event_embed | **pos_head / event_head fully independent** |
| LR schedule | ReduceLROnPlateau | **OneCycleLR** (per batch) |
| event loss | CE | **CE + label_smoothing=0.1** |
| enc_hidden | [128,128] | **[128,256]** |

`run_wmv3.sh`: h=256 × lr {3e-4, 1e-3} × seeds {0,1,2}, 6 runs total planned.
**Training started and stopped on 2026-09-05. No results. Direction changed afterwards.**

---

## Markov World Model · Event-based (2026-09-05)

Pivoted from LSTM-based trajectory prediction to a **purely event (collision)-based Markov model**.

### Model Structure

```
MarkovEncoder:    (obs(16) + act(2)) → first_event (type + state)
MarkovTransition: event_state(24)    → (next_type, next_cue_state, next_tgt_state)
```

**Event state dim 24:**
- cue_xy(2) + cue_vel(2) + cue_avel(3) — post-collision velocity
- tgt_xy(2) + tgt_vel(2) + tgt_avel(3)
- type_onehot(10): ball_ball / linear_cushion / circular_cushion / ball_pocket + 6 non-coll

**MarkovTransition input expansion (final in_dim=58):**
- type_embed(32) + phys(14) + pocket_dists(12) — 6 pockets × 2 balls

### Data (data_v4)

- 55k episodes: SAC 3 variants × 10k + random 25k
- **post-collision velocity** (`agent.final.vel`) used — more direct for predicting next event
- Normalization: pos/TABLE_WH, vel/12.0 m/s, avel/300 rad/s

### 5-stage Curriculum

| Stage | Data filter | max_per_class | trans_ep | enc_ep |
|-------|-----------|--------------|---------|-------|
| 0 | first_ball_ball (SAC) | 10,000 | 200 | 100 |
| 1 | any_ball_ball | 20,000 | 75 | 40 |
| 2 | any_ball_ball + new 15k | 40,000 | 50 | 25 |
| 3 | all | 80,000 | 40 | 20 |
| 4 | all (natural distribution) | None | 30 | 15 |

### Final Results (per-class accuracy by stage)

| Stage | ball_ball | linear_cushion | circular_cushion | ball_pocket |
|-------|-----------|---------------|-----------------|------------|
| 0 | 58.8% | 50.0% | 33.6% | 60.3% |
| 1 | 64.4% | 47.8% | 46.5% | 64.2% |
| 2 | 83.6% | 42.6% | 45.4% | 64.5% |
| 3 | 79.9% | 48.3% | 49.2% | 68.8% |
| **4** | **91.2%** | **39.1%** | **53.8%** | **70.3%** |

### Limitations

1. **linear_cushion 39%**: Accounts for 78% in natural distribution, confused with ball_ball at Stage 4
2. **No independence between two balls**: cue/tgt move independently after ball_ball, but predicted as a single chain
3. **Non-causal pairs**: Learns pairs of unrelated events like ④ target pocket → ⑤ cue ball cushion
4. **Gradient discontinuity**: Next event type is discrete argmax → unsuitable for planning

→ **Pivoting to fixed-Δt continuous state model**

---

## Fixed-Δt World Model (2026-09-06)

Adopted **fixed time interval (Δt=0.05s) continuous state prediction** to resolve structural issues of event-based approach.

### Design Motivation

| Problem (event-based) | Solution (Fixed-Δt) |
|-----------------|----------------|
| Discrete type argmax → gradient cut | Continuous state → fully differentiable |
| Two balls mixed after ball_ball | Joint state (cue+tgt) at every step |
| Non-causal event pairs | Sequential by time, continuous |

### Model Structure

```
StateEncoder φ  : s(14) + pocket_dist(12) → z(128)   ← Shared with SAC Critic (planned)
Transition f    : z(128) → z'(128)
StateHead       : z'(128) → ŝ(14)
CollisionHead   : z(128) → (p_coll, type_logit[4])
```

**State s (14dim, normalized):**
- cue_x, cue_y, cue_vx, cue_vy, cue_wx, cue_wy, cue_wz (7)
- tgt_x, tgt_y, tgt_vx, tgt_vy, tgt_wx, tgt_wy, tgt_wz (7)

**Pocket distances (12dim):** 6 pockets × 2 balls, same method as Markov model.

### Data (data_fixeddt)

- 45k episodes: SAC 3 variants × 10k + random 15k
- Δt=0.05s → avg 112 steps/shot, max 220 steps
- 4.8M (s_t, s_{t+1}) pairs, collision ratio 5.7%
- Query arbitrary time-point states using `pt.interpolate_ball_states()`

### Training Improvements Ported (from Markov model)

- **Pocket distances 12dim**: Built into StateEncoder
- **LR/TB data augmentation**: Left/right and top/bottom flip at 50% probability
- **Class weights**: ball_ball×0.60 / linear×0.075 / circular×0.87 / pocket×2.46

### Results

| Metric | Value |
|-----|---|
| Collision detection accuracy | 83.1% |
| Type classification (overall) | 88.7% |
| ball_ball | **99.3%** |
| linear_cushion | **87.4%** |
| circular_cushion | **85.9%** |
| ball_pocket | **91.8%** |

linear_cushion +48.3pp, circular +32.1pp compared to event-based.

### Limitation: AR structure → rollout error accumulation

```
step t: error ε_t
step t+1: ε_t fed as input → ε_{t+1} > ε_t
at collision: velocity changes abruptly → even a small directional error causes trajectory divergence
```

Max position error of 140cm during 6-second full shot rollout (table 99×198cm).
Single-step accuracy is high, but error accumulates in continuous rollout.

→ **Pivoting to Deterministic SSM (feature/wm-ssm)**

---

## Deterministic SSM · World Model (feature/wm-ssm, in progress)

Adopted **z-space closed-loop rollout** to resolve the rollout error accumulation problem of the Fixed-Δt AR structure.

### Problem: Current AR Structure

```
Current: s_t →enc→ z_t →trans→ z_{t+1} →dec→ ŝ_{t+1} →enc→ z_{t+1}' → ...
                                                              ↑
                                                        re-encode every step (AR)
```

During rollout, ŝ is decoded to observation space then re-encoded → error accumulates.

### SSM Formulation

$$z_{t+1} = f_\theta(z_t)$$
$$s_t = g_\phi(z_t)$$
$$z_0 = \text{enc}_\psi(s_0)$$

### Closed-loop rollout training

```
s_0 →enc→ z_0 →f→ z_1 →f→ z_2 → ... →f→ z_T   (z space only)
               ↓g     ↓g              ↓g
               ŝ_1    ŝ_2             ŝ_T

Loss = Σ ||ŝ_t - s_t||²   (gradient backpropagates all the way to z_0)
```

### Critic Sharing

```
Q(s, a) = Q_head( ψ(s), a )   ← Encoder ψ shared
```

### Key Changes

| | Fixed-Δt (AR) | SSM |
|--|--------------|-----|
| rollout | Back and forth through obs space | z space only |
| Transition training | Single-step loss | Multi-step rollout loss |
| Encoder | obs→z, every step | Only once for initial z_0 |
| gradient | 1 step | T steps backprop |

Skip connection added to Transition (z_{t+1} = z_t + f(z_t)) — prevents T-step gradient vanishing.

---

### SSM Model Implementation Details

```
world_model/ssm_model.py
  SSMWorldModel
    encoder    : StateEncoder (STATE_DIM → latent_dim=128)
    transition : ResTransition  z_{t+1} = LayerNorm(z_t + MLP(z_t))
    cue_head   : CueBallHead    z → ŝ_cue (7-dim, abs)
    tgt_head   : TgtBallHead    z → ŝ_tgt (7-dim, abs)
    coll_head  : CollisionHead  z → (p_coll, type_logit)
```

**Loss:**
- `loss_cue` : MSE over all steps (cue ball always moves, so uniform)
- `loss_tgt` : MSE over all steps (GT=stationary before collision → "stay still" is learned automatically)
- `loss_coll` : BCE / log(2)  (normalized)
- `loss_type` : CE / log(4)   (normalized)
- **Kendall uncertainty weighting** (auto-activated when w_coll > 0):
  `L = exp(-σ) * L_i + σ`  — higher σ down-weights that task

**mean_err definition:** 100 episodes × T steps, average Euclidean distance of cue+target ball xy (actual cm units, TABLE_W=0.99m, TABLE_H=1.98m)

---

### v10 · Position-only Curriculum (ssm_v10_t0fix)

**Setup:** from scratch, lr=3e-4, 120 epochs, 40,781 episodes (data_fixeddt)

**3-stage curriculum (gradually increasing T):**

| Stage | rollout T | w_coll | epochs | mean_err reached |
|-------|-----------|--------|--------|--------------|
| S1[all, T=16] | 16 steps | 0 | 1–67 | **22.5cm** (best) |
| S2[all, T=32] | 32 steps | 0 | 68–101 | ~29cm |
| S3[all, T=60] | 60 steps | 0 | 102–120 | 33.9cm (final) |

**milestone:** Ep1=38.6cm → S1_best=22.5cm → S3_final=33.9cm at T=60

**breakdown @ T=60 final:** 0.5s=22.3cm | 1.0s=30.9cm | 2.0s=38.8cm | 3.0s=47.2cm

**Two bugs discovered:**

1. **t=0 loss inclusion bug**: Loss computed with seq_s[:,0:] → fails to learn that s_hat[0] should always equal s[0], distorting the loss.
   Fix: Compute loss only against `seq_s[:, 1:]` (exclude t=0 since it is the identity)

2. **tgt_mask bug** (fixed in v11):
   ```python
   # Original: tgt loss only for steps after ball-ball collision
   bb_at_t  = seq_flags & (seq_types == BB_TYPE)
   tgt_mask = bb_at_t.float().cumsum(dim=1) > 0
   n_tgt    = tgt_mask.sum()
   if n_tgt > 0:
       loss_tgt = (tgt_sq.mean(-1) * tgt_mask).sum() / n_tgt
   else:
       loss_tgt = torch.zeros(...)   # episodes with no ball-ball collision → gradient=0
   ```
   **Result:** In episodes with zero target ball gradient, model predicts arbitrary values → hallucination.
   Visualization (tgt_x_s2.png) confirms pred oscillates even when GT is stationary.

**Saved file:** `world_model/results/ssm_v10_t0fix/final.pt` (epoch 120, T=60 basis)

---

### v11 · Collision Curriculum + tgt_mask Fix (ssm_v11_coll)

**Goal:** Eliminate tgt hallucination + learn collision detection

**Key changes:**
- Remove tgt_mask entirely → `loss_tgt = tgt_sq.mean()` (uniform across all steps)
- Add collision loss (gradual w_coll ramp, Kendall weighting auto-applied)
- Add scenario curriculum (easy shots → hard shots)

**4-stage curriculum:**

| Stage | Filter | w_coll | w_type | Purpose |
|-------|------|--------|--------|------|
| S0[all, T=60, tgt-fix] | None | 0 | 0 | v10 fine-tune, normalize tgt |
| S1[n≤3, T=60, pos+coll] | n_cush≤3 | 0→1 (20 epoch ramp) | 0 | Learn collision on simple shots |
| S2[n≤5, T=60, pos+coll] | n_cush≤5 | 1 | 0 | Medium difficulty |
| S3[all, T=60, full] | None | 1 | 0→1 (20 epoch ramp) | All shots + type |

**Fine-tuning:** v10 final.pt (T=60, 33.9cm) → lr=1e-4, 120 epochs

**Early results (6 epochs):**

| Epoch | Stage | mean_err | L_cue | L_tgt |
|-------|-------|----------|-------|-------|
| 1 | S0 | 29.1cm | 0.0139 (50.9%) | 0.0134 (49.1%) |
| 2 | S0 | 28.0cm | 0.0139 (51.7%) | 0.0130 (48.3%) |
| 4 | S0 | 26.6cm | 0.0138 (51.5%) | 0.0130 (48.5%) |

**Note:** Immediately after removing tgt_mask, L_tgt accounts for 49~51% → target ball starts receiving equal gradient compared to v10.
Starting ep1 at 29.1cm from v10 final (33.9cm) → immediate effect of tgt_mask fix confirmed.

**Final results:** mean_err=30.2cm, recall=0.169 (precision~0.25)

---

### v12 · Collision-Focused Curriculum (ssm_v12_coll)

**Goal:** Improve recall from 0.169. Introduce CollisionClipDataset for focused collision event learning.

**Key changes:**
- CollisionClipDataset: clip around collision t_c to include collision in every batch
- 50:50 balanced val (has_bb / no_bb)
- recall gate-based curriculum transition condition

**Final results:** mean_err≈37cm, recall=0.251, precision≈0.25

**Limitation:** Plateau at recall 0.25. Root cause → see architecture analysis below.

---

### v13 · Pocket Context (ssm_v13_pocket, failed)

**Goal:** Attempt to improve recall by injecting pocket coordinates as constants into transition.

**Result:** recall=0.15 (actually lower than v12). Stopped at epoch 69.

**Failure reason:** Pocket positions are fixed constants, so no new information is added. Cushion collision requires "current ball position," which was not provided.

---

### v14 · Scheduled Sampling + Wall Dists (ssm_v14_ss)

**Key architecture change:**

`wall_dists_t = [x, 1-x, y, 1-y]` (cue+tgt = 8dim) injected into transition.  
Transition can observe current ball position at every rollout step.

**Scheduled Sampling (ss_ratio):**
- `ss_ratio=1.0`: GT position → wall_dists (teacher forcing)
- `ss_ratio=0.0`: decoded prediction → wall_dists (pure inference)
- Curriculum: P1→P2→P3 (position) → C1→C2→C3 (collision) → S1(0.7)→S2(0.3)→S3(0.0)→S4(+type)

**3-phase curriculum:**

| Phase | Stages | Purpose |
|------|---------|------|
| Position | P1(T=16)→P2(T=32)→P3(T=60), ss=1.0, w_coll=0 | Stabilize position first in AR structure |
| Collision | C1(clip=8)→C2(clip=20)→C3(full), ss=1.0 | Learn collision with GT wall_dists |
| SS | S1(ss=0.7)→S2(ss=0.3)→S3(ss=0.0)→S4(+type) | Gradually transition to inference |

**epochs=500, patience=15, ckpt=v12 best.pt**

**Results by stage:**

| Stage | Transition epoch | recall | prec | err | Note |
|---------|----------|--------|------|-----|------|
| P1→P2 | 55 | - | - | 85.7cm | Inference still poor (expected) |
| P2→P3 | 71 | - | - | 89.1cm | |
| P3→C1 | 87 | - | - | 89.1cm | |
| C1→C2 | 121 | 0.168 | 0.45 | 80.3cm | Below recall gate, safety valve |
| C2→C3 | 169 | 0.234 | 0.30 | 71.2cm | Below recall gate, safety valve |
| C3→S1 | 200 | 0.224 | 0.29 | 79.1cm | Below recall gate, safety valve |
| S1→S2 | 216 | 0.241 | 0.51 | 66.9cm | **Precision spikes as ss drops** |
| S2→S3 | 244 | 0.248 | 0.68 | 57.2cm | |
| S3→S4 | 289 | 0.28 | 0.78 | 37cm | |
| S4 converge | ~390 | 0.32 | 0.79 | 35.6cm | |
| **S4 final** | **500** | **0.343** | **0.805** | **34.9cm** | best=34.87cm, 0.5s=28.8/1.0s=34.1/2.0s=39.5/3.0s=43.4cm |

**Key observation:** As ss_ratio decreases in the SS stage, precision explodes from 0.3→0.805 and err drops sharply from 79→34.9cm. A paradoxical phenomenon where "noise actually trains z better."

**v14 final performance:** recall=0.343, precision=0.805, err=34.9cm (bb=49.4cm / nbb=20.4cm)

---

### Architecture Analysis — Design Principles Derived from v14

#### Why Performance Improves: Open-loop vs Closed-loop

**Pure SSM (v10~v13):**
$$z_t = f^t(z_0)$$
Information starts from $z_0$ and only decreases through transitions. Predicting position 60 steps ahead requires $z_0$ to encode all future trajectories, which is inherently limited.

**SSM + wall_dists feedback (v14):**
$$z_{t+1} = \text{LN}(z_t + \text{MLP}([z_t;\ w_t])), \quad w_t = \phi(g_{xy}(z_t))$$
External information is injected at every step. Even if z loses information, position corrects it.

Physically: **dead reckoning (pure SSM)** vs **GPS + dead reckoning (SS + feedback)**.

#### wall_dists is essentially position AR

`wall_dists = [x, 1-x, y, 1-y]` is a simple transformation of position. The essence is:
$$z_{t+1} = f(z_t,\ \hat{s}^{xy}_t) \quad \text{where } \hat{s}^{xy}_t = g_{xy}(z_t)$$

**During SS training (ss>0):** AR where true $s^{xy}_t$ (GT) is fed in.  
**Inference (ss=0):** Closed-loop where position decoded from z is fed back.

SS training acts as a bridge connecting "train as AR, rollout with only z at inference."

#### Role of z

At ss=0, z is not a simple encoder output but a **belief state updated every step**:
- Decodes position itself and feeds back as wall_dists → must be self-consistent or the loop collapses
- velocity/spin are implicitly encoded in z
- Internalizes collision history + physics dynamics

This is analogous to a Kalman filter: z ≈ posterior estimate of the current physical state.

#### Why Noise Actually Helps in the SS Stage

With GT always present (ss=1.0), z learns "position will be provided externally anyway" and gives up on encoding position (cheating). Injecting noise (ss<1.0) acts like dropout, forcing z to encode position on its own.

#### Why Recall Is Still Low → v15 motivation

wall_dists only contains position, not velocity, so the transition doesn't know "how fast is the ball approaching the wall." Collision detection requires both **position + velocity direction**.

Cushion collision condition: (near wall) AND (velocity toward wall > 0) — currently only the former is provided.

---

### v15 · AR State Feedback + Feature-level Masking (ssm_v15_ar, in progress)

#### Architecture Changes

**wall_dists(8dim) → `ar_state`(14dim, full decoded state)**

```
v14: z_{t+1} = f(z_t, wall_dists_t)   # [x,1-x,y,1-y] × 2 balls (8dim)
v15: z_{t+1} = f(z_t, ar_state_t)     # [x,y,vx,vy,wx,wy,wz] × 2 balls (14dim)
```

`ar_state_t = [g_cue(z_t); g_tgt(z_t)]` — simply concat existing decode head outputs.

transition input: 128 + 14 = **142dim** (v14: 136).

BERT random masking: `ss_ratio < 0` sentinel, `ss_b ~ U(0,1)` per batch, per-feature Bernoulli.

val loop always uses pure inference (gt_states=None, ss_ratio=0.0) → clear interpretation.

**v14 best.pt fine-tune.** transition.net.0.weight shape changed (136→142) → auto skip.

#### 19-stage curriculum

| Phase | Stages | T | ss | w_coll | w_type | Notes |
|------|---------|---|----|--------|--------|---------|
| Phase 1 | P1a~P3b (6) | 16→32→60 | 1.0→0.0 | 0 | 0 | position MSE only |
| Phase 2 | C1a~C3b (6) | 60 | 1.0→0.0 | 1.0 | 0 | clip=8→20→full, recall gate 0.35/0.50/0.60 |
| Phase 3 | R1~R2 (2) | 60 | -1.0 | 1.0 | 0→1.0 ramp | BERT masking (U(0,1)) |
| Phase 3 | R3~R7 (5) | 60 | 0.0 | 1.0 | 0.3→0.3→0.6→0.9→1.0 | pure inference, w_type ramp |

#### Stage-by-stage observations

| Event | recall | prec | err | Note |
|--------|--------|------|-----|------|
| R1/R2 entry (BERT masking) | 0.23 | 0.81 | 65cm | Masking shock — recall temporarily drops, err spikes |
| R3 entry (pure inference) | 0.257 | 0.80 | 40cm | Fast recovery |
| R5 entry (wt=0.6) | ~0.31 | 0.80 | 36cm | |
| **R7 in progress (epoch 530/600)** | **0.307** | **0.819** | **35.7cm** | In progress |

#### Key Observations

- BERT masking (R1/R2) temporarily reduced recall and spiked err from 40→65cm, but recovered quickly after R3. A disruption, not a failure.
- v15 current err (35.7cm) converging similarly to v14 final (34.9cm). recall still lower than v14 (0.343) at 0.307.
- **Root cause of recall plateau:** class imbalance (5.7% collision steps, 16.6:1 ratio) + conservative BCE threshold `p_coll>0`.
- ar_state includes angular velocity (wx,wy,wz) but only vx,vy are relevant for collision detection → noise source.
- "Position error similar even with noise" is evidence that z encodes sufficient meaning.

---

### v16 · 5-class Unified Head + BERT Masking Curriculum

#### Design Motivation

Two problems identified from v15 analysis:
1. **Inefficiency of p_coll BCE + type CE split structure**: no_coll class absent from type CE, which only computes on collision steps → class imbalance unresolved
2. **No type information in ar_state**: p_coll/type computed from z_t but not included in ar_state (14dim) → transition cannot explicitly see "whether previous step was a collision"

#### Architecture Changes

| Item | v15 | v16 |
|------|-----|-----|
| collision head | p_coll BCE(1) + type CE(4) | **Unified type CE(5)** |
| AR_DIM | 14 | **19** (type_logit 5dim added) |
| transition input | 128+14=142 | **128+19=147** |
| ar_state feedback | raw logit (14dim) | **raw logit (19dim, no softmax)** |
| Kendall weights | [state, coll, type] | **[state, type]** |
| class weight | no p_coll pos_weight | **[1.0, 4.1, 4.1, 4.1, 4.1]** (sqrt inv-freq) |

**5-class:** 0=no_coll, 1=ball_ball, 2=linear, 3=circular, 4=pocket

**Data re-map (one line in loader):** `coll_type += 1` → -1(no_coll)→0, 0(bb)→1, 1→2, 2→3, 3→4

**Why raw logit (no softmax):**  
Logit magnitude encodes uncertainty as-is. `[0.1,0.1,0.1,0.1,0.0]` (uncertain) vs `[5.0,-3,-3,-3,-3]` (confident) look similar after softmax but differ as logits. Encourages ResTransition to rely less on uncertain ar_state.

**Why sqrt inv-freq weight:**  
Exact balance (16.6×) sacrifices precision excessively. `sqrt(16.6) ≈ 4.1` promotes recall↑ while preventing excessive precision drop. A good starting point.

#### BERT Masking Curriculum

In v15, BERT was on/off (ss=-1.0 fixed). In v16, `abs(ss_ratio)` is used as the **maximum masking ratio**:

```python
# _decode_ar_state() change (1 line)
if ss_ratio < 0.0:
    max_mask = abs(ss_ratio)               # Previously: always 1.0 → now variable
    ss_b = torch.rand(1).item() * max_mask  # U(0, max_mask)
    mask = torch.rand(B, AR_DIM, device=z.device) < ss_b
    return torch.where(mask, gt_states[:, t], decoded)
```

Negative domain version of the ss curriculum (1.0→0.0). Gradually reduces noise in Phase 3.

#### 17-stage curriculum

```
# (max_cush, T, w_type, label, coll_ratio, ss_ratio)
# no gate — all stages transition only via plateau (stall counter)

Phase 1: Position MSE only (w_type=0)
  P1a  T=16  w=0.0  coll=0.00  ss= 1.0
  P1b  T=16  w=0.0  coll=0.00  ss= 0.0
  P2a  T=32  w=0.0  coll=0.00  ss= 1.0
  P2b  T=32  w=0.0  coll=0.00  ss= 0.0
  P3a  T=60  w=0.0  coll=0.00  ss= 1.0
  P3b  T=60  w=0.0  coll=0.00  ss= 0.0

Phase 2: 5-class CE, difficulty × ss
  C1a  clip=8   w=1.0  coll=0.90  ss= 1.0
  C1b  clip=8   w=1.0  coll=0.90  ss= 0.0
  C2a  clip=20  w=1.0  coll=0.70  ss= 1.0
  C2b  clip=20  w=1.0  coll=0.70  ss= 0.0
  C3a  full     w=1.0  coll=0.30  ss= 1.0
  C3b  full     w=1.0  coll=0.30  ss= 0.0

Phase 3: BERT masking curriculum → pure inference fine-tuning
  R1   full  w=1.0  coll=0.00  ss=-0.2  (U(0,0.2))
  R2   full  w=1.0  coll=0.00  ss=-0.4  (U(0,0.4))
  R3   full  w=1.0  coll=0.00  ss=-0.6  (U(0,0.6))
  R4   full  w=1.0  coll=0.00  ss=-0.8  (U(0,0.8))
  R5   full  w=1.0  coll=0.00  ss= 0.0  (pure inference)
  R6   full  w=1.0  coll=0.00  ss= 0.0  (pure inference)
```

#### Summary of Changes vs v15

| | v15 | v16 |
|--|-----|-----|
| Phase 2 training objective | collision detection (BCE) | 5-class type (CE) |
| Phase 3 BERT | on/off | gradual decrease (-1.0→-0.7→-0.4→-0.1) |
| w_type ramp | 0→0.2→0.3→0.6→0.9→1.0 | fixed at w=1.0 from Phase 2 |
| number of stages | 19 | 17 |

#### Execution

```bash
.venv/bin/python world_model/train_ssm.py \
  --out-dir world_model/results/ssm_v16_5cls \
  --ckpt world_model/results/ssm_v15_ar/best.pt \
  --epochs 600 --lr 1e-4
```

Starts after v15 completion. transition (147dim) + type_head (5cls) auto-skipped, rest reused.

#### Training Trajectory

| Stage | Start epoch | Key metrics | Note |
|-------|----------|----------|------|
| P1a [T=16,ss=1.0] | 1 | val converges rapidly | |
| P1b [T=16,ss=0.0] | 18 | val 0.152→0.047 (drops within 1 epoch) | Stabilizes immediately after ss=0 switch |
| P2a [T=32,ss=1.0] | — | — | |
| P2b [T=32,ss=0.0] | — | — | |
| P3a [T=60,ss=1.0] | — | val~0.137, err≈78cm | err high under teacher forcing |
| P3b [T=60,ss=0.0] | 165 | val 0.137→0.031, err≈32.5cm | Best: **err=32.5cm** (ep238/241/245) |
| C1a [clip=8,ss=1.0] | 245 | val 0.642, err≈78cm | Entering type loss, recall 0.86→0.22 sudden drop |
| C1b [clip=8,ss=0.0] | — | recall recovers | Pattern repeats: recall↑ in b-stage |
| C2a [clip=20,ss=1.0] | — | recall↓ | |
| C2b [clip=20,ss=0.0] | — | recall↑ | |
| C3a [full,ss=1.0] | — | recall↓ | |
| C3b [full,ss=0.0] | — | **recall≈0.46, prec≈0.47, err≈37.5cm** | Phase 2 final |
| R1 [bert=0.2] | — | err temporarily rises | |
| R2 [bert=0.4] | — | err in 40→50cm range | |
| R3 [bert=0.6] | — | err oscillates 50→65cm | |
| R4 [bert=0.8] | ~478 | err oscillates 50→63cm | best.pt=50.49cm (reset bug) |
| R5/R6 [inf] | — | (incomplete) | Results not recorded due to context limit |

#### Phase 2 Recall/Precision Pattern

Regular pattern repeats at every a(teacher forcing)→b(inference) transition in Phase 2 (C stages):

| Stage | ss | recall | prec | Interpretation |
|-------|----|--------|------|------|
| C1a | 1.0 | ~0.22 | ~0.38 | GT feedback → only path ① (type_head CE) is active, path ② (MSE) is cut |
| C1b | 0.0 | ~0.35+ | ~0.35+ | path ② restored → recall recovers sharply |
| C2a | 1.0 | ~0.24 | ~0.40 | Same pattern |
| C2b | 0.0 | ~0.38+ | ~0.40+ | |
| C3a | 1.0 | ~0.25 | ~0.40 | |
| C3b | 0.0 | **~0.46** | **~0.47** | Phase 2 best: balanced recall/prec |

**v14/v15 vs v16 comparison:**

| Version | recall | prec | err | F1 |
|------|--------|------|-----|----|
| v14 | 0.343 | 0.805 | 34.9cm | 0.48 |
| v15 | 0.318 | 0.819 | 35.4cm | 0.46 |
| v16 C3b | ~0.46 | ~0.47 | ~37.5cm | ~0.46 |
| v16 P3b | — | — | **32.5cm** | — |

- v16 recall much higher than v14/v15 → effect of 5-class CE + class weight[4.1]
- prec is lower → class imbalance not fully resolved (4:1 effective ratio)
- P3b stage err (32.5cm) is best, but lost due to reset bug when entering C stage

#### Key Observations

**1. Teacher forcing and recall degradation mechanism**

Root cause of recall dropping to ~0.22 in ss=1.0 (teacher forcing) stage:
- CE loss (path ①) is active and directly trains type_head
- However, class imbalance: 0.94×1.0 (no_coll) vs 0.06×4.1 (collision) → effective ratio ≈ 4:1, no_coll dominates
- `_decode_ar_state(ss=1.0)` computes `decoded` but returns `gt_states[:,t]` → decoded is **completely detached** from the loss graph
- Therefore path ② (`state_MSE → transition(z, ar_state) → softmax(type_head(z)) → type_head params`) is cut
- Path ② especially provides physics-based signal penalizing false negatives (failed collision prediction → wrong state → large MSE)
- Upon switching to ss=0.0 (inference), path ② is restored → recall immediately recovers

**2. best_mean_err reset bug**

In `train_ssm.py`, `best_mean_err = float("inf")` is reset at every stage transition.  
P3b best point (32.5cm) is reset upon entering C1a → best.pt overwritten by 78cm model from C1a ep1.  
R4 stage best.pt = 50.49cm (much worse than actual P3b best).  
→ **Needs fix in v17:** Separate `global_best_err` to track across the entire training.

**3. BERT masking disruption**

err oscillates in the 50→65cm range during R3 (max_mask=0.6) / R4 (max_mask=0.8).  
Expected to recover in R5/R6 (pure inference), similar to the P3a→P3b pattern (P3a: err≈78cm → P3b: err≈32.5cm).  
Whether gradual BERT (0.2→0.4→0.6→0.8) reduces disruption compared to v15's on/off approach needs to be confirmed from R5/R6 results.

#### v17 Direction: Soft Labeling

Solutions for the path ② disconnection problem under teacher forcing:

| Option | Method | Characteristics |
|------|------|------|
| A | CE with label_smoothing=0.1 | Prevents type_head overconfidence, path ② still disconnected |
| B | Use soft targets instead of one-hot for type dims in gt_ar | Partial fix |
| C (recommended) | type dims: `α·one_hot + (1-α)·softmax(type_head(z_t))` | Directly restores path ②, α linked to ss_ratio |

Option C: Implemented as `gt_ar[:, :, 14:] = α × one_hot + (1-α) × softmax(type_head(z_t.detach()))`.  
If α=ss_ratio, the teacher forcing degree and type feedback ratio are naturally linked.

#### v16 curriculum final results

epoch 600 complete. best checkpoint: **epoch 588, err=34.2cm**, recall=0.507, prec=0.492, type_acc=0.896.

| Time axis | err |
|--------|-----|
| 0.5s   | 26.7cm |
| 1.0s   | 33.5cm |
| 2.0s   | 39.6cm |
| 3.0s   | 43.7cm |

**Saved file:** `world_model/results/ssm_v16_5cls/best.pt` (epoch 588, err=34.2cm)

---

### v16 no-curriculum ablation

#### Design Motivation

Analyzing why v16 curriculum was worse than v15 (35.4cm):
- The 17-stage curriculum starts with Phase 1 (position only), repeatedly re-introducing teacher forcing in Phase 2
- Phase 1 of the curriculum causes forgetting of collision prediction already learned from the v15 fine-tuning starting point
- Disruption in R3/R4 BERT stages worsened err to 50→65cm, significantly delaying convergence
- **Hypothesis: Without curriculum, fixed ss=0.0 + only random T~Uniform(10,60) will be better**

#### Setup

```python
# train_ssm_nocurr.py
MODE     = "no-curriculum"
ss_ratio = 0.0          # fixed — no teacher forcing
T        = Uniform(10,60)  # random T each epoch
w_type   = 1.0          # fixed — no curriculum weight ramp
coll_ratio = 0.0        # no oversampling
```

Initialization: `world_model/results/ssm_v15_ar/best.pt` (v15 fine-tune).  
v16 architecture (AR_DIM=19, 5-class head) same, transition (147dim) / type_head (5cls) auto-skipped.

#### Training Trajectory

| epoch | err | recall | prec | Note |
|-------|-----|--------|------|------|
| 1 | 49.4cm | 0.275 | 0.344 | Initial (after v16 skip) |
| 40 | 38.9cm | 0.438 | 0.445 | |
| 140 | 35.7cm | 0.508 | 0.471 | Approaching v16 curriculum final (34.2cm) |
| 160 | 34.6cm | 0.520 | 0.491 | |
| 200 | **34.1cm** | 0.519 | 0.523 | Surpasses v16 curriculum final |
| 310 | 32.5cm | 0.530 | 0.508 | Matches v16 P3b best |
| 450 | **31.0cm** | 0.549 | 0.540 | Best checkpoint |
| 459 | 31.0cm | 0.549 | 0.540 | Training ended (459/600 epoch) |

| Time axis | err (best, ep450) |
|--------|-------------------|
| 0.5s   | 21.9cm |
| 1.0s   | 29.5cm |
| 2.0s   | 36.1cm |
| 3.0s   | 41.3cm |

**Saved file:** `world_model/results/ssm_v16_nocurr/best.pt` (epoch 450, err=31.0cm)

#### v16 curriculum vs no-curriculum comparison

| | v16 curriculum | v16 no-curriculum |
|--|----------------|-------------------|
| Initialization | v15 best.pt | v15 best.pt |
| Architecture | Same | Same |
| ss_ratio | Curriculum schedule | Fixed 0.0 |
| T | Curriculum schedule | Uniform(10,60) |
| best epoch | 588 / 600 | 450 / 459 |
| best err | 34.2cm | **31.0cm** |
| recall | 0.507 | **0.549** |
| prec | 0.492 | **0.540** |
| type_acc | 0.896 | **0.905** |

#### Key Conclusions

**Curriculum is actually harmful during fine-tuning.**

- no-curriculum already surpasses v16 curriculum final (34.2cm) at epoch 200 with 34.1cm
- Curriculum Phase 1 (position-only) causes forgetting of collision prediction already learned in v15
- BERT masking disruption (R3/R4: 50→65cm) significantly delays convergence
- **Curriculum is unnecessary for fine-tuning, not scratch training**

Cases where curriculum is meaningful:
- v10→v12: Training directly at T=60 from scratch fails to converge → gradually increasing T from short values is essential
- v15: Phase 1 (position only) → Phase 2 (collision) → BERT structure contributed to stable convergence from scratch

#### Training Infrastructure Improvements

Optimizations introduced in this experiment:

| Improvement | Content | Effect |
|-----------|------|------|
| `_eval_batched` | 500 episodes sequentially → 1-batch forward | ~30% eval speed improvement |
| eval interval 10 epochs | eval every epoch → every 10 epochs | ~90% eval cost reduction |
| 60k item cap | 264 batches/epoch → 117 batches | ~55% train speed improvement |
| pickle cache | npz (with padding, 554MB OOM) → pickle (~200MB, no padding) | ~2-3s loading from 2nd run |

Final speed: ~47s/epoch (2.5× improvement over initial ~2min/epoch), total training ~6 hours.

---

### Pocket Prediction Performance Evaluation — RL Integration Feasibility

#### Evaluation Purpose

Determining whether SSM v16 nocurr best (epoch 530, err=30.5cm) can be used as Q-target supervision.  
Specifically: checking whether precision/recall of `type=4(pocket)` prediction in WM rollout is usable as RL labels.

#### 5-class per-class evaluation (val 500 episodes, T=60 rollout)

```
confusion matrix (rows=GT, cols=Pred)

             no_coll  ball_ball     linear   circular     pocket
  no_coll     23,759         31      1,258          9          0
ball_ball         61        202          9          0          0
   linear        994          4      1,160          4          0
 circular         81          0         78          8          0
   pocket         34          0         28          0          0
```

| class | support | recall | prec | F1 |
|-------|---------|--------|------|----|
| no_coll | 25,057 | 0.948 | 0.953 | 0.951 |
| ball_ball | 272 | 0.743 | 0.852 | 0.794 |
| linear | 2,162 | 0.537 | 0.458 | 0.494 |
| circular | 167 | 0.048 | 0.381 | 0.085 |
| **pocket** | **62** | **0.000** | **—** | **0.000** |

**episode-level pocket detection** (whether a pocket event exists within the episode):
- GT has pocket: 62 / 462 eps (13.4%)
- PR has pocket: **0** / 462 eps (0.0%)
- TP=0, FP=0, FN=62 → recall=0.000, prec=0.000

#### Root Cause Analysis

Pocket step ratio in val data:
```
no_coll   : 25,057 steps  (92.7%)
ball_ball :    272 steps  ( 1.0%)
linear    :  2,162 steps  ( 8.0%)
circular  :    167 steps  ( 0.6%)
pocket    :     62 steps  ( 0.2%)  ← linear:pocket = 35:1
```

Corrected with class_weight=4.1, but `linear:pocket = 35:1` ratio is too extreme.  
From the model's perspective, not predicting pocket minimizes CE loss → pocket class collapse.

#### Conclusion: Q-target Not Feasible with Current WM

Since WM never predicts pocket at all, it cannot determine "whether this action will pocket."  
Using this as Q-target would make all WM-predicted Q = 0 (no pocket) → useless for RL training.  
**Improving pocket prediction is a prerequisite before RL integration.**

---

### v17 · Pocket Focal Fine-tune (ssm_v17_pocket)

#### Design Motivation

pocket recall=0.000 confirmed in v16 nocurr best (err=30.5cm) → prerequisite fix before RL integration.  
pocket steps 0.2% (62/27,520) → class collapse inevitable with class_weight=4.1.

**Changes:**
- `class_weight[4]`: 4.1 → **20.0** (pocket 5× additional boost)
- `focal_gamma=2.0`: `L = -(1-p_t)^2 · log(p_t)` — down-weights easy samples (no_coll)
- v16 nocurr best.pt fine-tune, lr=5e-5, 200 epochs

#### Results (epoch 200 complete)

| Metric | v16 nocurr | v17 focal |
|------|------------|-----------|
| state err | 30.5cm | 30.7cm |
| coll recall | 0.549 | 0.572 |
| coll prec | 0.540 | 0.541 |
| pocket step recall | 0.000 | **0.109** |
| pocket episode recall | 0.000 | **0.250** |
| pocket episode prec | 0.000 | 0.320 |

**per-class detailed (val 458 eps, T=60):**

| class | recall | prec |
|-------|--------|------|
| no_coll | 0.947 | 0.953 |
| ball_ball | 0.760 | 0.787 |
| linear | 0.537 | 0.473 |
| circular | 0.076 | 0.310 |
| **pocket** | **0.109** | **0.123** |

episode-level: TP=16, FP=34, FN=48 → recall=0.250, prec=0.320.

**Conclusion:** pocket recall improved from 0.000→0.250 with focal loss + weight=20.  
However, below roadmap target (recall≥0.5, prec≥0.4). FP=34 > TP=16 — too many false alarms.  
Step-by-step error accumulation fundamentally limits pocket prediction.

**Saved file:** `world_model/results/ssm_v17_pocket/best.pt` (epoch 200, err=30.7cm)

---

### v18 · Pure Latent Dynamics (no AR state)

#### Design Motivation

Up to v17, ar_state (decoded state, 19dim) feedback was present in the transition:

```
v17: z_{t+1} = LN(z_t + MLP(cat(z_t, ar_state_t)))   # 147dim input
```

`ar_state_t = [g_cue(z_t); g_tgt(z_t); softmax(h_type(z_t))]` feeds values decoded from z_t back in as a shortcut. z_t doesn't need to encode all information, making the representation "lazy."

Removing this forces **z to encode all physical information on its own**:

```
v18: z_{t+1} = LN(z_t + MLP(z_t))   # only 128dim input
```

**Formulation:**

$$z_0 = \psi(s_0) \in \mathbb{R}^{128}$$

$$z_{t+1} = \text{LN}(z_t + f_\theta(z_t)), \quad f_\theta : \mathbb{R}^{128} \xrightarrow{\text{SiLU}} \mathbb{R}^{256} \xrightarrow{\text{SiLU}} \mathbb{R}^{256} \to \mathbb{R}^{128}$$

$$\hat{s}^{\text{cue}}_t = g_{\text{cue}}(z_t), \quad \hat{s}^{\text{tgt}}_t = g_{\text{tgt}}(z_t), \quad \hat{y}^{\text{type}}_t = h_{\text{type}}(z_t)$$

#### Architecture Comparison

| | v17 | v18 |
|--|-----|-----|
| transition input | z(128) + ar_state(19) = 147 | z(128) only |
| ar_state feedback | Present (decoded shortcut) | **None** |
| teacher forcing possible | Yes (ar_state = GT) | Not needed |
| role of z | Current state encoding | **Self-contained physics representation** |
| parameter count | 258,195 | 253,331 |

#### v17→v18 weight transfer

transition.net.0.weight: (256, 147) → (256, 128) — shape changed, skip.  
encoder + decoder heads (26 total): loaded directly from v17 best.pt.

#### v18-pretrained Setup

- v17 best.pt fine-tune (encoder/decoder loaded, transition random init)
- lr=1e-4, 400 epochs, pocket_weight=20, focal_gamma=2.0
- T~Uniform(10,60), no curriculum

#### v18-pretrained Training Progress

| epoch | err | coll recall | pocket ep. recall | Note |
|-------|-----|-------------|-------------------|------|
| 1 | 50.8cm | 0.284 | — | transition random init |
| 10 | 47.0cm | 0.398 | — | |
| 44 | 38.7cm | 0.440 | — | |
| 180 | 33.6cm | 0.509 | **0.234** | mid-run pocket eval |
| 252 | 32.6cm | 0.517 | — | |
| **400 (final)** | **31.6cm** | **0.549** | **0.312** | best ckpt |

Checkpoint: `world_model/results/ssm_v18/best.pt`

---

### v18-scratch · Pure Latent from Random Init

**Question**: Does v17 pretraining actually matter, or can v18 train from random init?

#### Setup

- Random init (no v17 weights loaded)
- Same hyperparams: lr=1e-4, 400 epochs, pocket_weight=20, focal_gamma=2.0, label_smoothing=0.0
- T~Uniform(10,60)

#### Results

| epoch | err | coll recall | Note |
|-------|-----|-------------|------|
| **400 (final)** | **32.7cm** | **0.512** | best ckpt=32.6cm |

Checkpoint: `world_model/results/ssm_v18_scratch/best.pt`

#### v18 Pretraining Comparison

| | v18-pretrained | v18-scratch |
|--|----------------|-------------|
| err (final) | **31.6cm** | 32.7cm |
| coll recall | **0.549** | 0.512 |
| ep. pocket recall | **0.312** | — |
| pretrain benefit | +1.1cm / +0.037 recall | — |

**Finding**: v17 pretraining yields only ~1cm improvement in state error and minimal recall gain. **The model trains well from scratch**, confirming that architectural choices (focal loss, pocket weighting) dominate over initialization. GNN rewrite can start from random init.

---

### GNN 2-ball · Set-based Multi-Agent World Model

**Question**: Does a GNN-based set architecture (shared-weight nodes, message passing) outperform the flat SSM on 2-ball prediction?

**Setup**: `world_model/gnn/train_gnn.py`, 400 epochs, lr=1e-4, batch=512, pocket_weight=20.0, focal_gamma=2.0, label_smoothing=0.0. Same data and val split as v18 for direct comparison. Random init (no pretraining). Checkpoint: `world_model/results/gnn_2ball_det/best.pt`

**Architecture**: Each ball is a node (BALL_DIM=7 + pocket_dists + is_cue). Message passing per rollout step: BallBallMsg (j→i, sum-aggregated) + BallPocketMsg (pocket→i, sum-aggregated) → BallUpdate (LN residual) → BallTransition. TypeHead (mean-pool → 5-class). 373,708 params (vs v18: 253,331).

#### Final Results (epoch 400)

| | GNN 2-ball | SSM v18 scratch | SSM v18 no-AR |
|--|:-----------:|:---------------:|:-------------:|
| **mean err (best)** | **32.4cm** | 32.6cm | 31.6cm |
| bb err | 46.8cm | 46.1cm | **43.7cm** |
| no-bb err | **16.5cm** | 17.7cm | 18.2cm |
| coll recall | **0.560** | 0.512 | 0.530 |
| coll precision | **0.508** | 0.456 | 0.493 |
| type_acc | **0.898** | 0.886 | 0.895 |
| 0.5s err | 24.7cm | 23.5cm | **21.2cm** |
| 1.0s err | 30.6cm | 30.9cm | **29.5cm** |
| 2.0s err | **36.7cm** | 38.2cm | 37.4cm |
| 3.0s err | **41.2cm** | 42.6cm | 42.8cm |

#### Key Findings

1. **GNN beats SSM scratch** (32.4 vs 32.6cm) — message passing provides a meaningful inductive bias even for 2-ball.
2. **GNN trails SSM no-AR by 0.8cm** — no-AR's MDN-free architecture still has an edge on short-horizon (0.5s, 1.0s), suggesting the deterministic GNN transition is the bottleneck.
3. **GNN collision detection is strongest** — recall 0.560 and precision 0.508 both exceed both SSM baselines. The explicit ball-ball edge features (Δpos, Δvel, dist) directly encode collision geometry.
4. **GNN long-horizon advantage** — 2.0s and 3.0s errors are best among all three. Message passing compounds well across rollout steps.
5. **GNN no-bb err is lowest** (16.5cm) — shared-weight node encoding is more data-efficient on simple trajectories.

**Conclusion**: GNN architecture is validated as a 2-ball baseline. Short-horizon gap vs no-AR points to the deterministic `BallTransition` MLP as the next bottleneck — exactly the motivation for SPR-MDN in step ③.

#### Key Findings from v16–v18 Series

1. **ar_state irrelevant**: removal gave +1.9cm error — focal loss + class weights drove v16→v17 improvement, not history tracking. Markov property holds with full [pos+vel+spin] state.
2. **Pretraining marginal**: scratch vs. pretrained ≈ 1cm difference. Architecture is the bottleneck, not initialization.
3. **Pocket recall plateau (~0.31)**: root cause is step-by-step chaos accumulation, not loss function or architecture of the 2-ball SSM. Accepted as-is; MDN mixture transition and BYOL addressed in GNN rewrite.
4. **No GRU needed**: validated by v17→v18 experiment.

#### Architectural Lessons for GNN Rewrite

1. **No GRU**: Markov property confirmed; message passing handles inter-ball dependencies
2. **SPR-MDN self-prediction** (replaces MSE + RSSM-lite): training uses own predicted latent as the next step input (not GT teacher forcing) — same condition as planning. EMA target encoder + stop-gradient (BYOL mechanism) provides stable self-prediction anchor. MDN (K=5 mixture) replaces SPR's deterministic predictor to handle chaotic bifurcations. Lineage: BYOL → SPR → SPR-MDN [ours].
   - NLL target: `z̄_{h+1} = sg(Enc_φ'(s_{t+h+1}))` — EMA-encoded GT, stable across gradient steps
   - Reconstruction `L_recon` grounds z to real ball states; closes the "NLL conspiracy" failure mode (encoder and transition minimizing NLL in a degenerate z-space)
4. **Label smoothing**: apply from the start (`--label-smoothing 0.1`)

---

### SPR-MDN · Self-Predictive MDN on v18 no-AR (idea validation)

**Hypothesis**: SPR-style self-chaining + MDN transition closes the short-horizon gap of the deterministic SSM (v18 no-AR: 0.5s=21.2cm / 1.0s=29.5cm) by removing teacher-forcing bias and modeling chaotic bifurcations with a mixture distribution.

**Why v18 no-AR as base (not GNN)**: Idea validation first — v18 no-AR is lighter (253K params vs 373K GNN) and the simplest architecture that eliminates the known irrelevant factors (GRU, pretraining). If SPR-MDN helps here, it will help GNN too. Avoids confounding architecture changes with training objective changes.

**Training from scratch** (random init).

#### Architecture changes vs v18 no-AR

| Component | v18 no-AR | SPR-MDN |
|-----------|-----------|---------|
| Transition | `z' = LN(z + MLP(z))` deterministic | `MixtureHead(z, ã) → (π, μ, σ)` K=5 MDN |
| Loss (pos) | MSE(ŝ, s_GT) | NLL against EMA-encoded target `z̄` |
| Training input | GT s_{t+h} at every step (teacher forcing) | own sampled ẑ_{h} for h≥1 (self-chaining) |
| EMA encoder | — | φ' ← τφ' + (1−τ)φ, stop-gradient on target |
| L_recon | — | `\|\|Dec(ẑ_h) − s_{t+h}\|\|²` at every step |
| λ weighting | fixed log_sigma (cue/tgt) | temperature taming (learnable log_σ_NLL, log_σ_recon) |
| Action | — | a_t at h=0, zero-padded for h≥1 |

#### Loss

```
L_NLL   = Σ_h  -log Σ_k π_k · N(z̄_{h+1} ; μ_k, diag(σ_k²))
L_recon = Σ_h  ||Dec_ψ(ẑ_h) - s_{t+h}||²
L_type  = CrossEntropy(type_logit, collision_label)  [focal, pocket_w=20]

L_total = exp(-σ_NLL)·L_NLL + σ_NLL + exp(-σ_recon)·L_recon + σ_recon + L_type
```

(σ_NLL, σ_recon: learnable scalars, initialized to 0)

#### Baseline for comparison

| Model | mean err | 0.5s | 1.0s | recall |
|-------|--------:|-----:|-----:|-------:|
| SSM v18 no-AR | 31.6cm | 21.2cm | 29.5cm | 0.530 |
| GNN 2-ball | 32.4cm | 24.7cm | 30.6cm | 0.560 |

**Success criterion**: SPR-MDN mean err ≤ 30cm OR short-horizon (0.5s ≤ 20cm, 1.0s ≤ 28cm).

---

## SPR-MDN Full Results

### SSM v18_longrun (24.8cm)

**문제**: SPRDataset 사용하더라도 400ep은 부족  
**해결**: 2000ep + patience=10 early stop  
**결과**: 24.8cm (0.5s=14.4/1.0s=22.6/2.0s=30.6/3.0s=37.2)

---

### SPR-MDN z-space 실험 (v17~v26)

#### 문제 1: truncated-BPTT 버그 (v9~v16 전부 무효)
predictor chain에 stop-grad → encoder gradient 완전 차단. v9~v16 실험 결과 신뢰 불가.  
**해결**: full-BPTT 적용 (v17) → 기준선 33.9cm 확립

#### 문제 2: SPR bootstrap 없음 (v17, 33.9cm)
latent prediction loss 없이 reconstruction만 → z-space 구조화 미흡  
**해결**: SPR L2 loss lam=0.01 추가 + SPRDataset 43k ep (v18a) → 28.3cm  
**검증**: lam=0 ablation(v18b) = 35.4cm → bootstrap 기여 7.1cm 확인

#### 문제 3: NLL이 L2보다 8cm 나쁨 (v19~v25)
K=5 NLL=39.9cm, K=1 NLL=41.4cm vs K=1 L2=28.3cm  
**원인**: `|∂L_nll/∂enc|` vs `|∂L_recon/∂enc|` = 3,363x 불균형 → NLL이 encoder 압도

- **시도 1 — EncoderLN** (v20a): gradient 안정화 → 1.8cm 비용만, 해결 안 됨
- **시도 2 — Kendall warmstart** (v21b/v22b): σ 사전 측정 후 warm-start → K=5 32.6cm, 여전히 L2에 못 미침
- **시도 3 — GradNorm** (v25): w_nll → 0.002 (NLL 신호 소멸), 최선 43.2cm. 실패
- **결론**: z-space NLL로 L2 기준선 돌파 불가. gradient 불균형이 구조적

#### 문제 4: direct state-space 예측 열위 (mdn_state_ewta)
가설: z 대신 직접 state 예측하면 encoder 오염 없지 않을까?  
결과: K=1 L2=39.7cm, K=5 EWTA=49.0cm → z-space(28.3cm) 대비 8~18cm 나쁨  
**결론**: encoder disentanglement + SPR bootstrap이 핵심. 방향 기각.

| 버전 | 변경 | err (3s) |
|------|------|----------|
| v17 | full-BPTT fix | 33.9cm |
| v18a | SPR lam=0.01 + SPRDataset | 28.3cm |
| v18b | lam=0 ablation | 35.4cm |
| v19 | K=5 NLL + EncoderLN | 39.9cm |
| v20a | K=1 L2 + EncoderLN | 32.9cm |
| v21b | K=1 NLL Kendall warmstart | 38.7cm |
| v22b | K=5 NLL Kendall warmstart | 32.6cm |
| v25 | GradNorm | 43.2cm |
| **v26_p1** | SPRDataset scratch 1190ep | **25.1cm** |

---

### SMDN + Segment 실험 (v27~v33)

#### 문제 5: per-step MDN collapse (v28_smdn, v28_anneal)
K=5 MDN 전 컴포넌트가 동일 평균 수렴 (mu_spread=0.001, pi=0.201 완전 균등)  
**원인**: EWTA + entropy_reg 조합이 분리 인센티브 소멸. 0.05s step이 너무 결정론적  
**시도**: entropy annealing β=0.05→0.0 (v28_anneal) → collapse 여전  
**결론**: per-step MDN 근본 부적합. 이벤트 경계에서만 MDN 필요

#### 문제 6: segment chaining covariate shift (v33_segment)
단일 세그먼트 예측 4.3cm이지만 연속 rollout 시 발산  
**원인**: 학습 시 항상 GT 초기 상태로 encoder 초기화. 실제 rollout에서는 이전 예측 끝점이 다음 시작점 → 분포 불일치 누적  
**시도 1**: SS noise injection (v33_ss, ep16 중단) — 미해결  
**시도 2**: Bengio-style SS on transition input (v34_ss, 진행 중)  
  - `ss_prob=1.0` (teacher forcing) → `0.0` (free running) over `ss_warmup` epochs  
  - v33은 처음부터 free running (ss=0.0). v34는 teacher forcing에서 시작해 점진적으로 낮춤  
  - ss_prob > 0이면 GT `seg_s[:, t+1]`을 transition 입력으로 사용, 0이면 `s_hat`  
  - gradient: use_gt=True → GT(상수), use_gt=False → s_hat (BPTT 통과) → ss 감소할수록 BPTT chain 자연히 길어짐

| 버전 | 변경 | err |
|------|------|-----|
| v27_p2 | 2-phase MDN | 24.9cm |
| v28_smdn | SMDN K=5 DT=0.05 (구 metric†) | 22.7cm |
| v30_obs_b | obs-space | 23.8cm |
| v31_bounce | bounce event 기반 | 34.2cm ↑ (악화) |
| **v33_segment** | segment 단위 (충돌~충돌 구간) | **4.3cm** (단일 세그먼트, chaining 미검증) |
| v34_ss | Bengio SS (ss=1.0→0.0) on transition input | TBD |

† 구 metric: len≥T_MAX+1 에피소드만 포함, 고정 T_MAX 스텝 평균. 아래 공정 비교와 수치 직접 비교 불가.

#### v28 DT 공정 비교 (2026-09-26)

**metric 수정**: `_eval_err`를 전수 에피소드 포함 + 에피소드별 실제 길이 정규화로 변경.
- 변경 전: `len >= T_MAX+1` 필터, 고정 T_MAX 스텝 평균
- 변경 후: `len >= 2` (사실상 전수), `T = min(len-1, T_MAX)` per-episode 평균
- 두 DT 모두 동일 metric 적용 → 공정 비교 가능

| 버전 | DT | epoch | mean_err | bb | nbb |
|------|-----|-------|----------|----|-----|
| v28_dt05_3s | 0.05s | ~1750 | **16.3cm** | ~28cm | ~5.6cm |
| v28_dt01_3s | 0.01s | ~1350 | **15.4cm** | ~32cm | ~5.4cm |

- DT=0.01이 0.9cm 앞섬. 물리 해상도 우위 (충돌 이벤트 5배 세밀) + per-step 예측 변화량 작음
- 두 모델 모두 학습 종료 시 spread=0.000 (MDN collapse) — K=5 컴포넌트가 동일 평균으로 수렴
- bb 오차가 nbb 대비 5~6배 — 공 충돌 케이스가 여전히 어려움

---

## Experiment Log

Detailed observation records.

---

### Exp-01 · Phase 0 single-ball benchmark

**Setup:** SAC/PPO/TQC, 1M steps, seeds {0,1,2}, n_balls=1, legacy batch

**Observations:**
- Off-policy (SAC/TQC) dominates overwhelmingly at Horizon=1. PPO's GAE degenerates to REINFORCE at a single transition.
- TQC seed variance very large (±27pp) — distribution estimation unstable. SAC best_model vs final gap also large (late-stage collapse).

**Cause of performance gap on reproduction (2026-03 ablation, branch: exp/phase0-placement-ablation):**

| Condition | Pocket% | Δ |
|------|---------|---|
| Legacy batch + no scratch (full original reproduction) | **81.4%** | baseline |
| Current batch + no scratch | 50.0% | −31pp |
| Current batch + scratch | 42.4% | −39pp |

- **Cause ①** scratch penalty: absent in original. ~26% of random actions are scratch → expected reward +0.026 → sign reversed to −0.099 → SAC learns to avoid scratch instead of pocketing. Fix: `simulator.py` `if scratch and n_balls > 1`
- **Cause ②** ball placement expansion (commit 89431bd): target y [0.6,0.9] → [0.30,0.85]. Intentional change when integrating n_balls=3, increasing difficulty.

---

### Exp-02 · Phase 1a multi-ball (ms=∞)

**Setup:** SAC, 1M, seed=42, n_balls=3, ms=15

pocket 98.3% / clear 95.8% — random also 40%. "Just keep shooting and they'll go in" strategy. Need to reduce horizon → Exp-03.

---

### Exp-03 · Phase 1a (ms=5)

**Setup:** SAC, 1M, seed=42, n_balls=3, ms=5

pocket 60.7% / clear 29.4%. 3 balls in 5 shots → avg 0.6 pocket per shot needed. Transfer experiment baseline.

---

### Exp-04 · Transfer A — zero-shot

**Method:** `ObsCollapseWrapper` reduces 23-dim → 16-dim. Each step, nearest unpocketed ball → "the ball". Phase 0 pretrained model (seed=0, 81.4%) used as-is.

63.6% / 31.4% — exceeds Exp-03 without additional training. Confirms direct transfer of aiming skill. However, other ball positions are unknown so interference avoidance is not possible.

---

### Exp-05 · Transfer B — warm-start

**Method:** Create new 23-dim SAC, copy n_balls=1 weights to shared neurons. ball2/ball3 neurons initialized to 0 then full fine-tune.

61.5% / 30.4%. Higher reward initially then gradual decline — classic weight dilution pattern. **zero-shot (Exp-04) is better with 0 training time.** Direct transfer dominates entire warm-start fine-tuning.

---

### Exp-06 · Progressive reward shaping

**Setup:** SAC, 1M, seed=42, ms=5, sp=0.1×step, tp=1.0

Change: fixed step penalty (−0.01) → progressive (step i: −0.1×i). truncation penalty none → −1.0. Reward difference between 3-step vs 5-step clear: 0.2 → 0.9.

63.9% / 33.2%, ep_len=4.48. ep_len distribution: step5 80.9% — **ep_len shortening failed**.

Root cause: step5 expected value = −0.5 + 1.0×30% ≈ −0.2. Still better than truncation (−1.0), so agent keeps consuming steps.

> Rerun in progress (multi-seed 0/1/2, 1M)

---

### Exp-07 · clear_bonus=2.0 + SAC vs TQC

**Setup:** SAC·TQC, 1M, seed=42, ms=5, sp=0.1 flat, tp=1.0, cb=2.0

SAC 62.0% / TQC 49.7%. No change in ep_len. clear_bonus also fails to shorten ep_len. TQC single-seed instability pattern reproduced.

---

### Exp-08 · shots_taken obs ablation

**Setup:** SAC, 1M, seed=42, ms=5

Added shots_taken (step count / max_steps) to obs. No change in ep_len — optimal action in billiards depends only on ball positions, not step count. shots_taken makes MDP complete but is uninformative in billiards.

08b: gs=10 (UTD=1.0) → Q-value overestimation cascade worsened in SAC.

---

### Exp-09 · ms × pp ablation grid

**Setup:** SAC, 1M, seed=42, sp=0.1, tp=1.0

pp shows no meaningful improvement at any ms:
- ms=4: pp=✓ is −3.9pp pocket — progressive penalty accumulation (−1.0) cancels out pocket reward (+1.0).
- ms=3: ep_len/ms = 99.3% — task structure inherently consumes all steps.

**pp discarded. ms=3 is the Phase 2 frontier.**

---

### Exp-10 · Phase 2 ms=3 algorithm benchmark

**Setup:** SAC/TQC/PPO × 3 seeds, ms=3, sp=0.1, tp=1.0, 2M

SAC s42: eval crash. PPO s1/s42: files lost, excluded.

- **SAC clear 8.4% reproduction confirmed.** Low variance across seeds — training stable.
- **TQC 2.0%:** Top quantile drop overconservatism backfires with sparse reward.
- **PPO ~0%:** 3-step credit assignment extremely noisy. **Phase 2 baseline = SAC.**

---

### Exp-11 · Curriculum ms=5→4→3

**Setup:** SAC, seed=42, 1M+500k+500k=2M, sp=0.1, tp=1.0

+1.3pp pocket / +2.0pp clear vs scratch at same 2M. Stage 3 (500k) alone is competitive with scratch 2M. ms=2 extension: double pocket required → absence of learning signal, failure confirmed as expected.

---

### Exp-12 · abs_angle ❌

**Setup:** SAC, ms=3, sp=0.1, tp=1.0, 3 seeds × 5M (seed=0/42 crash, seed=1 only complete)

seed=1 5M: 37.8% / 6.4% — lower than delta 2M (41.7%/8.4%). Falls behind despite 2.5× more steps. Missing inductive bias (delta=0 → aim directly at ball) explodes the search space.

**abs_angle discarded. delta_angle kept.**

---

### Exp-13 · Phase 0 reward shaping & steps scaling

**Setup:** SAC, n_balls=1, current placement, seed=42

**proximity reward (1M):** α ∈ {0.05, 0.1, 0.3, 0.5} all below baseline (50%).
Ball final position after cushion reflection ≠ action outcome → post-shot distance is noise. Sparse +1/0 is a cleaner signal.

**steps scaling:**

| Steps | Pocket% | Efficiency |
|-------|---------|------|
| 1M | 50.0% | — |
| 2M | 56.2% | 6.2pp/1M |
| 5M | 65.8% | 3.2pp/1M |

steps ∝ performance. Diminishing returns beginning. Due to the wide coverage space of current placement (ball y range 3×), it is a sample complexity problem that simply requires more steps.

---

## R-SSM World Model (Event-driven GNN+SSM)

**아키텍처**: 충돌 이벤트 단위로 동작하는 Relational SSM. 이벤트마다 GNN으로 공 간 상호작용 모델링, persistent latent h per ball.
**입력**: pre-collision rvw(pos, vel, avel) → **출력**: Δvel+Δavel (5D delta)
**파일**: `world_model/rssm_model.py`, `world_model/train_rssm.py`, `world_model/rssm_dataset.py`

### v1 (rssm_v1)

| 항목 | 값 |
|------|-----|
| 데이터 | on-the-fly random 시뮬, 2000 train / 400 val |
| 데이터 분포 | random 정책 → pocket ~0% |
| weight_decay | 1e-4 |
| Loss | raw MSE(vel) + 0.3 × CE(type), Kendall on |
| val RMSE | **12.29** |
| type_acc | 0.72 |
| 문제점 | 데이터 부족, pocket 전무, train/val gap 1.86× |

### v2 (rssm_v2)

| 항목 | 값 |
|------|-----|
| 데이터 | on-the-fly random 시뮬, 2000 train / 400 val |
| 데이터 분포 | random 정책 → pocket ~0% |
| weight_decay | 3e-4 |
| Loss | raw MSE(vel) + Kendall(vel+type) |
| val RMSE | **4.82** |
| type_acc | 0.76 |
| 문제점 | Kendall kw[type] → 5.07 폭발 (type loss 사실상 0), pocket 학습 없음 |

### v3 (rssm_v3) — epoch 10에서 중단 ❌

| 항목 | 값 |
|------|-----|
| 데이터 | SAC policy pkl, 40K train / 5K val (`world_model/data_rssm/`) |
| 데이터 분포 | pocket **13.1%**, ball_ball **11.8%** (SAC 정책, pocketed 65.7%) |
| Loss 개선 | **scale-weighted MSE**: 1/std² per component (Δvel vs Δω 9.2× 스케일 차이 보정) |
| | **type-freq-weighted MSE**: inv_freq 가중치로 rare type(tgt_circ 1.7%) 보정 |
| | **Focal CE** (γ=2.0): type 분류 rare class 강조 |
| | **Kendall + log_var clamp(-3,2)**: type weight 소멸 방지 |
| SS fix | pred 위치 GT snap 버그 수정 (위치 누적 방식) |
| 데이터 정밀도 | pooltool 이벤트 타임스탬프 기반 dt_to_next (float64, ~1e-15s) |
| val RMSE | 13.05 (epoch 10, 중단 시점) |
| type_acc | 0.822 |
| **버그 발견** | `compute_vel_type_weights`/`compute_type_class_weights`: count=0인 타입(stick_ball)을 `clamp(min=1)`로 처리 → 이 값이 `.mean()`/`.sum()`을 지배해서 실사용 타입 가중치가 전부 0에 가깝게 붕괴 (vel_mean 0.017, 정상값의 1/57). `kw_vel`이 clamp 상한(exp(2)=7.4) 근처인 20.14까지 치솟음 → epoch 10에서 중단. |

### v4 (rssm_v4) — 완료 ✅

| 항목 | 값 |
|------|-----|
| 데이터 | v3와 동일 (SAC policy pkl, 40K/5K) |
| 버그 수정 | weight 함수 2개: count=0 타입 제외하고 mean/sum 계산, `clamp(max=5.0)` 상한 추가 |
| | 수정 후 vel_mean 0.017→**0.35**, type_mean 0.97, pocket_mean 0.69 — Kendall 정상 스케일 |
| event_detector 수정 | `COLL_TYPE["stick_ball"]`: 4→**-1** (transition 취급, rollout 중 잘못된 type=4 유입 방지) |
| 신규 기능 | **per-ball pocket prediction head** (`pocket_mlp`, `predict_pocket(h)`): 이벤트마다 h[i]→P(공 i가 이번 샷에서 포켓될지) sigmoid, BCE loss, `lam_pocket=0.5` |
| | `will_pocket` 라벨 추가 (ShotData, pickle 호환 `__setstate__`) |
| | eval에 `pock_acc` 리포트 추가 |
| val RMSE | **1.98340** (epoch 130) — 500 epoch 전체 실행, early stop 없이 종료 |
| type_acc / pock_acc (epoch 130 기준) | 0.863 / 0.853 |

**SGDR(`CosineAnnealingWarmRestarts`, T_0=50, T_mult=2) 관찰**: restart가 epoch 51/151/351에서 발생, 매번 val_rmse가 일시적으로 2.6~3.0대로 튀었다가 LR이 decay되며 회복하는 패턴이 반복됨. 2번째 사이클(51~150)은 epoch 130에서 최종 best(1.98340)를 찍었고, 3번째 사이클(351~550)은 길이가 200 epoch이라 `max_epochs=500`으로 끝날 때까지 회복을 완료하지 못해(마지막 eval val=2.89) 결국 epoch 130 체크포인트가 최종 best로 남았다. 즉 SGDR 주기가 `max_epochs`보다 길게 잡히면 마지막 사이클이 미완주로 끝나 자원 낭비가 될 수 있음 — 다음 실험에서는 `T_mult` 또는 `max_epochs`를 사이클 경계에 맞춰 조정하는 게 나을 듯.

### v5 (rssm_v5) — 완료 (2026-09-28)

**가설**: v4의 SGDR restart는 매번 val_rmse를 일시적으로 망가뜨렸을 뿐, restart가 만든 mid-cycle optimum(epoch 130, 1.98340)을 넘어서는 이득을 준 적이 없었다. 특히 3번째 사이클은 `max_epochs`보다 길어서 미완주로 끝나 자원만 낭비했다. Warm restart 없이 단일 cosine decay로 학습하면 이런 디스럽션 없이 더 매끄럽게, 적어도 v4와 동등하거나 더 빠르게 수렴할 것이라는 가설.

| 항목 | 값 |
|------|-----|
| 데이터/설정 | v4와 동일 (40K/5K, focal_gamma=2.0, lam_pocket=0.5) |
| 변경점 | `CosineAnnealingWarmRestarts(T_0=50, T_mult=2)` → `CosineAnnealingLR(T_max=max_epochs)` — restart 없이 3e-4→3e-6로 500 epoch에 걸쳐 단조 감소. `T_0`/`T_mult` config·CLI 인자 제거. ⑥ 배치 forward(`--batch-size 32 --device cpu`)로 재시작 |
| val RMSE | **1.71900** (epoch 500, early stop 없이 500 epoch 완주) — v4(1.98340) 대비 개선 |
| type_acc / pock_acc (epoch 500 기준) | 0.831 / 0.857 |
| 총 소요시간 | 18:42~23:21, 약 4시간 39분 |

**가설 검증 결과**: warm restart 디스럽션은 확실히 사라졌으나(SGDR의 51/151/351 epoch 급락 패턴 없음), 대신 scheduled sampling(`ss` 1.0→0.0, `ss_warmup=200`)이 0으로 수렴하는 구간(epoch ~110~300)에서 약 190 epoch간 별도의 정체가 발생했다 — 상세 원인 논의는 아래 "eval 메서드..." / "ss_warmup vs LR anneal 타이밍" 참고. epoch 320 이후 정체 해소, 이후 매끄럽게 개선되어 최종 1.71900으로 v4를 갱신.

**출력**: `world_model/results/rssm_v5/` (best.pt는 epoch 500 시점 가중치 — 중간 체크포인트는 덮어쓰기로 소실됨), 로그 `/tmp/rssm_v5.log`

**진행 관찰 (2026-09-28)**: epoch 110에서 val=1.81041(당시 best) 찍은 뒤 epoch 320(val=1.78991)까지 약 90분간 non-improving 정체. epoch 110~300 구간은 scheduled sampling `ss`가 0.455→0.000으로 떨어지는 구간과 겹쳐서, 학습(train)은 점점 free-running(자기 예측 입력)에 맞춰지는데 eval은 계속 teacher-forced라 지표가 일시적으로 어긋났을 가능성을 논의함(아래 "eval 메서드 논의" 참고). ss=0.000 고정 이후 epoch 320에서 정체 해소, 다시 개선 시작.

#### eval 메서드가 항상 teacher-forcing이라는 점 (결정 필요)

**발견 경위**: 학습 중 epoch 110(ss=0.455)이 epoch 200+(ss=0.000)보다 val_rmse가 낮게 나오는 게 이상하다는 지적에서, `evaluate()`(`train_rssm.py:622-719`) 구현을 다시 확인함.

**사실 확인**:
- `evaluate()`는 매 이벤트마다 `shot.raw_rvws_i[k]`/`raw_rvws_j[k]`(GT)로 `node_i`/`node_j`를 만든다(line 656-664) — **학습 중 ss_prob 값과 무관하게 항상 완전 teacher-forcing(ss=1 상당)**. free-running 예측을 전혀 쓰지 않는다.
- 즉 val_rmse는 "GT가 항상 주어졌을 때 다음 델타를 얼마나 잘 맞히나"만 측정하고, 실제 배포 목표("큐샷 1회 → multi-step rollout imagining → Q-value MC 추정", `roadmap.md` ③)가 필요로 하는 "자기 예측을 계속 먹이는 free-running rollout에서 얼마나 안 어긋나는가"는 전혀 측정하지 않는다.
- 학습 curriculum(ss: 1.0→0.0)이 free-running 쪽으로 이동하는 동안, eval 지표는 계속 teacher-forcing만 재는 다른 것을 측정하므로, "val_rmse가 나빠졌다"가 곧 "모델이 실사용 목적에서 나빠졌다"를 의미하지 않을 수 있다.

**체크포인트 손실**: `train_rssm.py:942-944`(best.pt, 갱신 시 덮어씀) / `:986`(last.pt, 마지막 epoch만)만 저장하므로, epoch 110 시점 가중치는 epoch 320이 새 best가 되며 **이미 덮어써져 복구 불가**. teacher-forced 기준으로든 free-running 기준으로든 재평가 자체가 불가능해졌다.

**결정할 것**:
- **[x]** free-running(ss=0, 자기 예측을 다음 입력으로 사용) multi-step rollout eval을 별도로 추가해서, teacher-forced val_rmse가 뽑은 "best"가 실제 rollout 품질과 상관관계가 있는지 확인 필요. → `evaluate_free_running()` 추가, 결과는 아래 "v6 구현" 참고.
- **[x]** v6부터는 `best.pt`/`last.pt` 단일 덮어쓰기 대신 주기적 체크포인트(예: N epoch마다)를 보관해서, 사후에 여러 시점을 비교할 수 있게 할지 결정. → `ckpt_every` 추가.
- v5 자체는 재시작하지 않고 끝까지 진행 — 위 변경은 v6 이후 적용 대상.

#### ss_warmup vs LR anneal 타이밍 (v6 후보 튜닝)

**관찰**: `ss_warmup=200`(기본값, `train_rssm.py:103`) → ss는 epoch 201에서 이미 0.000 도달. 그 시점 `CosineAnnealingLR(T_max=500)` 기준 lr은 아직 ~1.97e-4(초기 3e-4의 66%)로 높은 채였다. 그런데 val_rmse 정체는 ss=0 도달 후로도 ~110 epoch 더 지속되다가 epoch 320(lr≈8.08e-5, 초기의 27%)에서야 풀렸다 — "ss가 완전히 self-conditioning으로 바뀌는 시점에 lr이 아직 크게 남아있는 것"이 정체를 길게 만든 요인일 수 있다는 가설.

**가설 (수정)**: 처음엔 "`ss_warmup`을 늘려서 ss=0 시점을 lr이 이미 낮아진 구간에 맞추자"고 생각했으나, 이는 방향이 반대일 수 있다 — `ss_warmup`만 늘리고 전체 LR 스케줄(`T_max=max_epochs`)을 그대로 두면 free-running(ss=0) 구간이 시작되는 시점 자체가 뒤로 밀려서, 정작 중요한 free-running 학습 구간에 남는 lr 예산과 epoch 수가 줄어든다. 실제로 v5에서 새 best 갱신은 전부 ss=0 이후 구간(epoch 320~460)에서 나왔고, 이 구간은 현재 스케줄상 전체 lr 예산의 60%(300 epoch, ss_warmup=200/max_epochs=500 기준)를 그대로 넘겨받아 자연 anneal된 결과다. 즉 free-running 구간이 "남은 부스러기 lr"이 아니라 **자기 전용 anneal 예산**을 갖는 게 핵심이지, ss=0 시점을 lr 저점에 맞추는 게 핵심이 아니다.

**[x]** 결정할 것 — v6에서는 `ss_warmup`을 건드리는 대신, **ss=0이 되는 시점부터 LR 스케줄을 새로 시작**(2단계: teacher-forcing 구간은 lr 완만하게 유지, ss=0부터 `CosineAnnealingLR(T_max=max_epochs-ss_warmup)`을 새로 fresh하게 적용)하는 방식을 구현함. 다만 이것도 가설 단계 — v5 로그만으로는 "현재 스케줄이 이미 충분히 괜찮다(free-running 구간에 60% 예산 할당됨)"와 "2단계로 명시적으로 분리하면 더 낫다"를 구분할 수 없어 v6 학습 A/B로 검증 필요 (아직 미실행).

#### v6 구현: 주기적 체크포인트 / 2단계 LR / free-running eval — 완료 (2026-09-28)

위 두 절의 "결정할 것"을 코드로 구현:

| 변경 | 내용 |
|------|------|
| 주기적 체크포인트 | `TrainConfig.ckpt_every`(기본 50) — `best.pt`/`last.pt`와 별개로 `epoch_NNNN.pt`를 N epoch마다 저장. `--ckpt-every 0`으로 비활성화 가능 |
| 2단계 LR | `sched`를 학습 시작 시점에 즉시 만들지 않고 `None`으로 시작. epoch 루프에서 `ss_prob`가 `ss_end`에 처음 도달하는 시점에 `CosineAnnealingLR(T_max=max_epochs-epoch+1, eta_min=lr*0.01)`을 그 시점부터 새로 생성. 그 전까지는 lr이 `cfg.lr`로 고정(스케줄러가 없으므로 step 자체가 없음) |
| free-running eval | `evaluate_free_running()` 추가 — `evaluate()`와 동일한 시그니처/반환값이지만, `_pick_node_i`/`_pick_node_j`/`_advance_rvw`를 `ss_prob=0.0`으로 호출해서 GT 대신 모델 자기 예측을 다음 이벤트의 입력으로 체이닝(=`compute_shot_ss_loss`의 free-running 분기와 동일한 시맨틱). `pocket_acc`는 각 공의 first-touch h 기준이라 첫 접촉은 항상 GT를 쓰므로(=`pred_rvws`가 비어있을 때만 GT 판정) 두 eval 함수에서 이론상 완전히 동일해야 함 |

**v5 `best.pt`(epoch 500)에 대한 teacher-forced vs free-running 비교** (val 5000샷, `world_model/data_rssm` 슬라이스 40000:45000, 동일 체크포인트):

| 지표 | teacher-forced (`evaluate`) | free-running (`evaluate_free_running`) |
|------|------|------|
| val RMSE | 1.71900 | **1.66458** |
| type_acc | 0.831 | 0.799 |
| pocket_acc | 0.857 | 0.857 (예상대로 동일 — 위 표 참고) |

**해석**: free-running RMSE가 teacher-forced보다 오히려 **더 낮다**. v5는 전체 500 epoch 중 마지막 300 epoch(60%)를 `ss=0`(완전 free-running)으로 학습했으므로, 모델이 "자기 예측을 이어받는 입력 분포"에 더 잘 적응한 상태고, 반대로 teacher-forced 입력(GT rvw)은 학습 후반부에는 거의 본 적 없는 분포라 오히려 약간 못 맞히는 것으로 보인다. type_acc는 free-running이 약간 낮은데(0.799 vs 0.831) — 자기 예측 오차가 누적되며 다음 이벤트 타입 분류가 더 어려워지는, 원래 기대했던 방향의 열화. pocket_acc는 정의상(first-touch h는 GT 고정) 완전히 동일하게 나온 것으로 구현이 의도대로 동작함을 재확인.

**결론**: "val_rmse가 항상 teacher-forcing만 잰다"는 우려와 달리, v5처럼 학습 후반부가 이미 ss=0인 경우 teacher-forced eval이 실사용(free-running) 성능을 과소평가하는 방향으로 어긋날 수 있다는 것이 확인됨 — 방향은 걱정했던 것과 반대(free-running이 더 나쁠 거라 예상했으나 더 좋음). 다만 type_acc는 예상대로 약간 열화되므로 두 지표를 분리해서 계속 같이 보는 게 안전. v6부터는 두 eval을 모두 기록해서 추이를 비교할 것.

#### v5 eval 심화 분석: 샷 길이별로 결과가 뒤집힌다 — 완료 (2026-09-28)

**배경**: 위 aggregate 비교(전체 5000샷 평균)만으로 "free-running이 전반적으로 더 낫다"고 결론 내리기엔 이르다는 문제의식에서, 5000개 val 샷 각각에 대해 `evaluate()`/`evaluate_free_running()`을 개별 호출해 per-shot RMSE를 뽑아 분포를 뜯어봄.

**per-shot 분포**:

| 지표 | teacher-forced | free-running |
|------|------|------|
| mean | 1.7190 | 1.6646 |
| median | 1.3884 | 1.2062 |
| p10 / p90 | 0.578 / 3.155 | 0.523 / 3.226 |
| max | 18.26 | **28.05** |

- `corr(tf_rmse, fr_rmse) = 0.880` — 대체로 같이 움직이지만 완전히 같지는 않음.
- free-running이 더 나은 샷: 2680/5000 (53.6%) — aggregate에서 본 "free-running이 근소 우위"라는 그림과 일치.

**샷 길이(event 수)별로 쪼개면 결과가 뒤집힌다**:

| 이벤트 수 | 샷 수 | tf_mean | fr_mean | gap(fr-tf) |
|------|------|------|------|------|
| 1-2   | 343  | 0.789 | 0.794 | +0.005 |
| 3-4   | 1403 | 1.225 | 1.066 | **-0.160** |
| 5-7   | 2223 | 1.730 | 1.572 | **-0.159** |
| 8-12  | 941  | 2.595 | 2.830 | **+0.235** |
| 13+   | 88   | 3.536 | 4.481 | **+0.944** |

**해석**: 짧은/중간 길이 샷(3~7 이벤트, 전체의 73%)에서는 free-running이 더 낫지만, 긴 샷(8+ 이벤트, 21%)에서는 정확히 반대로 **teacher-forced가 더 낫고 격차가 이벤트 수에 비례해 커진다** — free-running 특유의 자기 예측 오차 누적(compounding error)이 짧은 샷에서는 self-conditioning 적응 이득에 가려지다가, 이벤트 수가 늘어날수록 누적 오차가 그 이득을 역전시키는 것으로 해석됨. 전체 aggregate(mean=1.665 vs 1.719)는 73%를 차지하는 짧은 샷들의 개선분이 21%인 긴 샷들의 악화분을 상쇄하고도 남아서 "free-running이 근소 우위"로 보이는 것 — **aggregate 숫자 하나만 보면 이 반전을 완전히 놓친다.**

이는 로드맵 ③(큐샷 1회 → **60-step 같은 긴 rollout** → Q-value MC 추정)에 직접적인 시사점을 준다: 평균 5.76 이벤트인 이 val set의 aggregate free-running RMSE는 실제 목표인 "긴 rollout" 구간의 품질을 낙관적으로 과대평가하고 있을 가능성이 높다 — 정작 중요한 건 8+ 이벤트 구간의 성능인데, 그 구간에서는 free-running이 teacher-forced보다 뚜렷이 나쁘다.

**포켓 여부별**: 포켓된 샷(n=3300)은 free-running이 더 낫고(gap -0.156), 포켓 안 된 샷(n=1700)은 teacher-forced가 더 낫다(gap +0.142) — 포켓 샷이 대체로 짧은 샷과 상관관계가 있을 가능성이 있어 위 길이별 효과의 부산물일 수 있음(별도 검증 안 함).

**최악/최선 샷 (val_shots 인덱스, `world_model/data_rssm` 40000:45000 슬라이스 기준)**:
- teacher-forced와 free-running 둘 다에서 압도적 최악인 샷 2개(`#2185` RMSE 18.3/28.0, `#3261` RMSE 15.0/19.7, 둘 다 4-이벤트) — 두 방식 모두에서 나쁘다는 건 ss 방식과 무관한 별도 원인(이상치 물리 상황 등)일 가능성.
- free-running만 유독 나쁜 샷(`#1679`, `#1422`, 12/15 이벤트) — compounding error를 시각적으로 보여줄 좋은 후보.
- free-running이 teacher-forced보다 훨씬 나은 샷(`#3216`, `#1562` 등) — 반대 사례.

**결정할 것**:
- **[ ]** 8+ 이벤트 구간에서의 free-running 열화가 진짜 문제인지, v6 재학습(2단계 LR 적용 후) 이후에도 재현되는지 확인 필요.
- **[ ]** Q-value rollout 길이(60 step)에 더 가까운 긴 샷 전용 free-running 평가 지표를 별도로 트래킹할지 결정 (현재 val set은 평균 5.76 이벤트로 실제 rollout 길이와 괴리가 큼).

#### 위 분석의 시각 자료: GT/TF/FR 3-way 비교 영상 추출 — 완료 (2026-09-28)

**배경**: 위 length-bucket 반전 현상을 숫자표뿐 아니라 실제 궤적으로 확인하기 위해, 기존 `world_model/viz_rssm.py`(teacher-forcing 궤적만 렌더링하던 코드)를 확장. `shot_rmse()`가 fix⑤ 이후(node_i/node_j/edge가 항상 None) `ev.node_i.to(device)`를 직접 호출해 깨져 있던 것도 같이 고침(`make_node`/`make_edge`로 None-safe하게).

**구현**: `reconstruct(free_running=...)` — teacher-forcing(기존 동작, 매 이벤트 GT pre-state 입력) vs free-running(첫 접촉 이후 모델 자신의 이전 예측을 입력으로 사용, `evaluate_free_running()`/`compute_shot_ss_loss`의 free-running 분기와 동일 semantics)을 하나의 함수로 통합. `reconstruct_both()` + `make_video_3way()`로 GT(실선)/teacher-forced 예측(점선)/free-running 예측(점점선)을 한 화면에 동시 렌더링. `tests/test_viz_rssm.py`에 첫 접촉 시 두 모드 입력이 동일함을 확인하는 테스트, 재접촉 후 (미학습 모델 기준) 두 모드가 갈라짐을 확인하는 테스트 등 추가.

**실행**: `rssm_v5/best.pt` + val slice(`world_model/data_rssm`, offset 40000, n=5000) 기준, 위 분석에서 뽑은 관심 샷 15개(`worst_tf`, `worst_fr`, `fr_much_worse`, `fr_much_better` 그룹의 합집합: `#2185 #3261 #4544 #2478 #3389 #1679 #1422 #3621 #1990 #3537 #3216 #1562 #4553 #429 #2445`)를 3-way 모드로 렌더링:

```bash
python world_model/viz_rssm.py \
  --ckpt world_model/results/rssm_v5/best.pt \
  --data-dir world_model/data_rssm --val-offset 40000 --n-shots 5000 \
  --mode both --tag interesting \
  --indices "2185,3261,4544,2478,3389,1679,1422,3621,1990,3537,3216,1562,4553,429,2445" \
  --out-dir world_model/results/viz_rssm_v5_3way
```

결과물: `world_model/results/viz_rssm_v5_3way/interesting_{01..15}_rmse{...}.mp4` 15개(파일명 순서 = 위 인덱스 나열 순서, rmse는 free-running 기준 재계산값). `interesting_01~10`이 두 방식 모두/free-running만 나쁜 그룹, `interesting_11~15`가 free-running이 오히려 훨씬 나은 그룹.

**아직 안 한 것**: 영상을 실제로 육안 검토해서 "compounding error가 시각적으로 어떻게 나타나는지"(예: 공이 실제 궤적에서 점점 벗어나 이상한 방향으로 굴러가는지, 특정 이벤트 타입에서 급격히 어긋나는지)를 서술하는 것 — 이 세션에서는 렌더링까지만 완료, 시청 후 해석은 별도로 필요.

#### 렌더링한 15개 샷의 이벤트별 추론값 분해 — 완료 (2026-09-29)

**배경**: 영상/aggregate RMSE만으로는 "왜" 틀리는지 알 수 없어서, 위 15개 샷 각각을 이벤트 단위로 쪼개 teacher-forced/free-running 예측 델타(Δvel+Δavel)와 GT 델타를 직접 비교. 스크립트: `/tmp/analyze_interesting_shots_inference.py` (미커밋, ad-hoc), 결과: `/tmp/interesting_shots_trace.json`.

**발견 1 — 샷의 나쁜 RMSE는 "전체 이벤트에 걸친 균등한 누적"이 아니라 특정 이벤트 1~2개의 대형 오차 때문**: 15개 샷 전부에서 tf_err(teacher-forced 상태에서도!)가 유독 큰 이벤트가 1~2개씩 있고(`#2185` k=1: tf_err=53.3, `#4544` k=6: tf_err=87.4, `#2478` k=2: tf_err=77.1 등), 나머지 이벤트는 tf_err 1~10 수준으로 평범함. 즉 앞서 "샷 길이가 길수록 나쁘다"는 상관관계는 **긴 샷일수록 이런 극단 이벤트를 하나 포함할 확률이 높아서 생기는 결과**에 가깝고, 매 스텝 오차가 조금씩 쌓이는 그림이 아님.

**발견 2 — 이 대형 오차는 teacher-forcing으로도 못 고친다 (compounding error 문제가 아니라 회귀 자체의 문제)**: 위 대형-오차 이벤트들은 **GT 델타 자체가 원래 크다**(예: `#4544` k=6은 avel 성분이 실제로 105, -56 rad/s급 — 강한 스핀이 쿠션에서 급격히 전환되는 물리적으로 진짜 격렬한 반사). teacher-forced 모드는 정확한 GT 입력을 주는데도 이 이벤트의 델타를 크게 틀린다 — free-running의 상태 표류(compounding) 문제가 아니라 **dec_mlp가 극단적 스핀 전환 이벤트 자체를 회귀하지 못하는 것**이 1차 원인.

**발견 3 — 오차와 |GT delta| 크기의 관계는 U자형이다**: val shot 1000개(이벤트 8186개) 샘플로 |GT delta| 구간별 tf_err를 봤더니:

| \|GT delta\| 구간 | 비중 | tf_err 평균 |
|---|---|---|
| 0–1 (거의 무변화) | 20.5% | **4.75** |
| 1–3 | 2.5% | 1.79 |
| 3–10 | 19.1% | 1.61 |
| 10–30 | 23.0% | 3.29 |
| 30+ (극단적 변화) | 34.9% | **4.64** |

중간(1~10)은 잘 맞히는데 양 극단(거의 무변화 vs 극단적 변화)에서 둘 다 나쁘다 — 전형적인 "heavy-tailed/이봉(bimodal) 타깃 분포에 대한 MSE 회귀의 평균 회귀(regression-to-the-mean)" 패턴. 이벤트 타입별로도 `type=1`(cue_linear, gt_norm 평균 51/p99 192)과 `type=2`(cue_circular, gt_norm 평균 7이지만 p99 128 — 대부분 작다가 드물게 폭발)가 tf_err 평균이 가장 높은 두 그룹.

이건 이 repo에서 병행 중인 `world_model/spr_mdn/`(SPR + Laplace **Mixture** Density Network) 라인의 문제의식과 정확히 맞아떨어진다 — R-SSM의 단일-포인트(MSE) 델타 회귀가 이런 이봉 분포 이벤트에서 구조적으로 약하다는 걸 이 분석이 실측으로 보여준 셈. R-SSM 자체를 MDN으로 바꾸자는 얘기는 아니고(별도 exp 라인이므로 섞지 않음), R-SSM의 약점이 "compounding" 서사만으로는 설명 안 된다는 근거로 기록.

**발견 4 — type_mlp(다음 이벤트 타입 분류)는 두 가지 체계적 실수를 반복한다**: (a) 같은 물리적 상황의 "linear" vs "circular" 서브타입(1↔2, 5↔6)을 거의 항상 헷갈림(15개 샷 중 다수), (b) 공이 곧 포켓될 상황을 쿠션/충돌로 잘못 예측하는 경우가 반복됨(`#2185` k=2, `#3261` k=2, `#3389` k=2, `#1422` k=10, `#3621` k=0, `#1990` k=4, `#2445` k=3 등). **단, 이게 실제 자율 rollout(`rssm_rollout.py`의 `RolloutEngine`)의 안전성을 해치는 건 아님** — RolloutEngine은 type_mlp 예측이 아니라 `EventDetector`가 모델이 예측한 상태를 다시 물리 시뮬레이션해서 다음 이벤트를 직접 판정하므로, type_mlp 오분류는 보조 loss/metric 품질 문제일 뿐 rollout 분기 정확성과는 별개. (다만 발견 1~2의 상태 오차 자체는 `EventDetector`가 잘못된 위치/속도로 이벤트를 판정하게 만들어 rollout을 틀리게 만드는 진짜 원인.)

**결정할 것**:
- **[ ]** dec_mlp가 극단 스핀 이벤트(type=1/2, |delta|>30)에서 유독 약한 게 데이터 불균형(34.9%가 이 구간인데도 이 정도면 양 부족 문제는 아닐 수도) 때문인지 손실함수(MSE의 평균-회귀) 때문인지 확인 필요 — 후자라면 divergence/huber loss나 per-type loss weighting 검토.
- **[ ]** type_mlp의 linear/circular 혼동이 라벨 정의 자체의 애매함(연속적인 sliding→rolling 전이를 이분류로 나눈 것)인지 feature 부족인지 `event_detector.py`의 COLL_TYPE 정의를 봐야 판단 가능.

#### 위 "이봉 분포 → MSE 구조적 한계" 결론 정정 — 코드/수치로 재검증 (2026-09-29)

**계기**: 위 발견 3의 "heavy-tailed/이봉 분포에 대한 MSE 회귀의 구조적 한계"라는 결론이 근거 없이 강했다는 지적. `train_rssm.py`의 실제 손실 가중치 코드와 사후 분석 수치를 다시 짚어봄.

**가설 1 — [vel, avel] 스케일 불일치(9배 차이)로 MSE가 avel 위주로 최적화됐다**: 기각. `delta_stats.json` 기준 5개 출력 차원(dvx,dvy,dwx,dwy,dwz) 각각의 `rmse/std` 비율은 0.092~0.121로 균일 — 모델은 5개 차원 모두에서 자기 분산 대비 비례적으로 동등하게 잘 맞히고 있음. 참고로 `train_rssm.py`에는 이미 `load_delta_scale_weights()`(1/std² 가중, `--delta-stats` 플래그로만 활성화)가 있지만 v5 학습에는 미사용이었음 — 그런데 정작 스케일 불균형 자체가 없으므로 이건 우선순위 낮은 레버.

**가설 2 — "|delta| 극단(30+, 34.9%)에서 회귀가 못 따라간다"**: 절대오차 기준으로는 맞아 보이지만 **상대오차로 보면 틀렸다**. type별 median 상대오차: type0=0.109, type1=0.090, type3=0.054, type5=0.127, type6=0.106 — 전부 양호. 예외는 **`type=2`(cue_circular) 하나만 median=13.4, mean=19.5로 자릿수가 다름**. 즉 "이봉이라 전체적으로 평균회귀한다"가 아니라 **type=2라는 좁은 카테고리 하나에 국한된 문제**.

**가설 3 — type=2 내부를 |gt_delta| 구간별로 쪼개면**: 문제는 진짜 극단값(50+, 전체의 4.7%, rel_err=0.08 — 오히려 잘 맞음)이 아니라 **거의-무변화 구간(0~0.5, 전체의 83.6%, gt_mean=0.21인데 모델은 평균 4.8을 예측 — rel_err=23)**. 앞서 "발견 3"이 U자형이라고 본 것과 정반대로, 정확히는 "드문 큰 값은 잘 맞고 흔한 작은 값에서 망가진다."

**가설 4(핵심) — 이건 입사각(incidence angle) 커버리지 공백 때문이다, "MLP가 원래 이봉을 못 배운다"가 아니다**: type=2 이벤트의 `|cos(입사각)| = |dot(vel, normal)|/(|vel||normal|)`별로 GT/예측 크기를 봤음 (val 1500샷 샘플, 코드는 인라인 스크립트로 실행, 미커밋):

```
cos[0.00,0.02): n=1649  gt_mag_mean=0.211  gt_mag_max=0.389
cos[0.02,0.04): n=2     gt_mag_mean=0.730
cos[0.04,0.06): n=2     gt_mag_mean=1.096
cos[0.06,0.08): n=1     gt_mag_mean=1.397
cos[0.08,0.10): n=1     gt_mag_mean=1.655
cos[0.10,0.12): n=1     gt_mag_mean=1.936
cos[0.12,0.15): n=1     gt_mag_mean=2.565
cos[0.15,0.20): n=2     gt_mag_mean=22.375
cos[0.20,0.30): n=11    gt_mag_mean=21.267
```

GT 자체는 cos에 대해 **매끄럽게 증가하는 연속 함수**다(이봉이 아님). 그런데 `cos<0.02` 구간에 1500샷 중 1649개(전체 type=2의 83.6%)가 몰려 있고, 그 직후 전이구간(`cos 0.02~0.15`)에는 학습 데이터가 샷 1500개당 8개뿐이다. 즉 **모델이 가장 가파른 기울기를 배워야 하는 구간에 사실상 학습 샘플이 없다** — 이건 MSE/MLP의 구조적 한계가 아니라 **입력 분포(입사각)의 공백(coverage gap)** 문제다. 실제로 모델 예측도 이를 보여준다: `corr(cosang, pred_mag)=0.657`이 `corr(cosang, gt_mag)=0.712`에 근접 — 모델이 입사각-크기 관계 자체는 잘 학습했다는 뜻. 문제는 정확히 데이터가 빈 `cos<0.02` 코너에서, 모델이 옆 구간(`cos>0.15`, 잘 학습된 완만한 램프)의 함수 형태를 그대로 끌어와 과대예측(pred_mean=4.7)하는 것.

**정정**: "R-SSM의 단일-포인트 MSE 회귀가 이봉 분포에 구조적으로 약하다 → MDN 필요"라는 이전 결론은 과잉 일반화였음. 실제로는 (a) 문제가 `type=2` 하나로 국한되고, (b) 그 안에서도 원인이 "타깃이 이봉이라서"가 아니라 "입사각 분포에 실제 데이터 공백(gap)이 있어서"다. `spr_mdn`과의 연결고리로 이 사례를 인용하는 것은 근거 부족 — 철회.

**결정할 것 (갱신)**:
- **[ ]** 가장 저렴한 검증: `cos(incidence) = dot(vel,normal)/(|vel||normal|)`를 `make_node`/`step_single`의 명시적 스칼라 입력으로 추가해서(현재는 vel과 normal이 따로 들어가고 내적을 MLP가 암묵적으로 계산해야 함) type=2 rel_err가 개선되는지 — 데이터를 늘리지 않고도 도움이 될 수 있는 레버.
- **[x]** `cos∈[0.02,0.15]` 구간이 이렇게 비는 게 샷 생성 절차의 편향 때문인지, 실제로 희귀한 각도인지 확인 → 아래 후속 검증에서 답 나옴.
- **[ ]** `compute_vel_type_weights`는 type 단위로만 가중치를 주므로(type=2 전체 가중치 ≈1.09, 캡 5.0 근처도 아님) type 내부의 크기별 불균형(83.6% vs 4.7%)에는 애초에 대응하지 못함 — type 단위가 아니라 `|gt_delta|` 구간 단위 가중치가 필요할 수도 있음.

#### 후속 검증: 전체 5만 샷으로 다시 보니 "공백"이 아니라 "진짜 이봉형 소수 구간" — 완료 (2026-09-29)

**배경**: 위 coverage-gap 결론은 val 1500샷 서브샘플 기준이었음. 전체 데이터셋(50,000샷, type=2 이벤트 18,344개)으로 같은 `cos(incidence)` 히스토그램을 다시 뽑아서 이게 서브샘플링 아티팩트인지 실제 공백인지 확인.

```
cos[0.00,0.02): n=3500 (19.08%)  gt_mag_mean=0.209
cos[0.02,0.04): n=147  (0.80%)   gt_mag_mean=0.469
cos[0.04,0.06): n=31   (0.17%)   gt_mag_mean=0.877
cos[0.06,0.10): n=43   (0.24%)   gt_mag_mean~5.5
cos[0.10,0.15): n=60   (0.33%)   gt_mag_mean~10
cos[0.15,0.30): n=281  (1.53%)   gt_mag_mean~20
cos[0.30,0.50): n=880  (4.80%)   gt_mag_mean=36.5
cos[0.50,1.01): n=13402 (73.06%) gt_mag_mean=47.2
```

**결론**: 완전한 공백은 아니었다 — 전체 스케일에서는 전이구간(`cos 0.02~0.5`)에 실제로 ~1412개 이벤트(type=2의 7.7%)가 존재한다(1500샷 서브샘플에서 본 "1~2개"는 표본 크기(1500/50000≈3%)에 비례한 것일 뿐, 진짜 공백이 아니라 원래도 희귀한 구간이었음). 대신 드러난 그림은: **진짜 이봉형 입사각 분포** — 거의 접선(`cos<0.02`, 19%, 거의 무변화)과 거의 정면(`cos>0.5`, 73%, 큰 변화) 두 거대 모드 사이에 얇은 소수 구간(7.7%)이 끼어있는 형태. `event_detector.py`가 곡률 쿠션 세그먼트를 포켓 자입구 근처에 두는 기하 구조상 자연스러운 결과로 보임(레일 따라 미끄러지듯 들어가면 접선, 직접 겨냥하면 정면 — 그 중간 각도로 맞는 경우 자체가 실제로 드묾).

**의미**: `generate_rssm_data.py`의 샘플링 버그가 아니라 실제 물리 기하 구조의 성질이므로, "이 각도대를 겨냥한 합성 샷 추가 생성" 항목은 근본 해법이 아니라 차선책(불가능하진 않지만 효과 대비 비용이 큼)으로 강등. 대신 **이미 존재하는 소수 구간(1412개)이 두 거대 모드에 눌려 MSE에서 무시되는 것**이 핵심이므로, type 단위가 아닌 `|gt_delta|` 크기 단위 loss weighting(위 결정할 것의 3번째 항목)이 가장 직접적인 다음 레버로 우선순위가 올라감.

**결정**: 다음 실험은 magnitude-bin 기반 `compute_vel_magnitude_weights()`를 `train_rssm.py`에 추가해서 `_vel_loss_term()`에 적용 — 타입 가중치와 별개로 크기 구간별 역빈도 가중치를 곱한다. `cos(incidence)` explicit feature 추가(구조 변경, 재학습 필요)는 2순위로 미룸.

#### magnitude-bin weighting 방향 검증 (재학습 없이, v5 체크포인트 forward만) — 완료 (2026-09-29)

구현 직후 "이 weighting이 실제로 근사-제로 그룹(rel_err=23)을 겨냥하는 게 맞는지" 재학습 없이 먼저 확인. `world_model/results/rssm_v5/best.pt`로 val set type=2 이벤트에 teacher-forced forward만 돌려서 `|gt_delta|` 구간별 count/loss 기여도/절대오차를 직접 집계.

```
[0,1)    count=52.6%  loss기여=35.0%  mean_sqerr=28.36  mean_abserr=4.11
[1,3)    count= 1.1%  loss기여= 0.3%  mean_sqerr=12.06  mean_abserr=2.96
[3,10)   count= 5.8%  loss기여= 2.4%  mean_sqerr=17.47  mean_abserr=3.12
[10,30)  count=15.2%  loss기여=11.0%  mean_sqerr=30.91  mean_abserr=4.20
[30,inf) count=25.3%  loss기여=51.3%  mean_sqerr=86.36  mean_abserr=6.51
```

**정정 1**: "근사-제로 그룹의 relative error가 큰 건 GT가 작아서 나누기 때문에 생기는 착시"라는 설명은 틀렸다. `|gt|<1`인데 평균 절대오차가 4.11 — 실제로 큰 폭으로 못 맞히고 있다(0.1~0.3 세부구간만 봐도 절대오차 동일하게 4.1대). relative error 착시가 아니라 진짜 예측 실패.

**정정 2 (걱정했던 것과 반대)**: `compute_vel_magnitude_weights`는 type별이 아니라 전체 이벤트 풀링 기준으로 count를 세기 때문에, type=2 내부 비중(52.6%)만 보고 "다수 그룹이 더 downweight될 것"이라 걱정했던 건 틀린 계산이었다. 실제 학습에 쓰이는 (전체 이벤트 기준) weight `[1.49, 2.68, 0.35, 0.28, 0.20]`를 type=2 실측 오차에 곱해보면 loss 기여 비중이 근사-제로 35%→78%로 오히려 크게 올라가고 대형-delta는 51%→15%로 줄어든다 — 즉 구현 방향은 실제 문제(근사-제로 그룹의 절대오차)를 정확히 겨냥한다.

**결론**: `--vel-mag-weight`는 v7 학습으로 실증 검증해볼 가치가 있음. (아직 미실행)

#### bin 경계값을 하드코딩 대신 quantile로 자동화 — 완료 (2026-09-29)

**문제 제기**: `DEFAULT_MAG_BIN_EDGES=[0,1,3,10,30,inf]`는 지금 데이터 분포를 손으로 보고 정한 값이라, 물리 파라미터나 샷 수가 바뀔 때마다 다시 히스토그램을 찍고 경계값을 조정해야 하는 구조였다. `compute_vel_type_weights`(카테고리 수 고정, N_TYPE=7)와 달리 magnitude bin은 연속값을 자르는 거라 경계값 자체가 데이터셋에 의존적이라는 게 근본 문제.

**해결**: `compute_quantile_bin_edges(shots, n_bins=5)` 추가 — `|gt_delta|` 분포의 분위수(quantile)로 경계값을 매번 데이터에서 자동 계산. `compute_vel_magnitude_weights(bin_edges=None)`이 기본값이 되면서 자동으로 이 경로를 탐. `DEFAULT_MAG_BIN_EDGES` 상수는 제거. 기존 단위 테스트들은 반환된 edges를 그대로 재사용해서 bin을 계산하는 방식이라 전부 수정 없이 통과했다(우연이 아니라, quantile 분할이 다수/소수 클러스터를 자동으로 갈라주는 성질 덕분 — degenerate/동일값이 몰린 경우에도 검증됨).

**RL 통합 시 재고 필요 사항 (지금은 미해결로 남김)**: roadmap.md의 WM→RL 통합(`s_hat = wm(s_1, n_steps=60)`)에서 R-SSM을 frozen으로 쓸지 RL이 만든 데이터로 continual fine-tune할지 아직 미정. frozen이면 quantile 자동화만으로 충분하지만, continual이면 정책이 좋아질수록 방문 분포가 바뀌어서 학습 시작 시 1회 계산한 quantile 경계값이 stale해짐 — 이 경우 주기적 재계산이나 온라인(EMA) 추정으로 바꿔야 함. 지금은 이 결정이 안 났으므로 오프라인 전제 그대로 두고, WM→RL 통합을 실제로 설계할 때 다시 열어보기로 함.

### 물리 엔진 버그: ball_motion.py 마찰계수 (commit 4bbf568)

**배경**: long shot일수록 RSSM 예측 오차가 커지는 원인을 추적하던 중 발견.

| 항목 | 내용 |
|------|-----|
| 버그 | `world_model/ball_motion.py`의 `u_r`(rolling), `u_sp`(spin) 마찰계수가 pooltool 실제 기본값보다 과도하게 큼 |
| 영향 | free-flight 구간이 길어질수록(=long shot) 오차가 누적 — RSSM data 생성/rollout 전반에 영향 |
| 수정 | pooltool `BallParams.default()` 실측값으로 정정 |
| 검증 1 | `compare_pure_physics.py` — data_rssm 실 샷 전체에 대해 pooltool vs pure_physics 수치 비교 |
| 검증 2 (신규) | `viz_pure_physics.py` — 동일 GT 충돌-후 상태에서 pooltool·pure_physics 양쪽으로 자유주행시켜 궤적을 겹쳐 그린 worst/best 5개 mp4 |
| 결과 | 300샷 전체 min=max=mean=median **0.0000cm** — long shot 포함 전 구간에서 두 엔진 완전 일치 |

**`pure_physics.py` drop-in 대체 가능성**: `evolve_ball_motion(state, rvw, R, m, u_s, u_sp, u_r, g, t) -> (rvw, state)` signature 동일, state 상수(`STATIONARY=0, SPINNING=1, SLIDING=2, ROLLING=3, POCKETED=4`) 동일 → free-motion evolution 대체 가능함을 확인.

- 교체 대상 3곳 — **완료 (2026-09-28)**: `rssm_rollout.py::advance_balls`, `train_rssm.py::_advance_rvw()`, `viz_rssm.py::_evolve()` 전부 `pure_physics.evolve_ball_motion` 사용으로 전환
- 제외(별도 검증 필요): `event_detector.py::get_next_event()` 등 충돌시각 solver — 이번 검증 범위 밖

### R-SSM pocket head 검증: AUC=0.919 "Too good to be true?" 리크 헌팅 (`eval_pocket_head.py`)

**배경**: rssm_v4 학습 중 "공 흐름 예측이 부실해 보인다"는 관찰에서 시작 — per-event-type RMSE를 뜯어보니 실제 약점은 cushion 충돌(`cue_linear`/`cue_circ`/`tgt_linear`/`tgt_circ`, RMSE 2.7~3.4)이었고 `ball_ball`(RMSE~0.91)은 양호했다(SGDR warm restart로 인한 일시적 val_rmse 스파이크와도 별개). 이 김에 pocket 예측(`predict_pocket`)만 따로 정밀 평가(`eval_pocket_head.py`)했더니 AUC=0.919가 나왔고, 사용자가 "Too good to be true"라고 리크 가능성을 제기했다.

#### 가설

1. **Tautology leak**: first-touch 이벤트 자체가 포켓 이벤트인 공(사전 충돌 0회로 바로 포켓되는 경우)은, `make_node()`가 `event_type=pocket`을 입력 feature에 원-핫으로 직접 심어버린다. 즉 `predict_pocket(h)`가 미래를 맞히는 게 아니라 자기가 이미 받은 입력을 그대로 되읽는 것일 수 있다.
2. **Geometric shortcut**: 물리 시뮬레이션(다중 충돌, 마찰, 쿠션 반사) 없이, 충돌 직후(post-collision) 속도 방향을 직선으로 연장했을 때 포켓 근처를 지나가는지만 봐도 상당한 판별력이 나올 수 있다 — 즉 모델이 실제로 배운 것은 "물리"가 아니라 "방향이 대충 맞는지" 정도의 쉬운 패턴일 수 있다.

#### 검증 방법

1. **가설 ① 검증**: 전체 8267개 샘플 중 "first-touch 이벤트 == pocket 이벤트"인 tautological 샘플을 분리(92개), 제외한 뒤 AUC를 재계산해서 비교.
2. **가설 ② 검증**: `geometric_baseline_score()` — 물리 지식 0, 학습 파라미터 0인 baseline. 공의 post-collision 속도 방향으로 직선 ray를 쏴서 6개 포켓 중 최근접 거리를 구하고, 그 거리의 음수를 score로 사용(가까울수록 "포켓될 것 같음" 점수가 높음). Model AUC와 같은 rank-based AUC 공식으로 비교.
   - 여기서 시행착오가 있었다: post-collision 속도를 처음에 `node_i[2:4]`(그 이벤트가 일어나기 **전** 속도)로 잘못 구해서 baseline AUC가 0.266~0.283(랜덤 0.5보다 낮음)이 나왔다. 이 이상한 값 자체가 "충돌 전/후 방향이 실제로 크게 다르다"는 신호였고(한 실 샷에서 `gt_deltas_i[0]`의 Δvel_y=-13.25 확인, cushion bounce로 방향이 뒤집힌 사례), post-collision 속도(`node_i[2:4] + gt_deltas_i[k][0:2]`, 즉 pre-collision vel + 그 이벤트의 ground-truth Δvel)로 고쳐서 재계산했다.

#### 결과

| 가설 | 결과 |
|------|------|
| ① tautology leak | AUC 0.919 → 0.917 (tautological 92개 제외). **거의 무변화** |
| ② geometric shortcut | Baseline AUC = **0.827** (genuine 샘플 기준) vs Model AUC = 0.917 (genuine 기준) |

#### 논의

- **리크는 아니다** — tautology를 제외해도 AUC가 거의 안 바뀌고(0.919→0.917), baseline은 모델과 완전히 독립된 경로로 계산했으므로 두 가설 모두 "AUC=0.919가 사실은 가짜"라는 설명이 되지 못한다.
- 그러나 **헤드라인 숫자가 주는 인상보다 훨씬 소박한 결과**다. genuine AUC 0.917 중 0.827은 "충돌 직후 속도가 대략 포켓 쪽을 향하는가"라는 공짜 기하학 baseline만으로 이미 달성된다. 모델이 학습으로 추가한 판별력은 0.917-0.827 ≈ **+0.09 AUC**뿐이다. task 자체가 "공이 충돌 후 어느 쪽으로 튀는가"라는 쉬운 기하학적 구조를 갖고 있어서, 순수 baseline도 상당히 높은 점수를 낸다. 즉 AUC=0.919는 "모델이 멀티-바운스 물리를 이해했다"는 증거로는 과대해석이고, "단순 방향 정보 위에 약간의 추가 판별력을 더했다"는 정도로 읽어야 한다.
- **별도 flag**: `will_pocket=True` base rate가 46%로 매우 높다 (5000샷 중 first-touch된 공 8267개 기준, 즉 샷당 평균 1.65개 공만 터치되고 그중 46%가 포켓됨). 실제 플레이 대비 데이터 생성이 "생산적인" 샷 쪽으로 편향돼 있을 가능성이 있으나 아직 원인 미확인.

#### 후속 질문: 시퀀스가 실행되는 동안, 이벤트마다 예측이 맞는가?

**배경**: 위 검증은 전부 "first-touch h" 한 시점만 봤다. 그런데 실제로 `train_rssm.py`의 pocket loss(`train_rssm.py:271`)는 **매 이벤트마다, 모든 공에 대해** `predict_pocket(h)`를 호출해서 계산된다 — first-touch만 보는 게 아니다. 즉 실제 rollout에서 이 head를 쓴다면 이벤트마다 계속 호출될 텐데, first-touch 스냅샷 하나만 잘 나온 것일 수도 있다는 의문.

**가설**: 공이 여러 번 충돌하는 도중(2번째, 3번째 접촉 등)에는 아직 운명이 결정되지 않은 상태라서, first-touch보다 예측이 더 불안정하거나 부정확할 수 있다.

**검증 방법**: `collect_pocket_preds_all_touches()` — 각 공의 **모든** 접촉(포켓 이벤트 자체는 여전히 제외)에서 `predict_pocket(h)`를 기록하고, 그 공의 몇 번째 접촉인지(touch_idx)로 묶어서 AUC를 따로 계산. 추가로 "마지막 접촉"(공의 운명이 결정되기 바로 직전 — 포켓될 공이면 포켓 이벤트 한 스텝 전, 안 될 공이면 샷 내 실제 마지막 접촉)만 따로 모아서 AUC 계산.

**결과**:

| 시점 | N | pos_rate | AUC |
|------|---|----------|-----|
| 1번째 접촉 | 8175 | 0.458 | 0.917 |
| 2번째 접촉 | 5714 | 0.253 | 0.871 |
| 3번째 접촉 | 4468 | 0.156 | 0.828 |
| 4번째+ 접촉 | 9326 | 0.113 | 0.882 |
| **마지막 접촉(운명 결정 직전)** | 8175 | 0.458 | **0.996** |

**논의**:
- **답은 "아니오"다** — 매 이벤트마다 예측이 균일하게 정확하지 않다. 2~3번째 접촉(공이 아직 여러 번 더 튕길 수 있어 운명이 실제로 안 정해진 구간)에서 AUC가 0.83~0.87로 first-touch(0.917)보다 오히려 **더 나쁘다**. 처음 보고했던 0.919/0.917은 시퀀스 전체를 대표하는 값이 아니라, 마침 상대적으로 쉬운 시점(첫 접촉) 하나를 본 것이었다.
- 반면 "마지막 접촉"(포켓되기 직전 or 샷이 끝나기 직전)은 AUC 0.996으로 거의 완벽하다. 다만 이건 그다지 대단한 성과는 아닐 수 있다 — 그 시점엔 공이 이미 포켓 입구에 있거나 명백히 멀어지고 있어서, 단순 기하학으로도 맞히기 쉬운 "정답이 이미 드러난" 순간에 가깝다.
- 종합하면: 이 head는 **판단이 쉬운 시점(처음 한 번, 혹은 운명이 이미 정해진 순간)에는 잘 맞고, 진짜 불확실한 중간 구간에서는 신뢰도가 떨어진다.** "학습으로 물리를 이해했다"는 근거로 쓰기엔 여전히 약하고, 실시간 rollout 중간에 이 값을 의사결정에 쓴다면 중간 구간에서의 낮은 신뢰도를 감안해야 한다.

#### 후속 질문 2: 이 우위가 "물리를 배워서"인지, "적당히 학습해도 나오는" 건지?

**배경**: 위에서 모델의 진짜 기여는 genuine AUC 0.917 중 baseline(0.827)을 뺀 +0.09뿐이라고 확인했다. 그런데 이 +0.09가 어디서 오는지는 아직 모른다 — baseline이 이미 잘 맞히는 "쉬운" 샘플에서 우연히 조금 더 잘 맞혀서 생긴 차이라면 "물리를 배웠다"고 보기 어렵다. 반대로 baseline이 구조적으로 못 보는 케이스(직선 연장으로는 알 수 없는, 쿠션/공 충돌이 한 번 더 일어나는 경우)에서 우위가 집중된다면 최소한 단순 방향 정보 이상을 학습했다는 증거가 된다.

**가설**: baseline은 "이번 접촉 후 방향을 직선으로 연장"하는 방식이라, 그 다음에 쿠션이나 다른 공과 한 번이라도 더 충돌하면 궤적을 전혀 볼 수 없다(직선이 꺾이는 걸 모른다). 따라서 1번째 접촉이 곧 운명 결정 직전인 "easy" 그룹에서는 baseline이 원래 강하고, 1번째 접촉 후 최소 한 번 더 충돌이 있는 "hard" 그룹에서는 baseline이 구조적으로 불리할 것이다. 모델이 진짜 물리(충돌 후 방향 변화 등)를 학습했다면, 우위가 easy보다 hard에서 더 커야 한다.

**검증 방법**: `report_physics_vs_shortcut()` — 1번째 접촉 샘플을 `is_last_real_touch` 기준으로 easy(추가 충돌 없이 운명 결정)/hard(최소 한 번 더 충돌)로 나눠서 각각 모델 AUC, baseline AUC, 그 차이(gap)를 계산. 추가로 `pairwise_rescue()` — 전체 (positive, negative) 샘플 쌍 중 baseline이 순서를 잘못 매긴 쌍이 얼마나 있고, 그중 모델이 올바르게 순서를 매기는(rescue) 비율을 계산(50%=아무 정보도 안 더한 것, 100%=완벽히 고침).

- 시행착오: 처음에는 easy/hard를 `total_touches[b] == 1`(공이 딱 한 번만 접촉) 기준으로 나눴는데, 이것도 "마지막 접촉" 버그와 같은 종류의 tautology였다 — 포켓되는 공은 포켓 이벤트 자체가 접촉 횟수에 하나 더 포함되므로 `total_touches==1`은 애초에 포켓되지 않는 공만 걸러진다(실측: N=161, pos_rate=0.000, AUC=nan). 이미 검증된 `is_last_real_touch` 필드(포켓 이벤트를 접촉 횟수에서 제외하고 계산)로 다시 나눠서 easy N=2461(pos_rate=93.5%), hard N=5714(pos_rate=25.3%) — 둘 다 정상적인 분포임을 확인한 뒤 사용.

**결과**:

| 그룹 | N | pos_rate | Model AUC | Baseline AUC | gap |
|------|---|----------|-----------|--------------|-----|
| easy (추가 충돌 없음) | 2461 | 0.935 | 0.999 | 0.905 | +0.094 |
| hard (최소 1회 추가 충돌) | 5714 | 0.253 | 0.817 | 0.671 | +0.146 |

Pairwise rescue: baseline이 잘못 순서를 매긴 쌍 2,866,568개(전체 쌍의 17.3%) 중, 모델이 올바르게 고친 비율 = **73.9%**.

**논의**:
- **모델의 우위는 hard 그룹에서 더 크다** (+0.146 > +0.094). baseline이 구조적으로 못 보는 영역(쿠션/공 충돌이 최소 한 번 더 있는 경우)에서 모델이 오히려 더 많은 정보를 추가한다 — "baseline이 잘 맞히는 쉬운 사례에 무임승차"로는 설명되지 않는 패턴이다.
- hard 그룹 baseline AUC=0.671은 랜덤(0.5)에 가깝다 — 직선 연장으로는 거의 못 맞힌다는 뜻. 그런데도 모델은 0.817을 낸다. 이 정보는 직선 기하학이 원천적으로 제공할 수 없으므로, 모델이 "충돌 이후 방향이 바뀔 수 있다"는 사실을 어떤 형태로든 내재화했다는 증거다. pairwise rescue 73.9%(우연=50%)도 같은 결론을 보강한다.
- 다만 **hard 그룹 AUC(0.817) 자체는 easy 그룹(0.999)보다 훨씬 낮다** — 진짜 어려운 멀티바운스 상황에서는 모델도 상당히 헷갈려 한다. 그리고 이 결과는 "모델이 반사각/운동량 보존 같은 물리 법칙을 명시적으로 시뮬레이션한다"와 "충돌 후 궤적이 대략 이렇게 꺾이더라는 통계적 패턴을 학습했다"를 구별해주지 못한다 — 둘 다 이 실험 결과와 부합한다.
- **결론**: 완전한 무작위/얕은 학습(=baseline 재현)은 아니다 — 최소한 "충돌하면 방향이 바뀐다"는 구조를 학습으로 반영한 것은 맞다. 하지만 이게 "물리 엔진처럼 정밀하게 이해했다"는 뜻은 아니며, hard 그룹 정확도 자체가 낮다는 걸 보면 "적당히 학습된, 부분적인" 물리 이해로 보는 게 정확하다 — "전혀 안 배웠다"와 "물리를 완전히 이해했다"의 중간 어딘가.

#### 결정할 것

- **[ ]** roadmap.md QHead 후보 3(`pocket_prob` heuristic을 Q-target에 쓸지)에 이번 결과를 반영해서, 이 heuristic을 그대로 쓸지 / baseline 대비 보정해서 쓸지 / 폐기할지 결정 필요. **특히 rollout 중간 구간(2~3번째 접촉)에서 신뢰도가 낮다는 점을 반영할 것** — first-touch AUC만 보고 판단하면 과대평가.
- **[ ]** hard 그룹(멀티바운스) AUC=0.817은 "부분적 물리 이해"로 보이지만, 모델이 실제로 반사 법칙을 내재화했는지 vs 학습 데이터의 통계적 궤적 패턴을 외웠는지는 이번 검증으로 구별 불가 — out-of-distribution 각도/속도(학습 데이터 분포 밖)에서 AUC가 유지되는지 확인하면 구별 가능할 것.
- **[ ]** `will_pocket` base rate 46%가 데이터 생성(`generate_rssm_data.py`) 샷 샘플링 편향 때문인지 실제 분포를 반영하는지 확인 필요 — 편향이면 pocket head/QHead 전체의 학습 분포가 실제 사용 환경과 다를 수 있음.
- **[ ]** 모델의 진짜 +0.09 AUC 기여가 다운스트림(Q-value 추정, 정책 학습)에 실질적 가치가 있는지는 이번 검증 범위 밖 — 별도로 확인할 방법 필요.
- **[ ]** `train_rssm.py:271`의 pocket loss는 아직 접촉 안 한 공(h=0)까지 포함해 매 이벤트마다 전체 공에 대해 계산됨 — 이게 학습 신호를 얼마나 희석시키는지, first-touch 이후 시점만 loss에 포함하도록 바꿔야 하는지는 이번 검증 범위 밖(별도 확인 필요).

### R-SSM 배치 forward: GPU 활용률 개선 (2026-09-28)

**배경**: `compute_shot_ss_loss`가 샷 1개씩(`batch_size=1`) forward하고 있어서 GPU 활용률이 낮았다. `rssm_model.py`의 모든 MLP(`msg_mlp`/`upd_mlp`/`single_mlp`/`dec_mlp`/`type_mlp`)는 `nn.Linear`+`SiLU`만 쌓은 구조라 leading batch dim을 그대로 broadcast한다 — 즉 모델 구조 자체는 이미 배치를 지원하고 있었고, 병목은 100% 호출부(`compute_shot_ss_loss`)가 `(dim,)` 1-샘플 텐서만 만들어 넘기는 데이터 마샬링 문제였다. 새 수식이나 파라미터 없이 순수 오케스트레이션 리팩터로 풀 수 있는 문제라고 판단해 Phase 0→3 순서로 진행했다.

**구현**:
- **Phase 0** — wavefront 스케줄러 (`rssm_batch.py::iter_wavefronts`, `split_wavefront_by_type`): 샷마다 이벤트 개수가 다르므로, "아직 이벤트가 안 끝난 모든 샷의 다음 이벤트"를 한 배치로 묶어 순회하고, 샷은 이벤트가 끝나면 드롭아웃. 샷 간 공유 상태가 없으므로 배치 내 순서는 무관.
- **Phase 1** — `step_ball_ball_batch`/`step_single_batch` (`rssm_model.py`): 기존 `step_ball_ball`/`step_single`과 동일 가중치를 재사용하는 순수 배치 래퍼.
- **Phase 2** — `compute_batch_ss_loss` (`train_rssm.py`): wavefront 단위로 노드/엣지를 `torch.stack`해서 배치 forward, 결과를 다시 list 원소로 scatter. `h`/`pred_rvws`/loss 누적값 전부 Python list로 유지(텐서 in-place 금지 — 기존 `h.clone()` 제거 패턴의 일반화, autograd 안전성 확보).
- **Phase 3** — `train()` 루프를 shot당 forward에서 batch당 forward로 교체. `TrainConfig.batch_size` 추가. `accum_steps`의 의미가 "샷 개수"에서 "batch 개수"로 바뀜 (`effective_batch = batch_size * accum_steps`) — 기존 설정과 동일한 effective batch를 유지하려면 `batch_size=<기존 accum_steps>, accum_steps=1`로 설정할 것 (둘 다 기존 값으로 두면 effective batch가 중복 곱해짐).

**정합성 검증**:
- `ss_prob∈{0.0, 1.0}`에서 배치(wavefront 인터리빙) 결과가 순차 처리와 bit-identical함을 증명 — 이 두 지점에서는 `random.random() < ss_prob` 분기 결과가 실제 뽑힌 RNG 값과 무관하기 때문(`ss_prob=1.0`→항상 True, `ss_prob=0.0`→항상 False). `0<ss_prob<1`에서는 RNG 소비 순서가 달라 정확히 일치하진 않지만 finite/no-crash만 확인.
- `batch_loss = mean(per-shot loss)` 1회 backward가 기존 `sum((shot_loss/accum_steps).backward())`와 gradient 수준에서 정확히 동치임을 전용 비교 테스트로 검증 (선형성에 의해 수학적으로 동일).
- 리팩터 도중 발견한 버그: `rssm_dataset.py`의 `generate_shot_data`에서 `event_type==EVENT_BALL_BALL`이어도 두 번째 공이 tracked 안 되면 `ball_j=None`으로 떨어지는 케이스가 있는데, Phase 0의 `split_wavefront_by_type`이 `event_type`만 보고 라우팅해서 이 케이스를 놓쳤음 (기존 `compute_shot_ss_loss`는 `ev_type==EVENT_BALL_BALL and bj is not None` 가드가 있어서 문제없었음). `ball_j is not None` 체크 추가로 수정, 회귀 테스트 추가.
- 관련 없는 사전 존재 실패 7건(`test_rssm_dataset.py`, `test_rssm_rollout.py`, `test_spr_mdn_model.py`, `TestClassWeights::test_weights_positive`)은 git-stash 비교로 이번 작업 이전부터 있던 것으로 확인, 무관하므로 미수정.

**속도 실측** (`h_dim=32, hidden=[64,64]` 소형 모델, wall-clock 단순 forward+backward 1회 기준):

| 환경 | 샷 수 | 순차(기존) | 배치(신규) | 향상 |
|---|---|---|---|---|
| MPS | 25 | 0.39s | 0.23s | 1.7배 |
| MPS | 96 | 1.67s | 0.96s | 1.7배 |
| CPU | 69 | 0.11s | 0.03s | 3.7배 |

**흥미로운 부수 발견**: 이 모델 크기에서는 **MPS가 CPU보다 느리다** (예: 96샷 기준 MPS 0.96s vs CPU 0.03s대). MPS 커널 launch 오버헤드가 개별 연산의 실제 compute 시간보다 커서, 작은 텐서 위주의 이 워크로드에서는 GPU 이점이 없다. → rssm_v5 재학습 시 `--device cpu` 권장.

**pure_physics 벡터화(`evolve_ball_motion_batch()`) 보류 결정**: roadmap의 ⑤ 서브 아이템으로 남아있던 물리 호출 벡터화가 이 배치 forward 이후에도 의미 있는지 프로파일링으로 확인했다. `_advance_rvw`(`train_rssm.py`)는 `ss_prob>0.1`일 때 fast-linear 근사만 쓰고 `evolve_ball_motion`을 아예 호출하지 않으며, `ss_prob=0.0`(완전 free-running)일 때만 실제 물리 호출을 한다. `cProfile`로 `ss_prob=0.0` 케이스를 뜯어본 결과 `_advance_rvw`가 전체 iteration 시간의 ~10%에 불과했다 (`ss_prob=1.0`→`0.0` 비교 시 총 시간 증가도 18%뿐, 1443회 호출에 0.011s / 총 0.118s). 완전 벡터화해도 최선의 경우 ~10% 개선이 한계이므로, 방금 확인한 배치 forward의 1.7~3.7배 개선 대비 우선순위가 낮다고 판단해 보류.

### 3-ball 데이터 생성: 포켓 개수 불균형 확인 + balanced 샘플링 함수 (2026-09-29)

**배경**: roadmap ④(N-ball 일반화)의 첫 단계로 3-ball(cue+target 2개) 데이터 생성을 시작한다. `BilliardsEnv`의 `n_balls` 파라미터는 큐볼을 제외한 target 개수이므로, "3-ball"을 만들려면 `n_balls=2`로 호출해야 한다(`simulator.py:142-172` 확인 — `self._ball_ids = [str(i+1) for i in range(n_balls)]`). 또한 obs_dim이 `n_balls`에 따라 달라져서(`n_balls=1`→16-dim, `n_balls=2`→20-dim), 기존에 학습된 SAC 모델은 전부 `n_balls=1` 기준이라 재사용 불가 — 3-ball 데이터 생성은 `--policy random`만 가능하다.

**가설**: 랜덤 정책으로 3-ball 샷을 생성하면 포켓 개수(0/1/2)의 분포가 심하게 불균형할 것이다(2-포켓 동시 성공은 희귀 이벤트).

**검증 방법**: `n_balls=2` + random 정책으로 3000 에피소드를 돌려 각 샷의 `n_pocketed_targets()`(타깃 볼 중 포켓된 개수, 큐볼 스크래치는 제외)를 집계.

**결과**: 3000 에피소드 중 2221개가 유효 샷(이벤트 발생), 508.1 ep/s.

| n_pocketed | 개수 | 비율 |
|---|---|---|
| 0 | 2073 | 93.34% |
| 1 | 144 | 6.48% |
| 2 | 4 | 0.18% |

**결론**: 랜덤 정책 그대로 대량 생성하면 2-포켓 샘플이 극도로 희귀해(0.18%) 학습 데이터에서 사실상 누락된다. 그대로 두면 안 되고, 포켓 개수 버킷별로 quota를 채울 때까지 reject-sampling 하는 방식이 필요하다고 판단.

**구현**: `world_model/generate_rssm_data.py::generate_balanced(quotas, policy_fn, n_balls, seed_start, max_attempts, report_every)` 추가.
- `ShotData.n_pocketed_targets()`(`world_model/rssm_dataset.py`, 큐볼 인덱스 0 제외하고 포켓된 target 개수 반환)로 각 샷을 버킷(0/1/2)에 라우팅.
- 이미 quota가 찬 버킷은 버리고, 모든 버킷이 quota를 채우거나 `max_attempts`에 도달하면 종료 — 희귀 버킷(2-포켓) 때문에 무한 루프에 빠지지 않도록 안전장치.
- CLI에 `--balanced`, `--quota0/1/2`, `--max-attempts` 옵션 추가(기본 동작인 `generate()`는 그대로 유지, `--balanced` opt-in).
- 테스트: `tests/test_generate_rssm_data.py` — quota 초과 안 하는지, 도달 불가능한 버킷(quota가 매우 큰데 n_balls=1이라 2-포켓 자체가 불가능한 경우)에서 `max_attempts`로 정상 종료하는지 확인.

**다음**: 실제 quota를 정해 balanced 데이터셋을 생성 실행(2-포켓 quota가 클수록 시간이 오래 걸림 — 위 측정 기준 quota=1000이면 약 25분 추정). roadmap ④ 상태 업데이트는 실제 생성 완료 후 진행.

**실행 결과 (2026-09-29)**: `--balanced --quota0 1000 --quota1 1000 --quota2 1000 --n-balls 2 --policy random --seed 42`로 백그라운드 실행. 812,540 에피소드 시도 끝에 세 버킷(0/1/2-포켓) 모두 1000개씩 채워 총 3,000 샷 확보, 소요 27.4분. `world_model/data_rssm_3ball/`에 chunk 2개(pkl) + `metadata.json` 저장.

- 사전 측정 추정(quota=1000 → ~25분, ~751,000 에피소드)과 실측(27.4분, 812,540 에피소드)이 거의 일치 — 확률 추정이 신뢰할 만함을 확인.
- 버그 발견 및 수정: `generate_balanced()`가 `list[ShotData]`만 반환해서, `main()`의 통계/메타데이터 출력이 실제 시도 횟수(812,540) 대신 미사용 CLI 기본값(`--n-episodes` 기본 50,000)을 그대로 찍고 있었음 — `n_episodes` 필드가 완전히 틀린 값이었음. `generate_balanced()`가 `(shots, total_attempts)` 튜플을 반환하도록 수정하고, `main()`/metadata에 실제 시도 횟수를 쓰도록 고침. 이미 저장된 `metadata.json`의 `n_episodes` 값도 812540으로 수동 정정.
- 이 데이터셋은 `n_balls=2`(cue+target 2개) 기준이며, roadmap ④의 GNN N=3 확장 작업에서 학습 데이터로 사용 예정.

### "어느 공이 포켓될지" 예측 검증 계획 (2026-09-29)

**배경**: 사용자 질문 — 지금 R-SSM이 "포켓이 일어날지"뿐 아니라 "어느 공이 포켓될지"도 예측할 수 있는가? 코드 확인 결과 `RSSMModel.predict_pocket(h)`(`rssm_model.py:259-273`)는 공 전체를 뭉뚱그리지 않고 **공마다 자기 latent `h[i]`를 받아 독립적인 시그모이드 확률을 출력**한다 — 구조적으로는 이미 공별 예측이 가능하다. `init_hidden(n_balls)`도 공 개수에 가변적이라 모델 자체는 N-ball 공유 가중치 구조(공 개수와 무관하게 동일 MLP 재사용).

**문제**: 지금까지 학습된 체크포인트(`rssm_v4`, `rssm_v5`)는 전부 `n_balls=1`(큐+타깃 1개) 데이터로만 학습됨 — 타깃이 하나뿐이라 "포켓될 확률"과 "어느 공이 포켓될지"가 사실상 동일한 질문이었다. 여러 타깃 중 어느 것이 포켓될지 구별하는 능력은 한 번도 검증된 적이 없다. 방금 만든 3-ball 데이터셋(`world_model/data_rssm_3ball/`, n_pocketed 버킷별 1000샷)이 이걸 검증할 첫 기회.

**추가 발견**: 기존 진단 스크립트 `world_model/eval_pocket_head.py`가 `ev.node_i`/`ev.node_j`/`ev.edge`를 pkl에서 바로 읽어 쓰는데, 이 필드들은 이전 세션의 pkl 용량 최적화로 항상 `None`으로 저장되도록 바뀌었다(`train_rssm.py`는 `make_node`/`make_edge`로 재구성하는 패턴으로 이미 대응했지만 `eval_pocket_head.py`는 그때 안 고쳐짐) — 지금 그대로 돌리면 `NoneType.to()` 에러로 즉시 크래시하는 stale 버그. 이번 작업의 선행 조건으로 수정 필요.

**계획**:
1. `eval_pocket_head.py`의 `collect_pocket_preds`/`collect_pocket_preds_all_touches`를 `make_node`/`make_edge` 재구성 패턴으로 수정 (train_rssm.py:754-761과 동일 패턴).
2. 같은 샷 안에서 공끼리 비교하는 "which-ball" 판별력 지표 추가 — 기존 AUC는 서로 다른 샷의 공들을 전부 풀에 섞어서 비교하므로 "이 샷에서 어느 공이 포켓될지" 질문과는 다르다. 정확히 타깃 1개만 포켓되는 샷에서, 포켓된 타깃의 `predict_pocket` 확률이 포켓 안 된 타깃보다 높은 비율(same-shot pairwise accuracy)을 측정.
3. 기존 `rssm_v5/best.pt`(n_balls=1로만 학습된 체크포인트)를 3-ball 데이터에 **재학습 없이 zero-shot으로** 돌려 1차 진단 — 공유 가중치 구조라 코드 변경 없이 돌아갈 것으로 예상되지만, 멀티볼 충돌 동역학은 학습한 적이 없어 성능은 낮을 가능성 있음. 이 결과를 보고 재학습 필요 여부 결정.

**실행 결과** (`rssm_v5/best.pt`를 `world_model/data_rssm_3ball` 전체 3000샷에 zero-shot 평가, `--n-shots-train 0 --n-shots-val 3000`):

| 지표 | n_balls=1 데이터 (기존, 참고) | n_balls=2 데이터 (zero-shot) |
|------|------|------|
| First-touch AUC (genuine) | 0.923 | **0.487** (거의 chance, geometric baseline 0.622보다도 낮음) |
| Last real touch AUC | (미기록) | 0.882 |
| Which-ball top-1 accuracy | N/A (타깃 1개뿐) | **0.795** (chance=0.5) |
| Which-ball pairwise accuracy | N/A | **0.795** |

**결론 — 사용자 질문("어느 공이 포켓될지 예측 가능한가")에 대한 직접 답변**: **가능하다, 그것도 재학습 없이 zero-shot으로.** 정확히 타깃 1개만 포켓되는 1000개 샷에서, 두 타깃의 first-touch `h`를 같은 샷 안에서 비교했을 때 실제로 포켓된 쪽을 79.5% 확률로 더 높게 평가한다 — 동전 던지기(50%)보다 확연히 위, 그리고 `rssm_v4`/`v5`는 애초에 타깃이 1개뿐인 데이터로만 학습되어 2-타깃 상황을 한 번도 본 적이 없다는 점을 감안하면 의미 있는 일반화.

**단, 중요한 비대칭 발견**: 같은 first-touch `h`인데도 **풀링된(pooled) AUC는 0.487로 사실상 랜덤**이면서 **같은 샷 내부 상대 비교(which-ball)는 0.795로 강한 신호**를 보인다. 즉 모델은 "이 공이 절대적으로 포켓될 확률"을 서로 다른 샷 사이에서 비교 가능하게 캘리브레이션하는 데는 실패했지만(아마 n_balls=1→2 전환으로 입력 분포가 달라져 절대 확률 스케일이 틀어짐), "같은 샷 안에서 어느 쪽이 더 유력한가"라는 상대적 랭킹 정보는 latent에 살아있다. 이건 실전에서 유용한 형태이기도 하다 — 실제로 필요한 건 "이 샷에서 어느 공을 노려야 하는가"라는 상대 비교지, 샷을 넘나드는 절대 확률 비교가 아니기 때문.

부가 관찰: last-real-touch AUC(0.882)가 first-touch AUC(0.487)보다 훨씬 높다 — 샷이 진행되며(공끼리 부딪히고 방향이 정해지며) 예측이 크게 개선된다는 뜻으로, n_balls=1 결과에서 봤던 패턴과 일치.

**추가 검증: "몇 개나 포켓되냐" (개수 예측)**: 각 타깃 공의 확률을 0.5로 잘라 "포켓됨/안됨" 판정한 뒤 개수를 세서 실제 포켓 개수(0/1/2, 각 1000샷)와 비교.
- 정확히 맞춘 비율: **41.9%** (1256/3000). 균등분포 기준 랜덤 찍기(33.3%)보다는 낫지만 크게 안 좋음.
- confusion matrix를 보면 모델이 "0개"로 예측하는 쪽에 심하게 쏠려있음 — 실제 2개 포켓 샷의 64%(641/1000)를 "0개"로 오판.
- 절대 확률 캘리브레이션 실패(위 pooled AUC 0.487)가 그대로 드러남: 상대 비교엔 강하지만 개별 임계값(0.5) 판정에는 약함.

**추가 검증: "이번 턴에 곧 포켓되냐 / 다음 2턴 안에 포켓되냐" (근시간 예측)**: `predict_pocket(h)`는 원래 "이 샷이 끝날 때까지 결국 포켓되는가"를 예측하도록 학습된 head인데, 같은 출력을 "바로 다음 이벤트가 포켓 이벤트인가"(label_now) / "다음 2개 이벤트 안에 포켓이 있는가"(label_w2)라는 다른 질문에 그대로 재활용했을 때 얼마나 맞는지 사후 검증.

| 데이터 | label_now AUC | label_w2 AUC |
|---|---|---|
| n_balls=1 (학습 도메인, world_model/data_rssm 5000샷) | **0.944** | **0.927** |
| n_balls=2 (3-ball, zero-shot) | 0.655 | 0.612 |

학습 도메인(n_balls=1)에서는 "곧 포켓되나"도 강하게 맞춘다 — threshold=0.5 기준 precision 58%/recall 88%(label_now), precision 72%/recall 79%(label_w2). 하지만 3-ball로 넘어가면 랜덤보다는 낫지만 확 떨어진다(0.655, 0.612). "몇 개냐"와 같은 패턴: **공유 가중치 구조라 코드 변경 없이 굴러가지만, 공 2개가 상호작용하는 물리 상황 자체를 한 번도 학습한 적이 없어서 나오는 일반화 실패**로 해석.

**종합 결론**: n_balls=1→2 전환에서 일관되게 보이는 패턴은 "같은 샷 내부 상대 비교"는 zero-shot으로도 버티고(79.5%), "절대 기준 판정"(포켓 확률 몇 %, 개수, 곧 되나)은 모두 무너진다는 것. 구조적 결함이 아니라 **2-타깃 상호작용을 본 적 없는 학습 데이터의 한계**로 판단, 3-ball 데이터로 재학습 진행 결정.

### 3-ball 데이터로 재학습 (2026-09-29)

**가설**: 위에서 확인한 절대 확률 캘리브레이션 실패(pooled AUC, 개수 판정, 근시간 예측)는 모델 구조 문제가 아니라 학습 데이터가 n_balls=1(타깃 1개)뿐이라 2-타깃 상호작용을 한 번도 못 봐서 생긴 일반화 실패다. `world_model/data_rssm_3ball`(3000샷, n_balls=2)로 처음부터 재학습하면 이 지표들이 회복될 것이다.

**데이터 순서 이슈 발견**: `generate_balanced()`가 저장한 pkl은 포켓 개수 버킷(0/1/2) 순서로 쭉 이어붙여 저장됨(0-포켓 1000개 → 1-포켓 1000개 → 2-포켓 1000개). `train_rssm.py`의 train/val 분리는 앞쪽 N개를 train, 그 다음 M개를 val로 단순 슬라이싱하는 방식이라, 이 순서 그대로 쓰면 val이 특정 버킷(예: 2-포켓)에만 쏠리는 non-stratified split이 됨. `train_rssm.py` 코드를 건드리지 않고, 로드 후 seed 42로 셔플한 사본을 `world_model/data_rssm_3ball_shuffled/`에 새로 저장해서 우회.

**설정**: `--n-balls 2 --data-dir world_model/data_rssm_3ball_shuffled --n-shots-train 2400 --n-shots-val 600 --out-dir world_model/results/rssm_v7_3ball`. 처음부터 새로 학습(fine-tune 아님 — `train_rssm.py`에 체크포인트에서 이어 학습하는 기능이 없음).

**학습 결과**: 500 epoch 전부 완료. `best_val_rmse=6.044`(epoch 450), val loss 21.07→6.05로 하락. `type_acc`는 epoch 90-140 부근 0.66-0.67로 정점을 찍은 뒤 500epoch 시점 0.593까지 완만히 하락 — 동시에 magnitude class weight(`kw`)가 [0.14, 20.12]까지 커짐. 희귀 속도-구간 가중치가 후반부에 과도해져 type loss와 밸런스가 흔들렸을 가능성이 있음(원인 미확정, 후속 조사 필요).

**검증 결과 — n_balls=1 학습 모델(zero-shot) vs 3-ball 재학습 모델, 동일 3-ball held-out val(600샷)로 비교**:

| 지표 | zero-shot (rssm_v5) | 3-ball 재학습 (rssm_v7_3ball) |
|---|---|---|
| Pooled AUC (포켓될 확률, 볼 풀링) | 0.487 | 0.894 |
| Which-ball top-1/pairwise acc | 0.795 | 0.818 |
| 개수 예측 정확도 (exact match) | 0.419 | 0.757 |
| 근시간 예측 label_now AUC | 0.655 | 0.811 |
| 근시간 예측 label_w2 AUC | 0.612 | 0.837 |

가설 확인됨: 절대 기준 판정 실패는 구조 문제가 아니라 데이터 커버리지 문제였음. 3-ball 데이터로 재학습하자 모든 절대-기준 지표(pooled AUC, 개수, 근시간)가 큰 폭으로 회복. Which-ball은 zero-shot에서도 이미 강했던 만큼 소폭 개선에 그침 — 예상과 일치.

**남은 과제**: 개수 예측은 75.7%로 개선됐지만 여전히 완벽하지 않음 — confusion을 보면 실제 1개 포켓 상황을 2개로 과대 예측하는 오류(72/203)가 주된 오차원. `type_acc` 후반부 하락 원인 조사는 다음 실험으로 이월.

**`type_acc` 후반부 하락 원인 조사 결과**: `train_rssm.py`는 Kendall uncertainty weighting을 씀(`kw_v=exp(-log_var_vel)`, `kw_t=exp(-log_var_type)`, `log_var`는 [-3, 2]로 clamp). epoch 400 이후 로그의 `kw=[0.14, 20.12]`는 `log_var_type`이 clamp 하한(-3)에 완전히 눌린 것(`exp(3)=20.09`). 원인은 kw 자체가 아니라 **epoch 350 이후 train/val loss가 갈라지는 순수 overfitting**: val loss는 6.0~6.1에서 정체하는데 train loss는 계속 떨어짐(epoch350 train=1.68 → epoch500 train=1.13). train 기준 type loss가 작아지니 Kendall weighting이 "이 태스크 자신있다"고 판단해서 `kw_t`를 clamp 상한까지 밀어올리고, 그 사이 val type_acc는 0.665(epoch150) → 0.587(epoch400)로 하락. `best.pt`는 val_rmse 최소 기준(epoch450)으로 저장되는데, 이 지점은 이미 type_acc 정점(epoch150, 0.665) 대비 0.08 낮은 상태 — 체크포인트 선택 기준이 val_rmse 단일값이라 이런 트레이드오프가 반영 안 됨. (대안: 이미 저장된 `epoch_0150.pt`/`epoch_0200.pt`를 실제 다운스트림 지표로 재평가해서 best.pt 대신 쓰는 방법이 재학습 없이 가장 쌈. 아직 미실행.)

**이벤트별 포켓 확률 궤적 분석 (rssm_v7_3ball, 2026-09-30)**: `predict_pocket(h)`를 매 이벤트마다 찍어서 시간순으로 보면, 쿠션·충돌 같은 물리 이벤트에 반응해서 확률이 실제로 오르내리긴 하지만(예: 쿠션 맞고 0.435→0.091로 급락하는 정상 케이스도 있음), 방향이 통째로 뒤집힌 confident-wrong 케이스가 섞여 있음(정답=True인 공이 0.27~0.42로 계속 낮게 깔리는 동안, 같은 샷의 정답=False인 공은 0.97~0.998로 계속 확신).

틀린 케이스(FP/FN)만 따로 뽑아 특징 비교(600샷 val, n=745, 마지막 진짜 터치 기준):

| 그룹 | n | 평균확률 | 총터치수 | 쿠션횟수 | 큐볼충돌 |
|---|---|---|---|---|---|
| TP | 504 | 0.909 | 3.86 | 1.60 | 0.91 |
| TN | 102 | 0.152 | 3.20 | 1.93 | 1.01 |
| FP (오탐) | 60 | 0.845 | 3.93 | **2.48** | **1.12** |
| FN (놓침) | 79 | 0.208 | **4.10** | 1.90 | 0.91 |

FP는 쿠션 횟수가 전체 그룹 중 최다(2.48회) — 기하학적으로는 포켓 방향이 아닌데(baseline -0.2, TN과 동일) 쿠션을 여러 번 맞으면 과대평가하는 경향. FN은 총 터치수가 최다(4.10회) — 경로가 길고 복잡할수록 확신을 못 세움.

쿠션 횟수별 accuracy(전체 745개)는 매끄럽게 나빠지는 게 아니라 **3회에서 절벽처럼 떨어짐**: 0회 0.850 / 1회 0.836 / 2회 0.856 / 3회 **0.690** / 4회+ 0.760.

틀린 케이스를 쿠션 단위로 끊어서 시간순으로 보면, 새 쿠션 이벤트가 들어와도 확률이 **직전 값 근처에 눌어붙어서 잘 안 움직임** — 특히 FP는 초반에 한 번 0.99대로 확신하면 이후 쿠션을 몇 번 더 맞아도 거의 안 내려옴(예: 0.998→0.974, 0.993→0.989). FN 쪽은 증거가 쌓일수록 오히려 정답에서 더 멀어지는 경우도 있음(0.421→0.322, 정답=True인데 하락). **새 증거를 반영해서 갱신(update)하는 게 아니라 이전 판단을 그대로 우려먹는 것처럼 보임.**

**다음 방향 제안 — prior-posterior 구조**: 지금은 매 이벤트마다 `h → MLP → 확률`로 처음부터 다시 계산하는 구조라, 이전 판단을 새 증거로 갱신하는 메커니즘이 없음. "직전 확률(prior) + 이번 이벤트 증거(likelihood) → 갱신된 확률(posterior)"를 log-odds 덧셈 형태로 명시적으로 학습시키면 이벤트마다 일관되게 업데이트되도록 강제할 수 있을 것으로 기대. 단, 헤드 구조 자체를 재설계하고 재학습해야 해서 작업량이 큼 — 아직 설계 전, 다음 실험 후보.

**이 작업 전체의 동기**: pocket head를 이렇게 깊게 파는 이유는 최종적으로 **SAC critic의 Q-value 추정에 이 world model을 쓰기 위함**(exp16 wm-critic). 포켓 확률이 이벤트마다 제대로 갱신되지 않으면 critic이 받는 상태 표현(h) 자체의 신뢰도가 낮아지므로, RL 통합 전에 이 캘리브레이션 문제를 먼저 해결할 필요가 있음.

### 학습 데이터 버그 발견: target-target ball_ball collision normal 자리표시자 고정 → 수정 + 재학습 (rssm_v8_3ball, 2026-09-30)

**발견 경위**: SAC+R-SSM 통합(exp16 wm-critic, MBPO식 짧은 상상 롤아웃)을 설계하던 중, 그 전제인
`RolloutEngine`의 물리 엔진(`event_detector.py`)이 cue+타깃1(2공) 전용으로 하드코딩되어 있어
N-ball 일반화가 먼저 필요하다는 걸 확인했다. 이 조사 과정에서 실제 **학습 데이터 생성 경로**
(`rssm_dataset.py::_get_raw_type_and_normal`)에도 동일한 가정(ball_ball 충돌 중 하나는 반드시
cue)이 박혀 있는 걸 추가로 발견했다.

**원인**: `_get_raw_type_and_normal()`의 `ball_ball` 분기가 `cue_a = next(a.id=="cue")`,
`obj_a = next(a.id!="cue")`로 두 충돌 볼을 찾은 뒤 `cue_a is not None`일 때만 실제 contact
normal(`_contact_normal_ball_ball`)을 계산했다. 타깃-타깃 충돌(둘 다 non-cue, n_balls≥2에서는
흔함)은 이 조건을 만족하지 못해 `normal`이 함수 상단 기본값 `[1.0, 0.0]`(자리표시자)으로 남았다.
`data_rssm_3ball_shuffled`(rssm_v7_3ball의 실제 학습 데이터)를 직접 세어보니 전체 ball_ball
이벤트 4210개 중 711개(16.9%)가 타깃-타깃이었고, **711개 전부** normal이 `[1.0, 0.0]`으로
고정되어 있었다. 이 normal은 GNN의 edge feature로 직접 들어가므로, `rssm_v7_3ball`은 3-ball
데이터의 ball_ball 이벤트 중 약 17%를 물리적으로 틀린 edge 신호로 학습한 셈.

**수정**: 두 충돌 볼을 "cue vs non-cue"가 아니라 정렬 키(`(0,"") if id=="cue" else (1,int(id))`)로
뽑아 동일한 코드 경로에서 normal을 계산하도록 일반화. cue는 항상 낮은 정렬 키를 가지므로
기존 cue→target 방향 컨벤션은 그대로 유지되고, target-target 쌍도 같은 경로로 올바른 normal을
얻는다. 회귀 테스트(`tests/test_rssm_dataset.py::TestBallBallNormal`) 2개 추가: 타깃-타깃 normal이
더 이상 placeholder가 아님을 확인하는 테스트, cue-타깃 normal이 수정 전후 동일함을 확인하는
테스트. 전체 `pytest tests/ -v` 201 passed(기존에도 있던 시드 의존 flaky 테스트 1개는 무관,
`git stash`로 수정 전 코드에서도 동일하게 실패함을 확인).

**재생성 + 재학습**: 기존 `data_rssm_3ball`과 동일 설정(`--balanced --quota0 1000 --quota1 1000
--quota2 1000 --n-balls 2 --policy random --seed 42`)으로 `data_rssm_3ball_v2` 재생성
(819,520 attempts, 28.0분 — 원본 812,540/27.4분과 거의 동일, 설정이 정확히 재현됐음을 확인),
seed 42로 셔플한 사본(`data_rssm_3ball_v2_shuffled`)을 만들어 확인해보니 target-target
ball_ball 이벤트 720개 전부 더 이상 placeholder normal이 아님을 재확인. `rssm_v7_3ball`과
동일한 하이퍼파라미터(`--n-balls 2 --n-shots-train 2400 --n-shots-val 600`, 500 epoch)로
`rssm_v8_3ball`을 처음부터 재학습(파라미터 수 595,919로 v7과 동일, fine-tune 아님).

**v7 vs v8 비교 결과** (동일 600샷 held-out val, `eval_pocket_head.py` + target-target 전용 별도 스크립트):

| 지표 | v7 (버그 데이터) | v8 (수정 데이터) |
|---|---|---|
| 최종 val loss (epoch 500) | 6.052 | 4.834 |
| ball_ball 서브로스 | 2.048 | 2.059 |
| type_acc (전체) | 0.593 | 0.584 |
| pock_acc (전체) | 0.843 | 0.848 |
| Pooled pocket AUC | 0.894 | 0.906 |
| Which-ball top-1/pairwise acc | 0.818 | 0.862 |
| Pairwise rescue rate | 90.6% | 91.1% |
| **target-target ball_ball만: type_acc** | 0.447 (n=228) | 0.464 (n=261) |
| **target-target ball_ball만: delta RMSE** | 3.026 | 2.601 |

**해석**: 버그의 영향을 가장 직접적으로 받는 target-target ball_ball 이벤트에서 delta RMSE가
14% 개선(3.026→2.601)됐고 type_acc도 소폭 개선(0.447→0.464) — 버그 수정이 해당 이벤트에
실제로 도움이 됐다는 직접 증거. 흥미로운 건 **전체 val loss도 6.05→4.83으로 20% 가까이
줄었는데(cue_circ 13.40→10.85, cue_linear 6.95→5.87, pocket 9.07→6.88, tgt_circ 13.35→9.44,
tgt_linear 10.63→7.22 등 ball_ball 이외의 거의 모든 서브로스가 함께 개선)**, 이건 target-target
edge의 틀린 신호가 GNN의 공유 표현(shared hidden state)을 오염시켜 ball_ball과 무관한 다른
이벤트 타입 예측에도 전반적으로 영향을 줬을 가능성을 시사한다. Which-ball 판별력(top-1
0.818→0.862)도 개선폭이 커서, 3-ball 이상에서 "어느 공이 포켓될지" 판별이 이 버그로 인해
저평가되고 있었을 수 있다. 반면 전체 pooled `type_acc`(0.593→0.584)는 오히려 소폭 하락했는데,
이는 이전 실험에서 이미 진단된 "후반부 순수 overfitting + Kendall weighting이 `kw_t`를 clamp
상한까지 밀어올리는" 현상(§ 위 "`type_acc` 후반부 하락 원인 조사 결과")과 별개 이슈로, 이번
버그 수정과는 직접 관련 없는 것으로 판단.

**결론**: 데이터 버그가 실제로 존재했고, target-target 이벤트 및 which-ball 판별력에 측정
가능한 부정적 영향을 주고 있었음을 확인. `rssm_v8_3ball`을 새 기준 체크포인트로 채택.
SAC/MBPO 통합(EventDetector N-ball 일반화 포함)은 이 작업 완료 후 별도 브랜치에서 재개.

### EventDetector N-ball 일반화 (SAC/MBPO 통합 선결 작업, 2026-09-30)

**계획**: `world_model/event_detector.py::EventDetector`가 cue+타깃1(2공) 전용으로 하드코딩되어
있어(`has_cue`/`has_tgt = "1" in ball_ids`, `system.balls["cue"]`/`system.balls["1"]` 직접 참조),
`RolloutEngine`(`world_model/rssm_rollout.py`)을 n_balls≥2에 쓸 수 없다. `RolloutEngine` 자체는
이미 `ball_ids`/`id_to_idx` 기반으로 N-ball 일반적으로 짜여 있어(확인 완료), `EventDetector`만
고치면 파이프라인 전체가 N-ball을 지원하게 된다.

**수정 방향**: `rssm_dataset.py` 버그 수정 때 쓴 것과 동일한 정렬 키 패턴(`(0,"") if id=="cue"
else (1,int(id))`)을 적용. `next_event()`/`full_event_sequence()`에 중복 구현된 이벤트 분류
로직(ball_ball normal 계산, cushion/pocket normal 계산, TGT_LINEAR/TGT_CIRCULAR 재분류)을
`_classify_event()` 헬퍼로 통합해서 두 메서드가 같은 코드 경로를 쓰게 한다 — 이전에 `rssm_dataset.py`
버그처럼 한쪽만 고치고 다른 쪽을 놓치는 걸 구조적으로 방지하기 위함. cushion/pocket 등 단일-볼
이벤트의 `has_cue`/`has_tgt` 재분류 조건도 `not has_cue`만으로 일반화(기존엔 `not has_cue and
has_tgt`였는데, N-ball에서 has_tgt는 항상 참이므로 불필요).

**범위**: `EventDetector`만 수정. `RolloutEngine`/`rssm_dataset.py`는 이미 대응됐거나 이번 작업
대상 아님. SAC critic 연결(Option A/B/C)은 이 작업 이후 별도 범위.

**구현 결과**: 계획대로 `_ball_sort_key()` + `_classify_event()` 헬퍼로 리팩터링 완료
(`feature/wm-eventdetector-nball` 브랜치, `world_model/event_detector.py`). `next_event()`와
`full_event_sequence()` 둘 다 `_classify_event()`를 통해 같은 분류 로직을 공유하도록 통일했고,
`full_event_sequence()`의 "이벤트 참여 볼이 아직 테이블에 있는지" 체크도 단일 타깃 전용에서
`ball_ids.issubset(system.balls.keys())`로 일반화(포켓된 볼은 시뮬레이션 종료 시점에
`system.balls`에서 제거되므로).

**테스트**: `tests/test_event_detector.py`에 `TestEventDetectorNBall` 추가.
- `test_target_target_normal_not_placeholder`: n_targets=3, 60개 시드로 스캔해서 타깃-타깃
  ball_ball 충돌을 최소 1개 찾고, 그 normal이 예전 placeholder `[1.0, 0.0]`이 아니라 실제
  unit-length 벡터인지 확인. (n_targets=2로는 60시드 스캔해도 타깃-타깃 충돌이 0건이라 비현실적
  — 3-ball에서야 관측 빈도가 충분해서 n_targets=3으로 전환.)
- `test_ball_sort_key_orders_cue_first_then_numeric`: `_ball_sort_key`가 cue를 항상 먼저, 나머지는
  숫자 순(문자열 순 아님, "2" < "10")으로 정렬하는지 직접 검증.
- `test_cushion_reclassification_applies_to_any_target`: TGT_LINEAR/TGT_CIRCULAR 재분류가 예전엔
  볼 "1"에만 하드코딩돼 있었는데, 볼 "2"에서도 발생하는지 확인.
- 기존 11개 2-ball 테스트 전부 회귀 없이 통과. `pytest tests/ -v` 전체 204 passed, 1 failed
  (`TestClassWeights::test_weights_positive` — 이번 변경과 무관한 기존 flaky 테스트, `n_shots=10`
  샘플에서 특정 이벤트 타입이 0건이 되는 경우가 있어 발생, 이전부터 알려진 이슈).

**결론**: `EventDetector`가 이제 N-ball(cue + 임의 개수 타깃)을 완전히 지원한다. 이로써
`RolloutEngine`을 n_balls≥2에서 쓸 수 있는 마지막 선결 조건이 해소됐다. 다음 단계는 SAC critic에
R-SSM 상상 롤아웃을 연결하는 실제 통합(Option A/B/C, [[project_rssm_rl_integration_open_question]]
참고 — WM frozen vs continual fine-tune 결정 필요)이며, 별도 브랜치에서 진행한다.

### pocket head 포화(saturation) 진단 + 최소 개입 비교 실험 계획 (2026-10-01)

**배경**: "pocket head 캘리브레이션 진단"(위 2026-09-30 항목)에서 확률이 새 증거에도 안 바뀌는
현상을 발견했었다. `rssm_v8_3ball`의 주기적 체크포인트(50 epoch 간격)를 가지고 값싼 추가 진단을
세 가지 해봤다(재학습 없이 기존 체크포인트만으로):

1. **epoch150(peak type_acc≈0.665) vs best.pt(epoch460, val_rmse 최소지만 type_acc는 0.584로
   하락) 반응성 비교**: 이벤트 간 확률 변화 `|Δp|`의 median이 epoch150=0.026, best.pt=0.005로
   후반부가 더 심하게 얼어붙음(거의 안 움직이는 비율 35%→56%). 즉 문제가 학습 초반부터 고정된
   게 아니라 **학습이 진행될수록(특히 오버피팅 구간에서) 악화**된다.
2. **h 자체의 변화량(‖Δh‖) vs 확률 변화량(|Δp|) 분리**: 두 체크포인트 모두 `corr(|Δp|, ‖Δh‖)
   ≈ 0`(-0.045, -0.016)이고, 확률이 "얼어있을 때"와 "움직일 때"의 평균 ‖Δh‖가 거의 동일
   (0.972 vs 1.015 / 0.979 vs 1.026). 즉 **h는 매 이벤트마다 정상적으로 갱신되고 있고, 문제는
   `predict_pocket` head의 출력 단에 있다** — RolloutEngine/R-SSM 몸통(GNN+SSM 갱신 구조) 자체는
   문제가 아닐 가능성이 높다.
3. **직전 확률의 극단성(|p-0.5|)별 stuck 비율**: 확률이 아직 0.5 근처(미결정)일 때는 stuck
   비율이 1~4%로 낮지만, 이미 0/1 근처(확신)로 가 있으면 47~65%로 급증하고 이 경향은 학습이
   진행될수록(best.pt) 더 심해진다. 이는 **sigmoid 포화**의 전형적 서명이다.

**원인 가설**: `pocket_mlp`(h_dim→h_dim//2→1, `rssm_model.py:115`)는 표준 BCE
(`F.binary_cross_entropy`, `train_rssm.py:477`/`572`)로 0/1 하드 라벨을 맞히도록 학습된다.
BCE+하드라벨 조합은 이미 맞힌 샘플도 계속 더 확신하도록 무한히 gradient를 미는 성질이 있어서
(loss가 0에 완전히 수렴하지 않는 한), 500 epoch처럼 오래 학습하면 logit이 계속 커지다가
sigmoid가 포화된다 — 확인해보니 전역 `weight_decay=1e-4`(AdamW, 전체 파라미터 공통,
`train_rssm.py:82,1040`)가 이미 걸려있는데도 이 정도로는 억제가 안 된 것으로 보인다.

**계획 — 최소 개입 2개만 비교 (그리드 서치 안 함)**: 몸통을 갈아엎는 prior-posterior 재설계는
비용이 크므로, head 출력단만 건드리는 두 가지 표준 처방을 하이퍼파라미터 하나씩만 정해서
(문헌 기본값 그대로, 탐색 없이) 시험한다.

1. **Label smoothing (`rssm_v9_ls01_3ball`)**: pocket BCE 타깃을 `y' = y·(1-ε) + (1-y)·ε`로 완화.
   `ε=0.1`(Szegedy et al. 2016 표준 기본값) 단일값만 시험 — "확률이 0.9/0.1에 도달하면 이미
   목표 달성"이 되어 그 이상 확신을 밀어붙일 gradient가 사라지는 원리.
2. **Head 전용 weight decay (`rssm_v9_headwd1e2_3ball`)**: `pocket_mlp` 파라미터만 별도
   optimizer param group으로 분리해서 나머지 모델과 다른(더 큰) weight_decay를 준다. 전역
   1e-4가 이미 걸려있는데도 불충분했으므로, 두 자릿수 위인 `1e-2` 단일값만 시험 — head 가중치가
   커지는 것 자체를 억제해서 logit 폭주를 막는 원리. (그래도 안 잡히면 다음 후보는 1e-1이지만,
   이번 라운드에서는 시도하지 않는다.)

각 실험은 값 하나씩만 시험하고, 두 개입을 합치는 실험(`ls+headwd`)은 각각 단독으로 먼저 효과가
있는지 확인한 뒤 필요하면 추가한다. 나머지 설정(데이터, epoch 수, LR 스케줄 등)은 `rssm_v8_3ball`과
완전히 동일하게 유지해 다른 변수를 섞지 않는다.

**평가 기준**: 위 진단에서 쓴 것과 같은 3가지 값싼 지표(반응성 median |Δp|, stuck 비율, 극단
구간별 stuck 비율)를 새 체크포인트에 다시 돌려서 개선 여부 확인. 부작용 체크로 `type_acc`,
pooled pocket AUC, which-ball top-1도 `rssm_v8_3ball` 대비 퇴보하지 않는지 같이 본다.

**결과 (2026-10-01)**: `rssm_v8_3ball`과 동일 설정(`--n-balls 2 --data-dir
world_model/data_rssm_3ball_v2_shuffled --n-shots-train 2400 --n-shots-val 600`, 500 epoch)으로
두 실험 모두 처음부터 재학습 완료. `best.pt` 기준(각각 val_rmse 최소 epoch: v8=460, v9_ls01=440,
v9_headwd1e2=480) 진단 스크립트 3종 + `eval_pocket_head.py`를 동일 600샷 held-out val에 재실행:

| 지표 | v8 (기준) | v9_ls01 (label smoothing ε=0.1) | v9_headwd1e2 (head weight_decay=1e-2) |
|---|---|---|---|
| stuck 비율 (전체, \|Δp\|<0.01) | 55.6% | **12.7%** | 53.8% |
| median \|Δp\| | 0.0054 | **0.0473** | 0.0068 |
| stuck 비율 (직전 확률이 0/1 근처 "확신" 구간) | 64.5% | **19.2%** | 62.1% |
| pocket AUC (genuine, first-touch) | 0.906 | 0.903 | 0.888 |
| which-ball top-1 | 0.862 | **0.882** | 0.847 |
| type_acc (val) | 0.583 | 0.591 | 0.587 |
| pock_acc (val, @0.5) | 0.847 | 0.862 | 0.852 |
| best val_rmse | 4.841 | 4.953 (+2.3%) | 4.934 (+1.9%) |

**결론**: **Label smoothing이 확실한 승자.** 포화 지표(확신 구간 stuck 비율)가 64.5%→19.2%로
거의 3.4배 개선되었고, 부작용 지표(AUC/which-ball/type_acc/pock_acc)는 전혀 퇴보하지 않았으며
오히려 which-ball top-1(+0.02)과 type_acc(+0.008)가 미세하게 더 좋아졌다. val_rmse 손실은
+2.3%로 감내 가능한 수준. 반면 **head 전용 weight_decay=1e-2는 사실상 효과가 없었다**(stuck
비율 64.5%→62.1%, 거의 그대로) — 전역 weight_decay(1e-4)를 이미 훨씬 웃도는 값인데도 로짓
폭주를 억제하지 못했다는 뜻이므로, 계획에서 언급한 다음 후보(1e-1)로도 개선폭이 클 것으로
기대하기 어렵다고 판단해 추가 시도하지 않기로 함. 원인 가설과 일치하는 결과: **BCE+하드라벨의
"과확신을 향한 무한 gradient" 문제는 타깃 쪽을 누그러뜨리는 label smoothing으로 직접 없애는
것이 맞고, weight decay 같은 파라미터 크기 억제는 간접적이라 효과가 약했다.**

**결정**: `pocket_label_smoothing=0.1`을 앞으로 R-SSM pocket head 학습의 기본값으로 채택.
`rssm_v9_ls01_3ball`을 pocket head 신뢰도가 중요한 후속 작업(SAC critic 통합 등)의 새 기준
체크포인트로 삼는다. 비용이 큰 prior-posterior 재설계는 이번 라운드에서는 불필요 — head 레벨
개입만으로 포화 문제의 대부분이 해소됨을 확인했다.

### MBPO Option A: 실제 관측 샷을 frozen R-SSM으로 인코딩해 SAC critic에 연결 (2026-10-01)

**배경**: `world_model/rssm_model.py`의 QHead(`aggregate_q`, attention-pool → scalar Q)는
forward pass는 완전히 배선되어 있지만(`rssm_rollout.py:252`) 학습 신호가 전혀 없다
(`train_rssm.py`에 `q_head`/`aggregate_q` 참조 전무). Q-label 소스 후보 3개(MC return / SAC
critic bootstrap / `pocket_prob` 휴리스틱) 중, `pocket_prob`은 AUC 0.919 중 0.827이 0-파라미터
geometric baseline으로 설명되는 leak이 위("AUC=0.919 Too good to be true?" 항목)에 문서화돼
있어 신뢰하기 어렵다.

한편 `exp16_wm/`에는 이미 **다른 형태의** world-model critic(`WMSAC`, `exp16_wm/sac.py`)이
동작 중이었다 — R-SSM과 무관한, 이번 조사로 처음 확인된 기존 자산이다: `WorldModelCritic`은
blind MLP `M(obs,action)→ĥ`가 실제 관측된 샷 궤적(`h_real`, pooltool 이벤트 로그에서 추출한
flat (x,y,event_type) 배열, `simulator.py::_extract_trajectory`)을 지도학습으로 맞추고
(`wm_loss`), `q(ĥ)→Q`는 Bellman loss로 학습된다.

**선택한 방향(Option A)**: 매 실제 env step 후 "방금 일어난 진짜 이벤트 시퀀스"를
frozen·pretrained R-SSM(`rssm_v9_ls01_3ball`)에 통과시켜 얻은 per-ball latent(`h_final`,
shape `(n_balls_total, H_DIM=64)`)를, 기존 WMSAC의 `h_real` 타깃 자리에 그대로 꽂아 넣는다.
R-SSM 백본은 freeze, `WorldModelCritic`의 M(blind MLP)과 q(Bellman)만 학습한다. 검토한 대안:

- **Option B (진짜 MBPO 상상 롤아웃)**: 매 transition마다 `RolloutEngine`으로 모델 자신의
  예측(ground truth 없이)으로 n-step 상상 롤아웃을 돌려 Q_imagined를 Bellman target에 섞거나
  synthetic (s,a,r,s')를 생성. 가장 roadmap 원안에 가깝지만 `EventDetector`가 순수
  Python/NumPy 루프라 배치/GPU 불가 — 512 배치 크기로 매 gradient step마다 호출하면 처리량
  병목이 될 게 뻔해서 엔지니어링 리스크가 크다. 다음 단계로 유보.
- **Option C (pocket_prob 휴리스틱 증강)**: `q_target = r + gamma*V(s') + lambda*pocket_prob`.
  구현은 가장 간단하지만 위에서 언급한 leak 문제로 신호 품질이 의심스러워 기각.

Option A는 `WMSAC`/`WorldModelCritic`/`TrajectoryReplayBuffer`가 이미 `(max_events,
event_dim)`에 대해 완전히 범용적으로 짜여 있어서, 타깃의 **의미**만 바꾸고 코드는 거의
그대로 재사용할 수 있다(`max_events` 자리에 `n_balls_total`, `event_dim` 자리에 `H_DIM`).
`buffer.py`/`sac.py`/`networks.py`는 전혀 수정하지 않는다 — `simulator.py`가 채우는
`info["h_real"]`/`info["traj_len"]`의 내용물만 blind-trajectory에서 R-SSM latent로 바뀐다.

**Q(s, 임의의 action) 문제**: Bellman target(`Q(next_obs, next_action)`)과 actor loss
(`Q(obs, actor_action)`)는 아직 실행되지 않은 가상의 action에 대한 Q가 필요한데, R-SSM은
실제로 일어난 샷만 인코딩할 수 있어 "미래의 가상 action"에 대해서는 돌릴 수 없다. 이 문제는
이미 `WMSAC` 구조가 풀어놓았다 — `M(obs,action)`은 임의의 (s,a)에 대해 ĥ를 "예측"하는 학습된
함수이고, 실제 h(이번엔 R-SSM latent)는 그 예측을 지도학습으로 맞추는 타깃일 뿐 forward
pass에 직접 쓰이지 않는다. 즉 `RolloutEngine`(Option B, EventDetector 루프라 배치/GPU 불가)은
이번 phase에서 전혀 필요 없다.

**설계 요약**:
1. `world_model/rssm_encode.py`(신규) — `load_frozen_rssm()`, `encode_shot_to_latent(model,
   system, ball_ids)`. `generate_shot_data()`(`rssm_dataset.py`) + `make_node`/`make_edge`
   (`rssm_rollout.py`) + `RSSMModel.forward()`(`rssm_model.py`)를 그대로 재사용, 새 로직은
   그 셋을 잇는 글루코드뿐.
2. `simulator.py` — `trajectory_in_info: bool` 플래그를 `wm_target: str`(`none/traj/rssm`)로
   교체. `rssm` 모드에서 모듈 레벨 lazy-load 캐시로 frozen R-SSM을 한 번만 로드하고,
   `info["h_real"]`/`info["traj_len"]`에 동일한 키로 R-SSM latent를 채운다.
3. `exp16_wm/train.py` — `--wm-target {traj,rssm}` CLI 옵션 추가, `rssm` 선택 시
   `max_events=n_balls_total`, `event_dim=H_DIM`으로 차원만 교체.
4. R-SSM continual fine-tuning 여부는 이번 phase에서 건드리지 않는다 — frozen으로 고정
   ([[project_rssm_rl_integration_open_question]] 미결 이슈 그대로 보류). R-SSM 자체의
   `aggregate_q`/`q_head`를 in-place로 학습시키지도 않는다 — `exp16_wm/` 격리 규칙상 공유
   체크포인트를 exp16 전용 목적으로 변형하지 않기 위함.

**범위**: 이번 PR은 "꽂아서 학습이 도는지" 스모크 테스트까지. `wm_target=rssm`과 기존
`wm_target=traj`의 본격적인 pocket-rate 성능 비교 학습은 다음 세션으로 분리한다.
브랜치 `feature/wm-rssm-critic-latent`, `dev`로만 머지.

**결과**: 설계대로 구현 완료.
- `world_model/rssm_encode.py` + `tests/test_rssm_encode.py`(7개 — happy path shape/finite,
  무이벤트 샷 zero latent, frozen 결정성, `requires_grad=False` 확인, `simulator.py`의
  `wm_target=rssm/traj/none` 세 경로 info 내용 확인) 전부 통과.
- `simulator.py`: `trajectory_in_info` → `wm_target` 교체, `rssm_checkpoint` 생성자 인자 추가
  (기본값 `world_model/results/rssm_v9_ls01_3ball/best.pt`), 모듈 레벨 `_RSSM_MODEL_CACHE`로
  env 인스턴스 간 체크포인트 공유.
- `exp16_wm/train.py`: `--wm-target {traj,rssm}` 추가, `wm_dims()` 헬퍼로 `WMSAC`/
  `TrajectoryReplayBuffer` 생성 시 차원 전환.
- 스모크런(`--agent wm --n-balls 3 --n-envs 2 --total-steps 2000`) 두 경로(`traj`/`rssm`)
  모두 에러/NaN 없이 완주, `critic_loss`가 유한값으로 수렴 추세 확인(rssm: 0.74→0.62,
  traj: 0.31→0.19 — 스텝 수가 매우 적어 절대값 비교는 무의미, "도는지"만 확인).
- `pytest tests/ -v` 217 passed, 1 failed(`TestClassWeights::test_weights_positive`) —
  이번 변경과 무관한 파일(`test_train_rssm.py`)의 기존 seed-순서-의존 flaky 테스트로 확인
  (격리 실행 시 통과/실패 들쭉날쭉, 이번 커밋들이 건드린 적 없는 코드).
- 본격적인 `traj` vs `rssm` pocket-rate 비교 학습(수십만 step)은 다음 세션으로 유보.

### `wm_target=traj` vs `wm_target=rssm` 본격 학습 비교 (2026-10-01)

**가설**: frozen R-SSM latent(`h_final`)를 `WMSAC`의 `h_real` 타깃으로 쓰면(= `rssm`),
blind (x,y,event_type) flat 인코딩(= `traj`) 대비 critic이 더 유용한 supervision을 받아
pocket rate/clear rate가 더 높아질 것이다.

**설정**: 기존 `exp16_vanilla_multi3_ms5_s0_2026-03-28@1656` 기준 설정(이미 established
baseline)과 동일하게 맞춰 3-way 비교 가능하도록 함 — `n_balls=3 max_steps=5
step_penalty=0.1 trunc_penalty=1.0 total_steps=2,000,000 n_envs=10 learning_starts=5000
eval_freq=1000 eval_episodes=50 seed=0`. 이 baseline의 결과: `random_pocket_rate=13.93%,
trained_pocket_rate=65.87%, clear_rate=32.2%` (vanilla SAC, world model 없음).

`--agent wm --wm-target traj --seed 0`과 `--agent wm --wm-target rssm --seed 0`을
동일 설정으로 순차 실행(CPU 10코어 환경이라 동시 실행 시 자원 경합으로 둘 다 느려지는 걸
피하기 위해 순차 실행 선택). 각 실행 예상 소요: vanilla 2M step 기준 69분(482 fps) 참고,
wm 쪽은 critic이 추가 forward를 돌지만 env step 자체는 동일해 비슷한 자릿수로 예상.

**비교 축**: `trained_pocket_rate`, `clear_rate`, `best_mean_reward` (vanilla 대비, 그리고
traj 대비 rssm). `critic_loss`/`wm_loss`/`bellman_loss` 수렴 추이도 wandb run으로 비교.

**범위**: 이번 실행은 seed=0 1개 시드만 — 노이즈 가능성이 있어 결론은 "경향 확인" 수준,
여러 시드 반복은 결과가 유의미해 보일 때 후속 작업으로. 브랜치
`feature/wm-traj-vs-rssm-run`, `dev`로만 머지.

**결과**:

| | pocket rate | clear rate | best_mean_reward | training_time | exp_dir |
|---|---|---|---|---|---|
| vanilla (baseline) | 65.87% | 32.2% | 0.934 | 69 min | `exp16_vanilla_multi3_ms5_s0_2026-03-28@1656` |
| wm, `traj` | 58.87% | 27.8% | 0.878 | 112.2 min | `exp16_wm_multi3_ms5_s0_2026-10-01@1845` |
| wm, `rssm` | 62.2% | 31.4% | 0.986 | 116.4 min | `exp16_wm_multi3_ms5_s0_2026-10-01@2037` |

- **가설 지지**: `rssm`이 `traj` 대비 pocket +3.3pp, clear +3.6pp, best_mean_reward +0.108
  (0.878→0.986) 모두 우세. frozen R-SSM latent가 blind (x,y,event_type) flat 인코딩보다
  critic에 더 유용한 supervision을 준다는 가설과 방향이 일치한다.
- **그러나 둘 다 vanilla보다 낮음**: pocket rate 기준 vanilla(65.87%)가 wm/rssm(62.2%)보다도
  높다. 이번 설정에서는 world model critic을 추가하는 것 자체가 순수 SAC 대비 손해였다 —
  WMSAC의 추가 학습 목표(M의 supervised loss)가 critic 학습을 방해했을 가능성, 또는
  critic이 추가 forward/파라미터로 인해 같은 2M step 내에 덜 수렴했을 가능성. 학습 시간도
  wm 쪽이 vanilla의 ~1.6~1.7배(112~116분 vs 69분) — 추가 forward 비용이 실제로 큼.
- **노이즈 주의**: seed=0 단일 시드. `rssm` > `traj` 격차(3~4pp)는 여러 시드로 재확인 전엔
  "경향 확인" 수준. `wm` < `vanilla` 격차(3~7pp)는 더 크고 두 wm 런 모두에서 일관되게
  나타나 상대적으로 신뢰도가 높다.
- **결론/다음 질문**: R-SSM latent가 blind encoding보다는 낫다는 것은 확인됐지만, "WMSAC
  구조 자체가 이 task에 맞는가"는 별개 문제로 남는다. 여러 시드 반복은 보류 — 먼저
  WMSAC가 vanilla보다 떨어지는 원인(critic/actor loss가 학습 후반 계속 증가하는 경향이
  관찰됨, 수렴 전 종료 가능성)을 진단하는 게 우선순위로 보인다.

### 위 비교의 원인 진단 (2026-10-02)

wandb API로 세 런(vanilla `ggo1t1id`, traj `lkym9orr`, rssm `9ew9d17z`)의 전체 학습 곡선을
직접 당겨서 비교했다.

**진단 1 — critic/bellman loss 상승은 WM 전용 문제가 아님**: `critic_loss`(vanilla)/
`bellman_loss`(wm)가 초반 10% 구간 대비 후반 10% 구간에서 모두 ~4배 상승한다
(vanilla 0.56→2.11, traj 0.65→2.81, rssm 0.55→2.83). `wm_loss`는 오히려 작고 안정적
(traj 0.05→0.08, rssm 0.44→0.65). 즉 critic loss 상승은 WMSAC 구조 탓이 아니라 이
환경(max_steps=5, 짧은 호라이즌)에서 SAC 공통으로 나타나는 패턴으로 보인다.

**진단 2 — "중반 피크 → 후반 하락/진동" 패턴도 세 런 공통**:

| | peak pocket(50ep eval) | peak 시점(진행률) | 후반30% 평균±std |
|---|---|---|---|
| vanilla | 77.3% | 51% | 62.2% ± 4.4% |
| traj | 74.7% | 83% | 59.3% ± 5.8% |
| rssm | 76.0% | 51% | 61.7% ± 5.5% |

셋 다 2M step 중간쯤 74~77%를 찍고 이후 59~62%대로 떨어져 흔들린다. WM이 더 불안정한
것도 아니다 — 이 task 자체의 SAC 공통 불안정성.

**진단 3 (핵심) — best-checkpoint 선정 기준의 샘플링 노이즈**: `exp16_wm/train.py:343-346`
이 `pocket_rate > best_pocket_rate`(단일 50-episode eval)만으로 best_model을 저장한다.
n=50, p≈0.6에서 이항분포 표준편차 ≈ ±6.9pp — 지금 비교 중인 차이들(rssm-traj 3.3pp,
wm-vanilla 3~7pp)과 같은 크기다. 실제로 세 런 모두 저장된 best 체크포인트를 500ep로
재평가하면 11~16pp씩 떨어진다 (vanilla 77.3%→65.87%, traj 74.7%→58.87%,
rssm 76.0%→62.2%) — 체크포인트 선정 자체가 50ep 노이즈의 운 좋은 피크를 주운 것일
뿐이라는 뜻. 즉 지금까지의 "rssm>traj"/"wm<vanilla" 결론은 이 선정 노이즈 범위 안에
있어 신뢰하기 어렵다.

**결정**: 여러 시드를 늘리기 전에 먼저 best-checkpoint 선정 기준을 안정화한다 —
최근 N번 eval(`pocket_rate`)의 이동평균이 기존 최고 이동평균을 넘을 때만 저장
(`--best-ckpt-window`, 기본 5 → 노이즈 std 6.9pp → 추정 ~3.1pp로 감소). 선정 로직을
`_update_best_ma()` 순수 함수로 분리해 단위 테스트, `train.py` 루프에 연결 후 동일
3-way(vanilla/traj/rssm, seed=0)를 재실행해 결론이 바뀌는지 확인한다. 브랜치
`feature/wm-best-ckpt-ma`, `dev`로만 머지.

**구현 + 재실행 결과 (2026-10-02~03)**: `update_best_ma()`(`exp16_wm/train.py`) 구현,
7개 단위 테스트 통과, vanilla/wm 양쪽 스모크런 확인 후 동일 baseline 설정
(`best-ckpt-window=5` 추가)으로 vanilla/traj/rssm 3-way를 처음부터 다시 순차 실행.

| | pocket rate | clear rate | best_mean_reward | training_time | exp_dir |
|---|---|---|---|---|---|
| vanilla | 63.0% | 32.0% | 0.904 | 68.7 min | `exp16_vanilla_multi3_ms5_s0_2026-10-02@2057` |
| wm, `traj` | 61.87% | 30.0% | 0.930 | 110.5 min | `exp16_wm_multi3_ms5_s0_2026-10-02@2206` |
| wm, `rssm` | 65.6% | 34.2% | 1.040 | 115.0 min | `exp16_wm_multi3_ms5_s0_2026-10-02@2357` |

이전 비교(단일-eval 선정)와 순위 자체가 바뀌었다 — **이제 rssm이 vanilla보다도 우세**
(pocket +2.6pp, clear +2.2pp, reward +0.136), traj는 vanilla와 거의 동급이거나 약간
아래(pocket -1.1pp, clear -2.0pp, reward는 오히려 +0.026로 미세하게 높음). rssm이
traj보다 우세한 것은 이전과 동일하게 재확인됨(pocket +3.7pp, clear +4.2pp,
reward +0.11) — 이쪽은 두 번의 독립적인 실행(체크포인트 선정 기준이 다른)에서도 같은
방향으로 나왔다는 점에서 신뢰도가 더 올라갔다.

wandb 학습 곡선을 다시 확인한 결과, "중반 피크 → 후반 하락" 패턴 자체는 이번에도
그대로 남아 있다(세 런 모두 peak ma pocket_rate 68~70%를 학습 9~50% 지점에서
찍고 후반 30%는 57~61%대로 내려와 머무름) — 이건 체크포인트 선정 노이즈와는 무관한,
이 환경에서의 SAC 공통 현상으로 재확인. 다만 이동평균 선정 덕분에 "어느 체크포인트가
실제로 좋은지" 판단의 노이즈는 줄었고, 그 결과 traj vs rssm vs vanilla의 순위가 더
안정적으로 나왔다.

**결론**: frozen R-SSM latent를 critic 타깃으로 쓰는 쪽(`rssm`)이 blind flat
encoding(`traj`)뿐 아니라 world model 없는 순수 SAC(`vanilla`)보다도 낫다는 가설이
seed=0 기준으로는 지지된다. 다만 여전히 단일 시드이고, rssm-vanilla 격차(2.6pp)는
500ep 기준 이항분포 std(~2.2pp)와 비슷한 크기라 "확정" 수준은 아니다. 다음 단계는
여러 시드(예: 1,2,3) 반복으로 이 순위가 유지되는지 확인하는 것.

### 다중 시드 재확인 (2026-10-03 시작)

**가설**: seed=0에서 나온 순위(`rssm` > `vanilla` > `traj`)가 시드에 무관하게 유지되는지
확인한다. 격차(rssm-vanilla 2.6pp, rssm-traj 3.7pp)가 500ep 기준 이항 노이즈(~2.2pp)와
비슷한 크기라 단일 시드로는 결론을 내리기 어려웠다.

**설정**: seed=0과 동일한 baseline 설정(`n_balls=3 max_steps=5 step_penalty=0.1
trunc_penalty=1.0 total_steps=2,000,000 n_envs=10 learning_starts=5000 eval_freq=1000
eval_episodes=50 best_ckpt_window=5`)에 `seed ∈ {1, 2, 3}`을 추가해 vanilla/traj/rssm
각각 3회씩, 총 9개 런을 실행한다. CPU 10코어 제약으로 순차 실행(seed 1의 3-way →
seed 2의 3-way → seed 3의 3-way 순서), 런당 평균 ~70~115분이므로 총 소요 예상
~13~15시간.

**비교 축**: 각 시드에서 `rssm`이 `vanilla`/`traj`를 넘는지(순위), 그리고 seed=0을
포함한 4개 시드에서 방법별 평균±std로 최종 판단. 브랜치
`feature/wm-traj-vs-rssm-multiseed`, `dev`로만 머지.

**결과 (2026-10-03 완료)**: 9개 런 모두 정상 종료. seed=0 결과(이동평균 선정 재실행분)와 합쳐
4개 시드 × 3방법 = 12개 런으로 비교한다. 지표는 모두 best 체크포인트를 500ep 재평가한 값
(`results.json`). pocket rate는 1500볼(500ep×3볼), clear rate는 500ep 기준.

| seed | vanilla (pocket / clear / reward) | wm-`traj` | wm-`rssm` | pocket 1위 |
|---|---|---|---|---|
| 0 | 63.0 / 32.0 / 0.904 | 61.87 / 30.0 / 0.930 | **65.6** / 34.2 / 1.040 | rssm |
| 1 | 64.27 / 33.4 / 1.096 | **64.4** / 32.8 / 1.148 | 62.47 / 32.4 / 1.140 | traj |
| 2 | 61.67 / 30.4 / 0.938 | **63.6** / 31.8 / 0.998 | 60.0 / 27.8 / 0.944 | traj |
| 3 | 62.87 / 31.2 / 1.016 | **64.0** / 31.8 / 0.888 | 60.67 / 28.2 / 0.968 | traj |

exp_dir: vanilla `exp16_vanilla_multi3_ms5_s{1,2,3}_2026-10-03@{0720,1216,1717}`,
traj `exp16_wm_multi3_ms5_s{1,2,3}_2026-10-03@{0829,1326,1826}`,
rssm `exp16_wm_multi3_ms5_s{1,2,3}_2026-10-03@{1021,1520,2019}`.

4시드 평균 ± 표본 std:

| | pocket | clear | best_mean_reward | training_time |
|---|---|---|---|---|
| vanilla | 62.95 ± 1.06 | 31.75 ± 1.28 | 0.988 ± 0.086 | ~69분 |
| wm, `traj` | 63.47 ± 1.11 | 31.60 ± 1.17 | 0.991 ± 0.114 | ~112분 |
| wm, `rssm` | 62.19 ± 2.50 | 30.65 ± 3.15 | 1.023 ± 0.088 | ~115분 |

**통계 검정** (n=4 시드; final eval env가 `reset(seed=args.seed)`라 같은 시드끼리는 평가
배치가 동일 → 시드 단위 paired t-test를 주 검정으로, Welch t-test를 보조로):

| 비교 (pocket) | 평균 차이 | paired p | Welch p | 95% CI |
|---|---|---|---|---|
| traj − vanilla | +0.52pp | 0.49 | 0.53 | (−1.59, +2.62) |
| rssm − vanilla | −0.77pp | 0.55 | 0.60 | (−4.36, +2.82) |
| traj − rssm | +1.28pp | 0.51 | 0.40 | (−4.16, +6.73) |

3-way 전체: one-way ANOVA p=0.58, Friedman p=0.37. clear rate(모든 p ≥ 0.42),
best_mean_reward(모든 p ≥ 0.44)도 마찬가지로 유의한 차이 없음. 시드 분산을 무시하고
런 내부 이항 노이즈만으로 본 pooled two-proportion z-test조차 p ≥ 0.15.
검정력: 현재 시드 간 차이 std(~1.3pp) 기준 n=4 paired 설계로 80% 검정력으로 잡아낼 수 있는
최소 효과는 ~2.9pp — 그보다 작은 차이는 이번 설계로는 원래 판별 불가.

**해석**:
- seed=0의 `rssm > vanilla > traj` 순위는 **재현되지 않았다**. seed 1~3에서 rssm은 pocket
  기준 매번 3위, traj가 매번 1위. seed=0은 rssm에게 운 좋은 시드였던 것으로 보인다.
  rssm은 시드 간 분산도 가장 크다(pocket std 2.5pp vs 나머지 ~1.1pp).
- 세 방법의 평균 차이(≤1.3pp)는 모두 시드 노이즈 범위 안. **world model critic(traj/rssm
  어느 쪽이든)이 vanilla SAC보다 낫다는 근거는 없다** — "효과 없음이 증명됐다"가 아니라
  "~3pp 이상의 효과는 없을 가능성이 높고, 그 이하는 판별 불가"라는 의미.
- 비용 측면에서 wm은 vanilla 대비 학습 시간 ~1.6배.

**결론**: 이 설정(frozen 타깃 + `M(obs,action)→ĥ` 보조 MSE loss, Option A)으로는
world model이 SAC 성능에 유의미한 이득을 주지 못한다. 보조 loss로 critic 표현을 정규화하는
방식은 타깃이 blind 궤적이든 R-SSM latent든 차이를 만들지 못했다. 다음 방향 후보: (1) 보조
loss가 아니라 실제 상상 롤아웃으로 Bellman target/데이터를 늘리는 Option B, (2) 시드 수를
늘려 ~1pp 수준 효과를 검증하는 것은 비용 대비 의미가 작아 보류.

### 학습 곡선 구간별 비교 — 최종 성능이 아니라 학습 속도에서는 차이가 있는가 (2026-10-05)

**동기**: 위 결론은 best 체크포인트(=점근 성능)만 본 것이다. 보조 loss로 표현을 정규화하는
방식은 보통 점근 성능보다 **샘플 효율**(초반 학습 속도)에서 효과가 나기 때문에, 이미 있는
12개 런의 `train.log` eval 기록(1k step마다 50ep pocket rate)을 구간별로 다시 비교했다.
추가 학습 없음.

구간별 평균 eval pocket rate (4시드 평균 ± std), 아래 행은 vs vanilla paired t-test p:

| 구간 (env step) | 5–50k | 50–100k | 100–200k | 200–300k | 300–500k | 0.5–1M | 1–1.5M | 1.5–2M |
|---|---|---|---|---|---|---|---|---|
| vanilla | 41.5±0.9 | 15.9±0.6 | 41.4±2.0 | 63.4±0.5 | 62.7±0.3 | 62.7±0.2 | 62.5±0.3 | 62.2±0.6 |
| wm-`traj` | 38.9±1.9 | 17.3±0.8 | **53.9±1.3** | 63.4±0.3 | 62.7±0.3 | 61.6±0.7 | 59.8±1.0 | 59.1±1.7 |
| wm-`rssm` | 39.3±0.4 | 15.4±0.5 | **50.1±2.2** | 62.7±0.2 | 62.6±0.3 | 61.8±0.7 | 61.0±1.0 | 59.3±0.6 |
| p (traj) | 0.03 | 0.07 | **<0.01** | 0.99 | 0.98 | 0.04 | 0.01 | 0.04 |
| p (rssm) | 0.01 | 0.10 | **0.02** | 0.07 | 0.34 | 0.04 | 0.06 | <0.01 |

이동평균(20 eval) pocket ≥ 60% 도달 step (시드 1→4 순): vanilla 185k/174k/176k/166k,
traj 150k/166k/156k/159k, rssm 154k/146k/165k/149k. 0–300k 구간 곡선 아래 면적(평균
pocket): vanilla 43.7–45.2, traj 47.8–49.0, rssm 45.8–47.6 — 4시드 모두 wm > vanilla.

**해석**:
- **WM 보조 loss는 샘플 효율을 확실히 높인다.** 100–200k 구간(성능이 급상승하는 시기)에서
  traj +12.5pp, rssm +8.7pp, 60% 도달이 ~20k step 빠름. 4시드 모두 같은 방향이라
  best-ckpt 비교에서는 안 보이던 효과다.
- 그러나 200k 이후에는 세 방법이 같은 천장(~63%)에 붙고, **후반(1M+)에는 wm이 오히려
  vanilla보다 2–3pp 더 많이 떨어진다**(유의). 보조 loss가 후반 안정성을 해치는 것으로
  보인다(예: Bellman 신호가 작아진 뒤에도 wm_coef=1로 MSE가 critic 용량/gradient를
  잡아먹음 — 미검증 가설).
- traj와 rssm을 비교하면 오히려 traj가 초반에 더 빠르다. R-SSM latent가 blind 궤적보다
  "더 좋은 physics 정보"라는 가설은 이 지표로도 지지되지 않는다.
- 50–100k 구간에서 세 방법 모두 ~16%로 떨어지는 공통 dip은 WM과 무관한 SAC 초기 동작
  (아직 미조사).

### Q 품질 직접 진단: h가 pocket 예측/가치 추정에 실제로 정보를 더하는가 (2026-10-05 계획)

**원래 질문**: R-SSM latent h가 physics 정보를 충분히 담고 있다면, 그걸 거쳐 계산한 Q가
더 정확할 것이다. 특히 이 환경의 Q는 사실상 "이 샷으로 공이 들어가는가"(pocket
prediction)가 지배적이므로, h가 그 판별에 의미 있는 표현이 되는지 보고 싶다. MBPO로 가기 전
가장 값싸게 확인하는 경로로 Option A를 택했던 것.

pocket rate(정책 성능)는 Q 품질을 간접적으로만 반영하므로, 저장된 체크포인트로 **Q와 ĥ를
직접** 측정한다. 추가 학습 없음, 시드 0–3 × vanilla/traj/rssm의 best 체크포인트 사용.

- **D1 — pocket 판별력**: 시드별 공통 (s,a) 데이터셋(정책 행동 + 노이즈, uniform random
  행동 혼합으로 성공/실패가 고루 섞이게)을 수집하고, 각 샷의 실제 결과
  (`pocketed_this_step>0`)를 라벨로 Q(s,a)의 AUC를 비교. "어느 critic이 들어가는 샷을 더 잘
  알아보는가".
- **D2 — 가치 추정 정확도**: 각 모델 자기 정책으로 rollout하며 Q(s_t,a_t)와 실제 할인
  return G_t의 상관(Pearson/Spearman)·MSE 비교. (soft Q라 엔트로피 항만큼 bias는 있으니
  상관 위주로 해석.)
- **D3 — ĥ가 실제로 무엇을 담는가**: (a) M이 타깃을 얼마나 맞추나(ĥ vs h_real R²),
  (b) 선형 probe(5-fold CV AUC)로 pocket 라벨 판별: 원시 [s,a] / vanilla critic 은닉층 /
  wm ĥ / 실제 h_real(traj, rssm). 실제 h_real은 샷 결과를 이미 담고 있어 leak이 있는
  상한선 참고용이다. 핵심 비교는 **예측된 ĥ의 probe AUC가 원시 [s,a]·vanilla 은닉층보다
  높은가**.

**판단 기준**: rssm critic의 D1 AUC / D3 ĥ probe가 traj·vanilla보다 유의하게 높으면
"h가 pocket 관련 정보를 더 잘 구조화한다"는 가설이 살아 있다 → MBPO로 가볼 근거가 된다.
같거나 낮으면, 그 원인이 "M이 (s,a)→h 예측을 못해서"(D3a R² 낮음)인지 "h 자체가 pocket에
무관해서"(실제 h_real probe도 낮음)인지 가른다. 코드: `exp16_wm/diagnose_q.py`, 테스트
`tests/test_exp16_wm_diagnose_q.py`.

**결과 (2026-10-05)**: `python -m exp16_wm.diagnose_q --seeds 0 1 2 3 --n-shots 3000
--n-episodes 200` → `logs/diagnose_q.json`. 시드별 공통 데이터셋 3000샷(pocket 성공 비율
23%), 4시드 평균 ± std, p는 vs vanilla paired t-test.

| 지표 | vanilla | wm-`traj` | wm-`rssm` |
|---|---|---|---|
| D1 Q(s,a)의 pocket AUC | **0.654 ± 0.027** | 0.616 ± 0.013 (p=0.09) | 0.625 ± 0.016 (p=0.04) |
| D1 Spearman(Q, 즉시 reward) | **0.144** | 0.093 (p=0.05) | 0.086 (p=0.05) |
| D2 Pearson(Q, G_t), 전체 | −0.02 | −0.05 | −0.05 |
| D2 Pearson(Q, G_t), 샷 순번 t별 평균 | 0.12 | 0.09 | 0.09 |
| D2 bias(Q − G_t) | −3.46 | −4.76 | −3.53 |
| D3a ĥ vs h_real R² | — | 0.29 ± 0.08 | 0.25 ± 0.01 |
| D3b critic 표현 linear probe AUC | 0.699 | 0.682 | 0.697 |

참조 probe (모델 무관): 원시 [s,a] linear 0.646 / MLP 0.607, **실제 h_real(샷 이후)
traj 0.950 / rssm 0.962** (샷 결과를 이미 담고 있으므로 leak이 있는 상한선).

**해석**:
1. **h 자체의 정보량은 문제가 아니다.** 실제 샷의 h_real은 pocket을 거의 완벽히 판별한다
   (0.95–0.96). 다만 R-SSM latent(0.962)와 blind 궤적(0.950)의 차이는 작다 — "이미 일어난
   샷"을 기술하는 데에는 R-SSM의 구조화가 별 이득이 없다.
2. **병목은 (s,a)→h 예측이다.** critic 안의 M(512×3 MLP)이 h를 예측한 정도는 R² 0.25–0.29에
   그친다. Q가 실제로 보는 것은 h_real이 아니라 이 예측값 ĥ이므로, h_real의 풍부한 정보가
   Q로 전달되지 않는다. pocket 여부를 (s,a)로부터 예측하는 것 자체가 어려운 문제이고(원시
   [s,a] probe 0.61–0.65, 2M step 학습한 critic도 0.62–0.65), 보조 MSE loss는 이 어려운
   physics forward mapping을 M에게 가르치지 못했다.
3. **WM critic은 pocket 판별에서 오히려 vanilla보다 약간 나쁘다**(rssm p=0.04). 보조 loss가
   critic 용량을 h의 Q와 무관한 차원에 쓰게 한다는 앞의 가설과 맞는다.
4. **세 방법 공통: Q의 학습 목표 자체가 실제 return과 어긋나 있다.** 샷 순번 t가 커질수록
   실제 G_t는 내려가는데(seed 0 vanilla: 0.54→−0.57) Q는 반대로 올라간다(−3.53→−2.38).
   같은 t 안에서도 상관은 0.09–0.12. 원인 후보: (a) 5샷 truncation을 SB3가
   `TimeLimit.truncated`로 표시해 버퍼가 done을 마스킹(`buffer.py:86`) → 실제로는 게임이
   끝나고 trunc_penalty −1까지 받는데 critic은 무한히 이어지는 것처럼 bootstrap, (b) obs에
   샷 순번이 없어 남은 샷 수를 알 수 없음(`shots_taken` 옵션 미사용), (c) soft-Q 엔트로피 항
   (log π ≈ +2.4, α ≈ 0.12–0.15)이 매 step 누적돼 ~−3 bias. (a)+(b)는 Q가 무엇을
   추정해야 하는지 자체를 흐린다 — h가 아무리 좋아도 Q로 이득이 드러나기 어려운 구조.

**결론**: "h가 physics 정보를 담고 있으니 Q가 좋아질 것"이라는 가설은 이 Option A 설정으로는
검증 자체가 막혀 있다. h_real의 정보는 충분하지만, (1) critic이 h를 (s,a)에서 예측해야 하고
그 예측이 R² ~0.25 수준이며, (2) Q의 학습 목표가 truncation 처리 때문에 실제 return과
어긋나 있다. MBPO로 가는 판단에 직접 필요한 질문은 "R-SSM 롤아웃이 (s,a)로부터 pocket을
critic(AUC ~0.65)보다 잘 예측하는가"이며, 이는 R-SSM pocket head를 같은 데이터셋에 돌려
비교하면 값싸게 확인할 수 있다. 별개로 truncation 처리/샷 순번 obs 수정은 세 방법 모두에
영향을 주는 선결 문제다.

### D4: R-SSM 자유 롤아웃의 pre-shot pocket 예측력 vs critic (2026-10-05 계획)

**동기**: 위 D1–D3에서 병목은 "(s,a)→h 예측"으로 좁혀졌다. MBPO는 바로 이 예측을 R-SSM
롤아웃(`RolloutEngine`: EventDetector가 이벤트 시점·종류를 해석적으로 계산, R-SSM이 충돌별
Δvel 예측)으로 대신하는 방식이다. 그렇다면 MBPO로 갈 가치가 있는지는 "샷 전 상태 + 행동만
주고 R-SSM을 자유 롤아웃시켰을 때 pocket을 critic(AUC ~0.62–0.65)보다 잘 맞히는가"로 값싸게
판단할 수 있다. 기존 pocket head AUC 0.919는 **실제 이벤트를 teacher-forcing**한 h에서 측정한
것이라 이 질문에 대한 답이 아니다.

**방법**: D1과 같은 방식으로 시드별 3000샷 공통 데이터셋을 수집하되, 각 샷마다 샷 직전 공
배치와 strike 직후 큐볼 rvw(`stick_ball` 이벤트)를 함께 저장. 같은 샷에 대해 아래 점수의
pocket AUC를 비교한다(라벨 = `pocketed_this_step>0`).
- 각 시드 vanilla/traj/rssm critic의 Q(s,a) (D1 재현)
- R-SSM 롤아웃: (a) 롤아웃 안에서 타깃 공의 pocket 이벤트 발생 여부(개수),
  (b) `predict_pocket(h)` — 롤아웃 이벤트를 따라가며 공별 first-touch h 기준, 타깃 공 중 최댓값
- teacher-forced 참조: 실제 이벤트로 얻은 h의 `predict_pocket` (기존 0.919와 같은 방식, 상한선)
- 0-파라미터 pre-shot 기하 baseline: 큐볼 진행 방향으로 처음 맞는 공을 찾고, 그 공이
  중심선 방향으로 굴러갈 때 가장 가까운 포켓까지의 거리(가까울수록 높은 점수)

**판단 기준**: 롤아웃 기반 점수가 critic Q보다 확실히 높으면(예: AUC +0.1 이상) R-SSM 롤아웃이
critic이 못 배우는 physics를 제공한다는 뜻 → MBPO(또는 롤아웃 결과를 critic 입력으로 직접
주는 방식)로 갈 근거. 기하 baseline과도 비교해, 이득이 R-SSM 학습 덕분인지 EventDetector의
해석적 기하 계산 덕분인지 가른다.

**결과 (2026-10-05)**: `python -m exp16_wm.diagnose_q --d4 --seeds 0 1 2 3 --n-shots 3000`
→ `logs/diagnose_q_d4.json`. 시드별 3000샷(pocket 23%), 샷 단위 AUC, 4시드 평균 ± std.

| 점수 (샷 전 정보만 사용 / 참조) | pocket AUC | vs vanilla Q (paired p) |
|---|---|---|
| critic Q — vanilla | 0.662 ± 0.026 | — |
| critic Q — wm-`traj` | 0.622 ± 0.008 | |
| critic Q — wm-`rssm` | 0.632 ± 0.020 | |
| R-SSM 자유 롤아웃: 롤아웃 안 pocket 이벤트 개수 | 0.561 ± 0.005 | 0.006 (더 나쁨) |
| R-SSM 자유 롤아웃: first-touch `predict_pocket` | 0.716 ± 0.009 | 0.017 (더 좋음) |
| **0-파라미터 pre-shot 기하 baseline** | **0.746 ± 0.005** | 0.007 (더 좋음) |
| (참조) teacher-forced `predict_pocket` — 실제 이벤트 사용 | 0.746 ± 0.004 | |

롤아웃이 "pocket 있음"으로 판정한 비율은 13.5%(실제 23%), 실제 결과와의 일치율 74%는
"항상 실패" 예측(77%)보다도 낮다.

**해석**:
1. **R-SSM 롤아웃은 critic보다는 낫지만(0.716 vs 0.66), 직선 기하 계산(0.746)보다 못하다**
   (rollout_head − geom p=0.008). 즉 R-SSM이 critic에 줄 수 있는 pre-shot physics 정보는
   "큐볼 진행 방향 → 처음 맞는 공 → 중심선 방향으로 포켓을 향하는가"라는 기초 기하 수준에도
   못 미친다.
2. **롤아웃의 동역학 자체가 빠르게 어긋난다.** 롤아웃 안에서 실제로 공이 포켓에 들어가는
   이벤트로 판정하면 AUC 0.56으로 거의 무작위. 충돌별 Δvel 오차가 이벤트마다 누적되면서
   공의 경로가 실제와 달라지는 것으로 보인다. `predict_pocket` head(0.716)가 이보다 나은 건
   first-touch 시점(첫 충돌 직후)의 h만 쓰기 때문 — 누적 오차가 쌓이기 전이다.
3. **pocket head는 실제 이벤트를 넣어줘도 샷 단위 0.746**으로 기하 baseline과 같다. 기존
   0.919는 공 단위 + tautological 샘플(첫 이벤트가 곧 pocket)을 포함한 수치였다.
4. **critic의 pocket 판별에는 개선 여지가 크다.** 0-파라미터 기하조차 critic보다 0.08 높다.
   다만 그 여지를 채울 도구가 현재 R-SSM은 아니다.

**MBPO 판단에 대한 함의**: 현재 R-SSM(`rssm_v9_ls01_3ball`)으로 상상 롤아웃을 만들면, 그
롤아웃의 pocket 결과는 거의 무작위(AUC 0.56)라 critic에게 노이즈 데이터를 주게 된다. MBPO를
진행하려면 먼저 자유 롤아웃(free-running) 정확도를 올리는 것이 선결 과제 — R-SSM은 지금까지
teacher-forcing으로만 평가되어 왔다.

**부수 발견 (버그)**: `RolloutEngine`/`EventDetector`는 롤아웃 중 큐볼이 포켓되거나 공이
하나만 남으면 pooltool `System` 생성(cue ball id 누락) 또는 ball-ball 탐색(빈 pair 집합)에서
예외로 죽는다. MBPO 롤아웃에서도 그대로 발생할 문제. 진단 스크립트에서는
`CueSafeDetector`(테이블 밖 정지 더미 공으로 패딩)로 우회했고, `world_model/` 쪽 수정은 아직
하지 않았다.

### 설계: (G) critic에 pre-shot 기하 특징 주입 → (M) h masking/혼합 (2026-10-05)

**배경**: D1–D4로 정리된 사실 — (1) critic의 pocket 판별력(AUC ~0.66)은 0-파라미터 직선 기하
계산(0.746)보다 낮다, (2) WM critic의 `q`는 실제 h를 한 번도 보지 못하고 부정확한 ĥ(R² ~0.25)
로만 학습된다, (3) 현재 R-SSM 자유 롤아웃은 기하보다 못하다. 그래서 두 질문을 순서대로 분리해
확인한다.

- **G**: "샷 전에 계산 가능한 physics 정보가 critic 입력에 명시적으로 주어지면 Q(그리고
  정책)가 좋아지는가?" — WM 품질과 무관하게, *정보를 주면 쓰는가*를 먼저 본다. 효과가 있으면
  그 정보를 학습된 WM으로 대체할 가치가 있다는 상한선 근거가 된다.
- **M**: "q가 실제 h의 의미를 학습하도록 강제하면, WM critic이 h를 실제로 활용하는가?"

#### G — `GeomSAC` (`--agent geom`)

- critic 입력 = `[obs, action, geom(obs, action)]`. 나머지(actor, alpha, 버퍼, 하이퍼파라미터)는
  VanillaSAC와 동일.
- `geom()`은 **torch로 구현해 action에 대해 미분 가능**하게 만든다. actor는 ∂Q/∂a로
  학습되므로, numpy로 특징을 계산해 붙이면 기하 특징을 통한 gradient가 actor에 전달되지 않는다.
  첫 접촉 공 선택(argmin)만 이산 선택이고, 선택된 뒤의 특징은 a에 대해 연속이다.
- 특징(정규화 좌표 → 미터 변환 후 계산, env와 동일한 aim 규칙: 가장 가까운 남은 공 방향 +
  `delta_angle`):
  1. `hit` — 큐볼 직선 경로가 남은 타깃 공 중 하나에 닿는가 (0/1)
  2. `cos_cut` — 큐볼 진행 방향과 접촉 시 중심선(큐→타깃) 사이 cos (얇게 맞을수록 작음)
  3. `d_cue` — 큐볼이 접촉점까지 가는 거리
  4. `miss` — 타깃 공이 중심선 방향으로 굴러갈 때 가장 가까운 포켓까지의 최소 수직 거리
     (D4 기하 baseline 점수와 같은 양)
  5. `d_obj` — 그 포켓까지 진행 방향 거리
  6. `speed` — 정규화 속도 (action[1])
  미접촉 시 2–5는 고정 sentinel 값. 포켓 중심은 `pt.Table.default()` 상수.
- 비교 대상: 기존 vanilla 4시드(s0–3, 동일 설정 2M step) — 새로 돌리는 것은 geom 4런뿐.

#### M — WMSAC + h 혼합 (`--agent wm --wm-target rssm --h-mix-start 0.5 --h-mix-steps 1000000`)

- critic 업데이트의 **현재 (s,a)에 대한 Bellman 항**에서, 샘플별로 확률 p로 `q`의 입력을 ĥ 대신
  실제 h_real(그 transition에서 R-SSM이 인코딩한 값)로 바꾼다. 나머지는 그대로:
  - Bellman target(`Q(s', a')`)과 actor loss(`Q(s, π(s))`)는 가상의 action이라 실제 h가 없으므로
    항상 ĥ 사용.
  - wm_loss(MSE(ĥ, h_real))는 유지.
- p는 0.5에서 시작해 1M step(전체의 절반)에 걸쳐 선형으로 0까지 감소, 이후 0. 실제 h는 샷
  결과(pocket 이벤트 포함)를 담고 있어 leak이 있으므로, 끝에는 반드시 ĥ만으로 동작해야 한다.
- 기대 메커니즘: `q`가 실제 h에서 "h의 어느 방향이 pocket/가치와 연결되는지"를 배우고, ĥ로
  학습하는 샘플에서는 Bellman gradient가 q를 거쳐 M으로 흘러, M이 "q가 이해하는 방향"으로 ĥ를
  내도록 압박받는다.
- 알려진 위험: ĥ는 R² 0.25라 분포가 h_real보다 평균 쪽으로 수축되어 있어, q가 실제 h에서
  배운 것이 ĥ에 그대로 전이되지 않을 수 있다(분포 불일치). 이 경우 p가 0이 되는 후반에 성능이
  꺾일 것이다 — 학습 곡선에서 확인.
- 비교 대상: 기존 wm-`rssm` 4시드(s0–3). 새로 돌리는 것은 4런.

#### 공통

- 설정은 기존 12런과 동일(`n_balls=3 max_steps=5 step_penalty=0.1 trunc_penalty=1.0
  total_steps=2M n_envs=10 eval_freq=1000 eval_episodes=50 best_ckpt_window=5`), seed 0–3.
- 위에서 발견한 truncation/샷순번 문제는 **이번에는 고치지 않는다** — 기존 런과 비교 가능성을
  유지하기 위해서. 결론 해석 시 공통 교란 요인으로 명시.
- 평가 지표: (a) 최종 pocket/clear(500ep, best ckpt), (b) 학습 곡선 구간별 pocket(특히
  100–200k), (c) D1 Q pocket AUC / critic 표현 probe AUC (`diagnose_q` 확장), M은 추가로 ĥ R².
- 판단 기준:
  - G에서 Q AUC가 기하 수준(≥0.72)으로 오르고 pocket rate 또는 초반 샘플 효율이 유의하게
    오르면 → "pre-shot physics 정보는 Q에 쓸모 있다". Q AUC는 오르는데 정책이 안 오르면 →
    병목은 Q 목표(truncation)나 정책 쪽.
  - M에서 Q AUC·ĥ probe가 wm-rssm 대비 오르고 성능이 같거나 나으면 → masking으로 q가 h를
    활용하게 만들 수 있다.
- 순서: G 4런(~70분/런) → G 분석 → M 4런(~115분/런) → M 분석. 순차 실행 총 ~13시간.
- 코드: `exp16_wm/networks.py`(GeomCritic, `geom_features`), `exp16_wm/sac.py`(GeomSAC,
  WMSAC h-mix), `exp16_wm/train.py`(CLI), 테스트 `tests/test_exp16_wm_geom.py`,
  `tests/test_exp16_wm_hmix.py`.

**실행 상태 (2026-10-06, 사용자 요청으로 중단)**: G/M 구현·테스트·스모크런 완료(커밋됨) 후 G
seed 0부터 순차 실행했으나, Mac이 과열 비상(`Dark Wake Thermal Emergency`, AC 연결인데
배터리 0%)으로 15~20분마다 강제 잠자기에 들어가 학습이 실효 4~5배 느려졌다(깨어 있는 구간은
정상 ~600fps). 두 번의 G seed 0 부분 런(781k, 705k step)은 완료 전에 중단 —
`exp16_geom_multi3_ms5_s0_2026-10-06@{0024_ABORTED_slow,0352_ABORTED_stopped}`, 분석에서
제외. G/M 결과 없음. 동일 설정으로 재실행 필요. (※ 당시 "하드웨어 문제"로 본 진단은 오진 —
아래 2026-10-08 원인 정정 참조.)

**재실행 계획 (2026-10-06)**: Mac mini 대신 로컬 CPU 서버에서 **기준선 포함 전 arm을 동일 환경으로
다시 실행**한다 — 하드웨어/환경이 바뀌면 수치가 완전히 재현되지 않으므로, 비교 대상(vanilla,
rssm)도 같은 서버에서 다시 돌려야 공정하다. 서버가 Mac mini보다 느리므로 런당 시간은 더 길어질
것으로 예상.
- arm: `vanilla`, `rssm`(기존 WM 기준선), `geom`(G), `hmix`(M) × seed 0–3 = 16런.
  (`traj`는 선택 — 이전 결과에서 rssm과 유의차 없음.)
- 실행: `ARMS="vanilla rssm geom hmix" SEEDS="0 1 2 3" PARALLEL=<코어수/11> bash exp16_wm/run_grid.sh`
  — 설정은 기존 12런과 동일, `--n-envs 10` 고정(update-to-data 비율 유지).
- 이식성 정리: R-SSM 체크포인트 `world_model/results/rssm_v9_ls01_3ball/best.pt`(2.4MB)를 git에
  포함(`.gitignore` 예외로 강제 추가), `requirements.txt`에 wandb/scipy/scikit-learn/pytest 추가,
  `diagnose_q.find_run`의 날짜 필터 제거(서버 런도 탐색).

**원인 정정 (2026-10-08)**: 위 중단의 원인을 하드웨어(배터리/덮개/통풍)로 본 것은 오진이었다.
"이때까지 괜찮았는데 왜 이번에만?"이라는 질문에서 출발해 전원 로그(`pmset -g log`)와
unified log를 다시 조사했다.
- 기기는 Mac mini(Mac16,10, M4, 코어 10 = P4+E6)라 배터리·덮개가 없다. 로그의 `Charge:0%`는
  의미 없는 값이다.
- 10/4 07:20 ~ 10/5 19:33에는 sleep/wake 이벤트가 하나도 없다(계속 완전히 깨어 있었음). 10/1–3의
  12런도 길이가 일정했다(vanilla ~70분, wm ~115분/런) — 같은 부하에서 느려진 흔적이 없다.
- 10/5 19:34:33에 `Software Sleep pid=413`(= `loginwindow`, 수동 잠자기)이 걸렸고, 이후 Mac은 한 번도
  완전히 깨어나지 못한 채 Dark Wake만 반복했다. 첫 강제 잠자기(20:21)는 G 학습 시작(00:24)보다
  4시간 먼저였고 당시 부하는 D4 진단 정도였다 — **G/M 학습 자체가 원인이 아니었다.**
- 열 압력(`kThermalPressureContext`)은 20:04에 1 → 20:16 2 → 20:20 3 → 20:21:30 4에 도달하는 즉시
  `Dark Wake Thermal Emergency` 잠자기가 걸렸다. 깰 때마다 0으로 리셋되고, G 학습 중에는 깬 뒤 약
  3분 만에 4에 도달했다(10/6 00:24:28 1 → 00:27:32 4).
- 대조 실험(10/8 21:29, 완전히 깬 상태): `yes` 10개로 전코어 8분 부하 → 열 압력 변화 0회, 쓰로틀 기록
  없음. 같은 코어 수의 부하가 Dark Wake에서는 3분 만에 한계, 깬 상태에서는 8분간 무변화였다.
- 결론: 원인은 부하·하드웨어가 아니라 **학습 도중 수동 잠자기로 Dark Wake에 빠진 것**. 가설: Dark Wake에서는
  열 한계가 훨씬 낮게 걸림(팬 동작은 직접 확인하지 못함). 한계: 합성 부하 8분 대조일 뿐, 깬 상태에서
  실제 학습을 장시간 돌려 본 것은 아래 재실행으로 확인한다.
- 대책: 학습 중 Mac을 재우지 않는다(화면 꺼짐은 무방. `caffeinate`는 수동 잠자기를 막지 못함). 실행은
  `caffeinate -ims`로 감싸고 열 압력/잠자기 진입을 감시한다. `taskpolicy`·스레드 제한 같은 "살살
  돌리기"는 깬 상태에서 문제가 재현될 때의 예비책으로만 둔다.
- 재실행: 2026-10-08 21:39 G seed 0 (Mac mini, `ARMS=geom SEEDS=0 PARALLEL=1 run_grid.sh`, 설정 동일,
  PY=.venv/bin/python, `caffeinate -ims`로 감쌈). 2M step 정상 완주(21:39:53 → 22:53:52, 학습
  73.5분, 평균 456fps ≈ 계획의 ~70분/런). **전 구간(75분)에서 열 압력 변화 0회, 잠자기 0회** —
  깨어 있는 상태에서는 같은 부하가 아무 문제 없이 돈다는 것을 실학습으로 확인했다(위 가설 지지).
  - 결과(`exp16_geom_multi3_ms5_s0_2026-10-08@2139`, best ckpt 500ep): pocket 64.13%, clear
    31.2%, best_mean_reward 1.028. 기존 vanilla(n=4) 62.95±1.06 / 31.75±1.28 / 0.988±0.086 과
    같은 범위 — **시드 1개라 유의성 판단 불가**(p=nan). 학습 곡선은 초반(5–200k)에 vanilla보다
    낮고(32.2/9.6/31.6 vs 41.5/15.9/41.4) 300k 이후 비슷하다. G 해석은 4시드가 모인 뒤에.
- 계획에 미치는 영향: 위 10/6 "서버 재실행 계획"은 하드웨어 문제를 전제로 한 것이었다. 원인이 Mac이
  아니었으므로, 기존 12런과 같은 환경(Mac mini)에서 G/M을 이어가는 선택지가 생긴다 — 서버로 갈지
  Mac에서 계속할지는 재실행 결과를 보고 결정.

### 과제 재정의: "5샷 안에 몇 개" → "다 넣기까지 몇 샷" (2026-10-08 계획)

**왜 바꾸나**: 지금까지의 진단을 모아 보면, 병목은 WM의 물리 정보가 아니라 **Q가 배우는 목표 자체**다.
- Q 진단(D2): Q와 실제 return의 상관이 ≈0(전체 −0.02~−0.05, 샷 순번별 0.09~0.12). 원인 후보 (a)+(b):
  - (a) 5샷 제한은 과제 규칙이다(못 끝내면 −1 받고 게임 종료). 그런데 SB3 `SubprocVecEnv`가 이를
    `TimeLimit.truncated`로 표시하고 버퍼가 done을 마스킹해서(`buffer.py:86`, "잘렸을 뿐 게임은 계속된다"는
    SB3 관례) critic은 5번째 샷 뒤에도 게임이 이어지는 것처럼 bootstrap한다.
  - (b) obs에 샷 순번이 없다. 같은 공 배치라도 1샷째(기회 4번 남음)와 5샷째(못 넣으면 끝)는 가치가 완전히
    다른데 critic은 구분할 수 없다.
- 모든 arm 공통의 "중반 피크(ma 68~70%) → 후반 ~60%" 하락(7~8pp)이 지금까지 측정한 어떤 WM 효과(≤1.3pp)보다
  크다. 이 하락도 (a)(b)로 어긋난 Q 목표와 연결돼 있을 가능성이 있다.

**대안 검토**:
1. 5샷 종료를 done으로 처리 + `shots_taken` obs 추가 — 기존 과제를 유지한 채 고치는 방법.
2. "헛치면 종료"(실제 당구 규칙: 하나라도 넣어야 다음 샷) — 기각. 사용자 목표와 맞지 않는다.

**사용자 목표**: 공은 결국 다 넣는다. 중요한 것은 **(1) 다 넣는가, (2) 얼마나 빨리(몇 샷 만에) 다 넣는가**다.
5샷 안에 3개를 넣는 기준은 5번 중 2번을 헛쳐도 100%라 이 목표를 직접 보지 않는다.

**새 과제 정의**:
- 게임은 **공을 다 넣었을 때만 끝난다**(`terminated`). `max_steps=30`은 규칙이 아니라 학습 초반 정책이 끝내지 못하는
  게임을 막는 **안전장치**다.
- 리워드 = 공당 +1 − 샷당 1 (`--step-penalty 1.0`, 스크래치 −0.5는 기존대로). 다 넣으면 return = 3 − 샷 수
  (−0.5×스크래치) → 1샷에 3개 = 2 > 1개씩 3샷 = 0. 빠를수록 높다.
- 한 샷 다중 포켓에 별도 보너스는 주지 않는다. 그 이득은 "남은 샷의 실패 위험 감소"로 기대값에 이미 반영되고,
  보너스는 성공률 낮은 도박 샷을 유도할 수 있다.
- `--trunc-penalty 0`. 30샷 상한은 진짜 시간 제한(인위적 절단)이므로 **기존 버퍼의 bootstrap 처리가 이 경우엔
  정답**이고, 마감이 없으니 가치는 공 배치만의 함수라 샷 순번도 필요 없다. → (a)(b)가 **env 코드 수정 없이
  플래그만으로** 동시에 사라진다.

**필요한 코드 수정**: best ckpt 선정 기준이 pocket rate 이동평균인데, 30샷이면 pocket rate가 거의 100%로 포화돼
선정 기준으로 쓸 수 없다 → `--best-ckpt-metric {pocket_rate, mean_reward}` 추가(기본 pocket_rate로 기존 동작
유지). 최종 500ep 평가에 평균 샷 수(전체 / 다 넣은 게임만)와 **첫 5샷 안의 pocket rate**(기존 63%와 대략적
비교용 — 새 정책은 5샷 마감을 모르므로 참고치)를 추가로 기록한다. `run_grid.sh`는 `COMMON`을 환경변수로
덮어쓸 수 있게 한다.

**가설**: Q의 학습 목표가 실제 return과 일치하면, 기존의 중반 피크 → 후반 하락 없이 학습이 단조롭게 개선된다.

**설정**: vanilla SAC, seed 0~3 순차 실행. `--n-balls 3 --max-steps 30 --step-penalty 1.0 --trunc-penalty 0
--total-steps 2000000 --n-envs 10 --learning-starts 5000 --eval-freq 5000 --eval-episodes 50
--best-ckpt-window 5 --best-ckpt-metric mean_reward`. 나머지 하이퍼파라미터(γ=0.99 등) 동일. eval-freq를
1000→5000으로 올린 이유: 게임당 샷 수가 늘어(초반엔 최대 30) 평가 비용이 커진다.

**지표**: clear rate(30샷 안에 다 넣은 비율), 평균 샷 수(전체 / 다 넣은 게임만), mean reward, 첫 5샷 내 pocket
rate(참고). 학습 곡선에서 후반 하락 여부(peak MA vs 후반 30% 평균).

**판단 기준**: 후반 30% 평균이 peak MA에서 거의 떨어지지 않으면(기존은 7~8pp 하락) → 어긋난 Q 목표가 후반 붕괴의
원인이었다는 가설 지지. 결과와 무관하게 이 설정을 새 기준선으로 삼고, 이후 geom/WM arm은 이 위에서 비교한다.
기존 5샷 설정 수치(63% 등)와는 직접 비교하지 않는다.

**코드**: `exp16_wm/train.py`, `exp16_wm/run_grid.sh`, 테스트 `tests/test_exp16_wm_train.py`.
브랜치 `feature/exp16-clear-speed-reward`, `dev`로만 머지.

**결과**: 4시드 완료 후 TBD. 아래는 seed 0 중간 기록. **(2026-10-09 계획 변경)** 엔트로피 실험과 seed 0 하나씩 먼저 비교하기로 해서
seed 1은 820k step에서 중단했다(`exp16_vanilla_multi3_ms30_s1_2026-10-09@0741_ABORTED_stopped`, 분석 제외). seed 1~3은 필요하면 나중에.

**seed 0 (2026-10-09, `exp16_vanilla_multi3_ms30_s0_2026-10-09@0639`, 61.7분)**: best ckpt 500게임 — 평균
7.89샷(다 넣은 게임 7.81), clear 99.6%, mean reward −5.73, 첫 5샷 pocket 64.2%(참고: 예전 5샷 설정 vanilla 63%).
학습 곡선(50게임 eval): 0–100k 11.4샷 → 100–200k 7.4샷 → 이후 2M까지 7.7샷에서 평평. **예전 설정의 "중반 피크 →
후반 하락"은 사라졌지만, 15만 step 이후 개선도 없다.** 샷당 넣은 공은 ~0.39개(3 ÷ 7.7)로 예전 설정(1.89 ÷ 4.6
≈ 0.41)과 같다 — 과제 정의를 고쳐 학습은 안정됐지만 샷 실력은 그대로다. critic loss는 1.2 → ~9로 올랐다가 멈춤
(return 크기가 −5~−10으로 커진 영향).

**seed 0 정책은 무엇을 배웠나 — 조준 분석** (best ckpt, deterministic, 400샷, ghost-ball 각도와 비교):
- **조준 대신 "세게 치기"를 배웠다.** 속도 중앙값 7.14m/s(p10–p90: 6.77–7.33, 상한 8.0)로 거의 항상 최대 속도에 가깝게
  친다. 각도는 가장 가까운 공 중심에서 중앙값 1.18°(90%가 1.6° 이내)만 비튼다 — 가장 가까운 공을 거의 정면으로 세게
  맞혀 공들을 흩어 놓는 전략이다.
- 샷당 성공률 38.5%. 가장 가까운 ghost-ball 정답 각도와의 오차는 중앙값 2.05°이고, 0.2° 안에 드는 샷은 4.5%뿐이다.
  정답 각도에서 5° 넘게 벗어난 샷도 43%가 들어간다 — 성공이 조준이 아니라 세게 쳐서 생기는 우연에서 온다.
- 같은 속도로 각도만 그 ghost-ball 정답 각도로 바꾸면 70.8%가 들어간다(정책 자신은 38.5%). 플래너 분석에서 3m/s로
  쳤을 때 94%였던 것과 비교하면, 세게 치는 것 자체도 손해다.
- 해석: 포켓되는 각도 폭이 좁은(~0.07%) 지금 행동 공간에서는 정밀 조준이 바늘 찾기라, SAC가 각도에 둔감한 "세게
  치기" 국소 최적에 머문 것으로 보인다. roadmap의 행동 재파라미터화(δ = 0이 곧 정답 조준)가 정면으로 겨냥하는 문제.

**왜 "세게 치기"인가 — SAC 엔트로피 보너스 측정** ("intrinsic reward가 필요한가?"라는 질문에서 출발. SAC에는 이미
엔트로피 보너스라는 intrinsic reward가 있어서, 그게 학습 중 조준을 얼마나 흔드는지부터 쟀다. seed 0 best ckpt, 60개 상태):
- 학습 중(stochastic) 정책의 **조준각 흔들림은 std 4.5°**(p10–p90: 4.1–4.8°), 속도 흔들림은 std 1.28m/s다.
  `target_entropy = −action_dim = −2`(tanh 공간)이고 각도 범위가 ±180°라 이 정도 노이즈가 유지된다. 포켓되는 각도 폭
  (±0.1~0.2°)보다 20배 이상 넓다.
- 이 노이즈를 낀 채로 비교하면: 정책 자신(세게 치기) **33.0%** vs **정답 ghost-ball 조준(3m/s) + 같은 각도 노이즈 21.2%**
  (노이즈가 없으면 같은 상태들에서 100%).
- 즉 SAC가 최적화하는 목표(노이즈를 낀 기대 보상) 기준으로는 **세게 치기가 정밀 조준보다 실제로 낫다.** SAC는 합리적으로
  국소 최적을 고른 것이고, 원인은 탐험 부족이 아니라 엔트로피 보너스가 강제하는 노이즈 폭이다.
- 함의: (1) 호기심(예측 오차) 계열 intrinsic reward는 역효과일 가능성이 크다 — 세게 쳐서 공이 혼돈스럽게 튀는 샷이 가장
  예측하기 어려워 오히려 더 큰 보상을 받는다(noisy-TV 문제). 탐험도 부족하지 않다(샷의 1/3이 이미 들어감).
  (2) 해결 후보: **엔트로피 목표를 낮추기**(조준 흔들림 ~0.2°가 되려면 각도 차원만 ~−5 nats, 합계 ~−6~−7; 초반엔 −2로
  시작해 줄여 가는 annealing), 또는 **행동 재파라미터화** — δ 범위를 ±2°로 잡으면 같은 엔트로피 목표에서도 각도 흔들림이
  4.5° × 2/180 ≈ 0.05°로 자동으로 줄어든다.

### 기준점: pooltool 탐색 플래너 (같은 과제, 2026-10-09 계획)

**왜**: 지금까지 "63%"나 "68.6% 천장"이 높은지 낮은지 판단할 기준이 없었다. pooltool은 결정론적이고 샷 1회
시뮬레이션이 ~2ms라, 매 샷 후보를 미리 쳐보고 고르는 플래너를 RL과 **같은 과제·같은 평가**로 돌릴 수 있다.
이 값은 RL이 아니라 "이 과제에서 적어도 이만큼은 가능하다"는 기준점이다.

**방법 (greedy 1-샷 탐색, 학습 없음)** — 매 샷:
1. 남은 공 × 포켓 6개마다 ghost-ball 조준각(큐볼이 공을 포켓 방향으로 보내는 접촉 위치를 향한 각도)을 계산.
   컷 각도 > 80°인 조합은 제외.
2. 각 조준각에 오프셋 {−0.4, −0.2, 0, +0.2, +0.4}° × 속도 {1.5, 3.0, 5.0} m/s → 후보 ≤ 270개.
3. 후보마다 `env.system.copy()`에서 env와 **똑같은 조준 규칙**(가장 가까운 공 방향 + delta_angle)으로 쳐보고,
   즉시 리워드 기준(넣은 공 수 − 0.5×스크래치)으로 최선을 고른다. 동점이면 남은 공이 포켓에 더 가까운 쪽.
4. 1단계에서 넣는 후보가 없으면 무작위 행동 200개로 한 번 더 탐색(뱅크·키스 샷 우연 포함).
5. 고른 샷을 **행동 (delta_angle, speed)로 변환해 `env.step`에 넣는다** — 물리·리워드·종료 규칙이 RL과 완전히 같다.

**조건**: 위 재정의 과제와 동일 — `BilliardsEnv(n_balls=3, max_steps=30, step_penalty=1.0, trunc_penalty=0)`.
평가는 RL 최종 평가와 같은 방식(`reset(seed=s)` 후 500게임, s=0~3). 단, 스크래치 시 큐볼 재배치가 같은 RNG를
쓰므로 첫 스크래치 이후 게임들의 시작 배치는 RL과 1:1로 같지 않다(분포는 같음).

**지표**: RL과 동일 — clear rate, 평균 샷 수(전체 / 다 넣은 게임만), 첫 5샷 pocket rate, mean reward.
추가로 샷당 성공률(1개 이상 넣은 샷 비율)과 샷당 계획 시간.

**한계**: 이번 샷만 보는 greedy라 다음 샷 포지션을 고려하지 않는다 → 진짜 천장이 아니라 **하한 쪽 기준점**.

**판단 기준**: 플래너가 RL보다 확실히 적은 샷으로 다 넣으면 → RL이 아직 못 배우는 여지가 크다(학습 쪽 문제).
비슷하면 → 이 과제는 이미 천장 근처이고, 과제를 더 어렵게(포지션 플레이 등) 만들어야 의미 있는 비교가 된다.

**코드**: `exp16_wm/planner.py`, 테스트 `tests/test_exp16_wm_planner.py`. 실행은 학습과 나란히 코어 1개(`nice`).

**결과 (2026-10-09)**: `python -m exp16_wm.planner --seeds 0 1 2 3 --n-games 500` → `logs/planner_clear.json`.

| seed | clear | 평균 샷 (전체 / 다 넣은 게임) | 첫 5샷 pocket | mean reward | 샷당 성공률 | 무작위 단계 샷 | 예측 불일치 |
|---|---|---|---|---|---|---|---|
| 0 | 100.0% | 2.220 / 2.220 | 100.0% | +0.748 | 100.0% | 9 | 0 |
| 1 | 99.8% | 2.274 / 2.218 | 99.93% | +0.685 | 97.45% | 13 | 0 |
| 2 | 100.0% | 2.228 / 2.228 | 100.0% | +0.736 | 100.0% | 18 | 0 |
| 3 | 100.0% | 2.238 / 2.238 | 100.0% | +0.726 | 100.0% | 10 | 0 |
| **평균 ± std** | **99.95 ± 0.10%** | **2.240 ± 0.024 / 2.226 ± 0.009** | 99.98% | +0.724 ± 0.027 | 99.4% | | 0 |

계획 시간은 샷당 ~0.18초(seed 1은 아래 버그 게임 때문에 1.3초).

**샷 분해** (별도 분석, seed 0, 50게임/116샷):
- 한 샷에 넣은 개수: 0개 0%, 1개 72.4%, 2개 25.9%, 3개 1.7%. 게임 첫 샷은 2개 54%, 3개 4%.
- 무작위 행동은 6.7%만 포켓(200개 기준) — 행동 공간 대부분이 헛샷.
- ghost-ball 직선 조준(오프셋 0, 3m/s)만으로 94%의 샷에서 들어가는 (공, 포켓) 조합이 있다. 샷당 ghost 후보
  ~58개 중 50%가 실제로 들어간다.

**해석**:
1. **이 과제는 천장 근처가 아니다.** 같은 조건에서 2.24샷이 가능한데, vanilla RL은 seed 0 ~480k step 시점에
   ~7.6샷이다(학습 진행 중, 최종 비교는 4시드 완료 후). 병목은 학습 쪽이다.
2. **격차의 대부분은 조준 기하다.** 매 샷 1개씩 확실히 넣으면 3.0샷 — 7.6 → 3.0(~4.6샷)은 조준 기하로 해결되는
   부분이다. 3.0 → 2.2(~0.8샷)는 한 샷 다중 포켓, 즉 **충돌 뒤 물리**(큐볼이 꺾여 회전·마찰로 휘는 경로, 공끼리의
   연쇄)로, 직선 기하로는 계산되지 않고 플래너도 시뮬레이션으로 쳐보고 찾았다.
3. 기하만으로 94% → roadmap의 "행동 재파라미터화((공, 포켓) 선택 + ghost-ball 보정각 + 속도)" 제안의 근거.
4. 예측 불일치 0건 — 플래너가 복사본에서 본 결과와 실제 `env.step` 결과가 모든 샷에서 같았다(pooltool 결정론 확인).

**기하 없이 탐색만으로는?** ("bitter lesson을 생각하면 기하를 넣는 게 의미가 있나?"라는 질문에서 출발.) ghost-ball 후보를
끄고(`max_cut_deg=-1`) 무작위 행동 후보 + 시뮬레이션만으로 고르게 해서 seed 0, 50게임:

| 후보 수/샷 | 게임당 샷 | clear | 샷당 성공률 |
|---|---|---|---|
| 무작위 50개 | 2.98 | 100% | 94.0% |
| 무작위 200개 | 2.50 | 100% | 100% |
| (참고) ghost-ball 기하 + 탐색 | 2.24 | 99.95% | 99.4% |
| (참고) vanilla SAC | 7.89 | 99.6% | ~38% |

→ **사람의 기하 지식 없이 계산(탐색)만으로도 풀린다.** 기하는 탐색을 조금 더 효율적으로 만들 뿐 필수가 아니다. SAC가 못
푸는 이유는 지식 부족이 아니라, 노이즈 낀 정책 그래디언트로 좁은 정답 각도를 찾는 학습 신호가 약하기 때문으로 보인다.

**발견한 시뮬레이터 버그 — 포켓 입구에 낀 공**: seed 1의 233번째 게임에서 큐볼이 y ≈ 1.95~2.05(테이블 길이
1.981m 밖), 좌상단 코너 포켓(0, 1.981) 입구 안쪽에 있었는데 pooltool이 포켓으로 처리하지 않았다. 이 상태에서는
어떤 샷이든 시뮬레이션 1회가 ~340ms(평소 ~2ms, 이벤트 상한 2000까지 도는 것으로 보임)였고, 남은 공
(0.8466, 0.6576)은 30샷 내내 한 번도 맞지 않았다 — 4시드 2000게임 중 유일하게 못 끝낸 게임. 같은 시뮬레이터라
RL도 이 상태에 빠질 수 있고, obs는 좌표를 [0, 1]로 잘라서 보여 주므로 정책에는 "큐볼이 테이블 끝에 있다"로만
보인다. 드물어서 결과 영향은 작지만, **아직 고치지 않았다**(vanilla clear 4시드가 현재 env로 진행 중). 수정 후보:
테이블 경계 밖으로 나간 공은 포켓된 것으로 처리(큐볼이면 스크래치).

### 엔트로피 목표 낮추기: 조준 흔들림을 포켓 폭 수준으로 (2026-10-09 계획)

**왜**: 위 측정에서 학습 중 SAC 정책의 조준 흔들림이 std 4.5°였고, 이 노이즈를 낀 채로는 정답 조준(21%)이 세게
치기(33%)보다 못했다. SAC는 노이즈를 낀 기대 보상을 최적화하므로 세게 치기 국소 최적이 합리적인 선택이 된다.
`target_entropy = −2`(tanh 공간, 2차원)가 이 노이즈 폭을 유지시키는 원인이다. "intrinsic reward를 더 주자"는 질문에서
출발했지만, SAC의 엔트로피 보너스 자체가 intrinsic reward이고 이게 과하다는 결론이어서 방향을 반대로 잡는다.

**검토 후 미룬 대안 — actor/critic을 GNN으로**: 원인이 노이즈 폭이라, 노이즈가 4.5°인 동안에는 네트워크가 무엇이든
세게 치기가 최적이다. 같이 바꾸면 무엇이 효과였는지 가를 수 없으므로 이번에는 MLP 그대로 엔트로피만 바꾼다. 엔트로피를
고친 뒤에도 "좌표에서 정밀한 각도를 계산"하는 것이 병목으로 남으면 GNN(상대 위치 엣지)과 행동 재파라미터화가 후보다.

**변경**: `target_entropy`를 −2에서 −7로, 처음 1M env step 동안 선형으로 낮추고 이후 −7 유지.
- −7의 근거: 조준 흔들림 std ~0.2°(각도 범위 ±180° → tanh 공간 σ ≈ 0.0011)가 되려면 각도 차원 엔트로피가 ~−5.4 nats,
  속도 흔들림 ~0.3m/s면 속도 차원이 ~−1.5~−2 nats → 합계 ~−7. 실제 차원별 배분은 alpha 자동 조정과 Q가 정한다.
- 초반에 −2로 시작하는 이유: 처음부터 낮추면 탐험이 일찍 끊길 위험.
- 코드: `exp16_wm/train.py`에 `--target-entropy`, `--target-entropy-end`, `--target-entropy-steps` 추가(기본값은 기존
  동작 그대로 −action_dim 고정). sac.py는 수정하지 않는다(모든 SAC 클래스가 `target_entropy` 속성을 공유).

**가설**: 학습 중 조준 흔들림이 포켓 폭 수준으로 줄면, 정밀 조준이 학습 목표 기준으로도 세게 치기보다 나아져 정책이
세게 치기 국소 최적에서 벗어난다 → 게임당 샷 수가 vanilla clear(seed 0: 7.89)보다 줄어든다.

**설정**: vanilla clear와 동일(`--n-balls 3 --max-steps 30 --step-penalty 1.0 --trunc-penalty 0 --total-steps 2000000
--n-envs 10 --learning-starts 5000 --eval-freq 5000 --eval-episodes 50 --best-ckpt-window 5 --best-ckpt-metric
mean_reward`) + `--target-entropy -2 --target-entropy-end -7 --target-entropy-steps 1000000`.
**실행 계획 변경(2026-10-09)**: 4시드 대신 먼저 vanilla clear와 **같은 seed 0 하나만** 1:1로 비교한다(사용자 결정 —
방향부터 빠르게 확인). 효과가 보이면 시드를 늘린다. 2026-10-09 08:05 시작.

**지표**: 평균 샷 수(주 지표, vanilla clear와 시드별 paired 비교), 샷당 성공률, 학습 곡선. 정책 분석: 속도 분포(7m/s대에서
내려오는가), ghost-ball 정답 각도와의 오차, 학습 중 조준 흔들림 std.

**판단 기준**: 샷 수가 vanilla clear보다 4시드 일관되게 줄고 속도·각도 오차가 함께 줄면 → 엔트로피 노이즈가 국소 최적의
원인이었다는 가설 지지. 흔들림은 줄었는데 샷 수가 그대로면 → 노이즈가 아니라 critic/actor가 좁은 정답 각도를 표현하지
못하는 것이 병목 → 행동 재파라미터화로.

**결과 (2026-10-09, seed 0, `exp16_vanilla_ent-2to-7_multi3_ms30_s0_2026-10-09@0805`, 70.6분)**: best ckpt 500게임.

| | vanilla clear | 엔트로피 −2→−7 |
|---|---|---|
| 게임당 샷 (전체 / 다 넣은 게임) | 7.89 / 7.81 | **7.39 / 7.35** |
| clear | 99.6% | 99.8% |
| mean reward | −5.73 | −5.21 |
| 첫 5샷 pocket | 64.2% | 64.5% |
| 학습 곡선 1–2M 구간 (50게임 eval) | 7.7샷 | 7.3샷 |

정책 분석(best ckpt, 300샷, vanilla와 같은 방식):

| | vanilla clear | 엔트로피 −2→−7 |
|---|---|---|
| 학습 중 조준 흔들림 std | 4.39° | **0.34°** |
| 학습 중 속도 흔들림 std | 1.26m/s | 0.12m/s |
| 실제 치는 속도 (중앙값) | 7.13m/s | **7.68m/s** |
| ghost-ball 정답 각도와의 오차 (중앙값) | 2.07° | **2.08°** |
| 정답 0.2° 이내 샷 | 4.7% | 7.0% |
| 샷당 성공률 | 38.7% | 35.7% |

**해석**:
- 엔트로피 목표는 계획대로 내려갔다(1M step부터 log_pi +7.00). 학습 중 조준 흔들림이 4.39° → 0.34°로 포켓 폭 수준까지
  줄었다 — 노이즈 폭 자체는 의도대로 바뀌었다.
- **그런데 정책은 여전히 세게 친다.** 조준 오차는 2.08°로 그대로이고 속도는 오히려 더 세졌다(7.68m/s). 샷 수 개선은
  0.5샷(~6%)뿐 — 시드 1개라 유의성은 판단할 수 없다.
- 정책은 15만 step쯤, 엔트로피가 아직 −2 근처일 때 세게 치기로 굳었고, 이후 엔트로피를 낮추자 세게 치기를 더 확실하게
  하는 쪽으로만 갔다. 세게 치기 주변에서 Q의 기울기는 좁은 정답 각도를 가리키지 않으므로, 노이즈가 작아져도 빠져나올 길이 없다.
- **판단 기준상 "흔들림은 줄었는데 샷 수가 그대로"** — 노이즈는 국소 최적의 원인이 아니었거나 그것만으로는 부족하다. RL
  신호로는 좁은 정답 각도를 찾아가지 못한다. 행동 재파라미터화는 bitter lesson 관점에서 보류했으므로, 다음은 정답 각도를
  직접 주는 **모방학습**(아래 섹션)이다.
- 한계: 시드 1개. −2에서 시작해 낮췄기 때문에 이미 갇힌 뒤에 노이즈가 줄었다 — 처음부터 낮은 엔트로피로 시작한 경우는
  확인하지 않았다(탐험이 일찍 끊길 위험).

### 모방학습: 탐색 플래너를 교사로 초기 정책 만들기 (2026-10-09 계획)

**왜**: SAC는 노이즈 낀 정책 그래디언트로 행동 범위의 ~0.07%인 정답 각도를 찾아야 해서 "세게 치기" 국소 최적에 머문다.
탐색 플래너는 같은 과제를 2.2~2.5샷에 푼다. 그렇다면 플래너가 고른 샷을 정책이 **지도학습으로 따라 배우게** 해서 초기
정책을 만들면, 노이즈 그래디언트 없이 정답 각도를 직접 가르칠 수 있다(roadmap "탐색 + 학습"의 0단계). 그 자체로 진단도
된다 — 정답을 그대로 줘도 MLP가 좌표에서 정밀 조준을 따라 배우지 못하면, 병목은 RL 신호가 아니라 표현력이다.

**교사 선택 — 기하 플래너 먼저, 무작위 탐색 플래너는 다음**: 두 플래너는 후보를 만드는 방식만 다르고("어디를 쳐볼지"를
기하가 정하느냐, 무작위로 많이 쳐보느냐), 시뮬레이션으로 확인하고 고르는 과정은 같다. bitter lesson상 최종적으로는 사람
지식이 없는 무작위 탐색 교사가 맞지만, 그 라벨은 "들어가는 아무 각도·아무 속도"라 지저분해서 따라 배우기 어렵다. 기하 교사는
라벨이 깔끔해(정답 조준 ±0.4°, 속도 3개 값) **"MLP가 정밀 조준을 따라 배울 수 있나"를 빠르게 확인**하기 좋다(사용자 결정).
되면 무작위 탐색 교사(+ 라벨 다듬기: 들어가는 구간의 가운데로, 규칙으로 하나 고르기)로 넘어간다.

**알려진 위험**:
- 정답이 여러 개(멀티모달): 같은 배치에서도 들어가는 샷이 여러 (공, 포켓) 조합에 흩어져 있다. 가우시안 정책이 평균을 내면
  두 정답 사이를 쳐서 헛샷. 이번엔 교사의 동점 처리 규칙(남은 공이 포켓에 가까운 쪽)만으로 진행하고, 데이터에 상태마다
  "들어가는 후보 전체"도 저장해 두어 나중에 혼합분포 출력이나 집합 손실(가장 가까운 정답 기준)을 쓸 수 있게 한다.
- 분포 이동: 정책이 실수하면 교사가 본 적 없는 배치로 간다 → 필요하면 DAgger(정책이 간 상태에서 교사에게 다시 묻기).

**단계**:
1. 데이터: 기하 플래너로 게임을 진행하며 매 샷 (obs, 교사 행동, 들어가는 후보 전체) 저장. 시드별로 나눠 프로세스 3개 병렬.
   목표 ~5만 샷. 코드 `exp16_wm/teacher_data.py`.
2. 학습: SAC와 같은 TanhGaussian actor를 교사 행동의 log-likelihood로 학습(나중에 SAC 미세조정에 그대로 쓰기 위해).
   코드 `exp16_wm/bc.py`.
3. 평가: 탐색 없이 정책 단독(deterministic)으로 clear-speed 과제 500게임 — 게임당 샷, 샷당 성공률. 기준점: vanilla SAC
   7.89샷 / 기하 플래너 2.24샷. 보조: 검증 상태에서 교사 각도와의 오차, 정책 각도가 들어가는 후보 집합에 얼마나 가까운가.

**판단 기준**: 정책 단독이 SAC(7.89샷)보다 확실히 적은 샷으로 끝내면 → 정답을 주면 배운다(학습 신호 문제였다). SAC 수준이면 →
멀티모달/표현력 문제 → 혼합분포 출력이나 집합 손실로.

**브랜치**: `feature/exp16-imitation-teacher`(`feature/exp16-clear-speed-reward`에서 분기 — 플래너 코드 의존), `dev`로만 머지.

**중간 결과 (2026-10-09, 데이터 생성 중 ~1.2만 샷으로 먼저 테스트)**: 교사 데이터는 시드 1000~1002로 생성 중이고
(게임당 2.23~2.25샷, 교사 샷 성공률 99.7~99.9%), 500게임마다 중간 저장된 파일로 먼저 돌렸다. 평가는 정책 단독 500게임(seed 0).

| 학습 | 데이터 | 정책 단독 게임당 샷 | 샷당 성공률 | 정답 각도 오차 중앙값 (교사 / 가장 가까운 정답) | 0.2° 이내 (가장 가까운 정답) |
|---|---|---|---|---|---|
| `teacher`, [−1,1] 단위 MSE | 1.0만 | 19.17 | 13.5% | 8.28° / 3.19° | 9.2% |
| `nearest_good`, [−1,1] 단위 MSE | 1.0만 | 9.31 | 31.4% | 5.29° / 1.63° | 12.1% |
| `teacher`, 물리 단위 | 1.2만 | 20.78 | 12.0% | 8.15° / 3.15° | 7.7% |
| `nearest_good`, 물리 단위 | 1.3만 | 9.40 | 31.0% | 5.95° / 1.72° | 11.8% |
| `nearest_good`, 물리 단위, 1500 epoch(조기 종료 없음) | 1.3만 | 9.24 | 31.6% | 5.96° / 1.59° | 12.6% |
| (참고) vanilla SAC | — | 7.89 | 38.7% | — / 2.07°(ghost 정답 기준) | 4.7% |

- **손실 설계 실수와 수정**: 처음엔 [−1, 1] 단위로 MSE를 계산했는데, 이 단위에서 각도 1°는 0.0056, 속도 1m/s는 0.27이라
  손실이 거의 속도 오차로 채워졌다. 각도 0.2°·속도 0.5m/s를 오차 1로 정규화했지만(`--angle-scale-deg`, `--speed-scale`)
  결과는 거의 같았다 — 병목이 아니었다.
- **`teacher`(교사 행동 그대로)는 정답이 여러 개라 망가진다**: 교사 대비 오차 8°, 정책 단독 19~21샷. 가장 가까운 정답 기준으로 바꾼
  `nearest_good`이 훨씬 낫다(9.2~9.4샷) — 멀티모달 문제가 실재한다.
- **`nearest_good`도 과적합**: 조기 종료 없이 1500 epoch 학습하면 train loss 4557 → 145로 계속 내려가지만 val loss는
  3108(epoch 3) → 4665로 올라간다. 학습한 배치는 외우지만 처음 보는 배치로 일반화하지 못한다. 정책 단독 9.2샷으로 SAC(7.89)보다
  못하고, 정답 각도 오차 중앙값 1.6°도 SAC(2.07°)와 같은 수준이다.
- 해석: 좌표 → 0.1° 정밀도의 조준(행동 범위 대비 ~1/3600 정밀도)은 MLP가 1만 샷으로는 일반화하지 못하는 함수다. RL 신호(SAC)
  문제와 별개로 **표현/데이터 문제**가 있다는 첫 증거. 다음 확인: (1) 데이터 양을 늘리면 val 오차가 줄어드는가(데이터 크기별
  곡선), (2) 줄지 않으면 입력 표현(상대 좌표, 공·포켓 관계를 다루는 GNN 등)이 병목.

**모델을 키우면?** ("모델을 더 키우면 되잖냐"라는 질문에서 출발. 그 사이 데이터가 ~2.2만 샷으로 늘어 두 모델 모두 같은 데이터,
`nearest_good`, 500 epoch 조기 종료 없음, lr 1e-3.)

| | 기본 actor (256×2, SAC와 동일) | 큰 actor (1024×3) |
|---|---|---|
| train loss (epoch 0 → 490) | 3684 → 489 | 13300 → 209 |
| val loss 최저 (epoch) / 끝 | 3071 (1) / 4464 | 3293 (37) / 4009 |
| 가장 가까운 정답 각도 오차 중앙값 / 0.2° 이내 | 1.67° / 12.0% | 1.50° / 13.5% |
| 정책 단독 게임당 샷 / 샷당 성공률 | 10.44 / 27.9% | 11.15 / 26.0% |

- 큰 모델은 더 잘 외울 뿐(train 209) 일반화는 같다. 이미 과적합 상태(train↓ val↑)라 용량이 병목이 아니다 — 모델만 키우는 건 답이
  아니고, 데이터를 크게 늘릴 때 같이 키우는 것이 맞다.
- 데이터를 1.3만 → 2.2만으로 늘려도 각도 오차(1.6° → 1.67°)와 정책 단독 성능(9.2 → 10.4샷)이 나아지지 않았다. 두 배 정도로는 판단하기
  이르지만, 데이터 크기별 곡선이 이 구간에서 평평할 수 있다는 첫 신호.
- 두 모델 모두 val loss 최저가 학습 극초반(1~37 epoch)이다. 제곱 오차가 "다른 정답 쪽으로 크게 빗나간" 소수 샘플에 지배되어,
  학습이 진행될수록 외운 배치의 정답 쪽으로 가면서 처음 보는 배치에서 크게 틀리는 샘플이 늘어나는 것으로 보인다.

### 모방학습 입력 표현: 상대 좌표 (행동과 같은 기준틀) (2026-10-09 계획)

**왜**: 위 BC 결과에서 좌표 → 정밀 조준 함수를 MLP가 일반화하지 못했다(과적합, 모델을 키워도 같음, 데이터 1.3만 → 2.2만에도 그대로).
데이터를 수십 배 늘리기 전에, 입력 표현이 벽인지를 가장 싸게 확인한다. GNN(공·포켓 관계, 공 개수 일반화)이 결국 필요하지만,
상대 좌표가 훨씬 간단해서 먼저 한다(사용자 결정).

**표현** (`--obs-transform relative`, 23차원 obs만으로 계산):
- 큐볼을 원점으로 옮기고, **큐볼 → 가장 가까운 남은 공 방향이 x축이 되도록 회전**한다 — 행동 `delta_angle`의 기준 방향과 같은 틀.
  env의 조준 규칙(`_nearest_unpocketed_pos`)과 같은 방식으로 기준 방향을 정한다.
- 특징 = 공마다 (회전된 상대 위치 2 + 포켓됨 플래그 1) × 3 + 포켓마다 회전된 상대 위치 2 × 6 = 21차원(미터 단위). 포켓된 공은 0.
  큐볼의 절대 위치는 넣지 않는다 — 테이블 경계 정보는 포켓 상대 위치에 남아 있다.
- 당구 공식(ghost-ball 등)은 넣지 않는다. 좌표계 선택만 바꾸는 것이라 기존 `delta_angle`(공 방향 기준)과 같은 수준의 사전 지식이다.

**비교**: 같은 데이터 스냅샷에서 raw(기존) vs relative, `nearest_good`, 물리 단위 손실, 기본 actor(256×2), 500 epoch. 지표는 이전과 같다
(가장 가까운 정답 각도 오차 중앙값·0.2° 이내 비율, 정책 단독 500게임 샷 수).

**판단 기준**: relative가 각도 오차를 크게 줄이고(예: 1.6° → 0.5° 이하) 정책 단독 샷 수가 SAC(7.89)보다 나아지면 → 입력 표현이 주된
병목이었다 → 이 표현으로 데이터 크기별 곡선, 이후 GNN. 그대로면 → 데이터 양 쪽(50만 샷)을 확인.

**코드**: `exp16_wm/obs_transform.py`(+ `bc.py`의 `--obs-transform`), 테스트 `tests/test_exp16_wm_obs_transform.py`.

**결과 (2026-10-09, 같은 데이터 스냅샷 `logs/teacher_data/snap_0939`, 3.0만 샷)**: `nearest_good`, 물리 단위 손실, 기본 actor,
500 epoch, lr 1e-3. 정책 단독 500게임(seed 0).

| | raw 좌표 | 상대 좌표 |
|---|---|---|
| val loss 최저 (epoch) | 3204 (2) | **2540 (34)** |
| 가장 가까운 정답 각도 오차 중앙값 | 1.53° | **1.11°** |
| 정답 0.2° 이내 | 13.1% | **18.1%** |
| 정책 단독 게임당 샷 | **8.95** | 11.28 |
| 샷당 성공률 | **32.7%** | 25.6% |

- **상대 좌표는 일반화를 실제로 돕는다**: val loss 최저가 더 낮고 더 늦게(epoch 34) 나오며, 각도 오차가 1.53° → 1.11°(−28%),
  0.2° 이내가 13% → 18%로 좋아졌다. 입력 표현이 병목의 일부라는 증거.
- **그런데 정책 단독 성능은 오히려 나빠졌다** (8.95 → 11.28샷). 둘 다 1° 안팎 오차라 포켓 폭(~0.1~0.2°)에 한참 못 미치는 구간이고,
  여기서는 각도가 조금 나아져도 성공률이 오르지 않는다 — 성공 여부는 오차가 포켓 폭 안에 들어가느냐의 계단 함수에 가깝다.
  게다가 교사를 따라 **부드럽게(1.5~5m/s) 치는** 정책은 조준이 1°만 틀려도 깨끗이 빗나가고 공이 흩어지지도 않는다 — SAC의
  "세게 치기"(7m/s대, 우연 포함 ~38%)보다 못한 이유로 보인다. 그래서 정책 단독 샷 수는 이 구간에서 표현 품질을 재는 지표로
  둔감하다; 각도 오차 중앙값·0.2° 이내 비율을 주 지표로 본다.
- 다음: 상대 좌표로 데이터 크기별 곡선(전체 ~5만 샷의 25/50/100%) → 0.2° 이내 비율이 데이터와 함께 오르는지. 오르면 대량 생성(50만 샷),
  안 오르면 GNN.

### 모방학습 데이터 증강: 테이블 대칭 뒤집기 ×4 (2026-10-09 계획)

**왜**: BC의 병목은 데이터 부족으로 인한 일반화 실패다(과적합, 모델 키워도 같음). SSM 설계 때 쓰던 기법 중 이 문제에 정면으로
맞는 것이 **좌우/상하 뒤집기 증강**이다(Fixed-Δt WM에서 50% 확률로 사용). 테이블과 물리가 대칭이라, 공 위치를 뒤집으면 정답 행동도
정확히 따라 바뀐다 — 근사가 아닌 정확한 라벨로 데이터가 공짜로 4배. 당구 지식이 아니라 "테이블이 대칭"이라는 사실만 쓴다.
(다른 SSM 기법의 접목 검토: R-SSM의 상대 위치 엣지 + 공유 가중치 + attention pool → GNN 정책 설계로 다음 단계, 드문 큰 오차 대응
가중치 → robust 손실(Huber)로 나중에, scheduled sampling → DAgger로 정책이 쓸 만해진 뒤, 커리큘럼은 v16에서 망각을 일으켜 제외.)

**변환** (raw obs에서, 정규화 좌표):
- 좌우(LR): 큐볼·남은 공 x → 1 − x. 상하(TB): y → 1 − y. 포켓 목록은 그대로(뒤집어도 같은 포켓 집합·같은 순서), 포켓된 공은 (0, 0, 1) 그대로.
- 행동: 한 번 뒤집으면(LR 또는 TB) `delta_angle → −delta_angle`, 두 번(LR+TB = 180° 회전)이면 그대로. 속도는 그대로.
  가장 가까운 공(조준 기준)은 거리가 보존되므로 그대로다. 들어가는 후보 전체(`good_actions`)도 같은 규칙.

**물리 검증 (pooltool, 구현 전)**: seed 500, 40게임에서 매 샷 상태를 뒤집고 행동도 뒤집어 쳤을 때 "같은 공이 들어가고 스크래치 여부도
같은" 비율 — 교사 행동 99.2%(255건 중 2건 불일치, 긴 샷의 혼돈 효과로 보임), 무작위 행동 100%(255건). 증강 라벨은 99% 이상 정확하다.

**적용**: train/val을 **게임 단위로 먼저 나눈 뒤 train에만** 증강한다(val 게임의 뒤집힌 복사본이 train에 들어가 새는 것을 막기 위해).
코드 `exp16_wm/augment.py`, `bc.py --augment flip4`, 테스트 `tests/test_exp16_wm_augment.py`.

**비교 — 데이터 크기별 곡선**: 상대 좌표, `nearest_good`, 물리 단위 손실, 기본 actor. 데이터 25 / 50 / 100% × 증강 {없음, ×4}.
주 지표는 가장 가까운 정답 각도 오차 중앙값·0.2° 이내 비율(정책 단독 샷 수는 이 구간에서 둔감 — 위 상대 좌표 결과 참조).

**판단 기준**: 데이터·증강과 함께 0.2° 이내 비율이 꾸준히 오르면 → 데이터 문제 → 대량 생성(50만 샷). 평평하면 → GNN.

**결과 (2026-10-09, 스냅샷 `logs/teacher_data/snap_0951`, 4.35만 샷)**: 상대 좌표, `nearest_good`, 물리 단위 손실, 기본 actor(256×2),
500 epoch, lr 1e-3. val은 같은 게임(전체의 10%, 증강 없음). 정책 단독 500게임(seed 0).

첫 실행은 best 체크포인트를 val loss 최저로 골랐는데, val loss(제곱 오차 평균)는 다른 정답 쪽으로 크게 빗나간 소수 샘플에 지배되어 주
지표(0.2° 이내 비율)와 어긋났다 — 예: 50%+증강은 epoch 43에서 저장(0.2° 이내 18.2%), 25%+증강은 우연히 epoch 484에서 저장(28.1%).
그래서 **best를 val의 0.2° 이내 비율 최고로 고르도록 바꿔(`--select within`, 이제 기본값) 같은 6개를 다시 돌렸다**:

| 데이터 | 증강 | train 샘플 | best epoch | 가장 가까운 정답 각도 오차 중앙값 | 0.2° 이내 | 정책 단독 샷 | 샷당 성공률 |
|---|---|---|---|---|---|---|---|
| 25% | 없음 | 9,804 | 495 | 1.078° | 20.8% | 10.72 | 27.0% |
| 50% | 없음 | 19,622 | 422 | 0.944° | 22.7% | 11.39 | 25.5% |
| 100% | 없음 | 39,222 | 387 | **0.482°** | **34.8%** | 9.49 | 30.6% |
| 25% | ×4 | 39,216 | 477 | 0.600° | 31.3% | 9.80 | 29.4% |
| 50% | ×4 | 78,488 | 489 | 0.496° | 34.4% | 10.13 | 28.8% |
| 100% | ×4 | 156,888 | 408 | 0.581° | 30.3% | **8.92** | **32.7%** |
| (참고) 첫 실행 100% 없음 / ×4, val loss 기준 선택 | | | 34 / 72 | 1.18° / 0.73° | 17.9% / 25.5% | 10.69 / 9.59 | |
| (참고) vanilla SAC | | | | 2.07°(ghost 정답 기준) | 4.7% | 7.89 | 38.7% |

**해석**:
- **데이터가 늘면 조준이 좋아진다.** 증강 없이 25 → 50 → 100%에서 0.2° 이내가 20.8 → 22.7 → 34.8%, 각도 오차가 1.08 → 0.94 → 0.48°.
  판단 기준상 "데이터 문제" 쪽 신호다.
- **증강은 데이터가 적을 때 크게 돕는다**: 25%에서 20.8 → 31.3%(데이터 4배와 비슷한 효과). 하지만 데이터가 많아지면 30~34%에서
  평평해지고, 100%에서는 증강 없음(34.8%)과 비슷하거나 낮다(30.3%).
- **모든 실행의 best epoch가 학습 끝(387~495)** — 500 epoch로는 아직 수렴 전이다. 증강 쪽 평탄화가 데이터의 한계인지, 모델 크기
  (256×2)·학습량의 한계인지는 이 실험으로 가를 수 없다.
- 선택 기준만 바꿔도 0.2° 이내가 크게 올랐다(100% 없음 17.9 → 34.8%) — 지표에 맞는 체크포인트 선택이 중요했다.
- **정책 단독은 여전히 SAC보다 못하다**(최고 8.92샷 vs 7.89샷). 0.2° 이내 35% 수준으로는 샷 성공률이 SAC의 "세게 치기"(38.7%)를
  넘지 못한다.
- 다음: 데이터를 크게 늘리고(50만 샷) 모델·학습량도 함께 키워서 0.2° 이내가 계속 오르는지 본다.

### 모방학습: 데이터 50만 샷 + GNN 정책 (2026-10-09 계획)

**왜**: 데이터 크기별 곡선에서 데이터가 늘수록 조준이 좋아졌다(0.2° 이내 21 → 35%) — 지금 모델은 데이터가 부족하다. 두 방향을 동시에 친다.
1. **데이터를 10배로**: 기하 교사로 50만 샷(시드 2000~2006, 프로세스 7개 × 32,000게임, 코어당 ~3.6시간). CPU 작업이라 GNN 구현과 병렬.
2. **GNN 정책**: 지금 MLP는 21개 숫자를 한 줄로 받아 (a) "공 → 포켓" 같은 관계를 위치 차이부터 스스로 계산해야 하고, (b) "공1→포켓3"과
   "공2→포켓3"을 다른 문제로 따로 배우며(공 3 × 포켓 6 = 18조합이 데이터를 나눠 가짐), (c) 공 순서가 무의미하다는 것도 데이터로 배워야 한다.
   GNN은 관계(상대 위치 차이 — 단순 뺄셈, 당구 공식 아님)를 엣지로 직접 주고 모든 (공, 포켓) 쌍이 같은 가중치를 써서, 패턴당 데이터가 최대
   18배가 되는 효과를 구조로 얻는다. 대칭 증강(데이터 4배)이 적은 데이터에서 크게 도운 것과 같은 방향이다. 공 개수가 바뀌어도 쓸 수 있다.

**GNN 설계** (`exp16_wm/gnn_policy.py`, 입력은 상대 좌표 21차원을 구조로 다시 묶음):
- 공 노드: [rx, ry, |r|, 포켓됨, 기준공(가장 가까운 공) 여부]. 공↔공 메시지(상대 위치 차이·거리, 남은 공만)를 합쳐 공 임베딩 h_i.
- (공 i, 포켓 j) 쌍 엣지: [h_i, 포켓 위치, 포켓 − 공, 거리, 사이드 포켓 여부] → 공유 MLP → e_ij. 포켓된 공의 쌍은 마스킹.
- 출력: 쌍마다 행동 하나(delta, speed)와 선택 점수 하나 → **결정적 행동 = 점수 최고 쌍의 행동**. 정답이 여러 개인 문제를 구조로 다룬다
  (정답마다 다른 쌍이 맡으므로 평균을 쳐서 빗나가지 않는다). "샷은 (공, 포켓) 쌍 단위"라는 약한 사전 지식이 들어간다 — ghost-ball 공식보다
  훨씬 약하지만 들어간다는 점을 기록해 둔다.
- 손실: relaxed winner-takes-all — 들어가는 후보에 가장 가까운 쌍(승자)이 회귀(가중치 1−ε, 나머지 쌍 ε, ε=0.05) + 선택 점수는 승자 쌍을
  맞히는 cross-entropy. 회귀는 MLP와 같은 물리 단위 정규화.

**비교**: 같은 5만 샷에서 MLP(상대 좌표) vs GNN, 증강 ×4/없음, `--select within`. 50만 샷이 모이면 둘 다 다시. 지표는 이전과 같다.

**판단 기준**: GNN이 같은 데이터에서 0.2° 이내 비율을 확실히 높이면(예: 35% → 50%+) → 표현(관계·공유)이 병목의 큰 부분. 50만 샷에서 MLP도
따라오면 → 데이터만으로도 되는 문제.

**결과 1 — 같은 5만 샷에서 MLP vs GNN (2026-10-09, 스냅샷 `logs/teacher_data/snap_1014`, 4.9만 샷)**: 상대 좌표, `--select within`,
500 epoch, lr 1e-3, 물리 단위 손실. 정책 단독 500게임(seed 0).

| | MLP | MLP + 증강 ×4 | **GNN** | (참고) vanilla SAC / 엔트로피 낮춤 |
|---|---|---|---|---|
| best epoch | 238 | 170 | 364 | |
| 가장 가까운 정답 각도 오차 중앙값 | 0.581° | 0.504° | **0.419°** | 2.07° / 2.08° (ghost 정답 기준) |
| 정답 0.2° 이내 | 32.1% | 33.2% | **37.5%** | 4.7% / 7.0% |
| 정책 단독 게임당 샷 | 11.79 | 10.41 | **7.61** | 7.89 / 7.39 |
| 샷당 성공률 | 24.8% | 28.3% | **38.5%** | 38.7% / 35.7% |
| 학습 시간 | 196s | 674s | 1451s | |

- **GNN이 처음으로 모방학습 정책을 SAC 수준까지 올렸다**: 정책 단독 7.61샷(MLP 11.79), 샷당 성공률 38.5%(SAC 38.7%). SAC는 세게 쳐서
  우연에 기대는 정책이고, GNN은 교사처럼 부드럽게 조준하는 정책이라 같은 수치라도 내용이 다르다.
- 조준 정밀도(0.2° 이내)는 32 → 37.5%로 개선 폭이 크지 않다. 게임 성적 개선(11.8 → 7.6샷)이 훨씬 큰 이유는 **쌍마다 각도와 속도를 함께 내고
  그중 하나를 고르는 구조** 때문으로 보인다 — MLP는 "다른 정답의 각도 + 또 다른 정답의 속도"처럼 섞인 행동을 낼 수 있지만 GNN은 한 쌍의
  일관된 행동을 낸다. 첫 epoch val loss도 GNN 1081 vs MLP 3066으로 출발부터 낮았다(손실 형태가 달라 값 자체는 직접 비교 불가).
- MLP 실행 간 0.2° 이내 비율은 이전 곡선 실험(같은 조건 34.8%)과 ±3%p 차이가 난다 — 시드/스냅샷 노이즈 수준. GNN의 +5%p는 이 노이즈와
  비슷한 크기라 정밀도 개선 자체는 아직 확정하기 어렵다. 게임 성적 차이(4샷)는 노이즈보다 훨씬 크다.
- 여전히 플래너(2.24샷)와는 멀다. 병목은 정밀도(0.2° 이내 37.5%) — 다음은 GNN + 증강, GNN + 50만 샷.
- GNN + 증강 ×4는 처음에 CPU 경합(데이터 생성 7개 + 학습 4개)으로 epoch당 수십 초라 중단했다
  (`..._gnn_relative_flip4_..._ABORTED_slow`, 분석 제외). CPU가 비는 대로 다시 돌린다.

**GNN의 빗나간 샷은 정답에서 얼마나 떨어져 있나 — 실제 포켓 창 측정** ("37.5%만 0.2° 안이면 왜 7.61샷인가, 각도가 그렇게 결정적이면
GNN은 정답과 얼마나 다른가"라는 질문에서 출발). GNN(5만 샷) 정책 단독으로 250샷을 치며, 매 샷 **같은 속도로 각도만 ±2°를 0.02° 간격으로
pooltool에서 쳐봐서** 실제로 들어가는 각도 구간(창)을 찾았다.
- 이번 표본의 샷당 성공률 34.0%, 속도 중앙값 4.94m/s.
- **들어가는 창의 폭: 중앙값 0.04°**(p25 0.02° = 격자 한 칸, p75 0.22°). 지금까지 기준으로 쓴 "0.2°"보다 훨씬 좁다 — 0.2° 이내 비율도
  관대한 지표였다.
- **빗나간 샷의 58%는 0.1° 이내, 80%는 0.2° 이내에서 빗나갔다**(중앙값 0.08°). 0.5° 넘게 벗어난 것은 10%뿐이고, 그 속도로 ±2° 안에
  들어가는 각도가 아예 없는 경우는 1.2%. GNN은 맞는 곳을 겨누고 있지만 0.1° 안팎이 모자라 빗나간다.
- 그래서 7.61샷은 "각도가 결정적"이라는 것과 모순이 아니라 그 결과다: 창이 넓은 쉬운 샷(가깝고 곧은 샷)만 들어가 샷당 ~0.38개 →
  3 ÷ 0.38 ≈ 7.6~7.9샷.

**결과 1b — GNN + 증강, GNN + 데이터 8배** (같은 설정, 상대 좌표, `--select within`):

| | 데이터 | epoch | best epoch | 0.2° 이내 | 각도 오차 중앙값 | 정책 단독 샷 | 샷당 성공률 |
|---|---|---|---|---|---|---|---|
| GNN | 5만 | 500 | 364 | 37.5% | 0.419° | 7.61 | 38.5% |
| **GNN + 증강 ×4** | 5만 (×4 = 17.7만) | 500 | 32 | **39.0%** | **0.377°** | **7.43** | **39.4%** |
| GNN | 40만 (`big_snap_1421`) | 100 | 78 | 35.2% | 0.443° | 8.08 | 36.0% |

- GNN + 증강이 지금까지 가장 좋은 모방학습 정책(7.43샷)으로, SAC(7.89)·SAC 엔트로피 낮춤(7.39)과 같은 수준.
- **데이터를 8배로 늘려도 GNN도 나아지지 않았다**(MLP와 같음). 같은 라벨로는 0.2° 이내 ~35~39%에서 막힌다. 라벨이 "들어가기만 하면 아무거나"이고
  각도 격자도 0.2° 간격이라, 이 라벨 잡음이 도달 가능한 정밀도의 바닥을 정하고 있을 가능성이 크다 → 아래 "여유 있는 교사" 재라벨이 이를 직접 검증한다.

**결과 2 — 데이터 8배(40만 샷)에서 MLP** (스냅샷 `logs/teacher_data/big_snap_1421`, 39.9만 샷, 100 epoch, 상대 좌표, 증강 없음):
best epoch 71, 가장 가까운 정답 각도 오차 0.61°, 0.2° 이내 29.4%, 정책 단독 9.18샷(샷당 31.5%) — 5만 샷(32~35%, 9.5~11.8샷)과 같은 수준.
학습 곡선은 train 1163 / val 1386(epoch 80)으로 **과적합이 사라졌다**(5만 샷에선 train↓ val↑). 데이터가 과적합은 없앴지만 정밀도는 올리지
못했다 → 이제는 모델 용량/표현이 한계인 영역이다(100 epoch로 아직 val이 내려가는 중이라 학습량도 부족할 수 있음).

**함의**: 필요한 정밀도는 창 폭 ~0.04°, 행동 범위의 ~1/10,000이다. 좌표에서 이 정밀도의 함수를 처음 보는 배치에 일반화하는 건 데이터·용량·표현이
모두 필요해 어렵다. 반면 GNN은 빗나간 샷의 80%가 0.2° 안이므로, **결정 시점에 시뮬레이터를 쓸 수 있다면** GNN이 겨눈 곳 주변만 짧게 쳐보는
국소 탐색(±0.2°, 수십 번)으로 대부분 메울 수 있다 — 정책이 후보를 좁히고 시뮬레이터가 마지막 0.1°를 잡는 "탐색 + 학습" 구조.

**각도 × 속도 창 — 각도가 빗나가도 속도로 만회되나?** (사용자 질문에서 출발) GNN(5만 샷)이 실제 게임에서 만난 상태 60개에서 각도 ±0.5°
(0.02° 간격) × 속도 0.5~8m/s(0.25 간격) 격자를 전부 pooltool로 쳐봤다. 이번 표본의 정책 성공 37%, 정책 속도 중앙값 4.95m/s.

| | 값 |
|---|---|
| 빗나간 샷 중 **같은 각도로 다른 속도면 들어가는** 비율 | **66%** |
| 가장 가까운 들어가는 각도까지 (빗나간 샷, 중앙값) — 정책 속도 / 아무 속도 | 0.08° / **0.00°** |
| 가장 가까운 창의 폭 (중앙값) — 정책 속도 / 가장 넓은 속도 | 0.08° / **0.24°** (p75 0.56°) |
| 창이 가장 넓은 속도 (중앙값, p25–p75) | **2.50m/s** (0.75–4.75) |
| ±0.5° × 전체 속도 격자 중 들어가는 비율 (중앙값) | 27.1% |
| 속도에 따른 창 중심 이동(throw) | 0.010°/(m/s) — 거의 없음 |

- **속도로 만회된다 — 다만 창을 옮기는 게 아니라 넓히는 방식이다.** 속도를 바꿔도 창 중심은 거의 안 움직이지만(0.01°/(m/s)), 창의 폭은
  속도에 따라 크게 달라진다. 정책은 ~5m/s로 쳐서 창이 0.08°인데, 창이 가장 넓은 속도(중앙값 2.5m/s)에서는 0.24°로 3배다. 느린 공이
  포켓 입구에서 덜 튕겨 나오는 효과로 보인다.
- 원인은 교사 라벨: `nearest_good`은 들어가는 후보 중 각도가 가장 가까운 것을 따라 하므로, 같은 각도에서 여러 속도가 들어가면 속도는 사실상
  아무거나다. 교사(기하 플래너)도 "들어가기만 하면" 고르고 얼마나 여유 있게 들어가는지는 보지 않는다.
- 함의: **교사가 창이 가장 넓은(여유가 가장 큰) 각도·속도를 고르게** 하면(시뮬레이션만으로 가능, 당구 지식 불필요), 라벨이 더 깔끔해지고
  정책이 필요로 하는 정밀도도 ~0.04~0.08° → ~0.24°로 3배 완화된다. GNN의 빗나간 샷은 80%가 0.2° 이내였으므로 상당수가 들어갈 수 있다.

### 여유 있는 교사(robust teacher): 창이 가장 넓은 각도·속도를 라벨로 (2026-10-09 계획)

**왜**: 각도 × 속도 창 분석에서 GNN이 빗나간 샷의 66%는 같은 각도로 다른 속도였으면 들어갔다. 창은 느린 속도(중앙값 2.5m/s)에서 3배 넓다
(0.08° → 0.24°). 기존 교사(기하 플래너)는 "들어가기만 하면" 골라서 속도가 사실상 아무거나였고, GNN은 ~5m/s로 쳐서 좁은 창을 맞혀야 했다.
교사가 **가장 여유 있게 들어가는** 샷을 고르면 라벨이 깔끔해지고 정책이 필요로 하는 정밀도가 완화된다. 시뮬레이션만 쓰고 당구 지식은 쓰지 않는다.

**방법 (샷마다)**:
1. 기존 기하 플래너 후보(ghost-ball 각도 ±0.4° × 속도 1.5/3/5)를 쳐서 들어가는 (공, 포켓) 조합을 찾는다.
2. 들어간 후보가 많은 상위 2개 조합에 대해 ghost-ball 각도 주변 ±0.5°를 0.04° 간격 × 속도 {1, 2, 3, 4, 5}m/s로 촘촘히 쳐본다(조합당 130번).
3. 속도마다 연속으로 들어가는 각도 구간(창) 중 가장 넓은 것의 폭과 가운데를 구한다.
4. **창이 가장 넓은 (조합, 속도)를 고르고 각도는 그 창의 가운데** — 이것을 실제로 치고 라벨로 저장한다. 다른 조합의 최선 샷은 "다른 정답 후보"로
   함께 저장(`good_actions`). 찾지 못하면 기존 플래너 선택으로 대체.
- 한 샷 다중 포켓은 고려하지 않는다(창 기준은 1개 이상 포켓) — 교사 자체의 게임당 샷 수는 2.24보다 늘 수 있다. 같이 기록한다.

**비교 — 새로 만들지 않고 기존 데이터에 라벨만 다시 붙인다** (사용자 제안): 샷 직전에는 모든 공이 멈춰 있어 저장된 obs만으로 테이블이 정해진다.
obs에서 테이블을 복원해(포켓된 공은 pooltool의 pocketed 상태로 테이블 아래) 원래 상태와 비교한 결과 **340/340번 같은 결과**(85개 상태, 그중 55개는
포켓된 공 포함). 그래서 기존 기하 교사 데이터(`snap_1014/ghost_s1000.npz`, 16,748샷)의 **같은 상태들**에 여유 있는 교사 라벨을 붙이고, 같은 행·같은
게임 id로 기존 라벨 버전도 함께 저장한다 → train/val 분할까지 동일하고 **라벨만 다른 1:1 비교**. 게임 단위 8조각 병렬. GNN 동일 설정(상대 좌표,
`--select within`, 500 epoch). 지표: 정책 단독 게임당 샷·샷당 성공률(주 지표), 정책 속도 분포, 라벨의 창 폭·속도.

**코드**: `exp16_wm/teacher_data.py`(`robust_shot`, `--teacher robust`), `exp16_wm/relabel.py`(obs → 테이블 복원 + 재라벨),
`exp16_wm/planner.py`(조합별 ghost-ball 각도 헬퍼), 테스트 `tests/test_exp16_wm_relabel.py` 등.

**결과 (2026-10-09, 같은 1.67만 상태에서 라벨만 다름, GNN, 상대 좌표, `--select within`, 500 epoch, 정책 단독 500게임 seed 0)**:

| 라벨 | 증강 | train 샘플 | 정책 단독 게임당 샷 | 샷당 성공률 | clear |
|---|---|---|---|---|---|
| **여유 있는 교사** | 없음 | 15,051 | **6.18** | **48.1%** | 99.8% |
| **여유 있는 교사** | ×4 | 60,204 | 6.53 | 45.7% | 99.8% |
| 기존 (후보 8개 저장) | 없음 | 15,051 | 12.27 | 23.8% | 98.6% |
| 기존 (후보 8개 저장) | ×4 | 60,204 | 11.37 | 25.7% | 98.4% |
| (참고) 기존 라벨, 다른 상태 4.9만 샷 | 없음 / ×4 | 44,200 / 176,800 | 7.61 / 7.43 | 38.5% / 39.4% | |
| (참고) vanilla SAC / SAC 엔트로피 낮춤 | | | 7.89 / 7.39 | 38.7% / 35.7% | |
| (참고) 여유 있는 교사 자체 (40게임, seed 77) | | | 2.98 | 100% | |
| (참고) 기하 교사 / 무작위 탐색 200후보 | | | 2.24 / 2.50 | 99.4% / 100% | |

- **라벨만 바꿨는데 정책 단독 성능이 크게 좋아졌다**: 6.18샷, 샷당 성공률 48% — 처음으로 SAC(7.89 / 7.39샷)와 이전 최고 모방학습(7.43샷)을 모두
  넘었다. 그것도 3배 적은 학습 데이터(15k 샘플 vs 44~177k)로.
- 각도 오차(가장 가까운 정답까지, 라벨마다 기준 집합이 달라 직접 비교는 불가): 여유 있는 교사 0.46° / 0.43°, 기존 1.26° / 1.19°.
- 증강은 이번에는 도움이 되지 않았다(6.18 → 6.53, 12.27 → 11.37로 엇갈림). 데이터가 3배 적어도 잡음 수준 차이 — 더 큰 데이터에서 다시 본다.
- 교사 자체는 2.98샷이 걸려 기하 교사(2.24)보다 늦다 — 한 번에 2개 넣는 샷 대신 가장 여유 있는 1개짜리 샷(속도 1m/s)을 고르기 때문. 정책(6.18)과
  교사(2.98) 사이 격차는 아직 크다.
- 한계: 시드 1개·학습 1회. 기존 라벨 쪽은 후보를 8개로 줄여 저장했고(이전 GNN 실험은 32개) 이게 불리할 수 있어, 32개 그대로 쓴 대조군을 추가로 돌린다
  (`relabel_s1000/orig32/`, 결과는 아래 이어서). 속도·창 폭이 스캔 격자 끝(1m/s, ±0.5°)에서 잘린 점도 남아 있다.

**대조군 결과 — 기존 라벨을 후보 32개 그대로 저장** (`relabel_s1000/orig32/`, 같은 상태·같은 분할):

| 라벨 | 증강 | 정책 단독 게임당 샷 | 샷당 성공률 |
|---|---|---|---|
| 기존 (후보 32개) | 없음 | 9.48 | 30.8% |
| 기존 (후보 32개) | ×4 | 8.73 | 33.4% |
| 기존 (후보 8개) | 없음 / ×4 | 12.27 / 11.37 | 23.8% / 25.7% |
| **여유 있는 교사** | 없음 / ×4 | **6.18 / 6.53** | **48.1% / 45.7%** |

- 후보를 8개로 줄이면 실제로 불리했다(9.48 → 12.27샷). 공정한 비교는 32개 대조군이며, 그래도 **여유 있는 교사가 크게 앞선다**: 증강 없음 9.48 → 6.18샷
  (−35%), 증강 ×4 8.73 → 6.53샷(−25%). 샷당 성공률 31~33% → 46~48%.
- 기존 라벨은 증강이 도움(9.48 → 8.73)이고 새 라벨은 도움이 안 됐다(6.18 → 6.53): 새 라벨이 더 깔끔해 데이터가 적어도 충분한 것일 수 있다. 시드 1개라 확정은 아님.
- 새 라벨로 학습한 정책의 실제 행동(150샷): **속도 중앙값 1.31m/s**(p10–p90 1.17–1.57; 이전 GNN은 4.9m/s), **자기 속도에서 들어가는 각도 창의 폭 중앙값 0.61°**
  (이전 0.04°의 15배), 샷당 성공률 51.3%. 빗나간 73샷은 가장 가까운 들어가는 각도까지 중앙값 0.20°, 그러나 25%(18샷)는 ±1° 안에 들어가는 각도가 아예 없다
  (원인은 아래 "±1° 안에 들어가는 각도가 없는 빗나감의 원인" 참고 — 처음에 "엉뚱한 목표"로 짐작했으나 측정 결과 속도 문제였다).
- 남은 격차: 정책 6.18샷 vs 교사 2.98샷. 원인 후보 — (a) 빗나간 샷의 1/4은 각도 창이 없는 속도를 골랐음(아래 측정), (b) 나머지는 0.2° 안팎의 정밀도, (c) 교사가 한 샷 다중
  포켓을 포기해 하한이 3샷 근처.

**±1° 안에 들어가는 각도가 없는 빗나감의 원인** (여유 있는 교사로 학습한 GNN, 150샷, 사용자 질문에서 출발): 빗나간 샷 중 정책 자신의 속도에서 각도를 ±1°
(0.04° 간격)로 바꿔도 아무것도 안 들어가는 샷이 18개. 이 18샷의 같은 조준(±1°)에서 속도만 0.5~8m/s로 바꿔 보면:
- **17/18(94%)은 다른 속도에서는 들어간다** — 목표를 잘못 고른 게 아니라 속도가 안 맞은 것. 정책 속도는 이 샷들에서 중앙값 1.48m/s.
- 1/18만 어떤 속도로도 안 들어간다.
- 다만 이 샷들의 조준은 직선 ghost-ball 조준에서 중앙값 8.4° 떨어져 있어, 들어가는 경로가 직선 조준이 아니라 다른 공·쿠션을 거치는 간접 경로로 보인다
  (이런 샷은 속도에 민감). 표본이 18샷이라 비율은 대략적이다.

### 들어가는 조합 전체를 쓰기: 성공 지도(success map) 데이터 수집 (2026-10-09 계획)

**질문**: "창 안에서 얻은 값 중 왜 하나만 뽑나? 들어가는 조합(각도 × 속도 × 공·포켓) 전체를 쓰면 안 되나?"

**하나만 고르는 이유**: 정책은 샷 하나만 친다. 들어가는 조합 여러 개를 모두 정답으로 회귀시키면 평균을 내서 정답들 사이(대개 빗나감)를 친다
(교사 행동 그대로 따라 하기가 19샷으로 망가진 이유). 창 안에서는 가운데가 오차에 가장 강하다. 지금의 `nearest_good`은 이미 "집합 중 가장
가까운 것"을 쓰지만, 가장 가까운 점은 보통 **창의 가장자리**라 조금만 틀려도 빗나가는 위치를 배운다.

**그래도 버리는 건 낭비다 — 전체를 쓰는 방법**:
1. 창 자체를 배우기: (조합, 속도)마다 창의 가운데·폭을 예측.
2. **성공 지도 학습**: (상태, 각도, 속도) → 들어가나(공 수, 스크래치)를 맞히는 분류기. 들어간 샷뿐 아니라 **빗나간 샷도 라벨**이 된다. 칠 때는 후보
   수천 개를 신경망으로 평가해 "들어갈 확률이 높은 영역의 가운데"를 고른다 — 시뮬레이터 없이. SAC critic이 RL 신호로 배우려다 실패한
   (pocket AUC 0.66) "이 샷이 들어가나"를 시뮬레이터 라벨로 직접 가르치는 것이고, (상태, 행동) → 결과를 배우는 작은 world model이기도 하다.

**수집**: 재라벨 중 여유 있는 교사가 상태마다 하는 ~300번의 시뮬레이션(기하 후보 ~58 + 상위 2개 조합 × 5속도 × 26각도)을 전부 저장한다
(`relabel_s1000/sims/*.npz`: `sim_state`, `sim_action`, `sim_pocketed`, `sim_scratch`). 1만 6748 상태 × ~300 ≈ 500만 라벨. 저장된 샘플을 복원한
테이블에서 다시 쳐도 같은 결과가 나오는 것을 테스트로 확인했다. 재라벨을 처음부터 다시 시작했다(첫 실행은 저장 전에 중단, 출력 삭제).
성공 지도 학습 자체는 재라벨 비교 결과를 본 뒤 설계한다.

**재라벨 결과 (2026-10-09 16:23~17:07, 8조각 병렬, 약 45분)**: 기존 기하 교사 데이터 `snap_1014/ghost_s1000.npz`의 16,748 상태 전부에 여유 있는 교사를 적용.
- 시뮬레이션 기록(`relabel_s1000/sims/`): **약 500만 건**(조각당 62만, 상태당 ~298건), 깨끗이 들어간 샷 48.3%. 후보가 ghost-ball 조준 주변에 몰려 있어서
  행동 공간 전체에서 무작위로 뽑은 표본이 아니다(성공 지도를 학습할 때 분포 주의 — 무작위 행동 표본을 따로 섞어야 할 수 있음).
- 라벨 변화(조각별 중앙값): 속도 **3.00 → 1.00m/s**, 창 폭 **0.76~0.80°**. 이 값은 참고용이다: 스캔 각도 범위가 ±0.5°라 폭은 최대 1.04°에서 잘리고,
  최적 속도 중앙값이 스캔 속도 격자의 가장 느린 값(1.0m/s)이라 격자 밖(더 느린 속도)이 더 좋을 수 있다. 느린 샷은 다른 공을 흩지 못해 한 샷 다중
  포켓을 포기한다는 대가도 있다.
- 첫 학습 시도는 내 실수로 실패했다(zsh에서 `export -f`가 안 돼 `run: command not found`; 학습 단계만 실패, 재라벨 데이터는 정상). bash 스크립트
  `logs/runs/bc_relabel.sh`로 재시작.
- 비교 시 주의: 기존 라벨 버전(`orig/`)은 `good_actions`를 후보 8개로 줄여 저장했고 새 라벨은 쌍당 최선 1개씩(최대 2개)이라, **`within0.2` 지표는 라벨마다
  기준 집합이 달라 직접 비교할 수 없다.** 주 지표는 정책 단독 게임당 샷·샷당 성공률.

### 스캔 범위 확장: 각도 + 속도 여유(2D margin)를 함께 보는 교사 (2026-10-09 계획)

**왜**:
- 여유 있는 교사(v1)는 속도 5개(1~5m/s) 각각에서 **각도 창 폭**만 재서 가장 넓은 것을 골랐다. 속도 쪽 여유는 보지 않았고, 라벨 속도 중앙값이 스캔 격자의 최저값
  (1.00m/s)이며 창 폭은 ±0.5° 스캔 범위에서 잘렸다 — 격자 가장자리에 걸린 선택이다.
- 정책의 ±1° 창이 없는 빗나감 18샷 중 17샷(94%)은 같은 조준에서 속도만 바꾸면 들어갔다 → 속도 오차가 실제 실패 원인이다.

**방법** (샷마다; 시뮬레이션만 쓰고 당구 지식 없음):
1. 거친 단계는 v1과 같다(ghost-ball 후보 ±0.4° × 속도 {0.75, 1.5, 3, 5} — 느린 속도에서만 들어가는 쌍을 놓치지 않도록 0.75 추가)로 쌍을 고르고 상위 2쌍.
2. 쌍마다 **각도 ±2.0°(0.1° 간격 41개) × 속도 0.5~8m/s(로그 간격 15개, 인접 속도 비 1.22배) 전체 격자**를 쳐본다(쌍당 615번).
3. 들어간 칸(공 ≥1개, 스크래치 없음)마다 **가장 가까운 안 들어간 칸까지의 거리**(여유)를 구한다. 단위는 각도 0.2° = 1, 속도 20% = 1의 유클리드 거리.
   격자 가장자리는 벽이 아니라 바깥도 같은 값이라고 본다(각도 범위는 인공적인 한계; 속도는 행동 한계 0.5/8m/s에 걸리면 어차피 같은 값으로 잘림).
4. **여유가 가장 큰 칸이 라벨**. 여유가 같은 칸이 여럿이면 그 집합의 중심에 가장 가까운 칸. 두 쌍 중 여유가 큰 쪽을 치고, 다른 쌍의 최선은 `good_actions`에 넣는다.
5. 상태마다 조건을 만족하는 칸이 없으면 기존 플래너 선택으로 대체(v1과 동일).

**비용**: 상태당 시뮬레이션 ~1,290번(v1은 ~300) → 코어당 ~2.6초. 1.67만 상태 × 8코어 약 1.5시간. 먼저 30개 상태로 시간과 라벨 분포를 확인한다.
메모리: 시뮬레이션 기록이 21M건이라 상태마다 numpy로 바로 압축해 쌓는다.

**비교**: 같은 1.67만 상태, GNN 동일 설정(상대 좌표, `--select within`, 500 epoch), 증강 없음/×4. 주 지표: 정책 단독 게임당 샷·샷당 성공률 (v1 6.18 / 6.53, 기존 라벨
후보 32개 9.48 / 8.73), 정책 속도 분포, ±1° 창이 없는 빗나감 비율, 라벨의 여유 분포.

**판단 기준**: v1보다 정책 샷이 줄고(예: 5.5 미만) ±1° 창이 없는 빗나감이 줄면 → 속도 여유를 라벨에 넣은 효과. 그대로면 → 정책이 속도를 정밀하게 못 내는 표현 문제.

**코드**: `exp16_wm/teacher_data.py`(`best_margin_cell`, `margin_shot`, `TEACHERS["margin"]`), `exp16_wm/relabel.py`(`--teacher`), 테스트.

**결과**: TBD

### 정책 출력 vs 교사 라벨의 격차: 각도·속도 따로, 그리고 시뮬레이터에서 (2026-10-09)

**질문**: 학습된 정책의 행동(예측)이 교사의 정답 라벨과 실제로 얼마나 다른가? 기하 교사와 여유 있는 교사를 각도·속도 모두 비교.

**방법** (`exp16_wm/label_gap.py`, 테스트 `tests/test_exp16_wm_label_gap.py`): 학습이 끝난 GNN(증강 없음, seed 0)을 다시 불러와, 학습에 쓰지 않은 같은 검증
상태 1,697개에서 정책의 (각도, 속도)를 라벨과 비교한다. 두 모델은 같은 상태·같은 분할에서 라벨만 다르다(기하 교사 후보 32개 / 여유 있는 교사). 추가로 정책의
행동을 복원한 테이블에서 실제로 쳐서 "깨끗이 들어가나"를 보고, 각도와 속도를 하나씩 교사 값으로 바꾼 혼합 행동도 쳐서 어느 쪽 오차가 실패를 만드는지 분해한다.

| | 기하 교사 라벨 | 여유 있는 교사 라벨 |
|---|---|---|
| **각도 오차 vs 교사가 친 샷** — 중앙값 / 0.2° 이내 / 1° 이내 | 5.76° / 3.7% / 15.6% | **0.80° / 21.9% / 53.7%** |
| 각도 오차 vs 가장 가까운 정답 후보 (후보 수 32개 / ≤2개라 직접 비교 불가) | 0.99° / 18.8% / 50.1% | 0.46° / 28.2% / 68.5% |
| **속도 오차 vs 교사가 친 샷** — 중앙값 (상대) | 1.91m/s (64%) | **0.34m/s (31%)** |
| 속도 20% 이내 | 40.2% | 17.4% |
| 속도 중앙값: 라벨 → 정책 | 3.00 → 4.92 | 1.00 → 1.30 |
| **시뮬레이터에서 깨끗이 들어간 비율**: 교사 | 96.5% | 98.9% |
| 정책 | 25.9% | **52.6%** |
| 라벨 각도 + 정책 속도 (속도 오차만 남김) | 68.8% | **91.1%** |
| 정책 각도 + 라벨 속도 (각도 오차만 남김) | 22.9% | 55.0% |

**해석**:
- **병목은 각도다.** 각도를 교사 값으로 바꾸면 들어가는 비율이 25.9 → 68.8%(기하), 52.6 → 91.1%(여유)로 크게 오르지만, 속도만 바꾸면 거의 그대로다
  (22.9% / 55.0%). 정책 속도 오차의 비용은 기하 라벨에서 96.5 → 68.8%(−28%p), 여유 있는 교사에서 98.9 → 91.1%(−8%p)로, **여유 있는 라벨이 속도 오차를 3배
  이상 싸게 만든다** — 속도 여유를 라벨에 넣은 의도가 그대로 확인된다.
- 속도는 두 정책 모두 라벨보다 빠른 쪽으로 편향돼 있다(3.00 → 4.92, 1.00 → 1.30m/s). 여유 있는 라벨 쪽 상대 오차도 중앙값 31%로 작지 않다.
- 각도 오차 p90이 큰 이유(기하 59.8°, 여유 42.9°)는 정책이 교사와 **다른 공·포켓을 겨누기 때문**이다. 그 목표도 들어가면 틀린 게 아니다 — 가장 가까운 정답
  후보 기준 p90은 6.0° / 9.2°. 정책 행동이 실제로 들어가는 비율(25.9% / 52.6%)이 "교사 라벨과 다르다 = 틀렸다"가 아님을 보여 준다.
- 한계: 모델당 1개(seed 0), 증강 없음. "가장 가까운 정답 후보" 비교는 후보 집합 크기가 달라(32개 vs 최대 2개) 라벨 종류 간에는 직접 비교하지 않는다.

### 모방학습 데이터·모델 총정리 (2026-10-09 기준)

모든 모델은 **GNN/MLP 모두 학습 1회, seed 0**, 정책 단독 평가 = 500게임(seed 0, clear-speed 과제: 3개를 다 넣을 때까지, 샷당 −1). 기준점: SAC 7.89샷 / SAC 엔트로피 낮춤 7.39샷,
교사 자체 — 기하 2.24샷 · 무작위 탐색(200후보) 2.50샷 · 여유 있는 교사 v1 2.98샷.

**① 데이터** (`logs/teacher_data/`)

| 이름 | 크기 | 라벨(교사) | `good_actions` | 용도 |
|---|---|---|---|---|
| `ghost_s1000~1002` | 5.03만 샷 / 2.25만 게임 | 기하 교사(ghost-ball 후보 + 시뮬레이션 선택) | 최대 32 | 초기 BC 전부. 생성 중 복사한 스냅샷으로 학습: `snap_0939` 3.0만, `snap_0951` 4.35만, `snap_1014` 4.91만 샷 |
| `big/ghost_s2000~2006` | 49.9만 샷 / 22.4만 게임 | 기하 교사 | 최대 32 | 8배 데이터 실험 (`big_snap_1421` = 그 중 39.9만 샷) |
| `relabel_s1000/orig32` | 1.67만 상태 | 기하 교사 (`snap_1014/ghost_s1000`의 같은 상태) | 32 | 라벨 대조군 |
| `relabel_s1000/orig` | 1.67만 상태 | 같음 | 8로 잘림 | 대조군(불리했음) |
| `relabel_s1000/robust` | 1.67만 상태 | 여유 있는 교사 v1: 각도 ±0.5° × 속도 5개에서 가장 넓은 각도 창의 가운데 | 최대 2 | v1 학습 |
| `relabel_s1000/sims` | 499.7만 건 | v1이 쳐본 모든 시뮬레이션 (상태, 각도, 속도 → 들어간 공 수, 스크래치) | – | 성공 지도용 (아직 미사용) |
| `relabel_s1000/margin`, `sims_margin` | 1.67만 상태 / 약 2,100만 건 | 여유 있는 교사 v2: 각도 ±2° × 속도 0.5~8m/s 격자의 2D 여유 최대 칸 | 최대 2 | **생성 중** (20:30~20:50 완료 예정) |

**② 모델·실험** (정책 단독 게임당 샷 / 샷당 성공률 / 검증 상태에서 정답 0.2° 이내 비율 — 마지막 열은 라벨마다 기준 집합이 달라 **같은 계열 안에서만** 비교)

| 비교한 것 | 모델 | 입력 | 증강 | 학습 샘플 | 게임당 샷 | 성공률 | 0.2° 이내 |
|---|---|---|---|---|---|---|---|
| 손실 | MLP | raw | – | 9,031 | 19.17 (L2) / **9.31** (nearest_good) | 13.5 / 31.4% | 9.2 / 12.1% |
| 모델 크기 | MLP 256×2 / 1024×3 | raw | – | 2.0만 | 10.44 / 11.15 | 27.9 / 26.0% | 12.0 / 13.5% |
| 입력 표현 | MLP | raw / 상대 좌표 | – | 2.7만 | 8.95 / 11.28 | 32.7 / 25.6% | 13.1 / **18.1%** |
| 데이터 양 | MLP | 상대 | – | 9.8k / 19.6k / 39.2k | 10.72 / 11.39 / 9.49 | 27.0 / 25.5 / 30.6% | 20.8 / 22.7 / 34.8% |
| 데이터 양 + 증강 | MLP | 상대 | ×4 | 39.2k / 78.5k / 156.9k | 9.80 / 10.13 / 8.92 | 29.4 / 28.8 / 32.7% | 31.3 / 34.4 / 30.3% |
| 구조 | MLP / **GNN** | 상대 | – | 44,200 | 11.79 / **7.61** | 24.8 / 38.5% | 32.1 / 37.5% |
| 구조 + 증강 | MLP / **GNN** | 상대 | ×4 | 176,800 | 10.41 / **7.43** | 28.3 / 39.4% | 33.2 / 39.0% |
| 데이터 8배 (100 epoch) | MLP / GNN | 상대 | – | 359,167 | 9.18 / 8.08 | 31.5 / 36.0% | 29.4 / 35.2% |
| 라벨: 기하(후보 8개로 잘림) | GNN | 상대 | – / ×4 | 15,051 / 60,204 | 12.27 / 11.37 | 23.8 / 25.7% | 20.9 / 21.6% |
| 라벨: 기하(후보 32개) | GNN | 상대 | – / ×4 | 15,051 / 60,204 | 9.48 / 8.73 | 30.8 / 33.4% | 22.6 / 33.3% |
| 라벨: **여유 있는 교사 v1** | GNN | 상대 | – / ×4 | 15,051 / 60,204 | **6.18** / 6.53 | **48.1** / 45.7% | 28.3 / 29.1% |
| 라벨: 여유 있는 교사 v2 (2D 여유) | GNN | 상대 | – / ×4 | 15,051 / 60,204 | 학습 대기 | | |

- 선택 기준: 위 모든 행은 검증 상태의 "0.2° 이내 비율" 최고 시점을 best로 골랐다(`--select within`). 손실·크기·입력 비교와 첫 데이터 곡선(`val loss` 기준)은 `val loss` 최저로 골랐던 옛 실행이라 데이터 양 행만 `within`으로 다시 돌린 값이다.
- 증강 ×4 행의 학습 샘플은 뒤집기 4배가 포함된 수.
- 학습은 모두 seed 0 한 번이라, 차이가 1~2샷 이하인 것은 노이즈 수준일 수 있다.

### 학습 데이터에서의 격차: 과적합인가? (2026-10-09)

**질문**: 학습한 상태에서는 정책 행동과 라벨이 얼마나 다른가? (`label_gap.py --split train`, 학습 상태 15,051개 중 1,697개 무작위 표본; 두 GNN은 증강 없음, 검증 기준으로 고른 best epoch 체크포인트)

| | 기하 교사 train / val | 여유 있는 교사 train / val |
|---|---|---|
| 각도 오차 중앙값 (교사가 친 샷 대비) | 5.23° / 5.76° | 0.77° / 0.80° |
| 각도 0.2° 이내 | 3.9% / 3.7% | 21.9% / 21.9% |
| 속도 오차 중앙값 | 1.92 / 1.91m/s | 0.33 / 0.34m/s |
| 시뮬레이터 깨끗이 들어간 비율: 정책 | 26.4% / 25.9% | 51.3% / 52.6% |
| 교사 | 96.6% / 96.5% | 98.7% / 98.9% |
| 라벨 각도 + 정책 속도 | 66.8% / 68.8% | 91.4% / 91.1% |
| 정책 각도 + 라벨 속도 | 22.5% / 22.9% | 54.2% / 55.0% |

- **학습 상태에서도 검증 상태만큼 틀린다**(차이 ≤ 1~2%p). 일반화 실패(과적합)가 아니라, 모델이 **학습한 상태조차 라벨대로 재현하지 못한다**는 뜻이다.
  (옛 raw MLP에서 본 "train loss↓ val loss↑"와는 다른 상황.)
- 손실 곡선도 같은 이야기다: 여유 있는 교사 train/val 손실 398 / 694(epoch 299), 기하 교사 200 / 262(epoch 485)로 격차가 크지 않고, 둘 다 학습 내내 천천히 내려간다.
- **검증 지표가 epoch마다 크게 흔들린다**: 기하 교사 val 0.2° 이내가 epoch 485에서 22.6%, epoch 499에서 12.8%; 여유 있는 교사 epoch 299 28.3% → epoch 499 20.4%
  (lr 1e-3 고정, 감쇠 없음). 따라서 "검증 0.2° 이내가 최고인 epoch"를 고르는 방식은 **잡음 속 최댓값을 뽑는 것이라 보고된 0.2° 이내 비율은 낙관적**이다
  (게임 성적은 별도의 500게임 평가라 영향이 없다).
- 가설(미검증): (a) 최적화 잡음이 정밀도(~0.1°)를 가로막는다 — 학습률 감쇠/가중치 이동평균/더 큰 배치로 확인 가능, (b) 정답이 여러 개인 상태에서 교사와 다른 목표를
  고르는 것(각도 오차 p90 40~56°)은 허용되지만, 고른 목표 안에서의 정밀도가 부족하다.

### 학습이 잘못됐는가? — overfit test, lr decay, pair supervision (2026-10-09)

**Trigger**: train set에서도 policy action이 label과 크게 달랐다(train ≈ val). 지도학습에서 train set도 못 맞추면 학습 자체를 의심해야 한다.

**1. Overfit test** (robust label 512 states, GNN, 3000 epochs, 같은 states에서 평가): angle error median 2.62°, <0.2° 11%, simulator pocket 33.8% (teacher 98.4%). 512개도 못 외운다.
- 분해(lr을 1e-3→3e-5로 단계적으로 낮춤): winner pair의 regression error는 0.27°→0.06°로 내려가 97%가 0.2° 이내 — regression 자체는 된다.
  반면 **selection(어느 pair를 고르나) accuracy는 22%→48%**, CE 2.3→1.6 (ln 18 = 2.9)에 머문다.
- 원인: winner-takes-all(WTA)은 "현재 prediction이 label에 가장 가까운 pair"를 매 step 다시 고른다 → selection head의 target이 계속 바뀐다(moving target).

**2. Fix**: label을 만든 (ball, pocket) pair를 states에서 복원해(`exp16_wm/pair_labels.py`, simulation 없이 ghost-ball aim과 label angle 매칭; robust 95.8%, geom 99.4% 할당) **pair supervision**: 각 label은 자기 pair의 output으로 regress, selection head는 그 pair를 고르는 CE.
같은 512 states, cosine lr: WTA → pair supervision = angle error median 1.56° → 0.10°, <0.2° 27.7% → 64.5%, simulator pocket 45.9% → 79.1%.

**3. 전체 데이터 (GNN, relative obs, no augment, 15,051 train states, 500 epochs, `--select within`, policy alone 500 games)**:

| label | WTA, const lr (이전) | WTA, cosine lr | **pair supervision, cosine lr** |
|---|---|---|---|
| robust teacher v1 | 6.18 shots / 48.1% | 12.25 / 23.8% | **4.35 / 68.4%** |
| geom teacher (32 cand.) | 9.48 / 30.8% | 9.19 / 32.3% | **4.34 / 68.1%** |

(shots/game, shot success rate; SAC 7.89, entropy-lowered SAC 7.39, geom teacher itself 2.24, robust teacher itself 2.98)

- **lr decay만으로는 도움이 안 된다**: robust가 6.18→12.25로 오히려 나빠짐(train loss 267 vs val loss 673, 외우는 쪽으로 감). 위 "const lr이 문제"라는 처음 가설은 틀렸다 — 512 states overfit test에서만 도움이 됐다.
- **진짜 원인은 WTA의 moving-target selection**이었다. pair supervision이 두 label 모두 ~4.3 shots/game로 올렸다(SAC 7.89 대비 −45%).
- **Correction**: 이전의 "robust teacher label이 geom label보다 훨씬 낫다"(6.18 vs 9.48)는 결론은 **WTA 학습의 불안정성이 만든 격차**였다. pair supervision 하에서는 두 label이 같다(4.35 vs 4.34 shots). 그 전에 낸 모든 GNN 수치(총정리 표 포함)는 WTA + const lr로 학습된 값이라, 같은 조건으로 다시 비교해야 한다.
- 이상한 점: robust+pairs의 val loss가 2990(train 44.8)으로 매우 크다. within 0.2° 54.7%, 게임 성적은 좋아서 일부 outlier states(CE가 큰 states)가 val loss를 지배하는 것으로 보이지만 확인하지 않았다. orig32+pairs는 train 121 / val 146로 정상.
- **margin teacher v2 + pair supervision** (같은 설정, 이후 추가): 4.41 shots/game, success 67.7%, pocket_rate_at5 91.0%, within 0.2° 44.8% — robust(4.35), geom32(4.34)와 같은 수준. 세 label 모두 ~4.3-4.4. 여기서도 val loss가 큼(train 45.6 / val 3308): label이 ≤2 candidate인 robust, margin에서만 보이는 현상. (no-pairs baseline은 queue script 오류로 미실행)
- Caveat: seed 1개, run 1회. pair label은 teacher가 ghost-ball aim으로 만든 candidate에서 복원하므로 "(ball, pocket) pair 단위"라는 prior가 label에 들어간다(architecture의 pair slot과 같은 prior).

### Intent는 verify되는가? label의 (ball, pocket) pair vs simulator outcome (2026-10-09)

**Question**: pair supervision은 "이 action으로 이 ball을 이 pocket에 넣겠다"를 policy에 심는가?
현재는 pair slot이 암묵적 intent이고, label → slot 매핑은 label angle이 어느 pair의 ghost-ball aim과 가까운지(≤2.5°)로 정한다 — simulator outcome으로 확인하지 않는다.

**Check** (400 states, `good_actions[0]`을 restore한 table에서 쳐서 pooltool `BALL_POCKET` event의 (ball, pocket)을 읽음):

| label | intended (ball, pocket) 실제로 made | right ball, other pocket | other ball(s) | scratch | pair 미할당 |
|---|---|---|---|---|---|
| robust teacher v1 | **99.0%** | 0.0% | 1.0% | 0.5% | 5/400 |
| geom teacher (32 cand.) | 71.9% | 20.3% | 7.8% | 2.8% | 5/400 |

- robust label은 intent가 거의 verified (99%). geom label은 28%가 intended pair와 다른 outcome.
- 그럼에도 pair supervision 결과는 둘 다 ~4.3 shots/game — 약 28%의 noisy pair label이 있어도 학습이 됐다는 뜻이지만, intent가 clean한 것은 아니다.
- sims 파일에는 outcome의 개수만 있고 (ball, pocket) identity가 없다 → outcome-grounded pair label이나 goal-conditioned 학습을 하려면 simulation 기록에 identity를 추가해야 한다.

**Full-data result** (`exp16_wm/outcomes.py`: event log의 BALL_POCKET으로 label action이 실제로 made하는 (ball, pocket) slot을 `good_made (N,2,18)`, `good_scratch (N,2)`로 label npz에 추가; 앞 2개 candidate):

| label set | candidate | pair assigned | **intended pair really made** | any ball made | scratch |
|---|---|---|---|---|---|
| robust v1 | 0 / 1 | 16,502 / 13,960 | **97.7%** / 91.4% | 99.9% / 99.3% | 166 / 223 |
| margin v2 | 0 / 1 | 16,514 / 13,981 | **95.5%** / 84.6% | 100% / 100% | 108 / 108 |
| geom (32 cand.) | 0 / 1 | 16,496 / 16,455 | **73.5%** / 77.8% | 99.8% / 99.8% | 595 / 599 |

- 400-state 표본 결과(99.0%, 71.9%)와 일치. 거의 모든 label이 무언가는 넣지만, **angle matching으로 붙인 pair가 실제 outcome과 다른 비율**이 geom label은 ~26%, margin/robust는 2~5%(candidate 1은 8~15%).
- geom teacher label은 scratch도 ~3.6%(candidate 0)로 robust(1.0%)/margin(0.6%)보다 많다.
- 다음: label의 pair를 angle matching 대신 `good_made`(실제 made slots)로 바꾸는 outcome-grounded pair label과, pair별 "made?" aux head.

**Policy가 (i, j)를 실제로 "아는가"** (val 1,697 states, policy의 top-scored pair slot과 그 pair의 action을 restore한 table에서 쳐서 event log로 확인):

| model | clean pocket | chosen pair (i,j) really made (scratch 포함) | clean pocket 중 chosen pair였던 비율 |
|---|---|---|---|
| robust, WTA const lr (이전) | 52.6% | 55.3% | 95.6% |
| robust, pair supervision | 68.7% | 75.7% | **98.0%** |
| geom32, pair supervision | 61.2% | 68.8% | 93.7% |

- 모델은 이미 pair slot ↔ action을 일관되게 묶고 있다: shot이 clean하게 들어갈 때 94~98%는 선택한 pair가 made된 것이다 (intent-action consistency는 병목이 아님).
- 실패의 대부분은 **chosen pair를 못 넣는 것**(robust+pairs 24%, geom32+pairs 31%)이고, **chosen pair를 넣었지만 scratch**가 8~11%다(made − clean). 다른 pair를 넣은 clean pocket은 1~4%.
- 따라서 (i,j) intent label을 더 명시적으로 만드는 것의 가치는 "intent 일관성 개선"이 아니라 (a) geom label의 26% impure pair 정리, (b) pair별 P(made)를 verifier로 써서 선택/action 보정에 활용, (c) scratch를 별도 target으로 분리하는 데 있다.

**Label incompleteness (sample 0, robust v1 label)**: 18개 pair를 teacher와 같은 방식(coarse 15 sims + 2D margin scan)으로 전부 scan해 보니 feasible pair가 6개이고, 그중 4개가 거의 동일한 품질이다.

| (ball, pocket) | cut° | coarse hits/15 | 2D margin | in label | model p |
|---|---|---|---|---|---|
| (3,2) | 3.9 | 9 | 1.41 | yes | 0.299 |
| (2,5) | 34.1 | 8 | 1.41 | yes | 0.219 |
| (2,2) | 20.9 | 8 | 1.41 | **no** | 0.204 |
| (1,2) | 23.8 | 8 | 1.41 | **no** | 0.139 |
| (2,1) | 77.1 | 1 | 1.00 | no | 0.120 |

- teacher는 coarse hit 수 상위 **2개 pair만** 정밀 scan하고 label로 남기므로, 동점(8 hits)인 (2,2), (1,2)는 배열 순서에 따라 빠진다. selection CE는 이들을 **false negative**로 취급한다 — 그래도 model은 0.14~0.20을 준다(geometry가 positive와 비슷해서).
- 2D margin이 이산화 때문에 1.41 같은 값에 몰려 이 state에서는 네 pair를 구분하지 못한다(tie).
- 개선 후보: 상위 2개가 아니라 feasible한 모든 pair(coarse hit > 0 또는 margin ≥ 임계)를 positive로 label (scan 비용 ~3배), selection target을 coarse hit/margin에 비례한 soft distribution으로.

### Wrap-aware loss는 val loss를 고쳤지만 policy를 악화시켰다 + 모델 init seed 버그 (2026-10-09)

**val loss 이상의 원인(TODO #8)**: `delta_angle`이 ±180°에서 이어지는 원형인데 loss가 raw 차이로 계산 → label이 ±150° 밖인 backward shot(label의 1.3%, state의 2.3%)에서 loss가 1e6까지 폭발. val 5개 state가 val loss 합의 99%. median val loss(15.6)는 train(15.2)과 같음. wrap-aware 차이로 재계산하면 val loss robust 2807→261, margin 3287→614. pair 매핑 버그는 아님. 부수 발견: backward shot의 wrapped angle error 중앙값 1.1~1.3° vs 나머지 0.16~0.22°.
`action_diff`/`wrap_angle_unit`(gnn_policy.py)으로 GNN/MLP/label_gap loss를 wrap-aware로 고침.

**Seed 버그**: `bc.main()`에서 model이 `train_bc`의 `torch.manual_seed`보다 먼저 만들어져서 `--seed`와 무관하게 초기 weights가 매번 달랐다. `seed_everything(args.seed)`를 model 생성 전에 호출하도록 고침. 이 수정 전의 모든 BC 단일 실행은 init이 seed되지 않은 1회 결과다.

**Wrap fix의 효과** (robust label, GNN, pair supervision, cosine lr 1e-3, 500 epochs, seed 0/1/2 짝지어, policy alone 500 games):

| | seed 0 / 1 / 2 shots/game | mean ± std | success | val within 0.2° | val loss |
|---|---|---|---|---|---|
| raw angle diff (`--no-angle-wrap`) | 4.32 / 4.45 / 4.76 | **4.51 ± 0.23** | 66.2 ± 3.3% | 50.6 / 54.1 / 47.3% | 1,180~2,091 |
| wrap-aware (fix) | 5.96 / 5.46 / 5.34 | 5.59 ± 0.33 | 53.5 ± 2.9% | 31.2 / 38.5 / 35.3% | 112~858 |

- seed 3개 모두에서 wrap이 나빴고 차이(~1.1 shots)는 seed std(~0.3)의 3배 이상 → 잡음이 아닌 효과. seed 고정 전 1회 결과(robust 5.40, margin 5.02, geom32 6.90)도 같은 방향.
- 이전 loss로 seed를 달리한 결과(4.32~4.76)가 고정 전 단일 실행(4.35 robust, 4.41 margin, 4.34 geom32)과 같은 범위 → 그 숫자들은 신뢰할 만하다.
- **가설(검증 중)**: 이전 loss의 spike(1e6)가 Adam의 second-moment 추정을 키워 실효 learning rate를 낮추는 브레이크로 작용했다 → wrap으로 spike가 사라지면 같은 lr이 사실상 커져 정밀도가 떨어진다. 검증: wrap loss에서 lr 3e-4, 1e-4 (seed 0~2).
- **seed 간 std**: shots/game ~0.2-0.3, success ~3%p. 이 크기보다 작은 단일 실행 차이(예: robust 4.35 vs margin 4.41 vs geom32 4.34)는 label 효과로 해석할 수 없다.

**lr 검증 결과** (wrap-aware loss, robust, seed 0/1/2):

| | shots/game | success | val within 0.2° |
|---|---|---|---|
| wrap, lr 1e-3 | 5.59 ± 0.33 | 53.5% | 35.0% |
| wrap, lr 3e-4 | **4.94 ± 0.08** | 60.3% | 42.1% |
| wrap, lr 1e-4 | 6.30 ± 0.33 | 47.3% | 29.1% |
| no-wrap, lr 1e-3 | **4.51 ± 0.23** | 66.2% | 50.7% |

- lr은 비단조(3e-4가 최적, 1e-4는 500 epoch에 비해 느림). 그러나 어떤 lr에서도 wrap이 no-wrap(lr 1e-3)을 넘지 못했다 → "spike가 Adam의 브레이크" 가설은 lr을 낮추면 개선된다는 부분만 지지되고 완전히 설명하지 못한다.
- 결정: 학습 loss 기본값은 raw(no-wrap) 유지(`--angle-wrap`은 opt-in). val loss를 읽을 때는 median 또는 wrap-aware 값을 따로 계산한다. no-wrap에서 lr 3e-4 등과 wrap + grad clipping은 미검증(roadmap 1b).

### Pair-level outcome 통계로 selection 개선: TODO 2+3+4 (2026-10-09 계획)

**기준(baseline A)**: robust label, GNN, relative obs, pair supervision(angle matching + 전체 18개 softmax), raw angle loss, lr 1e-3 cosine, 500 epochs, `--select within`, seed 0/1/2 → **4.51 ± 0.23 shots/game, success 66.2 ± 3.3%** (seed std 이내 차이는 효과로 읽지 않는다).

**문제 (이전 분석)**:
- 한 state에서 feasible한 pair가 여러 개인데 teacher는 상위 2개만 label → 나머지는 selection CE의 false negative (sample 0: (1,2), (2,2)). 특히 ball 1은 supervision이 없다.
- label → pair 매핑이 angle matching이라 geom label은 26%가 intended pair와 다른 outcome을 만든다.
- policy가 chosen pair를 넣고도 cue ball이 빠지는 scratch가 8~11%.

**데이터** (`exp16_wm/pair_stats.py`, state 단위라 label 종류와 무관): 각 state에서 cut ≤ 80°인 **모든 pair**에 대해 ghost-ball aim 주변 coarse candidate 20개(offset {±0.4, ±0.2, 0}° × speed {0.75, 1.5, 3, 5}m/s)를 simulation하고 event log(BALL_POCKET)로 읽어 pair p=(ball i, pocket j)마다 저장:
- `pair_n (N,18)`: 시도한 candidate 수(20, cut 초과/포켓된 ball은 0)
- `pair_made (N,18)`: pair p의 candidate 중 **intended (i,j)를 실제로 made하고 scratch가 없는** 횟수 (intent verified, 3·4번)
- `pair_any (N,18)`: 아무 ball이라도 clean하게 넣은 횟수 (이전 teacher의 feasibility 정의, 비교용)
- `pair_scratch (N,18)`: cue ball이 빠진 횟수 (4번)
비용: 상태당 최대 360 sims(평균 ~150) → 1.67만 states 8조각 병렬 ~10-15분. 결과를 robust/margin/orig32 shard에 같은 key로 붙인다.

**Model** (`--selection factorized`, `PairGNNPolicy(factorized=True)`):
- ball head `b_i = Linear([h_i, max_j e_ij])`, pocket logit `l_ij = Linear(e_ij)`(per-ball softmax), scratch head `s_ij = Linear(e_ij)`. `forward`는 joint log p(i,j) = log softmax(b)_i + log softmax(l_i)_j 를 logits로 반환 → 기존 argmax inference/분석 코드 호환.
- Loss = reg + 10·(ball CE + pocket CE) + 10·scratch BCE
  - reg(3번): label candidate k의 action이 실제로 made한 모든 pair p(`good_made[k,p]`, scratch 없는 candidate만)에 대해 `‖(a_p − a_k)·w‖²`의 평균. (한 action이 두 pair를 made하면 두 head 모두 target)
  - ball CE(2번): target_i ∝ max_j pair_made[i,j] (feasible한 ball만), pocket CE: ball i마다 target_j ∝ pair_made[i,j] (합 > 0인 ball만). 둘 다 soft CE.
  - scratch BCE(4번): feasible pair에 대해 target = pair_scratch / pair_n.
- Inference: argmax of joint − λ·σ(s_ij); λ=0 기본, λ=3 평가를 추가(4번 효과 분리).

**실험**: robust label, factorized + outcome-grounded + scratch, seed 0/1/2 vs baseline A. 지표: shots/game, success, clean pocket 중 chosen pair 비율, chosen pair 미made율(miss), made지만 scratch 비율(label_gap 방식, val 1,697 states), selection이 고른 pair의 pair_made 순위. ablation(각 항목 기여 분리)은 효과가 보이면 그 다음.

**판단 기준**: A(4.51 ± 0.23)보다 shots가 seed std 이상 확실히 줄고(예: < 4.2) miss/scratch 비율이 줄면 → 채택. 아니면 어느 항목이 손해인지 ablation.

**결과 (2026-10-10, 첫 시도): 크게 나빠졌다 — 채택 불가.**

pair_stats(16,748 states, 8조각, ~17분): state당 시도한 pair 평균 4.0개, made-pair가 있는 state 97~98%, scratch 비율 ~16% (coarse candidate 기준).

| (robust label, seed 0/1/2) | shots/game | success | val within 0.2° |
|---|---|---|---|
| baseline A (global softmax + angle-matched pairs) | 4.32 / 4.45 / 4.76 → **4.51 ± 0.23** | **66.2 ± 3.3%** | 50.7% |
| factorized + outcome-grounded + scratch head, λ=0 | 7.98 / 7.36 / 7.94 → 7.76 ± 0.35 | 38.3 ± 1.6% | 25.3% |
| 같은 모델, scratch penalty λ=3 | 8.24 / 7.14 / 8.01 → 7.80 ± 0.58 | 38.2% | |

SAC(7.89) 수준으로 악화. 세 항목(2·3·4)을 한꺼번에 바꿔서 어느 것이 원인인지는 이 실험만으로는 분리되지 않는다.

**진단 (seed 0, val 1,000 states / train 1,000 states, policy의 top-1 pair를 restore한 table에서 쳐서 확인)**:
- chosen pair의 **22.5%는 action label이 없다**(어떤 label action도 그 pair를 made하지 않음). 이 pair의 clean pocket 20.9% vs label 있는 pair 46.1%. 원인: 새 selection target(`pair_made`)은 label 없는 feasible pair에도 확률을 주는데, action head는 label action이 made한 pair만 regression을 받는다 → **selection mass와 action supervision이 정렬되지 않는다**. (이전 baseline은 selection이 label pair에만 있어서 항상 정렬됐다.)
- 그것만으로는 설명이 안 된다: label 있는 pair를 고른 경우에도 clean pocket이 46%(train states 49.7%) — baseline의 66%보다 낮고 train에서도 낮다 → action regression 자체가 부실해졌다. 의심: selection 항목이 3개(ball CE, pocket CE, scratch BCE)로 늘어 각 ×10으로 regression을 압도(soft target의 irreducible entropy도 gradient를 유지). 미검증.
- scratch penalty λ=3은 효과 없음(7.76 → 7.80): chosen pair 선택이 이미 부정확해서 scratch head를 분리해 볼 단계가 아니다.
- 다음: (a) `--sel-weight` 1, 3으로 loss 균형 검증, (b) 모든 feasible pair에 coarse-stage action target 부여(`pair_action`)로 정렬 문제 해결, (c) 항목별 ablation(3만: global softmax + outcome-grounded pair).

**sel_weight 검증 (2026-10-10, factorized + outcome-grounded + scratch, seed 0/1/2, λ=0)**:

| sel_weight | shots/game | success |
|---|---|---|
| 10 (위 첫 시도) | 7.98 / 7.36 / 7.94 → 7.76 ± 0.35 | 38.3% |
| 3 | 8.00 / 7.28 / 7.74 → 7.67 ± 0.36 | 38.7% |
| 1 | 8.80 / 7.13 / 7.39 → 7.77 ± 0.90 | 38.2% |

- 셋 다 seed std 안 → "selection 항목 ×10이 regression을 압도" 가설은 기각.

**reg-only ablation (`--fact-terms reg`, seed 0): 무효 run.** 19.8 shots, clear 75%. selection head가 학습되지 않아 inference가 random init head로 pair를 고른다(chosen pair 중 label 있는 pair 15%, baseline 85%). `--select within`도 이 random selection을 거친 지표라 checkpoint가 epoch 28에서 골라졌다. regression 품질도 분리해서 볼 수 없다 → reg only + `--select within` 조합은 쓰지 않는다.

**Oracle-selection 진단 (seed 0, val 1,697 states)**: selection을 빼고 regression만 보려고, label action이 실제로 made한 (candidate k, pair p)마다 pair p head의 action을 label k와 비교했다.

| model | angle err median | within 0.2° | speed err median | top-1 pair가 made 가능 (`pair_made`>0) |
|---|---|---|---|---|
| baseline A (pairs, nowrap) | **0.19°** | **50.7%** | 0.52 | 91.5% |
| factorized sw3 | 0.52° | 24.7% | 0.54 | 89.6% |
| factorized sw10 | 0.64° | 20.6% | 0.59 | 89.6% |
| factorized sw1 | 0.83° | 16.7% | 0.70 | 87.1% |

- selection은 거의 같다(made 가능한 pair를 고른 비율 90% vs 92%). **악화의 주원인은 regression**: 정답 pair를 알려줘도 각도 오차가 2.5~4배. train(첫 3,000 states)도 같은 수치라 overfitting이 아니라 underfitting.
- 따라서 앞의 진단("chosen pair 22.5%가 label 없음")은 부차적이다.
- regression target 차이로는 설명이 안 된다: 한 pair가 두 candidate의 target이 되는 경우 2.5%, row당 label candidate ≤2가 99%, robust label에서 angle matching과 `good_made`가 98% 일치 → target이 사실상 같다. 남은 차이는 loss/head 구조.

### 변경점 하나씩 켜기: factorized 악화 원인 분리 (2026-10-11 계획)

**왜**: factorized 실험은 baseline A에서 4가지를 한꺼번에 바꿨고(아래 ①~④), 그 결과 regression이 망가졌다. 어느 변경이 원인인지 하나씩 켜서 확인한다.

| | baseline A | 변경 |
|---|---|---|
| ① selection 구조 | 18 pair global softmax | ball head → per-ball pocket softmax (`--selection factorized`) |
| ② selection target | label pair (candidate별 1개) | `pair_made` 비율 (label 없는 feasible pair 포함) |
| ③ regression target pair | angle matching (`good_pairs`) | 실제 made pair (`good_made`) |
| ④ scratch head | 없음 | scratch BCE (×sel_weight) |

**구현**: `PairGNNPolicy`에 4개 스위치를 따로 켤 수 있는 modular loss (`--sel-target label|made`, `--reg-target angle|made`, `--scratch-head`, ①은 기존 `--selection`).
- selection target은 18-pair 분포 t로 통일: label = labelled candidate의 pair 개수 비율(baseline의 CE와 동일), made = `pair_made`(live pair) 비율. factorized 구조에서는 ball target = t의 ball marginal, pocket target = ball별로 정규화한 t. (기존 factorized 코드의 ball target ∝ max_j pair_made와 다름 → 전부 켠 run도 다시 돌린다.)
- target이 없는 row는 해당 항목에서 제외 (baseline A는 label 없는 1.6% row를 WTA로 학습 — 차이는 control run으로 확인).
- val metric에 oracle 각도 오차(위 진단)와 top-1 pair made 가능 비율을 추가.

**실험**: robust label, GNN, relative obs, raw angle loss, lr 1e-3 cosine, 500 epochs, sel_weight 10, `--select within`, **seed 0 고정**. 6 runs: control(전부 off = baseline 재현), ①만, ②만, ③만, ④만, 전부 on.
지표: shots/game, success, oracle angle err / within 0.2°, top-1 made 가능 비율.

**판단 기준**: control이 baseline A seed 0(4.32)과 seed std(~0.3) 안이면 구현 검증 완료. 하나만 켠 run 중 oracle 각도 오차가 0.19° → 0.5° 수준으로 커지는 것이 원인. seed 1개라 shots 차이 0.5 미만은 효과로 읽지 않는다.

**결과 (2026-10-11, seed 0, policy alone 500 games): 원인은 ③ regression target(`good_made`)이다.**

| run | ① 구조 | ② sel target | ③ reg target | ④ scratch | shots/game | success | oracle angle err | oracle within 0.2° | top-1 made 가능 |
|---|---|---|---|---|---|---|---|---|---|
| control | global | label | angle | – | 4.37 | 68.0% | 0.186° | 52.9% | 91.8% |
| ①만 | **factorized** | label | angle | – | 4.27 | 69.9% | 0.176° | 54.6% | 91.5% |
| ②만 | global | **made** | angle | – | 4.50 | 66.1% | 0.194° | 50.9% | 91.3% |
| ③만 | global | label | **made** | – | **7.63** | 38.8% | **0.555°** | 23.1% | 91.1% |
| ④만 | global | label | angle | **on** | 4.09 | 72.6% | 0.162° | 57.9% | 92.0% |
| 전부 on | factorized | made | made | on | 7.96 | 37.3% | 0.571° | 23.0% | 90.0% |

- control 4.37은 baseline A seed 0(4.32)과 같다 → modular 구현 검증 완료 (label 없는 1.6% row를 WTA 대신 제외한 차이는 영향 없음).
- **③만 켜도 전부 켠 것과 같은 수준으로 망가진다** (7.63 vs 7.96, oracle 각도 오차 0.19° → 0.56°). ①②④는 하나씩 켜면 baseline과 seed std(~0.3) 안. ④의 4.09는 seed 1개라 개선으로 읽지 않는다.
- selection(top-1 made 가능 비율)은 6개 run 모두 90~92%로 같다 → selection 변경(①②)은 무해하다.

**③이 왜 망가뜨리나 (데이터 확인, 원인 가설)**: regression target을 (candidate, pair) 단위로 세면 angle matching 30,462개, `good_made` 30,819개, 겹치는 것 28,582개(93%). 앞의 "98% 일치"는 candidate 0만 본 수치였다. `good_made`에만 있는 target 2,237개 중 **61%는 다른 공의 pair**다: 공 A를 겨눈 action이 결과적으로 공 B를 넣은 경우(combination/kiss shot). ③은 공 B의 pair head에 "공 A를 겨누는 action"을 regression시킨다. pair head는 공 B와 포켓의 geometry로 action을 내는 구조라 이 target을 맞출 수 없고, squared error(0.2° = 1)가 커서 loss를 지배한다. 학습 중 train loss도 ③ run만 ~3,000~4,000으로 다른 run(200~700)의 5~10배였다.

**다음**:
- ③을 고치려면 `good_made` target에서 **겨눈 공과 다른 공의 pair를 빼거나**(made ∩ 같은 공), angle matching을 유지하고 made는 selection/verifier에만 쓴다. 위 가설은 "made ∩ 같은 공" run이 control 수준으로 돌아오면 검증된다.
- ①②④는 무해하므로, ③을 빼고 ①②④를 켠 조합을 seed 0/1/2로 돌려 baseline A(4.51 ± 0.23)와 비교. ④(4.09)가 실제 개선인지도 여기서 3 seed로 확인.

### ③ 원인 검증: 겨눈 공과 다른 공의 target 제거 (2026-10-11 계획)

**가설**: ③이 망가지는 이유는 combination shot target(공 A를 겨눈 action이 공 B를 넣음 → 공 B의 pair head가 공 A를 겨누는 action을 학습)이다.
**검증**: `--reg-target made_same` = `good_made` target 중 **겨눈 공(angle-matched pair의 공)과 같은 공의 pair만** 남긴다. angle pair가 없는 candidate는 제외. 나머지는 ③만 run과 동일 (global, label sel target, scratch off, seed 0).
**판단**: oracle 각도 오차가 control(0.19°) 근처, shots가 control(4.37)에서 ~0.5 이내로 돌아오면 가설 채택. ③만(0.56°, 7.63) 근처면 기각 → 다른 원인(같은 공 다른 포켓 target, scratch candidate 제외 등) 조사.

**결과 (2026-10-11, seed 0): 가설 채택 — combination shot target이 ③ 악화의 원인.**

| run | reg target | shots/game | success | oracle angle err | oracle within 0.2° |
|---|---|---|---|---|---|
| control | angle | 4.37 | 68.0% | 0.186° | 52.9% |
| ③만 | made | 7.63 | 38.8% | 0.555° | 23.1% |
| **③ made_same** | made ∩ 겨눈 공 | **4.37** | **68.1%** | **0.173°** | **55.4%** |

- 다른 공의 target(combination shot)만 빼자 control과 같아졌다(4.37 = 4.37, oracle 0.17° vs 0.19°). train loss도 ③만 run의 ~3,000에서 ~25(epoch 400)로 정상화.
- 즉 "실제로 넣은 pair"로 target을 붙이는 아이디어 자체는 무해하지만 이득도 없다(robust label에서는 같은 공 안에서 angle matching과 거의 같은 target).
- 결론: pair head가 학습할 target은 **겨눈 공의 pair**여야 한다. combination shot은 별도 처리(제외 또는 겨눈 공 head에 귀속)가 필요하다.
- 다음: ①②④ + (angle 또는 made_same) 조합을 seed 0/1/2로 baseline A(4.51 ± 0.23)와 비교, ④(seed 0 4.09)가 실제 개선인지 확인.

### ①②④ 조합 (2026-10-11 계획)

①(factorized 구조) + ②(pair_made selection target) + ④(scratch head)를 켜고 ③은 끈다(reg target = angle). seed 0, 나머지는 위 ablation과 동일. 하나씩 켰을 때 셋 다 무해했으므로(4.27 / 4.50 / 4.09), 합쳐도 control(4.37) 수준 이상인지 본다. seed 1개라 control과 0.5 미만 차이는 효과로 읽지 않는다.

**결과 (2026-10-11, seed 0): control과 같다 — 이득 없음.**

| run | shots/game | success | oracle angle err | oracle within 0.2° | top-1 made 가능 |
|---|---|---|---|---|---|
| control | 4.37 | 68.0% | 0.186° | 52.9% | 91.8% |
| ①②④ (③ off) | 4.39 | 68.0% | 0.175° | 54.1% | 91.2% |
| ①②④ + scratch penalty λ=3 | 4.41 | 67.5% | | | |

- 셋을 합쳐도 control과 차이 없음(4.39 vs 4.37). selection(top-1 made 가능 91%)과 regression(oracle 0.18°) 모두 그대로.
- scratch penalty(λ=3)도 효과 없음. ④만 켰을 때의 4.09는 ①②와 합치면 재현되지 않았다 → seed 잡음이었을 가능성이 높다.
- 해석: robust label에서는 selection 개선 여지가 작다(top-1이 이미 91% made 가능). 남은 실패는 고른 pair를 정확히 못 치는 것(oracle within 0.2° ~54%) → 다음 개선 대상은 selection이 아니라 regression 정밀도.

### ④ scratch head만, seed 3개 (2026-10-11 계획)

④만 켠 run(seed 0)이 4.09로 control(4.37)보다 좋았지만 ①②와 합치면 재현되지 않았다. seed 1, 2를 추가해(seed 0은 기존 run 재사용) baseline A(4.51 ± 0.23, 같은 설정 seed 0/1/2)와 비교한다. 판단: 평균 차이가 seed std(~0.3) 이상이면 개선, 아니면 잡음.

**결과 (2026-10-11): ④ scratch head는 seed 3개 모두에서 baseline A보다 좋다 — 채택 후보.**

| (robust, seed 0 / 1 / 2) | shots/game | success | oracle angle err | oracle within 0.2° | top-1 made 가능 |
|---|---|---|---|---|---|
| baseline A | 4.32 / 4.45 / 4.76 → 4.51 ± 0.23 | 66.2 ± 3.3% | 0.196 / 0.176 / 0.207° | 50.7 / 54.8 / 48.8% | 91.5 / 93.0 / 90.7% |
| ④ scratch head | 4.09 / 4.23 / 3.89 → **4.07 ± 0.17** | **73.3 ± 3.3%** | 0.162 / 0.154 / 0.133° | **57.9 / 60.1 / 65.2%** | 92.0 / 91.6 / 91.6% |

- seed별로 짝지으면 3개 모두 개선(−0.23, −0.22, −0.87 shots), 평균 차이 0.44는 seed std(0.17~0.23)의 2배 이상. success +7%p.
- inference에서 scratch penalty를 써서 좋아진 것이 아니다: penalty 없이(λ=0) 측정한 결과이고, λ=3은 오히려 같거나 약간 나쁨(4.12 / 4.17 / 3.97).
- 개선은 **regression 정밀도**에서 온다: oracle within 0.2°가 51% → 61%, selection(top-1 made 가능)은 그대로. scratch BCE가 auxiliary task로 공유 encoder `e`에 cue ball 경로 정보를 학습시켜 action head의 feature가 좋아진 것으로 추정(미검증).
- 주의: ①②④ 조합(seed 0)은 4.39로 이 개선이 보이지 않았다 → ①(factorized) 또는 ②(made selection target)가 ④의 이득을 상쇄할 가능성. seed 1개라 확정은 아님.
- 다음: (a) ④를 baseline에 기본 채택할지 결정, (b) 개선 메커니즘 확인 — scratch 대신 다른 auxiliary target(예: pair_made 회귀)으로도 regression이 좋아지는지, (c) ①②④ 조합 seed 1, 2.

### ①만 / ②만 / ①②④ seed 3개 (2026-10-11 계획)

④만 seed 3개와 같은 기준으로 비교하기 위해 seed 1, 2를 추가(seed 0은 기존 run 재사용): ①만(factorized 구조), ②만(made selection target), ①②④. 6 runs 병렬. ①②④가 ④만(4.07 ± 0.17)보다 나쁘면 ① 또는 ②가 ④의 이득을 상쇄하는 것.

### Look-ahead 보조 target 1단계: state-level made head (2026-10-11 계획)

**왜**: ④ scratch head는 inference에 쓰지 않았는데도(λ=0) regression 정밀도를 올렸다(oracle within 0.2° 51% → 61%, 3 seeds). 보조 과제가 공유 encoder `e`를 더 잘 학습시켰다는 가설이 맞다면, 같은 형태의 다른 보조 target으로도 좋아져야 한다. 그중 가장 직접적인 것이 "이 pair가 들어갈 수 있나"(look-ahead)다. 관련 기존 TODO: roadmap #5 Made head(action-conditional verifier) — 그 전 단계로 데이터가 이미 있는 state-level 버전부터 한다.

**설계**: `head_made(e_ij)` → logit, target = `pair_made / pair_n`(pair_stats의 coarse 후보 20개 중 intended pair를 scratch 없이 넣은 비율), 시도한 live pair(`pair_n > 0`)에 BCE × sel_weight. ④ scratch head와 형태·가중치가 같고 target만 다르다. inference에는 쓰지 않는다(보조 target only). `--made-head`.

**실험** (global, label sel target, angle reg target, sel_weight 10, seed 0/1/2):
- ⑤ made head만
- ④ + ⑤

**판단**: baseline A(4.51 ± 0.23), ④만(4.07 ± 0.17)과 비교.
- ⑤만이 baseline A보다 seed std 이상 좋고 oracle within이 오르면 → "보조 과제가 encoder를 개선" 가설 지지.
- ④+⑤가 ④만보다 좋으면 → 이득이 더해짐, 2단계(action-conditional made head)로.
- ⑤만이 baseline 수준이면 → ④의 이득은 scratch 정보 고유의 것(cue ball 경로)이고 일반적인 보조 과제 효과가 아님.

**결과 (2026-10-11): 모든 변경이 new-code control과 같다 — ④ "개선"은 비교 기준 오류였다 (정정).**

| (robust, seed 0 / 1 / 2) | shots/game | success | oracle within 0.2° |
|---|---|---|---|
| baseline A (이전 코드, 10-09) | 4.32 / 4.45 / 4.76 → 4.51 ± 0.23 | 66.2% | 50.7 / 54.8 / 48.8% |
| **control (new code, 변경 없음)** | 4.37 / 4.06 / 3.82 → **4.08 ± 0.28** | 72.9% | 52.9 / 60.4 / 64.0% |
| ① 구조만 | 4.27 / 4.18 / 3.95 → 4.13 ± 0.16 | 72.1% | 54.6 / 60.6 / 60.8% |
| ② sel target만 | 4.50 / 4.35 / 3.83 → 4.23 ± 0.35 | 70.8% | 50.9 / 57.8 / 63.4% |
| ④ scratch만 | 4.09 / 4.23 / 3.89 → 4.07 ± 0.17 | 73.3% | 57.9 / 60.1 / 65.2% |
| ①②④ | 4.39 / 3.93 / 4.12 → 4.15 ± 0.23 | 72.0% | 54.1 / 61.5 / 57.3% |
| ⑤ made head만 | 4.40 / 4.21 / 3.84 → 4.15 ± 0.28 | 71.8% | 54.1 / 55.1 / 64.4% |
| ④+⑤ | 4.45 / 4.15 / 3.88 → 4.16 ± 0.29 | 71.8% | 50.2 / 59.4 / 63.8% |

- **①②④⑤ 어느 것도, 어떤 조합도 control(4.08)과 차이가 없다** (전부 4.07~4.23, seed std 안). seed별 패턴도 같다(seed 2가 항상 가장 좋음 → split 난이도).
- 앞에서 "④는 3 seed 모두 baseline A보다 좋다 — 채택 후보"라고 쓴 것은 **틀렸다**: 비교 대상이 이전 코드의 baseline A였고, control을 seed 0에서만(4.37 vs 4.32) 검증했다. 같은 코드의 control 3 seed와 비교하면 ④의 효과는 0이다. 보조 과제가 encoder를 개선한다는 가설도 근거가 없어졌다(⑤ made head도 효과 0).
- **새 코드 control 자체가 baseline A보다 좋다**(4.08 vs 4.51, seed별 +0.05 / −0.39 / −0.94, oracle within 51% → 59%). 두 코드의 유일한 학습 차이는 label pair가 없는 row(1.6%)를 baseline A는 WTA loss로 학습하고 modular loss는 제외한다는 점 → **WTA fallback이 regression을 해치고 있었을 가능성**(미검증; seed 0에서는 차이가 없어 잡음일 수도 있다).
- 교훈: 효과를 볼 때는 같은 코드·같은 seed의 control을 반드시 함께 돌린다.
- 결론: robust label에서 selection 쪽 변경(①②)과 보조 head(④⑤)는 모두 무의미. 병목은 regression 정밀도(oracle within ~60%)이고 이 변경들로는 움직이지 않는다.
- 다음 후보: (a) WTA fallback 효과를 old code에서 fallback만 끈 run으로 확인(맞으면 baseline을 4.08로 갱신), (b) regression 정밀도 직접 공략 — action-conditional made head(2단계)를 보조 target이 아니라 inference 시 action 미세조정(∇_a P(made), roadmap #6)에 쓰기.

### WTA fallback 검증 (2026-10-11 계획)

**질문**: new-code control(4.08 ± 0.28)이 baseline A(4.51 ± 0.23)보다 좋은 이유가 label pair가 없는 row(robust 16,748 중 230개, 1.37%)의 처리 차이인가? baseline A(`--pair-supervision`, `_pair_loss`)는 이 row를 WTA loss(가장 가까운 pair head가 학습, 매 step 바뀔 수 있음)로 학습하고, modular loss는 loss에서 제외(gradient 0)한다.

**실험**: 이전 코드 경로(`--pair-supervision`)에 `--no-wta-fallback`(label pair 없는 row 제외) 스위치만 추가. 같은 batch에서 seed 0/1/2:
- old code 그대로 (baseline A 재현 확인 겸)
- old code + `--no-wta-fallback`

**판단**: fallback 끈 쪽이 ~4.1, 그대로가 ~4.5면 → WTA fallback이 원인, baseline을 fallback 없는 설정으로 갱신. 둘이 같으면 → 차이는 다른 곳(재현성 등).

**결과 (2026-10-11): 가설 채택 — WTA fallback이 baseline A를 끌어내리고 있었다.**

| (robust, seed 0 / 1 / 2) | shots/game | success | oracle within 0.2° |
|---|---|---|---|
| baseline A (10-09) | 4.32 / 4.45 / 4.76 → 4.51 ± 0.23 | 66.2% | – |
| old code 재실행 (fallback 켬) | 4.32 / 4.45 / 4.76 → 4.51 ± 0.23 | 66.2% | 50.7 / 54.8 / 48.8% |
| **old code + `--no-wta-fallback`** | 4.28 / 4.15 / 4.00 → **4.14 ± 0.14** | **71.9%** | 53.7 / 61.2 / 64.9% |
| new-code control | 4.37 / 4.06 / 3.82 → 4.08 ± 0.28 | 72.9% | 52.9 / 60.4 / 64.0% |

- 재실행이 baseline A와 소수점까지 같다 → 학습은 seed 고정 시 완전히 재현된다(차이가 잡음/비결정성이 아님).
- fallback만 끄면 seed 3개 모두 같거나 좋아지고(−0.04 / −0.30 / −0.76), new-code control과 같은 수준이 된다 → new code가 좋았던 이유는 이 한 가지.
- **왜 1.37%가 이렇게 큰가**: label pair가 없는 230개 state는 특이하다 — `good_actions`가 많고(최대 80개 이상, planner fallback이 후보를 여럿 저장), teacher action이 backward shot(|angle| > 150°)인 비율 10%(labelled 0.7%). 학습된 old-code 모델(seed 0)에서 이 row의 평균 loss는 2,944로 labelled row(22)의 ~130배 → **train 1.4% row가 전체 train loss의 66%**를 차지한다. raw angle loss라 큰 오차가 그대로 gradient를 지배하고, WTA target(가장 가까운 head)도 매 step 바뀐다.
- **결정**: baseline을 **fallback 없는 설정(4.14 ± 0.14 / new-code control 4.08 ± 0.28)**으로 갱신. 이후 비교는 이 baseline과 같은 코드로 한다.
- 함께 의심할 것: no-label row뿐 아니라 labelled row 중 backward shot 등 loss가 큰 소수 row가 여전히 gradient를 지배하는지 (raw loss 문제, roadmap 1b/#8과 연결).

### 데이터 조각별 loss / 정확도 분석 (2026-10-11, 학습 없음)

**왜**: WTA fallback을 끈 뒤에도 소수 row가 loss를 지배하는지, 어떤 state에서 regression이 부정확한지 확인. 질문 "angle이 speed보다 절대값이 작으니 loss signal을 덜 받는 것 아닌가?"도 같이 본다.

**방법**: 새 baseline(old code + `--no-wta-fallback`, seed 0/1/2 모델)에서 labelled row마다 pair loss(가중 squared error, angle 0.2° = 1, speed 0.5 m/s = 1)를 angle/speed로 나눠 계산. train 44,565 rows(3 seed 합침)로 loss 비율, val 4,989 rows로 정확도(candidate 0의 labelled pair 출력 vs label, 각도 오차). 조각은 candidate 0의 pair 기준. script: scratchpad `slice_loss.py`(저장소에 넣지 않음).

**angle vs speed**: loss는 이미 dim별 정밀도 단위로 정규화돼 있어서(`dim_weights`) 절대값을 비교하지 않는다. raw 단위로 환산하면 angle 가중치가 speed보다 ~120배 크다. 그런데도 **train loss의 59%가 speed, 41%가 angle** — 지금 scale에서는 오히려 speed 오차가 loss를 더 차지한다. scale을 바꾼 실험은 MLP 시절 한 번뿐(raw [-1,1] 단위 vs 정규화, 결과 거의 같음 → "병목 아님")이고, GNN + pair supervision 이후 92개 run은 전부 `a0.2_v0.5`. 지금은 angle 정밀도가 병목이라 다시 볼 가치가 있다.

**소수 row 지배?**: 아니다. 상위 0.1% / 1% / 5% / 20% row가 loss의 5.6% / 14.9% / 33.0% / 67.8% — 꼬리가 있지만 no-label row(1.4%가 66%)만큼 극단적이지 않다. backward shot은 데이터 0.7%에 loss 3.8%, 정확도는 매우 낮다(within 0.2° 17.5%, n_val 40) — 비중이 작아 전체에 큰 영향은 없음.

| 조각 | 데이터 | train loss | val angle err median | val within 0.2° |
|---|---|---|---|---|
| 전체 | 100% | 100% | ~0.13° | ~67% |
| **남은 공 2개** | 16.4% | **27.0%** | **0.230°** | **46.2%** |
| 남은 공 1개 / 3개 | 38.1% / 45.4% | 30.1% / 42.9% | 0.112° / 0.125° | 75.1% / 68.4% |
| cue–공 거리 < 0.3 | 17.8% | 22.9% | 0.181° | 55.4% |
| 공–포켓 거리 < 0.3 | 27.2% | 32.4% | 0.171° | 54.7% |
| label speed ≥ 2 m/s | 23.6% | 39.2% | 0.145° | 61.1% |
| cut angle 0–20° → 60°+ | 45% → 4% | 비례 | 0.123° → 0.154° | 69.6% → 58.7% |
| corner / side pocket | 57% / 43% | 비례 | 0.128° / 0.129° | 68.2% / 66.0% |

- **가장 큰 이상: 남은 공 2개 state.** 정확도가 46%로 1개(75%), 3개(68%)보다 크게 낮고 loss를 비중보다 많이 차지한다. 공 개수가 단조적으로 어려워지는 게 아니라 2개만 나쁘다 → 난이도보다 **표현 문제**(포켓된 공 1개가 섞인 입력의 처리) 의심. 데이터 분포도 이상하다(3개 45%, 2개 16%, 1개 38%).
- 가까운 거리(cue–공, 공–포켓 < 0.3)에서 부정확. cut angle, 포켓 종류는 영향이 작다.
- 다음 후보: (a) 남은 공 2개 원인 조사 — 포켓된 공의 입력값(rx, ry)이 무엇이고 encoder가 어떻게 받는지, 2개 state가 어떤 상황인지, (b) loss scale sweep(`--angle-scale-deg` 0.1 등 angle 비중 ↑, 같은 코드 control 포함 3 seed).

### 남은 공 2개 state 원인 조사 (2026-10-11, 학습 없음)

**확인한 것** (새 baseline 모델 seed 0/1/2, val; 실제 성공률만 seed 0):

| 가설 | 확인 | 결과 |
|---|---|---|
| 일반화 실패? | train 정확도 | **아니다** — train도 within 0.2° 50.8% (1개 76.4%, 3개 70.1%) → underfitting |
| 포켓된 공 입력 처리(slot 순열) 문제? | 포켓된 slot / 겨눈 공 slot별 정확도 | **아니다** — 모든 slot 44~48%로 같음. 포켓된 공은 `(0, 0, 1)`로 들어가고 GNN이 message/pair에서 mask함 |
| label이 state와 불일치? | label action을 obs에서 복원한 table에 쳐서 intended pair made 비율 | **아니다** — 96.4% (1개 98.3%, 3개 97.7%). `restore_state`도 포켓된 공을 올바르게 처리 |
| 다른 공이 방해? | 다른 live 공과 겨눈 공/샷 경로 거리 | **아니다** — 2개 state에서 다른 공은 대개 멀다(중앙값 0.74, 경로 0.06 이내 4.6%). 멀어도 정확도 47% |
| 넓은 창이라 label 위치가 임의적일 뿐? | policy action을 실제로 쳐서 clean pocket 비율 | **아니다** — 실제 성공률도 2개 57.3% vs 1개 70.2%, 3개 74.4% (val 1,697, seed 0) |
| 기하(가까운 거리, cushion) | 조각별 | **일부** — 2개 state는 겨눈 공이 포켓에 가깝고(중앙값 0.27 vs 0.5~0.58) cushion 근처(< 0.1)가 38%. 겨눈 공이 cushion 0.05 이내면 전체적으로도 48%(멀면 72%). 하지만 거리·cushion을 통제해도 2개 state는 53~57%로 여전히 낮다 |

- 2개 state는 전부 게임의 두 번째 샷(첫 샷에 정확히 1개를 넣은 뒤)이다. 3개 state(시작 배치)는 cushion 근처 공이 0% — 시작 배치가 별도 분포다.

**남은 가설 (미검증)**: **"live 이웃 공이 정확히 1개"인 regime의 데이터 부족.** GNN의 ball message는 live 이웃에 대한 sum이라 이웃 수에 따라 h의 분포가 다르다: 1개 state는 이웃 0개(message 없음, 6,503 states), 3개 state는 이웃 2개(7,500 states), 2개 state만 이웃 1개(2,745 states, 데이터 16%). 그런데 2개 state 자체가 cushion·포켓 근처라는 어려운 분포까지 겹친다.

**검증 후보**:
1. **증강**: 3개 state에서 겨누지 않은 공 하나를 포켓 처리해 2개 state를 만들고, label action을 다시 쳐서 여전히 들어가는 것만 추가(simulation ~1ms/샷이라 수 초). 2개 state 정확도가 올라가면 데이터 부족이 원인.
2. aggregation을 sum → mean(+ 이웃 수 feature)으로 바꿔 이웃 수 regime 차이를 줄이기.

### 공 2개로 시작하는 데이터 추가 (2026-10-11 계획)

**왜**: 남은 공 2개 state가 train에서도 부정확(within 0.2° 51%, 실제 성공률 57%)하고, 남은 가설은 "live 이웃 공 1개" regime의 데이터 부족(2,745 states, 16%)이다. 사용자 제안: 공 2개로 시작하는 게임을 만들고, 하나가 포켓되면(1개 남으면) 다시 생성 → 2개 state를 원하는 만큼 만든다. teacher가 첫 샷에 거의 항상 넣으므로 사실상 state마다 독립적인 무작위 2-ball 시작 배치다.

**주의(분포)**: env의 시작 배치는 테이블 안쪽(공 x 15~85%, y 30~85%; cue y 15~40%)뿐이라 cushion 근처 공이 0%다. 실제 2개 state는 첫 샷 이후라 겨눈 공의 38%가 cushion 0.1m 이내이고 포켓에 가깝다(중앙값 0.27m). 그래서 두 버전을 만든다:
- **A (env 기본 배치)**: env.reset() 배치에서 무작위 slot 하나를 포켓 처리 → "이웃 1개 데이터 부족" 가설만 검증.
- **B (테이블 전체 균일)**: cue와 공 2개를 테이블 전체(cushion에서 공 반지름 이상, 포켓 중심에서 일정 거리 이상, 공끼리 겹치지 않게)에 균일 배치 → cushion·포켓 근처의 어려운 배치까지 포함.

**생성** (`exp16_wm/two_ball_states.py`): 각 7,500 states, 포켓 처리할 slot은 무작위(slot별 균등). 이후 기존과 같은 파이프라인: robust relabel(`relabel.py`) → `pair_labels` → `outcomes` → `pair_stats`. 저장 위치 `logs/teacher_data/two_ball_{A,B}/`.

**학습**: 새 baseline 설정(`--pair-supervision --no-wta-fallback`, 기타 동일), seed 0/1/2. 새 데이터는 **train에만** 추가(`--train-only-data`) — 기존 데이터의 train/val split과 val set을 그대로 유지해서 기존 결과와 직접 비교. 비교: 기존 데이터만(이미 있음, 4.14 ± 0.14) / +A / +B.
지표: policy alone shots/game, success, val 2개 state의 within 0.2°와 실제 clean pocket 비율, 1개/3개 state에 손해가 없는지.

**판단**: +A에서 2개 state 정확도가 오르면 데이터 부족이 원인. +B만 오르면 어려운 배치(cushion·포켓 근처) 데이터가 원인. 둘 다 그대로면 데이터가 아니라 모델/표현 문제.

### Look-ahead 2단계: action-conditional made head (2026-10-11 계획)

**왜**: 1단계(state-level made head)는 보조 target으로 효과가 없었다(4.15 vs control 4.08). 하지만 2단계의 주 목적은 보조 target이 아니라 **칠 때 쓰는 것**이다. 남은 병목은 고른 pair를 정확히 못 치는 것(oracle within 0.2° ~60%, 실제 clean pocket ~70%)이고, "이 pair에 이 action을 치면 들어가나"를 예측할 수 있으면 (a) pair를 고를 때 verifier로, (b) action을 미세조정하는 데 쓸 수 있다. roadmap #5(Made head), #6(∇_a P(made) refinement).

**데이터**: relabel 때 저장한 sims(`relabel_s1000/sims/`, 499.7만 건, state·action·포켓된 공 수·scratch, clean 48.8%)를 재사용. 이 sims는 ghost-ball 조준 주변 + 상위 2개 pair의 정밀 scan(각도 26 × 속도 5)이라 **창 가장자리 근처의 성공/실패가 촘촘하다**(verifier에 필요한 바로 그 영역). 빠진 것은 어느 (ball, pocket)이 들어갔는지(identity) → obs에서 table을 복원해 다시 쳐서 made slot(18)과 scratch를 얻는다(`made_slots`, ~3ms/샷 → 8 cores ~30분).

**Model**: `head_made2(e_ij, a)` = MLP([e_ij, a, a − a_ij]) → logit (a_ij: pair head의 proposal, detach). sample마다 live pair 전부에 BCE, target_ij = 그 action이 pair ij를 scratch 없이 넣었나. 기존 BC loss와 함께 학습(sims batch를 따로 뽑아 더함), 가중치는 sweep 없이 1개로 시작.

**평가**:
1. verifier 자체: val state의 sims에서 AUC, calibration (trivial baseline = state-level made rate, 1단계 head와 비교).
2. inference 사용 (같은 모델, 학습 그대로):
   - (V) pair 선택을 selection logit 대신 P(made | e_ij, a_ij)로 / 또는 둘을 합쳐서
   - (R) 고른 pair의 a_ij 주변 후보(예: ±0.1°, ±0.2° × 속도 3개)를 head로 평가해 최고를 친다 (gradient 없이 sampling; ∇_a는 그다음)
   - 지표: policy alone shots/game, success. 같은 모델의 기본 inference가 control.
3. 학습 영향: made head를 켜도 기본 inference 성능이 control(새 baseline, 같은 코드)과 같은지.

**판단**: (V) 또는 (R)이 기본 inference보다 seed std 이상 좋으면 채택. verifier AUC가 높은데 (R)이 효과 없으면 → head가 0.1° 단위 차이를 구분 못 하는 것(해상도 문제), 창 가장자리 sims의 비중 조정 검토.

**순서**: 공 2개 데이터 학습이 CPU를 쓰는 동안 코드+테스트를 작성하고, sims identity 재simulation은 그 학습이 끝난 뒤 실행.

**결과 (2026-10-11): 공 2개 데이터는 2개 state를 개선하지 못했다 — "데이터 부족" 가설 기각.**

데이터 생성(각 7,500 states, 5 shards, relabel ~33분): A — candidate 0 intended pair made 97%대, B — 97.5%, scratch 거의 없음. pair_stats 시도 pair/state A 4.13, B 3.43.

| (seed 0/1/2, val은 기존 데이터의 같은 split) | shots/game | success | 2개 state within 0.2° / 실제 clean | 1개 state within / clean | 3개 state within / clean |
|---|---|---|---|---|---|
| base (기존 데이터만, fallback 끔) | 4.28 / 4.15 / 4.00 → **4.14 ± 0.14** | 71.9% | 46.4% / **60.6%** | 75.1% / 71.8% | 68.4% / 74.4% |
| + A (안쪽 배치 2-ball 7,500) | 4.55 / 4.20 / 4.45 → 4.40 ± 0.18 | 67.6% | 41.9% / 56.9% | 61.3% / 65.5% | 57.0% / 71.6% |
| + B (테이블 전체 2-ball 7,500) | 4.25 / 3.99 / 4.15 → 4.13 ± 0.13 | 72.0% | 46.6% / 61.1% | 69.6% / 67.4% | 61.6% / 71.7% |

- 2개 state 데이터를 2,745 → 10,245로 ~3.7배 늘려도 **2개 state 정확도·성공률이 그대로**(B 61.1% vs 60.6%)이거나 나빠졌다(A 56.9%). 데이터 부족은 원인이 아니다.
- 오히려 **다른 state가 나빠졌다**: 1개 state within 75% → 61%(A) / 70%(B), 3개 state 68% → 57% / 62%. 원인 미상 — 후보: 새 데이터가 train의 1/3을 차지해 기존 분포의 학습 비중이 줄어듦, checkpoint 선택(`--select within`, 기존 val 기준) 시점 변화. A(쉬운 안쪽 배치)가 B보다 더 해롭다.
- 정리: 실제 2개 state의 약점은 데이터 양으로도, 배치(cushion·포켓 근처)를 채워도 해결되지 않는다. 지금까지 기각: 일반화, slot 표현, label 불일치, 방해 공, 창 폭, 데이터 양. 남은 방향: 실패 사례를 직접 보기(2개 state 중 policy가 실패한 샷을 그려서 label·policy action·결과 비교).

### 남은 공 개수별 selection vs regression 분리 (2026-10-11, 학습 없음)

**질문**: 2개 state의 약점이 "공 2개 중 양자택일"(selection)에서 오는가? **방법**: base 모델(old code + `--no-wta-fallback`, seed 0/1/2), val 1,697 states ×3 seeds. policy가 고른 pair의 action으로 실제로 쳐서 clean pocket 비율을 재고, 같은 모델의 action을 **label pair(candidate 0)** 로 고정했을 때(oracle selection)와 비교.

| 남은 공 | 유효 pair | top-1이 label pair | top-1 prob / 2nd prob | top-1 pair coarse feasible | clean (policy 선택) | clean (label pair, 모델 action) | selection 손실 |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 95.4% | 0.75 / 0.21 | 91.6% | 71.8% | 74.1% | 2.3%p |
| **2** | 12 | 83.0% | **0.49 / 0.30** | 91.2% | **60.4%** | **70.2%** | **9.8%p** |
| 3 | 18 | 82.4% | 0.38 / 0.24 | 92.9% | 74.4% | 79.3% | 4.9%p |

- 2개 state의 격차는 **둘 다**에서 온다: (a) label pair를 줘도 70.2%(1개 74.1%, 3개 79.3%) → regression 약점, (b) policy가 고르면 9.8%p를 더 잃는다(1개 2.3, 3개 4.9) → selection 약점. 이전에 "selection은 변경과 무관하게 91% feasible"로 selection을 배제한 것은 전체 평균이었고, 공 개수별로는 2개 state에서 selection 손실이 가장 크다.
- 2개 state는 top-1 prob 0.49 / 2nd 0.30으로 가장 애매하다(두 공 사이에서 결정 못 함). top-1이 label 밖일 때(17%) 그 pair head는 그 state에 label action을 본 적이 없다.
- 미검증: ①(factorized, ball head 먼저)이 2개 state의 selection 손실을 줄이는지(해당 run의 모델이 있어 학습 없이 확인 가능).

**① / ②의 공 개수별 확인 (2026-10-11, 학습 없음, seed 0/1/2 합침, 같은 코드 control 포함)**:

| model | 남은 공 | clean (policy 선택) | clean (label pair) | selection 손실 | top-1이 label pair |
|---|---|---|---|---|---|
| control | 1 / **2** / 3 | 71.6 / **57.8** / 74.0% | 74.1 / 68.5 / 79.4% | 2.4 / **10.7** / 5.5%p | 95.4 / 82.5 / 81.7% |
| ① factorized | 1 / **2** / 3 | 69.6 / **60.0** / 73.6% | 72.2 / 69.4 / 80.3% | 2.6 / **9.4** / 6.7%p | 95.6 / 83.8 / 79.6% |
| ② made selection target | 1 / **2** / 3 | 70.3 / **60.2** / 71.1% | 72.1 / 68.8 / 78.0% | 1.8 / **8.6** / 6.8%p | 95.5 / 84.9 / 79.5% |

- 2개 state(n=829)에서 ①②는 control보다 +2%p(57.8 → 60.0 / 60.2), selection 손실 −1~2%p로 방향은 개선이지만 표본 오차(~±1.7%p)와 비슷한 크기라 효과로 읽기 어렵다. 1개/3개 state는 같거나 약간 나빠졌다.
- **결론**: ①②는 2개 state의 selection 약점(손실 ~9~11%p)과 regression 약점(label pair 성공 68~69%, 1개 72~74% / 3개 78~80%)을 해결하지 못한다. 구조/target 변경으로는 안 움직인다.

### label이 갈려서(flip) gradient가 상쇄되나? (2026-10-11, 학습 없음)

**가설(사용자)**: 2개 state에서 label이 이쪽/저쪽으로 갈려 loss를 반대 방향으로 받아 한쪽으로 못 배운다.

**1. pair 선택 label의 우열**: 남은 공 1/2/3개별로 label pair 2개(candidate 0/1)의 다른 공 비율 0% / 77% / 80%, coarse 성공 횟수 차이 ≤2/20(우열 없음) 25 / 24 / 28%, candidate 0이 argmax(pair_made)인 비율 95 / 88 / 86%. 2개 state가 3개 state보다 더 갈리지 않는다.

**2. label speed가 두 모드**: 모든 공 개수에서 label speed ≈1.0 m/s 76~77%, ≥2 m/s 23~24%(중간 값 거의 없음). control 모델이 label pair에 내는 speed: 1.1~1.4 32%, **1.4~2.0 63%**, 1.0 m/s 0% → 지배 모드(1.0)를 내지 않고 두 모드 사이로 평균을 낸다(MSE regression의 hedging). 공 개수별 차이는 없음(모드 사이 68 / 71 / 68%).

**3. hedging이 성공을 해치는가**: 정책 선택 action의 실제 clean 성공률 — speed ≤1.4 69.9%, 1.4~2.6 71.1%. 해치지 않는다(speed 창이 넓음). 2개 state만: ≤1.4 50.4%, 1.4~2.6 61.3%(비율 32 / 68%).

**결론**: label flip(speed 두 모드)은 실제로 있고 모델은 평균으로 hedge하지만, 모든 공 개수에서 같고 성공률에도 영향이 없어 **2개 state의 약점을 설명하지 못한다**. 같은 speed 모드로 맞춰도 2개 state의 angle 정확도는 낮다(label low·모델 low일 때 within 0.2°: 1개 84.4%, 2개 47.5%, 3개 66.3%). 미확인: 2개 state에서 angle label 자체가 flip하는지(같은 geometry의 비슷한 state끼리 label angle이 갈리는지).

### 남은 공 2개 실패 샷 분류와 시각화 (2026-10-11, 학습 없음)

control 모델(`mod-gla`) seed 0/1/2, val 2개 state 829개에서 policy가 실제로 쳐서 실패한 350건(42.2%)을 분류:

| 실패 유형 | 건수 | 비율 |
|---|---|---|
| label에 없는 pair를 골랐다 | 111 | 32% |
| label pair를 골랐고 각도 오차 ≥ 0.5° | 113 | 32% |
| label pair를 골랐고 scratch | 56 | 16% |
| label pair를 골랐고 각도 오차 0.2~0.5° | 53 | 15% |
| label pair, 각도 < 0.2°(speed 등) | 17 | 5% |

- policy가 **label pair를 고른 684건은 65%, label 밖 pair를 고른 145건은 23%만 성공**. label 밖 pair 실패 111건 중 coarse scan에서 그 pair가 한 번이라도 들어간 것은 54%(평균 성공 비율 0.14), 전혀 안 된 것이 46% → 가능한 pair를 label이 빠뜨려서가 아니라 주로 **되기 어렵거나 안 되는 pair를 고른 것**.
- label pair를 골라 성공한 샷의 각도 오차 중앙값 0.16°, 실패한 샷(scratch 제외)은 0.65°.
- 그림(`scratchpad/plots/fail_angle.png`, `fail_pair.png`, `ok.png`; policy 빨강 실선 / label 청록 점선): 각도 실패는 대부분 (a) cue 공이 겨냥한 공에 매우 가깝거나 (b) 겨냥한 공이 포켓에 매우 가까운 샷이고, 두 선이 거의 겹쳐 보여도 0.5~3°의 차이가 실패로 이어진다. label 밖 pair 실패는 label과 완전히 다른 공/포켓을 고른 경우이고 3건에 1건은 scratch.

**1개/3개 state와 같은 분류 (control, seed 0/1/2, 전체 state 대비 %)**:

| 실패 유형 (전체 state 대비) | 남은 공 1 | **2** | 3 |
|---|---|---|---|
| 실패율 | 28.4% | **42.2%** | 26.0% |
| label 밖 pair를 골라 실패 | 3.2% | 13.4% | 10.6% |
| label pair, 각도 오차 ≥ 0.5° | 1.2% | **13.6%** | 3.1% |
| label pair, 각도 0.2~0.5° | 6.3% | 6.4% | 4.0% |
| label pair, scratch | 9.1% | 6.8% | 6.9% |
| label pair, 각도 < 0.2° | 5.6% | 2.1% | 1.4% |
| label 없음 | 2.9% | 0 | 0 |

| 지표 | 1 | 2 | 3 |
|---|---|---|---|
| label 밖 pair를 고른 비율 | 4.4% | 17.5% | 18.3% |
| 성공률: label pair / label 밖 pair | 76.0 / 27.6% | 65.1 / **23.4%** | 81.1 / **42.0%** |
| 같은 pair의 각도 오차 중앙값: 성공 / 실패 | 0.09° / 0.22° | **0.16° / 0.65°** | 0.11° / 0.37° |
| label 밖 pair 실패 중 coarse로 가능했던 pair | 27% | 54% | 50% |

- **공통(2개만의 문제 아님)**: label 밖 pair를 고르는 비율(2개 17.5% ≈ 3개 18.3%), scratch(6.8% ≈ 6.9%), label 밖 pair가 가능한 pair였던 비율(54% ≈ 50%).
- **2개 state에서만 두드러진 것**: (1) 같은 pair를 골랐을 때 각도 오차가 큰 실패(≥0.5°)가 전체의 13.6%(1개 1.2%, 3개 3.1%), 각도 오차 중앙값도 성공 0.16°/실패 0.65°로 1·3개(0.09~0.11°/0.22~0.37°)의 1.5~3배; (2) label 밖 pair를 골라도 3개 state는 42%가 성공하지만(다른 공이 맞아 들어가는 효과 추정) 2개는 23%. 이전의 "선택이 2개 state의 주 약점"은 label 밖 pair 선택 빈도로는 3개와 같고, 선택이 틀렸을 때 구제되지 않는 점이 다르다.

**타겟볼이 아닌 공이 들어가나? (control, seed 0/1/2, val 전체 state, 실제로 쳐서 event log 확인; 타겟볼 = policy가 고른 pair의 공)**:

| 남은 공 | n | 타겟볼 들어감(scratch 없음) | 타겟볼은 안 들어가고 다른 공만 | 둘 다 | 아무것도 안 들어감 | scratch |
|---|---|---|---|---|---|---|
| 1 | 1971 | 71.6% | 0% | 0% | 15.8% | 12.6% |
| **2** | 829 | 56.3% | 1.4% | 0.7% | **30.9%** | 11.3% |
| 3 | 2250 | 73.2% | 0.8% | 0.8% | 16.0% | 10.0% |

- clean 성공 중 **타겟볼이 들어간 비율 97.5~100%**(2개 97.5%, 3개 98.9%). 타겟볼이 아닌 공이 들어가는 일은 ~1%뿐이고, 의도한 포켓이 아닌 다른 포켓으로 타겟볼이 들어가는 것도 0.1~0.8%. **"3개 state는 공이 흩어져 다른 공이 들어가 구제된다"는 앞의 추정은 틀렸다.**
- 2개 state의 차이는 "아무것도 안 들어감"이 31%로 1·3개(16%)의 2배, scratch는 세 경우 모두 10~13%. 즉 2개 state의 격차는 거의 전부 타겟볼이 안 들어가는 것에서 나온다.
- 앞에서 label 밖 pair를 골랐을 때 성공률이 3개 42% vs 2개 23%였던 이유는 scatter가 아니라, 3개 state에는 label에 없지만 들어가는 pair가 더 많기 때문일 가능성(미검증).

### 2개 state에서 각도가 유독 크게 틀리는 이유 조사 (2026-10-11, 학습 없음)

control(`mod-gla`) seed 0/1/2, val, label pair(candidate 0) 기준.

| 확인 | 결과 |
|---|---|
| **목표가 더 어려운가?** (teacher의 성공 창 폭) | 아니다 — 창 폭 중앙값 1개 0.64° / 2개 0.96° / 3개 0.92°(스캔 범위 ±0.5°에서 잘림). 오히려 1개가 가장 좁다 |
| **모델 오차** | 각도 오차 중앙값 0.11° / **0.24°** / 0.13°, 오차 > 창 반폭(= 맞히기 실패 예상) 16.2% / **37.0%** / 15.6% → 2개 state는 목표가 같은데 모델이 2배 부정확 |
| **label이 불규칙한가?** (label − 순수 기하 조준각, speed ≤1.5 m/s) | 아니다 — |잔차| 중앙값 0.040° / 0.000° / 0.020°, p90 0.16 / 0.14 / 0.12°. label은 모든 공 개수에서 기하 조준각의 ±0.05° 안이고 2개 state가 오히려 가장 규칙적 |
| 큰 delta_angle 샷의 비중이 원인인가? | 일부만: 모든 공 개수에서 label \|delta\|가 클수록(기준 공이 아닌 먼 공을 겨눌수록) 오차가 커지고(3개: 0.09° → 0.14° → 0.30° → 0.79°), 2개 state에는 \|delta\| > 90° 샷이 12.4%(3개 2.7%)로 많다. 하지만 3개의 delta 구성을 적용해도 2개 state 정확도는 42.7% → 45%뿐이고, 같은 delta 구간 안에서도 2개가 나쁘다(5~30°: 33% vs 65%) |
| 먼 공을 겨눌 때 | 2개 0.54° / within 20% vs 3개 0.23° / 45%. 가장 가까운 공을 겨눌 때도 0.15° / 59% vs 0.10° / 80% |

- 지금까지 기각: 일반화(train도 나쁨), 데이터 양(3.7배 추가에도 불변), 포켓된 공 slot 표현, label 불일치·불규칙, 방해 공, 창 폭/난이도, speed hedging, selection, scatter 구제.
- 남은 가설(미검증): **이웃 공이 정확히 1개인 regime에서 ball message(sum)가 pair embedding에 섞여 각도 계산을 방해한다.** 각도는 cue–공–포켓 기하만으로 결정되는데(label이 ±0.05°로 기하와 일치) 현재 action head는 이웃 정보가 섞인 `e_ij`에서 나온다. 검증: action head 입력에서 이웃 message를 빼고(selection head는 유지) 2개 state의 각도 오차가 줄어드는지.

### 각도 head에서 이웃 message 빼기 (2026-10-11 계획)

**가설**: 정답 각도는 cue–겨눈 공–포켓 기하만으로 결정된다(label이 순수 기하 조준각의 ±0.05° 안). 그런데 현재 action head는 이웃 공의 ball message(sum)가 섞인 `e_ij`에서 나온다. 이웃이 정확히 1개인 2개 state에서 이 섞임이 각도 계산을 방해해 오차가 2배(0.24° vs 0.13°)라는 가설.

**변경 (`--geom-angle`)**: angle 출력만 이웃 message 없이 계산한다 — ball embedding에 message 대신 0을 넣어(`h⁰ = ball_mlp([ball_feat, 0])`) pair MLP를 한 번 더 통과시킨 `e⁰_ij`에서 별도 Linear head로 각도를 낸다. speed head와 selection head는 기존 그대로(이웃 정보 사용). 파라미터는 Linear(H, 1) 하나만 늘고 pair MLP/ball MLP는 공유.

**실험**: 새 baseline 설정(`--pair-supervision --no-wta-fallback`, 나머지 동일), seed 0/1/2. 같은 batch에 control(`--geom-angle` 없음)도 다시 돌려 재현성과 같은 코드 비교를 확인(기존 4.14 ± 0.14와 같아야 함).
**지표**: policy alone shots/game, success; 남은 공 1/2/3개별 각도 오차(label pair 기준 중앙값, within 0.2°)와 실제 clean pocket 비율.
**판단**: 2개 state의 각도 오차가 control(0.24°)에서 1·3개 수준(0.11~0.13°)으로 줄면 이웃 message 간섭이 원인. 2개 state만 그대로고 1·3개만 변하면 원인 아님(각도가 이웃 정보 없이도 같은 정확도) → 모델 구조가 아니라 optimization/표현 쪽.

**결과: `--geom-angle` (2026-10-11, seed 0/1/2, control 재실행 포함): 원인이 아니다.**

| | shots/game | success | oracle within 0.2° | 각도 오차 중앙값 1 / **2** / 3개 | within 0.2° 1 / **2** / 3개 | clean 성공 1 / **2** / 3개 |
|---|---|---|---|---|---|---|
| control (재실행, 이전 4.14와 소수점까지 동일) | 4.28 / 4.15 / 4.00 → 4.14 ± 0.14 | 71.9% | 59.9% | 0.112 / **0.230** / 0.125° | 75.1 / **46.2** / 68.4% | 71.8 / **60.4** / 74.4% |
| geom-angle (이웃 message 없이 각도) | 3.94 / 4.43 / 3.90 → 4.09 ± 0.29 | 73.0% | 59.3% | 0.118 / **0.215** / 0.137° | 71.6 / **48.4** / 66.4% | 66.2 / **61.4** / 75.7% |

- 이웃 정보 없이 각도를 내도 정확도가 전혀 달라지지 않고(전체 within 59.9% → 59.3%), 2개 state의 각도 오차도 0.230° → 0.215°로 1·3개(0.12~0.14°)와의 격차가 그대로다. **이웃 message 간섭 가설 기각.** 각도는 이웃 정보 없는 기하만으로도 같은 정확도로 계산되는데 2개 state에서만 2배 부정확하다 → 모델 구조의 정보 경로가 아니라 2개 state 입력 분포/최적화 쪽 문제.
- 지금까지 기각: 일반화, 데이터 양, slot 표현, label 불일치·불규칙, 방해 공, 창 폭, speed hedging, selection, scatter, 이웃 message.

**결과: look-ahead 2단계 (action-conditional made head, 2026-10-11, seed 0/1/2)**

verifier 품질(val states의 teacher sims, 3 seeds): AUC **0.957 / 0.958 / 0.958**(양성 3.7%). state-level coarse 성공률(action을 모름)은 0.90, **가능한 pair(pair_made>0)만** 보면 0.95 vs 0.80 — action을 보는 verifier가 창 안/밖을 잘 구분한다.

| inference (같은 모델, 학습 그대로) | seed 0 / 1 / 2 | 평균 shots/game | select 대비 |
|---|---|---|---|
| select(기본) | 4.28 / 4.14 / 3.89 | 4.11 ± 0.20 | – |
| verify(P(made)로 pair 재선택) | 4.19 / 4.05 / 3.91 | 4.05 ± 0.14 | −0.06 |
| refine(고른 pair 주변 ±0.3° × speed ×{0.8,1,1.25} 27후보 중 P(made) 최대) | 4.19 / 4.11 / 3.80 | 4.03 ± 0.21 | −0.08 |
| **verify + refine** | **4.06 / 4.02 / 3.73** | **3.94 ± 0.18** | **−0.17** (seed별 −0.22 / −0.12 / −0.16) |

- made2 head를 학습시켜도 기본 inference는 그대로(4.11 vs 4.14) — 해롭지 않다. baseline(made2 없음) 4.14 → 3.94(−0.20).
- 같은 모델·같은 평가에서 inference만 바꾼 비교라 seed별 차이가 3개 모두 개선이지만 크기가 작다(−4%, 평가 500 games의 잡음 ~0.05~0.08 shots 수준과 비교해 약 2~3배). 각도 오차를 줄이는 직접적 효과는 refine 후보가 27개로 성기고(각도 9개), head가 0.1° 단위 차이를 구분하는 정도에 달려 있다.
- 지금까지 시도한 것 중 전체 수치를 움직인 것은 이것이 처음(WTA fallback 제거 4.51 → 4.14 제외).

**2단계 inference를 남은 공 개수별로 (2026-10-11, made2 모델 seed 0/1/2, val 합침, 실제로 쳐서 clean pocket 비율)**:

| 방식 | 남은 공 1 | **2** | 3 | 전체 |
|---|---|---|---|---|
| select (기본) | 71.0% | 58.7% | 74.8% | 70.7% |
| verify | 71.3% | 62.0% | 76.3% | 72.0% |
| refine | 74.1% | 61.5% | 75.3% | 72.6% |
| **verify + refine** | **74.6%** | **64.8%** | **77.6%** | **74.3%** |

select 대비 개선(seed별 0 / 1 / 2): verify+refine 1개 +3.6%p(+3.7, +4.0, +3.2), **2개 +6.0%p(+7.5, +6.4, +4.1)**, 3개 +2.8%p(+3.1, +2.4, +2.8); verify 1개 +0.3 / 2개 +3.3 / 3개 +1.5; refine 1개 +3.1 / 2개 +2.8 / 3개 +0.4.

- verify+refine은 **모든 공 개수, 모든 seed에서 개선**(전체 +3.6%p). **2개 state에서 가장 크다**(+6.0%p).
- 기여가 다르다: **verify(pair 재선택)는 선택이 여러 갈래인 2·3개 state에서 효과**(+3.3 / +1.5), 선택이 거의 없는 1개 state에서는 없음(+0.3). **refine(action 미세조정)는 각도 정밀도 쪽이라 1·2개 state에서 효과**(+3.1 / +2.8), 3개에서는 거의 없음(+0.4). 둘은 서로 다른 약점을 겨냥하고 합치면 더해진다(2개: 3.3 + 2.8 ≈ 6.0).
- 2개 state의 격차는 줄었지만 남아 있다: 1개와의 차이 12.3 → 9.8%p, 3개와의 차이 16.1 → 12.8%p.

### refine 후보 격자를 촘촘하게 (2026-10-11 계획, 학습 없음)

**왜**: refine은 각도 9개(±0.3°, 간격 0.05~0.1°) × speed 3개 = 27후보만 본다. 후보가 성기거나 범위가 좁아서 이득이 제한됐을 수 있다.
**방법**: made2 모델(seed 0/1/2)의 inference 격자만 바꾼다(speed {0.8, 1, 1.25}× 고정). 각도 격자: base(27후보) / 0.02° 간격 ±0.3° / 0.02° ±0.6° / 0.02° ±1.0° / 0.01° ±0.3°. 방식은 verify+refine. 지표: val에서 실제로 쳐서 clean pocket 비율(공 개수별).
**판단**: 촘촘할수록/넓을수록 오르면 후보 부족이 한계였던 것. 오르다 꺾이면(head가 작은 차이를 구분 못 하거나 범위 밖 후보에서 과신) 그 지점이 head의 해상도·신뢰 한계.

**결과 (2026-10-11): 각도 격자를 촘촘하게/넓게 해도 거의 오르지 않는다 — 한계는 후보 수가 아니다.**

| verify+refine, 각도 격자 (× speed 3) | 남은 공 1 | 2 | 3 | 전체 |
|---|---|---|---|---|
| select (refine 없음) | 71.0% | 58.7% | 74.8% | 70.7% |
| base: 9각도 ±0.3° | 74.6% | 64.8% | 77.6% | 74.3% |
| 0.02° 간격 ±0.3° (31) | 75.1% | 65.0% | 77.6% | 74.6% |
| 0.01° 간격 ±0.3° (61) | 74.8% | 65.3% | 77.7% | 74.6% |
| 0.02° 간격 ±0.6° (61) | 75.1% | 64.8% | 77.6% | 74.5% |
| 0.02° 간격 ±1.0° (101) | 75.1% | 64.8% | 77.6% | 74.5% |

- 후보를 27 → 303개로 늘리거나 범위를 ±1°로 넓혀도 전체 +0.2~0.3%p(74.3 → 74.5~74.6)뿐이다. 간격 0.05° → 0.01°도, 범위 ±0.3° → ±1.0°도 효과가 없다.
- 따라서 refine의 이득(+3.6%p)은 이미 base 격자에서 거의 다 얻었고, 더 얻으려면 후보를 늘릴 게 아니라 **made head 자체의 해상도/정확도**를 올리거나 다른 축(speed 격자는 3개뿐)을 봐야 한다.

### refine: speed 격자 확대 + made head 해상도 측정 (2026-10-11 계획, 학습 없음)

**1. speed 격자**: 각도 격자는 포화(0.02° ±0.3°면 충분)했으니 각도를 31개로 고정하고 speed만 바꾼다. 기존은 기준 speed의 ×{0.8, 1, 1.25} 3개. 후보: (a) ×{0.7…1.4} 7개, (b) ×0.5~2.0 로그 간격 13개, (c) 기준 speed와 무관한 절대 speed {0.75, 1, 1.25, 1.5, 2, 3, 5} m/s (teacher label은 1.0 m/s가 76%인데 모델은 1.4~2.0 m/s로 hedge하므로). 지표: 공 개수별 실제 clean 성공률.
**2. made head의 해상도**: (a) val 시뮬레이션에서 AUC·calibration, (b) 직접 측정 — val state 450개(seed별 150)에서 label action의 각도를 ±0.6°(0.05° 간격 25개) 흔들며 실제로 쳐서 성공 창을 구하고, head의 P(made)가 그 창 안/밖을 구분하는지(AUC), head의 argmax가 진짜 창 안에 드는 비율, head가 예측한 창의 폭 vs 실제 폭.

**결과 1: speed 격자 (각도 0.02° ±0.3° 31개 고정, verify+refine, val 실제 clean 성공률, 3 seed 합침)**:

| speed 격자 | 남은 공 1 | 2 | 3 | 전체 |
|---|---|---|---|---|
| select (refine 없음) | 71.0% | 58.7% | 74.8% | 70.7% |
| 기존 ×{0.8, 1, 1.25} (3) | 75.1% | 65.0% | 77.6% | 74.6% |
| ×0.7~1.4 (7) | 76.8% | 65.7% | 79.2% | 76.0% |
| ×0.5~2.0 로그 (13) | 77.7% | 65.0% | 78.2% | 75.8% |
| **절대 {0.75, 1, 1.25, 1.5, 2, 3, 5} m/s (7)** | **78.8%** | **66.8%** | **80.1%** | **77.4%** |

- 각도와 달리 speed 격자는 효과가 있다: 3 → 7개로 +1.4%p, 모델 speed와 무관한 절대 격자는 +2.8%p(select 대비 +6.7%p). 모델이 speed를 1.4~2.0 m/s로 hedge하므로 기준 speed ×배율보다 teacher가 실제로 쓰는 값(1.0 / 2~5 m/s)을 직접 후보로 두는 것이 낫다.
- **실제 게임(정책 단독 500 games, eval seed 0)에서는 이득이 작다**: select 4.11 ± 0.20 / 기존 격자 3.94 ± 0.18 / **각도 31 × 절대 speed 7: 3.90 ± 0.21** (success 72.5 / 75.9 / 76.7%). val(teacher states)의 +2.8%p가 게임에서는 +0.8%p, −0.04 shots로 줄었다(평가 잡음 ~0.05~0.08 shots 수준). 게임 중 state 분포가 teacher states와 다르기 때문으로 추정(미검증).

**결과 2: made head 정확도**:
- (a) val 시뮬레이션 160만 건: AUC 0.958(양성 3.7%). calibration이 좋다 — 예측 0.4~0.6 → 실제 50.8%, 0.6~0.8 → 69.9%, 0.8~0.95 → 83.2%, 0.95 이상 → 92.4%, 0.05 미만 → 0.5%.
- (b) refine이 실제로 쓰는 영역(label 각도 ±0.6°, 0.05° 간격, 450 states를 실제로 쳐서 성공 창 확인): **AUC 0.841**로 전체(0.958)보다 훨씬 낮다 — 쉬운 샷을 구분하는 것은 잘하지만 창 안의 작은 차이는 덜 구분한다. head의 argmax가 실제 성공 창 안에 드는 비율 **81.4%**(나머지 18.6%는 창 밖 후보를 고른다). 창 폭은 정확하다(실제 중앙값 0.70° vs head의 P>0.5 구간 0.75°) → 문제는 폭이 아니라 위치.

### made head가 부정확한 이유: sample 다양성인가 interpolation인가 (2026-10-11, 학습 없음, seed 0 모델)

**방법**: 해상도 측정(label 각도 ±0.6°를 0.05° 간격 25점으로 실제로 쳐서 성공 창 확인, head의 P(made)와 비교)을 **train state 200개 vs val state 200개**에서 같이 측정. 추가로 각 스캔 점 근처(각도 ±0.05°, speed ±10%)에 있는 같은 state의 학습 sims 수별로 정확도를 나눔.

| | AUC | Brier | argmax가 실제 창 안 |
|---|---|---|---|
| train states | 0.831 | 0.160 | 77.1% |
| val states | 0.846 | 0.163 | 78.8% |

| 스캔 점 근처 학습 sims 수 | 점 수 | AUC | Brier |
|---|---|---|---|
| 0 | 1,344 | 0.780 | 0.183 |
| 1~2 | 4,156 | 0.838 | 0.161 |
| 3~7 | 4,500 | 0.842 | 0.156 |

- **train과 val이 같다**(AUC 0.831 vs 0.846). 일반화 격차가 없으므로 state 다양성 부족이 아니다 — 학습한 state에서도 창의 위치를 못 맞힌다(**underfitting**).
- **sims 밀도는 약하게만 영향**: 근처 sims가 하나도 없는 점은 AUC 0.78이지만, 1~2개든 3~7개든 0.84로 같다. 창 가장자리 샘플을 늘려도 이득은 제한적이다.
- 따라서 "sample이 다양하지 못함"이 아니라 **head가 창의 위치(= 정답 각도가 pair 제안에서 얼마나 떨어져 있나)를 0.1° 단위로 표현·보간하지 못한다**. 이 head는 `e_ij`와 (action − pair 제안)만 받는데, 창 위치는 pair 제안의 오차(중앙값 0.15°)에 의존한다 → head가 `e_ij`에서 그 오차를 스스로 추정해야 하고 이는 pair head의 각도 정밀도와 같은 병목이다.
- 검증 제안: (a) 진단용으로 head에 기하 정보(조준각과 순수 기하 조준각의 차)를 직접 넣어 AUC가 오르면 병목은 계산 정밀도, (b) 오르지 않으면 head 용량/학습(offset의 고주파 표현, 더 긴 학습).

### made head 진단: 순수 기하 조준각과의 차이를 입력으로 주기 (2026-10-11 계획)

**질문**: made head가 창의 위치를 못 맞히는 이유가 "각도를 0.1° 단위로 계산하는 정밀도 병목"인가, 아니면 head의 용량·학습 문제인가.
**변경 (`--made2-geom`, 진단 전용)**: made head 입력에 `g_ij = wrap(action 각도 − 순수 기하 조준각 aim_ij) / 0.2°`를 추가한다. `aim_ij`는 relative 좌표(cue가 원점, 기준 공이 +x)에서 ghost ball 지점 `b − 2R·û(p−b)`로의 방향. label이 이 값의 ±0.05° 안이므로(experiments.md "label이 불규칙한가") 창이 `g ≈ 0` 근처에 있다 — head가 해야 할 일이 1차원 함수(창 폭 학습)로 줄어든다. 이 기하 공식은 진단에만 쓰고 policy/최종 모델에는 넣지 않는다(bitter lesson 방침).
**실험**: made2 head, seed 0/1/2, 나머지 동일(이전 `m2w1` run이 같은 코드 경로의 control). 지표: val 시뮬레이션 AUC, label ±0.6° 스캔의 AUC·argmax가 창 안에 드는 비율, 공 개수별 verify+refine 실제 clean 성공률.
**판단**: 스캔 AUC가 0.84 → 0.95 이상으로 오르면 병목은 head가 각도 정밀도를 계산하지 못하는 것(→ offset의 정밀한 표현이 필요). 거의 안 오르면 head 용량/학습 쪽.

**결과 (2026-10-11): made head 병목은 각도 정밀도 계산이다 — 기하 조준각 차이를 입력으로 주면 창 위치를 거의 맞히고 성능이 크게 오른다.**

(seed 0/1/2, 이전 `m2w1` run이 같은 코드 경로의 control)

| | val 시뮬레이션 AUC | 스캔 AUC (label ±0.6°) | argmax가 실제 창 안 | Brier |
|---|---|---|---|---|
| control | 0.957 | 0.857 | 81.0% | 0.156 |
| **+ 기하 입력 g** | 0.961 | **0.915** | **95.1%** | **0.121** |

| 실제 clean 성공률 (val, 3 seed) | 남은 공 1 | **2** | 3 | 전체 |
|---|---|---|---|---|
| control, select (refine 없음) | 71.0% | 58.7% | 74.8% | 70.7% |
| control, verify+refine 기존 격자 | 74.6% | 64.8% | 77.6% | 74.3% |
| control, verify+refine 각도 31 × 절대 speed 7 | 78.8% | 66.8% | 80.1% | 77.4% |
| g 입력, select | 70.3% | 58.9% | 74.2% | 70.1% |
| g 입력, verify+refine 기존 격자 | 79.7% | **77.0%** | 81.8% | 80.2% |
| **g 입력, verify+refine 각도 31 × 절대 speed 7** | **82.3%** | **76.8%** | **85.0%** | **82.6%** |

| 정책 단독 500 games (eval seed 0) | shots/game | success |
|---|---|---|
| baseline (fallback 끔) | 4.14 ± 0.14 | 71.9% |
| made2, select | 4.11 ± 0.20 | 72.5% |
| made2, verify+refine (각도 31 × 절대 speed 7) | 3.90 ± 0.21 | 76.7% |
| **g 입력 + verify+refine (각도 31 × 절대 speed 7)** | **3.59 / 3.45 / 3.35 → 3.46 ± 0.12** | **86.4%** |

- g를 주자 head의 argmax가 실제 성공 창 안에 드는 비율이 81% → 95%로 오르고(스캔 AUC는 0.857 → 0.915, 사전에 정한 기준 0.95에는 못 미침), 실제 성공이 크게 올랐다(val 전체 77.4% → 82.6%, **2개 state 66.8% → 76.8%**로 1개(82.3%)·3개(85.0%)와의 격차가 대폭 줄었다). select(refine 없음)는 70.1%로 그대로 → g는 made head에만 들어가고 pair head는 안 바뀐다.
- **게임에서 3.46 shots/game**(baseline 4.14, g 없는 2단계 3.90). teacher(robust 2.98, geom 2.24)에 가까워졌다.
- 해석: "head가 창 위치를 모른다"가 병목이었고, 위치를 알려주는 정보(순수 기하 조준각과의 차이)만 주면 head는 창 폭·경계를 잘 학습한다. 즉 문제는 head 용량이 아니라 **`e_ij`가 각도를 0.1° 정밀도로 담지 못하는 것**(= 앞의 pair head 각도 오차와 같은 뿌리).
- 주의: g는 ghost-ball 기하 공식으로 계산한 값이다. 진단용이라 정책에 넣지 않기로 했던 값을 쓴 결과이므로 "최종 성능"으로 보면 안 되고, 정밀도 병목이 해결되면 어디까지 갈 수 있는지의 상한에 가깝다. 다음 질문: 공식 없이(학습으로) 같은 정밀도를 얻을 수 있는가 — encoder 용량 확대, 좌표의 주파수(sin/cos) 표현.

### encoder 용량 키우기 (2026-10-11 계획)

**질문**: made head 진단에서 병목이 `e_ij`가 각도를 0.1° 정밀도로 담지 못하는 것으로 확인됐다(기하 입력 g를 주면 argmax 창 안 81% → 95%). 공식 없이 encoder 용량만 키워도 이 정밀도가 오르는가.
**변경**: (a) 폭: `--gnn-hidden 256`(기존 128), (b) 깊이: encoder MLP(ball message / ball / pair)의 hidden layer를 2 → 4(`--gnn-layers 4`). head는 그대로. 둘을 따로 본다(함께는 결과를 보고).
**실험**: made2 head 포함 설정(`--pair-supervision --no-wta-fallback --made2-head`), seed 0/1/2. control은 이전 `m2w1` run(같은 설정, hidden 128, layers 2).
**지표**: ① pair head 각도 오차(공 개수별 중앙값·within 0.2°) ② made head: label ±0.6° 스캔 AUC·argmax가 창 안 비율 ③ 실제 clean 성공률(select / verify+refine, 공 개수별) ④ 정책 단독 500 games shots/game.
**판단**: 각도 오차가 0.15°에서 의미 있게 줄고 스캔 argmax가 81%에서 오르면 용량이 병목(→ 더 키움). 거의 안 변하면 용량이 아니라 표현(sin/cos 좌표)이나 학습 쪽.

#### encoder 용량 실험 — 결과별 판단표 (2026-10-11, 결과를 보기 전에 정해 둠)

**기준점** (3 seed 합침, 같은 val state·같은 측정 방법)

| 지표 | control (`m2w1`, hidden 128, 2층) | 상한 참고 (기하 입력 g, 진단용) |
|---|---|---|
| pair head 각도 오차 중앙값 (전체 / 남은 공 1 / **2** / 3) | 0.153° / 0.112° / **0.230°** / 0.125° | – |
| pair head within 0.2° (전체) | 59.9% | – |
| made head 스캔 argmax가 실제 창 안 | **81.0%** | **95.1%** |
| made head 스캔 AUC | 0.857 | 0.915 |
| 실제 clean 성공률(val) verify+refine(각도 31 × 절대 speed 7): 전체 / 남은 공 2 | 77.4% / 66.8% | 82.6% / 76.8% |
| 정책 단독 500 games shots/game | 3.90 ± 0.21 | 3.46 ± 0.12 |

**회복률** = (실험 스캔 argmax − 81.0) / (95.1 − 81.0). 격차는 14.1%p. 효과는 **seed 3개 모두 control보다 좋고** 평균 차이가 seed 표준편차(각도 오차 ±0.01°, argmax ±2%p 수준) 이상일 때만 인정한다.

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| A | 폭 또는 깊이로 argmax ≥ 88%(회복률 ≥ 50%) **그리고** 각도 오차 ≤ 0.115°(−25%), seed 3개 모두 | 용량이 병목이다 | 폭+깊이를 결합(256 × 4층)해 포화점을 찾고, 같은 지표로 sin/cos 표현과 파라미터 효율을 비교. 학습으로 g 없이 상한에 가까워지면 g(기하 공식)는 쓰지 않는다 |
| B | 한 축(폭 또는 깊이)만 A를 충족 | 그 축이 병목이다 | 그 축만 더 키우고 다른 축은 고정. 결합은 하지 않는다 |
| C | argmax 83~88%, 각도 오차 −10~−25% | 용량이 일부만 기여, 단독으로는 부족 | sin/cos(Fourier) 좌표 표현을 같은 지표로 시험. 용량과 표현을 합친 조건은 C가 확인된 뒤에만 |
| D | argmax < 83%, 각도 오차 ±10% 이내 | **용량이 아니다** | (1) 좌표 sin/cos 표현 (2) 학습 쪽: 더 긴 학습·lr·EMA/마지막 epoch checkpoint(roadmap #9) (3) loss 스케일(`--angle-scale-deg` 0.1). 순서대로 하나씩 |
| E | train 지표는 오르는데 val 지표(스캔·각도 오차)가 내려감 | 과적합 | 데이터 양/정규화. 용량 확대는 중단 |
| F | 평균은 좋아졌지만 seed 3개 중 하나가 control보다 나쁨 | 불확실 | 결론 보류, seed 5개로 반복 후 판단(다음 실험으로 넘어가지 않는다) |
| G | 학습이 불안정(loss spike, 수렴 실패) | 큰 모델에서 최적화 문제 | lr를 낮춰(3e-4) 같은 설정으로 재시도 후 A~E 적용 |

**이 실험 이후의 결정 (판단표와 별개)**
- 학습만으로 argmax ≥ 90%에 도달하면 → gradient로 action을 연속 공간에서 조정하는 방법(`∇_a P(made)`)을 시도한다(head의 기울기가 정확할 때만 의미가 있다).
- 용량과 표현을 모두 시도해도 argmax < 90% → **사용자 결정 필요**: (a) ghost-ball 기하 조준각 차이를 made head에만 입력으로 허용(정책 본체는 학습으로 유지, verifier만 기하 사용) vs (b) 학습만으로 계속. 이 결정은 "탐색 + 학습" 방침(bitter lesson)과 충돌하므로 임의로 정하지 않는다.
- 게임(3.90)과 val(teacher state)의 격차: val에서 +2.8%p 이득이 게임에서는 +0.8%p였다. 정밀도 병목이 풀린 뒤에도 이 격차가 남으면 policy가 가는 state에서 다시 label을 받는 방법(DAgger, roadmap #10)을 시도한다.

#### 판단표 기준점 정정 (2026-10-12, 용량 실험 결과를 열어 보기 전에 수정)

위 판단표를 쓸 때 (1) seed 표준편차를 측정 없이 적었고(±0.01°, ±2%p), (2) 각도 오차 기준점에 정의가 다른 두 값을 섞었다(전체 0.153°는 results.json의 oracle 지표 = 실제로 made한 모든 (candidate, pair), 공 개수별 값은 label pair(candidate 0) 기준). 재사용 가능한 평가 도구 `exp16_wm/eval_by_balls.py`로 control(`m2w1`)과 g 진단 run을 같은 방법으로 다시 측정해 기준점을 확정한다. 이 도구는 이전 임시 스크립트의 수치를 그대로 재현했다(스캔 AUC 0.857 / 0.915, argmax 81.0% / 95.1%, 공 개수별 clean 성공률, 게임 3.90 / 3.46).

| 지표 (도구 정의: label pair(candidate 0)의 pair head 출력 vs label) | control | seed별 | 상한 참고 (g 입력) |
|---|---|---|---|
| pair head 각도 오차 중앙값 전체 | **0.127°** | 0.130 / 0.135 / 0.118 (표준편차 0.009°) | 0.136° |
| 〃 공 1 / **2** / 3 | 0.111 / **0.217** / 0.124° | | 0.119 / 0.234 / 0.129° |
| pair head within 0.2° 공 1 / **2** / 3 | 74.1 / **47.5** / 68.8% | | 73.8 / 43.9 / 68.0% |
| made head 스캔 argmax가 창 안 | **81.0%** | 77.6 / 81.4 / 84.2% (**표준편차 3.3%p**) | 95.1% (94.4 / 96.6 / 94.1) |
| made head 스캔 AUC / Brier | 0.857 / 0.156 | | 0.915 / 0.121 |
| clean 성공률 verify+refine 31×7: 공 1 / **2** / 3 / 전체 | 78.8 / **66.8** / 80.1 / 77.4% | 전체 75.2 / 78.7 / 78.4% | 82.3 / 76.8 / 85.0 / 82.6% |
| 게임 shots/game | 3.90 ± 0.21 | 4.09 / 3.94 / 3.68 | 3.46 ± 0.12 |

**정정된 임계값** (회복률 = (argmax − 81.0) / 14.1; "효과 인정" = seed 3개 모두 control보다 좋고 평균 차이가 위 seed 표준편차 이상):
- A: argmax ≥ 88%(회복률 ≥ 50%) **그리고** 각도 오차 ≤ 0.095°(control 0.127°의 −25%)
- B: 한 축만 A 충족
- C: argmax 84~88%(control + 표준편차 1개 이상) 또는 각도 오차 0.095~0.114°(−10~−25%)
- D: argmax < 84% **그리고** 각도 오차 0.114~0.140°(±10% 이내) → 용량이 아니다
- E, F, G: 위 판단표와 같다. 단 F의 "control보다 나쁜 seed"는 seed별 control 값과 짝지어 비교한다(seed 0 argmax 77.6%처럼 control 자체의 seed 편차가 크다).

**결과: encoder 깊이 4 (hidden 128, encoder MLP hidden layer 2 → 4, made head 포함, seed 0/1/2; 2026-10-12)**

`exp16_wm/eval_by_balls.py`, control = `m2w1`. seed별 짝지은 비교. 판단표는 결과를 열기 전에 정한 것(위 "정정" 포함).

| 지표 | control | **깊이 4** | seed별 (control → 깊이 4) |
|---|---|---|---|
| pair head 각도 오차 중앙값 전체 | 0.127° | **0.097°** (−24%) | 0.130→0.101 / 0.135→0.095 / 0.118→0.093 |
| 〃 공 1 / **2** / 3 | 0.111 / 0.217 / 0.124° | 0.085 / **0.158** / 0.095° | |
| pair head within 0.2° 공 1 / **2** / 3 | 74.1 / 47.5 / 68.8% | 84.5 / **57.7** / 79.9% | |
| made head 스캔 argmax가 창 안 | 81.0% | **86.5%** (회복률 39%) | 0.776→0.860 / 0.814→0.881 / 0.842→0.851 |
| made head 스캔 AUC / Brier | 0.857 / 0.156 | 0.880 / 0.148 | |
| val 시뮬레이션 AUC(made head) | 0.957 / 0.958 / 0.958 | 0.958 / 0.956 / 0.955 | 같음 |
| clean 성공률 select: 공 1 / **2** / 3 / 전체 | 71.0 / 58.7 / 74.8 / 70.7% | 75.0 / **64.5** / 77.8 / **74.5%** | 전체 0.676→0.744 / 0.721→0.759 / 0.724→0.733 |
| clean 성공률 verify+refine(31×절대 speed 7) | 78.8 / 66.8 / 80.1 / 77.4% | 82.5 / **74.9** / 82.6 / **81.3%** | 전체 0.752→0.816 / 0.787→0.830 / 0.784→0.793 |
| **게임 shots/game (verify+refine)** | 3.90 ± 0.21 | **3.55 ± 0.10** (success 84.2%) | 4.09→3.52 / 3.94→3.66 / 3.68→3.47 |

**판단표 적용**: 각도 오차 −24%는 A의 기준(≤ 0.095°, −25%)에 0.002° 못 미치고, argmax 86.5%는 A의 기준(≥ 88%)에 못 미쳐 C 구간(84~88%)이다. 사전에 정한 규칙대로라면 **C: 용량이 일부 기여**. 다만 효과는 모든 지표에서 seed 3개 전부 control보다 좋고 평균 차이도 seed 표준편차(각도 0.009°, argmax 3.3%p)보다 크다(각도 0.030°, argmax +5.5%p).

- **실제 성능은 기하 입력 진단에 거의 도달**: 게임 3.55(진단 3.46, control 3.90), 남은 공 2개 state 성공 74.9%(진단 76.8%, control 66.8%). 단 argmax는 진단(95.1%)에 한참 못 미친다 → 이득의 상당 부분이 **made head가 아니라 pair head**에서 온다: refine 없는 select에서도 70.7% → 74.5%(+3.8%p), 각도 오차 −24%.
- 깊이가 pair head의 각도 정밀도를 직접 올린다: 2개 state 0.217° → 0.158°(−27%), within 0.2° 47.5% → 57.7%. 남은 공 2개 state 격차는 줄었지만 남아 있다(0.158° vs 1/3개 0.085 / 0.095°).
- made head의 val 시뮬레이션 AUC는 그대로(0.958 → 0.955~0.958) — 전체 구분력은 이미 포화이고 늘어난 것은 창 안 해상도다.
- 다음: 폭 256 결과를 같은 표로 본 뒤 (1) 폭+깊이 결합 (2) 깊이를 더(6층). 판단표 C의 규칙("용량과 표현을 합친 조건은 C가 확인된 뒤에만")에 따라 sin/cos 표현은 그 다음.

**결과: encoder 폭 256 (hidden 256, 2층, made head 포함, seed 0/1/2; 2026-10-12)**

| 지표 | control | 깊이 4 | **폭 256** |
|---|---|---|---|
| pair head 각도 오차 중앙값 전체 | 0.127° | 0.097° | **0.110°** (−13%) |
| 〃 공 1 / **2** / 3 | 0.111 / 0.217 / 0.124° | 0.085 / 0.158 / 0.095° | 0.095 / **0.198** / 0.106° |
| made head 스캔 argmax가 창 안 | 81.0% | 86.5% | **83.1%** (seed별 0.832 / 0.864 / **0.792**, control 0.776 / 0.814 / 0.842) |
| 스캔 AUC / Brier | 0.857 / 0.156 | 0.880 / 0.148 | 0.879 / 0.144 |
| clean select 전체 | 70.7% | 74.5% | 73.5% |
| clean verify+refine(31×절대 speed 7): 공 1 / **2** / 3 / 전체 | 78.8 / 66.8 / 80.1 / 77.4% | 82.5 / 74.9 / 82.6 / 81.3% | 81.8 / **70.3** / 80.8 / **79.5%** |
| 게임 shots/game | 3.90 ± 0.21 | **3.55 ± 0.10** | 3.84 ± 0.18 (success 77.9%) |
| val 시뮬레이션 AUC | 0.958 | 0.956 | 0.958 |

- 판단표: 각도 오차 −13%는 C 구간(−10~−25%)이지만 argmax 83.1%는 control의 seed 편차(3.3%p) 안이고 seed 2는 control보다 **나쁘다**(0.792 vs 0.842) → F(한 seed가 control보다 나쁨): **폭은 결론 보류**, 효과는 있어도 작다. **깊이 > 폭**: 같은 정밀도 개선을 깊이는 −24%/+5.5%p, 폭은 −13%/+2.1%p로 낸다(폭 256은 파라미터가 더 많은데도).
- 해석: 정밀도 병목은 encoder가 거치는 비선형 변환의 **깊이**(각도를 여러 단계에 걸쳐 계산)에 더 민감하고 단순히 표현 폭을 늘리는 것으로는 덜 풀린다.
- 확인한 인자(results.json): 깊이 4와 폭 256 run 모두 `made2_geom=False`, `geom_angle=False` — **기하 입력은 쓰지 않았고** control과 다른 것은 `gnn_layers`(4) / `gnn_hidden`(256)뿐이다. control(`m2w1`)은 이 옵션들이 추가되기 전 코드로 학습했으므로 현재 코드로 다시 돌려 동일성을 확인한다(`logs/experiments_recheck`).

### encoder 깊이 스케일링: 6층, 8층 (2026-10-12 계획)

**왜**: 깊이 4는 control 대비 pair head 각도 오차 −24%, 창 적중 +5.5%p, 게임 3.90 → 3.55로 모든 지표에서 seed 3개 모두 개선했다(폭 256은 효과가 작고 seed 하나가 control보다 나빠 보류). 같은 축으로 더 가면 더 오르는지, 어디서 포화하는지 본다.
**변경**: `--gnn-layers 6`, `--gnn-layers 8`(encoder MLP hidden layer 수, hidden 128 고정, residual 없음). 나머지는 control과 동일(`--pair-supervision --no-wta-fallback --made2-head`), seed 0/1/2. 폭은 건드리지 않는다(폭은 결론 보류, 깊이 포화 확인 뒤에 결합).
**기준점** (같은 도구 `exp16_wm/eval_by_balls.py`, 3 seed 합침, 괄호는 seed별 표준편차):

| 지표 | control | 깊이 4 (직전 최선) |
|---|---|---|
| pair head 각도 오차 중앙값 전체 / 남은 공 **2** | 0.127° (±0.009) / 0.217° | **0.097°** (±0.004) / **0.158°** |
| made head 스캔 argmax가 창 안 | 81.0% (±3.3%p) | **86.5%** (±1.5%p) |
| clean 성공률 verify+refine 31×7 전체 / 공 **2** | 77.4% / 66.8% | **81.3%** / **74.9%** |
| 게임 shots/game | 3.90 ± 0.21 | **3.55 ± 0.10** |
| 상한 참고(기하 입력 g) argmax / 게임 | – | 95.1% / 3.46 |

**"효과 인정" 규칙**: 깊이 4와 seed별로 짝지어 3개 모두 더 좋고 평균 차이가 control의 seed 표준편차(각도 0.009°, argmax 3.3%p) 이상일 때. 그 미만이면 "깊이 4와 같다"로 본다.

| # | 관측 (6층·8층, 깊이 4 대비) | 해석 | 다음 행동 |
|---|---|---|---|
| A | 각도 오차 ≤ 0.088°(−0.009°) **또는** argmax ≥ 89.8%(+3.3%p), 더 깊을수록 더 좋음 | 깊이가 계속 병목 | 더 깊게(12층) 또는 폭과 결합. 남은 공 2개 state 오차가 1·3개의 1.3배 이내가 되면 2개 state 약점은 용량 문제였다고 결론 |
| B | 6층까지 개선, 8층은 6층과 같음(효과 인정 미만) | 깊이 포화 | 6층을 새 기준으로 고정. 다음은 (1) 폭 256 결합 (2) 좌표 sin/cos 표현 (3) 학습 쪽(긴 학습, lr, EMA checkpoint) 순서대로 |
| C | 6층·8층 모두 깊이 4와 같음 | 4층에서 포화 | 깊이 4를 기준으로 고정, 위 B의 (1)~(3) |
| D | 6층 또는 8층이 깊이 4보다 나쁨(특히 train loss가 높거나 학습 불안정) | 최적화 문제 | residual connection 또는 LayerNorm을 넣어 같은 깊이를 재시도, lr 3e-4. 이 경우 깊이가 병목인지는 아직 모른다 |
| E | 3 seed 중 하나라도 깊이 4보다 나쁨 | 불확실 | 결론 보류, seed 5개로 반복 |

**이 실험 이후의 결정**
- 학습만으로 argmax ≥ 90%(g 진단 95.1%의 회복률 약 70%)에 도달하고 게임이 ≤ 3.5이면 → ghost-ball 기하 입력은 쓰지 않는다(열린 결정 해소). 그 구성에서 `∇_a P(made)` gradient refine(roadmap #15)을 시도한다.
- 깊이·폭·표현을 다 해도 argmax < 90%이고 게임 > 3.5이면 → 사용자 결정(made head에만 기하 입력 허용 여부).
- 비용도 기록한다: 파라미터 수와 epoch당 시간(깊이 4: 5.6초/epoch, 6 runs 병렬).

**control 동일성 확인** (진행 중): 현재 코드로 control(`m2w1` 설정)을 3 seed 다시 돌려(`logs/experiments_recheck`) 이전 control과 `results.json`의 eval/val 수치가 같은지 본다. 같으면 위 비교는 같은 코드 기준, 다르면 모든 비교의 control을 재실행한 값으로 바꾼다.

**control 동일성 확인 결과 (2026-10-12)**: 현재 코드로 control(`--pair-supervision --no-wta-fallback --made2-head`)을 seed 0/1/2 다시 돌려(`logs/experiments_recheck`) 이전 `m2w1`과 비교 — eval(shots/game, success), val(oracle 각도 오차, made2 AUC), best epoch, best val loss **전부 소수점까지 동일**(seed 0: 4.284 / 69.4678% / 0.167174° / 0.957115 / epoch 430 / 2074.09). 깊이·폭·기하 입력 옵션을 추가한 뒤에도 기본 설정의 동작이 변하지 않았으므로 control(`m2w1`)은 현재 코드와 같은 코드 기준이다. 깊이 4·폭 256·깊이 6/8과의 비교는 이 control을 그대로 쓴다.

**결과: encoder 깊이 6, 8 (hidden 128, made head 포함, seed 0/1/2; 2026-10-12)**

변경점은 `gnn_layers` 하나(2 → 4 → 6 → 8)다. `results.json` 인자를 control과 대조해 확인했다(다른 것은 `gnn_layers`와 저장 경로뿐, `made2_geom=False`, `geom_angle=False`). 평가는 `exp16_wm/eval_by_balls.py`(control·깊이 4와 같은 도구, 같은 val state), 깊이 4와 seed별로 짝지은 비교.

| 지표 | control (2층) | **깊이 4** | 깊이 6 | 깊이 8 |
|---|---|---|---|---|
| 파라미터 수 | 168,836 | 267,908 | 366,980 | 466,052 |
| pair head 각도 오차 중앙값 전체 | 0.127° | **0.097°** | 0.113° | 0.117° |
| 〃 seed별 | 0.130 / 0.135 / 0.118 | 0.101 / 0.095 / 0.093 | 0.101 / 0.101 / **0.136** | 0.115 / 0.098 / **0.140** |
| 〃 남은 공 2개 | 0.217° | **0.158°** | 0.169° | 0.189° |
| within 0.2° 남은 공 2개 | 47.5% | **57.7%** | 55.2% | 52.1% |
| made head 스캔 argmax가 창 안 | 81.0% | **86.5%** | 82.5% | 81.9% |
| 〃 seed별 | 0.776 / 0.814 / 0.842 | 0.860 / 0.881 / 0.851 | 0.879 / 0.881 / **0.703** | 0.822 / 0.881 / **0.743** |
| 스캔 AUC / Brier | 0.857 / 0.156 | **0.880 / 0.148** | 0.827 / 0.173 | 0.819 / 0.182 |
| clean 성공률 verify+refine 31×7: 전체 / 공 2개 | 77.4% / 66.8% | **81.3% / 74.9%** | 77.2% / 71.3% | 76.7% / 70.4% |
| **게임 shots/game** (success) | 3.90 ± 0.21 (76.7%) | **3.55 ± 0.10** (84.2%) | 3.77 ± 0.16 (79.6%) | 3.94 ± 0.11 (76.0%) |
| 〃 seed별 | 4.09 / 3.94 / 3.68 | 3.52 / 3.66 / 3.47 | 3.69 / 3.66 / **3.95** | 3.87 / 3.89 / **4.06** |
| 학습 끝 train loss (마지막 20 epoch 평균) seed별 | 18.2 / 18.7 / 16.0 | 16.5 / 16.8 / 15.7 | 16.7 / 17.0 / **25.4** | 17.4 / 18.0 / **20.3** |
| val within 0.2° 최댓값 seed별 | .572 / .582 / .622 | .672 / .683 / .675 | .658 / .656 / .591 | .636 / .652 / .547 |
| 500 epoch 학습 시간 (병렬 6 runs) | 약 970초(3 runs 병렬) | 약 2,650초 | 약 3,530초 | 약 3,990초 |

**판단표 적용** (결과를 보기 전에 정한 표, 위 "encoder 깊이 스케일링" 절):
- **A, B 아님**: 6층은 깊이 4보다 개선하지 못했다(각도 0.113° > 기준 0.088°, argmax 82.5% < 89.8%).
- **C 아님**: "깊이 4와 같음"이 아니라 **더 나쁘다**. 6층 / 8층의 깊이 4 대비 차이는 각도 +0.016° / +0.020°(seed 표준편차 0.009° 이상), argmax −4.0 / −4.6%p(표준편차 3.3%p 이상), 게임 +0.22 / +0.39 shots. 8층은 게임이 control(3.90)과 같은 수준(3.94)으로 돌아갔다.
- **D 해당**: 6층과 8층이 깊이 4보다 나쁘고 **train loss가 깊이 4보다 낮아지지 않는다**. seed 2는 25.4(6층) / 20.3(8층)으로 깊이 4의 15.7보다 높다. 나머지 seed도 깊이 4와 같거나 약간 높다(6층 16.7 / 17.0, 8층 17.4 / 18.0 vs 깊이 4 16.5 / 16.8). 파라미터가 1.4~1.7배인데 train loss가 안 내려가므로 **과적합(train ↓ val ↑)이 아니라 최적화 문제**로 읽는다.
- **E도 해당**: 깊이 4보다 나쁜 seed가 있다. 6층은 seed 2(각도·argmax·게임 모두), 8층은 게임 shots가 3 seed 모두(스캔 argmax는 seed 1이 0.881로 깊이 4와 같고, 각도도 seed 1은 0.098 vs 0.095로 거의 같음).

**결론**: 이 설정(residual·normalization 없음, lr 1e-3)에서 **깊이 4가 최선**이고, 더 깊게는 오히려 나빠진다. 판단표 D의 말대로 **깊이 자체가 병목인지는 아직 모른다**: 더 깊은 모델이 학습되지 않은 것이지, 더 깊은 모델의 한계를 본 것이 아니다. 사전에 정한 다음 행동(D): residual connection 또는 LayerNorm을 넣어 같은 깊이(6/8)를 재시도하고 lr 3e-4도 시험한다. 가설일 뿐이고 이 변경이 최적화를 고친다는 것은 아직 검증하지 않았다. 비용은 깊이 4 → 6 → 8에서 학습 시간이 약 +33% / +51%.

### 학습 길이 3배: 깊이 4 vs 깊이 6, 1500 epoch (2026-10-12 계획)

**왜**: 깊이 6/8은 깊이 4보다 나빴고 train loss가 안 내려갔다(판단표 D). 학습 곡선(`history.npy`, 100 epoch 구간별 train loss 중앙값, seed 0 / 1 / 2)을 보면 두 가지가 보인다.
- **시작이 느리다**: epoch 100~199에서 깊이 4는 37.9 / 35.1 / 31.0인데 깊이 6은 38.3 / 290 / 52.6, 깊이 8은 146 / 283 / 51.8. 깊이 8은 epoch 300 부근에서야 따라온다. 같은 500 epoch 예산에서 쓸모 있는 학습 시간이 깊이 4보다 짧았다.
- **seed 0·1은 마지막에 깊이 4 수준에 도달**(epoch 400~499: 깊이 6 16.9 / 17.2 vs 깊이 4 16.6 / 16.9)했는데도 깊이 4보다 낫지 않았다. **seed 2는 폭주**(깊이 6 epoch 300~399 중앙값 122, 끝 27.2)했고 회복 중에 lr이 0으로 내려갔다.

가설: (a) 깊은 모델은 500 epoch이 부족했다(학습량 문제) / (b) 깊이 > 4는 구조상 최적화가 불안정하다(residual·normalization 문제, TODO 18). 이 실험은 (a)를 가린다. (b)는 폭주(seed 2)에 해당하고 학습을 늘려도 안 고쳐질 수 있다.

**"plateau에서 종료"를 쓰지 않는 이유**: lr이 cosine으로 1e-3 → 1e-5까지 내려가므로 마지막 구간의 loss는 어떤 모델이든 평평해진다. plateau 감지로 끊으면 수렴이 아니라 스케줄 끝을 감지하게 된다. 스케줄을 바꾸면 이전 run과 같은 코드로 비교하는 조건도 깨진다. 그래서 길이를 고정한다.

**변경**: 깊이 4와 깊이 6 모두 `--epochs 1500 --patience 1500`. cosine도 1500 epoch에 걸쳐 감쇠한다(그 때문에 epoch 500 시점의 lr은 약 0.75e-3으로 500 epoch 스케줄보다 높다). `--patience 1500`은 early stop을 끄기 위해서다(patience 500이면 val 최댓값이 늦게 안 갱신될 때 cosine 꼬리가 잘린다). 나머지는 control과 동일(`--pair-supervision --no-wta-fallback --made2-head`, hidden 128), seed 0/1/2. 출력은 `logs/experiments_long`: run 이름에 epoch 수가 들어가지 않아 기존 500 epoch run과 glob가 겹치기 때문이다. **같은 길이의 깊이 4가 이 실험의 control이다**(길게 돌려서 좋아진 것과 깊이 효과가 섞이지 않게).
**비용**: 학습 시간이 epoch에 비례(3배). 500 epoch 기준 깊이 4 약 2,650초, 깊이 6 약 3,530초(6 runs 병렬)였으므로 약 2.2~2.9시간(추정). 깊이 8은 깊이 6 결과를 본 뒤에 결정한다.
**지표**: `eval_by_balls`(각도 오차, 스캔 argmax, clean 성공률, 게임 shots/game 500 games), 학습 곡선(구간별 train loss 중앙값, 폭주 여부).

**기준점** (500 epoch, 3 seed 평균)

| 지표 | 깊이 4 | 깊이 6 |
|---|---|---|
| pair head 각도 오차 중앙값 전체 | 0.097° | 0.113° |
| 스캔 argmax가 창 안 | 86.5% | 82.5% |
| 게임 shots/game | 3.55 ± 0.10 | 3.77 ± 0.16 |

**"효과 인정" 규칙**: 같은 길이 깊이 4와 seed별로 짝지어 3개 모두 더 좋고 평균 차이가 각도 0.009°, argmax 3.3%p, 게임 0.15 shots(두 깊이의 seed 표준편차 0.10 / 0.16 근처) 이상.

| # | 관측 (1500 epoch) | 해석 | 다음 행동 |
|---|---|---|---|
| A | 깊이 6이 같은 길이 깊이 4보다 "효과 인정" 기준으로 좋다 | 깊이 6은 학습량 부족이었다. 깊이가 더 도움이 된다 | 깊이 8도 1500 epoch. 시간 3배 대비 이득을 같이 기록 |
| B | 깊이 6 ≈ 깊이 4(기준 미만), 둘 다 500 epoch 때보다 좋아짐 | 병목은 학습량이고 깊이가 아니다 | 깊이 4 고정. 이전 비교(control 500 epoch 등)를 같은 길이로 다시 봐야 하는지 결정(모든 절대 수치가 바뀔 수 있음) |
| C | 깊이 6 ≈ 깊이 4, 둘 다 500 epoch 때와 같음 | 학습량도 깊이도 아니다 | residual/LayerNorm 재시도(TODO 18) 또는 search → distill(TODO 19) |
| D | 깊이 6이 여전히 깊이 4보다 나쁨, 또는 train loss 폭주 재발 | 깊이 > 4의 최적화 불안정은 길이로 안 풀린다 | residual 또는 LayerNorm + lr 3e-4(TODO 18) |
| E | 깊이 4가 1500 epoch에서 500 epoch보다 나빠짐 | 길게 돌리면 과적합 | 500 epoch 유지. 정규화·checkpoint 선택(#9) 검토 |
| F | 위 중 seed 하나만 반대 방향 | 불확실 | 결론 보류, seed 5개 |


### 깊이 4에서 남은 실패는 어디서 오는가 (2026-10-12, 학습 없음)

**질문** (깊이 4 결과를 보고 사용자가): 깊이 4에서도 샷 성공률이 아직 낮은 것은 (a) pair를 잘못 골라서인가, (b) 고른 pair의 action 정밀도가 모자라서인가, (c) made head의 gradient로 action을 찾으면 되는가.
**왜 다시 쟀나**: 2026-10-11의 "selection vs regression 분리"는 control(2층) 모델 기준이었고 깊이 4에는 없었다. 또 용량 실험 판단표는 "스캔 argmax가 창 안" 비율을 made head의 해상도로 읽었는데, 그 값을 기준선(우연, pair head 자신의 proposal)과 대조한 적이 없었다.
**방법**: `exp16_wm/diagnose_action.py`(테스트 있음), control·깊이 4·6·8 × seed 0/1/2, val state 중 label pair가 있는 것(남은 공 1/2/3개: 1910 / 829 / 2250, 3 seed 합침)을 **실제로 쳐서** 측정. clean = pocket이 들어가고 scratch 없음(`eval_by_balls`와 같은 정의). 임시 스크립트로 먼저 낸 수치(깊이 4, control, 깊이 6)와 저장소 도구 결과가 소수점까지 같음을 확인했다.

**1. pair 선택 vs action 정밀도** (남은 공 전체; 같은 모델의 action으로, 정책이 고른 pair와 label pair[후보 0]를 비교)

| 모델 | select: 정책이 고른 pair | select: label pair로 고정 | 선택 손실 | verify+refine: 정책이 고른 pair | verify+refine: label pair로 고정 | 선택 손실 | label pair를 줘도 실패 (정밀도) |
|---|---|---|---|---|---|---|---|
| control (2층) | 71.5% | 77.2% | 5.8%p | 78.3% | 81.8% | **3.5%p** | **18.2%p** |
| **깊이 4** | 75.3% | 81.6% | 6.3%p | 82.2% | 86.3% | **4.2%p** | **13.7%p** |
| 깊이 6 | 71.7% | 77.0% | 5.3%p | 78.1% | 81.0% | **2.9%p** | **19.0%p** |
| 깊이 8 | 71.4% | 77.8% | 6.4%p | 77.5% | 82.0% | **4.5%p** | **18.0%p** |

남은 공 2개 state만:

| 모델 | select: 정책이 고른 pair | select: label pair로 고정 | 선택 손실 | verify+refine: 정책이 고른 pair | verify+refine: label pair로 고정 | 선택 손실 | label pair를 줘도 실패 (정밀도) |
|---|---|---|---|---|---|---|---|
| control (2층) | 58.7% | 72.3% | 13.5%p | 66.8% | 73.9% | **7.1%p** | **26.1%p** |
| **깊이 4** | 64.5% | 74.5% | 10.0%p | 74.9% | 77.8% | **2.9%p** | **22.2%p** |
| 깊이 6 | 63.6% | 71.3% | 7.7%p | 71.3% | 74.1% | **2.8%p** | **25.9%p** |
| 깊이 8 | 64.4% | 71.0% | 6.6%p | 70.4% | 72.9% | **2.4%p** | **27.1%p** |

- **깊이 4: 선택 손실은 4.2%p, label pair를 정확히 줘도 13.7%p가 실패한다.** 남은 실패의 약 77%가 선택이 아니라 그 pair를 정확히 치지 못하는 데서 온다. 남은 공 2개 state도 같다(선택 손실 2.9%p, 정밀도 실패 22.2%p).
- **선택 손실은 모든 모델에서 비슷하다**(verify+refine에서 2.9~4.5%p). 깊이의 이득은 **label pair를 줬을 때의 정밀도**에서 나온다: 81.8% / 86.3% / 81.0% / 82.0% (control / 4 / 6 / 8층). 깊이 6·8은 control 수준이다.
- 남은 공 3개 state의 선택은 깊이가 4를 넘으면 나빠진다(정책 pair가 label의 상위 2개 pair 안인 비율 control 81.7% / 깊이 4 82.8% / 깊이 6 78.6% / 깊이 8 73.6%).

**2. 각도만 고치는 refine: made head 격자 vs gradient** (label pair, speed는 proposal 그대로, 각도만 변경; 남은 공 전체 clean)

| 모델 | proposal 그대로 | made head 격자 ±0.3° × 31 | gradient ascent (12 × 0.04°) | gradient ascent (24 × 0.01°) |
|---|---|---|---|---|
| control (2층) | 77.2% | 76.8% | 76.9% | 76.6% |
| **깊이 4** | 81.6% | 81.6% | 81.5% | 81.5% |
| 깊이 6 | 77.0% | 76.3% | 76.6% | 76.4% |
| 깊이 8 | 77.8% | 77.6% | 77.5% | 77.5% |

- 어느 모델에서도 **각도만 조정해서는 이득이 없다**(proposal 대비 −0.7 ~ 0%p). 격자도 gradient도 같다.
- 같은 label pair에 verify+refine(각도 31 × 절대 speed 7)을 하면 깊이 4가 81.6% → 86.3%로 오른다. 각도만으로는 0인데 speed 격자를 넣으면 +4.8%p이므로, refine의 이득은 각도가 아니라 **speed**에서 나온다(각도와 speed의 결합 효과는 따로 분리하지 않았다).

**3. made head 스캔의 기준선** (label action을 ±0.6°, 0.05° 간격으로 옮겨 실제로 침. 창이 스캔 범위의 일부인 state 326개)

| 모델 | made head argmax가 창 안 | 우연(아무 offset) | pair head 자신의 proposal에 가장 가까운 offset | offset 0(label) | proposal이 창 밖인 state | 그때 head가 창 안으로 옮김 | proposal이 창 안일 때 head가 유지 | gradient 방향이 창 쪽(창 밖 offset) |
|---|---|---|---|---|---|---|---|---|
| control (2층) | 81.0% | 53.8% | 78.8% | 98.8% | 69 | 20.3% | 97.3% | 95.6% |
| **깊이 4** | 86.5% | 53.8% | 86.5% | 98.8% | 44 | 4.5% | 99.3% | 96.5% |
| 깊이 6 | 82.5% | 53.8% | 83.7% | 98.8% | 53 | 9.4% | 96.7% | 95.1% |
| 깊이 8 | 81.9% | 53.8% | 83.4% | 98.8% | 54 | 7.4% | 96.7% | 94.9% |

- **made head argmax는 pair head의 proposal 적중률과 거의 같다.** 깊이 4는 정확히 같고(86.5% = 86.5%), proposal이 창 밖인 44개 state에서 head가 창 안으로 옮긴 것은 4.5%(약 2개)뿐이다. control만 +2.1%p, 깊이 6·8은 오히려 낮다. 우연 수준은 53.8%.
- 그래서 **"argmax가 창 안 86.5%"는 made head의 해상도가 아니라 proposal 자체의 정확도**다. 기하 조준각 차이를 입력으로 준 진단 run의 95.1%만 proposal 밖의 정보를 쓴 값이다. 용량 실험 판단표(위)는 이 값을 made head 지표로 읽었으므로, 그 표에서 argmax 임계값은 사실상 pair head 정밀도의 지표였다. 이전 절은 기록으로 두고 해석만 여기서 정정한다.
- gradient 방향은 창 밖 offset에서 모델별로 94.9~96.5% 창 쪽을 가리킨다. 해석: made head는 입력으로 받는 `action − proposal` 차이가 0인 곳에서 확률이 가장 높게 학습돼서(proposal에 앵커됨), gradient가 proposal 쪽으로 돌려보낸다. proposal이 대개 창 안이라 "proposal 쪽 = 창 쪽"이 되어 방향은 맞지만 새 정보는 없다. 위 2번의 "이득 0"과 일치한다. 직접 검증한 것은 아니다(입력 차이를 바꿔 확률이 어떻게 변하는지는 따로 재지 않았다).

**한계**: label pair는 teacher가 고른 후보 0뿐이다(다른 pair가 더 쉬울 수 있다). 스캔은 seed당 150 state, 창이 일부인 326개, proposal이 창 밖인 state는 44개로 작아 "4.5%"는 몇 개 차이다. gradient는 sign-step 한 방식, 각도만 시험했다(speed 방향 gradient는 안 함). speed 격자의 효과는 공 개수별로 나누어 보지 않았다.

### head에서 샘플링해서 치면? 분포 head와 RL 대응 (2026-10-12, 학습 없음)

**질문** (사용자): 점추정 대신 head가 분포(Laplace 등)를 내고, scan 샘플을 posterior/likelihood로 보고 샘플링(Thompson 등)해서 치면 샘플 효율이 오르지 않겠느냐. 이것이 RL에서 continuous action을 뽑는 것(policy gradient 계열)과 비슷하지 않느냐.
**실험**: 위와 같은 설정에서(label pair, speed는 proposal 그대로) 각도 = proposal + Laplace(b)를 실제로 침. 상태당 6개를 뽑아 (i) 샘플 하나를 그대로 쳤을 때의 기대 성공률, (ii) 6개 중 하나라도 성공하는 비율(시뮬레이터가 골라 주는 best-of-6)을 잼. seed당 600 state, 3 seed 합침.

| 각도 = proposal + Laplace(b) | 샘플 1개로 침 (깊이 4) | 〃 남은 공 2개 | 6개 중 시뮬레이터가 고름 (깊이 4) | 6개 중 시뮬레이터가 고름 (control / 6 / 8층) |
|---|---|---|---|---|
| b = 0 (proposal 그대로) | 81.8% | 72.4% | 81.8% | 77.7% / 75.8% / 76.6% |
| b = 0.05° | 80.3% | 72.2% | 89.4% | 86.9% / 85.7% / 85.7% |
| b = 0.10° | 76.6% | 72.0% | 92.6% | 89.6% / 89.7% / 89.6% |
| b = 0.20° | 68.2% | 66.8% | 93.4% | 92.4% / 91.4% / 91.3% |

- **샘플 하나를 그대로 치면 b가 클수록 나빠진다**(모든 모델). label이 순수 기하 조준각의 ±0.05° 안이라 label noise는 거의 없고, 남은 오차는 무작위성이 아니라 근사 오차(bias)다. proposal이 이미 대부분 창 안이라 노이즈는 손해만 더한다. Laplace NLL로 학습하면 scale은 학습되지만 오차 자체는 줄지 않는다. Thompson sampling은 같은 문제를 반복해 풀며 불확실성을 탐색하는 기법이고, 샷 한 번의 0/1 성공만 보면 가장 확률이 높은 action을 치는 것이 최적이다.
- **시뮬레이터가 검증해 주면 샘플링이 큰 가치가 있다**: b = 0.10°, 6개 중 best가 깊이 4에서 81.8% → 92.6%(teacher shot 성공은 99.8%). made head는 verifier로 못 쓴다(위 3번, proposal에 앵커).
- **샷 단위로 보면 이 문제는 SAC를 게임이 아니라 샷에 가져가는 것과 같다.** shots/game ≈ 3 / (샷 성공률)이 모든 모델에서 맞는다(control 3.91 vs 실제 3.90, 깊이 4 3.56 vs 3.55, 깊이 6 3.77 vs 3.77, 깊이 8 3.95 vs 3.94). 정책이 position play를 쓰지 않는 것으로 보이며, 게임 성능은 샷 성공률 하나로 정해진다. 따라서 샷 하나는 reset이 되는 contextual bandit(γ·Q bootstrapping 불필요)이다. 이 틀에서는 position play를 배울 수 없고, 그것은 게임 단위 RL(roadmap #11, #12)의 몫이다.

| 우리가 얘기한 것 | RL에서의 이름 | 이 프로젝트에서 확인한 것 |
|---|---|---|
| head가 분포(Laplace)를 내고 샘플링 | stochastic policy(SAC/PPO의 Gaussian policy). 샘플링은 학습 중 exploration용, 평가는 보통 mean | 추론 때 샘플링은 손해 |
| 샘플 N개를 평가해 best를 침 | CEM / QT-Opt, best-of-N | 시뮬레이터로 6개 중 선택하면 92.6% |
| P(made)를 likelihood로 써서 샘플을 가중 | MPO / AWR. Thompson sampling은 bandit 쪽 용어 | made head가 proposal에 묶여 못 씀 |
| ∇ₐ P(made)로 action 미세조정 | DDPG/TD3의 ∇ₐQ | 방향은 맞지만 이득 0 |
| 샘플 결과(reward)로 policy를 업데이트 | policy gradient(REINFORCE/PPO/SAC) | 아직 안 함. roadmap #12 |

- **search → distill (expert iteration)** = 정책이 각도를 흔들어 샘플 N개를 만들고, 시뮬레이터로 전부 쳐서, 성공한 샘플을 label로 정책을 다시 supervised로 학습한다. 손실은 supervised지만 정답을 고르는 것은 시뮬레이터의 binary reward이므로 reward로 거른 BC(filtered BC, CEM·MPO·AWR 계열)다. SAC로 말하면 actor loss의 Q 자리에 **critic 대신 시뮬레이터의 진짜 1-step return**이 들어간다(critic 근사 오차 없음). SAC의 entropy 항에 해당하는 것은 분포의 폭 b이며, 성공 창 반폭(중앙값 약 0.3°)에 대해 현재 오차 중앙값은 약 0.1°이다. 추론 때는 시뮬레이터 없이 정책만 쓴다. 비용: 시뮬레이터 샷 약 2.5ms(이번 측정에서 역산)이므로 16.7k state × 샘플 6개 ≈ 100k 샷, 약 4분(추정).
- **policy gradient**는 같은 샘플을 reward로 직접 업데이트하는 대안이다. reset이 되는 1-step 문제라 둘은 거의 같은 일을 한다(창 밖 proposal의 tail state에서 노이즈 샘플 중 창에 들어간 것이 mean을 창 쪽으로 이동). 검증하지 않은 가설이다. from-scratch SAC는 7.89 shots로 실패했지만 BC init은 proposal이 이미 창 근처라 sparse reward 문제가 훨씬 작을 것으로 기대한다(마찬가지로 미검증).
- **열린 결정**: 다음 단계를 (1) search → distill(TODO 19)로 갈지, (2) residual/LayerNorm 깊이 재시도(TODO 18)와 학습 길이 실험을 먼저 볼지. 현재 학습 길이 3배 실험이 돌고 있다.
- 별개의 작은 실험 후보: 제곱 오차 대신 Laplace NLL(L1 계열)을 robust loss로 쓰기. 이전 분석에서 train loss의 66%가 label 없는 1.4% row에서 나왔다. 단 샘플별 scale을 학습하면 어려운 2개 state를 덜 보게 될 위험이 있어 scale은 고정하는 쪽이 안전하다.

### 진단 두 가지 (학습 없음): oracle 교체, 정책이 가는 state (2026-10-12 계획)

학습 길이 3배 실험이 도는 동안 할 수 있는 학습 없는 진단. 둘 다 다음 결정을 바꾼다. 도구는 `exp16_wm/diagnose_action.py`(`--parts oracle visited`). 판단표는 결과를 보기 전에 정한다.

#### 1. oracle 교체: 정밀도 실패가 각도 때문인가 speed 때문인가

**질문**: 깊이 4에서 label pair를 줘도 select에서 18.4%p, verify+refine에서 13.7%p가 실패한다. 그 실패는 각도 오차, speed 오차, 둘의 결합 중 무엇인가. 앞선 분해에서 "각도만 고치는 refine은 이득 0, speed 격자를 넣으면 +4.8%p"를 봤지만, 각도와 speed 중 어느 쪽이 얼마나 틀리는지는 직접 안 쟀다. 답에 따라 mode-aware speed head를 올릴지 접을지가 정해진다.
**방법**: 깊이 4(대조로 control), seed 0/1/2, label이 있는 모든 val state에서 label pair(후보 0)의 **pair head 출력**(select, made head 없이)을 기준으로 실제로 친다(clean = pocket + scratch 없음).
- M = 모델 각도 + 모델 speed (기준, 81.6%)
- Ls = 모델 각도 + **label speed**, La = **label 각도** + 모델 speed, L = label 각도 + label speed(= teacher action, 성공률이 teacher의 99.8% 근처여야 한다: 하네스 검증)
- 격자 oracle(시뮬레이터가 고름, seed당 300 state 표본): 모델 각도에서 speed 7개(0.75~5 m/s) 중 하나라도 성공 / 모델 speed에서 각도 31개(±0.3°) 중 하나라도 성공 / 각도 31 × speed 7 중 하나라도 성공. 비교용으로 같은 표본에서 M과 made head verify+refine(같은 label pair).

**판단표**: r_a = La − M, r_s = Ls − M (%p). 표본 오차가 약 ±1%p이므로 2%p 미만의 차이는 같다고 본다. 위에서부터 먼저 맞는 행.

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| A | L < 97% | 하네스(label 행동 재현)에 문제 | 해석 보류, 원인부터 찾는다 |
| B | r_a + r_s < 0.5 · (L − M) | 각도와 speed가 **함께** 맞아야 한다(결합 오차) | 각도·speed를 따로 고치는 방법은 효과가 작다. 결합을 다루는 쪽(joint 모델링, 탐색 TODO 19) |
| C | r_a ≥ 2 · r_s (그리고 r_a − r_s ≥ 2%p) | 각도 오차가 지배 | mode-aware speed head는 접는다. 깊이·loss 스케일·탐색 계속 |
| D | r_s ≥ 2 · r_a (그리고 r_s − r_a ≥ 2%p) | speed 오차가 지배 | mode-aware speed head, speed 표현/loss를 올린다 |
| E | 그 외 | 둘 다 기여 | 병행 |

격자 oracle 보조 해석: (각도 31 × speed 7 oracle) − (같은 표본의 verify+refine) ≥ 3%p이면 같은 격자에서 verifier(made head)의 선택 손실이 크다 → 시뮬레이터 검증·탐색의 가치(TODO 19). 3%p 미만이면 격자 안에서는 made head가 이미 거의 최선이고 남은 실패는 격자 밖(큰 오차)이다.

#### 2. 정책이 가는 state vs teacher state (TODO 16)

**질문**: 게임에서 정책이 만나는 state는 teacher state와 다른가. 다르면 정책이 가는 state에서 label을 받는 방법(DAgger, TODO 10)과 search → distill(TODO 19)의 가치가 커지고, 같으면 teacher state만으로 충분하다. 앞서 "val에서 +2.8%p 이득이 게임에서는 +0.8%p"였던 격차의 원인도 이것일 수 있다.
**방법**: 깊이 4(대조로 control), seed 0/1/2, 각 모델로 500 games(verify+refine 31×7, `eval_by_balls`의 게임과 같은 프로토콜: `reset(seed=0)` 후 reset). 매 샷의 (남은 공 수, 직전 샷 결과{첫 샷 / 직전 적중 / 직전 miss / 직전 scratch}, clean 여부)를 기록한다. 게임의 `shot_success_rate`는 scratch를 무시하고 pocket만 세므로 **clean으로 다시 센다**(val 지표와 같은 정의). val 기준은 같은 모델이 label이 있는 val state에서 verify+refine으로 고른 pair의 clean(깊이 4: 84.8 / 74.9 / 82.6%, 전체 82.2%).
**비교**: (i) 남은 공 수별 clean 성공률 게임 vs val, (ii) 게임에서 남은 공 수 분포, (iii) 직전 샷 결과별 성공률(teacher state에는 거의 없는 "실패 직후 state"가 얼마나 어려운가).

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| A | 남은 공 수별 게임 성공률이 val과 5%p 이내이고 실패 직후 state도 정상 state와 3%p 이내 | 분포 이동이 작다 | DAgger·정책 state distill의 기대 이득이 작다. TODO 19는 teacher state에서 진행 |
| B | 실패 직후(miss·scratch) state의 성공률이 첫 샷·적중 직후보다 5%p 이상 낮음 | 정책이 스스로 만드는 어려운 state가 있다 | 정책이 가는 state에서 label을 받는 방법(TODO 10)·distill의 우선순위를 올린다 |
| C | 게임 성공률이 남은 공 수별로 val보다 5%p 이상 낮음(전반적) | train/val teacher state와의 분포 차이 또는 평가 방식 차이 | 원인 분리(첫 state vs 이후 state, ball-in-hand 위치) |
| D | 게임 성공률이 val보다 5%p 이상 높음 | val의 teacher state가 더 어렵다 | teacher의 state 선택 편향 가능성, 해석 주의 |

표본: 게임 성공률의 표준오차는 샷 수 약 1,700/seed로 약 ±1%p, 실패 직후 state는 수가 적어 표본 수를 함께 적는다.


#### 결과: oracle 교체 (2026-10-12, 학습 없음, 3 seed 합침)

`diagnose_action --parts oracle`, 계획과 판단표는 위 "진단 두 가지". label이 있는 val state 4989개, label pair에서 pair head의 **출력 그대로**(select, made head 없음)를 실제로 침. **하네스 검증 통과**: L(label action)이 98.8%로 A행의 기준 97%를 넘고 teacher 성공률과 같은 수준이다.

**깊이 4**

| clean (실제로 침) | 남은 공 1 | 남은 공 2 | 남은 공 3 | 전체 |
|---|---|---|---|---|
| M: 모델 각도 + 모델 speed | 79.6% | 74.5% | 85.8% | 81.6% |
| Ls: 모델 각도 + **label speed** | 89.2% | 81.8% | 92.8% | 89.6% |
| La: **label 각도** + 모델 speed | 88.8% | 93.2% | 93.0% | 91.5% |
| L: label 각도 + label speed (teacher action) | 97.5% | 99.8% | 99.6% | 98.8% |
| r_a = La − M | +9.2%p | +18.7%p | +7.2%p | +9.9%p |
| r_s = Ls − M | +9.5%p | +7.2%p | +7.0%p | +8.0%p |

**control (2층)** (비교)

| clean (실제로 침) | 남은 공 1 | 남은 공 2 | 남은 공 3 | 전체 |
|---|---|---|---|---|
| M: 모델 각도 + 모델 speed | 75.3% | 72.3% | 80.7% | 77.2% |
| Ls: 모델 각도 + **label speed** | 84.2% | 78.0% | 87.6% | 84.7% |
| La: **label 각도** + 모델 speed | 89.3% | 93.1% | 93.0% | 91.6% |
| L: label 각도 + label speed (teacher action) | 97.5% | 99.8% | 99.6% | 98.8% |
| r_a = La − M | +13.9%p | +20.9%p | +12.3%p | +14.4%p |
| r_s = Ls − M | +8.9%p | +5.8%p | +6.9%p | +7.5%p |

**판단표 적용: 깊이 4 → E ("둘 다 기여, 병행"), control도 E.** r_a = +9.9%p, r_s = +8.0%p, L − M = +17.3%p. B(결합 오차)는 아니다: r_a + r_s = +17.9%p가 L − M과 거의 같아 **각도 오차와 speed 오차는 거의 가산적**이다. C/D도 아니다(비율 1.23배, 2배 미만). control은 r_a / r_s = 1.92배로 C(2배)에 아슬아슬하게 못 미친다.

- **speed 오차는 약 7.4%p의 실패를 만든다**(각도가 완벽해도 모델 speed 때문에 91.5%에 머문다). 각도 오차는 speed가 완벽할 때 9.3%p의 실패를 만든다.
- **깊이 4는 각도 쪽만 줄였다**: r_a가 control +14.4%p → 깊이 4 +9.9%p, 각도 오차의 비용(speed 완벽 시)은 14.1%p → 9.3%p. speed 쪽은 그대로다(r_s +7.5%p → +8.0%p, speed 오차의 비용 7.3%p → 7.4%p). **깊이 4의 이득은 전부 각도에서 왔고 speed는 손대지 않은 상태**다.
- **남은 공 2개 state의 약점은 각도다**: r_a +18.7%p vs r_s +7.2%p(2.6배). 1개·3개 state는 각도와 speed가 거의 같다(r_a +9.2%p / +7.2%p, r_s +9.5%p / +7.0%p).
- 따라서 **mode-aware speed head는 접지 않는다**: speed를 label대로 맞추면 81.6% → 89.6%. 단 이것은 "teacher가 고른 특정 speed"로 바꾼 상한이고, 학습으로 그만큼 얻는다는 뜻이 아니다. (정정: 이 +8.0%p는 select 기준이고 verify+refine 파이프라인의 개선 여지가 아니었다. 아래 "속도 분포 head" 결과 참고.)

**격자 oracle** (시뮬레이터가 후보 중에서 고름, seed당 300 state 표본 900개, 모델의 pair head 출력을 중심으로)

| 모델 | 모델 action 그대로 | speed 7개 중 best | 각도 31개(±0.3°) 중 best | 각도 31 × speed 7 중 best | made head verify+refine(같은 label pair) | (best of 217) − (verify+refine) |
|---|---|---|---|---|---|---|
| 깊이 4 | 80.3% | 94.4% | 94.6% | **99.2%** | 84.8% | **+14.4%p** |
| control (2층) | 75.4% | 90.2% | 93.3% | **98.7%** | 80.0% | **+18.7%p** |

- **같은 격자 안에 거의 모든 state의 해가 있다**: 깊이 4에서 99.2%. speed만 바꿔도 94.4%, 각도만 바꿔도 94.6%가 성공하는 후보가 있다.
- 그런데 made head는 그중 84.8%만 찾는다. 차이 +14.4%p는 보조 해석의 기준 3%p를 훨씬 넘는다 → **같은 격자에서 verifier(made head)의 선택 손실이 크다**. "각도만 refine하면 이득 0"은 격자에 해가 없어서가 아니라 made head가 해를 고르지 못해서였다.
- 이 상한은 시뮬레이터가 후보를 골라 줄 때의 값이다. 학습된 verifier가 얼마나 따라갈 수 있는지는 별개이고, 이전 진단에서 기하 조준각 입력을 준 made head도 verify+refine이 82.6%였다(2층 모델, 이 oracle의 98.7%보다 한참 낮음). 즉 격자의 해를 실제로 얻으려면 학습만으로는 어려울 가능성이 높고, **시뮬레이터 검증/탐색(TODO 19)의 가치를 뒷받침**한다.
- 한계: 표본 900개(표준오차 약 ±1.3%p). label pair 하나에 대해서만 쟀다. "best of 217"은 후보 수가 많아서 높게 나오는 면이 있다(speed 7개나 각도 31개만으로도 각각 94%대).

#### 결과: 정책이 가는 state vs teacher state (2026-10-12, 학습 없음, 3 seed × 500 games)

`diagnose_action --parts visited`. **롤아웃 검증**: 샷 수가 깊이 4 5328개 = 3.552 shots/game, control 5856개 = 3.904 shots/game로 `eval_by_balls`의 게임 결과(3.55, 3.90)를 그대로 재현한다.

| 모델 | | 남은 공 1 | 남은 공 2 | 남은 공 3 | 전체 |
|---|---|---|---|---|---|
| 깊이 4 | 게임 | 83.3% | 76.1% | 81.6% | 80.2% |
| | val(teacher state) | 84.8% | 74.9% | 82.6% | 82.2% |
| control (2층) | 게임 | 74.9% | 67.4% | 75.9% | 72.6% |
| | val(teacher state) | 81.0% | 66.8% | 80.1% | 78.3% |

게임의 clean 성공률을 **직전 샷이 어떻게 끝났는지**로 나누면:

| 모델 | 첫 샷 | 직전 적중 | 직전 miss | 직전 scratch | 첫 샷 + 직전 적중 | 직전 miss·scratch |
|---|---|---|---|---|---|---|
| 깊이 4 | 83.8% (n 1500) | 81.3% (n 2849) | 70.4% (n 797) | 75.3% (n 182) | 82.2% | 71.3% (n 979) |
| control (2층) | 79.7% (n 1500) | 73.6% (n 2827) | 61.6% (n 1298) | 74.5% (n 231) | 75.8% | 63.5% (n 1529) |

**판단표 적용: 깊이 4, control 모두 B만 해당.** 남은 공 수별 게임 성공률은 val과 깊이 4에서 1.5%p 이내(분포 이동 없음)지만, 실패 직후 state의 성공률이 정상 state보다 깊이 4에서 10.9%p, control에서 12.3%p 낮다(기준 5%p). 실패 직후 격차가 A의 기준 3%p를 넘으므로 A는 해당하지 않는다. control은 남은 공 1개·3개 state에서 게임이 val보다 6.1 / 4.2%p 낮지만(2개 state는 +0.6%p) 모든 그룹이 5%p 이상 낮은 것은 아니라 C는 아니다. control은 실패가 많아 실패 직후 state가 전체의 26%를 차지하는 영향으로 보인다(미분리).

- **실패 직후 state는 어렵다**(깊이 4: miss 직후 70.4%, 정상 82.2%). 그러나 **실패의 대부분은 거기서 나오지 않는다**: 깊이 4 게임의 실패 1056개 중 27%만 miss·scratch 직후 state에서 나오고, 73%는 첫 샷과 적중 직후(val과 같은 성공률 82.2%)에서 나온다. 실패 직후 state가 정상 수준이 된다면 clean 성공률은 약 +2.0%p 오른다(대략의 상한, 정책이 가는 state에서 label을 받는 방법의 이득이 이 정도로 제한됨).
- **해석 주의**: 정책이 결정론적이라 miss 직후 state는 직전과 거의 같을 수 있고, 그러면 같은 action을 반복해 또 실패하는 것이다. 즉 "teacher 분포 밖의 state가 어렵다"가 아니라 "정책이 이미 한 번 실패한 어려운 state"일 수 있다. 직전과 얼마나 같은지는 이번에 재지 않았다(미검증). 같다면 재시도 때 action에 노이즈를 주는 것(샘플링이 이득이 되는 경우)을 시험할 수 있다.
- 부수 발견: 게임의 clean 성공률(깊이 4 80.2%)은 `shot_success_rate` 84.2%(pocket만 세고 scratch는 무시)보다 낮다. 정의가 달라서이고 차이가 scratch 동반 pocket 때문인지는 따로 세지 않았다.

#### 두 진단의 종합 (2026-10-12)

1. **각도와 speed가 각각 약 8~10%p씩 거의 가산적으로 실패를 만든다.** 깊이 4는 각도만 개선했다. 남은 공 2개의 약점은 각도다. → speed 쪽(mode-aware head 등)은 아직 손대지 않은 큰 항목이다(상한 +8.0%p).
2. **격자 안에 해는 거의 항상 있는데 made head가 못 고른다**(상한 99.2% vs 84.8%). 학습된 verifier로 줄이기는 어려워 보이고, 시뮬레이터 검증/탐색(TODO 19)이 이 격차를 겨냥한다.
3. **teacher state에서 개선한 것은 게임으로 이어진다**(첫 샷·적중 직후 성공률이 val과 같고 실패의 약 73%가 거기서 나온다). 정책이 가는 state(실패 직후)에서 label을 받는 것은 이득이 있지만 상한이 clean 성공률 +2.0%p 정도다.

**결과: 학습 길이 3배 (깊이 4 vs 깊이 6, 1500 epoch, seed 0/1/2; 2026-10-12)**

`eval_by_balls`(같은 도구, 같은 val state, 게임 500), 출력 `logs/experiments_long`. 세 run 모두 `epochs_run = 1500`(early stop 없음), best epoch은 깊이 4가 1452 / 1462 / 1468, 깊이 6이 1369 / 1209 / 1203이다. 500 epoch 값은 위 "깊이 6, 8 결과"의 표.

| 지표 | 깊이 4, 500 | 깊이 4, **1500** | 깊이 6, 500 | 깊이 6, **1500** |
|---|---|---|---|---|
| pair head 각도 오차 중앙값 전체 | 0.097° | 0.091° | 0.113° | **0.082°** |
| 〃 seed별 | 0.101 / 0.095 / 0.093 | 0.088 / 0.085 / 0.099 | 0.101 / 0.101 / 0.136 | 0.082 / 0.082 / 0.082 |
| 〃 남은 공 2개 | 0.158° | 0.147° | 0.169° | 0.118° |
| within 0.2° 남은 공 2개 | 57.7% | 61.2% | 55.2% | 65.5% |
| 스캔 argmax가 창 안 (proposal 적중률과 같은 값) | 86.5% | 88.3% | 82.5% | 88.7% |
| 〃 seed별 | 0.860 / 0.881 / 0.851 | 0.897 / 0.932 / 0.812 | 0.879 / 0.881 / 0.703 | 0.925 / 0.881 / 0.851 |
| clean select 전체 | 74.5% | 76.2% | 70.9% | 76.8% |
| clean verify+refine 전체 / 공 2개 | 81.3% / 74.9% | 80.1% / 69.7% | 77.2% / 71.3% | 81.4% / 73.7% |
| **게임 shots/game** (success) | 3.55 ± 0.10 (84.2%) | 3.59 ± 0.10 (83.4%) | 3.77 ± 0.16 (79.6%) | **3.51 ± 0.05** (85.1%) |
| 〃 seed별 | 3.52 / 3.66 / 3.47 | 3.66 / 3.47 / 3.63 | 3.69 / 3.66 / 3.95 | 3.51 / 3.56 / 3.47 |
| 학습 끝 train loss (마지막 100 epoch 중앙값) seed별 | 16.6 / 16.9 / 15.8 | 15.0 / 14.9 / 15.7 | 16.9 / 17.2 / 27.2 | 14.9 / 14.8 / 15.2 |
| 학습 시간 | 약 2,650초 | 약 7,930초 | 약 3,530초 | 약 9,315초 |

**같은 길이의 깊이 6 − 깊이 4** (seed별 짝지음, 사전에 정한 "효과 인정" = 3 seed 모두 개선 + 평균 차이 ≥ 각도 0.009° / argmax 3.3%p / 게임 0.15 shots):

| | 500 epoch | 1500 epoch |
|---|---|---|
| 각도 오차 개선 seed별 / 평균 | 0 / −0.006 / −0.043° / **−0.016°** (깊이 6이 나쁨) | +0.006 / +0.003 / +0.017° / **+0.0087°** (3 seed 모두 개선, 기준 0.009°에 0.0003° 못 미침) |
| argmax 차이 seed별 / 평균 | +1.9 / 0 / −14.9 / −4.3%p | +2.8 / −5.1 / +4.0 / +0.6%p |
| 게임 shots 차이 (깊이 6 − 깊이 4, 음수가 깊이 6 우세) seed별 / 평균 | +0.17 / 0.00 / +0.48 (평균 +0.22, 깊이 6이 나쁨) | −0.15 / +0.09 / −0.16 (평균 −0.07, 깊이 6이 좋음, seed 1은 반대) |
| verify+refine clean 차이 | −2.9 / −3.9 / −5.4 (평균 −4.1%p) | +2.3 / −0.7 / +2.4 (평균 +1.3%p) |

**판단표 적용**: 표의 어느 행도 정확히 맞지 않는다.
- **A 아님**: 깊이 6이 "효과 인정" 기준을 충족한 지표가 없다. 각도는 3 seed 모두 낫지만 평균 0.0087°로 기준 0.009°에 못 미치고(사실상 경계), argmax(+0.6%p)와 게임(−0.07 shots, seed 1은 반대)은 기준 미만이다.
- **B와 가장 가깝지만 한 조건이 다르다**: "깊이 6 ≈ 깊이 4"는 맞다. 그러나 "둘 다 500 epoch 때보다 좋아짐"은 **깊이 6에만** 맞다(게임 3.77 → 3.51, 각도 0.113 → 0.082°, 3 seed 모두 개선). 깊이 4는 길게 돌려도 노이즈 범위다(게임 3.55 → 3.59, 각도 0.097 → 0.091°는 표준편차 0.009° 안, verify+refine 81.3 → 80.1%). C(둘 다 500 epoch과 같음)도 깊이 6이 변했으므로 아니다. D(깊이 6이 여전히 나쁨/폭주 재발)도 아니다. E(깊이 4가 길어서 나빠짐)는 기준 미만이라 아니다.

**결론**:
- **가설 (a) "깊은 모델은 500 epoch이 부족했다"는 깊이 6에 대해 맞다.** 500 epoch에서의 열세(3.77)가 사라졌고(3.51), 느린 시작과 seed 2의 폭주도 3배 길이에서는 수렴했다(마지막 구간 train loss가 깊이 4와 같은 14.9 / 14.8 / 15.2). 앞 절에서 "최적화 문제"로 읽은 것은 **구조적 불안정이 아니라 학습 예산 부족**이었다. 따라서 residual/LayerNorm(TODO 18)의 필요성은 크게 줄었다.
- **그러나 같은 길이에서 깊이 6이 깊이 4보다 낫다는 증거는 없다.** 각도 정밀도에서 경계 수준의 우위(−0.009°), 게임은 같음(3.51 vs 3.59, 표준편차 안). 비용은 학습 시간이 깊이 4(500 epoch)의 약 3.5배(9,315초 vs 2,650초). **깊이 4, 500 epoch이 비용 대비 최선**이다.
- 사전 지표는 아니지만 남은 공 2개의 각도 오차는 깊이 6이 0.118° vs 깊이 4 0.147°(−20%)로 가장 크게 다르다. 2개 state가 병목이라면 깊이 6이 거기서 의미가 있을 수 있으나, 이는 사후에 본 값이라 가설로만 둔다(게임 지표로는 확인되지 않음).
- **깊이 8을 1500 epoch으로 돌릴 이유는 약하다**(깊이 6이 깊이 4를 못 넘었다). 깊이 축은 일단 닫는다.
- 열린 점: 깊이 2(control)는 500 epoch에서 충분했는지 길게 돌려 보지 않았다. 깊이 4는 충분했음이 확인되어 이전의 500 epoch 비교는 유효하다.
- 한계: seed 3개, 게임 표준편차 0.05~0.10. "최댓값 선택" 편향은 best epoch이 끝 근처(깊이 4 1452~1468, 깊이 6 1203~1369)라 작다. 깊이 6의 best epoch 두 개는 끝에서 약 300 epoch 앞이다.

### TODO 22: miss 직후 state는 직전과 같은가 (2026-10-12 계획, 학습 없음)

**질문**: 게임에서 miss·scratch 직후 샷의 성공률이 정상 state보다 낮다(깊이 4: 71.3% vs 82.2%). 결정론적 정책이 같은 target을 같은 방식으로 반복해 또 실패하는 것인가, 아니면 실패한 샷이 만든 새로운 배치가 원래 어려운 것인가. 반복 실패라면 재시도 때 action에 노이즈를 주는 롤아웃(2단계)이 의미가 있다.
**사전 점검 (원래 가설이 약해지는 점)**: 샷마다 cue ball이 굴러가 위치가 바뀌므로 miss 직후 state가 직전과 **같을 수는 없다**. 그래서 "같은 action을 반복"은 성립하지 않고, 확인할 수 있는 약한 형태는 "**object ball 배치가 그대로인가**"(cue ball만 움직이고 target은 그대로)다.
**방법**: `diagnose_action --parts visited`의 롤아웃(깊이 4, 대조로 control, seed 3개 × 500 games)에 직전 state와의 변위를 기록한다: cue ball 이동량, 아직 안 들어간 object ball 중 가장 많이 움직인 것의 이동량(obs 단위, table 좌표 정규화). miss·scratch 직후 샷만 본다. **object ball 이동량 < 1e-3을 "배치 그대로"** 로 본다(임계값에 둔감한지 보려고 분포도 적는다).
**비교**: (i) miss·scratch 직후 샷 중 배치 그대로인 비율, (ii) 배치 그대로 / 바뀜 그룹의 clean 성공률, (iii) cue ball 이동량 분포.

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| A | 배치 그대로 비율 ≥ 50% 이고 그 그룹의 성공률이 바뀜 그룹보다 ≥ 5%p 낮음 | 같은 target에서 재실패하는 경향이 있다 | 2단계(재시도 때 각도 노이즈를 주는 롤아웃)로 진행 |
| B | 배치 그대로 비율 < 50% | 대부분 새 배치. 결정론적 반복 실패 가설은 기각 | 2단계를 접는다. 난이도 원인은 "실패한 샷이 만든 어려운 배치"로 보고, 같은 state에서 teacher(planner) 성공률을 재서 내재 난이도인지 확인 |
| C | 배치 그대로 비율 ≥ 50% 인데 성공률 차이 < 5%p | 같은 target이어도 재실패 경향 없음 | 2단계를 접는다 |
| D | 배치 그대로 그룹이 30개 미만 | 표본 부족 | 판단 보류, 게임 수를 늘려 재측정 |

**결과: TODO 22 (2026-10-12, 학습 없음, 3 seed × 500 games)**

`diagnose_action --parts visited`. 게임 지표는 이전 실행과 소수점까지 같다(깊이 4 clean 80.2%, 실패 직후 71.3% 등: 롤아웃은 결정론적).

| 모델 | miss·scratch 직후 샷 | object ball 배치 그대로 (이동 < 1e-3) | 그대로일 때 clean | 바뀌었을 때 clean | object ball 이동 p10 / p50 / p90 | cue ball 이동 p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| 깊이 4 | 979 | 214 (**21.9%**) | 63.6% | 73.5% | 0.0000 / 0.3074 / 0.6476 | 0.166 / 0.492 / 0.820 |
| control (2층) | 1529 | 368 (**24.1%**) | 58.7% | 65.0% | 0.0000 / 0.2797 / 0.6650 | 0.175 / 0.515 / 0.849 |

**판단표 적용: 깊이 4, control 모두 B** (배치 그대로 비율 < 50%).
- **대부분의 miss·scratch 직후 샷은 새 배치다.** object ball 이동 중앙값이 0.31(obs 단위)로 크다: 실패한 샷은 대개 공을 맞혀 배치를 흩어 놓지만 넣지 못한다. cue ball은 당연히 항상 움직인다(중앙값 0.49).
- 따라서 **"결정론적 정책이 같은 action을 반복해 또 실패한다"는 가설은 지지되지 않는다.** 재시도 때 노이즈를 주는 롤아웃(2단계)은 접는다.
- 배치가 그대로인 그룹은 실제로 더 어렵다(깊이 4: 63.6% vs 73.5%, −9.9%p; control −6.3%p). 같은 target을 다시 칠 때 재실패하는 경향이 약하게 있다는 뜻이지만, 이 그룹은 전체 샷의 약 4%(214 / 5328)라 노이즈를 줘서 정상 수준이 되어도 clean 성공률은 약 +0.4%p 오른다(상한).
- 남은 설명은 "실패한 샷이 만든 배치가 원래 어렵다"(미검증). 판단표 B의 다음 행동인 teacher(planner) 성공률 확인은, 이득 상한이 앞서 계산한 +2.0%p(실패 직후 state 전체)에 머물러 우선순위를 낮추고 하지 않았다.
- 정정: 위 "정책이 가는 state vs teacher state"의 "정책이 결정론적이라 같은 action을 반복할 수 있다"는 해석 주의는 이 측정으로 기각됐다.

### speed 쪽 사전 점검: speed window map (TODO 21의 1단계, 2026-10-12 계획, 학습 없음)

**왜**: oracle 교체에서 speed를 label로 바꾸면 깊이 4가 81.6 → 89.6%(+8.0%p)이고 각도 오차와 거의 가산적이었다. 깊이 4는 각도만 개선했고 speed는 못 고쳤다. speed가 틀리는 이유 후보:
- (a) **mode hedging**: label speed가 1.0 m/s 76% / ≥2 m/s 24%의 두 모드인데 모델이 사이로 평균을 내서 성공 speed 사이의 **틈**에 떨어진다 → 처방: mode-aware head(분류 + 회귀) 또는 median(L1) 손실.
- (b) **단순 정밀도 오차**: 성공 speed 구간 바깥(위·아래)에 떨어진다 → 처방: 성공 구간 안이면 gradient 0인 tolerant 손실, 또는 폭이 좁은 state에 가중.
- (c) **label 각도에서 어떤 speed도 성공하지 못함**: label 각도는 label speed에서만 유효한 결합이라 speed 단독 접근에 한계.

**관련 논의 (사용자 지적)**: 점 label은 성공 창의 **폭**을 모른다. negative(실패) 샘플이 주는 정보는 창의 경계, 즉 폭이다. 각도 창 폭은 state마다 크게 다르다: 750개 state 중 45%는 ±0.6° 스캔보다 넓었고(폭 > 1.2°), 스캔 안에 드는 385개의 폭은 p10 / p50 / p90 = 0.25° / 0.55° / 0.90°(3.6배 차이), 0.3° 미만이 10.4%(전체의 약 5.7%). 점 label에 대한 제곱 오차는 모든 state에 같은 정밀도를 요구하므로, 용량이 모자란 모델이 폭이 좁은 state에 용량을 쓰도록 폭 정보를 주는 손실은 시험할 만하다(미검증 가설). 이 점검의 speed map은 speed의 같은 정보(성공 speed 집합)를 준다.
(정정 범위: 앞서 "경계 중앙 label은 teacher label과 같아 의미 없다"는 **중앙 위치**에 대한 것이다. 창의 폭·경계 정보가 새로운 정보라는 것은 이 측정으로 지지된다.)

**방법**: 깊이 4(대조로 control), seed 0/1/2, label이 있는 모든 val state에서 **label 각도**를 고정하고 speed 격자 20개(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0 m/s)를 실제로 쳐서 성공 speed 집합을 만든다(맵은 state에만 의존해 seed별 한 번 계산). 모델 speed(label pair의 pair head 출력)로 label 각도에서 친 결과가 실패한 state를 모아, 모델 speed가 성공 집합에 대해 어디에 있는지 분류한다:
- empty = 격자 어디서도 성공 없음, gap = 모델 speed가 성공 speed의 최소·최대 사이인데 실패(성공 구간 사이의 틈), below / above = 모델 speed가 모든 성공 speed보다 낮음 / 높음.
- 맵 자체의 통계: 성공 구간이 둘 이상인 state 비율, 성공 speed 범위(최대/최소 비), 격자에서 성공이 하나도 없는 비율.

**판단표** (실패 state 수 ≥ 30일 때; 위에서부터 먼저 맞는 행. 격자 간격이 성긴 곳의 좁은 speed 창은 틈이나 empty로 잘못 잡힐 수 있다)

| # | 관측 (모델 speed 실패 state 중) | 해석 | 다음 행동 |
|---|---|---|---|
| E | 실패 state < 30 | 표본 부족 | 판단 보류 |
| C | empty ≥ 30% | label 각도에서 speed 단독으로는 못 고침(결합) | speed 단독 처방 한계. 각도·speed 결합 모델링 또는 탐색 |
| A | gap ≥ 40% | mode hedging이 주된 원인 | mode-aware head(분류 + 회귀) 또는 L1 손실(TODO 21) |
| B | below + above ≥ 60% | 모드 문제가 아니라 단순 정밀도 오차 | tolerant speed 손실(성공 구간 안이면 gradient 0) 또는 폭 가중. mode head는 접는다 |
| D | 그 외 | 섞여 있음 | 둘 다 시험 |

**결과: speed window map (2026-10-12, 학습 없음, 3 seed 합침)**

`diagnose_action --parts speedmap`, label이 있는 val state 4989개, label 각도에서 speed 격자 20개를 실제로 침.

**맵 자체의 구조 (state에만 의존, 두 모델 같음)**
- label 각도에서 어떤 speed도 성공하지 못하는 state: **0.7%**(거의 없음). 성공 speed 구간이 하나: 46.4%, **둘 이상으로 갈라짐: 53.0%**.
- 가장 빠른 성공 speed / 가장 느린 성공 speed의 비는 p10 / p50 / p90 = 5.0 / 12.0 / 16.0배: 성공하는 speed 범위가 매우 넓다.

**모델 speed(label 각도에서)가 실패하는 state의 위치**

| 모델 | label 각도 + 모델 speed 성공 | 실패 state | **틈(gap)** | 아래 / 위 | empty | 실패한 모델 speed p10 / p50 / p90 (m/s) | 판단표 |
|---|---|---|---|---|---|---|---|
| 깊이 4 | 91.5% | 426 | 70.4% | 14.3% / 7.5% | 7.7% | 1.36 / 1.57 / 2.22 | **A** |
| control (2층) | 91.6% | 420 | 71.4% | 13.8% / 6.9% | 7.9% | 1.36 / 1.62 / 2.23 | **A** |

**판단표 적용: 깊이 4, control 모두 A (gap ≥ 40%)** → mode hedging이 주된 원인. empty는 8%로 C 기준(30%)에 한참 못 미친다.
- 모델 speed 실패의 약 70%는 성공 speed **사이의 틈**에 떨어진다. 실패한 모델 speed의 중앙값이 1.57 m/s(p10 1.36 ~ p90 2.22)로, label speed의 두 모드(1.0 m/s 76% / ≥2 m/s 24%) 사이다. MSE 회귀가 두 모드의 평균 쪽으로 가서 어느 모드도 아닌 곳에 떨어진다는 가설과 일치한다.
- 구간 바깥(아래 14.3% / 위 7.5%)의 단순 오차는 약 21.8%, label 각도에서 해가 없는 경우는 7.7%다. 따라서 mode hedging을 완벽히 고치면 label 각도에서 약 6.0%p(= 틈 실패 300개 / 4989), 단순 오차까지 고치면 약 7.9%p가 회복된다(각도가 label일 때의 상한).
- **깊이는 이 문제에 영향이 없다**: control과 깊이 4가 거의 같다(틈 71.4% vs 70.4%, 성공 91.6% vs 91.5%). 앞선 oracle 교체의 "깊이 4는 speed를 안 고쳤다"와 일치한다.
- **처방 후보**: (a) mode-aware head(분류 + 회귀) 또는 L1(median) 손실, (b) **성공 speed 집합 중 모델 출력에 가장 가까운 것을 목표로 하는 tolerant 손실**. (b)는 이 맵의 성공·실패 정보(틈의 위치와 폭)를 직접 쓰는 방식이라, 앞서 사용자가 말한 "negative가 주는 창 정보"를 speed에서 구현한 것이다.

**한계**: 격자가 20개라 격자 사이에 낀 좁은 speed 창은 틈이나 empty로 잘못 잡힐 수 있다. clean은 scratch를 실패로 보므로 틈의 일부는 scratch일 수 있다(틈에서 실패의 원인이 miss인지 scratch인지는 따로 세지 않았다). 각도는 label로 고정하고 쟀다(실제 정책은 각도도 틀린다). 맵은 val state에서만 만들었다(학습 state용 맵은 아직 없음: tolerant 손실에는 학습 state 약 15k개 × 20 speed ≈ 300k 샷, 약 12~15분이 필요).

### 속도 분포 head: Laplace(L1) vs 구간별 확률(categorical) (TODO 21, 2026-10-12 계획)

**왜**: speed window map에서 모델 speed 실패의 70%가 성공 speed 사이의 **틈**에 떨어졌고(판단표 A), 성공 speed 구간이 둘 이상인 state가 53%였다. label speed는 두 모드(1.0 m/s 76% / ≥2 m/s 24%)라서 상태 → 속도가 1:N이고, 제곱 오차 점 추정은 모드를 구분하지 못할 때 평균(틈)으로 간다. 분포를 내는 head가 맞다(사용자 제안). 각도는 성공 창이 한 덩어리로 보이고 label이 창 중앙이라 점 추정을 유지한다.
**변경 (속도 항만, 각도·선택·made head·encoder는 그대로)**: 깊이 4(hidden 128, `--gnn-layers 4`), 500 epoch, `--pair-supervision --no-wta-fallback --made2-head`, seed 0/1/2.
- **L1**: `--speed-loss l1`. `_pair_loss`의 속도 항을 (Δ·w)² 대신 |Δ·w|로 바꾼다(w = 0.5 m/s가 1인 정밀도 단위). 단일 Laplace의 최우도 = 중앙값이라, 모델은 확률이 높은 덩어리 쪽 값을 낸다. 새 파라미터 없음.
- **BIN**: `--speed-bins 21`. 속도 [0.5, 8.0] m/s에 로그 간격 **중심 21개**(중심 비율 16^(1/20) ≈ 1.15, 0.5·1.0·2.0·4.0·8.0 m/s가 중심)를 두고 pair마다 구간 logit 21개를 내는 새 head. (정정, 결과 보기 전: 처음에는 20개 구간으로 계획했는데 경계가 정확히 1.0 m/s에 놓였다. label의 76%가 정확히 1.0 m/s라서 float 반올림에 따라 인접 두 구간으로 무작위로 갈리는 label 노이즈가 생긴다. 1.0과 2.0이 중심이 되도록 21개로 바꾼다.) 학습: label 속도가 속한 구간에 cross-entropy(가중치 `sel_weight` = 10, 선택 CE와 같은 값), 속도의 제곱 오차 항은 제거(각도 항은 그대로). 추론: 확률이 가장 높은 구간의 중심 속도를 `means[..., 1]`로 쓴다(made head·refine·verify는 그대로 이 값을 proposal로 받음). 확률대로 뽑는 추론(`speed_sample`)은 학습 후 같은 모델로 따로 비교한다.
- **control(같은 코드)**: 같은 옵션에서 위 두 옵션만 끈 깊이 4. 코드를 바꾸므로 이전 깊이 4 결과(`logs/experiments/*_L4_m2w1_*`)와 **소수점까지 같은지**를 확인한다(기본 동작 불변 검증).
**지표**: ① 게임 shots/game (verify+refine 31×7, 500 games, `eval_by_balls`) ② speed window map 재측정: "label 각도 + 모델 speed" 성공률(깊이 4 기준 91.5%)과 실패 중 틈 비율(70.4%) ③ 보조: 각도 오차, verify+refine clean 전체·공 2개.
**"효과 인정"**: control과 seed별로 짝지어 3개 모두 개선 + 평균 차이 ≥ 게임 0.15 shots (또는 ② 성공률 ≥ +2%p). 각도 오차 악화 기준은 +0.009°.

**판단표** (위에서부터 먼저 맞는 행)

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| F | 같은 코드 control이 이전 깊이 4 결과와 다름 | 기본 경로가 바뀜(버그) | 원인을 찾기 전까지 비교 보류 |
| E | L1 또는 BIN의 각도 오차가 control보다 0.009° 이상 나쁨 | 속도 항 변경이 공유 encoder를 해침 | 손실 가중 조정 후 재시도 |
| A | BIN이 효과 인정(게임 또는 ② 성공률)이고 틈 비율이 줄어듦, L1보다 낫다(차이 ≥ 0.1 shots 또는 ② ≥ 2%p) | 분포 head가 틈을 줄인다 | BIN 채택. 다음: 구간 확률의 목표를 teacher 모드 → 속도 맵의 성공 구간(여러 개)으로 바꾸기, 확률대로 뽑기 vs 최고 확률 비교, 혼합 분포 |
| B | L1이 효과 인정이고 BIN과 같은 수준(차이 < 0.1 shots, ② < 2%p) | 중앙값 선택으로 충분 | L1 채택(더 단순) |
| C | ② 성공률·틈 비율은 좋아졌는데 게임은 효과 인정에 못 미침 | 속도는 고쳐졌지만 다른 항목이 병목 | 각도·선택·verify 쪽으로 복귀 |
| D | L1·BIN 모두 ② 변화 < 2%p이고 틈 비율 변화 < 5%p | 출력 분포를 바꿔도 모드를 못 고른다(teacher의 모드 선택이 state에서 예측 불가) | 모드 확률의 목표를 속도 맵(성공 구간)으로 바꾸거나 made head로 모드를 검증하는 쪽, 또는 속도 쪽 접음 |
| G | 위 중 seed 하나만 반대 방향 | 불확실 | 결론 보류, seed 5개 |


**결과: 속도 분포 head (2026-10-12, 깊이 4, 500 epoch, seed 0/1/2, 같은 길이)**

`eval_by_balls`(게임 500판)와 `diagnose_action --parts speedmap`. 9개 run 모두 정상 종료(약 71분). **F행 해당 없음**: 같은 코드 control이 이전 깊이 4(`logs/experiments/*_L4_m2w1_*_2026-10-11@2312`)와 세 seed 모두 best epoch, best val loss, select 게임, 각도 오차까지 동일하다.

| 지표 | control (같은 코드) | **L1** | **BIN** (21구간) |
|---|---|---|---|
| pair head 각도 오차 중앙값 전체 | 0.097° | 0.109° | 0.101° |
| 〃 seed별 | 0.101 / 0.095 / 0.093 | 0.113 / 0.101 / 0.115 | 0.114 / 0.094 / 0.095 |
| 〃 남은 공 2개 | 0.158° | 0.177° | 0.174° |
| 스캔 argmax가 창 안 (seed별) | 86.5% (0.860 / 0.881 / 0.851) | 85.6% (0.869 / 0.890 / 0.802) | 81.3% (0.794 / 0.814 / 0.832) |
| clean **select** 전체 / 공 2개 | 74.5% / 64.5% | 77.7% / 65.4% | 78.1% / 67.6% |
| clean **verify+refine 31×7** 전체 / 공 2개 | 81.3% / 74.9% | 78.5% / 69.5% | 78.3% / 69.8% |
| verify+refine − select (전체) | +6.8%p | +0.8%p | +0.2%p |
| **게임 shots/game** (success) | 3.55 ± 0.10 (84.2%) | 3.78 ± 0.04 (79.1%) | 3.82 ± 0.04 (78.3%) |
| 〃 seed별 | 3.52 / 3.66 / 3.47 | 3.79 / 3.74 / 3.81 | 3.79 / 3.86 / 3.81 |

게임 shots 차이 (head − control, seed별, 양수가 나쁨): L1 0.27 / 0.08 / 0.34 (평균 +0.23), BIN 0.27 / 0.20 / 0.34 (평균 +0.27). **두 variant 모두 3 seed 전부 control보다 나쁘다.**

**speed window map 재측정 (label 각도에서 모델 speed가 들어가는가)**

| 지표 (label 각도 + 모델 speed) | control (같은 코드) | **L1** | **BIN** (21구간) | BIN, 확률대로 뽑기 |
|---|---|---|---|---|
| 성공률 | 91.5% | 93.8% | 93.7% | 91.0% |
| 실패 state | 426 | 307 | 312 | 451 |
| 실패 중 틈(gap) | 70.4% | 43.0% | 43.3% | 53.7% |
| 실패 중 아래(below) | 14.3% | 45.3% | 45.8% | 25.3% |
| 실패 중 위(above) | 7.5% | 1.0% | 0.3% | 13.7% |
| 실패 중 empty | 7.7% | 10.7% | 10.6% | 7.3% |
| 실패한 모델 speed p10 / p50 / p90 (m/s) | 1.36 / 1.57 / 2.22 | 1.00 / 1.08 / 1.55 | 1.00 / 1.00 / 2.00 | 1.00 / 2.00 / 5.28 |

**판단표 적용**
- **F 아님**: 같은 코드 control이 이전 깊이 4와 동일.
- **E: L1에 해당**. 각도 오차가 control보다 +0.012°(기준 +0.009°), seed별 0.113 / 0.101 / 0.115 vs 0.101 / 0.095 / 0.093로 3 seed 모두 나쁘다. BIN은 +0.005°로 기준 미만이다(seed 0만 0.114로 나쁨).
- **A, B 아님**: A는 BIN이 L1보다 나아야 하는데 같은 수준이다(La 93.7% vs 93.8%, 게임 3.82 vs 3.78). B는 효과 인정(게임 개선)이 전제인데 게임이 나빠졌다.
- **D 아님**: 틈 비율이 70.4%에서 43.0% / 43.3%로 줄고 La가 +2.4 / +2.3%p 올랐다(D는 둘 다 작을 때의 행: La 변화 < 2%p이고 틈 비율 변화 < 5%p).
- **C에 해당**: speed 지표(La, 틈 비율, select clean)는 좋아졌는데 게임은 효과 인정은커녕 **오히려 나빠졌다**(표의 C는 "못 미침"이라고만 썼고 악화는 예상하지 않았다). 해석 "속도는 고쳐졌지만 다른 항목이 병목"에 더해 아래 관찰이 중요하다.

**관찰과 해석**
1. **speed head는 pair head 출력(select) 수준에서는 의도대로 동작했다**: select clean이 74.5% → 77.7% / 78.1%(+3.2 / +3.6%p), 틈 실패 70% → 43%.
2. **그러나 상한 대비 회복은 작다**: La가 91.5% → 93.7%로 +2.3%p(label 속도를 쓰면 98.8%까지 가능한 7.3%p 여지 중 약 30%). 실패가 틈에서 **아래**로 옮겨갔다(아래 14.3% → 45.3% / 45.8%, 실패한 모델 speed의 p50이 1.00~1.08 m/s). 모델이 **느린 모드를 골랐는데 그 state는 빠른 모드가 필요한** 경우로, 어느 모드가 필요한지를 state에서 못 맞히는 것이 남은 병목이다. 평균(틈)은 없앴지만 모드 선택 자체는 맞히지 못한다.
3. **전체 파이프라인(verify+refine)에서는 이득이 사라지고 뒤집힌다**: verify+refine이 select보다 주던 이득이 control +6.8%p에서 L1 +0.8, BIN +0.2%p로 줄었고, 전체 verify+refine clean이 81.3% → 78.5% / 78.3%로 낮아졌다. proposal의 speed를 좋게 만들어도 verify+refine 위에 더해지지 않는다. refine의 절대 speed 격자 7개가 이미 hedging을 상당 부분 풀고 있었기 때문으로 보인다(가설, 아래 5의 (a)).
4. **정정**: 앞서 oracle 교체의 "speed를 label로 바꾸면 +8.0%p"는 **select(pair head 출력) 기준** 상한이다. 실제 정책은 verify+refine이라 refine이 speed 문제를 일부 이미 풀고 있어서, 그 8.0%p는 파이프라인의 개선 여지가 아니었다. 이 점을 oracle 결과에서 구분하지 못했다.
5. **원인 가설 (미검증)**: (a) refine의 speed 격자와 중복, (b) 속도 손실 변경이 공유 encoder를 해침(L1: 각도 오차 +0.012°), (c) BIN은 made head의 스캔 적중률이 86.5% → 81.3%(3 seed 모두 하락)로 떨어져 proposal을 기준으로 하는 made head와의 연결이 어긋남. 어느 것이 지배적인지는 가르지 않았다.
6. **BIN 확률대로 뽑기는 도움이 안 된다**: La 93.7%(argmax) → 91.0%(샘플), 틈 43.3% → 53.7%. 게임 shots 3.82(argmax) vs 3.78(샘플)은 표준편차 0.04 근처의 차이다. 어느 쪽도 control(3.55)에 못 미친다.

**결론**: 속도 분포 head(L1, 구간별 확률)는 **채택하지 않는다.** select 수준의 speed 오류는 줄지만 verify+refine 파이프라인에서 게임이 나빠진다(+0.23 / +0.27 shots, 3 seed 모두). 속도 쪽 개선 여지는 **verify+refine 출력 기준으로 다시 재야 한다**(select 기준의 +8.0%p를 그대로 믿지 않는다). 시험하지 않은 것: 구간 수·CE 가중치 등 하이퍼파라미터(한 설정만 시험), 구간 확률의 목표를 teacher 모드가 아니라 속도 맵의 성공 구간으로 바꾸는 2단계, 혼합 분포. 한계: seed 3개, 게임 표준편차 0.04~0.10.

**정정 2: verify+refine에서 실제로 쳐지는 속도는 speed head의 출력이 아니다 (2026-10-12, 학습 없음, 3 seed × 500 games, 임시 스크립트 — 저장소에 넣지 않음)**

질문(사용자): 속도 head가 중앙값/모드를 내도록 바꿨는데도 안 좋아졌다면, 실제 inference에서 여전히 중간값이 나오는 것 아닌가. 확인: 게임 롤아웃에서 매 샷마다 (a) select 모드의 head 출력 속도와 (b) 실제로 치는 verify+refine 속도(절대 격자 7개 중 하나)를 같이 기록.

| 모델 | head 출력 속도 p10 / p50 / p90 (m/s) | head 출력이 중간 구간(1.2~1.9 m/s)인 샷 | 실제로 친 속도가 중간 구간인 샷 | 친 속도가 head 출력의 ±10% 안 |
|---|---|---|---|---|
| control | 1.29 / 1.45 / 1.86 | 90.2% | 15.6% | 0.8% |
| L1 | 0.97 / 1.02 / 1.15 | 5.7% | 18.3% | 79.0% |
| BIN | 1.00 / 1.00 / 1.00 | 0.0% | 0.9% | 27.4% |

실제로 친 속도의 비율 (verify+refine, 절대 격자 값, 괄호는 그 속도에서의 clean 성공률):

| 모델 | 0.75 | 1.00 | 1.25 | 1.50 | 2.00 | 3.00 / 5.00 |
|---|---|---|---|---|---|---|
| control | 53.8% (83.4%) | 28.2% (79.7%) | 10.0% (79.1%) | 5.7% (71.9%) | 1.6% (53.0%) | 0.4% / 0.4% (0% / 0%) |
| L1 | 5.9% (70.2%) | 74.8% (77.0%) | 16.8% (67.6%) | 1.6% (50.6%) | 0.3% | 0.2% / 0.4% |
| BIN | 70.8% (76.5%) | 26.2% (72.2%) | 0.2% | 0.7% | 1.4% | 0.3% / 0.4% |

- **게임에서 치는 속도는 head의 출력이 아니라 made head가 격자 7개 중에서 고른 값이다.** control에서 head는 90%의 샷에서 중간값(중앙값 1.45 m/s)을 내지만, 실제로는 0.75 m/s(54%)나 1.0 m/s(28%)를 치고 중간 구간을 치는 샷은 15.6%뿐이다. head 출력의 ±10% 안에서 친 샷은 0.8%다.
- 그래서 앞의 speed map의 "실패의 70%가 틈"은 **head의 날것 출력**에 대한 것이고, 실제 정책이 치는 속도에 대한 것이 아니었다. 같은 이유로 head를 고쳐도 최종 속도는 거의 안 바뀌는 부분이 있다(control에서 head 출력과 친 속도가 거의 무관).
- L1 / BIN의 head는 의도대로 중간값을 없앴다(중앙값 1.02 / 1.00). 그러나 친 속도는 made head의 선택이라 L1은 1.0(75%), BIN은 0.75(71%)로 바뀌었고, **같은 속도에서도 clean 성공률이 control보다 낮다**(1.0 m/s: 79.7% → 77.0% / 72.2%, 0.75 m/s: 83.4% → 70.2% / 76.5%; 상태 구성이 달라 엄밀한 비교는 아님). 즉 악화는 속도 선택이 아니라 각도·pair 선택·made head 쪽 변화에서 왔을 가능성이 크다(L1: 각도 오차 +0.012°, BIN: made head 스캔 적중률 −5.2%p; 미분리).
- 이전 절의 "refine의 speed 격자와 중복" 가설은 이 측정으로 지지된다.
- 이 측정으로 oracle 교체와 speed map의 대상이 **정책이 실제로 치는 action이 아니었다**는 것이 확인됐다. verify+refine 출력 기준의 재측정이 필요하다(roadmap 우선순위 1번).

### verify+refine 출력 기준 oracle 교체와 실패 예산 (2026-10-12 계획, 학습 없음)

**왜**: 앞선 oracle 교체와 speed map은 pair head의 날것 출력(select) 기준이었다. 실제 정책은 verify+refine이라 made head가 각도 31 × 절대 speed 7개 격자에서 고른 값을 친다(위 "정정 2": head 출력과 친 속도가 거의 무관). 그래서 그 상한(각도 +9.9%p, speed +8.0%p)은 파이프라인의 개선 여지를 대표하지 못했다. **정책이 실제로 치는 최종 action 기준으로** 남은 실패가 각도, 속도, pair 선택 중 무엇인지 다시 잰다.
**방법**: 깊이 4(대조로 control 2층, 학습 길이 실험의 깊이 6 1500 epoch), seed 0/1/2, label이 있는 모든 val state. label pair(후보 0)에서 made head의 verify+refine 격자(각도 ±0.3° 31개 × 절대 speed 7개)로 고른 **refined action**을 기준 R로 하고 실제로 친다.
- R: refined action 그대로 (앞선 "label pair + refine"과 같은 값이어야 한다: 하네스 일치 확인)
- Rs: refined 각도 + **label speed**, Ra: **label 각도** + refined speed
- L: label action (teacher action, 성공 ≈ 98.8%여야 함)
- P: 정책이 verify로 고른 pair의 refined action (실제 게임에서 치는 것)
**실패 예산** (전체 실패 100% − P):
- 선택 손실 = R − P (정책이 pair를 잘못 골라서 잃는 것)
- 각도 손실 = Ra − R, speed 손실 = Rs − R (label pair에서 refine 후에도 남는 각도·speed 오차)
- 그 외(결합·기타) = (L − R) − 각도 손실 − speed 손실, 그리고 L에도 못 미치는 나머지 = 100% − L
**판단표** (R, Rs, Ra, L로 `oracle_row`와 같은 규칙; 위에서부터 먼저 맞는 행. 표본 오차 ±1%p)

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| A | L < 97% | 하네스 문제 | 해석 보류 |
| B | 각도 손실 + speed 손실 < 0.5 · (L − R) | 각도와 speed가 함께 맞아야 한다(결합) | 따로 고치는 방법은 효과가 작다. 결합을 다루는 쪽(joint 모델링, 탐색) |
| C | 각도 손실 ≥ 2 · speed 손실 (차이 ≥ 2%p) | refine 후에도 **각도**가 지배 | 각도 정밀도 쪽(loss 스케일, 폭 가중 손실, 표현, 탐색). speed 쪽은 접는다 |
| D | speed 손실 ≥ 2 · 각도 손실 (차이 ≥ 2%p) | refine 후에도 **speed**가 지배 | speed 개선이 의미 있다. 단 proposal(head 출력)이 아니라 made head가 고르는 값을 겨냥해야 한다(격자 세분화, made head의 speed 정확도) |
| E | 그 외 | 둘 다 기여 | 병행 |
| 보조 | 선택 손실(R − P)이 각도·speed 손실보다 크다 | pair 선택이 최우선 | 선택/verify 쪽 |

**결과: verify+refine 출력 기준 oracle 교체와 실패 예산 (2026-10-12, 학습 없음, 3 seed 합침)**

`diagnose_action --parts vroracle`, label이 있는 val state 4989개. **하네스 일치 확인**: L(label action)이 98.8%로 A행 기준 97%를 넘고, 깊이 4의 R이 86.3%, P가 82.2%로 앞선 분해("label pair + refine" 86.3%, 정책 pair 82.2%)와 같다(control도 81.8% / 78.3%로 같음).

| clean (실제로 침, 전체) | **깊이 4** | control (2층) | 깊이 6 (1500 epoch) |
|---|---|---|---|
| P: 정책이 고른 pair + refine (게임에서 치는 것) | 82.2% | 78.3% | 82.3% |
| R: label pair + refine | 86.3% | 81.8% | 87.0% |
| Rs: refined 각도 + **label speed** | 89.3% | 84.2% | 91.1% |
| La: **label 각도** + refined speed | 94.0% | 93.5% | 93.3% |
| L: label action (teacher) | 98.8% | 98.8% | 98.8% |

| 실패 예산 (정책 P의 실패, %p) | **깊이 4** | control (2층) | 깊이 6 (1500 epoch) |
|---|---|---|---|
| pair 선택 (R − P) | +4.2 | +3.5 | +4.7 |
| **각도** (La − R) | +7.7 | +11.7 | +6.3 |
| **speed** (Rs − R) | +2.9 | +2.3 | +4.1 |
| 둘이 함께 / 기타 | +1.9 | +3.0 | +1.4 |
| label action으로도 못 고침 (1 − L) | +1.2 | +1.2 | +1.2 |
| **P의 실패 합계** | **17.8** | **21.7** | **17.7** |
| 판단표 행 | **C** | **C** | **E** |

| 깊이 4, 남은 공 수별 실패 예산 (%p) | 남은 공 1 | 남은 공 2 | 남은 공 3 |
|---|---|---|---|
| pair 선택 | +2.0 | +2.9 | +6.5 |
| 각도 | +5.5 | +16.3 | +6.4 |
| speed | +2.5 | +3.0 | +3.2 |
| 둘이 함께 / 기타 | +2.7 | +2.7 | +0.9 |
| 못 고침 | +2.5 | +0.2 | +0.4 |


**판단표 적용**: 깊이 4 → **C** (각도 손실 +7.7%p ≥ 2 × speed 손실 +2.9%p, 차이 ≥ 2%p). B(결합)는 아니다: 각도 + speed 손실 10.6%p가 0.5 × (L − R) = 6.2%p보다 크다. control도 C, 깊이 6(1500 epoch)은 각도 +6.3 vs speed +4.1(비율 1.5배, 2배 미만)라 E.

**해석**
- **정책의 실패 17.8%p(깊이 4)는 각도 7.7 (43%), pair 선택 4.2 (23%), speed 2.9 (16%), 결합·기타 1.9, label로도 못 고침 1.2로 나뉜다.** 최종 행동 기준으로 가장 큰 것은 각도다.
- **speed의 여지는 select 기준 +8.0%p가 아니라 2.9%p다.** select 기준 oracle(각도 +9.9 / speed +8.0)에서 verify+refine 기준(각도 +7.7 / speed +2.9)으로 오면서 speed 여지가 크게 줄었다(refine이 풀어 준 부분). 각도 여지는 2.2%p만 줄었다. 앞선 "속도 head가 게임에서 이득이 없었다"와 일치한다: 고칠 수 있는 속도 여지가 애초에 작았다.
- **남은 공 2개 state의 각도 손실이 +16.3%p로 두드러진다**(1개 +5.5, 3개 +6.4). pair 선택은 3개 state에서 +6.5%p로 각도와 비슷한 크기다.
- **각도와 speed의 결합 손실(기타)은 1.9%p로 작다**: 두 오차는 거의 독립적으로 더해진다. 따라서 각도를 먼저 구하고 speed를 조건부로 구하는 cascade 구조나 결합 모델링이 얻을 수 있는 몫은 이 분해 안에서 최대 약 2%p다(구조 변경 비용 대비 작음).
- **깊이의 효과**: control → 깊이 4에서 각도 손실이 +11.7 → +7.7로 줄고 speed(+2.3 → +2.9)와 선택(+3.5 → +4.2)은 그대로다. 깊이 6(1500 epoch)은 각도 +6.3로 더 낮지만 speed +4.1와 선택 +4.7이 커져 P가 같다(82.2% vs 82.3%).
- 이전 격자 oracle(해가 99.2% state에 있음)과 합치면: 각도 해는 격자 안에 거의 항상 있는데 made head가 고르지 못해 각도 손실 7.7%p가 남는다. 따라서 각도 개선은 (1) pair head의 proposal 정밀도 또는 (2) 각도를 고르는 verifier(made head 또는 시뮬레이터) 중 하나를 겨냥해야 한다.
- 한계: label pair(후보 0)에서의 측정이고, 선택 손실은 정책 pair vs label pair의 차이다. label과 다른 pair가 더 쉬운 경우는 반영하지 않는다. 격자의 speed는 7개다.

### 각도와 speed의 결합: 2차원 성공 지도 (2026-10-12 계획, 학습 없음)

**질문 (사용자)**: 이전 결과에서 성공·실패가 각도와 "빠르게 / 살살" 조합에 따라 갈리지 않았는가. 그렇다면 "각도·speed 결합 손실이 1.9%p로 작아서 cascade가 불필요하다"는 결론이 틀릴 수 있다.
**지금까지의 근거와 한계**: (a) oracle 교체의 결합 항(verify+refine 기준 1.9%p, select 기준 −0.6%p)은 **모델 오차와 label의 교체**로 본 간접 증거다. 각도·speed가 독립적으로 각각 가끔 틀리는 경우와 성공 영역이 기울어진 경우를 구분하지 못한다. (b) speed map은 **label 각도에 고정**해서 speed를 바꿨다(성공 speed 구간이 둘 이상인 state 53%). (c) 각도 스캔은 **label speed에 고정**했다. 즉 **각도 창의 위치가 speed에 따라 움직이는지(기울기)** 는 직접 재지 않았다.
**방법 (임시 스크립트, 저장소에 넣지 않음)**: 깊이와 무관한 측정(모델 없음). val label state를 seed당 150개(450개), label pair에서 각도 offset 25개(label 각도 ±0.6°, 0.05° 간격) × speed 10개(0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.6, 3.5, 5.0, 8.0 m/s)를 전부 쳐서 (각도, speed) 성공 지도를 만든다. 성공 = 의도한 pair가 들어가고 scratch 없음(각도 스캔과 같은 정의). label speed의 각도 창(`scan_truth`)이 스캔 안에 완전히 들어오는 state만 기준으로 삼고, 다른 speed에서 창이 스캔 안에 들어오는 (state, speed) 쌍마다 **창 중심의 이동** Δc = c(speed) − c(label speed)를 label 창 반폭으로 나눈다.
**판단표** (비교 가능한 쌍 ≥ 100개일 때; 위에서부터 먼저 맞는 행)

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| D | 비교 가능한 (state, speed) 쌍 < 100 | 표본 부족 | 판단 보류, state를 늘려 재측정 |
| A | \|Δc\| ≤ 반폭인 쌍이 ≥ 90%이고 \|Δc\|/반폭의 중앙값 < 0.25 | 각도 창이 speed에 거의 안 움직인다(곱 구조에 가까움) | "결합 작음, cascade 불필요" 유지 |
| B | \|Δc\| ≤ 반폭인 쌍이 < 70% | 각도 창이 speed에 의존한다(기울어진 영역) | 결합 모델링(각도를 speed 조건부로 또는 후보를 결합 평가)이 정당화됨 |
| C | 그 사이 | 부분적 의존 | 상세 분해(speed 구간별, 2개 state만) 필요 |

### 각도 loss 가중치: 수동 4배 vs Kendall 자동 조절 (2026-10-12 계획)

**왜**: verify+refine 기준 정책 실패 17.8%p 중 각도가 7.7%p(43%)로 최대이고, 남은 공 2개 state에서는 16.3%p다. 각도 가중치를 바꿔 본 실험은 깊이 4 기준으로 없다(이전은 MLP 시절 1회). 사용자 제안: 항별 가중치를 자동으로 조절하는 Kendall 등(2018)의 uncertainty weighting. 먼저 수동으로 각도 가중치를 올리면 도움이 되는지 보고, 같은 배치에서 자동 조절과 비교한다.
**우려 (결과 보기 전의 예상)**: Kendall 방식은 최적에서 σ² ≈ 항의 평균 손실이 되어 손실이 큰(꼬리가 두꺼운) 항의 가중치를 내린다. 각도 손실은 train loss의 66%가 1.4% row에서 나올 만큼 꼬리가 두꺼워 각도 가중치가 **내려갈** 가능성이 있다. 이는 각도 정밀도를 올리려는 방향과 반대다(예상일 뿐 검증 전).
**실험 (깊이 4, 500 epoch, `--pair-supervision --no-wta-fallback --made2-head`, seed 0/1/2, 출력 `logs/experiments_loss`)**
- **기준**: 같은 코드 깊이 4(이미 있음: `logs/experiments_speed/*_L4_m2w1_*`, 이전 깊이 4와 동일함을 확인한 것).
- **S01 (수동)**: `--angle-scale-deg 0.1`. 각도 항의 가중치가 4배(제곱 오차 기준)가 된다. 이것만 다르다(`results.json` 인자로 확인).
- **KEN (자동)**: `--uncertainty-weights`. 각도·속도·선택 세 항에 학습 가능한 `s = log σ²`를 두고 `손실 = Σ exp(−s_i) · L_i + s_i`. L_i는 현재 손실의 각 항(각도 제곱 오차, 속도 제곱 오차, 선택 CE × `sel_weight`)이라 **s = 0이면 기준과 정확히 같다**. s는 [−3, 3]으로 제한. 학습된 s의 최종값을 기록해서 각도 가중치가 올랐는지 내렸는지 본다. (코드는 따로 작성 후 합류)
**지표**: ① pair head 각도 오차 중앙값(전체, 남은 공 2개) ② 게임 shots/game(verify+refine, 500판) ③ 보조: verify+refine clean 전체·공 2개, select clean, 실패 예산(`vroracle`)의 각도 손실.
**"효과 인정"**: 기준과 seed별로 짝지어 3개 모두 개선 + 평균 차이 ≥ 각도 0.009° (게임은 0.15 shots).

| # | 관측 (S01 vs 기준) | 해석 | 다음 행동 |
|---|---|---|---|
| F | S01의 인자가 `angle_scale_deg` 외에 기준과 다름 | 실험 설정 오류 | 비교 보류 |
| D | S01의 각도 오차가 기준보다 0.009° 이상 나쁨 | 각도 가중치를 올리면 다른 항을 희생하거나 최적화가 불안정 | 반대 방향(0.4)을 확인하거나 이 축을 접음 |
| A | 각도 오차가 효과 인정이고 게임도 개선(≥ 0.15 shots 또는 verify+refine clean ≥ +2%p) | 각도 가중치가 병목이었다 | 더 올리기(0.05), 창 폭 가중 손실(TODO 2)로 이어감 |
| B | 각도 오차는 효과 인정인데 게임·verify+refine clean 개선은 기준 미만 | 각도 proposal은 좋아졌지만 verifier/선택이 병목 | 각도 verifier 쪽(TODO 3) |
| C | 각도 오차 변화 < 0.009° | 각도 가중치는 병목이 아니다 | 가중치 조정은 접고 표현·verifier 쪽 |
| G | seed 하나만 반대 | 불확실 | seed 5개 |

KEN은 같은 표로 읽되 S01과 비교한다: KEN이 S01 이상이고 기준보다 좋으면 "자동 조절이 수동과 같은 방향을 찾았다", 기준과 같거나 나쁘면 "자동 조절은 이 문제에서 필요 없다".

**결과: 각도·speed 2차원 성공 지도 (2026-10-12, 학습 없음, 450 state × 각도 25 × speed 10, 임시 스크립트 — 저장소에 넣지 않음)**

**1. 각도 창이 speed에 따라 움직이는가 → 판단표 A** (label speed의 각도 창이 스캔 안에 완전히 드는 state 242개, 비교 가능한 (state, speed) 쌍 1,883개 ≥ 100)
- 창 중심의 이동 |Δc| / label 창 반폭: 중앙값 **0.14**, p75 0.28, p90 0.47. 반폭 이내인 쌍 **98.7%**(반폭의 절반 이내 91.6%). 절대 이동은 중앙값 0.031°, p90 0.125°(label 창 폭 중앙값 0.55°, 다른 speed의 창 폭 0.45°).
- slow(< 1.2 m/s) / middle / fast(≥ 2.2) 98.8 / 99.6 / 98.1%, 남은 공 1 / 2 / 3개 97.6 / 98.4 / 99.8%로 어디서나 비슷하다.
→ **창이 있는 speed에서는 각도 창의 위치가 speed에 거의 안 움직인다.**

**2. 그런데 label 각도에서 실패하는 (state, speed) 칸은 각도를 옮기면 성공하는가** (저장한 지도로 계산)
- label 각도에서 실패하는 칸 858개(전체 4,500칸의 19%) 중 **53.0%는 ±0.6° 안의 어떤 각도에서 성공**한다(47.0%는 어떤 각도로도 안 됨).
- **"틈" 칸**(label 각도에서 더 느린 speed와 더 빠른 speed는 성공하는데 그 speed만 실패, 203칸)은 **92.1%가 각도를 옮기면 성공**한다. 필요한 이동은 작다: 가장 가까운 성공 각도까지 p25 / p50 / p75 / p90 = 0.05 / 0.05 / 0.15 / 0.30°(0.05°는 스캔 해상도의 최솟값이라 중앙값은 "해상도 이하"), label 반폭의 중앙값 0.25배, 1 반폭을 넘는 경우 5.3%(틈 칸은 2.2%).
- speed별로 보면 느린 speed의 실패(0.6 m/s에서 12%만 각도로 구제됨)는 각도와 무관하게 안 되고, 빠른 speed의 실패(2 m/s 이상 69~78%)는 각도로 대부분 구제된다.

**해석**
- **질문(성공·실패가 각도와 빠르게/살살의 조합으로 갈리는가)의 답은 "맞다, 다만 이동량이 작다"**: label 각도에서 실패하는 칸의 절반 이상, 틈 칸의 92%는 각도를 0.05~0.15°만 옮겨도 성공한다. 즉 일부 speed에서 창이 label 각도 바로 옆으로 살짝 이동해 있다. 창 폭(중앙값 0.45~0.55°)에 비해 이동량(중앙값 0.03°)은 작지만, 창이 좁은 state에서는 label 각도 한 점이 가장자리 밖에 놓여 실패한다.
- **정정**: speed map에서 "모델 speed 실패의 70%가 두 모드 사이 틈"을 mode hedging의 증거로 읽은 것은 과했다. 그 측정은 각도를 label 각도 한 점에 고정했기 때문에, "그 speed가 안 된다"와 "label 각도가 그 speed의 창 가장자리 바로 밖이다"를 구분하지 못했다. 틈의 92%는 후자(각도 0.05° 안팎의 이동으로 해소)다. speed 분포 head가 게임에서 이득이 없었던 것과 일치한다.
- **cascade / 결합 모델링 결론은 직접 근거를 얻어 유지된다**: 각도 창 이동이 작고(98.7%가 반폭 이내) 틈이 작은 각도 이동으로 풀리므로 각도를 speed에 조건부로 구해서 얻을 몫은 작다(oracle 교체의 결합 항 1.9%p와도 맞음). 단 결합이 전혀 없는 것은 아니다.
- 한계: state 450개·speed 10개, 각도 해상도 0.05°, 틈 칸 중 46%는 성공 영역이 스캔 가장자리(±0.6°)에 닿아 더 멀리 있을 수 있다. label 창이 스캔보다 넓은 state(약 45%)는 이동량 계산에서 빠졌다. 의도한 pair 기준.

### made head(verify+refine)의 실패 프로파일: 나머지 14%는 왜 틀리는가 (2026-10-12, 학습 없음, 임시 스크립트 — 저장소에 넣지 않음)

**질문 (사용자)**: made head가 해를 찾는 비율이 84%일 때 나머지는 왜 틀리는가. raw 데이터에 속도가 빠르다는 식의 경향이 있는가. (84.8%는 격자 oracle 표본 900개의 값이고, 전체 4,989개에서 label pair + verify+refine 행동 R은 86.3%, 실패 13.7%다.)
**방법**: 깊이 4, seed 3개, label이 있는 val state 4,989개에서 R(label pair의 pair head proposal 주변 각도 31 × speed 7 격자에서 made head가 고른 행동)을 실제로 쳐서 성공 여부와 특징을 기록: 고른 speed, teacher(label)의 speed, 고른 각도와 label 각도·proposal의 차이, 컷 각도(실제 테이블 크기 0.991 × 1.981 m로 환산), cue→ball·ball→pocket 거리, 포켓 종류, 남은 공 수. 실패 = 포켓이 없거나 scratch. 실패 13.7%는 scratch 5.5%p + 포켓 없음(miss) 8.1%p.

| 구분 | 구간 | 상태 비중 | 실패율 |
|---|---|---|---|
| 고른 speed (m/s) | 0.75 / 1.0 / 1.25 / 1.5 / 2.0 / 3.0 / 5.0 | 53 / 29 / 9 / 5 / 2.5 / 0.9 / 0.8% | 9.2 / 14.4 / 19.3 / 23.1 / 25.4 / 59.1 / 73.8% |
| teacher의 speed 모드 | 느림(≤ 1.5) / 빠름(> 1.5) | 76% / 24% | **7.9% / 31.7%** |
| \|고른 각도 − label 각도\| (°) | < 0.05 / 0.05~0.15 / 0.15~0.3 / 0.3~0.6 / > 0.6 | 28 / 39 / 20 / 8 / 5% | 3.4 / 7.4 / 16.7 / 42.6 / 59.3% |
| \|고른 각도 − proposal\| (°) | < 0.01 / 0.01~0.1 / 0.1~0.2 / > 0.2 | 35 / 60 / 3.5 / 1.6% | 12.5 / 12.1 / 32.4 / 56.2% |
| 컷 각도 (°) | < 15 / 15~30 / 30~45 / 45~60 / > 60 | 34 / 30 / 20 / 11 / 5% | 8.7 / 10.7 / 16.8 / 17.3 / 44.5% |
| cue→ball 거리 | 가까움 / 중간 / 멀음 (3분위) | 33 / 34 / 33% | 11.4 / 11.7 / 17.9% |
| 포켓 | corner / side | 58% / 42% | 15.4% / 11.2% (실패 중 scratch 32% / 56%) |
| 남은 공 | 1 / 2 / 3 | 38 / 17 / 45% | 13.2 / 22.2 / 10.9% |

**teacher 모드 × 고른 speed**

| teacher 모드 | 고른 speed | 상태 수 | 실패율 | 전체 실패 중 비중 |
|---|---|---|---|---|
| 느림 (3,807) | 느림 ≤ 1.0 / 중간 / 빠름 ≥ 2.0 | 88% / 11% / 1.2% | 7.3 / 13.6 / 37.8% | 36 / 8.4 / 2.5% |
| **빠름 (1,182)** | 느림 / 중간 / 빠름 | 62.5% / 23% / 14% | 27.6 / 31.2 / 43.1% | 30 / 12.6 / 10.6% |

**관찰**
1. **teacher가 빠른 샷을 쓰는 state(24%)가 전체 실패의 53%를 차지한다**(실패율 30.6% vs 8.4%). 그 실패의 57%가 scratch다. 이 state에서 정책이 빠른 speed를 고르는 것은 14%뿐이고 62.5%는 느린 speed를 고르는데, 느린 speed를 골라도 27.6%, 빠른 speed를 골라도 43.1% 실패한다. 각도가 label에서 0.15° 이내여도 이 state는 20.3% 실패한다(느린 모드 state는 1.9%).
2. **이 경향은 컷 각도·거리와 겹치지만 그것만으로 설명되지 않는다**: 빠른 모드 비중은 컷 < 15°에서 14%, > 60°에서 64%로 오르지만(겹침), 컷 각도 구간마다 빠른 모드 state가 느린 모드보다 2~5배 더 실패한다(25 vs 6%, 25 vs 7%, 33 vs 10%, 26 vs 13%, 51 vs 33%). cue→ball 거리 3분위에서도 같은 방향(27 vs 6%, 25 vs 8%, 43 vs 12%).
3. **고른 speed가 빠를수록 실패율이 단조롭게 오른다**(0.75 m/s 9.2% → 5.0 m/s 73.8%). 다만 2 m/s 이상은 상태의 4%뿐이고 실패의 약 13%다. 가장 큰 덩어리는 느린 speed를 고른 상태(53%·29%)의 실패 66%다.
4. **각도 오차가 가장 강한 단일 요인**: label과 0.3° 넘게 어긋난 상태(13%)가 실패의 약 47%를 만든다. 0.05° 이내면 3.4%만 실패한다(그중 67%가 scratch).
5. **made head가 proposal에서 0.1° 넘게 옮기면 대개 틀린다**: 0.1~0.2°는 실패율 32%, 0.2° 초과는 56%(상태의 5%). 0.1° 이내 조정은 12%대로 proposal을 그대로 쓸 때와 같다. made head의 큰 보정은 이득이 아니라 위험이다(앞선 "made head가 proposal에 앵커돼 있고 각도 해상도가 없다"와 일치).
6. **기하**: 컷이 얇을수록(> 60°에서 44.5%), cue→ball이 멀수록(17.9%), 남은 공 2개(22.2%)일수록, corner 포켓일수록 실패가 많다. side 포켓 실패는 56%가 scratch다.

**한계**: 상관이지 인과가 아니다(컷 각도, 거리, teacher 모드가 서로 겹침). 각 요인의 기여는 더해지지 않는다(예: 각도 > 0.3°의 47%와 빠른 모드의 53%는 겹침). label pair(후보 0) 기준이고 teacher가 빠른 샷을 고른 이유는 데이터에 없다. 빠른 모드 state에서 정책이 느린 speed를 고르는 것이 틀린 모드 선택인지, 느린 speed로도 가능한 샷을 단지 못 친 것인지는 가르지 않았다.

### made head 실패 케이스의 패턴: 지도·비지도 분석 (2026-10-12, 학습 없음, 탐색적 분석, 임시 스크립트 — 저장소에 넣지 않음)

**질문 (사용자)**: made head의 선택이 틀리는 나머지 약 14%에 속도 말고 데이터상의 패턴이 있는가. GMM, t-SNE, DBSCAN으로 클러스터링해 볼 것.
**방법**: 깊이 4(500 epoch), seed 3개, label이 있는 val state 4,989개. 대상 행동 R = label pair에서 made head가 verify+refine 격자(각도 31 × speed 7)에서 고른 행동, 실패 = 포켓 없음 또는 scratch(681개, 13.7%). 실패 중 made head의 후보 격자 안에 성공하는 후보가 있었는지도 시뮬레이터로 확인(early exit). **클러스터링 입력은 state 기하 14개만**(cut 각도, cue→ball·ball→pocket 거리, 포켓 종류, ball·cue 위치, ball의 쿠션 거리, cue→ball·ball→pocket 경로의 다른 공까지 최단거리, 0.2 m 안의 이웃 공 수, 남은 공 수, 포켓 입구 편차). speed, teacher 모드, 각도 오차, 결과는 입력에서 제외하고 사후 설명에만 썼다. 그림: `logs/diagnose_action/figs/mh_tsne.png`, `mh_clusters.png`(logs는 git에 포함되지 않음).

**0. 실패의 94%는 "해가 있는데 못 고른" 선택 실패**: 실패 681개 중 made head의 후보 격자에 성공 후보가 있었던 것이 640개(94.0%, 전체 상태의 12.8%). 해가 격자에 아예 없던 실패는 0.9%뿐이다.

**A. 패턴이 있는가 (5-fold 교차검증 AUC, 0.5 = 패턴 없음)**

| 입력 | AUC |
|---|---|
| state 기하 14개만 | **0.801 ± 0.024** |
| (대조) 같은 입력, 결과를 섞음 | 0.496 ± 0.020 |
| 기하 + teacher speed | 0.839 ± 0.017 |
| 고른 각도의 label 대비 오차만 | 0.791 ± 0.020 |
| made head 확신도 P(chosen)만 | 0.689 ± 0.017 |
| 기하 + teacher speed + 고른 speed + 각도 오차 + 확신도 | 0.933 ± 0.016 |
| "선택 실패"로 한정, 기하만 / 기하 + teacher speed | 0.794 / 0.833 |

→ **state 기하만으로 실패를 예측할 수 있다(AUC 0.80).** teacher speed를 더해도 +0.04뿐이라 speed는 부차적이다.

**B. 기하로 전체 state 클러스터링**: GMM(K=8, full covariance)의 BIC는 K에 따라 계속 줄어(2: 115,172 → 10: 19,535 → 11: 16,436 → 12: 18,717) 뚜렷한 팔꿈치가 없고 silhouette 0.155로 낮다 — **state는 잘 분리된 덩어리가 아니라 연속적으로 분포한다.** 다만 분할은 재현된다(두 번 적합한 결과의 ARI 0.99). 클러스터별 실패율은 9.0% ~ 24.3%이고, 결과를 섞은 permutation 검정에서 클러스터 간 최대 차이(10.7%p)가 우연(95퍼센타일 4.9%p)보다 크다(p < 0.001).

| 클러스터 | n | 실패율 | lift | z | 특징 |
|---|---|---|---|---|---|
| 7 | 218 | **24.3%** | 1.78 | 4.6 | corner 포켓, cue→ball 0.78 m(멂), ball이 쿠션에 가까움(0.16 m), ball→pocket 0.60 m |
| 6 | 315 | **22.9%** | 1.67 | 4.8 | corner 포켓, cue→ball 0.75 m, ball이 쿠션에 가까움(0.16 m), 경로 근처 다른 공(0.55 m) |
| 3 | 341 | 16.1% | 1.18 | 1.3 | side 포켓, 중간 거리 |
| 5 | 341 | 14.7% | 1.07 | 0.5 | 다른 공이 목표 공 바로 옆(0.14 m) |
| 0 / 2 / 4 | 500 / 1,410 / 782 | 13.4 / 13.1 / 13.0% | 0.96~0.98 | ≈ −0.5 | side·corner, 경로 깨끗 |
| 1 | 1,082 | **9.0%** | 0.66 | −4.5 | side 포켓, 거리 짧음(0.42 / 0.38 m) |

DBSCAN(PCA 6성분, eps 1.50): 큰 덩어리 하나와 **이상치 148개(3%)** 로 갈린다. 이상치 state(기하가 특이한 경우)는 실패율 **32.4%**(lift 2.4)다.

**C. 실패 케이스만 클러스터링** (681개; 기하 + 고른 speed + teacher speed + 각도 오차 + 확신도 + scratch): K = 2~7 모두 silhouette 0.15~0.18로 **뚜렷한 실패 유형은 없다.** 대략 K=4로 나누면 (n 249) 얇은 컷 41°·먼 거리·빠른 teacher·scratch 52% / (n 222) corner 먼 거리에 다른 공이 경로 근처·각도 오차 0.42°·scratch 19% / (n 159) side 포켓 짧은 거리·scratch 57%·확신도 0.82 / (n 51) made head가 proposal에서 크게 벗어난 후보를 고름(각도 오차 3.15°).

**D. made head의 확신도**: 고른 행동의 P(made)와 실제 성공률 — 0.50~0.70 구간 평균 P 0.62 → 성공 75.8%, 0.70~0.85: 0.79 → 88.4%, 0.85~0.95: 0.90 → 92.4%, ≥ 0.95: 0.96 → **91.1%**(n 179). 확신도가 0.85를 넘으면 성공률이 더 오르지 않는다. 실패한 샷의 확신도 중앙값은 0.73(성공한 샷 0.85)이고 실패의 27%는 확신도 ≥ 0.85였다. 확신도만으로는 실패를 약하게만 가른다(AUC 0.689).

**해석**
1. **패턴은 있다.** 속도가 아니라 **기하**다: corner 포켓 + 먼 cue ball + 쿠션 가까운 공의 조합(실패율 23~24%), 기하가 특이한 이상치(32%)가 어렵고, side 포켓 + 짧은 거리(9%)가 쉽다.
2. **그러나 연속적인 경향이지 분리된 유형이 아니다.** silhouette 0.15, DBSCAN이 한 덩어리 + 이상치라서 K=8 클러스터는 설명용 분할이다. 실패 케이스끼리도 뚜렷한 유형으로 갈리지 않는다.
3. **실패의 94%는 made head가 성공 후보를 못 고른 것**이다. made head의 확신도는 0.85 이상에서 정보가 없고 자신의 실패를 잘 가르지 못한다. 반면 state 기하는 실패를 AUC 0.80으로 예측한다 → made head가 이미 가진 입력(embedding, 행동, proposal 대비 차이)에서 이 기하 신호를 충분히 못 쓰고 있을 가능성이 있다(가설). 이는 열린 결정 1(made head에 명시적 기하 입력을 허용할지)을 뒷받침하는 정황이다. 앞선 기하 입력 진단(made head 스캔 적중 95%)과 방향이 같다.

**한계**: 관찰이고 인과가 아니다. 클러스터링 입력은 내가 고른 기하 14개이고 다른 특징이면 다른 분할이 나올 수 있다. t-SNE의 거리는 해석하지 않았다(시각화용). label pair(후보 0)에서의 R이고 시뮬레이터 후보 확인은 made head의 격자 안에서만 했다. 그림의 팔레트 검증 스크립트(node)는 환경 문제로 실행하지 못했고, 스킬이 이미 검증한 슬롯 1·2(파랑·주황)를 값 그대로 썼다.

### made head가 oracle(완벽한 verifier)일 때의 상한 (2026-10-12, 학습 없음, 임시 스크립트 — 저장소에 넣지 않음)

**질문 (사용자)**: made head를 oracle로 바꿔서, 후보를 모두 시뮬레이션해 들어가는 것만 고르면 숫자가 얼마나 개선되는가.
**방법**: 깊이 4(500 epoch), seed 0/1/2. 각 샷에서 verify로 고른 pair의 후보 217개(각도 31 × 절대 speed 7)를 **made head 점수 순**으로 정렬하고, 그 순서대로 시뮬레이터에 쳐서 **처음으로 성공(포켓 + scratch 없음)하는 후보**를 쓴다(없으면 made head의 1순위). 별도의 시뮬레이터 인스턴스로 후보를 시험하고 실제 게임 환경에서 그 행동을 친다.
- A: teacher val state 4,989개에서 상위 k개 안에 성공 후보가 있는 비율(success@k)
- B1: 게임 500판 × 3 seed, verify로 고른 pair만 검증(샷당 최대 217회)
- B2: B1에서 해가 없으면 verify 순서의 다음 pair들도 검증(pair당 상위 20개)

**A. made head 순위의 success@k** (정책이 verify로 고른 pair, n = 4,989)

| k | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 217 |
|---|---|---|---|---|---|---|---|---|
| success@k | 82.2% | 83.0% | 83.7% | 84.9% | 87.1% | 89.1% | 91.7% | **98.8%** |

k = 1이 82.2%로 앞선 "정책 pair + refine"(P)과 같다(하네스 일치). 217개 안에 해가 없는 경우는 1.2%뿐이다. 첫 성공까지 시뮬레이터 확인 평균 12.8회(상한 217), 첫 성공 순위의 중앙값은 1.

**B. 게임 (500판 × 3 seed = 1,500판)**

| | shots/game | 샷 성공률(실제 clean) | 시뮬레이터 확인/샷 (평균 / 중앙값 / p90) |
|---|---|---|---|
| 현재 정책 (verify+refine, made head 1순위) | **3.55** | 80.2% | 0 |
| **B1 완벽한 verifier, verify pair만** | **3.015** (3.02 / 3.01 / 3.02) | **99.2%** | 11.1 / 1 / 20 |
| **B2 + 해가 없으면 다음 pair** | **2.997** (2.99 / 3.00 / 3.00) | **99.9%** | 11.3 / 1 / 21 |
| 참고: robust teacher | 2.98 | – | – |

- **완벽한 verifier는 게임 평균을 3.55 → 3.02 shots(−0.54)로 줄이고, 샷 성공률을 80% → 99%로 올린다.** 다음 pair까지 검증하면 3.00(−0.55)으로 robust teacher(2.98)와 같은 수준이다.
- 시뮬레이터가 성공으로 판정한 행동은 실제 게임에서도 99.9% 성공해 **시뮬레이터와 실제가 일치**한다.
- **검증 비용은 작다**: 샷당 평균 11회, 중앙값 1회(대부분의 샷은 made head의 1순위가 이미 성공해 1회로 끝남), p90 20회. 실패하는 18%의 state만 평균 약 67회가 필요하다((12.8 − 0.82) / 0.18).
- **made head의 순위는 1순위 이후로는 거의 쓸모가 없다**: success@k가 k = 2, 3, 5에서 +0.8, +0.7, +1.2%p씩만 오르고, 91.7%에 도달하려면 50개를 확인해야 한다. 상위 후보들이 서로 격자에서 이웃한 사실상 같은 행동일 가능성이 있다(미검증).
- 한계: **추론 때 시뮬레이터를 쓰는 값**이라 정책 단독 성능이 아니다. 깊이 4 500 epoch 모델(seed 3개) 기준이고, 후보 격자는 pair head의 proposal 주변 ±0.3°·7개 speed로 한정된다. 게임 성공률은 포켓 + scratch 없음 기준이다.

### search → distill 1단계: 시뮬레이터 결과로 made head(verifier)를 on-policy 학습 (TODO 19, 2026-10-12 계획)

**왜 (사용자가 계속 제안한 것)**: 시뮬레이터로 후보를 쳐서 positive(성공)와 negative(실패)를 모두 써서 학습한다. 앞선 oracle 결과(정책 3.55 vs 완벽한 verifier 3.02 shots)가 격차 0.54 shots의 거의 전부가 **made head의 선택 오류**임을 보였고, made head의 순위는 1순위 이후 거의 쓸모가 없었다(success@5 84.9%, @20 89.1%, @217 98.8%). 현재 made head는 teacher 계획기가 만든 coarse 후보의 시뮬레이션으로 학습돼 있어, 실제로 순위를 매기는 **pair head proposal 주변 격자(각도 ±0.3° × 절대 speed 7개)** 의 분포와 다르다. 그 분포에서 시뮬레이터 결과로 직접 학습한다. 이것은 "검증을 증류한다"는 뜻이고, 같은 데이터는 이후 pair head(actor) 증류(2단계)에도 쓴다.
**데이터 생성**: 모델(깊이 4, 500 epoch)의 seed별 **train state**(label pair가 있는 것, 약 15k개) 각각에서 label pair의 proposal 주변 후보 217개 중 24개를 뽑아 시뮬레이터로 친다: **12개는 현재 made head의 상위 20개 중에서(어려운 negative 포함), 12개는 나머지에서 균등**. 각 시뮬레이션에서 18 pair 전체의 `made`와 `scratch`를 기록한다(`made2_loss`의 목표 `made & ~scratch`와 같은 정의). seed당 약 36만 샷.
**학습**: encoder, pair head, selection head는 **고정**(proposal 분포를 데이터 생성 때와 같게 유지)하고 `head_made2`만 `made2_loss`(18-way BCE)로 fine-tune. lr 3e-4, 최대 30 epoch, train state의 10%를 holdout으로 두고 holdout BCE가 최소인 epoch을 쓴다.
**평가 (val state, 학습에 안 쓴 state)**: ① success@k: 정책이 verify로 고른 pair의 후보 217개를 새 made head 점수 순으로 시뮬레이터에 쳐서 상위 k개 안의 성공 비율(k = 1, 2, 5, 20, 217). 기존 made head와 같은 proposal·같은 후보에서 비교(k = 1은 verify+refine clean과 같다) ② 게임 500판 shots/game(verify+refine, 새 made head) ③ 보조: 같은 후보 표본에서 BCE·state 내 AUC.
**비교 기준**: 같은 모델의 기존 made head(결과를 같은 스크립트로 다시 잰다). seed 0/1/2.
**"효과 인정"**: seed 3개 모두 개선 + 평균 success@1 ≥ +2%p (게임은 ≥ 0.15 shots).

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| E | 새 made head가 기존보다 나쁨(success@1 −2%p 이하) | on-policy 데이터가 과적합이거나 분포 이동 | 원래 sims와 혼합, 정규화 |
| A | success@1 효과 인정이고 게임도 효과 인정 | 시뮬레이터 결과를 on-policy로 학습하면 verifier가 개선된다 | 데이터 늘리기(후보 수·pair), 2단계(pair head 증류), 반복(expert iteration) |
| B | success@1은 효과 인정인데 게임은 미달 | 순위는 좋아졌지만 게임에서는 pair 선택 등 다른 항목이 병목 | pair 선택 쪽 |
| D | success@1 변화 < 2%p인데 success@5·20은 개선 | 상위권 순위만 개선 | 상위 k개를 시뮬레이터로 검증하는 하이브리드(비용 대비 효과 평가) |
| C | success@1과 success@20 모두 변화 < 2%p | 이 입력 표현의 made head는 on-policy 데이터로도 순위를 못 올린다(표현·해상도 한계) | 기하 입력 허용(열린 결정 1)이나 추론 때 시뮬레이터 검증 유지 |
| G | seed 하나만 반대 | 불확실 | seed 5개 |

**결과: 각도 loss 가중치 (2026-10-12, 깊이 4, 500 epoch, seed 0/1/2, 같은 코드 control)**

설정 검증(F): S01은 `angle_scale_deg`(0.2 → 0.1)만, KEN은 `uncertainty_weights`만 control과 다르다(`results.json` 인자).

| 지표 | control (기준) | **S01** 각도 가중치 4배 | **KEN** Kendall |
|---|---|---|---|
| pair head 각도 오차 중앙값 전체 | 0.097° | 0.096° | 0.114° |
| 〃 seed별 | 0.101 / 0.095 / 0.093 | 0.098 / 0.093 / 0.095 | 0.142 / 0.103 / 0.099 |
| 〃 남은 공 2개 | 0.158° | 0.148° | 0.181° |
| within 0.2° 남은 공 2개 | 57.7% | 59.5% | 53.1% |
| 스캔 argmax가 창 안 | 86.5% | 88.0% | 83.1% |
| clean select 전체 | 74.5% | 73.4% | 70.5% |
| clean verify+refine 전체 / 공 2개 | 81.3% / 74.9% | 79.9% / 71.8% | 79.4% / 73.1% |
| **게임 shots/game** (success) | 3.55 ± 0.10 (84.2%) | 3.62 ± 0.03 (82.5%) | 3.83 ± 0.16 (78.3%) |
| 〃 seed별 | 3.52 / 3.66 / 3.47 | 3.65 / 3.59 / 3.62 | 4.01 / 3.73 / 3.74 |

S01 − control: 각도 오차 seed별 [-0.003 -0.002  0.002] (평균 -0.001°), 게임 shots seed별 [ 0.13 -0.08  0.15] (평균 +0.07)

KEN − control: 각도 오차 seed별 [0.041 0.008 0.006] (평균 +0.018°), 게임 shots seed별 [0.49 0.06 0.27] (평균 +0.28)

**Kendall이 학습한 가중치 (best epoch, log σ² → 가중치 exp(−s), seed 0 / 1 / 2)**: 각도 +1.28 / +0.88 / +0.62 → 0.28 / 0.41 / 0.54, 속도 +1.49 / +1.48 / +1.47 → 0.22 / 0.23 / 0.23, 선택 +2.57 / +2.42 / +2.46 → **0.08 / 0.09 / 0.09**. 세 항의 가중치가 모두 1보다 작아졌고 선택 항이 가장 많이(약 11배) 내려갔다. 각도 대 선택의 **상대** 가중치는 오히려 3.5 ~ 6배 커졌다.

**판단표 적용**
- **F 아님** (설정이 의도한 한 가지만 다름).
- **S01 → C**: 각도 오차 변화가 −0.001°로 기준(0.009°) 미만이고 seed별 방향도 섞여 있다(−0.003, −0.002, +0.002). 각도 가중치를 4배로 올려도 각도 정밀도가 안 변한다 → **각도 가중치는 병목이 아니다.** 게임은 3.62로 기준(3.55) 대비 +0.07(표준편차 안, seed 0·2는 나쁘고 seed 1만 좋음), verify+refine clean은 −1.4%p로 약간 나쁘다. 남은 공 2개 각도 오차는 −0.010°(−6%)로 줄었으나 사전 지표가 아니고 효과 인정 기준에도 못 미친다.
- **KEN → D**: 각도 오차가 기준보다 +0.018°(≥ 0.009°) 나쁘다(seed 3개 모두 나쁨: +0.041, +0.008, +0.006). 게임은 3.83으로 +0.28(3 seed 모두 나쁨), select clean이 −4.0%p, 스캔 argmax가 −3.4%p다. **자동 조절은 기준보다 나쁘다.** 이 문제에서는 필요 없다.
- 사전에 적어 둔 예상("각도 손실이 두꺼운 꼬리라 각도 가중치가 내려가 정밀도에 도움이 안 됨")은 반만 맞았다: 각도 가중치(0.28~0.54)는 내려갔지만 선택 가중치가 더 크게 내려가(0.08) 각도의 상대 가중치는 올랐다. 그래도 결과는 기준보다 나빴다 — 선택 head의 학습 신호가 줄어든 것(select clean 하락)이 원인일 가능성이 크다(미분리).

**결론**: 각도 loss 가중치는 손대지 않는다. 수동 4배는 효과가 없고, Kendall 자동 조절은 오히려 나쁘다. 각도 정밀도는 가중치 문제가 아니라 표현·verifier 쪽이다(판단표 C의 다음 행동). 한계: 가중치를 4배 한 가지만 시험했고(0.05 등은 안 함), seed 3개, 게임 표준편차 0.03~0.16.

**결과: search → distill 1단계, made head를 시뮬레이터 결과로 on-policy fine-tune (2026-10-12, 깊이 4, 500 epoch, seed 0/1/2)**

데이터: seed당 train state 약 14,850개 × 후보 24개 = 약 35.7만 샷(`logs/vdistill/s{seed}.npz`). `head_made2`만 fine-tune(30 epoch, lr 3e-4, holdout BCE 0.047 → 0.044로 약 7% 감소, 최적 epoch 27~29). 평가는 val state(학습에 안 쓴 state)에서 같은 모델의 원래 made head와 같은 후보로 비교. 원래 head의 수치는 이전 평가와 소수점까지 같다(재현). 평가 속도는 첫 성공에서 멈추도록 고친 뒤 약 2분(이전 방식은 후보 217개를 전부 시뮬레이션해 20분 이상).

| 지표 | 원래 made head (seed 0 / 1 / 2, 평균) | distilled (seed 0 / 1 / 2, 평균) | 차이 평균 |
|---|---|---|---|
| success@1 (= verify+refine clean) | 82.4 / 83.9 / 80.2 (82.2%) | 82.7 / 84.1 / 80.3 (82.4%) | **+0.2%p** |
| success@5 | 85.4 / 86.3 / 83.2 (85.0%) | 85.8 / 86.2 / 82.7 (84.9%) | −0.1%p |
| success@20 | 89.5 / 90.4 / 87.6 (89.2%) | 89.2 / 89.4 / 87.3 (88.6%) | −0.5%p |
| success@50 | 92.1 / 92.7 / 90.3 (91.7%) | 91.3 / 92.3 / 89.8 (91.1%) | −0.6%p |
| success@217 | 98.6 / 99.0 / 98.7 | 98.6 / 98.9 / 98.7 | 0 |
| state 내 AUC | 0.783 / 0.807 / 0.805 (0.798) | 0.790 / 0.811 / 0.813 (0.805) | +0.006 |
| 후보 표본 BCE | 0.493 / 0.478 / 0.468 (0.479) | 0.463 / 0.447 / 0.440 (0.450) | −6% |
| **게임 shots/game** | 3.522 / 3.664 / 3.470 (3.55) | 3.586 / 3.774 / 3.520 (3.63) | **+0.075** (3 seed 모두 나쁨) |

**판단표 적용: C** (success@1과 success@20 모두 변화 < 2%p). E는 아니다(악화 −2%p 이하가 아님), A·B·D도 아니다(개선 없음).
- **BCE는 6% 좋아졌는데 순위는 안 좋아졌다.** 새 head는 시뮬레이터 결과를 더 잘 맞히지만(보정이 개선), 같은 state 안에서 후보를 가르는 능력은 거의 그대로다(AUC +0.006). 즉 배운 것은 주로 확률의 크기이고 후보 간 구분이 아니다.
- **게임은 오히려 약간 나빠졌다**(+0.075 shots, 3 seed 모두). 표준편차 안이지만 개선은 없다.
- 해석: **encoder를 고정한 made head는 on-policy 데이터가 있어도 순위를 더 못 올린다.** made head가 읽는 정보(pair embedding `e_ij`, 행동, proposal 대비 차이)에 후보를 가를 세밀한 기하 정보가 없거나 해상도가 부족하다는 가설과 일치한다(앞선 "기하만으로 실패 예측 AUC 0.80", "기하 입력 made head 진단 95%"와 같은 방향).
- 한계: encoder·pair head를 고정했다(풀면 proposal이 바뀌어 데이터 분포가 달라짐). 설정 한 가지(30 epoch, 최적 epoch이 끝 근처라 수렴 전일 수 있으나 BCE만 6% 더 줄 뿐 순위 지표는 거의 안 움직였다), label pair만, 후보 24개/state, seed 3개.
- 다음(판단표 C): 기하 입력 허용(열린 결정 1) 또는 추론 때 시뮬레이터 검증 유지. 이번 데이터(`state`, 후보, made, scratch)를 그대로 써서 **기하 특징을 입력에 더한 verifier**를 시뮬레이션 없이 오프라인으로 시험할 수 있다.

### encoder가 기하 정보를 담고 있는가: probe (2026-10-12, 학습 없음, 임시 스크립트 — 저장소에 넣지 않음)

**질문 (사용자)**: 기하값을 입력으로 주는 것은 정답을 알려 주는 것이다. 대신 기하값을 보조 목표로 예측하게(look-ahead) 학습하면 어떤가. 이 방법이 의미가 있으려면 현재 encoder가 그 기하를 **이미 담고 있지 않아야** 한다.
**방법**: 깊이 4(500 epoch) 모델 3 seed의 **고정된** pair embedding `e`(label pair)로 기하 값을 예측하는 probe(선형 ridge, 2층 MLP)를 train state(약 14.9k)로 학습하고 held-out val state(약 1.66k)에서 R²를 쟀다. 비교용으로 **원시 좌표 21개 + pair one-hot**만으로 같은 MLP probe를 학습했다. 목표값은 원시 좌표에서 계산한 기하(실제 테이블 크기 환산): 컷 각도, cue→ball 거리, ball→pocket 거리, 포켓 입구 편차(공이 포켓 축에서 벗어난 각), 공의 쿠션 거리, cue→ball·ball→pocket 경로에서 다른 공까지의 최단거리, side 포켓 여부.

| 기하 값 | `e` + 선형 | `e` + MLP | 원시 좌표 + MLP |
|---|---|---|---|
| 컷 각도 | 0.767 | **0.918** | 0.630 |
| cue→ball 거리 | 0.922 | **0.976** | 0.965 |
| ball→pocket 거리 | 0.883 | **0.965** | 0.949 |
| side 포켓 여부 | 0.835 | 0.938 | 0.996 |
| 포켓 입구 편차 | 0.289 | **0.392** | 0.665 |
| 공의 쿠션 거리 | 0.422 | **0.572** | 0.795 |
| 경로상 다른 공 (cue→ball) | 0.403 | **0.484** | **0.978** |
| 경로상 다른 공 (ball→pocket) | 0.349 | **0.431** | **0.977** |

**해석**
- **encoder는 컷 각도와 거리는 이미 잘 담고 있다**(R² 0.92~0.98, 컷 각도는 원시 좌표 probe(0.63)보다 오히려 낫다). 이 값들에 보조 목표를 줘도 얻을 것이 적다.
- **그러나 포켓 입구 편차(0.39), 쿠션 거리(0.57), 경로상 다른 공(0.43~0.48)은 거의 버린다.** 원시 좌표에서는 같은 probe가 0.80~0.98로 읽으므로 **정보는 입력에 있는데 `e`가 안 남긴 것**이다. pair head는 label 각도를 맞히는 데 이 값들이 필요 없어서 학습 중 버려진 것으로 보인다(가설).
- 실패 프로파일에서 실패율이 높았던 클러스터(공이 쿠션 가까이, 경로 근처에 다른 공)의 특징과 겹친다. 고정 encoder를 쓴 1단계 made head 증류가 순위를 못 올린 것도 이 정보가 `e`에 없어서였을 가능성과 일치한다(미검증).
- 따라서 사용자의 제안(기하를 look-ahead 보조 목표로 학습)은 **컷·거리가 아니라 포켓 입구 편차·쿠션 거리·경로상 다른 공**에 대해서 여지가 있다. 한계: probe의 R²는 정보가 선형/MLP로 읽히는지를 보일 뿐 그 정보가 실패의 원인인지는 보이지 않는다.

### made head의 한계는 정보인가 학습인가: 학습 곡선과 더 큰 모델 (2026-10-12, 학습 없음, 임시 스크립트 — 저장소에 넣지 않음)

**질문 (사용자)**: encoder가 기하를 가지고 있다면 made head에는 충분한 정보가 있는데 학습이 제대로 안 되는 것 아닌가.
**방법**: 1단계 증류 데이터(seed 0/1, 후보 약 32만 개, 24개/state)에서 held-out state 10%(약 1,485개)를 두고, made head가 읽는 입력 `[e_ij(128), 행동(2), proposal 대비 차이(2)]`로 label pair의 성공(`made & ~scratch`)을 예측했을 때의 **state 내 AUC**(같은 state의 후보끼리의 순위 품질)를 측정했다.
**(a) 학습 곡선** (원래 made head에서 시작해 train state의 일부로 30 epoch fine-tune):

| 사용한 train state | 2% | 5% | 10% | 25% | 50% | 100% | (원래 head) |
|---|---|---|---|---|---|---|---|
| seed 0 AUC | 0.802 | 0.806 | 0.807 | 0.812 | 0.813 | 0.813 | 0.802 |
| seed 0 BCE | 0.439 | 0.434 | 0.432 | 0.428 | 0.424 | 0.419 | 0.456 |
| seed 1 AUC | 0.810 | 0.805 | 0.812 | 0.812 | 0.816 | 0.816 | 0.810 |
| seed 1 BCE | 0.427 | 0.427 | 0.423 | 0.415 | 0.413 | 0.410 | 0.445 |

**(b) 같은 입력에 더 큰 모델** (seed 0, 같은 train/held-out):

| 모델 | state 내 AUC | 전체 풀링 AUC |
|---|---|---|
| 원래 made head | 0.802 | 0.768 |
| 1단계 fine-tune (100% 데이터) | 0.813 | – |
| MLP 3 × 512 (30초, 최적 epoch 기준) | 0.824 (마지막 epoch 0.813) | 0.801 |
| gradient boosting (10초) | **0.828** | **0.818** |

**해석**
- **학습 곡선은 25%에서 이미 평평하다**: 데이터를 4배(25% → 100%) 늘려도 AUC가 +0.001(seed 0), +0.004(seed 1)뿐이다. BCE만 천천히 계속 줄어(보정이 개선됨) 순위 품질과는 다른 양상이다. 순위가 **데이터 부족 때문에 막힌 것이 아니다.**
- **같은 입력으로 훨씬 강한 학습기(gradient boosting, 큰 MLP)를 써도 +0.02 정도**(0.802 → 0.824~0.828)다. 완벽한 순위(1.0)까지는 0.17이 남아 있다. 학습 용량·최적화로 풀 수 있는 몫은 작다.
- 따라서 made head의 한계는 **학습이 아니라 입력 `[e, 행동, 차이]`에 담긴 정보**다. 앞선 probe(고정 encoder가 쿠션 거리·경로상 다른 공·포켓 입구 편차를 거의 버림)와 같은 방향이다.
- **"encoder가 기하를 가지고 있다"는 거친 수준의 기하(컷 각도·거리, R² 0.92~0.98)에 대한 말이다.** 후보 순위를 가르려면 성공 창의 위치를 0.05° 단위로 알아야 하는데, 이 정밀도는 pair head 자신의 각도 오차(중앙값 0.1°)와 같은 한계에 걸려 있다. R² 0.92는 컷 각도 오차가 수 도에 해당해 이 정밀도에 한참 못 미친다.
- 한계: seed 0/1, 한 가지 후보 분포(상위 12 + 균등 12), label pair만. AUC는 후보 간 순위 품질이고 success@k와 같지 않다.

### 기하 look-ahead 보조 목표: encoder가 버린 기하 정보를 남기게 하기 (2026-10-12 계획)

**왜 (사용자 제안)**: 기하값을 입력으로 주면 정답을 알려 주는 것이다. 대신 기하값을 **보조 목표로 예측**하게 학습하면(look-ahead) encoder가 그 정보를 담게 된다. probe에서 고정 encoder가 쿠션 거리(R² 0.57), 경로상 다른 공(0.43~0.48), 포켓 입구 편차(0.39)를 거의 버렸고(원시 좌표로는 0.80~0.98 읽힘), made head의 순위는 데이터·용량이 아니라 입력 정보에 막혀 있었다(학습 곡선이 25%에서 평평, 더 큰 모델도 +0.02). 보조 목표는 pair head가 label 각도를 맞히는 데 필요 없어서 버려진 정보를 남기도록 강제한다.
**변경 (보조 손실 하나, 입력·추론은 그대로)**: `--geom-aux-weight 5`. pair embedding `e_ij` 위의 보조 head(MLP 128 → 64 → 4)가 모든 live pair(18개)에 대해 4개 기하 값을 예측한다: **포켓 입구 편차**(공이 포켓 축에서 벗어난 각, °), **공의 쿠션 거리**(m), **cue→ball 경로**와 **ball→pocket 경로**에서 다른 공까지의 최단거리(m, 0.6 m로 cap). 목표는 원시 좌표에서 학습 때만 계산하고 train 통계로 표준화, 손실은 live pair에 대한 제곱 오차 평균 × 5(주 손실이 약 16~25라 초기에 약 20%). 컷 각도·거리는 encoder가 이미 담고 있어(R² 0.92~0.98) 뺐다. 보조 head는 추론에 쓰이지 않는다. 나머지는 control과 동일(깊이 4, 500 epoch, `--pair-supervision --no-wta-fallback --made2-head`), seed 0/1/2, 출력 `logs/experiments_geom`.
**control**: 같은 코드 깊이 4(`logs/experiments_speed/*_L4_m2w1_*`, 이전 깊이 4와 동일함을 확인한 것).
**지표**: ① **메커니즘**: 학습된 encoder로 같은 probe(`e` + MLP)를 다시 해서 4개 값의 R² (정보가 실제로 남았는가) ② pair head 각도 오차(전체, 남은 공 2개) ③ **made head 순위**: success@1·5·20(`verifier_distill eval --only original`, 이 학습에서 함께 학습된 made head). state 내 AUC는 후보 표본이 모델마다 달라 참고용 ④ 게임 shots/game.
**"효과 인정"**: control과 seed별로 짝지어 3개 모두 개선 + 평균 success@1 ≥ +2%p (게임 ≥ 0.15 shots, 각도 ≥ 0.009°).

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| F | 인자가 `geom_aux_weight` 외에 control과 다름 | 설정 오류 | 비교 보류 |
| M | probe R²가 4개 중 하나라도 0.8 미만 | 보조 head가 정보를 안 남김(가중치·학습 부족) | 가중치·head 용량을 키워 재시도. 유용성에 대해서는 아직 모름 |
| E | M 아님, 각도 오차가 control보다 ≥ 0.009° 나쁘거나 게임이 ≥ 0.15 shots 나쁨 | 보조 목표가 주 과제를 해침 | 가중치를 1로 낮춰 재시도 |
| A | M 아님, success@1이 효과 인정이고 게임도 개선(≥ 0.15 shots) | 기하 look-ahead가 verifier를 개선한다 | 보조 목표 확대(컷·창 폭 등), 가중치 탐색 |
| B | M 아님, success@1이 효과 인정인데 게임은 개선 미달 | 순위는 좋아졌지만 다른 항목이 병목 | pair 선택 쪽 |
| C | M 아님, success@1 변화 < 2%p | 버려졌던 기하가 순위의 병목이 아니다(0.05° 정밀도가 한계) | verifier 전용 encoder 또는 추론 때 시뮬레이터 검증 |
| G | 위 중 seed 하나만 반대 | 불확실 | seed 5개 |
