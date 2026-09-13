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

#### Key Findings from v16–v18 Series

1. **ar_state irrelevant**: removal gave +1.9cm error — focal loss + class weights drove v16→v17 improvement, not history tracking. Markov property holds with full [pos+vel+spin] state.
2. **Pretraining marginal**: scratch vs. pretrained ≈ 1cm difference. Architecture is the bottleneck, not initialization.
3. **Pocket recall plateau (~0.31)**: root cause is step-by-step chaos accumulation, not loss function or architecture of the 2-ball SSM. Accepted as-is; MDN mixture transition and BYOL addressed in GNN rewrite.
4. **No GRU needed**: validated by v17→v18 experiment.

#### Architectural Lessons for GNN Rewrite

1. **No GRU**: Markov property confirmed; message passing handles inter-ball dependencies
2. **MDN mixture transition** (not single Gaussian): single Gaussian mode-averages at bifurcation points → physically impossible mean predictions. Mixture + NLL loss, no KL.
3. **BYOL (Transition-chained)**: chain actual `BallTransition` k times as online path; EMA encoder on GT states as target. Gradient flows into Transition directly.
4. **Label smoothing**: apply from the start (`--label-smoothing 0.1`)

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
