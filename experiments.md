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
