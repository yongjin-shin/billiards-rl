# Roadmap

Experiment plans and next directions. For completed experiment results, see [experiments.md](experiments.md).

---

## Progress

```
[x] Exp-13   Phase 0: proximity reward failed + steps scaling confirmed (1M:50% / 2M:56% / 5M:65.8%)
[x] Exp-14   gradient_steps=4, 5M → 68.6% (+2.8pp, 2x time) — flat policy ceiling confirmed
[x] Exp-15   Trajectory VAE exploration — physics latent structure identified, M decoder architecture finalized
             (VAE itself discarded in final architecture; purpose was decoder architecture and hyperparameter search)

[~] Exp-16   World Model Critic — Q(s,a) = q(M(s,a))
             └─ [x] SSM v16 no-curriculum: err=30.5cm, recall=0.549 (epoch 530)
             └─ [x] SSM v17 focal fine-tune: pocket recall 0.000→0.250 (episode-level)
             └─ [~] SSM v18 pure latent (no ar-state): in progress (epoch 44/400, err=38.7cm)
             └─ [ ] WM-augmented Q-target integration
[ ] Exp-17   Phase 1 HRL — System 2 (ball selection discrete 3) + System 1 (Phase 1 Exp-10 freeze)

[ ] cushion / bank shots
[ ] self-play / full 8-ball
```

---

## Completed: Deterministic SSM (feature/wm-ssm)

Adopted **z-space closed-loop rollout** to address rollout error accumulation in the fixed-Δt AR architecture.

```
s_0 →enc→ z_0 →f→ z_1 →f→ z_2 → ... →f→ z_T   (z space only)
               ↓g     ↓g              ↓g
               ŝ_1    ŝ_2             ŝ_T
```

**v16 no-curriculum final results** (epoch 450, err=30.5cm):
- state err: 30.5cm (1.0s: 29.5cm / 2.0s: 36.1cm / 3.0s: 41.3cm)
- collision recall/prec: 0.549 / 0.540, type_acc: 0.905
- pocket recall: 0.000

**v17 focal fine-tune final results** (epoch 200, err=30.7cm):
- pocket class_weight=20 + focal_gamma=2.0
- pocket step recall: 0.000 → 0.109
- pocket episode recall: 0.000 → 0.250 (TP=16, FP=34, FN=48)
- Roadmap target (recall≥0.5, prec≥0.4) not met — step-by-step error accumulation is the fundamental bottleneck

**v18 pure latent (in progress, epoch 252/400)**:
- ar_state feedback fully removed → z encodes all physics information on its own
- transition random init (v17 encoder/decoder retained)
- err=32.6cm, collision recall=0.517 (epoch 252)
- episode-level pocket recall=0.234 (evaluated at epoch 180)

---

## Next: WM → RL Integration

### Design Rationale

The fundamental problem with SAC critic Q(s,a): with only binary sparse reward (+1 pocket), it is difficult to learn "whether this action will pocket the ball."
Since WM predicts ball trajectories + collision types over 60-step rollouts, this can be leveraged as Q-target supervision.

### Prerequisites (blockers)

Two fundamental issues to resolve before attaching WM to RL:

#### Blocker 1: WM only supports 2-ball (current state=14-dim)

```
Current WM:  state = [cue(7) | tgt(7)]         = 14-dim  (2-ball)
RL problem:  state = [cue(7) | ball1(7) | ball2(7)] = 21-dim  (3-ball, n_balls=3)
```

- Current WM cannot model ball-ball interactions (target1 → target2 deflection)
- Must accept full ball layout as input to use as Q-target
- **Architecture change + data regeneration required**

**Direction**: Extend to fixed n_balls=3 with Stochastic SSM architecture

```
state   : 21-dim [cue(7), ball1(7), ball2(7)]
```

#### Next Architecture: Stochastic SSM (no GRU)

Key design decisions derived from v16–v18 experiments:

**1. No GRU / history tracking**
- Evidence: v17 (with ar_state RNN feedback) vs v18 (no feedback) show minimal performance difference
- Billiards is Markovian: current state [pos + vel + spin] fully determines next state
- GRU adds complexity without meaningful benefit for this problem

**2. Stochastic z transition (replacing deterministic f)**
```
# Current (v16–v18): deterministic
z_{t+1} = LayerNorm(z_t + MLP(z_t))

# Next: stochastic
z_{t+1} ~ p(z | z_t) = N(μ_θ(z_t), σ_θ(z_t))
```
Motivation: billiards is deterministic at the physics level, but it is a chaotic system (positive Lyapunov exponent). Small encoder representation errors grow exponentially through collisions. From the model's perspective, this creates genuine **epistemic uncertainty** — not because the physics is random, but because our representation is imperfect. Stochastic z captures this uncertainty without requiring GRU history.

This is a simplified RSSM: keep the stochastic latent, drop the deterministic recurrent path (h_t).

**3. BYOL auxiliary loss**
The current reconstruction loss (MSE on positions + CE on collision type) teaches z to represent *where the ball is*, but not *where it is going*. BYOL-style temporal prediction forces z to encode dynamics:

```
online:  f_θ(s_t)  → q_θ  →  ẑ_{t+k}
target:  f_ξ(s_{t+k})  →  z̄_{t+k}   (EMA of f_θ, stop-grad)

L_byol = -cosine_sim(ẑ_{t+k}, z̄_{t+k})
```

A state heading toward a pocket must produce a different z_t than one that is not — the BYOL objective enforces this by requiring z_t to predict the future representation.

**4. Label smoothing**
Added to type classification loss to prevent overconfident predictions on the minority pocket class:
```python
F.cross_entropy(logits, targets, label_smoothing=0.1)
```
Already implemented in `ssm_model.py`; apply via `--label-smoothing 0.1` in training.

#### Blocker 2: pocket recall improvement (resolved — accept current level)

v17 (focal+w=20): episode recall 0.250, prec=0.320. Target (≥0.5) not met.
v18 (no ar_state, epoch 180): episode recall 0.234. No meaningful improvement over v17.

**Decision: move forward with current recall (~0.25) rather than continuing to chase the target.**

Rationale:
- The fundamental bottleneck is chaos-induced error accumulation in step-by-step prediction, not the ar_state or training objective
- `pocket_prob = max(type_logit[..., 4])` still provides a useful (if noisy) signal for Q-target augmentation
- Architectural improvements (stochastic z, BYOL auxiliary loss) are better addressed in the 3-ball rewrite than incrementally

**Key finding from v17 vs v18 comparison**:
- ar_state removal had minimal impact on performance (err: 30.7 → 32.6cm; recall: 0.549 → 0.517)
- The v16→v17 improvement came from focal loss + class weights, not from ar_state
- History tracking (GRU/ar_state) is not necessary for this Markovian physics problem

---

### Plan (in priority order)

| Step | Content | Status |
|------|---------|--------|
| **① pocket prediction fix** | v17 focal+weight=20 → recall 0.250; v18 no-ar → 0.234 | [x] Done (accepted as-is) |
| **① v18 pure latent** | Remove ar_state → self-contained z representation | [~] In progress (ep.252/400) |
| **② WM multi-ball extension** | 3-ball Stochastic SSM + BYOL + label smoothing | [ ] Pending |
| **③ Q-target augmentation** | WM(s_1, T=60) → pocket probability → Q-target label | [ ] Pending |
| ④ Auxiliary loss | critic loss + WM pocket prediction parallel training | [ ] Pending |
| ⑤ Reward shaping | WM dense reward → SAC | [ ] Pending |

### ③ Q-target Augmentation (WM-augmented critic)

```python
s_hat, type_logit = wm(s_1, n_steps=60)         # s_1: actual state after env step
pocket_prob = type_logit[0, :, 4].max().item()   # max pocket probability within 60 steps
q_target = r + gamma * V(s') + lambda * pocket_prob
```

Difference from Dyna: WM provides Q-labels directly rather than generating (s,a,r,s') → **WM-augmented critic**

---

## Next: Exp-17 · Phase 0 HRL

### Design Rationale

Problem with Phase 0 flat policy: since the delta_angle reference is the nearest ball direction, **which pocket to aim for is only determined implicitly.**
A nearest pocket bias can emerge during training via curriculum and similar processes.

```
System 2 (newly trained): pocket selection (discrete 6)
                       ↓
obs rearrangement: target_pocket_xy → injected at front of obs
                       ↓
System 1 (pocket-conditioned, after freeze):
  learns "the angle needed to sink the ball into this pocket"
```

**Credit assignment solution:**
- Train System 1 sufficiently first, then freeze
- During subsequent System 2 training, failure is attributed to System 2's pocket selection
- System 2 converges statistically (over thousands of episodes)

**Implementation:** DQN (System 2, discrete 6) + SAC (System 1, continuous) + custom env wrapper

### Obs Design

**Current Phase 0 obs (16-dim):**
```
[cue_x, cue_y,        # 2
 ball_x, ball_y,      # 2
 p0_x, p0_y,          # 12 (6 pockets, fixed order)
 p1_x, p1_y,
 ...
 p5_x, p5_y]
```

**System 1 obs (16-dim, same size):**
```
[cue_x, cue_y,              # 2
 ball_x, ball_y,            # 2
 target_x, target_y,        # 2  ← always obs[4:6] = selected pocket
 other_p0_x, other_p0_y,    # 10 (remaining 5 pockets)
 ...
 other_p4_x, other_p4_y]
```

When System 2 selects a pocket idx, that pocket is placed at obs[4:6] and the remaining 5 are filled in after.
→ obs size preserved (16-dim), System 1 always trains with "obs[4:6] is the target pocket."

**System 1 training reward change:**
```
Current: +1 (when ball enters any pocket)
Changed: +1 (only when ball enters the target pocket)
```

**Feasible pocket selection (resolving random target pocket issue):**
```
cut_angle(pocket_i) = arccos(dot(normalize(B-C), normalize(P_i - B)))
feasible = [i for i in range(6) if cut_angle(i) < 60°]
target_pocket = random.choice(feasible)  # fallback: argmin(cut_angle) if empty
```

---

## Next: Exp-17 · Phase 1 HRL

### Design Rationale

The bottleneck of the current flat MLP is mixing two levels of problems:

| | System 1 (aiming) | System 2 (strategy) |
|---|---|---|
| **Role** | Does shooting at this angle+speed sink that ball? | Which ball first, into which pocket |
| **Reward** | Immediate (+1 per pocket) | Sparse and delayed (9% clear) |
| **Current problem** | Hardcoded to nearest-ball greedy | Almost no gradient signal |

```
System 2 (newly trained):  target ball selection (discrete 3)
                       ↓
obs rearrangement:  [cue_xy, target_xyz, other1_xyz, other2_xyz, 6pockets]
             target ball → moved to ball[0] position
                       ↓
System 1 (Phase 1 Exp-10 freeze):
  delta_angle = 0 → aims toward ball[0] direction (= target)
```

### Exp-17 variants

| | 17a | 17b | 17c |
|---|-----|-----|-----|
| **System 2 action** | ball selection (discrete 3) | ball+pocket (discrete 18) | ball+pocket (discrete 18) |
| **System 1** | Phase 1 freeze | Phase 1 freeze | joint training |
| **OOD risk** | low | medium | none |
| **Key question** | Is the HRL structure itself valid? | Does pocket info help? | Possible without pretraining? |

**Experiment order:** 17a → 17b → 17c
