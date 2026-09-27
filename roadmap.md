# Roadmap

Experiment plans and next directions. For completed experiment results, see [experiments.md](experiments.md).

---

## Progress

```
[x] Exp-13   Phase 0: proximity reward failed + steps scaling confirmed (1M:50% / 2M:56% / 5M:65.8%)
[x] Exp-14   gradient_steps=4, 5M → 68.6% (+2.8pp, 2x time) — flat policy ceiling confirmed
[x] Exp-15   Trajectory VAE exploration — physics latent structure identified, decoder architecture finalized

[~] Exp-16   World Model Critic — Q(s,a) = q(M(s,a))
             └─ [x] SSM v10~v18: full-BPTT 버그 수정, collision aux, ar_state 제거 → 31.6cm
             └─ [x] SSM v18_longrun: SPRDataset 43k + 2000ep → 24.8cm
             └─ [x] GNN 2-ball: set-based, message passing → 32.4cm
             └─ [x] SPR-MDN z-space (v17~v26): NLL gradient 불균형 3363x 확인 → L2(28.3cm) 돌파 불가
             └─ [x] SPR-MDN v26_p1: SPRDataset scratch 1190ep → 25.1cm (현재 최고)
             └─ [x] SMDN per-step (v28): MDN collapse 확인 → per-step MDN 근본 부적합
             └─ [x] v28 DT 공정 비교: DT=0.05→16.3cm / DT=0.01→15.4cm (공정 metric, 전수 ep 정규화)
             └─ [x] v33_segment: segment 단위 4.3cm — chaining covariate shift 미해결
             └─ [ ] v33 chaining 성능 검증
             └─ [ ] v34 event-boundary MDN
             └─ [ ] WM → RL 통합
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

**v18 pure latent final results** (epoch 400, err=31.6cm):
- ar_state feedback fully removed → z encodes all physics information on its own
- transition random init (v17 encoder/decoder retained)
- err=31.6cm, collision recall=0.549, episode pocket recall=0.312

**v18 scratch (random init) final results** (epoch 400, err=32.7cm):
- Same hyperparams as v18, no pretrained weights
- err=32.7cm, collision recall=0.512
- Pretraining benefit: ~1cm err / ~0.037 recall — marginal; GNN can train from scratch

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

**Direction**: GNN-based multi-agent architecture (n_balls dynamic)

#### Next Architecture: GNN World Model

Key design decisions derived from v16–v18 experiments and architectural analysis:

**1. Set-based / multi-agent (replaces fixed-dim concatenation)**

Each ball is a node with shared-weight encoder/decoder. n_balls becomes a runtime parameter.

```
Current SSM:  state = [cue(7) | tgt(7)] = 14-dim  (hardcoded 2-ball)
GNN:          state = N × ball(7),  N = any number of balls

node:  z_i = BallEncoder(ball_i_state, pocket_dists, is_cue)  [shared weights]
edge:  m_ji = BallBallMsg(z_i, z_j, Δpos, Δvel, dist)         [shared weights]
       m_pi = BallPocketMsg(z_i, Δpos_to_pocket, dist)         [shared weights]
update: z_i' = LN(z_i + MLP(cat(z_i, Σ_j m_ji, Σ_p m_pi)))
```

Message passing runs at every rollout step — necessary for multi-collision chains (ball1 → ball2 → ball3).

**Status**: `world_model/gnn/gnn_model.py` implemented and shape-verified (2-ball).
- Same weights handle N=2 and N=3 (tested).
- Parameters: 373,708 (vs v18: 253,331).

**2. No GRU / history tracking**
- Evidence: v17 (ar_state RNN) vs v18 (no ar_state) — minimal performance difference
- Billiards is Markovian: current state [pos + vel + spin] fully determines future
- Message passing handles inter-ball dependencies; no temporal memory needed

**3. Multi-step latent self-prediction with MDN (SPR-MDN)**

**One-line idea**: train the model to predict the next latent using its *own previous prediction* as input — the same condition it faces during planning, where GT states are never available beyond step 0.

**The training-deployment gap**: at planning time, only $s_t$ (step 0) is real; steps 1, 2, 3, ... use the model's own sampled outputs as the next input. If training always feeds GT at every step (teacher forcing), the model never practices "self-chained" rollout and fails when compounding errors enter at deployment.

**Why this is BYOL-family, not just NLL**: the defining mechanism of BYOL is EMA target encoder + stop-gradient for stable self-prediction without collapse. Both are present here:
- $\phi' \leftarrow \tau\phi' + (1-\tau)\phi$ — EMA target
- $\bar{z}_{h+1} = \mathrm{sg}(\mathrm{Enc}_{\phi'}(s_{t+h+1}))$ — stop-gradient

This is exactly what SPR (Self-Predictive Representations, Schwarzer et al. 2021) does: BYOL + multi-step latent transition model. Our variant replaces SPR's deterministic predictor with an MDN, adding mixture density to capture bifurcations in chaotic billiards. The cosine similarity loss in vanilla BYOL becomes NLL here — stricter because it requires the correct distribution shape, not just direction.

**State / action indexing**:
- $s_t$ = pre-strike state (all balls at rest)
- $a_t$ = strike parameters (angle, speed, spin)
- $s_{t+1}, \ldots, s_{t+H}$ = post-strike states at fixed Δt intervals (autonomous physics)

**Design (MDN transition, no KL, no posterior)**:

Notation: $z_t = \mathrm{Enc}_\phi(s_t)$, $\phi'$ = EMA copy of $\phi$ (no gradient), $H$ = rollout length.

Action enters only at $h=0$ (the impulse moment); all subsequent steps are autonomous:
$$\tilde{a}_h = \begin{cases} a_t & h=0 \\ \mathbf{0} & h=1,\dots,H-1 \end{cases}$$

```
# Starting point: only step 0 uses GT
ẑ_0 = Enc_φ(s_t)

# Per step h = 0 … H-1:

# MDN transition — conditioned on action at h=0, zero otherwise
(π, μ, σ) = MixtureHead_θ(ẑ_h, ã_h)   # π:(B,N,K)  μ:(B,N,K,7)  σ:(B,N,K,7)  K=5
σ_k = softplus(σ_raw_k) + ε

# Scoring target: EMA-encoded GT next state (stable anchor)
z̄_{h+1} = sg( Enc_φ'(s_{t+h+1}) )

# NLL loss at step h: how likely is the actual future under my predicted mixture?
L_NLL^h = -log Σ_k π_k · N(z̄_{h+1} ; μ_k, diag(σ_k²))

# Next input: sample from own prediction, NOT from GT (the key)
k ~ Cat(π),  ẑ_{h+1} = sg( μ_k + σ_k ⊙ ε ),  ε ~ N(0, I)
# sg: treat sampled value as a constant; gradient flows into MixtureHead only via L_NLL

L_NLL = Σ_{h=0}^{H-1} L_NLL^h
```

**Where mixture density actually matters**: $h=0$ (the strike) is a deterministic physical impulse — given $a_t$, the outcome is fully determined. The chaotic bifurcations arise at $h \geq 1$, when balls collide with each other or cushions. A sub-mm difference in contact point at a grazing collision sends the ball to completely different regions. The $K$-component mixture is doing real work precisely on the $\tilde{a}_h = \mathbf{0}$ steps — the autonomous phase, not the action phase.

**Reconstruction** (encoder/decoder grounding — closes the NLL conspiracy failure mode):
```
L_recon = Σ_{h=0}^{H} || Dec_ψ(ẑ_h) - s_{t+h} ||²
```
Without this, encoder and transition could jointly find a degenerate z-space that minimizes NLL while losing physical meaning. $L_{\text{recon}}$ keeps z grounded to real ball states at every rollout step.

**Total loss**:
```
L_total = L_NLL + λ · L_recon + L_type
```

**Lineage**:
```
BYOL (EMA + stop-grad self-prediction)
  └─ SPR (BYOL + multi-step latent transition)
       └─ SPR-MDN [ours] (SPR + mixture density for chaotic bifurcations)
```
The specific combination (EMA + stop-grad + MDN + multi-step chaining) has not been published as far as we know — each component is validated independently.

**4. Label smoothing**
```python
F.cross_entropy(logits, targets, label_smoothing=0.1)
```
Implemented in `ssm_model.py` and `gnn_model.py` via `--label-smoothing` arg.

**5. Label smoothing**
```python
F.cross_entropy(logits, targets, label_smoothing=0.1)
```
Implemented in `ssm_model.py` and `gnn_model.py` via `--label-smoothing` arg.

#### Blocker 2: pocket recall improvement (resolved — accept current level)

v17 (focal+w=20): episode recall 0.250, prec=0.320. Target (≥0.5) not met.
v18 (no ar_state, epoch 180): episode recall 0.234. No meaningful improvement over v17.

**Decision: move forward with current recall (~0.25) rather than continuing to chase the target.**

Rationale:
- The fundamental bottleneck is chaos-induced error accumulation in step-by-step prediction, not the ar_state or training objective
- `pocket_prob = max(type_logit[..., 4])` still provides a useful (if noisy) signal for Q-target augmentation
- Architectural improvements (MDN mixture transition, SPR-MDN self-prediction) are better addressed in the 3-ball rewrite than incrementally

**Key finding from v17 vs v18 comparison**:
- ar_state removal had minimal impact on performance (err: 30.7 → 32.6cm; recall: 0.549 → 0.517)
- The v16→v17 improvement came from focal loss + class weights, not from ar_state
- History tracking (GRU/ar_state) is not necessary for this Markovian physics problem

---

### 현재 상태 (2026-09-26)

**최고 성능**: v28_dt01_3s 15.4cm (SMDN K=5, DT=0.01, 공정 metric — 전수 에피소드 길이 정규화)  
**모든 training 프로세스 종료됨**

### 확정된 결론 요약

| 결론 | 내용 |
|------|------|
| full-BPTT 필수 | stop-grad on predictor chain이 encoder gradient 차단 (v9~v16 전부 무효) |
| SPR bootstrap 기여 7.1cm | lam=0.01 vs lam=0 ablation 확인 |
| NLL z-space 한계 확정 | gradient 불균형 3363x, GradNorm/Kendall 모두 실패 |
| direct state 열위 | z-space 대비 8~18cm 나쁨 |
| per-step MDN collapse | 이벤트 경계가 아닌 per-step은 결정론적 → MDN 분기 인센티브 없음 |
| DT=0.01 > DT=0.05 | 15.4cm vs 16.3cm — 물리 해상도 우위 확인 |
| v33 segment 4.3cm | 단일 세그먼트만, chaining covariate shift 미해결 |

### Plan (in priority order)

| Step | Content | Status |
|------|---------|--------|
| **① v33 chaining 검증** | 연속 세그먼트 rollout 성능 측정 (4.3cm → rollout 몇 cm?) | [ ] Next |
| **② v34 event-boundary MDN** | 충돌 순간에만 K=5 MDN, 구간 내는 결정론적 → collapse 자연 해결 | [ ] Planned |
| **③ WM → RL 통합** | v28_dt01_3s best.pt(15.4cm) 기반 SAC critic 보강 | [ ] Pending |
| **④ 3-ball data + GNN extension** | Generate 3-ball data; extend GNN to N=3 | [ ] Pending |

### ③ Q-target Augmentation (WM-augmented critic)

```python
s_hat, type_logit = wm(s_1, n_steps=60)         # s_1: actual state after env step
pocket_prob = type_logit[0, :, 4].max().item()   # max pocket probability within 60 steps
q_target = r + gamma * V(s') + lambda * pocket_prob
```

Difference from Dyna: WM provides Q-labels directly rather than generating (s,a,r,s') → **WM-augmented critic**

### R-SSM QHead: 배선 완료, 학습 신호 없음

Q-value 추출 목표 대비 현재 위치 점검 (2026-09-27):

- `rssm_model.py:220 aggregate_q(h)` — attention pool(`q_proj`) → `q_head` → scalar Q. `project_rssm_architecture.md` 설계(attention pool, N-independent) 그대로 구현되어 있음
- `rssm_rollout.py:253 Q = self.model.aggregate_q(h)` → `RolloutResult.Q`로 rollout 끝까지 연결됨 — forward pass 자체는 이미 end-to-end
- **`train_rssm.py`에 `q_head`/`aggregate_q` 참조 전혀 없음** — Q label도 loss도 없어서, 진행 중인 rssm_v4 학습에서도 QHead는 계산만 되고 학습되지 않는 죽은 출력 상태

**[ ] Pending** — Q label 소스 결정 필요, 후보:
1. MC return (에피소드 종료 후 실제 reward로 역산)
2. SAC critic bootstrap
3. 위 ③ Q-target Augmentation(`pocket_prob` heuristic)과 병행할지, QHead로 대체할지

결정 후 `train_rssm.py`에 Q loss 추가.

### R-SSM 물리 엔진: pure_physics.py 대체 검토 (보류)

마찰계수 버그 수정(`ball_motion.py`, [experiments.md](experiments.md) 참고) 검증 과정에서 pooltool-free 재구현체 `pure_physics.py`가 free-motion evolution 용도로 `ph.evolve_ball_motion`의 drop-in 대체가 가능함을 확인 (300샷 0.0000cm 일치).

- 안전 교체 후보: `rssm_rollout.py::advance_balls`, `train_rssm.py::compute_shot_ss_loss._advance()`, `viz_rssm.py::_evolve()`
- 제외(미검증): `event_detector.py`의 충돌시각 solver (`get_next_event()`)
- **[ ] Pending** — 교체 이득(속도) 미검증, pooltool 의존성 제거가 시급하지 않아 보류. 필요 시 재검토.

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
