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
             └─ [x] R-SSM 피벗 (2026-09-28): 이벤트 경계만 확률적/구간 내부 결정론적 물리 — SMDN 계열(v33 chaining, v34) 폐기
             └─ [x] R-SSM: pure_physics 물리 엔진 교체 + 배치 forward (1.7~3.7배)
             └─ [x] R-SSM v5 재학습(val_rmse 1.71900) + free-running eval + magnitude-bin weighting(방향 검증 완료)
             └─ [ ] v7: magnitude-bin weighting 실제 학습 검증
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

**우선순위 변경 (2026-09-28)**: 아래 ①②③(v33 chaining 검증 / v34 event-boundary MDN / v28 기반 WM→RL 통합)을 폐기한다. 셋 다 SMDN/segment 계열(z-space MDN, per-step 또는 세그먼트 단위) 라인의 다음 단계였는데, R-SSM이 이미 "이벤트 경계에서만 확률적, 구간 내부는 결정론적 물리"로 v34가 풀려던 문제(per-step MDN collapse)를 아키텍처 차원에서 해결한 상태다. 두 라인을 병행 유지할 이유가 없으므로 이벤트 드리븐(R-SSM) 하나로 완전히 전환하고, SMDN 계열은 여기서 종료. ④는 R-SSM의 N-ball 일반화 목표와 직결되므로 유지.

| Step | Content | Status |
|------|---------|--------|
| ~~① v33 chaining 검증~~ | ~~연속 세그먼트 rollout 성능 측정~~ | ❌ 폐기 (이벤트 드리븐 전환) |
| ~~② v34 event-boundary MDN~~ | ~~충돌 순간에만 K=5 MDN~~ | ❌ 폐기 (R-SSM이 이미 이 설계) |
| ~~③ WM → RL 통합 (v28 기반)~~ | ~~v28_dt01_3s best.pt(15.4cm) 기반 SAC critic 보강~~ | ❌ 폐기 (R-SSM 기반으로 대체) |
| **④ 3-ball data + GNN extension** | Generate 3-ball data; extend GNN to N=3 | ✅ 완료 — balanced 3-ball 데이터(3000샷)로 처음부터 재학습(`rssm_v7_3ball`, best_val_rmse=6.044). zero-shot에서 무너졌던 절대 기준 판정이 재학습 후 모두 회복: pooled AUC 0.487→0.894, 개수 판정 41.9%→75.7%, 근시간 예측 AUC 0.61~0.66→0.81~0.84. which-ball(상대 비교)은 zero-shot에서도 강했던 만큼 0.795→0.818 소폭 개선. 가설(구조 문제 아닌 데이터 커버리지 문제) 확인됨. 남은 이슈: 개수 예측 잔여 오차(1개 포켓을 2개로 과대예측), `type_acc` 후반부 하락(0.66→0.59, kw 극값과 연관 추정, 미조사). **→ 2026-09-30: `rssm_v7_3ball` 데이터에 target-target ball_ball normal 버그(17% 이벤트가 placeholder) 발견, 수정 후 `rssm_v8_3ball`로 재학습 — pooled AUC 0.894→0.906, which-ball top-1 0.818→0.862, target-target 이벤트 delta RMSE 14% 개선. `rssm_v8_3ball`이 새 기준 체크포인트.** 상세는 [experiments.md](experiments.md) "3-ball 데이터로 재학습" / "학습 데이터 버그 발견: target-target ball_ball collision normal" 참고 |
| **⑤ R-SSM 물리 엔진 pure_physics 교체** | 아래 "R-SSM 물리 엔진" 섹션 참고 | ✅ 핵심 3곳 완료 (벡터화 서브아이템은 저효용으로 보류) |
| **⑥ R-SSM 배치 forward** | 아래 "R-SSM 배치 forward" 섹션 참고 | ✅ 완료 (Phase 0~3) |
| **⑦ rssm_v5 재학습** | ⑥의 `batch_size` 옵션으로 재시작 필요 — `results/rssm_v5/best.pt`만 있고 history 없이 중단됨 | ✅ 완료 (val_rmse 1.71900, v4 대비 개선) |
| **⑧ v6 코드 구현 + v5로 free-running eval** | `ckpt_every`, ss=0 시점부터 fresh `CosineAnnealingLR`, `evaluate_free_running()` 등 v6용 코드는 구현했으나 **v6 학습은 실행한 적 없음** — `results/`에 `rssm_v5`까지만 존재. 아래 분석은 전부 기존 `rssm_v5/best.pt`에 이 신규 eval 코드를 돌려서 나온 결과. per-shot 심화 분석 결과 **샷 길이에 따라 결과가 뒤집힘**: 짧은 샷(3-7 이벤트)은 free-running이 더 낫지만 긴 샷(8+ 이벤트)은 teacher-forced가 더 낫고 격차가 커짐(compounding error) — aggregate(free-running 1.665 vs 1.719 우위)만 보면 이 반전을 놓침. `viz_rssm.py` 3-way 영상 + 이벤트별 추론값 분해로 더 파보니, 진짜 원인은 "매 스텝 조금씩 누적"이 아니라 **특정 이벤트 1~2개의 대형 회귀 오차**(teacher-forcing으로도 못 고침 — compounding 문제가 아님). 다만 이후 코드 재검증 결과 "MSE의 이봉 타깃 회귀 실패"라는 최초 진단은 과잉 일반화였음이 드러남: 문제는 `type=2`(cue_circular) 하나로 국한되고, 그 안에서도 원인은 이봉 분포가 아니라 입사각(incidence angle) 분포 편중 — 1500샷 서브샘플에서는 `cos 0.02~0.15` 전이구간이 "공백"처럼 보였으나, **전체 5만 샷으로 재검증한 결과 공백이 아니라 실제로 존재하는 얇은 소수 구간(type=2의 7.7%, ~1412건)임을 확인** — 거의 접선(`cos<0.02`, 19%)과 거의 정면(`cos>0.5`, 73%) 두 거대 모드 사이에 낀 진짜 이봉형 분포. 모델은 입사각-크기 관계 자체는 잘 배웠지만(corr 0.657 vs GT 0.712) 이 소수 구간에서 옆 모드의 함수형태를 잘못 끌어씀. 로드맵 ③의 60-step rollout 목표와 직결. 상세는 [experiments.md](experiments.md) "v5 eval 심화 분석" / "렌더링한 15개 샷의 이벤트별 추론값 분해" / "위 결론 정정 — 코드/수치로 재검증" / "후속 검증: 전체 5만 샷" 참고 | ✅ 코드+영상+원인분석+정정+전체데이터 재검증 완료. 데이터 커버리지 보강은 근본 해법 아님(실제 물리 기하 구조의 성질)으로 판단해 보류, magnitude-bin weighting(⑨)으로 우선순위 이동 |
| **⑨ magnitude-bin loss weighting** | `type=2` 내부의 크기별 불균형(근사-제로 다수 vs 대형-delta 소수)은 `compute_vel_type_weights`(type 단위)로 대응 불가 → `compute_vel_magnitude_weights()` 추가, `--vel-mag-weight`로 opt-in. 재학습 없이 v5 체크포인트로 방향 검증: 근사-제로 그룹은 relative error 착시가 아니라 실제 절대오차 실패(mean_abserr=4.1, `\|gt\|<1`인데도)였고, weighting은 이 그룹에 정확히 gradient를 더 실어주는 방향(loss 기여 35%→78%)으로 계산됨을 확인. bin 경계값은 하드코딩(`[0,1,3,10,30,inf]`) 대신 `compute_quantile_bin_edges()`로 데이터에서 자동 파생하도록 리팩터(데이터 분포가 바뀔 때마다 손으로 재조정해야 하는 문제 제거). WM→RL 통합 시 R-SSM이 frozen인지 continual fine-tune인지에 따라 이 자동화가 온라인 재계산으로 확장돼야 할 수 있음 — 아직 미정이라 지금은 보류. 상세는 [experiments.md](experiments.md) "magnitude-bin weighting 방향 검증" / "bin 경계값을 하드코딩 대신 quantile로 자동화" 참고 | ✅ 코드+테스트+방향검증 완료, **v7 학습으로 실제 성능 개선 여부는 미실행** |

### ③ Q-target Augmentation (WM-augmented critic)

```python
s_hat, type_logit = wm(s_1, n_steps=60)         # s_1: actual state after env step
pocket_prob = type_logit[0, :, 4].max().item()   # max pocket probability within 60 steps
q_target = r + gamma * V(s') + lambda * pocket_prob
```

Difference from Dyna: WM provides Q-labels directly rather than generating (s,a,r,s') → **WM-augmented critic**

### R-SSM이 여기 있는 이유 (오늘 pure_physics를 검증한 이유)

Exp-16의 목표는 "큐샷 1회 → multi-step rollout imagining → Q-value MC 추정"(`project_spr_mdn_state.md`). 이 목표가 SSM → SPR-MDN(z-space NLL이 encoder gradient 오염, 문제 7) → SMDN(per-step MDN collapse, 문제 9) 순으로 point-prediction의 한계에 계속 부딪혔고, 그래서 "이벤트 경계에서만 확률적, 구간 내부는 결정론적 물리"로 가는 **R-SSM**으로 피벗했다. R-SSM은 설계 단계부터 `QHead`(attention pool → scalar Q)를 내장하고 있다 (`project_rssm_architecture.md`).

R-SSM이 만드는 rollout은 (1) 이벤트 감지(`event_detector.py`)와 (2) 이벤트 사이 자유운동(`evolve_ball_motion`) 두 개로 이루어진다. 이 substrate가 틀리면 QHead가 아무리 잘 학습돼도 "틀린 rollout에서 뽑은 Q"가 될 뿐이다. 그래서 R-SSM 학습(rssm_v4)을 신뢰하기 전에, long shot에서 오차가 커지는 원인을 추적해 (2) 자유운동 쪽 마찰계수 버그를 잡았고 (`ball_motion.py`, commit 4bbf568), `pure_physics.py`(pooltool-free 재구현체)로 300샷 0.0000cm 일치를 검증했다 — **이게 오늘 pure_physics 작업을 한 이유**. 자세한 내용은 [experiments.md](experiments.md) 참고.

substrate 검증이 끝난 지금, 아래 두 항목이 다음 단계다: QHead 배선 상태 점검, 그리고 substrate 재사용(대체) 검토.

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

**후보 3 관련 경고** — `pocket_prob`(= `predict_pocket` head) 자체를 `eval_pocket_head.py`로 리크 헌팅한 결과(상세: [experiments.md](experiments.md)), AUC=0.919 중 0.827은 물리 시뮬레이션 없는 0-파라미터 기하학 baseline(post-collision 속도 방향 직선 연장)만으로 이미 나오는 값이었다. 즉 이 heuristic이 "학습된 물리 이해"를 반영한다고 보기엔 근거가 약함. QHead도 같은 h를 입력으로 쓰므로, 학습 신호가 생긴 뒤 평가할 때 반드시 같은 방식(trivial/geometric baseline 대비)으로 검증할 것 — 정확도나 AUC 단독 숫자를 그대로 믿지 말 것.

**→ 2026-10-01 결정 (Option A 채택)**: 위 세 후보 중 어느 것도 그대로 쓰지 않기로 했다.
대신 `exp16_wm/sac.py`에 이미 동작 중이던 **다른** world-model critic(`WMSAC`,
R-SSM과 무관한 blind-MLP 버전)을 발견했고, 이 구조를 그대로 재사용해 R-SSM 통합의
첫 단계로 삼기로 했다: 실제 관측된 샷 이벤트 시퀀스를 frozen `rssm_v9_ls01_3ball`에
통과시켜 얻은 `h_final`을 `WMSAC`의 기존 `h_real` 타깃 자리에 꽂는다(R-SSM 자체의
`aggregate_q`/`q_head`는 학습하지 않음 — exp16 전용 목적으로 공유 체크포인트를
in-place 변형하지 않기 위해). 상세 설계와 기각된 대안(Option B 진짜 MBPO 상상 롤아웃,
Option C `pocket_prob` 증강)은 [experiments.md](experiments.md) "MBPO Option A" 항목 참고.

**→ 2026-10-01 구현 완료**: `world_model/rssm_encode.py`(인코더) + `simulator.py`의
`wm_target` 리팩터 + `exp16_wm/train.py --wm-target {traj,rssm}` 전부 구현, 테스트
(7개) 통과, `--wm-target rssm`/`traj` 양쪽 스모크런(2k step) 에러 없이 완주 확인.
QHead는 여전히 R-SSM 자체 학습 신호는 없는 채로 남지만(`exp16_wm`에 별도 q head를
두는 방식으로 우회), "R-SSM latent가 critic에 실제로 쓸모있는지"를 검증할 수 있는
최소 경로가 이제 존재한다. 본격적인 `traj` vs `rssm` pocket-rate 비교 학습은 다음
세션 과제. `feature/wm-rssm-critic-latent` → `dev` 머지 완료. 상세는
[experiments.md](experiments.md) "MBPO Option A" 항목의 "결과" 참고.

**→ 2026-10-01 본격 학습 비교 완료 (seed=0 1개 시드)**: `rssm`이 `traj` 대비 pocket
+3.3pp(58.87%→62.2%), clear +3.6pp(27.8%→31.4%), best_mean_reward +0.108(0.878→0.986)
전부 우세 — R-SSM latent가 blind flat encoding보다 critic에 더 유용하다는 가설 방향은
지지됨. **그러나 `wm`(traj/rssm 둘 다)이 vanilla SAC(pocket 65.87%, clear 32.2%)보다
낮다** — WMSAC 구조 자체를 추가하는 것이 이 설정에서는 순수 SAC 대비 손해였다(학습
시간도 ~1.6~1.7배). 다음 우선순위는 "R-SSM이 traj보다 나은가"의 시드 반복 확인이
아니라, **"WMSAC가 왜 vanilla보다 떨어지는가"** 진단(critic/actor loss가 학습 후반
계속 증가하는 경향 관찰됨 — 수렴 전 종료 가능성). 상세: [experiments.md](experiments.md)
"`wm_target=traj` vs `wm_target=rssm` 본격 학습 비교" 항목의 "결과".

### pocket head 캘리브레이션 진단 (2026-09-30) — QHead/critic 통합 전 선결 과제

`rssm_v7_3ball`(3-ball 재학습, ④ 참고)로 절대 기준 판정(pooled AUC, 개수, 근시간)은 회복됐지만, 이벤트별 확률 궤적을 직접 추적해보니 **h가 새 증거를 제대로 반영해서 갱신되지 않는 문제**가 남아있음을 확인:

- `type_acc` 후반부 하락(epoch150 0.665 → epoch400 0.587)의 원인은 Kendall uncertainty weighting의 `log_var` clamp 포화가 아니라, epoch 350 이후 train/val loss가 갈라지는 **순수 overfitting**. `best.pt`는 val_rmse 최소(epoch450) 기준으로 저장되는데 이 지점은 이미 type_acc 정점보다 낮은 상태 — 체크포인트 선택 기준이 val_rmse 단일값이라 이런 트레이드오프를 못 잡음.
- 틀린 예측(FP/FN)의 패턴: FP는 쿠션 횟수가 많을수록 과대평가, FN은 총 터치수가 많을수록 과소평가. 쿠션 횟수별 accuracy는 3회에서 절벽처럼 떨어짐(0.856→0.690).
- 쿠션 단위로 끊어서 시간순으로 보면 확률이 **직전 값에 눌어붙어서 새 이벤트가 들어와도 잘 안 바뀜** — prior를 posterior로 갱신하는 게 아니라 이전 판단을 우려먹는 것처럼 보임.

**다음 방향**: `h → MLP → 확률`을 매번 새로 계산하는 대신, "직전 확률(prior) + 이번 이벤트 증거(likelihood) → 갱신된 확률(posterior)"를 log-odds 누적 형태로 명시적으로 학습시키는 prior-posterior 구조 제안. 아직 설계 전, 다음 실험 후보. **이 진단을 하는 이유는 QHead/critic이 같은 h를 입력으로 쓰기 때문** — h의 갱신 신뢰도가 낮으면 Q-value 추정도 노이즈가 됨. RL 통합(위 "[ ] Pending" 항목들) 전에 먼저 해결해야 할 선결 과제로 판단. 상세는 [experiments.md](experiments.md) "이벤트별 포켓 확률 궤적 분석" 참고.

**→ 2026-10-01 후속 진단**: `rssm_v8_3ball`의 주기적 체크포인트로 값싸게 더 파본 결과, `h`의
변화량(‖Δh‖)은 확률이 멈춰있을 때나 움직일 때나 거의 동일(corr≈0)해서 **h 자체는 정상 갱신되고
있고, 문제는 `predict_pocket` head 출력단의 sigmoid 포화**로 좁혀졌다. 확신(0/1 근처) 구간에서만
stuck 비율이 급증하고 학습이 진행될수록 심해지는 패턴이 이를 뒷받침. 그래서 몸통을 갈아엎는
prior-posterior 재설계보다 먼저, head 출력단만 건드리는 싼 개입(label smoothing / head 전용
weight decay) 2개를 비교하는 실험을 설계함 — 상세는 [experiments.md](experiments.md) "pocket
head 포화 진단 + 최소 개입 비교 실험 계획" 참고. 이 시도로 안 잡히면 그때 prior-posterior
재설계로 넘어간다.

**→ 2026-10-01 실험 결과 및 결정**: `rssm_v9_ls01_3ball`(label smoothing ε=0.1)과
`rssm_v9_headwd1e2_3ball`(head 전용 weight_decay=1e-2)을 `rssm_v8_3ball`과 동일 설정으로
재학습해 비교. **Label smoothing이 확실한 승자** — 확신 구간 stuck 비율 64.5%→19.2%로 개선,
side-effect 지표(AUC/which-ball/type_acc/pock_acc)는 퇴보 없이 오히려 소폭 개선. **head 전용
weight decay는 사실상 무효**(stuck 비율 64.5%→62.1%, 거의 그대로) — BCE+하드라벨의 과확신
문제는 타깃 자체를 누그러뜨려야 직접 해소되고, 파라미터 크기 억제는 간접적이라 효과가 약함을
확인. **결정**: `pocket_label_smoothing=0.1`을 R-SSM pocket head 학습 기본값으로 채택,
`rssm_v9_ls01_3ball`을 새 기준 체크포인트로 삼는다. prior-posterior 재설계는 불필요 —
head 레벨 개입만으로 포화 문제 대부분 해소 확인. 상세 수치는
[experiments.md](experiments.md) 참고.

### 학습 데이터 버그 수정: target-target ball_ball normal + rssm_v8_3ball (완료, 2026-09-30)

SAC+R-SSM 통합(MBPO식 롤아웃)을 설계하던 중 `rssm_dataset.py::_get_raw_type_and_normal`이
ball_ball 충돌 중 하나가 반드시 cue라고 가정하고 있어서, 타깃-타깃 충돌(cue 비개입, n_balls≥2에서
흔함)의 contact normal이 자리표시자 `[1.0, 0.0]`으로 고정되던 버그를 발견. `rssm_v7_3ball` 학습
데이터의 ball_ball 이벤트 중 16.9%(711/4210)가 이 버그의 영향을 받았음. 두 충돌 볼을 정렬 키로
뽑는 방식으로 일반화 수정 후 데이터 재생성(`data_rssm_3ball_v2`) + 동일 설정으로 재학습
(`rssm_v8_3ball`). target-target 이벤트에서 delta RMSE 14% 개선, pooled pocket AUC 0.894→0.906,
which-ball top-1 0.818→0.862로 개선 확인 — `rssm_v8_3ball`을 새 기준 체크포인트로 채택.
상세는 [experiments.md](experiments.md) "학습 데이터 버그 발견: target-target ball_ball collision
normal" 참고. SAC/MBPO 통합(EventDetector N-ball 일반화 포함)은 이 작업 완료 후 별도 브랜치에서
재개 예정.

### EventDetector N-ball 일반화 (완료, 2026-09-30)

위 데이터 버그 수정과 같은 근본 원인(ball_ball 충돌 중 하나가 반드시 cue라는 가정)이
`world_model/event_detector.py::EventDetector`에도 있어서, `RolloutEngine`을 n_balls≥2에 쓸 수
없는 상태였다(`RolloutEngine` 자체는 이미 N-ball 일반적으로 구현되어 있었음, 확인 완료).
`_ball_sort_key()`(cue 우선, 나머지 숫자 순) + `_classify_event()` 공통 헬퍼로 리팩터링해서
`next_event()`/`full_event_sequence()`가 동일 로직을 공유하도록 수정 — 이전 버그처럼 한쪽만
고치고 한쪽을 놓치는 걸 구조적으로 방지. `TestEventDetectorNBall` 3개 테스트 추가, 기존 2-ball
테스트 전부 회귀 없음. 이제 `RolloutEngine`을 SAC/MBPO 통합에 쓸 수 있는 마지막 선결 조건이
해소됐다. `feature/wm-eventdetector-nball` 브랜치, `dev` 머지 완료. 상세는
[experiments.md](experiments.md) "EventDetector N-ball 일반화" 참고. 다음 단계는 SAC critic에
R-SSM 상상 롤아웃을 연결하는 실제 통합(Option A/B/C) — WM frozen vs continual fine-tune 결정이
아직 미정.

### R-SSM 물리 엔진: pure_physics.py 대체 (완료, 2026-09-28)

교체 대상 3곳(`rssm_rollout.py::advance_balls`, `train_rssm.py::_advance_rvw()`, `viz_rssm.py::_evolve()`) 전부 완료. 벡터화(`evolve_ball_motion_batch()`) 서브 아이템은 저효용으로 판단해 보류. 상세 및 근거는 [experiments.md](experiments.md) 참고.

### R-SSM 배치 forward (완료, 2026-09-28)

`compute_shot_ss_loss`의 shot당 forward(batch_size=1)를 wavefront 기반 배치 forward로 교체 (Phase 0~3). 속도 1.7~3.7배 개선 확인. 상세, 검증 방법, 벤치마크 수치는 [experiments.md](experiments.md) 참고.

rssm_v5 재학습 시 이 `--batch-size` 옵션을 쓸 것. 단, 이 모델 크기(h_dim=32)에서는 MPS 커널 오버헤드가 커서 CPU가 MPS보다 빠르므로 `--device cpu` 권장.

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
