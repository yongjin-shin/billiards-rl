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
             └─ [x] 과제 재정의(clear-speed: 다 넣을 때만 종료, 공당 +1, 샷당 −1, 30샷 안전 상한) + vanilla seed 0 → 7.89샷
                    (후반 하락은 사라졌지만 15만 step 이후 개선 없음, 조준 대신 "세게 치기" 국소 최적)
             └─ [x] pooltool 탐색 플래너 기준점: 기하 + 탐색 4시드 2.24샷 / 기하 없는 무작위 탐색 200후보 2.50샷
             └─ [x] SAC 엔트로피 −2→−7 (seed 0) → 7.39샷: 조준 흔들림 4.4°→0.34°로 줄었지만 조준 오차 그대로(2.08°)
                    → RL 신호로는 세게 치기를 못 벗어남
[-] 행동 재파라미터화 — (공, 포켓) 선택 + ghost-ball 보정각 + 속도 — 보류 (2026-10-09, bitter lesson: 도메인 지식 주입)
[~] 탐색 + 학습 (expert iteration) — 시뮬레이터 탐색이 찾은 샷을 정책이 배우고, 정책이 탐색 후보를 제안 (2026-10-09 제안)
             └─ [~] 0단계: 모방학습(behaviour cloning) — 기하 플래너 교사로 빠른 검증 → 무작위 탐색 교사로
                    └─ [x] raw 좌표로는 과적합(정책 단독 9~11샷) → relative obs → GNN(공·포켓 pair, N-ball 일반화)
                    └─ [x] robust/margin teacher label, overfit test로 학습 결함 발견 → pair supervision: 3개 label 모두 ~4.35 shots/game (SAC 7.89)
                    └─ [ ] 후속 TODO 12개 → 아래 "Next: BC 후속 작업" 참고 (현재 실행 중인 실험 없음, 2026-10-09 22:30)
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

**→ 2026-10-02 원인 진단 + 재실행 (seed=0)**: wandb 곡선 비교 결과 critic loss 상승과
"중반 피크→후반 하락" 패턴은 vanilla에서도 동일하게 나타나 WMSAC 전용 결함이
아니었음을 확인. 대신 `train.py`의 best-checkpoint 선정이 단일 50-episode eval
(이항분포 std ~6.9pp)만으로 이뤄져, 위 비교의 3~7pp 차이가 선정 노이즈 범위 안에
있었다는 게 핵심 원인이었다. 최근 N번 eval의 이동평균 기준으로 선정하도록 고친 뒤
(`--best-ckpt-window`, 기본 5) 3-way를 다시 실행하자 **순위가 바뀌었다**: `rssm`
(pocket 65.6%, clear 34.2%, reward 1.040)이 `vanilla`(63.0%, 32.0%, 0.904)와
`traj`(61.87%, 30.0%, 0.930) 둘 다를 앞섬 — "WMSAC가 vanilla보다 나쁘다"는 이전
결론은 선정 노이즈의 산물이었고, R-SSM latent 자체는 critic에 실질적으로 도움이
된다는 가설이 지지된다. `traj`는 vanilla와 거의 동급. 여전히 단일 시드라 확정은
아님 — 다음 단계는 여러 시드 반복. 상세: [experiments.md](experiments.md)
"위 비교의 원인 진단" 항목.

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

## Next: 탐색 + 학습 (expert iteration) (2026-10-09 제안)

### 왜

- **bitter lesson**: 계산이 늘수록 강해지는 일반적 방법은 탐색과 학습이고, 사람 지식(기하 주입 등)은 결국 이 둘에 밀린다.
  이 과제에서는 탐색이 이미 사람 지식 없이 이긴다 — 무작위 후보 + pooltool 시뮬레이션만으로 50후보/샷 2.98샷,
  200후보/샷 2.50샷(샷당 성공률 94% → 100%). vanilla SAC는 7.89샷(샷당 ~38%).
- **SAC가 못 푸는 이유는 학습 신호**: 노이즈 낀 정책 그래디언트로 행동 범위의 ~0.07%인 정답 각도를 더듬어 찾아야 한다.
  학습 중 조준 흔들림(std 4.5°) 아래서는 정답 조준(21%)이 세게 치기(33%)보다 못해 국소 최적에 머문다
  (experiments.md "SAC 엔트로피 보너스 측정"). 탐색은 같은 시뮬레이터로 정답 샷을 직접 찾는다.
- → 탐색이 찾은 샷을 정책이 **지도학습 목표로** 배우게 하면, 노이즈 그래디언트보다 훨씬 정확하고 강한 신호가 된다.

### 루프

```
반복:
  1. 데이터: 현재 정책으로 게임을 진행. 매 샷 후보 = 정책 샘플 K개 + 무작위 M개
     → 각 후보를 pooltool 복사본에서 쳐보고 최선 a* 선택 (처음엔 즉시 리워드 기준)
     → (s, a*) 저장, 실제로는 a*를 친다
  2. 학습: 정책 π(a|s)를 a*에 대한 지도학습(log π(a*|s) 최대화 또는 회귀)으로 갱신
  3. 정책이 좋아질수록 정책 샘플이 좋은 후보가 되어 탐색 예산(K+M)을 줄일 수 있다
평가: 정책만으로(탐색 없이, 시뮬레이터 없이 한 번에) 친 게임당 샷 수
```

- 사람 지식 0: 후보는 정책 샘플과 무작위뿐, 기하 계산 없음.
- 핵심 지표는 **탐색 없는 정책 단독 성능** — 탐색을 학습으로 얼마나 압축했는가. 기준점: vanilla SAC 7.89샷(하한),
  무작위 탐색 200후보 2.50샷 / 기하 플래너 2.24샷(상한 쪽).
- 보조 지표: 같은 성능을 내는 데 필요한 탐색 예산, 학습 곡선.

### 단계

1. **즉시 리워드 기준 탐색 + 정책 증류** — 위 루프 그대로. 매 샷 넣는 것(조준)을 사람 지식 없이 배우는지.
2. **가치 기반 탐색** — 후보 점수를 즉시 리워드 대신 `r + γ·V(s')`로(V는 탐색으로 개선된 플레이의 return으로 학습).
   다음 샷 포지션과 한 샷 다중 포켓(3.0 → 2.2샷 구간, 충돌 뒤 물리)을 학습으로 다루는 단계.
3. **world model 연결 (MuZero 방식)** — 탐색에 쓰는 pooltool을 학습된 모델로 대체. 그동안의 WM 작업이 의미를 갖는 자리:
   critic 보조 loss가 아니라 탐색용 모델. 선결 조건은 자유 롤아웃 정확도 — 현재 R-SSM 자유 롤아웃의 pocket 판별은
   AUC 0.56(D4)으로 탐색에 쓰기엔 부족하다.

### 비용

- 후보 1개 시뮬레이션 ~2ms → 200후보/샷이면 ~0.4초/샷. 20만 샷 데이터 = 코어 1개로 ~22시간, 10코어로 ~2.2시간.
  50후보/샷이면 1/4. 정책이 좋아져 예산을 줄이면 더 싸진다.

### 열린 질문

- 정책 형태: 지금의 TanhGaussian actor를 그대로 쓰고 지도학습만 바꿀지, 좁은 정답 각도를 잘 표현하는 출력(혼합분포 등)이 필요할지.
- 즉시 리워드 기준 a*는 탐욕적이다 — 2단계 전까지는 포지션 플레이를 배우지 못한다.
- ~~SAC(엔트로피 낮춤) 결과와의 관계~~ → 답 나옴(2026-10-09, seed 0): 조준 흔들림을 0.34°까지 줄여도 정책은 세게 치기에서
  벗어나지 못했다(조준 오차 2.08° 그대로, 7.89 → 7.39샷). 학습 신호 문제가 노이즈보다 근본적이라는 근거 → 정답을 직접 주는
  모방학습(0단계)부터 진행 중.

---

## Next: BC 후속 작업 (pair supervision 이후, 2026-10-09)

### 현재까지의 결과 (GNN, relative obs, 증강 없음, 1.67만 states, policy alone 500 games, seed 0 1회)

| label | WTA, const lr (이전) | **pair supervision, cosine lr** | intended pair made (label 기준) |
|---|---|---|---|
| robust teacher v1 | 6.18 shots/game | **4.35** | 97.7% |
| margin teacher v2 (각도×속도 2D margin) | (미실행) | **4.41** | 95.5% |
| geom teacher (32 cand.) | 9.48 | **4.34** | 73.5% |

SAC 7.89, 엔트로피 낮춘 SAC 7.39, teacher 자체: geom 2.24 / random search 2.50 / robust 2.98. **label 종류의 차이는 pair supervision에서 사라졌다** (이전 6.18 vs 9.48은 WTA 불안정성의 산물).
policy의 shot 실패 분해(robust+pairs, val): chosen pair miss ~24%, chosen pair made지만 scratch 8%, 다른 pair를 clean하게 넣음 1~4%.
선택한 pair와 action의 일관성(clean pocket 중 chosen pair였던 비율)은 94~98%.

### TODO (우선순위순)

| # | 항목 | 이유 | 비용 | 의존 |
|---|---|---|---|---|
| 1 | **Seed 반복** (pair supervision 3개 label × seed ≥3) | 지금까지 전부 seed 1회. 4.34~4.41 차이는 판단 불가 | BC 20분/run | – |
| 2 | **coarse_hits (N,18) 저장 + factorized selection** (ball head + per-ball pocket softmax, soft target ∝ coarse hits) | label은 top-2 pair만 → feasible한데 unlabeled인 pair가 false negative (sample 0: (1,2), (2,2)). ball 1은 supervision 없음 | coarse stage만 재계산 ~15-20분 | – |
| 3 | **Outcome-grounded pair label** (`good_made`로 pair 교체) | geom label의 26%가 intended pair와 다른 outcome | 이미 `good_made` 있음, 코드만 | – |
| 4 | **Scratch를 별도 target으로** | chosen pair를 넣고도 scratch 8-11% | 낮음 | `good_scratch` 있음 |
| 5 | **Made head** P(made_ij \| e_ij, a_ij) → pair 선택 보정(verifier) | selection이 "이 action이 성공할까"를 직접 보지 않음 | 중 | 3 |
| 6 | **Dense success map** (sims 500만-2,100만 건) + test-time refinement ∇_a P(made) | miss 24%(0.1° 정밀도) 직접 겨냥. 단 sims에 (ball, pocket) identity 없음 → 기록 추가 후 재simulation 또는 angle matching(purity 72-99%) | 높음 | 5 |
| 7 | **pair supervision 조건으로 이전 비교 재실행**: MLP vs GNN, augmentation(flip4는 pocket 순열 필요), data 8×, model size | 이전 모든 GNN 수치는 WTA 학습 | BC 여러 번 | 1 |
| 8 | **robust+pairs의 val loss 이상(train 45, val 2990-3300)** 원인 조사 | outlier state가 val loss를 지배; label/pair 매핑 버그 가능성 배제 필요 | 낮음 | – |
| 9 | **Checkpoint 선택 잡음**: val within-0.2° 비율이 epoch마다 ±8%p 흔들림 → 최댓값 선택은 낙관적 | EMA weights / 더 큰 val / 마지막 epoch 평균 | 낮음 | – |
| 10 | **DAgger**: policy가 가는 state에서 teacher에게 다시 label | train/val은 teacher states만. policy rollout은 다른 분포 | 중 | 1 |
| 11 | **Multi-ball / position play**: robust teacher는 1.0m/s 1-ball shot을 골라 2.98 shots (geom 2.24) | "창이 충분히 넓은 shot 중 more balls" 기준 | 중 | – |
| 12 | **SAC fine-tune from BC init** (엔트로피 낮춤) | BC 4.3 shots → teacher 2.2~3.0 격차, position play는 RL이 필요할 수 있음 | 중 | 1, 8 |

실행 순서 제안: 1(seed) → 8(val loss 이상) → 2+3+4(label/selection 개선, 한 번에) → 5 → 7 → (6, 10, 11, 12는 결과 보고 결정).

### 보류/기각
- 행동 재파라미터화(ghost-ball 보정각): bitter lesson, 아래 보류 섹션.
- lr decay 단독: 전체 데이터에서 도움 없음(robust 6.18 → 12.25).
- margin v2 no-pairs(WTA) baseline: queue script 오류로 미실행, pair supervision이 대체해서 필요하면 나중에.

---

## 보류: 행동 재파라미터화 — (공, 포켓) 선택 + ghost-ball 보정각 + 속도 (2026-10-09 제안 → 같은 날 보류)

> **보류 이유 (bitter lesson)**: 기하를 행동에 넣으면 문제의 대부분을 사람이 푸는 셈이고 학습이 할 일이 거의 남지 않는다.
> 그럴 거면 플래너로 끝이다. 실제로 기하 없이 무작위 후보 + 시뮬레이션 탐색만으로도 200후보/샷이면 2.50샷·샷당 성공률
> 100%가 나온다(experiments.md "기하 없이 탐색만으로는?"). 즉 이 과제는 사람 지식 없이 계산(탐색)으로 풀리고, SAC가 못
> 푸는 이유는 지식 부족이 아니라 학습 신호가 약하기 때문이다. 그래서 방향을 "탐색 + 학습"(아래 섹션)으로 바꾼다.
> 아래 분석(조준 기하 vs 충돌 뒤 물리 구분 등)은 문제 이해용 기록으로 남긴다.

### 왜

pooltool 탐색 플래너를 RL과 같은 clear-speed 과제로 분석하다가 나온 방향이다(experiments.md "과제 재정의",
"기준점: pooltool 탐색 플래너" 참조).

- **무작위로 치면 6.7%만 들어간다**(무작위 행동 200개 기준). 행동 공간 대부분이 헛샷이다.
- **ghost-ball 직선 조준(보정 0, 속도 3m/s)만으로 94%의 샷에서 들어가는 (공, 포켓) 조합이 있다.** 그 각도 주변
  후보(±0.4°, 속도 3개)는 절반이 실제로 들어간다. 플래너는 평균 2.2샷에 다 넣고 샷당 성공률은 100%다
  (seed 0, 50게임/116샷 분석 + 500게임 본평가).
- **RL(vanilla, 같은 과제)은 게임당 ~7.6샷**, 샷당 ~0.4개다(seed 0, ~480k step 시점).
- 같은 이야기를 하는 기존 결과: D4에서 critic Q의 pocket AUC 0.66 < 0-파라미터 직선 기하 0.746. Exp-12에서
  행동 파라미터화만 바꿔도(delta → abs) 5M step에서 −3.9pp.

→ 이 과제의 어려움은 물리가 아니라 **조준 기하**다. 지금 행동 공간(`delta_angle ∈ [−π, π]`, 가장 가까운 공 방향
기준)에서는 RL이 좌표로부터 ±0.1° 수준의 조준을 스스로 배워야 하고, 포켓되는 각도 폭이 행동 범위의 ~0.07%다.
`delta_angle`이 "공 중심 조준"을 행동에 넣어서 abs_angle보다 나았듯이, 한 단계 더 나가 **"공을 포켓으로 보내는
조준"을 행동에 넣는다.**

### 행동 정의

```
a = (k, δ, v)
  k = (공 i, 포켓 j)   이산, 남은 공 × 6포켓 ≤ 18. 컷 각도 > 80° 등 불가능한 조합은 마스킹
  δ ∈ [−Δ, +Δ]        연속, ghost-ball 조준각 φ_ghost(i, j)에 더하는 작은 보정각 (예: Δ = 2°)
  v ∈ [0.5, 8.0] m/s   연속, 속도
실제 타격 각도 = φ_ghost(i, j) + δ
```

- δ = 0이면 "공 i를 포켓 j로 보내는 직선 기하 조준"이다. 플래너 분석상 이것만으로 대부분 들어가므로, 정책은
  throw·속도에 따른 작은 어긋남만 보정하면 된다.
- 포켓되는 각도 폭(±0.1~0.2°)이 행동 범위의 ~0.07%에서, ±2° 범위 기준 ~5% 수준으로 커진다.
- Exp-17(HRL) 17b/17c와 이산 선택은 같다. 차이는 연속 부분이 각도 전체가 아니라 **기하 조준 주변의 보정**만
  배운다는 점이다.

### 구현 후보 (순서)

1. **k를 휴리스틱으로 고정**(예: 컷 각도 최소 조합)하고 (δ, v)만 SAC로 학습 — 조준 재파라미터화 효과만 먼저 분리 확인.
2. **k까지 학습** — hybrid SAC(이산 categorical + 연속) 또는 Exp-17 HRL 구조(System 2 = k, System 1 = (δ, v)) 재사용.

### 조건 / 비교

- 과제: clear-speed 과제(다 넣을 때만 종료, 공당 +1, 샷당 −1, 30샷 안전 상한).
- 비교 대상: vanilla(clear) 4시드, 플래너 기준점(~2.2샷), **학습 없는 휴리스틱**(컷 각도 최소 조합 + δ = 0)
  — 학습이 기하 이상의 것을 더하는지 확인하기 위해.
- 지표: 평균 샷 수, 샷당 성공률, 학습 곡선.

### 열린 질문

- **기하로 되는 부분과 안 되는 부분이 다르다.** ghost-ball 기하는 "첫 번째 공을 포켓으로 보내는 조준"까지만 준다.
  충돌 뒤의 일 — 큐볼이 꺾여 나가 회전·마찰로 휘어 굴러가는 경로, 첫 공이 다른 공을 맞히는 것, 큐볼이 두 번째
  공을 넣는 것 — 은 직선 기하로 계산되지 않는다. 플래너가 한 샷에 2개를 넣은 샷(seed 0 분석에서 26%, 첫 샷은
  54%)도 기하로 계산한 게 아니라 후보를 pooltool로 쳐보고 우연히 2개가 들어간 것을 고른 결과다.
  - 7.6샷(RL) → 3.0샷(매 샷 1개씩 확실히): **조준 기하**로 해결되는 부분, 격차 ~4.6샷. 이 재파라미터화가 겨냥하는 곳.
  - 3.0샷 → 2.2샷(플래너): **충돌 뒤 물리**(한 샷 다중 포켓), 격차 ~0.8샷. 기하로는 안 되고 미리 쳐보기나 학습이
    필요하다. 재파라미터화 후에는 δ·v 범위 안에서 RL이 경험으로 배워야 하는 부분이고, 물리를 아는 모델(world
    model)이 실제로 도움이 될 수 있는 자리도 여기다.
- 다음 샷 포지션(큐볼 위치)까지 고려하려면 k 선택이 장기 가치를 봐야 한다 → 이산 Q가 필요할 수 있다.
- 뱅크/키스 샷은 이 행동 공간에서 표현할 수 없다. 플래너 본평가(seed 0)에서 무작위 탐색이 필요했던 샷은
  9/1110이라 영향은 작아 보인다.

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
