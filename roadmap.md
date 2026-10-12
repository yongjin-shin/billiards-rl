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

### 현재 상태 (2026-10-12) — 결과 사다리와 병목

정책 단독 500 games, seed 3개 평균(별도 표기 없으면 GNN, relative obs, robust label, pair supervision). 상세와 근거는 experiments.md의 2026-10-11 항목들.

| 단계 | shots/game | success | 무엇을 바꿨나 |
|---|---|---|---|
| vanilla SAC (참고) | 7.89 | 38.7% | – |
| BC baseline A | 4.51 ± 0.23 | 66.2% | label pair 없는 1.37% row를 WTA loss로 학습(그 row가 train loss의 66%를 차지) |
| **BC 새 baseline** (`--no-wta-fallback`) | **4.14 ± 0.14** | 71.9% | 그 row를 loss에서 제외 |
| + made head(look-ahead 2단계) select | 4.11 ± 0.20 | 72.5% | action을 받는 made head를 학습(칠 때는 안 씀) → 해롭지 않음 |
| + verify + refine(각도 31 × 절대 speed 7) | **3.90 ± 0.21** | 76.7% | made head로 pair 재선택 + action 후보 평가(시뮬레이터 없이) |
| **+ encoder 깊이 4**(2층 → 4층, made head 포함, verify+refine) | **3.55 ± 0.10** | 84.2% | encoder MLP를 깊게(기하 입력 없이 학습만으로) — pair head 각도 오차 −24%, 창 적중 81 → 86.5% |
| 깊이 6 / 깊이 8 (층 수만 변경, 500 epoch) | 3.77 ± 0.16 / 3.94 ± 0.11 | 79.6% / 76.0% | 깊이 4보다 **나쁨**(8층은 control 수준). train loss가 안 내려갔다 → **1500 epoch에서 깊이 6이 회복(아래 행): 학습 예산 부족이었음** |
| 깊이 6, 1500 epoch (깊이 4도 1500 epoch) | 3.51 ± 0.05 (깊이 4: 3.59 ± 0.10) | 85.1% (83.4%) | 같은 길이 깊이 4 대비 "효과 인정" 기준 미충족(각도 −0.009°는 경계, 게임 차이는 표준편차 안). 깊이 4는 길게 돌려도 불변. 비용 약 3.5배 |
| (진단 전용) + 기하 조준각 차이 입력 | 3.46 ± 0.12 | 86.4% | made head에 ghost-ball 조준각과의 차이를 입력 — **상한 참고, 최종 모델이 아님** |
| teacher (robust / geom) | 2.98 / 2.24 | – | – |

**병목 = `e_ij`(pair embedding)가 각도를 0.1° 정밀도로 담지 못하는 것.** 근거: (1) 남은 공 2개 state는 성공 창 폭이 같은데(0.96° vs 0.92°) 모델 각도 오차가 2배(0.22° vs 0.12°); (2) label은 순수 기하 조준각의 ±0.05° 안이라 깨끗하다; (3) made head는 train과 val 정확도가 같아(AUC 0.831 vs 0.846) 다양성이 아니라 해상도 문제; (4) head에 기하 조준각 차이를 직접 주면 창 위치 적중이 81% → 95%, 2개 state 성공이 66.8% → 76.8%.

**2026-10-12 보강·정정** (experiments.md "깊이 4에서 남은 실패는 어디서 오는가"): (5) 깊이 4에서 label pair를 정확히 줘도 verify+refine이 13.7%p 실패하고 선택 손실은 4.2%p다 → 남은 실패의 약 77%가 선택이 아니라 정밀도다. (6) 각도만 고치는 refine(made head 격자·gradient)은 이득 0이고 refine의 이득은 speed 격자에서 나온다. **정정**: (4)와 용량 실험에서 쓴 "스캔 argmax가 창 안"은 made head의 해상도가 아니라 **pair head proposal의 적중률과 같은 값**이다(깊이 4: 86.5% = 86.5%, 우연 53.8%; proposal이 창 밖인 44개 state에서 head가 창 안으로 옮긴 것은 4.5%). made head는 proposal에 앵커돼 있어 proposal 밖의 창 위치 정보가 없고, 기하 입력(95.1%)만 그 정보를 준다. 이전 서술은 기록으로 두고 해석만 바로잡는다.

**기각된 가설** (전부 실험으로): 일반화/데이터 양(공 2개 데이터 3.7배 추가에도 불변), 포켓된 공 slot 표현, label 불일치·불규칙(flip), 방해 공, 창 폭/난이도, speed hedging(모델이 두 모드 사이로 평균을 내지만 성공에 영향 없음), selection 구조/target(①②), scratch head(④), made head 보조 target(⑤), 이웃 message 간섭(`--geom-angle`), 다른 공의 target(③ `good_made`: combination shot이 pair head 학습을 망가뜨림), scatter 구제(타겟이 아닌 공이 들어가는 비율 ~1%), refine 후보 수/각도 범위(포화).

**열린 결정** (사용자가 정한다):
1. 학습만으로 정밀도를 못 올리면 ghost-ball 기하 조준각 차이를 made head(verifier)에만 입력으로 허용할지. "탐색 + 학습" 방침(bitter lesson)과 충돌한다. 단 위 정정대로 "argmax ≥ 90%"는 made head 해상도가 아니라 proposal 정확도의 기준이다.
2. **다음 단계 (2026-10-12)**: (a) search → distill(TODO 19: 분포 head + 시뮬레이터 best-of-N → 성공 샘플로 재학습), (b) speed 쪽 개선(TODO 21), (c) loss 스케일. 학습 길이 실험은 끝났다(깊이 축은 일단 닫음: 깊이 4, 500 epoch이 비용 대비 최선).

### 아이디어 지도 (2026-10-12) — 주제별 2단계 정리

현재 병목은 **고른 pair의 action 정밀도**다(깊이 4에서 label pair를 줘도 13.7%p 실패, pair 선택 손실은 4.2%p). 그 실패는 (pair head 출력, select 기준으로) **각도(+9.9%p)와 speed(+8.0%p)가 거의 가산적**으로 만들고, 깊이 4는 각도만 개선했다. 단 실제 정책은 verify+refine이라 refine의 speed 격자가 speed 문제를 일부 풀고 있어, speed 상한 +8.0%p는 파이프라인의 개선 여지가 아니었다(speed 분포 head로 확인). 상태 표기: **[완료]** 결과 있음 / **[진행]** 실행 중 / **[미착수]** 아직 안 함 / **[보류]** 사전에 멈춤. 숫자는 experiments.md의 해당 절.

**1. 데이터와 teacher**
- **[완료]** label 없는 1.37% row를 WTA loss로 학습하던 문제 → `--no-wta-fallback`으로 4.51 → 4.14 (그 row가 train loss의 66%)
- **[완료]** 공 2개 데이터 3.7배 추가 → 불변. label 불일치·flip, 방해 공, 창 폭/난이도 → 기각
- **[완료]** 게임 vs val 격차 규명(#16): 분포 이동 없음(남은 공 수별 성공률이 val과 1.5%p 이내). 실패 직후 state만 어렵고(71.3% vs 82.2%) 실패의 27%만 거기서 나옴 → 정책이 가는 state에서 label을 받는 이득의 상한은 clean +2.0%p 정도
- **[보류]** DAgger(#10): 위 결과로 우선순위 낮음
- **[미착수]** position play·multi-ball teacher(#11) (지금 게임 성능은 shots/game ≈ 3 / 샷 성공률로만 결정됨)

**2. Pair 선택**
- **[완료]** factorized ①, made selection target ②, scratch head ④ → control과 동일. `good_made` ③ → 7.63으로 악화. made head 보조 target ⑤ → 효과 없음
- **[완료]** 깊이 4에서 선택 손실 4.2%p (병목 아님). 깊이 > 4에서는 3개 state의 선택이 나빠짐

**3. Action 정밀도: 모델 구조와 학습 (현재 병목)**
- **[완료]** GNN + pair supervision, encoder 깊이 4 → 3.90 → 3.55. 폭 256 → 작은 효과(보류). 깊이 6/8 → 3.77 / 3.94(최적화 문제)
- **[완료]** 이웃 message 빼기(`--geom-angle`), 포켓된 공 slot 표현 → 효과 없음. angle wrap-aware loss → policy 악화(이유 미규명). lr decay 단독 → 악화
- **[완료]** 학습 길이 3배(1500 epoch): 깊이 6의 500 epoch 열세(3.77)는 학습 예산 부족이었고(1500 epoch에서 3.51), 같은 길이의 깊이 4(3.59)와는 차이 없음. 깊이 4는 길게 돌려도 불변
- **[미착수]** residual/LayerNorm + lr 3e-4(#18, 필요성 크게 줄어듦: 깊이 6도 길이로 수렴함), 좌표 sin/cos(#14), 폭+깊이 결합
- **[미착수]** loss 스케일 `--angle-scale-deg 0.1`, EMA/마지막 epoch checkpoint(#9), lr/clip 정리(#1b), Laplace NLL robust loss(#20)
- **[미착수]** 2개 state 약점의 구조적 원인(#17), 이전 비교의 pair supervision 조건 재실행(#7)

**4. 추론 때 보정 (made head, verify/refine)**
- **[완료]** made head + verify + refine(각도 31 × 절대 speed 7) → 3.90. refine의 이득은 각도가 아니라 speed에서 나옴. 각도만 조정(격자·gradient)은 이득 0
- **[완료]** made head의 "스캔 argmax가 창 안"은 proposal 적중률과 같은 값(정정), head는 proposal에 앵커됨
- **[완료]** verify+refine 출력 기준 oracle 교체와 실패 예산(판단표 C): 정책 실패 17.8%p = 각도 7.7 + pair 선택 4.2 + speed 2.9 + 결합·기타 1.9 + 못 고침 1.2. speed 여지는 select 기준 8.0이 아니라 2.9%p, 각도·speed 결합은 1.9%p로 작음(cascade 불필요). 남은 공 2개 각도 손실 16.3%p
- **[완료]** oracle 교체(label의 각도/speed로 바꿔 치기): 각도 +9.9%p, speed +8.0%p로 거의 가산적(판단표 E). (select 기준, 위 실패 예산으로 대체됨) **깊이 4는 각도 쪽만 개선**(speed는 그대로), 남은 공 2개의 약점은 각도
- **[완료·참고용]** 기하 조준각 차이 입력 → 3.46 (상한 참고, 최종 모델 아님)
- **[보류]** gradient refine ∇ₐP(made)(#15) — head가 proposal 밖 정보를 가질 때까지
- **[미착수·아이디어]** made head를 proposal 앵커에서 풀기(입력 차이 제거 또는 proposal 무작위 섭동). 격자 oracle의 상한(99.2% vs 84.8%)은 크지만, 기하 입력을 준 made head도 verify+refine이 82.6%(2층)에 그쳐 학습된 verifier로 얼마나 따라갈지는 불확실하다
- **[완료·기각]** 재시도 시 탐색(TODO 22): miss·scratch 직후 샷의 object ball 배치가 그대로인 비율은 22~24%뿐(대부분 새 배치). 그대로인 그룹이 더 어렵지만(−9.9%p) 전체 샷의 4%라 이득 상한 +0.4%p → 접음
- **[완료·부정적]** speed 분포 head(TODO 21): L1과 구간별 확률(21구간) 모두 select 수준에서는 의도대로(select clean +3.2/+3.6%p, 틈 실패 70% → 43%) 그러나 verify+refine 파이프라인에서 게임이 나빠짐(3.55 → 3.78/3.82, 3 seed 모두). 실패가 틈에서 "잘못된 모드(아래)"로 이동. 채택하지 않음. 확률대로 뽑기도 도움 안 됨

**5. 탐색 + 학습 (expert iteration, RL 연결)**
- **[완료]** Laplace 샘플링: 샘플 하나를 그대로 치면 손해, 시뮬레이터가 6개 중 고르면 81.8 → 92.6% (추론 때 시뮬레이터가 있을 때의 상한)
- **[완료]** 틀 정리: 샷 단위 contextual bandit, RL 용어 대응
- **[완료]** 격자 oracle: 모델 action 주변(각도 ±0.3° × speed 7개)에 해가 99.2% state에 있는데 made head는 84.8%만 고름 → 시뮬레이터 검증/탐색의 근거
- **[미착수]** search → distill(#19): A positives-only / B baseline REINFORCE + negatives / C 경계 중앙 label
- **[미착수]** 분포 head(Laplace/mixture) 학습, policy gradient 계열 fine-tune(#12)

**6. 방법론과 검증**
- **[완료]** control 동일성 확인, 판단표 사전 등록, 기준선 정정, `eval_by_balls` / `diagnose_action` 도구화
- **[미착수]** seed 5개 반복(판단표 E/F 행이 나올 때), margin/geom32 label의 seed 반복(#1)

**결정 대기**: (1) 기하 조준각 입력을 made head에만 허용할지(bitter lesson과 충돌), (2) 다음 단계를 search → distill로 갈지 residual/LayerNorm 깊이 재시도로 갈지.

### 지금 실험할 만한 것 (2026-10-12, 우선순위, verify+refine 기준 실패 예산 반영)

**기준 데이터 — 정책이 실제로 치는 행동의 실패 예산(깊이 4, 실패 17.8%p)**: **각도 7.7 (43%)** / pair 선택 4.2 (23%) / speed 2.9 (16%) / 각도·speed 결합·기타 1.9 / label action으로도 못 고침 1.2. 남은 공 2개 state의 각도 손실은 16.3%p, 3개 state의 pair 선택은 6.5%p. 판단표 C(각도 지배). 상세는 experiments.md "verify+refine 출력 기준 oracle 교체와 실패 예산".
자원: 학습 길이 실험이 끝나 CPU가 비었다. 판단표는 실행 직전에 결과를 보기 전에 정한다(experiments.md 규칙).

| 순위 | 실험 | 겨냥하는 몫과 근거 | 비용 | 시점 |
|---|---|---|---|---|
| 1 | **각도 정밀도: loss 스케일 `--angle-scale-deg 0.1`** (깊이 4, 500 epoch, seed 0/1/2) | 각도 손실 7.7%p(2개 state 16.3%p)가 최대 항목. 이전 스케일 실험은 MLP 시절 1회뿐, 깊이 4에서는 안 잼. 코드 변경 없음 | BC 약 45분 | 지금 가능 |
| 2 | **각도 정밀도: 창 폭 가중 손실** (좁은 창에 정밀도를 몰아주기) | 각도 창 폭이 state마다 3.6배 이상 다름(실측). 점 label은 폭을 모름. 가설(미검증). 폭 계산이 선행(약 16분, 학습 없음) | 폭 계산 + 손실 코드·테스트 + BC 3 runs | 1번 결과 후 |
| 3 | **각도 verifier**: made head가 각도를 고르지 못함(격자에 해 99.2% vs made head 84.8%). 시뮬레이터 검증/탐색(#19) 또는 made head의 각도 해상도 개선 | 각도 손실이 proposal 정밀도 때문인지 verifier 때문인지 가르는 실험이 먼저. #19의 각도 이득은 teacher label이 이미 창 중앙이라 약할 수 있음 | 설계 필요 | 1·2번 뒤 |
| 4 | **pair 선택** (손실 4.2%p, 3개 state 6.5%p) | 새 아이디어 없음. 3개 state에서 선택 손실이 각도와 비슷한 크기 | – | 보류 |
| – | speed 개선 | **접음.** 최종 행동 기준 여지가 2.9%p뿐이고(select 기준 8.0%p는 refine이 풀어 줌), speed 분포 head는 게임이 나빠짐 | – | – |
| – | cascade / 결합 모델링 | **접음.** 각도·speed 결합 손실이 1.9%p뿐이라 얻을 몫이 작음 | – | – |
| – | 깊이 / 학습 길이 / 재시도 노이즈 / DAgger | **닫음.** 깊이 4, 500 epoch이 비용 대비 최선(각도 손실은 깊이가 줄이나 speed·선택이 늘어 깊이 6도 P가 같음). 재시도 이득 상한 +0.4%p, 정책 state 이득 상한 +2.0%p | – | – |
| 상시 | 새 run부터 **마지막 epoch 가중치도 저장**(#9) | 최댓값 선택의 낙관 편향 보정(best-vs-last 비교) | 코드 약 5줄 + 테스트 | 다음 학습 run 전에 |

**일단 보류**: sin/cos 표현(#14)과 Laplace NLL(#20)은 loss 스케일 결과가 나온 뒤, gradient refine(#15)과 made head 앵커 해소는 기하 입력 결정 후(상한은 크지만 학습된 verifier가 따라갈지 불확실), 폭 256 + 깊이 4 결합은 효과가 작아 보류.

### 다음 방향 (판단 트리)

결과별 판단 세부표는 experiments.md "encoder 용량 실험 — 결과별 판단표"(기준점과 정정 포함)에 있다.

| 지금 | 결과 | 다음 |
|---|---|---|
| encoder 용량 — 폭 256: 효과 작음(보류). 깊이 4: 효과 큼(판단표 C). 깊이 6/8(500 epoch): 깊이 4보다 나쁨. **학습 길이 3배: 깊이 6은 회복(3.77 → 3.51), 같은 길이 깊이 4(3.59)와 차이 없음, 깊이 4는 불변** | **깊이 축은 닫는다**: 깊이 4, 500 epoch 유지(비용 대비 최선). 깊이 8·residual/LayerNorm은 보류 | 남은 공 2개의 각도에서 깊이 6의 −20%(사후 관찰)가 중요해지면 재검토 |
| 깊이 4 고정 상태에서 남은 실패 분해(experiments.md "깊이 4에서 남은 실패는 어디서 오는가") — **pair를 맞춰도 13.7%p 실패(선택 손실 4.2%p의 약 3배), 각도만 고치는 refine은 이득 0** | 학습 중 각도 정밀도를 더 짜내는 것 vs 시뮬레이터로 후보를 검증하는 탐색 | **사용자 결정 필요**: search → distill(TODO 19)을 먼저 할지, 깊이 재시도(TODO 18)를 먼저 할지 |
| 깊이 재시도에서 개선 | 소폭 ↑ | sin/cos(Fourier) 좌표 표현을 같은 지표로 시험 |
| | 변화 없음 | 학습 쪽(긴 학습, lr, EMA/마지막 epoch checkpoint = #9, loss 스케일 `--angle-scale-deg` 0.1) |
| | train ↑ val ↓ | 데이터/정규화 |
| made head가 proposal 밖의 정보를 가질 때(proposal이 창 밖인 state에서 head가 창 안으로 옮기는 비율이 의미 있게 오를 때; 깊이 4는 4.5%) | – | gradient로 action을 연속 공간에서 조정(`∇_a P(made)`, #6의 남은 부분). 지금 구조에서는 각도 gradient/격자 이득 0 |
| 용량·표현·학습을 다 해도 proposal 정확도(= 스캔 argmax)가 90%에 못 미침 | – | **사용자 결정**: made head에만 기하 입력 허용 vs 학습만 계속 vs 탐색(TODO 19) |
| 게임과 val 격차 | **규명 완료**: 분포 이동 없음, 정책 state의 이득 상한 clean +2.0%p | DAgger(#10)는 우선순위 낮음 |

### 진행 상태 (2026-10-12, 아래 TODO 표 번호 기준)

| # | 상태 |
|---|---|
| 1 | **완료.** 새 baseline은 `--no-wta-fallback` 4.14 ± 0.14 (baseline A 4.51은 fallback 때문에 높았음) |
| 2+3+4 | **완료, 효과 없음.** ①구조·②target·④scratch는 control(4.08 ± 0.28)과 동일, ③ `good_made` target은 combination shot 때문에 7.63으로 악화(같은 공만 남기면 복구). 상세는 experiments.md |
| 5 | **완료.** action-conditional made head(`--made2-head`), verify로 선택 재정렬 |
| 6 | **부분 완료.** sims에 (ball, pocket) identity를 복원(`sim_identity.py`)하고 후보 평가 방식 refine까지 완료(각도는 포화, speed는 절대 격자가 최선). **∇_a P(made)는 head 정밀도(argmax ≥ 90%)가 오른 뒤로 보류** |
| 9 | 아직. 용량 실험에서 학습 쪽 시도가 필요해지면 먼저 한다 |
| 10 | 보류(위 판단 트리) |

**신규 TODO** (2026-10-12)

| # | 항목 | 이유 | 비용 | 의존 |
|---|---|---|---|---|
| 13 | **encoder 용량: 완료(2026-10-12).** 깊이 4가 비용 대비 최선(3.90 → 3.55), 폭 256은 효과 작음(3.84, 보류), 깊이 6 / 8은 500 epoch에서 3.77 / 3.94로 나빴으나 **1500 epoch에서 깊이 6이 3.51로 회복**(학습 예산 부족), 같은 길이 깊이 4(3.59)와는 차이 없음. 깊이 4는 길게 돌려도 불변 | 정밀도 병목이 용량(특히 깊이)인지 | – | – |
| 14 | **좌표 sin/cos(Fourier) 표현** | 작은 각도 차이를 더 잘 구분하도록 | BC | 13 결과에 따라 |
| 15 | **gradient refine** `∇_a P(made)` — **2026-10-12: 보류.** 4개 모델 모두 각도 gradient·격자 refine 이득 0(방향은 95~96% 맞지만 proposal로 되돌리는 방향, made head가 proposal에 앵커) | 후보 격자 없이 연속 탐색. 단 head가 proposal 밖의 정보를 가져야 의미 | 낮음(추론) | made head를 proposal 앵커에서 풀기(미착수) 또는 기하 입력 허용(열린 결정 1) |
| 16 | **게임 vs val 격차 확인: 완료(2026-10-12).** 분포 이동 없음(남은 공 수별 성공률이 val과 1.5%p 이내), 실패 직후 state만 어려움(71.3% vs 82.2%, 실패의 27%). made head 이득이 게임에서 작았던 이유는 이 분석으로는 설명되지 않는다(미해결) | policy가 가는 state는 teacher state와 분포가 다름 | – | – |
| 17 | **2개 state 약점의 구조적 원인** | 용량으로 안 풀리면 입력 분포·최적화 쪽(남은 공 2개 state만 각도 오차 2배) | – | 13 결과에 따라 |
| 18 | ~~깊이 > 4 최적화 안정화~~ **[보류]**: encoder MLP에 residual connection 또는 LayerNorm, lr 3e-4로 6/8층 재시도 | 깊이 6이 1500 epoch에서 수렴해 필요성이 줄었고, 깊이 6이 깊이 4를 못 넘음 | – | – |
| 19 | **search → distill (expert iteration)**: head가 분포(Laplace/mixture)를 내고, 샘플 N개를 시뮬레이터로 쳐서 성공한 샘플을 label로 재학습(filtered BC). 샷 단위 contextual bandit(shots/game ≈ 3 / 샷 성공률이 모든 모델에서 맞음) | 같은 label pair에서 샘플 6개 중 하나라도 성공하는 비율이 b = 0.10°에서 깊이 4 81.8% → 92.6%, 격자 oracle은 각도 31 × speed 7 중 99.2%(made head는 84.8%). critic이 필요 없고(시뮬레이터가 진짜 1-step return) 추론은 정책 단독. RL 대응은 experiments.md "head에서 샘플링해서 치면?" | 시뮬레이터 약 100k 샷/라운드(약 4분, 추정) + BC | **사용자 결정**(열린 결정 2) |
| 21 | ~~speed 쪽 개선~~ **완료·부정적(2026-10-12)**: speed head를 L1 / 구간별 확률(21구간)로 바꿨더니 select clean은 +3.2/+3.6%p 오르고 틈 실패 70% → 43%였지만 verify+refine 게임이 3.55 → 3.78/3.82(3 seed 모두 나쁨). 채택하지 않음. 남은 의문: verify+refine 출력 기준으로 speed·각도 중 무엇이 남았는가(아래 우선순위 1번) | oracle 교체의 speed +8.0%p는 select 기준이라 파이프라인의 여지가 아니었음(정정) | – | – |
| 22 | ~~재시도 시 탐색~~ **완료·기각(2026-10-12)**: miss 직후 배치가 그대로인 비율 22~24%(판단표 B), 이득 상한 +0.4%p. 원래 제안: 직전 샷이 miss였고 state가 거의 같으면 각도에 노이즈를 주는 롤아웃 | miss 직후 성공률 70.4% vs 정상 82.2%. 결정론적 정책이 같은 action을 반복해 또 실패하는 것일 수 있음(미검증). 샘플링이 이득이 되는 경우 | 학습 없음, 롤아웃 | – |
| 20 | **Laplace NLL(고정 scale)을 robust loss로**: 제곱 오차 대체 | train loss의 66%가 label 없는 1.4% row에서 나왔고 loss가 소수 row에 지배됨. 샘플별 scale 학습은 어려운 2개 state를 덜 보게 할 위험이 있어 고정 | BC | – |

### TODO 19 설계 메모: search → distill의 loss 비교 (2026-10-12 초안, 실행 전에 확정)

**틀**: 샷 하나 = reset이 되는 contextual bandit. 정책이 각도(와 speed)를 흔들어 샘플 N개를 만들고, 시뮬레이터로 쳐서 성공/실패를 얻고, 그 결과로 정책을 다시 학습한다. 추론은 정책 단독. SAC로 말하면 actor loss의 Q 자리에 critic 대신 시뮬레이터의 1-step return이 들어가고, entropy ↔ 분포 폭 b. RL 용어 대응은 experiments.md "head에서 샘플링해서 치면?".

**세 loss를 같은 state·같은 샘플로 비교** (실패 샘플을 어떻게 쓰는가)

| | 방법 | 실패 샘플의 역할 | 위험 |
|---|---|---|---|
| A | **positives-only (filtered BC)**: 성공 샘플을 label로 supervised | 쓰지 않음 | 상태당 샘플이 전부 성공한 state(b = 0.10°에서 52.8%)는 정보 없이 "샘플 평균 쪽" 노이즈 업데이트. 전부 실패(7.4%)는 신호 없음 |
| B | **per-state baseline REINFORCE**: advantage = R − (그 state 샘플들의 평균 성공률), 실패는 반대 방향으로 밀기, clip | 성공과 대비되는 음의 advantage | 한계 없는 밀기(평균이 멀리 가고 σ 증가) → advantage·ratio clip. 실패 원인이 각도가 아닐 수 있음(credit assignment) |
| C | **경계 중앙 label**: 각도를 정렬해 `실패…성공…실패`에서 양쪽 첫 실패의 중점을 label로 | 창의 경계를 알려줌(저분산 supervised target) | 샘플이 창 양쪽 바깥에 있어야 함(좁은 노이즈로는 혼합 state 40%만). 한쪽만 있을 때의 처리는 미정 |

- 기댓값 gradient는 A와 B가 같고(baseline이 state에만 의존) B는 분산만 줄인다. 새 정보는 혼합 state(약 40%)의 경계에서 온다.
- 1단계는 **각도만**(label pair, speed = proposal)으로 한다: 실패 원인이 각도로 한정돼 credit assignment가 깨끗하다. speed·pair는 그다음. 실패 원인이 speed인 state(각도 노이즈로 닿지 못하는 ~7%)는 각도 샘플링만으로 신호가 없다.
- 샘플링 예산은 세 방법이 같아야 한다(같은 state, 같은 샘플). 시작은 Laplace(b = 0.10°) × N = 6~8, 학습 state는 teacher train state(이전 run과 같은 코드·같은 split). 시뮬레이터 약 15k state × 8 ≈ 120k 샷, 약 5분(추정).

**주의 (앞선 설명 정정)**
- "샘플 6개로 81.8% → 92.6%"는 **추론 때 시뮬레이터가 고를 때의 상한**이지 distill 후 정책 단독 성능이 아니다.
- teacher state에서는 teacher label이 이미 거의 창 중앙이다(기하 조준각 ±0.05°, teacher 성공 99.8%). search로 얻는 점 label은 teacher보다 정밀하지 않다. 기대하는 이득은 (i) 정책이 실제로 가는 state(게임 vs val 격차, #16)와 (ii) 창 안이면 gradient가 0인 tail 집중 loss이다. 따라서 teacher state에서만 비교하면 이득이 안 보일 수 있고, 그 경우 판단 D가 아니라 "state 분포를 정책이 가는 곳으로 옮긴 뒤 재시도"가 다음 행동이다.

**비교 기준과 판단표 (초안)**: 정책 단독 500 games shots/game, seed 3개. control = 같은 길이·같은 구조의 현재 최선(깊이 4, 500 epoch 3.55 ± 0.10, 학습 길이 실험 결과로 바뀔 수 있음). "효과 인정" = seed 3개 모두 개선 + 평균 차이 ≥ 0.15 shots.

| # | 관측 | 해석 | 다음 행동 |
|---|---|---|---|
| A | A·B·C 모두 control 안(차이 < 0.15) | teacher state에서 distill은 이득 없음 | 정책이 가는 state(DAgger식)로 분포를 옮겨 재시도, 또는 열린 결정 1(기하 입력) |
| B | A만 개선, B·C는 A와 같음 | positives로 충분 | negative는 쓰지 않는다 |
| C | B가 A보다 좋음(효과 인정) | 분산 감소 또는 경계 정보가 도움 | B 유지, speed·pair로 확장 |
| D | C가 A·B보다 좋음(효과 인정) | 경계 중앙 label이 최선 | C로 확장, 양쪽 경계가 없는 state의 처리 설계 |
| E | B가 A보다 나쁘거나 불안정(σ 증가, loss spike) | negative의 위험(한계 없는 밀기·credit assignment) | clip 강화 또는 A/C |
| F | seed 하나만 반대 | 불확실 | seed 5개 |

### TODO (우선순위순)

| # | 항목 | 이유 | 비용 | 의존 |
|---|---|---|---|---|
| 1 | **Seed 반복** (pair supervision 3개 label × seed ≥3) | **seed 버그 수정 후** robust label은 3 seeds 완료: shots 4.51±0.23, success 66±3%. margin, geom32는 아직 1회 → seed std ~0.2-0.3 shots 때문에 label 간 4.34~4.41 차이는 판단 불가 | BC 5분/run | – |
| 1b | **lr/clip 정리**: wrap 없음 + lr 3e-4, wrap + grad clip 등으로 baseline 설정 확정 | wrap 있음에서 lr 1e-3→3e-4가 5.59→4.94 (lr 민감). 1e-4는 6.30. wrap 없음은 lr 1e-3만 해 봄 | 6 runs ~10분 | – |
| 2+3+4 | **한 번에 진행 중 (2026-10-09)** — 설계는 experiments.md "Pair-level outcome 통계로 selection 개선" 참고. 같은 데이터 하나로 묶임: 모든 pair(≤18)의 coarse candidate 20개를 simulation하고 event log로 pair별 {n, intended pair made(no scratch), any clean pocket, scratch} 저장(`exp16_wm/pair_stats.py`) | – | ~15분 + BC 3 seeds | – |
| 2 | ↳ **factorized selection**: ball head + per-ball pocket softmax, target ∝ pair별 made hits (feasible한데 unlabeled인 pair가 false negative였던 문제) | label은 top-2 pair만 → 이 문제 | | |
| 3 | ↳ **Outcome-grounded pair**: angle matching 대신 `good_made`(event log)로 regression/selection target (한 action이 여러 pair를 made하면 각 pair head가 그 action을 target으로) | geom label의 26%가 intended pair와 다른 outcome | | |
| 4 | ↳ **Scratch**: "made" 정의에 no-scratch 포함 + pair별 scratch head(inference에 penalty 옵션) | chosen pair를 넣고도 scratch 8-11% | | |
| 5 | **Made head** P(made_ij \| e_ij, a_ij) → pair 선택 보정(verifier) | selection이 "이 action이 성공할까"를 직접 보지 않음 | 중 | 3 |
| 6 | **Dense success map** (sims 500만-2,100만 건) + test-time refinement ∇_a P(made) | miss 24%(0.1° 정밀도) 직접 겨냥. 단 sims에 (ball, pocket) identity 없음 → 기록 추가 후 재simulation 또는 angle matching(purity 72-99%) | 높음 | 5 |
| 7 | **pair supervision 조건으로 이전 비교 재실행**: MLP vs GNN, augmentation(flip4는 pocket 순열 필요), data 8×, model size | 이전 모든 GNN 수치는 WTA 학습 | BC 여러 번 | 1 |
| 8 | ~~robust+pairs의 val loss 이상~~ **완료 (2026-10-09)**: 원인 = ±180° seam에서 각도 loss 폭발(label의 1.3%가 backward shot, val 5개 state가 loss의 99%). pair 매핑 버그 아님. wrap-aware로 고치면 val loss는 정상이 되나 **policy는 악화**(4.51→5.59 shots, lr 3e-4에서도 4.94) → 기본값은 raw loss 유지(`--angle-wrap` opt-in). 이유 미규명 | – | – |
| 9 | **Checkpoint 선택 잡음**: val within-0.2° 비율이 epoch마다 ±8%p 흔들림 → 최댓값 선택은 낙관적 | EMA weights / 더 큰 val / 마지막 epoch 평균 | 낮음 | – |
| 10 | **DAgger**: policy가 가는 state에서 teacher에게 다시 label | train/val은 teacher states만. policy rollout은 다른 분포 | 중 | 1 |
| 11 | **Multi-ball / position play**: robust teacher는 1.0m/s 1-ball shot을 골라 2.98 shots (geom 2.24) | "창이 충분히 넓은 shot 중 more balls" 기준 | 중 | – |
| 12 | **SAC fine-tune from BC init** (엔트로피 낮춤) | BC 4.3 shots → teacher 2.2~3.0 격차, position play는 RL이 필요할 수 있음 | 중 | 1, 8 |

실행 순서 제안: 1(seed) → 8(val loss 이상) → 2+3+4(label/selection 개선, 한 번에) → 5 → 7 → (6, 10, 11, 12는 결과 보고 결정).

### 보류/기각
- 행동 재파라미터화(ghost-ball 보정각): bitter lesson, 아래 보류 섹션.
- lr decay 단독: 전체 데이터에서 도움 없음(robust 6.18 → 12.25).
- **angle wrap-aware loss** (2026-10-09): val loss 이상(±180° seam spike)의 원인이었고 함수는 수학적으로 검증됐지만(주기, 범위, sin/cos 일치), 3 seeds 모두에서 policy가 나빠짐(4.51→5.59 shots; lr 3e-4에서도 4.94). 검증으로 기각된 가설: Adam 브레이크(step당 update 크기 0.0499 vs 0.0525로 동일), tanh seam trap(갇힌 출력 0개, non-seam label의 회귀 오차 동일). 원인 미규명 → 기본값은 raw loss 유지, `--angle-wrap` opt-in. 규명하려면 epoch별 학습 곡선이 갈라지는 시점 추적(미실행).
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
