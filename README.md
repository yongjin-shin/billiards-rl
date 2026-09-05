# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).
진행 상황과 관찰을 기록하는 실험 노트.

---

## Quick Status

| | |
|---|---|
| **현재 위치** | WMPredictor v3 아키텍처 완성 (Δpos·2-layer LSTM·OneCycleLR), 훈련 미완료 — 다음 실험 방향 탐색 중 |
| **다음 실험** | TBD |
| **Exp-16 vanilla** | Phase 1: pocket 62.2% ≈ SAC 63.6% (p=0.49) / Phase 0: pocket 56.9% ≈ SAC 55.3% (p=0.62) |

---

## Environments

두 환경 모두 동일한 2-dim 연속 액션:

| Dim | Range | 의미 |
|-----|-------|------|
| `delta_angle` | [−π, π] | nearest unpocketed ball 방향 기준 offset (0 = 직접 겨냥) |
| `speed` | [0.5, 8.0] | 큐볼 타격 속도 (m/s) |

### Phase 0 — single-ball (`n_balls=1`)

| | |
|---|---|
| **Observation** | 16-dim: `[cue_xy, ball_xy, p0~p5_xy]` |
| **Reward** | +1 pocketed, 0 otherwise |
| **Episode** | 항상 1 step 후 종료 |
| **Ball placement** | cue y∈[0.15,0.40] / ball y∈[0.30,0.85] (current)<br>cue y∈[0.20,0.40] / ball y∈[0.60,0.90] (legacy — 원본 Exp-01 조건) |

### Phase 1 — multi-ball (`n_balls=3`)

| | |
|---|---|
| **Observation** | 23-dim: `[cue_xy, b1_xyz, b2_xyz, b3_xyz, p0~p5_xy]` (flag = pocketed 여부) |
| **Reward** | +1.0 per ball pocketed · −step_penalty×i (progressive) / flat · −0.5 scratch · −trunc_penalty if truncated |
| **Episode** | 공 3개 모두 pocketed OR step ≥ max_steps |
| **Ball-in-hand** | scratch 시 큐볼 재배치 |

---

## Results

### Phase 0 · Exp-01 algorithm benchmark

SAC / PPO / TQC, 1M steps, seeds {0, 1, 2}, legacy 배치 조건.

| Algo | s0 / s1 / s2 | avg | std |
|------|-------------|-----|-----|
| **SAC** | 81.4 / 73.6 / 77.8 | **77.6%** | ±3.9pp |
| TQC | 84.6 / 80.6 / 35.6 | 66.9% | ±27pp |
| PPO | 24.0 / 30.0 / 31.6 | 28.5% | ±4.0pp |
| Random | — | ~6% | — |

> **재현 시 주의:** 현재 코드로 재훈련 시 ~50% (legacy 배치, no-scratch) 또는 ~42% (current 배치).
> 원인 및 ablation 결과는 [Exp-01 log](#exp-01--phase-0-single-ball-benchmark) 참고.

---

### Phase 1 · ms=5 시리즈 (Exp-02~09)

#### Exp-02~06: baseline → reward shaping

| Exp | 조건 | Pocket% | Clear% | Ep Len | 비고 |
|-----|------|---------|--------|--------|------|
| 02 | ms=∞ (ms=15) | 98.3% | 95.8% | — | 너무 느슨, random도 40% |
| 03 | ms=5 scratch | 60.7% | 29.4% | 4.60 | Phase 1 baseline |
| 04 | Transfer A zero-shot | 63.6% | 31.4% | — | 추가 훈련 없이 baseline 초과 |
| 05 | Transfer B warm-start | 61.5% | 30.4% | — | zero-shot보다 낮음 |
| **06** | **pp=✓ scratch** | **63.9%** | **33.2%** | **4.48** | **ms=5 best** |

> Exp-06 재실험: seed=0 완료 63.1%/31.2% (원래 63.9%/33.2%). seed=1/2 미완.

#### Exp-07~08: ep_len 단축 시도 → 전부 실패

| Exp | 변경 조건 | Pocket% | Clear% | Ep Len |
|-----|----------|---------|--------|--------|
| 06 | pp=✓ (기준) | 63.9% | 33.2% | 4.48 |
| 07 SAC | cb=2.0 | 62.0% | 29.2% | 4.50 |
| 07 TQC | cb=2.0 | 49.7% | 17.6% | 4.80 |
| 08a | shots_taken + cb=2.0 | 63.7% | 30.0% | 4.60 |
| 08b | shots_taken + lr=1e-4 + gs=10 | 62.1% | 28.8% | 4.50 |

> **결론:** ep_len은 reward shaping / obs로 단축 불가. task 구조 자체가 매 step 기댓값 > 0.

#### Exp-09: ms × pp ablation grid

SAC, 1M, seed=42, sp=0.1, tp=1.0.

| ms | pp=✗ (pocket/clear) | pp=✓ (pocket/clear) | Δ clear | ep_len/ms |
|----|---------------------|---------------------|---------|-----------|
| 5 | 63.6% / 32.2% | 63.9% / 33.2% | +1.0pp | 88% |
| 4 | 55.1% / 17.6% | 51.2% / 15.8% | −1.8pp | 97% |
| 3 | 41.4% / 9.0% | 41.5% / 7.6% | −1.4pp | **99%** |

> **결론:** pp 폐기. ms=3이 Phase 2 frontier — reward shaping이 아닌 알고리즘 수준 개선 필요.

---

### Phase 2 · ms=3 frontier (Exp-10~12)

#### Exp-10: algorithm benchmark

SAC/TQC/PPO × 3 seeds, ms=3, sp=0.1, tp=1.0, 2M steps.

| Algo | avg Pocket% | avg Clear% | 비고 |
|------|------------|------------|------|
| **SAC** | **41.7%** | **8.4%** | 안정적 (s0/s1 분산 작음) |
| TQC | 27.1% | 2.0% | overconservatism |
| PPO | ~6.5% | ~0.0% | credit assignment 한계 |
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

> curriculum 2M이 scratch 2M 대비 +1.3pp pocket / +2.0pp clear. ms=2 extension은 학습 신호 부재로 실패.

#### Exp-12: abs_angle ❌ 폐기

delta_angle → absolute angle [0, 2π] 교체 실험.

| Seed | Steps | Pocket% | Clear% |
|------|-------|---------|--------|
| 1 | 5M | 37.8% | 6.4% |
| Exp-10 SAC | 2M | **41.7%** | **8.4%** |

> 5M을 써도 delta 2M보다 낮음. inductive bias(delta=0 → 공 직접 겨냥) 부재가 치명적. **delta_angle 유지.**

---

## Exp-13 결과 · Phase 0 Reward Shaping & Steps Scaling

### Proximity Reward ❌

| α | Pocket% | vs baseline |
|---|---------|-------------|
| 0.0 (baseline, 1M) | **50.0%** | — |
| 0.05 | 40.8% | −9.2pp |
| 0.1 | 37.8% | −12.2pp |
| 0.3 | 41.4% | −8.6pp |
| 0.5 | 42.0% | −8.0pp |

모든 α에서 baseline 하회. **post-shot 최종 거리는 유효한 gradient signal이 아니다.**
쿠션에 1~2번만 맞으면 볼의 최종 위치와 초기 action 사이의 인과관계가 끊김 — `f(action) → chaos`.

### Steps Scaling ✅ · gradient_steps 한계 확인

| Steps | gs | Pocket% | 학습시간 | 효율 |
|-------|----|---------|---------|------|
| 1M | 1 | 50.0% | ~20분 | — |
| 2M | 1 | 56.2% | ~35분 | 6.2pp/1M |
| 5M | 1 | 65.8% | 91분 | 3.2pp/1M |
| 5M | 4 | **68.6%** | 187분 | — |

steps 늘릴수록 일관된 성능 향상. 단, **diminishing returns 확연** — 효율 절반으로 하락.

`gradient_steps=4`는 +2.8pp에 그치며 학습시간 2배 소요 → ROI 낮음.

**결론: flat policy의 ceiling 확인.** `Q(o,a) ≈ P(pocket | positions, angle)`을 binary feedback만으로 근사하는 구조적 한계. physics dynamics를 모르는 상태에서 sample을 아무리 늘려도 벽에 부딪힘. → **World Model (Exp-15)로 방향 전환.**

---

## Roadmap

```
[x] Exp-13   Phase 0: proximity reward 실패 + steps scaling 확인 (1M:50% / 2M:56% / 5M:65.8%)
[x] Exp-14   gradient_steps=4, 5M → 68.6% (+2.8pp, 시간 2배) — flat policy ceiling 확인
[x] Exp-15   Trajectory VAE 탐색 — physics latent 구조 파악, M decoder 구조 확정
             (VAE 자체는 최종 구조에서 폐기, decoder 아키텍처·하이퍼파라미터 탐색 목적)

[~] Exp-16   World Model Critic — Q(s,a) = q(M(s,a))
             vanilla 구현 검증 완료 (≈ SB3 SAC), WM variant 실험 중
[ ] Exp-17   Phase 1 HRL — System 2 (ball 선택 discrete 3) + System 1 (Phase 1 Exp-10 freeze)

[ ] cushion / bank shots
[ ] self-play / full 8-ball
```

---

## Exp-15 · Trajectory VAE (World Model 기초)

### 근본 문제 (flat policy 한계)

```
Q(o, a) ≈ P(pocketed | cue_pos, ball_pos, angle, speed)
```

이 함수는 billiards physics를 완전히 내포해야 하는데:
- **불연속**: 0.1° 차이로 in/out
- **chaotic**: 다중 쿠션 후 초기 방향과 인과관계 단절
- binary reward만으론 막대한 샘플 필요 → 5M steps에서도 68.6% ceiling

proximity reward 실패 원인도 동일 — post-shot 최종 위치는 physics causal chain이 끊긴 이후의 노이즈.

### 수식

```
f(O_t | a_t) = (O_{t+1}, ..., O_{t+n})    # dynamics: 궤적
g(O_{t+1}, ..., O_{t+n}) = h_t             # encoder → latent z
p(O_t, h_t) = a_t                          # policy
```

순환 의존성 해결 → **Recurrent** (h from past) 또는 **MPC/imagination** (TD-MPC / DreamerV3 방향).

### Exp-15 구현: Trajectory VAE

**데이터**: pooltool `system.events`에서 `stick_ball` 부터 전체 궤적 추출
(cue 접근 경로 + ball_ball 충돌 + target ball 이후 경로 모두 포함)
```
event = (x, y, type_one_hot_10)  → 12-dim
sequence: variable length, max 32 events
```

**이벤트 타입 (10종)**:
`none` / `stick_ball` / `ball_ball` / `ball_linear_cushion` /
`ball_circular_cushion` / `ball_pocket` / `sliding_rolling` /
`rolling_spinning` / `rolling_stationary` / `spinning_stationary`

**모델**:
```
Encoder: LSTM(12 → 64) → h_final → μ, log σ² → z ∈ R^z_dim
Decoder: MLP(z_dim → 128 → 12 × 32)
Loss:    MSE(pos) + CE(event_type) + β·KL
```

**분석**:
- t-SNE: pocketed / n_bounces / tag(SAC vs random) 별 latent 구조
- action 상관관계: z_dim vs delta_angle / speed
- latent traversal: 각 dim을 ±3σ로 변화시켜 궤적 디코딩

### 파일 구조

```
world_model/
├── data/              # 생성된 trajectory 데이터셋 (.npz + metadata.json)
├── checkpoints/       # 학습된 VAE 체크포인트
├── results/           # t-SNE, correlation, traversal 이미지
├── generate_data.py   # SAC/random 모델로 데이터 수집
├── model.py           # LSTM encoder + VAE
├── train_vae.py       # VAE 학습
├── visualize.py       # t-SNE + latent traversal
└── analyze.py         # linear probe + correlation 분석
```

### 실행 순서

```bash
# 1. 데이터 생성
python world_model/generate_data.py --tag sac_5m_gs4 \
    --model logs/experiments/SAC_5000k_s42_sp0.0_tp0.0_gs4_20260322_150734/best_model/best_model \
    --n-episodes 5000
python world_model/generate_data.py --tag random --n-episodes 5000

# 2. VAE 학습 (z_dim 비교)
python world_model/train_vae.py --z-dim 8
python world_model/train_vae.py --z-dim 16
python world_model/train_vae.py --z-dim 32

# 3. 분석
python world_model/visualize.py --ckpt world_model/checkpoints/vae_z16_*.pt
python world_model/analyze.py   --ckpt world_model/checkpoints/vae_z16_*.pt
```

---

## Exp-16 · World Model Critic

### 근본 문제 (flat policy 한계 복기)

```
Q(s, a) ≈ E[r | s, a]
```

Q가 physics를 implicit하게 배워야 하는데:
- **(s, a) → scalar** 로 압축하는 순간 physics causal chain 소실
- `∂Q/∂a` 가 noisy — "이 angle이 통계적으로 나빴다"만 알고 "왜 나빴는지" 모름
- sparse binary reward에서 역추론 → 5M steps 68.6% ceiling

### 아키텍처

```
Actor:   π_θ(s) → a                     표준 SAC, 무수정
Critic:  Q(s, a) = q( M(s, a) )         M이 critic 안에서만 존재
```

Critic 내부:

```
s, a ──→ M ──→ ĥ ──→ q ──→ Q
          ↑           ↑
   L_WM (dense)   L_Bellman (sparse)
   MSE(ĥ, h_real)  Bellman backup
```

- `M: (s, a) → ĥ` — 단순 MLP. trajectory 전체를 예측. dense supervision으로 physics 명시적 학습
- `q: ĥ → Q` — value head. Bellman으로 학습. physics는 M이 처리했으니 q의 문제가 단순해짐

### 왜 기존 Q(s,a)보다 나은가

```
기존: ∂Q/∂a            — physics를 scalar gradient 하나로 압축
새것: ∂q/∂ĥ · ∂ĥ/∂a  — ∂ĥ/∂a 가 physics Jacobian, M이 dense 학습으로 정확
```

Actor update 시 gradient path:
```
actor_loss = -Q + entropy = -q(M(s, π(s))) + entropy
∂/∂θ: ∂q/∂ĥ · ∂ĥ/∂a · ∂a/∂θ   ← physics gradient 도달, shared weight 없음
```

### 학습

```python
# Critic update (M + q 동시)
h_hat    = M(s, a)
L_WM      = MSE(h_hat, h_real)           # dense, trajectory 전 step
L_Bellman = MSE(q(h_hat), target_Q)      # sparse, reward
L_critic  = L_Bellman + λ · L_WM
L_critic.backward()
critic_optimizer.step()

# Actor update (표준 SAC)
a_pi = π_θ(s)
actor_loss = -q(M(s, a_pi)) + α·log π_θ(a_pi)
actor_loss.backward()   # 새 forward → 새 graph, double-backward 없음
actor_optimizer.step()
```

### SB3 수정 범위

```
Actor                → 무수정
TrajectoryBuffer     → h_real 저장 추가          (~60줄)
WorldModelCritic     → ContinuousCritic 상속      (~80줄)
WorldModelSAC        → train() 오버라이드         (~30줄)
simulator.py         → info에 trajectory 포함     (~20줄)
─────────────────────────────────────────────────────────
총                   ~190줄, SB3 core 무손상
```

### 결과 · Vanilla SAC 구현 검증 (2026-03-28)

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

**유의성 검정 (paired t-test):**
- pocket: diff=−1.4pp, t=−0.77, **p=0.49** → not significant
- clear:  diff=−2.7pp, t=−1.21, **p=0.29** → not significant

**결론: VanillaSAC ≈ SB3 SAC. 구현 검증 완료.**

### 결과 · Vanilla SAC 구현 검증 — Phase 0 (2026-03-29)

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
> best checkpoint (eval 중 peak): vanilla 74.0% vs SAC 72.4%
> SAC s3 — best 66%에서 학습 말기 8%로 collapse 관찰

**유의성 검정 (paired t-test):**
- best pocket: diff=+1.6pp, t=0.63, **p=0.57** → not significant
- final pocket: diff=+1.6pp, t=0.54, **p=0.62** → not significant

**결론: Phase 0에서도 VanillaSAC ≈ SB3 SAC. 구현 검증 완료.**

---

## WMPredictor 실험 · Exp-16 World Model 구성요소 개발

Exp-16 WM critic의 `M(s, a) → ĥ` 컴포넌트를 위한 독립 trajectory predictor 개발 트랙.
SAC/random 데이터로 사전훈련 후 Exp-16 critic에 통합 예정이었으나, 훈련 결과 확인 전에 방향 전환.

### WMPredictor v1 (2026-03-28)

`world_model/predictor.py` + `train_predictor.py`

**비교축:** 아키텍처(MLP vs LSTM) × 훈련전략(curriculum vs tf_ratio) × scale-up(h=256/512/1024)

- LSTM curriculum: TF ratio 1→0 점진 감소 (80 epoch에 걸쳐)
- LSTM tf_ratio: scheduled sampling 방식
- scale-up: h=256 → 512 → 1024 (scaleup 실험)

### WMPredictor v2 (2026-03-29 ~ 2026-04-03)

`world_model/wm_predictor.py` + `train_wm_predictor.py` (신규 포맷)

**변경:** cue / target ball 궤적 분리 데이터 포맷(data_v2/) + dual-head LSTM

```
Encoder : MLP (obs_norm, act) → h0, c0
Decoder : LSTM step input [e_emb | cue_xy | tgt_xy] (d+4)
          ├── event_head : hidden → logits (K=10)
          └── pos_head   : [hidden | event_embed] → abs cue_xy, tgt_xy
```

**실험 범위:** lstm_hidden {256, 512} × lstm_layers {1, 2} × aug {on, off} × seeds {0,1,2,42}
**마지막 체크포인트:** `wmv2_enc128_256_h256_l2_emb32_s*_20260403_*`

### WMPredictor v3 (2026-09-05, 아키텍처만 완성 — 훈련 미완료)

**동기:** v2에서 abs좌표 직접 예측 시 오차 누적 + pos/event head 간 gradient 간섭 문제 관찰.

**변경 사항:**

| | v2 | v3 |
|---|---|---|
| pos 예측 | 절대좌표 직접 | **Δpos** 예측 후 누적 |
| decoder input | `[e_emb\|cue\|tgt]` d+4 | `[e_emb\|cue\|tgt\|Δcue\|Δtgt]` d+8 |
| LSTM | 1-layer | **2-layer + dropout=0.1** |
| head 구조 | pos_head = hidden+event_embed | **pos_head / event_head 완전 독립** |
| LR 스케줄 | ReduceLROnPlateau | **OneCycleLR** (batch 단위) |
| event loss | CE | **CE + label_smoothing=0.1** |
| enc_hidden | [128,128] | **[128,256]** |

`run_wmv3.sh`: h=256 × lr {3e-4, 1e-3} × seeds {0,1,2} 총 6 run 계획.
**2026-09-05 훈련 시작 직후 종료. 결과 없음. 이후 실험 방향 전환.**

---

## Markov World Model · 이벤트 기반 (2026-09-05)

LSTM 기반 trajectory 예측에서 **순수 이벤트(충돌) 기반 Markov 모델**로 방향 전환.

### 모델 구조

```
MarkovEncoder:    (obs(16) + act(2)) → first_event (type + state)
MarkovTransition: event_state(24)    → (next_type, next_cue_state, next_tgt_state)
```

**Event state dim 24:**
- cue_xy(2) + cue_vel(2) + cue_avel(3) — 충돌 직후(post-collision) 속도
- tgt_xy(2) + tgt_vel(2) + tgt_avel(3)
- type_onehot(10): ball_ball / linear_cushion / circular_cushion / ball_pocket + 6 non-coll

**MarkovTransition 입력 확장 (최종 in_dim=58):**
- type_embed(32) + phys(14) + pocket_dists(12) — 6포켓 × 2공

### 데이터 (data_v4)

- 55k 에피소드: SAC 3종 × 10k + random 25k
- **post-collision velocity** (`agent.final.vel`) 사용 — 다음 이벤트 예측에 더 직접적
- 정규화: pos/TABLE_WH, vel/12.0 m/s, avel/300 rad/s

### 커리큘럼 5-stage

| Stage | 데이터 필터 | max_per_class | trans_ep | enc_ep |
|-------|-----------|--------------|---------|-------|
| 0 | first_ball_ball (SAC) | 10,000 | 200 | 100 |
| 1 | any_ball_ball | 20,000 | 75 | 40 |
| 2 | any_ball_ball + 신규 15k | 40,000 | 50 | 25 |
| 3 | all | 80,000 | 40 | 20 |
| 4 | all (자연분포) | None | 30 | 15 |

### 최종 결과 (Stage별 per-class accuracy)

| Stage | ball_ball | linear_cushion | circular_cushion | ball_pocket |
|-------|-----------|---------------|-----------------|------------|
| 0 | 58.8% | 50.0% | 33.6% | 60.3% |
| 1 | 64.4% | 47.8% | 46.5% | 64.2% |
| 2 | 83.6% | 42.6% | 45.4% | 64.5% |
| 3 | 79.9% | 48.3% | 49.2% | 68.8% |
| **4** | **91.2%** | **39.1%** | **53.8%** | **70.3%** |

### 한계

1. **linear_cushion 39%**: 자연분포에서 78% 차지, Stage 4에서 ball_ball과 혼동
2. **두 공 독립성 없음**: ball_ball 이후 cue/tgt가 독립적으로 움직이는데 단일 체인으로 예측
3. **인과 없는 쌍**: ④ 타겟 포켓 → ⑤ 큐볼 쿠션처럼 관계없는 이벤트 쌍을 학습
4. **gradient 불연속**: 다음 이벤트 타입이 discrete argmax → planning에 부적합

→ **Fixed-Δt 연속 상태 모델로 방향 전환**

---

## Fixed-Δt World Model (2026-09-06)

이벤트 기반의 구조적 문제를 해결하기 위해 **고정 시간 간격(Δt=0.05s) 연속 상태 예측** 방식 채택.

### 설계 동기

| 문제 (이벤트 기반) | 해결 (Fixed-Δt) |
|-----------------|----------------|
| discrete type argmax → gradient 끊김 | 연속 상태 → fully differentiable |
| ball_ball 이후 두 공 혼재 | joint state (cue+tgt) 매 스텝 |
| 인과 없는 이벤트 쌍 | 시간 순서 그대로, 연속 |

### 모델 구조

```
StateEncoder φ  : s(14) + pocket_dist(12) → z(128)   ← SAC Critic 공유 예정
Transition f    : z(128) → z'(128)
StateHead       : z'(128) → ŝ(14)
CollisionHead   : z(128) → (p_coll, type_logit[4])
```

**State s (14dim, normalized):**
- cue_x, cue_y, cue_vx, cue_vy, cue_wx, cue_wy, cue_wz (7)
- tgt_x, tgt_y, tgt_vx, tgt_vy, tgt_wx, tgt_wy, tgt_wz (7)

**포켓 거리 (12dim):** 6포켓 × 2공, Markov 모델과 동일 방식 이식.

### 데이터 (data_fixeddt)

- 45k 에피소드: SAC 3종 × 10k + random 15k
- Δt=0.05s → 평균 112 스텝/shot, 최대 220 스텝
- 480만 (s_t, s_{t+1}) 쌍, 충돌 비율 5.7%
- `pt.interpolate_ball_states()` 로 임의 시점 상태 쿼리

### 학습 개선 이식 (Markov 모델에서)

- **포켓 거리 12dim**: StateEncoder에 내장
- **LR/TB 데이터 증강**: 좌우/상하 반전 50% 확률
- **클래스 가중치**: ball_ball×0.60 / linear×0.075 / circular×0.87 / pocket×2.46

### 결과

| 항목 | 값 |
|-----|---|
| 충돌 감지 정확도 | 83.1% |
| 타입 분류 (전체) | 88.7% |
| ball_ball | **99.3%** |
| linear_cushion | **87.4%** |
| circular_cushion | **85.9%** |
| ball_pocket | **91.8%** |

이벤트 기반 대비 linear_cushion +48.3pp, circular +32.1pp.

### 한계: AR 구조 → rollout 오차 누적

```
step t: 오차 ε_t
step t+1: ε_t가 입력 → ε_{t+1} > ε_t
충돌 순간: 속도 급변 → 방향 조금만 틀려도 이후 궤적 발산
```

6초 full shot rollout 시 위치 오차 최대 140cm (테이블 99×198cm).
단일 스텝 정확도는 높으나, 연속 rollout에서 오차 누적.

→ **Deterministic SSM으로 방향 전환 (feature/wm-ssm)**

---

## Next: Deterministic SSM (feature/wm-ssm, 예정)

### 문제: 현재는 AR (Autoregressive)

```
현재: s_t →enc→ z_t →trans→ z_{t+1} →dec→ ŝ_{t+1} →enc→ z_{t+1}' → ...
                                                              ↑
                                                        매 스텝 re-encode (AR)
```

rollout 때 ŝ를 observation space로 decode 후 다시 encode → 오차 누적.

### SSM 구조식

$$z_{t+1} = f_\theta(z_t)$$
$$s_t = g_\phi(z_t)$$
$$z_0 = \text{enc}_\psi(s_0)$$

### closed-loop rollout 학습

```
s_0 →enc→ z_0 →f→ z_1 →f→ z_2 → ... →f→ z_T   (z space에서만)
               ↓g     ↓g              ↓g
               ŝ_1    ŝ_2             ŝ_T

Loss = Σ ||ŝ_t - s_t||²   (gradient가 z_0까지 역전파)
```

### Critic 공유

```
Q(s, a) = Q_head( ψ(s), a )   ← Encoder ψ 공유
```

### 핵심 변경

| | Fixed-Δt (AR) | SSM |
|--|--------------|-----|
| rollout | obs space 왔다갔다 | z space에서만 |
| Transition 학습 | single-step loss | multi-step rollout loss |
| Encoder | obs→z, 매 스텝 | 초기 z_0 한 번만 |
| gradient | 1 스텝 | T 스텝 역전파 |

Transition에 skip connection (z_{t+1} = z_t + f(z_t)) 추가 — T step gradient vanishing 방지.

---

## Future: Exp-17 · Phase 0 HRL

### 설계 근거

Phase 0 flat policy의 문제: delta_angle 기준이 nearest ball 방향이라 **어느 포켓을 노릴지 implicit하게만 결정**됨.
커리큘럼 등 학습 과정에서 nearest pocket bias가 생길 수 있음.

```
System 2 (새로 학습): 포켓 선택 (discrete 6)
                       ↓
obs 재배열: target_pocket_xy → obs 앞에 주입
                       ↓
System 1 (pocket-conditioned, freeze 후):
  "이 포켓으로 공을 넣기 위한 각도"를 학습
```

**Credit assignment 해결 방법:**
- System 1을 먼저 충분히 학습 후 freeze
- 이후 System 2 학습 시 실패 원인 = System 2의 포켓 선택 탓으로 귀결
- System 2는 통계적으로 수렴 (에피소드 수천 개 기준)

**구현:** DQN (System 2, discrete 6) + SAC (System 1, continuous) + custom env wrapper

---

### Obs 설계

**현재 Phase 0 obs (16-dim):**
```
[cue_x, cue_y,        # 2
 ball_x, ball_y,      # 2
 p0_x, p0_y,          # 12 (6 pockets, 순서 고정)
 p1_x, p1_y,
 ...
 p5_x, p5_y]
```

**Exp-15 System 1 obs (16-dim, 동일 크기):**
```
[cue_x, cue_y,              # 2
 ball_x, ball_y,            # 2
 target_x, target_y,        # 2  ← 항상 obs[4:6] = 선택된 포켓
 other_p0_x, other_p0_y,    # 10 (나머지 5 pockets)
 ...
 other_p4_x, other_p4_y]
```

System 2가 포켓 idx를 선택하면 해당 포켓을 obs[4:6]으로 배치, 나머지 5개를 뒤에 채움.
→ obs 크기 유지 (16-dim), System 1은 항상 "obs[4:6]이 목표 포켓"으로 학습.

**System 1 학습 시 reward 변경:**
```
현재: +1 (어느 포켓이든 들어가면)
변경: +1 (target pocket에 들어갔을 때만)
```

**Feasible pocket 선택 (random target pocket 문제 해결):**
```
cut_angle(pocket_i) = arccos(dot(normalize(B-C), normalize(P_i - B)))
feasible = [i for i in range(6) if cut_angle(i) < 60°]
target_pocket = random.choice(feasible)  # fallback: argmin(cut_angle) if empty
```

---

## Future: Exp-17 · Phase 1 HRL

### 설계 근거

현재 flat MLP의 병목은 두 레벨 문제를 혼합:

| | System 1 (aiming) | System 2 (strategy) |
|---|---|---|
| **역할** | 이 각도+속도로 쏘면 저 공이 들어가는가 | 어떤 공을 먼저, 어느 포켓으로 |
| **reward** | 즉각적 (+1 per pocket) | 희박하고 지연됨 (9% clear) |
| **현재 문제** | nearest-ball greedy에 하드코딩됨 | gradient 신호 거의 없음 |

```
System 2 (새로 학습):  target ball 선택 (discrete 3)
                       ↓
obs 재배열:  [cue_xy, target_xyz, other1_xyz, other2_xyz, 6pockets]
             target ball → ball[0] 위치로 이동
                       ↓
System 1 (Phase 1 Exp-10 freeze):
  delta_angle = 0 → ball[0] 방향(= target) 겨냥
```

### Exp-16 variants

| | 16a | 16b | 16c |
|---|-----|-----|-----|
| **System 2 action** | 공 선택 (discrete 3) | 공+포켓 (discrete 18) | 공+포켓 (discrete 18) |
| **System 1** | Phase 1 freeze | Phase 1 freeze | joint 학습 |
| **OOD 리스크** | 낮음 | 중간 | 없음 |
| **핵심 질문** | HRL 구조 자체 유효한가? | 포켓 info가 도움되는가? | pretraining 없이 가능한가? |

**실험 순서:** 16a → 16b → 16c

---

## Experiment Log

상세 관찰 기록. 빠른 확인은 위 Results 테이블 참고.

---

### Exp-01 · Phase 0 single-ball benchmark

**설정:** SAC/PPO/TQC, 1M steps, seeds {0,1,2}, n_balls=1, legacy 배치

**관찰:**
- Horizon=1에서 off-policy(SAC/TQC) 압도적. PPO의 GAE가 단일 transition에서 REINFORCE로 퇴화.
- TQC seed 분산 매우 큼 (±27pp) — 분포 추정 불안정. SAC best_model vs final 격차도 큼 (말기 collapse).

**재현 시 성능 차이 원인 (2026-03 ablation, branch: exp/phase0-placement-ablation):**

| 조건 | Pocket% | Δ |
|------|---------|---|
| Legacy 배치 + no scratch (원본 완전 재현) | **81.4%** | 기준 |
| Current 배치 + no scratch | 50.0% | −31pp |
| Current 배치 + scratch | 42.4% | −39pp |

- **원인 ①** scratch penalty: 원본에 없었음. random action의 ~26%가 scratch → expected reward +0.026 → −0.099로 부호 역전 → SAC가 포켓 대신 scratch 회피 학습. 수정: `simulator.py` `if scratch and n_balls > 1`
- **원인 ②** ball placement 확장 (commit 89431bd): target y [0.6,0.9] → [0.30,0.85]. n_balls=3 통합 시 의도적 변경으로 난이도 상승.

---

### Exp-02 · Phase 1a multi-ball (ms=∞)

**설정:** SAC, 1M, seed=42, n_balls=3, ms=15

pocket 98.3% / clear 95.8% — random도 40%. "많이 쏘면 들어간다" 전략. horizon 축소 필요 → Exp-03.

---

### Exp-03 · Phase 1a (ms=5)

**설정:** SAC, 1M, seed=42, n_balls=3, ms=5

pocket 60.7% / clear 29.4%. 5번 안에 3개 → 샷당 평균 0.6 pocket 필요. Transfer 실험 baseline.

---

### Exp-04 · Transfer A — zero-shot

**방법:** `ObsCollapseWrapper`로 23-dim → 16-dim 축소. 매 step nearest unpocketed ball → "the ball". Phase 0 pretrained model (seed=0, 81.4%) 그대로.

63.6% / 31.4% — 추가 훈련 없이 Exp-03 초과. aiming skill 직접 전이 확인. 단, 다른 공 위치를 모르므로 interference 회피 불가.

---

### Exp-05 · Transfer B — warm-start

**방법:** 23-dim SAC 새로 만들고, shared 뉴런에 n_balls=1 weight 복사. ball2/ball3 뉴런은 0 초기화 후 전체 fine-tune.

61.5% / 30.4%. 초반 reward 높다가 점차 하락 — weight dilution 전형 패턴. **zero-shot(Exp-04)이 훈련 0분으로 더 좋음.** direct transfer가 warm-start fine-tuning 전체를 압도.

---

### Exp-06 · Progressive reward shaping

**설정:** SAC, 1M, seed=42, ms=5, sp=0.1×step, tp=1.0

변경: step penalty 고정(−0.01) → progressive(step i: −0.1×i). truncation penalty 없음 → −1.0. 3-step vs 5-step clear 보상 차이 0.2 → 0.9.

63.9% / 33.2%, ep_len=4.48. ep_len 분포: step5 80.9% — **ep_len 단축 실패**.

근본 원인: step5 기댓값 = −0.5 + 1.0×30% ≈ −0.2. truncation(−1.0)보다 여전히 나으므로 step을 계속 소비.

> 재실험 진행 중 (multi-seed 0/1/2, 1M)

---

### Exp-07 · clear_bonus=2.0 + SAC vs TQC

**설정:** SAC·TQC, 1M, seed=42, ms=5, sp=0.1 flat, tp=1.0, cb=2.0

SAC 62.0% / TQC 49.7%. ep_len 변화 없음. clear_bonus도 ep_len 단축 실패. TQC single-seed 불안정 패턴 재현.

---

### Exp-08 · shots_taken obs ablation

**설정:** SAC, 1M, seed=42, ms=5

shots_taken(step count / max_steps)을 obs에 추가. ep_len 변화 없음 — 당구에서 최적 action은 공 위치에만 의존, step count 무관. shots_taken은 MDP를 완전하게 만들지만 당구에서는 uninformative.

08b: gs=10(UTD=1.0) → SAC에서 Q-value overestimation cascade 심화.

---

### Exp-09 · ms × pp ablation grid

**설정:** SAC, 1M, seed=42, sp=0.1, tp=1.0

pp는 어떤 ms에서도 유의미한 개선 없음:
- ms=4: pp=✓가 −3.9pp pocket — progressive penalty 누적(−1.0)이 pocket reward(+1.0)와 상쇄.
- ms=3: ep_len/ms = 99.3% — task 자체가 모든 step을 소모하는 구조.

**pp 폐기. ms=3이 Phase 2 frontier.**

---

### Exp-10 · Phase 2 ms=3 algorithm benchmark

**설정:** SAC/TQC/PPO × 3 seeds, ms=3, sp=0.1, tp=1.0, 2M

SAC s42: eval crash. PPO s1/s42: 파일 손실 제외.

- **SAC clear 8.4% 재현 확인.** seed 간 분산 작음 — 학습 안정적.
- **TQC 2.0%:** top quantile drop overconservatism이 sparse reward에서 역효과.
- **PPO ~0%:** 3-step credit assignment 극도로 노이즈. **Phase 2 baseline = SAC.**

---

### Exp-11 · Curriculum ms=5→4→3

**설정:** SAC, seed=42, 1M+500k+500k=2M, sp=0.1, tp=1.0

같은 2M에서 scratch 대비 +1.3pp pocket / +2.0pp clear. Stage 3(500k)만으로도 scratch 2M과 경쟁. ms=2 extension: 더블포켓 필수 → 학습 신호 부재, 실패 예상대로 확인.

---

### Exp-12 · abs_angle ❌

**설정:** SAC, ms=3, sp=0.1, tp=1.0, 3 seeds × 5M (seed=0/42 crash, seed=1만 완료)

seed=1 5M: 37.8% / 6.4% — delta 2M(41.7%/8.4%)보다 낮음. 2.5× 스텝 소모로 오히려 뒤처짐. inductive bias(delta=0 → 공 직접 겨냥) 부재가 탐색 공간을 폭발시킴.

**abs_angle 폐기. delta_angle 유지.**

---

### Exp-13 · Phase 0 reward shaping & steps scaling

**설정:** SAC, n_balls=1, current placement, seed=42

**proximity reward (1M):** α ∈ {0.05, 0.1, 0.3, 0.5} 전부 baseline(50%) 하회.
쿠션 반사 후 볼 최종 위치 ≠ action 결과 → post-shot 거리는 노이즈. sparse +1/0이 더 깨끗한 신호.

**steps scaling:**

| Steps | Pocket% | 효율 |
|-------|---------|------|
| 1M | 50.0% | — |
| 2M | 56.2% | 6.2pp/1M |
| 5M | 65.8% | 3.2pp/1M |

steps ∝ 성능. diminishing returns 시작. current placement의 넓은 커버 공간 (ball y범위 3배) 때문에 단순히 더 많은 steps가 필요한 sample complexity 문제.

---

## Project Structure

```
billiards-rl/
├── simulator.py          # BilliardsEnv (n_balls=1: Phase 0, n_balls=3: Phase 1)
├── train.py              # SAC/PPO/TQC 훈련 CLI (주요 진입점)
├── train_curriculum.py   # Curriculum ms=5→4→3
├── logger.py             # ExperimentLogger + BilliardsEvalCallback (Aim 연동)
├── compare.py            # 실험 결과 비교 테이블 + 학습 곡선 PNG
├── visualize.py          # 이미지 그리드 / MP4 영상 / before-after 비교
├── benchmark.py          # 하드웨어 벤치마크 (vec_env × device 조합)
├── run_phase0.py         # Phase 0 재훈련 shortcut (n_balls=1, 1M)
├── requirements.txt
└── logs/
    ├── experiments/      # 실험별 디렉토리 (config.json, results.json, best_model/)
    └── tensorboard/
```

## Setup

```bash
cd ~/Documents/billiards-rl
bash setup.sh              # Python 3.13 venv 생성 + 의존성 설치
```

## 주요 명령어

```bash
# 훈련
python train.py --n-balls 3 --max-steps 5 --step-penalty 0.1 --trunc-penalty 1.0
python train.py --n-balls 3 --max-steps 3 --steps 2000000 --seed 0
python train_curriculum.py --seed 0

# 실험 비교
python compare.py                      # 전체 실험 테이블 + 학습곡선 PNG
python compare.py --filter multi3      # multi-ball 실험만
python compare.py --list               # 실험 디렉토리 목록만

# 시각화
python visualize.py --n-balls 3 --model <exp_dir>/best_model/best_model
python visualize.py --n-balls 3 --mode video --model <path>
python visualize.py --mode compare --before before.mp4 --after after.mp4 \
    --before-label "Random" --after-label "SAC 43%"

# 실시간 메트릭 (Aim)
aim up                                 # localhost:43800

# TensorBoard
tensorboard --logdir logs/tensorboard
```
