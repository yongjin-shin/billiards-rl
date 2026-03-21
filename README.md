# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).
진행 상황과 관찰을 기록하는 실험 노트.

---

## Quick Status

| | |
|---|---|
| **현재 위치** | Exp-13 proximity reward — 모든 α에서 baseline 50% 하회. 방향 재검토 중 |
| **진행 중** | 2M steps 실험 (α=0.0 / α=0.3) — steps 증가 자체의 효과 확인 중 |
| **Exp-06 재실험** | seed=0 완료 63.1%/31.2% ≈ 원래 63.9%/33.2% (재현 확인) |

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

## Exp-13 결과 · Phase 0 Proximity Reward ❌

### 결과

| α | Pocket% | vs baseline |
|---|---------|-------------|
| 0.0 (baseline) | **50.0%** | — |
| 0.05 | 40.8% | −9.2pp |
| 0.1 | 37.8% | −12.2pp |
| 0.3 | 41.4% | −8.6pp |
| 0.5 | 42.0% | −8.0pp |

모든 α에서 baseline 하회. α값에 무관하게 일관된 역효과.

### 관찰

**왜 proximity reward가 작동하지 않는가:**

쿠션에 1~2번만 맞아도 볼의 최종 위치와 초기 조준 방향 사이의 인과관계가 끊긴다.
`f(action) → final_pos`가 아니라 `f(action) → chaos`가 되어버리므로,
포켓 근처에 튕겨 나온 것이 좋은 샷이었다는 보장이 없다.
**post-shot 최종 거리는 유효한 gradient signal이 아니다.**

**sample complexity 문제:**

current placement (y=[0.30,0.85])는 legacy (y=[0.60,0.90])보다 훨씬 넓은 공간을 커버하므로,
에이전트가 6개 포켓 방향을 모두 학습해야 한다. 동일한 1M steps에서 각 구성의 방문 빈도가 줄고,
coverage가 크게 부족해질 수 있다. 2M 이상이 필요할 수 있으며, world model 기반 접근이
sample efficiency 측면에서 유리할 수 있다.

> 2M steps 실험 (α=0.0 / α=0.3) 진행 중 — steps 증가 자체의 효과 확인 후 방향 결정.

---

## Roadmap

```
[x] Exp-13   Phase 0 proximity reward → 실패 (post-shot 거리 = noise after bounces)

[ ] Exp-13b  2M steps (α=0.0 / α=0.3) 진행 중 — steps 효과 단독 확인
[ ] Exp-14   Phase 0 방향 결정 후 (curriculum placement / world model / 기타)
[ ] Exp-15   HRL

[ ] cushion / bank shots
[ ] self-play / full 8-ball
[ ] DreamerV3 (model-based, sample efficiency)
```

---

## Future: Exp-14 HRL

### 설계 근거

현재 flat MLP의 병목은 두 레벨 문제를 혼합:

| | System 1 (aiming) | System 2 (strategy) |
|---|---|---|
| **역할** | 이 각도+속도로 쏘면 저 공이 들어가는가 | 어떤 공을 먼저, 어느 포켓으로 |
| **reward** | 즉각적 (+1 per pocket) | 희박하고 지연됨 (9% clear) |
| **현재 문제** | nearest-ball greedy에 하드코딩됨 | gradient 신호 거의 없음 |

**해결:** System 2가 target ball을 명시적으로 지정 → obs 재배열 → Phase 1 policy(System 1)가 실행.

```
System 2 (새로 학습):  target ball 선택 (discrete)
                       ↓
obs 재배열:  [cue_xy, target_xyz, other1_xyz, other2_xyz, 6pockets]
             target ball → ball[0] 위치로 이동
                       ↓
System 1 (Phase 1 Exp-10 freeze):
  delta_angle = 0 → ball[0] 방향(= target) 겨냥
```

**왜 Phase 1인가:** Phase 0(n_balls=1)은 다른 공과의 간섭/충돌 경험 없음. Phase 1은 3볼 경험 + 23-dim obs 그대로 재사용 가능.

### Exp-14 variants

| | 14a | 14b | 14c | 14d |
|---|-----|-----|-----|-----|
| **System 2 action** | 공 선택 (discrete 3) | 공+포켓 (discrete 18) | 공+포켓 (discrete 18) | 공+포켓 (discrete 18) |
| **System 1** | Phase 1 freeze | Phase 1 freeze | Phase 1 freeze | joint 학습 (no pretrain) |
| **포켓 처리** | System 1 자율 | 비목표 포켓 마스킹 | target pocket 첫 번째 | System 2 지정 |
| **OOD 리스크** | 낮음 | 중간 | 중간 | 없음 |
| **핵심 질문** | HRL 구조 자체 유효한가? | masking으로 포켓 강제 가능한가? | 포켓 info가 도움되는가? | pretraining 없이 가능한가? |

**실험 순서:** 14a → 14b/c → 14d (OOD 리스크 낮은 순으로, 14d는 ablation baseline)

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

### Exp-13 · Phase 0 proximity reward ❌

**설정:** SAC, 1M, seed=42, n_balls=1, current placement, α ∈ {0.05, 0.1, 0.3, 0.5}

```
r = +1.0 (pocketed) + α · (−min_dist(ball→pocket) / d_max)
```

모든 α에서 baseline(50.0%) 하회 (37.8%~42.0%). α값에 무관한 일관된 역효과.

**근본 원인:** 쿠션 반사 후 볼 최종 위치는 초기 action과 인과관계가 끊김. post-shot 거리는 gradient signal이 아닌 노이즈. sparse +1/0가 오히려 더 깨끗한 신호.

**추가 관찰:** current placement는 legacy 대비 커버 공간이 크게 넓어 (y범위 3배), 동일 steps에서 각 구성 방문 빈도 하락 → sample efficiency 문제. 2M+ 또는 world model 기반 접근 검토 필요.

---

### Exp-12 · abs_angle ❌

**설정:** SAC, ms=3, sp=0.1, tp=1.0, 3 seeds × 5M (seed=0/42 crash, seed=1만 완료)

seed=1 5M: 37.8% / 6.4% — delta 2M(41.7%/8.4%)보다 낮음. 2.5× 스텝 소모로 오히려 뒤처짐. inductive bias(delta=0 → 공 직접 겨냥) 부재가 탐색 공간을 폭발시킴.

**abs_angle 폐기. delta_angle + System 2 명시적 target 지정(Exp-13)으로 공 선택 자유도 해결.**

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
