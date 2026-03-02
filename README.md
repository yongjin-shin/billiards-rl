# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).
진행 상황과 관찰을 기록하는 실험 노트.

---

## Roadmap

```
Phase 0  Single-ball aiming
  [x] Exp-01  SAC / PPO / TQC benchmark (multi-seed)

Phase 1  Multi-ball clearing — reward shaping & obs ablation (ms=5)
  [x] Exp-02  SAC from scratch — max_steps=∞   →  pocket 98.3%, clear 95.8% (너무 느슨함)
  [x] Exp-03  SAC from scratch — max_steps=5   →  pocket 60.7%, clear 29.4%
  [x] Exp-04  Transfer A · obs-collapse zero-shot  →  pocket 63.6%, clear 31.4% (0 min!)
  [x] Exp-05  Transfer B · weight-copy warm-start  →  pocket 61.5%, clear 30.4%
  [x] Exp-06  Progressive penalty (sp=0.1×i, tp=1.0) → 63.9% / 33.2%  ← 현재 ms=5 best
              ep_len=4.48, 80.9% use all 5 steps — aiming↑ but efficiency unchanged
  [x] Exp-07  SAC vs TQC + clear_bonus(=2.0/steps_used) → SAC 62.0%/29.2%, TQC 49.7%/17.6%
              ep_len unchanged — clear_bonus도 efficiency 개선 없음
  [x] Exp-08  shots_taken obs (24-dim) + cb=2.0   → 63.7%/30.0%, ep_len=4.60 — 효과 없음
  [x] Exp-09  shots_taken + lr=1e-4 + gs=10       → 62.1%/28.8%, ep_len=4.50 — 오히려 worse

  ★ 결론: ep_len은 reward shaping / obs 변경으로 단축 불가. task 자체가 5-step이 최적.

Phase 1  Multi-ball clearing — max_steps ablation
  [x] Exp-10  ms=4, pp=False  → 55.1% / 17.6%, ep_len=3.86 (96.5% use all 4 steps)
  [x] Exp-11  ms=3, pp=False  → 41.4% /  9.0%, ep_len=2.98 (99.3% use all 3 steps)
  [x] Exp-12  ms=5, pp=False  → 63.6% / 32.2%, ep_len=4.40  (Exp-06의 pp=True와 사실상 동등)
  [x] Exp-13  ms=4, pp=True   → 51.2% / 15.8%, ep_len=3.86  (pp가 ms=4에선 역효과)
  [x] Exp-14  ms=3, pp=True   → 41.5% /  7.6%, ep_len=3.00  (pp가 ms=3에선 무의미)

  ★ 결론: pp는 ms=5에서만 소폭 유효. ms=4/3에서는 penalty 누적이 학습을 방해.

Phase 2  (진행 예정)
  [ ] ms=3 frontier — 더 타이트한 환경에서 알고리즘 개선 가능성 탐색
  [ ] Phase 1b  cushion/bank shots (action space 확장)
  [ ] Phase 2   self-play / full 8-ball
  [ ] DreamerV3 (model-based)
```

---

## Environments

두 환경 모두 동일한 2-dim 연속 액션을 사용:

| Dim | Range | 의미 |
|-----|-------|------|
| `delta_angle` | [−π, π] | 가장 가까운 unpocketed ball 방향 기준 aim offset (0 = 직접 겨냥) |
| `speed` | [0.5, 8.0] | 큐볼 타격 속도 (m/s) |

### Phase 0 — single-ball (`n_balls=1`)

공 1개, 샷 1번으로 끝나는 단일 에피소드.

| | |
|---|---|
| **Observation** | 16-dim: `[cue_x, cue_y, ball_x, ball_y, p0x,p0y, …, p5x,p5y]` — [0,1] 정규화 |
| **Reward** | +1 pocketed / 0 otherwise |
| **Episode** | 항상 1 step 후 종료 |

### Phase 1a — multi-ball (`n_balls=3`)

공 3개, 최대 `max_steps` 샷.

| | |
|---|---|
| **Observation** | 23-dim: `[cue_x, cue_y, b1x,b1y,b1_flag, b2x,b2y,b2_flag, b3x,b3y,b3_flag, p0x,p0y,…,p5x,p5y]` |
| **Reward** | +1.0 per ball pocketed · −step_penalty (flat) or −step_penalty×i (progressive) per step · −0.5 for scratch · −trunc_penalty if truncated · +clear_bonus/steps_used on termination (all cleared) |
| **Episode** | 공 3개 전부 pocketed OR step ≥ max_steps |
| **Ball-in-hand** | scratch 시 큐볼을 임의 위치에 재배치 |

---

## Experiment Summary Table

### Phase 0 — n_balls=1

| Exp | Algo | Seed | Pocket% | Mean R | Ep Len |
|-----|------|------|---------|--------|--------|
| 01 | SAC | 0/1/2 | 81.4 / 73.6 / 77.8 (avg **77.6%**) | 0.82/0.84/0.82 | 1.00 |
| 01 | TQC | 0/1/2 | 84.6 / 80.6 / 35.6 (avg 66.9%, **±27pp**) | 0.90/0.80/0.44 | 1.00 |
| 01 | PPO | 0/1/2 | 24.0 / 30.0 / 31.6 (avg 28.5%) | 0.54/0.38/0.46 | 1.00 |

### Phase 1 — n_balls=3

| Exp | Algo | ms | pp | cb | st | Pocket% | Clear% | Mean R | Ep Len | 비고 |
|-----|------|----|----|----|----|---------|--------|--------|--------|------|
| 02 | SAC | ∞  | ✗ | 0   | ✗ | 98.3% | 95.8% | 2.432 | 6.84 | 너무 느슨 |
| 03 | SAC | 5  | ✗ | 0   | ✗ | 60.7% | 29.4% | 1.714 | 4.60 | scratch baseline |
| 06 | SAC | 5  | ✓ | 0   | ✗ | **63.9%** | **33.2%** | 0.038 | 4.48 | **ms=5 best** |
| 07 | SAC | 5  | ✗ | 2.0 | ✗ | 62.0% | 29.2% | 1.028 | 4.50 | cb 효과 없음 |
| 07 | TQC | 5  | ✗ | 2.0 | ✗ | 49.7% | 17.6% | 0.072 | 4.80 | single-seed 불안정 |
| 08 | SAC | 5  | ✗ | 2.0 | ✓ | 63.7% | 30.0% | 1.091 | 4.60 | shots_taken 효과 없음 |
| 09 | SAC | 5  | ✗ | 0   | ✓ | 62.1% | 28.8% | 0.740 | 4.50 | lr↓+gs↑ 오히려 worse |
| 10 | SAC | 4  | ✗ | 0   | ✗ | 55.1% | 17.6% | 0.424 | 3.86 | ms 축소 효과 |
| 11 | SAC | 3  | ✗ | 0   | ✗ | 41.4% |  9.0% | 0.182 | 2.98 | ms 축소 효과 |
| 12 | SAC | 5  | ✗ | 0   | ✗ | 63.6% | 32.2% | 0.990 | 4.40 | pp 없어도 동등 |
| 13 | SAC | 4  | ✓ | 0   | ✗ | 51.2% | 15.8% | −0.056 | 3.86 | pp가 ms=4에서 역효과 |
| 14 | SAC | 3  | ✓ | 0   | ✗ | 41.5% |  7.6% | −0.270 | 3.00 | pp가 ms=3에서 무의미 |

**pp pair 비교 (ms별):**

| ms | pp=✗ | pp=✓ | Δ pocket | Δ clear |
|----|------|------|----------|---------|
| 5  | 63.6% / 32.2% (Exp-12) | 63.9% / 33.2% (Exp-06) | +0.3pp | +1.0pp |
| 4  | 55.1% / 17.6% (Exp-10) | 51.2% / 15.8% (Exp-13) | **−3.9pp** | −1.8pp |
| 3  | 41.4% /  9.0% (Exp-11) | 41.5% /  7.6% (Exp-14) | +0.1pp | −1.4pp |

---

## Experiments — Done

### Exp-01 · Phase 0 single-ball benchmark

**목표:** SAC / PPO / TQC를 동일 조건에서 비교. horizon=1에서 각 알고리즘의 특성 파악.

**설정:** 1M steps, seed {0, 1, 42}, `n_balls=1`

**결과:**

| Algorithm | Pocket rate (mean ± std) | vs. random |
|-----------|--------------------------|------------|
| SAC | 77.6% ± 3.9 pp | +71 pp |
| TQC | 66.9% ± 27.2 pp | +61 pp |
| PPO | 28.5% ± 4.0 pp | +22 pp |
| Random | ~6% | — |

**관찰:**
- Horizon=1에서 off-policy(SAC/TQC)가 압도적. PPO의 GAE는 단일 transition에서 REINFORCE로 퇴화.
- TQC는 seed 간 분산이 매우 큼 (어떤 seed는 SAC 수준, 어떤 seed는 붕괴). 분포 추정의 불안정성으로 추정.
- SAC의 best_model vs final_model 격차가 종종 큼 — 훈련 말기에 collapse 발생. EvalCallback의 best_model을 사용해야 함.

---

### Exp-02 · Phase 1a multi-ball (max_steps=15)

**목표:** multi-ball 환경에서 SAC scratch 훈련 가능성 확인.

**설정:** SAC, 1M steps, seed=42, `n_balls=3`, `max_steps=15`

**결과:**

| Metric | Value |
|--------|-------|
| Pocket rate | 98.27% |
| Clear rate (all 3) | 95.8% |
| Random baseline | ~40% |
| Training time | 31.1 min |

**관찰:**
- 수치 자체는 높지만, max_steps=15가 너무 넉넉해서 random agent도 40%를 기록.
- 에이전트가 비효율적으로 여러 번 샷을 반복하는 경향. "일단 많이 쏘면 들어간다" 전략에 가까움.
- horizon을 줄여 shot efficiency를 강제할 필요 → Exp-03.

---

### Exp-03 · Phase 1a multi-ball (max_steps=5)

**목표:** 타이트한 horizon으로 shot efficiency를 강제. Exp-02와 비교.

**설정:** SAC, 1M steps, seed=42, `n_balls=3`, `max_steps=5`

**결과:**

| Metric | Exp-02 (ms=15) | Exp-03 (ms=5) |
|--------|----------------|---------------|
| Pocket rate | 98.3% | 60.7% |
| Clear rate (all 3) | 95.8% | 29.4% |
| Random baseline | ~40% | ~17% |
| Training time | 31.1 min | 25.4 min |

**관찰:**
- max_steps=5는 훨씬 어려운 문제. 5번 안에 3개를 넣으려면 샷당 평균 0.6개 이상 pocketing해야 함.
- clear rate 29.4%는 Exp-02의 95.8%와 극명한 차이 — horizon이 difficulty를 얼마나 바꾸는지 확인.
- Transfer 실험(Exp-04, 05)의 scratch baseline.

---

### Exp-04 · Transfer A — obs-collapse zero-shot eval

**목표:** Phase 0에서 학습한 aiming skill이 추가 훈련 없이 multi-ball에 얼마나 transfer되는지 측정.

**방법:** `ObsCollapseWrapper`로 23-dim obs → 16-dim 축소. 매 step 가장 가까운 unpocketed ball을 "the ball"로 제시. Pretrained n_balls=1 모델(seed=0, 81.4%) 그대로 사용.

**결과:**

| Metric | Exp-03 scratch | Exp-04 zero-shot |
|--------|----------------|-----------------|
| Pocket rate | 60.7% | **63.6%** |
| Clear rate (all 3) | 29.4% | **31.4%** |
| Training | 25.4 min | 0 min |

**관찰:**
- 추가 훈련 없이 scratch(Exp-03)와 거의 동등한 성능. aiming skill이 multi-ball로 그대로 transfer됨.
- 공을 1개씩 순서대로 보여주는 wrapper 덕분에 n_balls=1 모델이 n_balls=3 환경에서도 유효하게 작동.
- 단, ball2/ball3 위치를 모르므로 interference는 피할 수 없음 — 이 이상의 개선은 full obs 없이는 불가.

---

### Exp-05 · Transfer B — weight-copy warm-start

**목표:** Phase 0 가중치를 warm-start로 활용했을 때 Exp-03(scratch)보다 빠르게 수렴하는지 확인.

**방법:** 23-dim SAC를 새로 만들고, 첫 번째 input layer의 shared 뉴런(cue, ball1, pocket 열)에 n_balls=1 모델 가중치 복사. ball2/ball3 뉴런은 0 초기화 후 전체 fine-tune.

**결과:**

| Metric | Exp-03 scratch | Exp-04 zero-shot (A) | Exp-05 warm-start (B) |
|--------|----------------|----------------------|----------------------|
| Pocket rate | 60.7% | 63.6% | **61.5%** |
| Clear rate | 29.4% | 31.4% | **30.4%** |
| Training time | 25.4 min | 0 min | 22.6 min |

**관찰:**
- 학습 곡선을 보면 **초반 reward가 높다가 점점 떨어지는** 패턴이 관찰됨. weight dilution의 전형적인 증거.
  - b2/b3 뉴런이 0에서 시작 → 초반에는 shared 뉴런(pretrained)이 주도 → 높은 reward
  - 학습이 진행되면서 multi-ball gradient가 shared 뉴런을 점진적으로 덮어씀 → reward 하락
  - 결국 pretrained weight의 이점이 희석되어 scratch(Exp-03)와 거의 수렴
- **ep_len_mean ≈ 4.7** — 5번 step 한도를 거의 다 소진하고 끝남. agent가 "빨리 끝내는" 전략을 배우지 못했음을 시사. 현재 step penalty(-0.01)가 너무 작아서 agent 입장에서는 step을 아낄 동기가 없음.
- 가장 충격적인 결과는 **Exp-04(zero-shot)** — 훈련 0분으로 B보다 더 좋음. aiming skill의 direct transfer 효과가 weight-copy fine-tuning 전체를 압도.

**다음 방향:** → Exp-06에서 progressive step penalty + truncation penalty로 개선 시도.

---

### Exp-06 · Progressive reward shaping (sp=0.1×step, tp=1.0)

**목표:** ep_len≈4.7 문제 해결. Progressive penalty로 urgency gradient를 만들고, truncation penalty로 미완료 에피소드를 명시적으로 패널티. Exp-03/04/05와 동일 구조(scratch / Transfer A / Transfer B)로 비교.

**변경 사항 (simulator.py):**
- step penalty: 고정 −0.01 → **progressive: step i에서 −0.1×i**
  - step 1: −0.1 / step 2: −0.2 / ... / step 5: −0.5
  - 3-step vs 5-step clear 보상 차이: 0.2pp → **0.9pp (4.5×)**
- truncation penalty: 없음 → **−1.0** (step limit 도달 시)

**결과:**

| Metric | Exp-03 scratch | Exp-04 Transfer A | Exp-05 Transfer B | Exp-06 scratch | Exp-06 Transfer A | Exp-06 Transfer B |
|--------|----------------|-------------------|-------------------|----------------|-------------------|-------------------|
| Pocket rate | 60.7% | 63.6% | 61.5% | **63.9%** | **64.3%** | **64.8%** |
| Clear rate | 29.4% | 31.4% | 30.4% | **33.2%** | **30.2%** | **32.2%** |
| Training time | 25.4 min | 0 min | 22.6 min | 36.2 min | 0 min | 36.6 min |

**Episode length 분포 (Exp-06 scratch best_model, n=1000):**

| step | count | % |
|------|-------|---|
| 2 | 15 | 1.5% |
| 3 | 64 | 6.4% |
| 4 | 112 | 11.2% |
| 5 | 809 | **80.9%** |

Clear 에피소드의 step 분포: step2 4.7% / step3 20.0% / step4 35.0% / **step5 40.3%** — avg clear step **4.11**

**관찰:**
- 세 조건 모두 pocket rate 3~4pp 향상. progressive penalty가 더 나은 aiming을 유도함.
- **Transfer B가 처음으로 A를 역전 (64.8% > 64.3%).** 강화된 reward structure에서 warm-start가 비로소 추가 가치를 냄.
- **목표였던 ep_len 단축은 실패.** Exp-03/05와 동일하게 80.9%가 5 step을 전부 소진. clear rate 향상(+3~4pp)은 shot efficiency 증가가 아니라 aiming 정확도 향상에서 기인.
- **근본 원인:** agent가 현재 step이 몇 번째인지 모름. obs에 step_remaining이 없으므로 urgency를 인식할 수 없음. 단, step_remaining은 실제 당구에 존재하지 않는 정보이므로 obs에 추가하는 것은 현실성이 없음.
- **step 5 penalty(-0.5) + 공 하나 기댓값(+1.0 × ~30%)** → 5번째 샷 기댓값 ≈ −0.2. 여전히 쏘는 게 안 쏘는 것(terminal이 아니라 truncated)보다 낫기 때문에 agent가 5번째 step을 계속 사용함. reward만으로 ep_len을 줄이려면 step 5 penalty를 −1.5+ 수준으로 올려야 하지만, 그러면 학습 자체가 불안정해질 위험이 있음.

**다음 방향:** progressive penalty는 ep_len을 줄이지 못함 (에이전트가 step count를 모르므로 urgency 인식 불가). 대신 **clear_bonus = 2.0/steps_used**로 전환 — 빠른 클리어에 직접적인 보너스. SAC vs TQC 비교 → Exp-07.

---

### Exp-08–14 · Ablation series (요약)

상세 기록은 Experiment Summary Table 참고. 핵심 발견:

- **Exp-08/09 (shots_taken):** obs에 urgency 정보 추가해도 ep_len 변화 없음. 당구에서 최적 action은 step-dependent하지 않음 — shots_taken이 uninformative feature.
- **Exp-10/11 (ms 축소):** ms를 줄여도 ep_len/ms 비율이 오히려 상승 (ms=5: 90% → ms=3: 99%). agent가 lazy한 게 아니라 task 자체가 모든 step을 소모하는 구조.
- **Exp-12/13/14 (pp pair):** pp는 ms=5에서 +1pp 수준의 미미한 효과. ms=4/3에선 penalty 누적이 학습을 방해해 역효과. **pp의 효용 없음** 결론.

---

### Exp-08 · shots_taken obs + clear_bonus ablation

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=5, sp=0.1, tp=1.0, cb=2.0, shots_taken=True

**관찰:**
- 63.7% / 30.0%, ep_len=4.60 — Exp-07 SAC baseline(62.0%/29.2%)과 사실상 동등, 개선 없음.
- 훈련 초반 critic loss explosion 발생 — cb=2.0의 높은 reward scale(+3.37 max)이 SAC Q값을 불안정하게 만든 것으로 추정. SAC는 gradient clipping 없음(SB3 기본).
- shots_taken이 urgency 신호로 작동할 것을 기대했으나 ep_len이 오히려 4.50→4.60으로 증가. 당구에서 최적 action은 위치(obs)에만 의존하고 step count와 독립적임.
- shots_taken은 Markov property를 강화하는 feature이지 non-stationarity를 유발하지 않음. uninformative feature를 추가하면 오히려 network 용량을 낭비.

---

### Exp-09 · shots_taken + SAC stability tuning (lr=1e-4, gs=10)

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=5, sp=0.1, tp=1.0, cb=0.0, shots_taken=True, lr=1e-4, gradient_steps=10

**관찰:**
- 62.1% / 28.8%, ep_len=4.50 — Exp-08보다 오히려 worse. clear_bonus 제거 후에도 shots_taken이 도움 안 됨.
- gs=10 → UTD(Update-To-Data)=1.0 (N_ENVS=10, gs=10 → 업데이트 : 데이터 = 1:1). vanilla SAC에서 high UTD는 critic overestimation cascade를 심화시킴.
- lr=1e-4은 안정성 향상에 기여하지 못함. SAC hyperparameter 튜닝만으로는 critic instability 해결 불가.
- **결론:** shots_taken 불채택. ep_len은 obs 변경으로도 단축 불가 — task 구조 자체의 문제.

---

### Exp-10 · ms=4, pp=False (max_steps 축소 ablation)

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=4, sp=0.1, tp=1.0

**관찰:**
- 55.1% / 17.6%, ep_len=3.86 — ms=5 baseline(Exp-12: 63.6%/32.2%) 대비 pocket −8.5pp, clear −14.6pp.
- ep_len/ms = 3.86/4 = 96.5% (ms=5의 88%보다 높음). ms를 줄여도 모든 step을 소모하는 경향이 강해짐.
- clear rate가 pocket rate에 비해 급감: pocket 55.1% → clear 17.6%. 3개 ball을 4번 안에 모두 넣는 조합적 난이도가 급상승.

---

### Exp-11 · ms=3, pp=False (max_steps 최소화)

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=3, sp=0.1, tp=1.0

**관찰:**
- 41.4% / 9.0%, ep_len=2.98 — ep_len/ms = 2.98/3 = **99.3%**. 사실상 모든 에피소드가 3번 full 소비.
- clear rate 9.0% — 3번 안에 3개를 모두 넣어야 하므로 매 샷이 pocket이어야 함. reward가 매우 sparse.
- ms=5에서 ms=3으로 줄였을 때 clear rate가 33.2% → 9.0%로 3.7배 감소. 조합적 난이도 폭발적 상승.
- **Phase 2 frontier로 선정.** reward shaping이 아닌 알고리즘적 개선(탐색, 다단계 계획)이 필요한 구간.

---

### Exp-12 · ms=5, pp=False (pp ablation 페어 기준)

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=5, sp=0.1, tp=1.0 ← Exp-03와 동일 구조

**관찰:**
- 63.6% / 32.2%, ep_len=4.40 — Exp-06(pp=True) 63.9%/33.2%와 사실상 동등.
- pp의 ms=5 기여: pocket +0.3pp, clear +1.0pp — 노이즈 수준의 차이.
- **ms=5에서 pp는 실질적으로 무효.**

---

### Exp-13 · ms=4, pp=True (pp ablation 페어)

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=4, sp=0.1, tp=1.0, progressive_penalty=True

**관찰:**
- 51.2% / 15.8% — Exp-10(pp=False) 55.1%/17.6% 대비 pocket **−3.9pp**, clear −1.8pp. pp가 역효과.
- ms=4에서 progressive penalty 누적: step 1(−0.1) + step 2(−0.2) + step 3(−0.3) + step 4(−0.4) = **−1.0 total**. pocket 1개(+1.0)가 4번의 step penalty와 상쇄 → 학습 신호 약화.
- penalty가 너무 커서 agent가 적극적으로 샷을 시도하지 않는 쪽으로 수렴.

---

### Exp-14 · ms=3, pp=True (pp ablation 페어)

**설정:** SAC, 1M steps, seed=42, n_balls=3, ms=3, sp=0.1, tp=1.0, progressive_penalty=True

**관찰:**
- 41.5% / 7.6% — Exp-11(pp=False) 41.4%/9.0% 대비 pocket +0.1pp(무의미), clear **−1.4pp**.
- ms=3 pp 누적: step 1(−0.1)+2(−0.2)+3(−0.3) = −0.6. pocket 기댓값(~41% × 1.0 = +0.41) < penalty(0.6/3 = 0.2 per step) → 첫 샷부터 기댓값이 거의 0.
- pp의 ms=3에서의 효과는 없음(pocket 동률) 혹은 해로움(clear 감소).
- **★ 결론 (Exp-12~14 종합):** pp는 어떤 ms에서도 유의미한 개선 없음. 폐기.

---

## Experiments — Planned

### Exp-07 · SAC vs TQC — clear_bonus reward shaping

**목표:** (1) progressive penalty 대신 clear_bonus(=2.0/steps_used)로 빠른 클리어에 직접 보상. (2) SAC vs TQC 직접 비교 — 고분산 reward 환경에서 distributional Q-학습의 이점 측정.

**설정:** SAC·TQC 각 1회, 1M steps, seed=42, n_balls=3, max_steps=5
reward: sp=0.1 (flat), tp=1.0, **clear_bonus=2.0**

**Reward 구조 (예시, n_balls=3):**

| 클리어 step | ball reward | step pen | clear bonus | total |
|------------|------------|----------|-------------|-------|
| 3-step | +3.0 | −0.3 | +0.667 | **+3.37** |
| 5-step | +3.0 | −0.5 | +0.400 | **+2.90** |
| truncated | varies | −0.5 | 0 | ≤ −1.5 |

3-step vs 5-step gap: **+0.47** (clear_bonus 없을 때 +0.20, 2.3× 향상)

**결과:**

| Metric | SAC (Exp-07) | TQC (Exp-07) | SAC Exp-06 (참고) |
|--------|-------------|-------------|-----------------|
| Pocket rate | **62.0%** | 49.7% | 63.9% |
| Clear rate | **29.2%** | 17.6% | 33.2% |
| ep_len (5-step %) | 81.5% | 91.0% | 80.9% |
| avg clear step | 4.14 | 4.16 | 4.11 |
| Training time | 32.4 min | 26.3 min | 36.2 min |

**Episode length 분포 (SAC Exp-07, n=1000):**

| step | count | % |
|------|-------|---|
| 1 | 3 | 0.3% |
| 2 | 11 | 1.1% |
| 3 | 64 | 6.4% |
| 4 | 107 | 10.7% |
| 5 | 815 | **81.5%** |

**관찰:**
- **clear_bonus도 ep_len 단축 실패.** 81.5%가 여전히 5-step 소진 — Exp-06(80.9%)과 사실상 동일. 3-step vs 5-step 보상 차이(+0.47)가 policy gradient를 바꾸기엔 부족하거나, 에이전트가 step count를 모르므로 이 차이를 활용할 수 없음. **ep_len 단축은 obs 변경 없이는 불가능해 보임 → Exp-08 shots_taken 필요.**
- **TQC가 SAC보다 크게 열세 (49.7% vs 62.0%).** Exp-01에서 관찰된 seed 간 불안정성(±27pp)이 여기서도 재현. 단일 seed로는 TQC가 나쁜 local optima에 수렴할 가능성이 있음. multi-seed 비교 없이 TQC 우위를 주장하기 어려움.
- SAC Exp-07 (62.0%)이 Exp-06 (63.9%)보다 소폭 낮음 — clear_bonus가 reward scale을 높여(최대 +3.37) Q-learning을 오히려 불안정하게 만들었을 가능성.

**다음 방향:** → Exp-08~14 ablation 시리즈 (완료, 위 요약 참고).

---

## Project Structure

```
billiards-rl/
├── simulator.py          # BilliardsEnv — pooltool gymnasium wrapper
│                         #   n_balls=1  →  Phase 0 (16-dim obs, horizon=1)
│                         #   n_balls=3  →  Phase 1a (23-dim obs, multi-shot)
├── train.py              # SAC / PPO / TQC 훈련 (--n-balls, --max-steps 지원)
├── train_pretrained.py   # 전이학습: obs-collapse (A) + weight-copy (B)
├── compare.py            # 실험 결과 비교 테이블 + 학습 곡선 PNG
├── visualize.py          # 이미지 그리드 / MP4 영상 / before-after 비교
├── benchmark.py          # 멀티시드 벤치마크 러너
├── run_comparison.sh     # SAC + PPO + TQC 배치 훈련 → 비교
├── requirements.txt
├── setup.sh
└── logs/
    ├── experiments/      # 실험별 디렉토리 (config, results, checkpoints)
    └── tensorboard/
```

---

## Visualization & Analysis

```bash
# 이미지 그리드 (12 에피소드)
python visualize.py --n-balls 1
python visualize.py --n-balls 3 --model <exp_dir>/best_model/best_model

# MP4 영상
python visualize.py --n-balls 3 --mode video --model <path>

# Before / After 비교 영상 연결
python visualize.py --mode compare --before before.mp4 --after after.mp4

# 실험 비교 테이블 + 학습 곡선
python compare.py
python compare.py --filter multi3      # multi-ball 실험만
python compare.py --filter transfer    # 전이학습 실험만

# TensorBoard
tensorboard --logdir logs/tensorboard
```

---

## Setup

```bash
cd ~/Documents/billiards-rl
bash setup.sh
```

Python 3.13 (Homebrew), `.venv` 생성, 의존성 설치.
