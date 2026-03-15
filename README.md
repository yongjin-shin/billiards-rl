# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).
진행 상황과 관찰을 기록하는 실험 노트.

---

## Roadmap

```
Phase 0  Single-ball aiming
  [x] Exp-01  SAC / PPO / TQC benchmark (multi-seed)
              → SAC 77.6% (best), PPO 28.5%

Phase 1  Multi-ball clearing (n_balls=3)

  1-1  Feasibility & baseline
  [x] Exp-02  ms=∞  →  98.3% / 95.8%  (너무 느슨)
  [x] Exp-03  ms=5 scratch  →  60.7% / 29.4%  (Phase 1 baseline)
  [x] Exp-04  Transfer A — zero-shot  →  63.6% / 31.4%  (훈련 0분!)
  [x] Exp-05  Transfer B — warm-start  →  61.5% / 30.4%

  1-2  Reward shaping — ep_len 단축 시도 (모두 실패)
  [x] Exp-06  Progressive penalty (pp)  →  63.9% / 33.2%  ← ms=5 best
              ep_len=4.48, 80.9% use all 5 steps — aiming↑, efficiency 변화 없음
  [x] Exp-07  clear_bonus=2.0 + SAC vs TQC  →  SAC 62.0% / TQC 49.7%
              ep_len 변화 없음 — clear_bonus도 efficiency 개선 불가
  [x] Exp-08  shots_taken obs ablation
       08a  shots_taken + cb=2.0  →  63.7% / 30.0%, ep_len=4.60  (효과 없음)
       08b  shots_taken + lr=1e-4 + gs=10  →  62.1% / 28.8%  (오히려 worse)

  ★ 결론: ep_len은 reward shaping / obs 변경으로 단축 불가.
          task 자체가 모든 step을 소모하는 구조 (per-step 기댓값 > 0).

  1-3  max_steps × progressive_penalty ablation grid
  [x] Exp-09  ms ∈ {5, 4, 3}  ×  pp ∈ {✗, ✓}  (6-cell grid)
       09a  ms=5, pp=✗  →  63.6% / 32.2%  (Exp-06과 사실상 동등 → pp 효용 없음)
       09b  ms=5, pp=✓  →  Exp-06 참고
       09c  ms=4, pp=✗  →  55.1% / 17.6%
       09d  ms=4, pp=✓  →  51.2% / 15.8%  (pp가 역효과: −3.9pp)
       09e  ms=3, pp=✗  →  41.4% /  9.0%
       09f  ms=3, pp=✓  →  41.5% /  7.6%  (pp가 무의미)

  ★ 결론: pp 폐기. ms=3이 Phase 2 frontier (sparse reward, 알고리즘 개선 필요).

Phase 2  ms=3 frontier
  [x] Exp-10  SAC/TQC/PPO × 3-seed benchmark  (ms=3, 2M steps)
              SAC 41.7%/8.4% · TQC 27.1%/2.0% · PPO 6.5%/0.0%  → SAC 압도적 우위
  [x] Exp-11  Curriculum: SAC ms=5 → ms=4 → ms=3
              43.0%/10.4% (2M total) — Exp-10 대비 +1.3pp/+2.0pp 개선
              (+) ms=2 extension: 28.6%/1.6% → 더블포켓 필수, 학습 신호 부재 확인
  [x] Exp-12  Action space 교체: delta_angle → absolute angle [0, 2π]  ★ 폐기
              seed=1 5M: 37.8%/6.4%  — delta_angle 2M(41.7%/8.4%)보다 낮음
              결론: abs_angle 불필요. delta_angle + HRL(Exp-13)이 올바른 방향.
  [x] Phase 0 재훈련 (부산물): n_balls=1, 1M, seed=42 → 42.4% / random 3% (14× 개선)
              ※ 구 Exp-01(77.6%)은 target 배치가 y=[0.6,0.9]로 상단 고정 → 쉬운 직선샷 위주.
                 현재 코드는 target y=[0.30,0.85]로 확장(89431bd) → 더 다양하고 어려운 배치.
                 배치 난이도 차이가 77.6%→42.4% 하락의 주원인. horizon=1 단일샷은 동일.
                 Exp-13은 Phase 1(Exp-10 best_model, 23-dim) 기반으로 설계 → Phase 0 미사용.
  [ ] Exp-13a HRL-A: 공 선택 (discrete 3) · Phase 1 freeze · obs 재배열(target→ball[0])
  [ ] Exp-13b HRL-B: 공+포켓 선택 (discrete 18) · Phase 1 freeze · 비목표포켓 마스킹
  [ ] Exp-13c HRL-C: 공+포켓 선택 (discrete 18) · Phase 1 freeze · 전체 포켓 공개
  [ ] Exp-13d HRL-D: scratch (Phase 1 없이 joint 학습)
  [ ] cushion/bank shots (action space 확장)
  [ ] self-play / full 8-ball
  [ ] DreamerV3 (model-based)
```

---

## 분석 — Phase 1 병목과 Phase 2 방향

### System 1 / System 2 분리

Phase 1 실험 전체를 돌아보면, 현재 병목이 두 가지 서로 다른 레벨의 문제를 혼동하고 있음을 알 수 있다.

**System 1 — aiming execution (저수준 운동 제어)**
> "이 방향, 이 속도로 쏘면 저 공이 저 포켓에 들어간다"

- 물리 역학 학습: cut angle, speed → ball trajectory
- **Phase 0에서 이미 검증 완료**: SAC 77.6% (단일공)
- **Transfer A (zero-shot)가 훈련 0분으로 63.6%** = System 1이 multi-ball로 그대로 전이됨
- reward signal이 즉각적 (공이 들어갔는가?) → 학습이 쉬움

**System 2 — strategic planning (고수준 의사결정)**
> "3개를 3번에 넣으려면 어떤 공을 먼저, 어떤 포켓으로, 다음 샷을 위해 큐볼을 어디에?"

- 어떤 공을 먼저 칠지 (ordering)
- 어떤 포켓에 넣을지 (pocket selection)
- 큐볼 포지셔닝 (position play)
- reward signal이 희박하고 지연됨 (ms=3 clear rate 9%) → 학습이 매우 어려움

**현재 flat MLP 정책의 문제:**
단일 [256, 256] MLP가 System 1과 System 2를 동시에 처리하는데, 두 시스템의 reward signal 밀도가 극단적으로 다름. 결과적으로 System 1(즉각 reward)에 압도적으로 편향되어 "nearest ball을 향해 일단 쏜다"에 수렴. 이것이 ep_len ≈ ms (모든 step 소비)이면서 clear rate는 낮은 이유.

---

### Action Space 구조적 문제 → Exp-12 배경

현재 `delta_angle ∈ [−π, π]`는 **nearest unpocketed ball 방향을 기준점**으로 삼는다.

```
실제 발사 각도 = angle(cue → nearest_ball) + delta_angle
```

이 설계는 **System 2를 greedy nearest-first로 하드코딩**한 것과 동일하다:

| 문제 | 설명 |
|------|------|
| 공 선택 불가 | 두 번째로 가까운 공을 먼저 치려면 우연히 큰 delta_angle을 발견해야 함 |
| 기준 non-stationarity | 공이 포켓될 때마다 nearest ball이 바뀌어 같은 delta_angle이 다른 방향을 의미 |
| 포켓 선택 불가 | 6개 포켓 중 어느 쪽으로 넣을지 표현 수단 없음 |

**왜 처음에 상대 각도를 썼나:**
Phase 0 초기에는 obs가 `[cue_x, cue_y, ball_x, ball_y]` 4-dim뿐으로 포켓 위치가 없었음. 이 상태에서 절대 각도를 쓰면 agent가 "어느 방향이 포켓인지"를 obs에서 알 수 없어 학습이 불가능했음. `delta_angle = 0 → 공 직접 겨냥`이라는 inductive bias가 필수였고, 이것이 Phase 0 SAC 77.6% 달성의 핵심이었음.

**지금 조건이 바뀌었다:**
Phase 1부터 obs에 포켓 좌표 12개가 추가됨 (23-dim). agent는 모든 공과 포켓의 위치를 다 알고 있으므로, 절대 각도 `[0, 2π]`를 사용해도 geometry에서 직접 최적 각도를 학습할 수 있음. 또한 nearest-ball 기준은 애초에 Phase 1b에서 explicit ball selection으로 교체할 예정이었던 임시 설계.

**Exp-12에서 확인할 것:**
> 절대 각도로 교체했을 때 agent가 nearest-first 제약 없이 최적 공 순서를 스스로 발견하는가?
> 같은 SAC 알고리즘, 같은 obs에서 action 표현만 바꿨을 때 clear rate가 오르는가?

**Exp-12 결과 (완료):** ❌ **효과 없음 → 폐기**

| | delta 2M (Exp-10) | abs 2M (s42) | abs 5M (s1) |
|---|---|---|---|
| Pocket | 41.7% | 32.5% | 37.8% |
| Clear | 8.4% | 2.0% | 6.4% |

abs_angle은 inductive bias(delta=0이면 공 직접 겨냥)가 없어 탐색 공간이 넓어지고 학습 속도가 급감. 5M을 써도 delta 2M을 따라잡지 못했다. **System 1(aiming)에 delta_angle의 inductive bias는 여전히 필수.**

공 선택 자유도 문제는 Exp-13(HRL)에서 System 2가 target ball을 명시적으로 지정하는 방식으로 해결 → abs_angle 불필요.

---

### System 1/2 명시적 분리 → Exp-13 배경

Exp-12(절대 각도)로 action 표현력 문제를 해결해도, 근본적인 구조 문제가 남는다. 단일 flat MLP가 여전히 두 시스템을 동시에 학습하면서 gradient signal 밀도 차이가 지속됨.

**핵심 모순:**
- System 1(공 하나 pocketing)은 에피소드마다 0~3회 즉각 reward
- System 2(3개 모두 clear)는 ms=3에서 9%만 성공 → 대부분 에피소드에서 학습 신호 없음
- 같은 네트워크, 같은 optimizer가 두 신호를 동시에 처리 → System 1에 편향 불가피

**HRL 기본 구조:**
```
High-level (System 2, 새로 학습):
  obs → 목표 선택 → System 1에 전달
  reward: clear 여부

Low-level (System 1, Phase 1 기반 — Exp-10 best_model):
  obs(재배열) → (angle, speed) 실행
  reward: 목표 공이 (목표 포켓에) 들어갔는가
```

**왜 Phase 0이 아닌 Phase 1인가:**
Phase 0 (n_balls=1)은 단일공 환경만 경험해 다른 공과의 간섭, 충돌을 모른다.
목표는 3볼 환경에서의 클리어이므로, 3볼 경험이 있는 Phase 1 policy가 System 1으로 더 적합하다.
Phase 1은 이미 23-dim obs를 사용하므로 obs 구조 변경 없이 ball 순서만 재배열하면 된다.

```
System 2: target_ball 선택 (discrete 3)
           ↓
obs 재배열: [cue_xy, target_ball_xyz, other1_xyz, other2_xyz, 6pockets]
           → target ball을 obs에서 ball[0] 위치로
           ↓
System 1 (Exp-10 weight, frozen):
  delta_angle = 0 → ball[0] 방향(= target ball) 겨냥
```

**System 1 설계 선택지 → 각각 별도 실험:**

| | Exp-13a | Exp-13b | Exp-13c | Exp-13d |
|---|---------|---------|---------|---------|
| **System 2 action** | 공 선택 (discrete 3) | 공+포켓 (discrete 18) | 공+포켓 (discrete 18) | 공+포켓 (discrete 18) |
| **System 1** | Phase 1 완전 freeze | Phase 1 freeze | Phase 1 freeze | scratch (joint 학습) |
| **System 1 obs** | `[cue_xy, target_xyz, other1_xyz, other2_xyz, 6pockets]` (ball 재배열) | 위 + 비목표 포켓 마스킹 | 위 + target pocket 첫 번째로 | goal-conditioned 새 설계 |
| **Phase 1 재사용** | 완전 그대로 | 완전 그대로 (OOD: 비nearest ball[0]) | 완전 그대로 (OOD) | 없음 |
| **포켓 선택** | System 1이 자율 결정 | System 2 지정 (masking) | System 2 지정 (soft) | System 2 지정 |
| **OOD 리스크** | 낮음 (ball[0]만 바뀜) | 중간 (ball[0] + pocket 마스킹) | 중간 (pocket 순서 변경) | 없음 |
| **핵심 질문** | HRL 구조 자체가 유효한가? | masking으로 포켓 강제 가능한가? | 포켓 info 추가가 도움되는가? | Phase 1 없이 end-to-end 가능한가? |

**리워드 구조 (13a~d 공통):**
- 어느 포켓이든 들어가면 +1 (포켓 지정 여부 무관)
- 포켓 지정 준수 여부는 reward에 반영하지 않음 — System 2가 episode clear reward로 간접 학습
- **평가 지표 추가 (13b/c/d)**: `pocket_accuracy` = 지정 포켓에 실제로 들어간 비율 (reward 아님, 사후 분석용)

**실험 순서 근거:**
- 13a → Phase 1 OOD 정도가 가장 낮음. HRL 구조 자체 유효성 검증에 가장 깨끗한 조건.
- 13b/c → 포켓 지정이 실제로 System 1의 행동을 바꾸는가 (masking vs 전체 공개)
- 13d → Phase 1 pretraining 없을 때 대비 ablation

**Exp-12 vs Exp-13의 관계 (업데이트):**
Exp-12 실패 → abs_angle 불채택. Exp-13은 delta_angle + Phase 1 재배열 구조 사용.
공 선택 자유도는 abs_angle이 아닌 System 2의 명시적 target 지정 + obs 재배열로 해결.

---

### Phase 1 obs 재배열 분석 — Exp-13a/b/c 설계

Phase 1 obs: `[cue_xy, b1_xyz, b2_xyz, b3_xyz, 6pockets]` = **23-dim, delta_angle**
(훈련 시 b1 = nearest ball 순서로 정렬)

System 2가 target ball(idx=k)을 선택하면, obs를 재배열해서 target을 ball[0] 위치로 올린다.
delta_angle=0 → ball[0] 방향 겨냥이므로, Phase 1은 재배열된 obs에서 target ball을 향해 쏜다.

| Exp | System 1 obs 구성 | OOD 정도 | 핵심 변수 |
|-----|------------------|----------|----------|
| 13a | ball[0]=target, ball[1/2]=others | 낮음 (순서만 바뀜) | HRL 구조 유효성 |
| 13b | 위 + 비목표 포켓 0 마스킹 | 중간 (포켓 일부 누락) | 포켓 지정 강제 가능성 |
| 13c | 위 + target pocket 첫 번째 | 중간 (포켓 순서 변경) | 포켓 info 추가 효과 |
| 13d | goal-conditioned 새 설계 | — | Phase 1 없이 end-to-end |

**OOD 리스크:** Phase 1은 ball[0]=nearest ball로 훈련됐으나, Exp-13에서 ball[0]=target ball (≠ nearest 가능).
non-nearest ball을 겨냥할 때 Phase 1이 얼마나 잘 동작하는지가 Exp-13a의 핵심 확인 사항.

---

### 쿠션 학습으로의 확장 → Exp-14 방향

Exp-12/13이 성공하면 (직접샷만으로 3개 클리어), 다음 자연스러운 확장은 **쿠션샷 학습**이다.

**왜 쿠션이 지금보다 훨씬 어려운가:**

쿠션샷은 직접샷 대비 각도 민감도가 5~10배 높고, 경로가 완전히 달라진다. 현재 flat MLP에 각도 하나만 주고 "직접 겨냥"과 "1쿠션 후 입사각 계산"을 동시에 배우도록 기대하는 건 불가능에 가깝다.

근본 원인: **목표가 암묵적(implicit)**이라는 것.

```
현재: action = (delta_angle, speed)
      → "어떤 공을 어느 포켓에 몇 쿠션으로" 전부 angle 하나에 함축
      → 탐색 공간이 너무 넓고, 쿠션 경로는 reward 신호가 거의 없음
```

**목표를 명시적(explicit)으로 만들어야 한다:**

```
High-level (System 2, goal selection):
  어떤 공 (0~2) × 어느 포켓 (0~5) × 몇 쿠션 (0~2) → 목표 선택
  → 3 × 6 × 3 = 54가지 조합

Low-level (System 1, execution):
  obs + [목표 공 좌표, 목표 포켓 좌표, n_cushions] → (angle, speed) 실행
  → n_cushions가 obs에 없으면 "직접샷인지 쿠션샷인지" 문맥을 모름
```

**n_cushions를 obs에 추가해야 하는 이유:**
같은 (공, 포켓) 목표라도 0쿠션 경로와 1쿠션 경로는 완전히 다른 각도를 요구한다. low-level이 어떤 전략을 실행해야 하는지 알려면 `n_cushions`가 조건(context)으로 주어져야 한다.

**구현 선결 조건:**
- simulator가 벽 충돌 횟수를 trajectory에서 count해야 함 (현재 없음)
- reward: "n쿠션으로 넣었는가" 판별 로직 추가 필요

**단계적 로드맵:**

```
Exp-12   절대각도 ❌    → 5M써도 delta 2M보다 낮음. abs_angle 폐기.
Exp-13a  HRL-A         → 공 선택 (discrete 3), Phase 1 freeze, obs 재배열(target→ball[0])
                          확인: HRL 구조 자체가 flat MLP 대비 유리한가? (변수 최소)
Exp-13b  HRL-B         → 공+포켓 선택 (discrete 18), Phase 1 freeze, 비목표포켓 마스킹
                          확인: masking으로 System 1이 지정 포켓을 실제로 겨냥하는가?
Exp-13c  HRL-C         → 공+포켓 선택 (discrete 18), Phase 1 freeze, target pocket 첫 번째
                          확인: 포켓 정보 추가가 도움되는가?
Exp-13d  HRL-D         → 공+포켓 선택 (discrete 18), scratch (Phase 1 없이 joint 학습)
                          확인: Phase 1 pretraining이 없으면 얼마나 못하나? (ablation)
Exp-14   쿠션 확장     → simulator에 cushion count 추가
                          obs에 n_cushions 포함, high-level에 쿠션 선택 추가
                          확인: goal-conditioned low-level이 0쿠션/1쿠션/2쿠션을
                                각각 다른 각도 전략으로 학습하는가?
```

---

### Phase 2 실험 방향 (우선순위)

```
① Exp-10  multi-seed benchmark ✓ (완료)
          확인: SAC clear 8.4% (2-seed)는 실력. TQC/PPO는 ms=3 sparse reward에서 기대 이하.

② Exp-11  curriculum ms=5→4→3 ✓ (완료) → 43.0%/10.4%
          (+) ms=2 extension ✓ (완료) → 28.6%/1.6% 더블포켓 필수, 학습 신호 부재 확인
          확인: curriculum이 flat scratch 대비 유효 (+2.0pp clear)

③ Exp-12  absolute angle [0, 2π] ✓ (완료) → ❌ 폐기
          확인: 5M 써도 delta 2M(41.7%/8.4%)보다 낮음(37.8%/6.4%)
               abs_angle의 inductive bias 부재가 치명적. delta_angle 유지.

④ Exp-13a  HRL-A — 공 선택 (discrete 3), Phase 1 freeze, obs 재배열(target→ball[0])
           확인: HRL 구조 자체가 flat MLP보다 유리한가? (변수 최소, 가장 깨끗한 검증)
           선행 조건: Exp-10 best_model (SAC s0 or s1, ms=3, 2M)

⑤ Exp-13b  HRL-B — 공+포켓 선택 (discrete 18), Phase 1 freeze, 비목표포켓 마스킹
           확인: masking으로 System 1이 지정 포켓을 실제로 겨냥하는가?
           평가: pocket_accuracy (지정 포켓 적중률) 추가 기록

⑥ Exp-13c  HRL-C — 공+포켓 선택 (discrete 18), Phase 1 freeze, target pocket 첫 번째
           확인: 포켓 정보를 온전히 줄 때 masking 대비 차이?
           평가: pocket_accuracy 추가 기록

⑦ Exp-13d  HRL-D — 공+포켓 선택 (discrete 18), scratch (Phase 1 없이 joint)
           확인: Phase 1 pretraining 없이 end-to-end 가능한가? (13a~c 대비 ablation)
           평가: pocket_accuracy 추가 기록

⑨ Exp-14  쿠션 확장  (Exp-13 성공 후)
          확인: goal-conditioned low-level이 n_cushions 조건에 따라
               다른 각도 전략을 학습하는가?
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

## Experiment Summary Tables

### Phase 0 — algorithm benchmark

| Exp | Algo | Seeds | Pocket% (per seed) | avg | std |
|-----|------|-------|---------------------|-----|-----|
| 01 | SAC | 0/1/2 | 81.4 / 73.6 / 77.8 | **77.6%** | ±3.9pp |
| 01 | TQC | 0/1/2 | 84.6 / 80.6 / 35.6 | 66.9% | **±27pp** |
| 01 | PPO | 0/1/2 | 24.0 / 30.0 / 31.6 | 28.5% | ±4.0pp |

---

### Phase 1 — ms=5 baseline 진화

scratch에서 reward shaping까지 순서대로 비교.

| Exp | 조건 | Pocket% | Clear% | Ep Len | 비고 |
|-----|------|---------|--------|--------|------|
| 03 | SAC scratch | 60.7% | 29.4% | 4.60 | Phase 1 baseline |
| 04 | Transfer A (zero-shot) | 63.6% | 31.4% | — | 훈련 0분 |
| 05 | Transfer B (warm-start) | 61.5% | 30.4% | — | 초반↑ 후 희석 |
| 06 | pp (scratch) | **63.9%** | **33.2%** | 4.48 | ms=5 best |
| 06 | pp (Transfer A) | 64.3% | 30.2% | — | — |
| 06 | pp (Transfer B) | 64.8% | 32.2% | — | — |

---

### Phase 1 — reward shaping: ep_len 단축 시도

모두 ms=5, SAC, 1M steps. Exp-06(pp)을 기준선으로 비교.

| Exp | 변경 조건 | Pocket% | Clear% | Ep Len | 결론 |
|-----|----------|---------|--------|--------|------|
| 06 | pp=✓ (baseline) | 63.9% | 33.2% | 4.48 | — |
| 07 SAC | cb=2.0 | 62.0% | 29.2% | 4.50 | cb 효과 없음 |
| 07 TQC | cb=2.0 | 49.7% | 17.6% | 4.80 | single-seed 불안정 |
| 08a | shots_taken + cb=2.0 | 63.7% | 30.0% | 4.60 | ep_len 오히려 증가 |
| 08b | shots_taken + lr=1e-4 + gs=10 | 62.1% | 28.8% | 4.50 | worse |

---

### Phase 1 — ms × pp ablation grid

SAC, 1M steps, sp=0.1, tp=1.0. pp=✓ at ms=5는 Exp-06 참고.

| ms | pp=✗ (pocket / clear) | pp=✓ (pocket / clear) | Δ pocket | Δ clear | ep_len (pp=✗) |
|----|----------------------|----------------------|---------|---------|---------------|
| 5 | 63.6% / 32.2% (09a) | 63.9% / 33.2% (06) | +0.3pp | +1.0pp | 4.40 |
| 4 | 55.1% / 17.6% (09c) | 51.2% / 15.8% (09d) | **−3.9pp** | −1.8pp | 3.86 |
| 3 | 41.4% /  9.0% (09e) | 41.5% /  7.6% (09f) | +0.1pp | −1.4pp | 2.98 |

ep_len/ms: ms=5 → 88%, ms=4 → 96.5%, ms=3 → **99.3%** — ms가 줄수록 step 효율 악화.

---

### Phase 2 — ms=3 algorithm benchmark (Exp-10)

SAC/TQC/PPO × 3 seeds, ms=3, sp=0.1, tp=1.0, 2M steps. (SAC s42, PPO s1/s42는 실험 중 오류로 제외)

| Algo | Seeds (pocket%) | Seeds (clear%) | avg pocket | avg clear |
|------|-----------------|----------------|------------|-----------|
| **SAC** | 41.5 / 41.9 / — | 8.2 / 8.6 / — | **41.7%** | **8.4%** |
| TQC | 24.6 / 24.9 / 31.9 | 2.2 / 0.4 / 3.4 | 27.1% | 2.0% |
| PPO | 6.5 / — / — | 0.0 / — / — | ~6.5% | ~0.0% |
| Random | — | — | ~9% | ~0% |

**관찰:** SAC이 ms=3 sparse reward 환경에서 유일하게 의미 있는 학습. TQC는 top quantile drop의 overconservatism, PPO는 on-policy credit assignment 한계로 clear 거의 불가.

---

### Phase 2 — Curriculum (Exp-11) vs abs_angle (Exp-12) 비교

| Exp | 방법 | Steps | Pocket% | Clear% | 비고 |
|-----|------|-------|---------|--------|------|
| 10 | SAC delta_angle | 2M | 41.7% | 8.4% | baseline |
| **11** | **SAC curriculum ms=5→4→3** | **2M** | **43.0%** | **10.4%** | **+2.0pp clear** |
| 11+ | ms=2 extension (curriculum s4) | +500k | 28.6% | 1.6% | 더블포켓 필수, 사실상 불가 |
| 12 | SAC abs_angle (s1 only) | 5M | 37.8% | 6.4% | 2.5× 스텝 소모, 오히려 낮음 |

**Exp-12 결론:** abs_angle은 delta_angle 대비 개선 없음. 5M을 써도 delta 2M보다 낮다. nearest-ball 기준의 inductive bias가 Phase 2에서도 여전히 유효. Exp-13(HRL)에서 System 1은 delta_angle + Phase 0 그대로 사용.

---

## Experiments

### Exp-01 · Phase 0 single-ball benchmark

**목표:** SAC / PPO / TQC를 동일 조건에서 비교. horizon=1에서 각 알고리즘의 특성 파악.

**설정:** 1M steps, seed {0, 1, 42}, `n_balls=1`

> **📌 재현 시 성능 차이 원인 (2026-03 확인):**
> 초기 커밋(0f55628~85e3855)의 배치: cue y∈[0.2,0.4], target y∈[0.6,0.9] — **항상 상하 분리**, 직선샷 위주.
> MultiBallEnv 통합 커밋(89431bd)에서 배치가 통일: cue y∈[0.15,0.40], target y∈[0.30,0.85] — 측면/근거리샷 포함.
> **배치 난이도 상승**이 77.6%→42.4% 하락의 원인. horizon=1 단일샷은 초기 커밋부터 동일.
> 2026-03 재훈련(seed=42, 1M, sp=0.0): **42.4%** / random **3%** → 14× 개선 (현재 배치 기준).
> **Exp-13은 Phase 1(n_balls=3) 기반으로 설계 변경 → Phase 0 가중치 미사용.**

**결과 (당시 기록):**

| Algorithm | Pocket rate (mean ± std) | vs. random |
|-----------|--------------------------|------------|
| SAC | 77.6% ± 3.9 pp | +71 pp |
| TQC | 66.9% ± 27.2 pp | +61 pp |
| PPO | 28.5% ± 4.0 pp | +22 pp |
| Random | ~6% | — |

**관찰:**
- Horizon=1 단일샷에서 off-policy(SAC/TQC)가 압도적. PPO의 GAE는 단일 transition에서 REINFORCE로 퇴화.
- TQC는 seed 간 분산이 매우 큼 (어떤 seed는 SAC 수준, 어떤 seed는 붕괴). 분포 추정의 불안정성으로 추정.
- SAC의 best_model vs final_model 격차가 종종 큼 — 훈련 말기에 collapse 발생. EvalCallback의 best_model을 사용해야 함.

---

### Exp-02 · Phase 1a multi-ball (max_steps=∞)

**목표:** multi-ball 환경에서 SAC scratch 훈련 가능성 확인.

**설정:** SAC, 1M steps, seed=42, `n_balls=3`, `max_steps=15`

**관찰:**
- pocket 98.3% / clear 95.8% — 수치는 높지만 random agent도 40%를 기록. max_steps=15가 너무 넉넉함.
- "일단 많이 쏘면 들어간다" 전략에 가까움. horizon을 줄여 shot efficiency를 강제할 필요 → Exp-03.

---

### Exp-03 · Phase 1a multi-ball (max_steps=5)

**목표:** 타이트한 horizon으로 shot efficiency를 강제. Exp-02와 비교.

**설정:** SAC, 1M steps, seed=42, `n_balls=3`, `max_steps=5`

**관찰:**
- pocket 60.7% / clear 29.4%. 5번 안에 3개 넣으려면 샷당 평균 0.6개 이상 pocket 필요 — 훨씬 어려운 문제.
- Transfer 실험(Exp-04, 05)의 scratch baseline으로 사용.

---

### Exp-04 · Transfer A — obs-collapse zero-shot

**목표:** Phase 0에서 학습한 aiming skill이 추가 훈련 없이 multi-ball에 얼마나 transfer되는지 측정.

**방법:** `ObsCollapseWrapper`로 23-dim obs → 16-dim 축소. 매 step 가장 가까운 unpocketed ball을 "the ball"로 제시. Pretrained n_balls=1 모델(seed=0, 81.4%) 그대로 사용.

**관찰:**
- 추가 훈련 없이 scratch(Exp-03: 60.7%)를 상회 — 63.6% / 31.4%. aiming skill이 직접 전이됨.
- ball2/ball3 위치를 모르므로 interference는 피할 수 없음 — full obs 없이는 이 이상 개선 불가.

---

### Exp-05 · Transfer B — weight-copy warm-start

**목표:** Phase 0 가중치를 warm-start로 활용했을 때 Exp-03(scratch)보다 빠르게 수렴하는지 확인.

**방법:** 23-dim SAC를 새로 만들고, 첫 번째 input layer의 shared 뉴런(cue, ball1, pocket 열)에 n_balls=1 모델 가중치 복사. ball2/ball3 뉴런은 0 초기화 후 전체 fine-tune.

**관찰:**
- 학습 초반 reward가 높다가 점점 떨어지는 패턴 — weight dilution의 전형적인 증거.
  b2/b3 뉴런(0 초기화)이 학습되면서 pretrained weight을 점진적으로 덮어씀.
- 가장 충격적인 결과: **Exp-04(zero-shot)가 훈련 0분으로 B보다 더 좋음**. direct transfer가 weight-copy fine-tuning 전체를 압도.
- ep_len_mean ≈ 4.7 — 5번 step 한도를 거의 다 소진. step penalty(-0.01)가 너무 작아 urgency 없음.

---

### Exp-06 · Progressive reward shaping (pp, sp=0.1×step, tp=1.0)

**목표:** ep_len≈4.7 문제 해결. Progressive penalty로 urgency gradient를 만들고, truncation penalty로 미완료 에피소드를 명시적으로 패널티. 세 가지 훈련 방식(scratch / Transfer A / B) 모두 적용.

**변경 사항:**
- step penalty: 고정 −0.01 → progressive: step i에서 −0.1×i (step1: −0.1 ... step5: −0.5)
- truncation penalty: 없음 → **−1.0** (step limit 도달 시)
- 3-step vs 5-step clear 보상 차이: 0.2 → **0.9 (4.5×)**

**결과:**

| 조건 | Pocket% | Clear% |
|------|---------|--------|
| Exp-03 scratch (참고) | 60.7% | 29.4% |
| Exp-06 scratch | **63.9%** | **33.2%** |
| Exp-06 Transfer A | 64.3% | 30.2% |
| Exp-06 Transfer B | 64.8% | 32.2% |

**ep_len 분포 (scratch best_model, n=1000):**

| step | 2 | 3 | 4 | 5 |
|------|---|---|---|---|
| % | 1.5% | 6.4% | 11.2% | **80.9%** |

**관찰:**
- 세 조건 모두 pocket rate 3~4pp 향상. progressive penalty가 더 나은 aiming을 유도.
- Transfer B가 처음으로 A를 역전(64.8% > 64.3%) — 강화된 reward structure에서 warm-start 가치가 나타남.
- **ep_len 단축은 실패.** 80.9%가 여전히 5 step 전부 소진 — aiming 정확도 향상이지 step efficiency 향상이 아님.
- 근본 원인: step 5의 기댓값 = −0.5(penalty) + 1.0 × ~30%(pocket chance) ≈ −0.2. truncation(−1.0)보다 여전히 나으므로 agent가 5번째 step을 계속 사용.

---

### Exp-07 · clear_bonus=2.0 + SAC vs TQC

**목표:** clear_bonus = 2.0/steps_used로 빠른 클리어에 직접 보상. SAC vs TQC 비교.

**설정:** SAC·TQC 각 1회, 1M steps, seed=42, ms=5, sp=0.1(flat), tp=1.0, cb=2.0

**Reward 구조 (예시):**

| 클리어 step | ball reward | step pen | clear bonus | total |
|------------|------------|----------|-------------|-------|
| 3-step | +3.0 | −0.3 | +0.667 | **+3.37** |
| 5-step | +3.0 | −0.5 | +0.400 | **+2.90** |

**결과:**

| | SAC | TQC | Exp-06 참고 |
|---|-----|-----|-------------|
| Pocket% | 62.0% | 49.7% | 63.9% |
| Clear% | 29.2% | 17.6% | 33.2% |
| ep_len (5-step %) | 81.5% | 91.0% | 80.9% |

**관찰:**
- **clear_bonus도 ep_len 단축 실패.** 3-step vs 5-step 차이(+0.47)가 policy를 바꾸기에 부족하고, agent가 step count를 모르므로 이 차이를 활용할 수 없음.
- TQC Exp-01과 동일한 패턴 재현 — single-seed에서 나쁜 local optima에 수렴. multi-seed 없이 TQC 우위 주장 불가.
- SAC Exp-07(62.0%)이 Exp-06(63.9%)보다 낮음 — reward scale이 높아져(+3.37 max) Q-learning이 불안정해진 것으로 추정.

---

### Exp-08 · shots_taken obs ablation

**목표:** step count 정보를 obs에 추가(shots_taken/max_steps ∈ (0,1])하면 urgency 인식이 가능해져 ep_len이 줄어드는지 확인. 두 가지 variant 비교.

**설정:** SAC, 1M steps, seed=42, ms=5, n_balls=3

| variant | 추가 변경 | Pocket% | Clear% | Ep Len |
|---------|----------|---------|--------|--------|
| 08a | shots_taken + cb=2.0 (sp=0.1 flat) | 63.7% | 30.0% | 4.60 |
| 08b | shots_taken + lr=1e-4 + gs=10, cb=0 | 62.1% | 28.8% | 4.50 |
| Exp-06 (참고) | pp, cb=0 | 63.9% | 33.2% | 4.48 |

**관찰:**
- **ep_len 변화 없음 (08a: 4.60, 08b: 4.50).** shots_taken이 urgency 신호로 작동하지 않음.
  → 당구에서 최적 action은 공의 위치(obs)에만 의존하고 step count와 독립적이기 때문.
  shots_taken은 MDP를 더 완전하게(Markov) 만드는 feature이지만, 당구에서는 uninformative → network 용량 낭비.
- **08a:** cb=2.0의 높은 reward scale이 훈련 초반 critic loss explosion 유발. SAC는 gradient clipping 없음(SB3 기본).
- **08b:** gs=10 → UTD(Update-To-Data)=1.0. vanilla SAC에서 high UTD는 Q-value overestimation cascade를 심화시킴. lr=1e-4도 도움 안 됨.
- **결론:** shots_taken, clear_bonus, gradient_steps 모두 ep_len 단축에 기여 없음. ep_len은 task 구조 자체의 문제.

---

### Exp-09 · ms × pp ablation grid

**목표:** (1) pp가 ms=5 이외에도 유효한지 확인. (2) ms=3/4의 학습 가능성 탐색.

**설정:** SAC, 1M steps, seed=42, n_balls=3, sp=0.1, tp=1.0

| | ms=5 | ms=4 | ms=3 |
|---|------|------|------|
| **pp=✗** | 63.6% / 32.2% (09a) | 55.1% / 17.6% (09c) | 41.4% / 9.0% (09e) |
| **pp=✓** | 63.9% / 33.2% (Exp-06) | 51.2% / 15.8% (09d) | 41.5% / 7.6% (09f) |
| **Δ pp** | +0.3pp / +1.0pp | **−3.9pp** / −1.8pp | +0.1pp / −1.4pp |
| **ep_len (pp=✗)** | 4.40 | 3.86 | **2.98** |
| **ep_len/ms** | 88% | 96.5% | **99.3%** |

**관찰:**
- **pp는 어떤 ms에서도 유의미한 개선 없음. 폐기.**
  - ms=5: +0.3pp pocket (노이즈 수준)
  - ms=4: −3.9pp. progressive penalty 누적(−0.1−0.2−0.3−0.4 = −1.0)이 pocket reward(+1.0)와 상쇄 → 학습 신호 약화.
  - ms=3: −1.4pp clear. penalty 누적(−0.6)으로 첫 샷부터 기댓값이 0에 가까움.
- **ms를 줄일수록 ep_len/ms 비율이 상승.** agent가 lazy한 게 아니라 task 자체가 모든 step을 소모하는 구조. ms=3에서 사실상 전 에피소드가 3-step full.
- **ms=3이 Phase 2 frontier.** clear rate 9.0% — sparse reward + 매 샷이 pocket이어야 하는 고난도. reward shaping이 아닌 알고리즘 수준의 개선이 필요.

---

### Exp-10 · Phase 2 ms=3 algorithm benchmark

**목표:** SAC 단일 seed 9% clear가 재현 가능한 실력인지 확인. SAC/TQC/PPO 공정 비교.

**설정:** SAC/TQC/PPO × 3 seeds {0, 1, 42}, ms=3, sp=0.1, tp=1.0, 2M steps

**결과:**

| Algorithm | Pocket% (s0 / s1 / s42) | Clear% (s0 / s1 / s42) | avg pocket | avg clear |
|-----------|------------------------|------------------------|------------|-----------|
| **SAC** | 41.5 / 41.9 / — | 8.2 / 8.6 / — | **41.7%** | **8.4%** |
| TQC | 24.6 / 24.9 / 31.9 | 2.2 / 0.4 / 3.4 | 27.1% | 2.0% |
| PPO | 6.5 / — / — | 0.0 / — / — | ~6.5% | ~0.0% |
| Random | ~9% | ~0% | — | — |

(SAC s42: eval 단계에서 crash. PPO s1/s42: TB 디렉토리 이동 중 파일 손실로 제외)

**관찰:**
- **SAC clear 8.4% 재현 확인.** seed 간 분산이 작음 (41.5% vs 41.9%) — 학습이 안정적.
- **TQC는 clear 2.0%로 저조.** top quantile drop의 overconservatism이 sparse reward에서 역효과. Q값 추정이 지나치게 낮아지면 탐색 동기 감소.
- **PPO는 clear ≈ 0%.** on-policy 특성상 3-step credit assignment가 극도로 노이즈. 에피소드 중 한 번이라도 실패하면 모든 이전 action이 동등하게 패널티를 받음.
- **Phase 2에서는 SAC가 baseline.** TQC/PPO는 현 설정에서 ms=3 task에 부적합.

---

### Exp-11 · Curriculum ms=5 → ms=4 → ms=3

**목표:** 쉬운 task(ms=5)에서 학습한 전략이 어려운 task(ms=3)로 전이되는가. scratch 2M(Exp-10) 대비 curriculum 2M 효과 측정.

**설정:** SAC, seed=42, sp=0.1, tp=1.0. Stage 1: ms=5 1M / Stage 2: ms=4 500k / Stage 3: ms=3 500k

| Stage | ms | Steps | Pocket% | Clear% | Ep Len |
|-------|----|-------|---------|--------|--------|
| 1 | 5 | 1M | 65.1% | 33.2% | 4.65 |
| 2 | 4 | 500k | 53.6% | 21.6% | 3.86 |
| 3 | 3 | 500k | **43.0%** | **10.4%** | 2.98 |
| Exp-10 baseline | 3 | 2M scratch | 41.7% | 8.4% | — |

**Extension — Stage 4 ms=2 (500k):** pocket 28.6% / clear **1.6%**
- 2샷으로 3포켓 → 한 번 이상 더블포켓 필수. reward 신호 극히 희박. 예상대로 학습 실패.

**관찰:**
- curriculum이 같은 2M steps에서 Exp-10 대비 +1.3pp pocket / +2.0pp clear 개선.
- Stage 3(500k)만으로 Stage 1(1M)보다 낮지만, warm-start 덕분에 scratch 2M과 경쟁.
- ms=3 frontier에서는 curriculum의 benefit이 작음 — System 2 전략 자체가 부재한 것이 병목.

---

### Exp-12 · abs_angle action space ❌ 폐기

**목표:** delta_angle(nearest-ball 기준)을 absolute angle [0, 2π]로 교체 시 ball ordering 개선 여부.

**설정:** SAC, ms=3, sp=0.1, tp=1.0. 3 seeds × 5M steps 계획 → seed=0/42 5M은 crash(결과 없음), seed=42 2M은 조기종료

**결과 (완료된 run만):**

| Seed | Steps | Pocket% | Clear% | 비고 |
|------|-------|---------|--------|------|
| 42 | 2M | 32.5% | 2.0% | 조기종료 |
| 0 | 5M | — | — | crash |
| 1 | 5M | 37.8% | 6.4% | **유일하게 완료** |
| 42 | 5M | — | — | crash |
| Exp-10 SAC baseline | 2M | **41.7%** | **8.4%** | delta_angle |

**관찰:**
- abs_angle은 5M을 써도 delta 2M보다 낮다. 2.5× 스텝으로 오히려 뒤처짐.
- 원인: delta=0 → 공 직접 겨냥이라는 inductive bias가 없어져 탐색 공간 폭발. System 1(aiming)에는 이 bias가 필수.
- 공 선택 자유도 문제는 abs_angle이 아닌 System 2(HRL)가 target을 명시적으로 지정하는 방식으로 해결.
- **결론: abs_angle 폐기. Exp-13 이후 모두 delta_angle 유지.**

---

## Project Structure

```
billiards-rl/
├── simulator.py           # BilliardsEnv — pooltool gymnasium wrapper
│                          #   n_balls=1  →  Phase 0 (16-dim obs, horizon=1)
│                          #   n_balls=3  →  Phase 1a (23-dim obs, multi-shot)
├── train.py               # SAC / PPO / TQC 훈련 (--n-balls, --max-steps 등)
├── train_pretrained.py    # 전이학습: obs-collapse (A) + weight-copy (B)
├── train_curriculum.py    # SAC curriculum: ms=5 → ms=4 → ms=3
├── compare.py             # 실험 결과 비교 테이블 + 학습 곡선 PNG
├── visualize.py           # 이미지 그리드 / MP4 영상 / before-after 비교
├── benchmark.py           # 하드웨어 벤치마크 (vec_env / device 조합)
├── run_multiseed_bench.py # SAC/TQC/PPO × 3 seeds, ms=3, 2M steps
├── requirements.txt
├── setup.sh
└── logs/
    ├── experiments/       # 실험별 디렉토리 (config, results, checkpoints)
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

# TensorBoard (네이밍: {ALGO}_ms{X}_sp{Y}_s{seed})
tensorboard --logdir logs/tensorboard

# Phase 2 multi-seed benchmark 백그라운드 실행 (로그 자동 저장)
python run_multiseed_bench.py &
tail -f logs/ppo_bench.log            # 진행 확인 (bench_YYYYMMDD_HHMMSS.log)

# Curriculum training
python train_curriculum.py                              # 1M+500k+500k, seed=42
python train_curriculum.py --steps1 1000000 --steps3 1000000 --seed 0
```

---

## Setup

```bash
cd ~/Documents/billiards-rl
bash setup.sh
```

Python 3.13 (Homebrew), `.venv` 생성, 의존성 설치.
