# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).
진행 상황과 관찰을 기록하는 실험 노트.

---

## Roadmap

```
Phase 0  Single-ball aiming
  [x] Exp-01  SAC / PPO / TQC benchmark (multi-seed)

Phase 1  Multi-ball clearing
  [x] Exp-02  SAC from scratch — max_steps=15  →  너무 느슨함, step 낭비
  [x] Exp-03  SAC from scratch — max_steps=5   →  pocket 60.7%, clear 29.4%
  [x] Exp-04  Transfer A · obs-collapse zero-shot  →  pocket 63.6%, clear 31.4% (0 min!)
  [x] Exp-05  Transfer B · weight-copy warm-start  →  pocket 61.5%, clear 30.4%

  [x] Exp-06  Progressive penalty (sp=0.1×step, tp=1.0) → scratch 63.9% / A 64.3% / B 64.8%

Phase 2  (미정)
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
| **Reward** | +1.0 per ball pocketed · −step_penalty (flat) or −step_penalty×i (progressive) per step · −0.5 for scratch · −trunc_penalty if truncated |
| **Episode** | 공 3개 전부 pocketed OR step ≥ max_steps |
| **Ball-in-hand** | scratch 시 큐볼을 임의 위치에 재배치 |

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

**관찰:**
- 세 조건 모두 pocket rate 3~4pp 향상. progressive penalty가 reward signal을 실질적으로 개선함.
- **Transfer B가 처음으로 A를 역전 (64.8% > 64.3%).** Exp-05에서는 A(zero-shot)가 B(trained)보다 좋았는데, 강화된 reward structure에서는 warm-start training이 추가 가치를 만들어냄.
- Clear rate도 모든 조건에서 향상 (+2~4pp) — shot efficiency 개선의 직접적인 증거.
- Transfer B의 weight dilution 문제가 완전히 해소되지는 않았지만, 충분히 강한 penalty 덕분에 최종 수렴점이 개선됨.
- Training time이 더 늘어난 것(~36분 vs ~25분)은 episode가 짧아지면서 env reset 횟수가 늘고 오버헤드 증가 때문으로 추정.

---

## Experiments — Planned

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
