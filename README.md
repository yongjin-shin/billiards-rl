# billiards-rl

Reinforcement learning project using [pooltool](https://github.com/ekiefl/pooltool) — a physics-accurate billiards simulator.

**Phase 0 (single-ball):** SAC/TQC → ~77–82% pocket rate (random baseline: ~6%)
**Phase 1a (multi-ball, 3 balls, max\_steps=5):** SAC → ~98% pocket rate, 95% clear rate (random baseline: ~17%)

---

## Setup

```bash
cd ~/Documents/billiards-rl
bash setup.sh
```

Installs Python 3.13 via Homebrew, creates `.venv`, and installs all dependencies.

---

## Quick start

```bash
# Verify the simulator (single-ball + multi-ball sanity tests)
python simulator.py

# Train (SAC, single-ball, 1M steps)
python train.py --algo SAC --steps 1000000 --seed 42

# Train multi-ball (3 balls, max 5 shots per episode)
python train.py --algo SAC --n-balls 3 --max-steps 5 --steps 1000000 --seed 42

# Visualize (image grid)
python visualize.py --n-balls 1                           # single-ball, random agent
python visualize.py --n-balls 3 --model <path>            # multi-ball, trained agent
python visualize.py --n-balls 1 --mode video --model <path>  # MP4 video
python visualize.py --mode compare --before before.mp4 --after after.mp4  # side-by-side

# Compare finished experiments
python compare.py
python compare.py --filter multi3   # only multi-ball experiments
```

---

## Training

```bash
python train.py --algo {SAC,PPO,TQC} --steps STEPS --seed SEED [--n-balls {1,3}] [--max-steps N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `SAC` | Algorithm: `SAC`, `PPO`, or `TQC` |
| `--steps` | `1000000` | Total environment steps |
| `--seed` | `42` | Random seed |
| `--n-balls` | `1` | `1` = single-shot (Phase 0), `3` = multi-ball (Phase 1a) |
| `--max-steps` | `5` | Max shots per episode (multi-ball only) |

Each run saves a self-contained experiment directory:

```
logs/experiments/{ALGO}_{steps}k_s{seed}[_multi{N}_ms{M}]_{timestamp}/
  ├── config.json       ← hyperparameters (includes n_balls, max_steps)
  ├── results.json      ← pocket rates, clear rate, fps, training time
  ├── best_model/       ← best checkpoint (saved by EvalCallback)
  ├── eval/             ← evaluations.npz (learning curve data)
  └── train/            ← per-worker Monitor CSVs
```

---

## Transfer learning

Transfers a pretrained n\_balls=1 SAC model to the n\_balls=3 multi-ball environment. Two strategies:

**Strategy A — obs-collapse** (zero-shot baseline)

The `ObsCollapseWrapper` collapses the 23-dim multi-ball obs to 16-dim by always presenting the nearest unpocketed ball as "the ball". The pretrained model pockets balls one at a time without ever seeing ball2/ball3. No fine-tuning needed — evaluates how much raw aiming skill transfers.

**Strategy B — weight-copy** (warm-start fine-tuning)

Builds a fresh SAC with 23-dim obs, copies pretrained weights into shared input neurons (cue, ball1, pockets columns), and zeros out the new ball2/ball3 input neurons. Fine-tuned from this warm start so the model retains aiming skill while freely learning multi-ball strategy.

```bash
# Strategy A: zero-shot eval
python train_pretrained.py \
    --strategy obs-collapse \
    --pretrained logs/experiments/SAC_1000k_s42_.../best_model/best_model \
    --eval-only

# Strategy B: warm-start fine-tune (1M steps)
python train_pretrained.py \
    --strategy weight-copy \
    --pretrained logs/experiments/SAC_1000k_s42_.../best_model/best_model \
    --steps 1000000 --seed 42
```

| Strategy | Obs | Training | What it measures |
|----------|-----|----------|-----------------|
| A (obs-collapse) | 16-dim (collapsed) | none / fine-tune | Zero-shot aiming transfer; ceiling limited by info bottleneck |
| B (weight-copy) | 23-dim (full) | warm-start fine-tune | Faster convergence vs scratch |
| Scratch baseline | 23-dim (full) | from random init | Baseline to compare B against |

---

## Comparison

```bash
python compare.py                        # all finished experiments
python compare.py --filter multi3        # only multi-ball runs
python compare.py --filter transfer      # only transfer experiments
python compare.py --out my_plot.png
```

Prints a summary table and saves a learning-curve PNG.

---

## Project structure

```
billiards-rl/
├── simulator.py          # BilliardsEnv — gymnasium wrapper around pooltool
│                         #   n_balls=1 → Phase 0 (single-shot, 16-dim obs)
│                         #   n_balls=3 → Phase 1a (multi-ball, 23-dim obs)
├── train.py              # Train SAC / PPO / TQC; supports --n-balls, --max-steps
├── train_pretrained.py   # Transfer learning: obs-collapse (A) + weight-copy (B)
├── compare.py            # Load experiments → summary table + learning curves
├── visualize.py          # Unified visualizer: image grid / MP4 video / compare
├── benchmark.py          # Multi-seed benchmark runner
├── run_comparison.sh     # Batch: train SAC + PPO + TQC → compare
├── requirements.txt
├── setup.sh
└── logs/
    ├── experiments/      # Per-run experiment directories
    └── tensorboard/      # TensorBoard event files
```

---

## Environment

### Phase 0 — single-ball (n\_balls=1)

| | |
|---|---|
| **Observation** | 16-dim: `[cue_x, cue_y, ball_x, ball_y, p0x,p0y, …, p5x,p5y]` normalized to [0,1] |
| **Action** | 2-dim: `[delta_angle ∈ [-π,π], speed ∈ [0.5,8.0]]` |
| **Reward** | +1 if ball pocketed, else 0 |
| **Episode** | Single shot (horizon = 1) |

### Phase 1a — multi-ball (n\_balls=3)

| | |
|---|---|
| **Observation** | 23-dim: `[cue_x, cue_y, b1x,b1y,b1_flag, b2x,b2y,b2_flag, b3x,b3y,b3_flag, p0x,p0y,…,p5x,p5y]` |
| **Action** | 2-dim: same as Phase 0 — `delta_angle` is offset from nearest unpocketed ball direction |
| **Reward** | +1.0 per ball pocketed · −0.01 per step · −0.5 for scratch |
| **Episode** | Ends when all 3 balls pocketed OR step ≥ max\_steps (default 5) |
| **Ball-in-hand** | On scratch, cue ball respawns at a random valid position |

`delta_angle = 0` always aims at the nearest unpocketed ball; non-zero values produce cut shots.

---

## Algorithm comparison (Phase 0, single-ball, 1M steps, seed 42)

| Algorithm | Pocket rate | Random baseline | Notes |
|-----------|-------------|-----------------|-------|
| **SAC** | ~77–82% | ~6% | Strong; can collapse near end — use `best_model` |
| **TQC** | ~67–82% | ~6% | Distributional; higher variance across seeds |
| **PPO** | ~29% | ~6% | GAE degenerates at horizon=1; structurally disadvantaged |

> **Why off-policy wins at horizon=1:** PPO's multi-step advantage estimation degenerates to REINFORCE with a single transition. SAC/TQC's replay buffer recycles every step efficiently.

---

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Roadmap

- [x] Phase 0: single-ball SAC/PPO/TQC benchmark
- [x] Phase 1a: multi-ball env (n\_balls=3, max\_steps=5)
- [x] Unified visualizer (image / video / compare mode)
- [ ] Transfer learning experiment: obs-collapse zero-shot eval (Strategy A)
- [ ] Transfer learning experiment: weight-copy warm-start vs scratch (Strategy B)
- [ ] Phase 1b: cushion shots (richer action space — bank angle)
- [ ] Phase 2: self-play / opponent modelling (full 8-ball game)
- [ ] DreamerV3 for model-based planning
