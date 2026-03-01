# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).

Two environments are implemented as progressive phases:

| Phase | Task | Best result | Random baseline |
|-------|------|-------------|-----------------|
| **0 — single-ball** | Pocket one ball in one shot | SAC/TQC ~77–82% | ~6% |
| **1a — multi-ball** | Pocket 3 balls in ≤ 5 shots | SAC ~98% pocket, 95% clear | ~17% |

---

## Setup

```bash
cd ~/Documents/billiards-rl
bash setup.sh
```

Installs Python 3.13 via Homebrew, creates `.venv`, and installs all dependencies.

---

## Environments

Both environments share a 2-dim continuous action space:

| Dim | Range | Meaning |
|-----|-------|---------|
| `delta_angle` | [-π, π] | Aim offset from nearest unpocketed ball (0 = straight at it) |
| `speed` | [0.5, 8.0] | Cue ball strike speed (m/s) |

### Phase 0 — single-ball (`n_balls=1`)

One target ball, one shot per episode.

| | |
|---|---|
| **Observation** | 16-dim: `[cue_x, cue_y, ball_x, ball_y, p0x,p0y, …, p5x,p5y]` normalized to [0,1] |
| **Reward** | +1 if ball pocketed, else 0 |
| **Episode** | Always terminates after 1 shot |

### Phase 1a — multi-ball (`n_balls=3`)

Three target balls, up to `max_steps` shots per episode.

| | |
|---|---|
| **Observation** | 23-dim: `[cue_x, cue_y, b1x,b1y,b1_flag, b2x,b2y,b2_flag, b3x,b3y,b3_flag, p0x,p0y,…,p5x,p5y]` |
| **Reward** | +1.0 per ball pocketed · −0.01 per step · −0.5 for scratch |
| **Episode** | Ends when all 3 balls pocketed OR `step ≥ max_steps` |
| **Ball-in-hand** | On scratch, cue ball respawns at a random valid position |

---

## Training

```bash
python train.py --algo {SAC,PPO,TQC} [--n-balls {1,3}] [--max-steps N] [--steps N] [--seed N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `SAC` | `SAC`, `PPO`, or `TQC` (TQC requires `sb3-contrib`) |
| `--n-balls` | `1` | Phase 0 (`1`) or Phase 1a (`3`) |
| `--max-steps` | `5` | Episode horizon for multi-ball (ignored for `n_balls=1`) |
| `--steps` | `1000000` | Total training timesteps |
| `--seed` | `42` | Random seed — use the same seed across algos for fair comparison |

Each run produces a self-contained experiment directory:

```
logs/experiments/{ALGO}_{steps}k_s{seed}[_multi{N}_ms{M}]_{timestamp}/
  ├── config.json      ← all hyperparameters
  ├── results.json     ← pocket rate, clear rate, fps, training time
  ├── best_model/      ← best checkpoint (EvalCallback)
  ├── eval/            ← evaluations.npz (learning curve)
  └── train/           ← per-worker Monitor CSVs
```

### Phase 0 results (1M steps, seed 42)

| Algorithm | Pocket rate | vs. random |
|-----------|-------------|------------|
| SAC | ~77–82% | +71–76 pp |
| TQC | ~67–82% | +61–76 pp |
| PPO | ~29% | +23 pp |

> **Why off-policy wins at horizon=1:** GAE degenerates to REINFORCE with a single transition. SAC/TQC's replay buffer recycles every step efficiently.

### Phase 1a results (1M steps, seed 42, max_steps=5)

| Algorithm | Pocket rate | Clear rate (all 3) | vs. random |
|-----------|-------------|-------------------|------------|
| SAC | ~98% | ~95% | +81 pp |

---

## Transfer learning (Phase 0 → Phase 1a)

Hypothesis: a model that already knows how to aim (Phase 0) should learn multi-ball play faster than starting from scratch.

Two strategies are implemented in `train_pretrained.py`:

**Strategy A — obs-collapse** (zero-shot baseline, no training)

`ObsCollapseWrapper` collapses the 23-dim obs to 16-dim by always feeding the nearest unpocketed ball to the pretrained model. The model pockets balls one at a time, never seeing ball2/ball3. This measures how much raw aiming skill transfers without any additional training. Fine-tuning is possible but has a hard ceiling — the model cannot learn multi-ball strategy because ball2/ball3 are invisible.

**Strategy B — weight-copy** (warm-start, then fine-tune)

Copies pretrained weights into the shared input neurons of a fresh 23-dim SAC (cue, ball1, pocket columns); zeros out new ball2/ball3 neurons. Fine-tunes on the full multi-ball env. The model starts with working aiming skills and adapts to multi-ball strategy freely.

```bash
# Strategy A: zero-shot eval
python train_pretrained.py \
    --strategy obs-collapse \
    --pretrained logs/experiments/SAC_1000k_s42_.../best_model/best_model \
    --eval-only

# Strategy B: warm-start → fine-tune (1M steps)
python train_pretrained.py \
    --strategy weight-copy \
    --pretrained logs/experiments/SAC_1000k_s42_.../best_model/best_model \
    --steps 1000000 --seed 42
```

| | Strategy A | Strategy B | Scratch (baseline) |
|---|---|---|---|
| **Obs space** | 16-dim (collapsed) | 23-dim (full) | 23-dim (full) |
| **Init** | pretrained weights | pretrained weights (partial) | random |
| **Training** | none (or fine-tune w/ ceiling) | full fine-tune | full |
| **Goal** | measure zero-shot transfer | faster convergence? | comparison target for B |

---

## Visualization & analysis

### Visualize

```bash
# Image grid (12 episodes)
python visualize.py --n-balls 1                              # random agent
python visualize.py --n-balls 3 --model <exp_dir>/best_model/best_model  # trained

# MP4 video
python visualize.py --n-balls 1 --mode video --model <path>

# Before/after comparison video
python visualize.py --mode compare --before before.mp4 --after after.mp4
```

### Compare experiments

```bash
python compare.py                        # all finished experiments
python compare.py --filter multi3        # only multi-ball
python compare.py --filter transfer      # only transfer experiments
python compare.py --out my_plot.png
```

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Project structure

```
billiards-rl/
├── simulator.py          # BilliardsEnv — pooltool gymnasium wrapper
│                         #   n_balls=1  →  Phase 0 (16-dim obs, horizon=1)
│                         #   n_balls=3  →  Phase 1a (23-dim obs, multi-shot)
├── train.py              # Train SAC / PPO / TQC
├── train_pretrained.py   # Transfer learning (obs-collapse + weight-copy)
├── compare.py            # Summary table + learning curve plots
├── visualize.py          # Image grid / MP4 video / before-after compare
├── benchmark.py          # Multi-seed benchmark runner
├── run_comparison.sh     # Batch: train all algos → compare
├── requirements.txt
├── setup.sh
└── logs/
    ├── experiments/      # Per-run experiment directories
    └── tensorboard/      # TensorBoard event files
```

---

## Roadmap

- [x] Phase 0: single-ball benchmark (SAC / PPO / TQC, multi-seed)
- [x] Phase 1a: multi-ball env (n\_balls=3, max\_steps=5)
- [x] Unified visualizer (image / video / compare mode)
- [x] Transfer learning code (obs-collapse + weight-copy)
- [ ] Transfer learning experiments (run Strategy A zero-shot + Strategy B vs scratch)
- [ ] Phase 1b: cushion/bank shots (extended action space)
- [ ] Phase 2: self-play / full 8-ball game
- [ ] DreamerV3 (model-based planning)
