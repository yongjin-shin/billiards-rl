# billiards-rl

Reinforcement learning project using [pooltool](https://github.com/ekiefl/pooltool) — a physics-accurate billiards simulator.

Trains SAC, PPO, and TQC agents to pocket a target ball with a single cue-ball shot.
**Multi-seed results (3 seeds):** SAC 77.6% ± 3.9pp · TQC 66.9% ± 27.2pp · PPO 28.5% ± 4.0pp (random: ~3%)

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
# Verify the simulator
python simulator.py

# Train a single algorithm (SAC by default)
python train.py --algo SAC --steps 1000000 --seed 42

# Train all three and compare
bash run_comparison.sh

# Visualize the trained agent
python visualize.py         # PNG grid of 12 shots
python visualize_video.py   # MP4 video
```

---

## Training

```bash
python train.py --algo {SAC,PPO,TQC} --steps STEPS --seed SEED
```

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `SAC` | Algorithm: `SAC`, `PPO`, or `TQC` |
| `--steps` | `1000000` | Total environment steps |
| `--seed` | `42` | Random seed — use the **same seed** across algos for a fair comparison |

Each run saves a self-contained experiment directory:

```
logs/experiments/{ALGO}_{steps}k_s{seed}_{timestamp}/
  ├── config.json       ← hyperparameters
  ├── results.json      ← pocket rates, fps, training time
  ├── best_model/       ← best checkpoint (saved by EvalCallback)
  ├── eval/             ← evaluations.npz (learning curve data)
  └── train/            ← per-worker Monitor CSVs
```

---

## Comparison

```bash
# Compare all finished experiments
python compare.py

# Filter by algorithm name
python compare.py --filter TQC

# Save plot to custom path
python compare.py --out outputs/my_comparison.png
```

Outputs a summary table and a learning-curve PNG (pocket rate vs. timesteps).

---

## Batch comparison run

```bash
bash run_comparison.sh                    # SAC + PPO + TQC, 1M steps, seed 42

# Override via environment variables
ALGOS="SAC TQC" bash run_comparison.sh
SEED=0 STEPS=500000 bash run_comparison.sh
```

---

## Project structure

```
billiards-rl/
├── simulator.py          # BilliardsEnv (gymnasium wrapper around pooltool)
├── train.py              # Training: SAC / PPO / TQC
├── compare.py            # Load experiments, print table, plot learning curves
├── visualize.py          # PNG grid of agent shots
├── visualize_video.py    # MP4 video of agent shots
├── run_comparison.sh     # Batch: train all algos → compare
├── requirements.txt
├── setup.sh              # One-time environment setup
└── logs/
    ├── experiments/      # Per-run experiment directories
    └── tensorboard/      # TensorBoard event files
```

---

## Environment

| | |
|---|---|
| **Observation** | 16-dim: `[cue_x, cue_y, target_x, target_y, p0x, p0y, …, p5x, p5y]` — all normalized to [0, 1] |
| **Action** | 2-dim continuous: `[delta_angle ∈ [-π, π], speed ∈ [0.5, 8.0]]` |
| **Reward** | +1 if target ball pocketed, else 0 (binary) |
| **Episode** | Single shot (horizon = 1) |
| **Pockets** | 6 standard pocket positions (fixed per episode) |

`delta_angle = 0` aims directly at the target ball; non-zero values produce cut shots.

---

## Algorithm comparison

All runs: 1M steps, 3 seeds (0, 1, 2), single-shot environment (horizon = 1).

| Algorithm | Type | Mean ± std | Best seed | Notes |
|-----------|------|------------|-----------|-------|
| 🥇 **SAC** | Off-policy | **77.6% ± 3.9pp** | 81.4% | Most consistent; all seeds collapse near end but best_model saved |
| 🥈 **TQC** | Off-policy, distributional | **66.9% ± 27.2pp** | 84.6% | Highest ceiling but high variance — seed 2 catastrophically failed (35.6%) |
| 🥉 **PPO** | On-policy | **28.5% ± 4.0pp** | 31.6% | Consistent but structurally disadvantaged at horizon=1 |
| Random | — | ~3% | — | Uniform random action |

> **Why off-policy wins here:** Single-shot episodes mean PPO's multi-step advantage estimation (GAE) degenerates to simple REINFORCE. SAC/TQC's replay buffer recycles every transition efficiently.
>
> **SAC vs TQC (multi-seed):** TQC reaches the highest single-seed peak (84.6%) but shows catastrophic instability on seed 2. SAC is more robust across seeds (77.6% ± 3.9pp). Both algorithms exhibit end-of-training collapse; EvalCallback's best_model checkpoint is essential.
>
> **SAC collapse pattern:** All 3 SAC seeds collapse to near-zero pocket rate at the end of training. The best checkpoint (saved mid-training) is used for evaluation.

---

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Roadmap

### Phase 0 — Single-shot benchmark ✅
SAC / PPO / TQC on single-shot env. Multi-seed (×3) statistical comparison in progress.

### Phase 1 — Multi-ball (`feature/multi-ball`)
3 target balls. Progressive difficulty within this phase:

| Level | Reward | Action | Scratch | Notes |
|-------|--------|--------|---------|-------|
| **1a** (easy) | sparse `+1` · `-0.01/step` | `[angle, speed]` | `-0.5` penalty | Entry point |
| **1b** (medium) | dense (distance-based) | `[ball_idx, angle, speed]` | `-0.5` penalty | Explicit targeting |
| **1c** (hard) | dense | `[ball_idx, angle, speed]` | episode ends | Full difficulty |

- Obs: 23-dim `[cue(2) + balls(3×2) + pocketed_flags(3) + pockets(12)]`
- Episode ends: all pocketed **or** step ≥ 15
- Longer horizon → PPO / RecurrentPPO become competitive

### Phase 2 — Cushion shots (`feature/cushion-env`)
- `BilliardsEnv(mode="direct" | "cushion" | "free")`
- `mode="cushion"`: direct pocket = reward 0; cushion-first pocket = +1
- Geometric forcing: ball placement that makes direct shots sub-optimal
- Curriculum: start `mode="direct"` (warm-start), then switch to `mode="cushion"`

### Phase 3 — Multi-ball + cushion
- Combine Phase 1 + 2: 3 balls, agent chooses when to use cushion
- Algorithm focus: DreamerV3 / RecurrentPPO (longer horizon + planning)

### Phase 4 — Self-play (`feature/self-play`)
- Two agents alternate shots (full game format)
- Self-play PPO or MAPPO
- Opponent modelling via recurrent policy
