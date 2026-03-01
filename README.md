# billiards-rl

Reinforcement learning project using [pooltool](https://github.com/ekiefl/pooltool) — a physics-accurate billiards simulator.

Trains SAC, PPO, and TQC agents to pocket a target ball with a single cue-ball shot.
**Current best: TQC / SAC → ~81%+ pocket rate** (random baseline: ~6%)

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

| Algorithm | Type | Best pocket rate | Notes |
|-----------|------|-----------------|-------|
| **TQC** | Off-policy, distributional | TBD | Expected best — drops top quantiles to reduce overestimation |
| **SAC** | Off-policy | ~81% | Strong baseline; can collapse at end of training (use best_model) |
| **PPO** | On-policy | ~29% | Structurally disadvantaged at horizon=1: GAE provides no signal |
| Random | — | ~6% | Uniform random action |

> **Why off-policy wins here:** Single-shot episodes mean PPO's multi-step advantage estimation (GAE) degenerates to simple REINFORCE. SAC/TQC's replay buffer recycles every transition efficiently.

---

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Next steps

- [ ] Multi-ball environment (3–7 balls, ordered pocketing)
- [ ] Multi-shot episodes → longer horizon → PPO / RecurrentPPO become competitive
- [ ] Cushion shots (bank shots) via richer action space
- [ ] Self-play / opponent modelling for full 8-ball game
- [ ] DreamerV3 for model-based planning
