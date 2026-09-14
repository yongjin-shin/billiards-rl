# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).

---

## Quick Status

| | |
|---|---|
| **Current status** | Deterministic SSM World Model (feature/wm-ssm) in progress |
| **Next experiment** | Review Exp-17 HRL direction after SSM results |
| **Exp-16 vanilla** | Phase 1: pocket 62.2% ≈ SAC 63.6% (p=0.49) / Phase 0: pocket 56.9% ≈ SAC 55.3% (p=0.62) |

→ Experiment log: [experiments.md](experiments.md) | Next plan: [roadmap.md](roadmap.md)

---

## Environments

Both environments share the same 2-dim continuous action space:

| Dim | Range | Description |
|-----|-------|------|
| `delta_angle` | [−π, π] | Angle offset relative to nearest unpocketed ball direction (0 = aim directly) |
| `speed` | [0.5, 8.0] | Cue ball strike speed (m/s) |

### Phase 0 — single-ball (`n_balls=1`)

| | |
|---|---|
| **Observation** | 16-dim: `[cue_xy, ball_xy, p0~p5_xy]` |
| **Reward** | +1 pocketed, 0 otherwise |
| **Episode** | Always ends after 1 step |
| **Ball placement** | cue y∈[0.15,0.40] / ball y∈[0.30,0.85] (current)<br>cue y∈[0.20,0.40] / ball y∈[0.60,0.90] (legacy — original Exp-01 conditions) |

### Phase 1 — multi-ball (`n_balls=3`)

| | |
|---|---|
| **Observation** | 23-dim: `[cue_xy, b1_xyz, b2_xyz, b3_xyz, p0~p5_xy]` (flag = pocketed status) |
| **Reward** | +1.0 per ball pocketed · −step_penalty×i (progressive) / flat · −0.5 scratch · −trunc_penalty if truncated |
| **Episode** | All 3 balls pocketed OR step ≥ max_steps |
| **Ball-in-hand** | Cue ball repositioned after scratch |

---

## Project Structure

```
billiards-rl/
├── simulator.py          # Core physics environment (Gymnasium-compatible)
├── train.py              # Main SAC training entry point
├── train_curriculum.py   # Curriculum training (multi-stage ms5→ms4→ms3)
├── benchmark.py          # Evaluate and benchmark trained models
├── compare.py            # Compare results across experiments
├── logger.py             # Shared logging utilities
├── exp16_wm/             # Current experiment: world model critic
│   ├── sac.py            # SAC implementation
│   ├── networks.py       # Actor/Critic network architectures
│   ├── buffer.py         # Replay buffer
│   └── train.py          # Exp16-specific training loop
├── world_model/          # VAE+LSTM world model (exp16 dependency)
│   ├── model.py          # VAE / VAE-LSTM architecture
│   └── train_vae.py      # World model training
└── logs/experiments/     # All experiment outputs (auto-named)
```

---

## Setup

```bash
cd ~/Documents/billiards-rl
bash setup.sh              # Create Python 3.13 venv + install dependencies
```

## Key Commands

```bash
# Train
python train.py --n-balls 3 --max-steps 5 --step-penalty 0.1 --trunc-penalty 1.0
python train.py --n-balls 3 --max-steps 3 --steps 2000000 --seed 0
python train_curriculum.py --seed 0

# Compare experiments
python compare.py
python compare.py --filter multi3
python compare.py --list

# Visualize
python visualize.py --n-balls 3 --model <exp_dir>/best_model/best_model
python visualize.py --n-balls 3 --mode video --model <path>

# World model training
python world_model/train_vae.py

# TensorBoard
tensorboard --logdir logs/tensorboard
```
