# billiards-rl

Reinforcement learning on a physics-accurate billiards simulator ([pooltool](https://github.com/ekiefl/pooltool)).

---

## Quick Status

| | |
|---|---|
| **Current status** | Exp-16, teacher-free phase (2026-10-16 ~ 19, `feature/exp16-teacher-free`, tag `exp16-teacherfree-2026-10-19`): the made head alone chooses pair and action from simulator outcomes of candidates around each pair's aim, no teacher labels. Best: **made head ensemble of 11 runs 3.088 shots/game** (single runs 3.130, training-run sd 0.025; made-head val clean 89.9 → 90.9%). End-to-end (one network learns aim + made, no frozen aim encoder) is close: 3.18 single, 3.11–3.12 as an ensemble, and works with almost no aim target (λ = 0.01) but collapses with none (λ = 0: the aim drifts and inference candidates miss the window). Runs on AWS Spot (c7a.16xlarge) with checkpoint resume — summary table in experiments.md ("정리: teacher-free · 클라우드 단계") |
| **Next experiment** | S series: replace the ghost-ball formula by a simulator scan (360°, 0.05° steps) that finds every pair's success-window centre, use it as the end-to-end aim target, then loop (simulate around the model's own aim, retrain) — see roadmap.md |
| **Behaviour cloning, policy alone (3 seeds)** | shots to clear 3 balls: BC GNN 4.14 ± 0.14 → + made head verify/refine 3.90 ± 0.21 → + encoder depth 4 **3.55 ± 0.10** → + learned-aim cascade (4 frozen aim encoders) **3.43** vs 3.60 control (2000 games × 5 seeds, 2026-10-16); teacher 2.98 — details in experiments.md (E2–E11) |
| **Clear-speed task (seed 0)** | shots to clear 3 balls: vanilla SAC 7.89 / SAC low-entropy 7.39 / random search 2.50 / geometry search planner 2.24 |
| **Exp-16 vanilla** | Phase 1: pocket 62.2% ≈ SAC 63.6% (p=0.49) / Phase 0: pocket 56.9% ≈ SAC 55.3% (p=0.62) |
| **Exp-16 WM critic (4 seeds)** | pocket vanilla 62.95% / traj 63.47% / rssm 62.19% — no significant difference (ANOVA p=0.58) |

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
| **Clear-speed setting** (2026-10-09) | `--max-steps 30 --step-penalty 1.0 --trunc-penalty 0`: game ends only when all balls are in (30 shots is a safety cap, bootstrapped), reward +1/ball −1/shot (−0.5 scratch) → return = 3 − shots |

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
