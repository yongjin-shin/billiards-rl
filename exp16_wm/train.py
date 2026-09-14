"""
exp16_wm/train.py — Training script for Exp-16 (VanillaSAC and WMSAC).

Usage:
    python -m exp16_wm.train --agent vanilla --seed 0
    python -m exp16_wm.train --agent wm --seed 0

Defaults match root train.py SAC preset:
    lr=3e-4, buffer_size=200k, batch_size=512, tau=0.005, gamma=0.99,
    train_freq=1, gradient_steps=1, learning_starts=5000
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import wandb
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv, TRAJ_MAX_EVENTS, TRAJ_EVENT_DIM
from train import _tee_output
from exp16_wm.buffer import ReplayBuffer, TrajectoryReplayBuffer
from exp16_wm.sac import VanillaSAC, WMSAC


ACTION_DIM = 2
ACT_LOW    = np.array([-np.pi, 0.5],  dtype=np.float32)
ACT_HIGH   = np.array([ np.pi, 8.0],  dtype=np.float32)


def get_obs_dim(n_balls: int) -> int:
    # 2(cue) + n_balls*(2 if n_balls==1 else 3)(ball pos [+ pocketed flag]) + 12(pockets)
    return 2 + n_balls * (2 if n_balls == 1 else 3) + 12


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",           type=str,   default="vanilla",
                   choices=["vanilla", "wm"])
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--n-envs",          type=int,   default=10)
    p.add_argument("--total-steps",     type=int,   default=1_000_000)
    p.add_argument("--buffer-size",     type=int,   default=200_000)
    p.add_argument("--batch-size",      type=int,   default=512)
    p.add_argument("--learning-starts", type=int,   default=5_000)
    p.add_argument("--train-freq",      type=int,   default=1)
    p.add_argument("--gradient-steps",  type=int,   default=1)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--tau",             type=float, default=0.005)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--eval-freq",       type=int,   default=10_000)
    p.add_argument("--eval-episodes",   type=int,   default=50)
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--wandb-project",   type=str,   default="billiards-rl-exp16")
    p.add_argument("--no-wandb",        action="store_true")
    # env params
    p.add_argument("--n-balls",         type=int,   default=1, choices=[1, 3])
    p.add_argument("--max-steps",       type=int,   default=5)
    p.add_argument("--step-penalty",    type=float, default=0.01)
    p.add_argument("--trunc-penalty",   type=float, default=0.0)
    # WM-only
    p.add_argument("--wm-coef",         type=float, default=1.0)
    return p.parse_args()


# ──────────────────────────────────────────────
# Build agent / buffer
# ──────────────────────────────────────────────

def build_agent(args):
    obs_dim = get_obs_dim(args.n_balls)
    if args.agent == "vanilla":
        return VanillaSAC(
            obs_dim    = obs_dim,
            action_dim = ACTION_DIM,
            act_low    = ACT_LOW,
            act_high   = ACT_HIGH,
            lr         = args.lr,
            tau        = args.tau,
            gamma      = args.gamma,
            device     = args.device,
        )
    else:
        return WMSAC(
            obs_dim    = obs_dim,
            action_dim = ACTION_DIM,
            act_low    = ACT_LOW,
            act_high   = ACT_HIGH,
            max_events = TRAJ_MAX_EVENTS,
            event_dim  = TRAJ_EVENT_DIM,
            wm_coef    = args.wm_coef,
            lr         = args.lr,
            tau        = args.tau,
            gamma      = args.gamma,
            device     = args.device,
        )


def build_buffer(args):
    obs_dim = get_obs_dim(args.n_balls)
    if args.agent == "vanilla":
        return ReplayBuffer(obs_dim, ACTION_DIM, args.buffer_size)
    else:
        return TrajectoryReplayBuffer(
            obs_dim, ACTION_DIM, TRAJ_MAX_EVENTS, TRAJ_EVENT_DIM, args.buffer_size
        )


# ──────────────────────────────────────────────
# Experiment directory
# ──────────────────────────────────────────────

def make_exp_dir(args) -> str:
    ts      = time.strftime("%Y-%m-%d@%H%M")
    env_tag = f"_multi{args.n_balls}_ms{args.max_steps}" if args.n_balls > 1 else ""
    name    = f"exp16_{args.agent}{env_tag}_s{args.seed}_{ts}"
    path = os.path.join("logs", "experiments", name)
    os.makedirs(os.path.join(path, "eval"),       exist_ok=True)
    os.makedirs(os.path.join(path, "best_model"), exist_ok=True)
    return path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def train(args):
    exp_dir = make_exp_dir(args)
    with _tee_output(os.path.join(exp_dir, "train.log")):
        _train_inner(args, exp_dir)


def _train_inner(args, exp_dir):
    # ── reproducibility ───────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── config ────────────────────────────────
    config = {
        "agent":           args.agent,
        "seed":            args.seed,
        "n_envs":          args.n_envs,
        "total_steps":     args.total_steps,
        "buffer_size":     args.buffer_size,
        "batch_size":      args.batch_size,
        "learning_starts": args.learning_starts,
        "gradient_steps":  args.gradient_steps,
        "lr":              args.lr,
        "tau":             args.tau,
        "gamma":           args.gamma,
        "eval_freq":       args.eval_freq,
        "eval_episodes":   args.eval_episodes,
        "device":          args.device,
        "n_balls":         args.n_balls,
        "max_steps":       args.max_steps,
        "step_penalty":    args.step_penalty,
        "trunc_penalty":   args.trunc_penalty,
        "wm_coef":         args.wm_coef,
        "exp_dir":         exp_dir,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── wandb ─────────────────────────────────
    run_name = os.path.basename(exp_dir)
    if not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = config,
            tags    = [f"agent:{args.agent}", f"seed:{args.seed}"],
        )

    W = 58
    obs_dim = get_obs_dim(args.n_balls)
    print(f"\n{'═' * W}")
    print(f"  Exp-16  agent={args.agent}  seed={args.seed}")
    print(f"  env: n_balls={args.n_balls}  max_steps={args.max_steps}"
          f"  step_pen={args.step_penalty}  trunc_pen={args.trunc_penalty}")
    print(f"  obs_dim={obs_dim}  action_dim={ACTION_DIM}")
    print(f"  total_steps={args.total_steps:,}  n_envs={args.n_envs}"
          f"  lr={args.lr}  gamma={args.gamma}")
    print(f"  buffer={args.buffer_size:,}  batch={args.batch_size}"
          f"  learning_starts={args.learning_starts:,}")
    print(f"  eval_freq={args.eval_freq:,}  eval_episodes={args.eval_episodes}")
    print(f"  exp_dir: {exp_dir}")
    print(f"{'═' * W}")

    # ── environments ──────────────────────────
    traj_in_info = (args.agent == "wm")

    print(f"  [1/4] Spawning {args.n_envs} subproc envs ...", end=" ", flush=True)

    def _make_env():
        e = BilliardsEnv(
            n_balls       = args.n_balls,
            max_steps     = args.max_steps,
            step_penalty  = args.step_penalty,
            trunc_penalty = args.trunc_penalty,
        )
        e.trajectory_in_info = traj_in_info
        return e

    vec_env  = SubprocVecEnv([_make_env] * args.n_envs)
    vec_env.seed(args.seed)
    print("done")

    print(f"  [2/4] Creating eval env ...", end=" ", flush=True)
    eval_env = BilliardsEnv(
        n_balls       = args.n_balls,
        max_steps     = args.max_steps,
        step_penalty  = args.step_penalty,
        trunc_penalty = args.trunc_penalty,
    )
    eval_env.reset(seed=args.seed)
    print("done")

    # ── agent & buffer ────────────────────────
    print(f"  [3/4] Building {args.agent} agent ...", end=" ", flush=True)
    agent  = build_agent(args)
    n_params = sum(p.numel() for p in agent.actor.parameters()) + \
               sum(p.numel() for p in agent.critic.parameters())
    print(f"done  ({n_params:,} params  device={args.device})")

    print(f"  [4/4] Allocating replay buffer ({args.buffer_size:,} × {obs_dim}+{ACTION_DIM}) ...", end=" ", flush=True)
    buffer = build_buffer(args)
    print("done")

    # re-seed so buffer sampling is isolated from setup RNG consumption
    random.seed(args.seed)
    np.random.seed(args.seed)

    obs = vec_env.reset()

    best_pocket_rate = 0.0
    best_mean_reward = float("-inf")
    total_steps      = 0
    last_eval_step   = 0
    t_start          = time.time()
    eval_timesteps: list = []
    eval_results:   list = []

    # accumulate train metrics between evals
    train_metrics_acc: dict = {}
    train_metrics_cnt: int  = 0

    print(f"{'─' * W}")
    print(f"  Warmup: filling buffer with {args.learning_starts:,} random steps ...")
    _last_warmup_print = 0

    while total_steps < args.total_steps:
        # ── collect ────────────────────────────────────────────────────────
        if total_steps < args.learning_starts:
            actions = np.array([vec_env.action_space.sample()
                                for _ in range(args.n_envs)])
        else:
            actions = agent.act_batch(obs)

        next_obs, rewards, dones, infos = vec_env.step(actions)
        total_steps += args.n_envs

        # stored_next: only done envs need terminal_observation
        if dones.any():
            stored_next = next_obs.copy()
            timeouts    = np.zeros(args.n_envs, dtype=bool)
            for i in np.where(dones)[0]:
                stored_next[i] = infos[i]["terminal_observation"]
                timeouts[i]    = infos[i].get("TimeLimit.truncated", False)
        else:
            stored_next = next_obs
            timeouts    = np.zeros(args.n_envs, dtype=bool)

        if args.agent == "wm":
            for i in range(args.n_envs):
                buffer.add(
                    obs[i], actions[i], rewards[i], stored_next[i], dones[i],
                    timeout  = bool(timeouts[i]),
                    h_real   = infos[i].get("h_real",   np.zeros(
                                    (TRAJ_MAX_EVENTS, TRAJ_EVENT_DIM), dtype=np.float32)),
                    traj_len = infos[i].get("traj_len", 0),
                )
        else:
            buffer.add_batch(obs, actions, rewards.astype(np.float32),
                             stored_next, dones, timeouts)

        obs = next_obs

        # ── warmup progress ────────────────────────────────────────────────
        if total_steps < args.learning_starts:
            if total_steps - _last_warmup_print >= max(args.learning_starts // 5, args.n_envs):
                _last_warmup_print = total_steps
                pct = total_steps / args.learning_starts * 100
                bar = int(pct / 5)
                print(f"    [{('#' * bar):{'<'}20s}] {total_steps:>6,}/{args.learning_starts:,}  ({pct:.0f}%)")
        elif total_steps == args.learning_starts or \
                (total_steps - args.n_envs < args.learning_starts <= total_steps):
            print(f"  Buffer ready — starting SAC updates (batch={args.batch_size})\n")

        # ── update ─────────────────────────────────────────────────────────
        if total_steps >= args.learning_starts and len(buffer) >= args.batch_size:
            for _ in range(args.gradient_steps):
                batch   = buffer.sample(args.batch_size)
                metrics = agent.update(batch)
                for k, v in metrics.items():
                    train_metrics_acc[k] = train_metrics_acc.get(k, 0.0) + float(v)
                train_metrics_cnt += 1

        # ── evaluate ────────────────────────────────────────────────────────
        if total_steps - last_eval_step >= args.eval_freq:
            last_eval_step = total_steps
            ep_rewards, ep_lens, pocket_rate, clear_rate = _evaluate(agent, eval_env, args.eval_episodes)
            mean_r  = float(np.mean(ep_rewards))
            std_r   = float(np.std(ep_rewards))
            elapsed = time.time() - t_start
            fps     = total_steps / elapsed

            if mean_r       > best_mean_reward:  best_mean_reward  = mean_r
            if pocket_rate  > best_pocket_rate:
                best_pocket_rate = pocket_rate
                agent.save(os.path.join(exp_dir, "best_model", "best_model.pt"))

            avg_train = {k: v / train_metrics_cnt
                         for k, v in train_metrics_acc.items()} \
                         if train_metrics_cnt > 0 else {}
            train_metrics_acc.clear()
            train_metrics_cnt = 0

            log_dict = {
                "eval/mean_reward":      mean_r,
                "eval/std_reward":       std_r,
                "eval/pocket_rate":      pocket_rate * 100,
                "eval/clear_rate":       clear_rate  * 100,
                "eval/best_mean_reward": best_mean_reward,
                "eval/best_pocket_rate": best_pocket_rate * 100,
                "eval/ep_len_mean":      float(np.mean(ep_lens)),
                "train/fps":             fps,
                "train/buffer_size":     len(buffer),
            }
            _key_map = {"alpha": "ent_coef", "alpha_loss": "ent_coef_loss"}
            log_dict.update({f"train/{_key_map.get(k, k)}": v for k, v in avg_train.items()})

            pct        = total_steps / args.total_steps * 100
            eta_sec    = (args.total_steps - total_steps) / fps if fps > 0 else 0
            elapsed_s  = int(elapsed)
            eta_s      = int(eta_sec)
            ep_len_mean = float(np.mean(ep_lens))

            W = 58
            print(f"\n{'─' * W}")
            print(
                f"  [{total_steps:>9,} / {args.total_steps:,}]  {pct:5.1f}%  "
                f"elapsed={elapsed_s//3600:02d}:{elapsed_s%3600//60:02d}:{elapsed_s%60:02d}  "
                f"eta={eta_s//3600:02d}:{eta_s%3600//60:02d}:{eta_s%60:02d}"
            )
            print(f"{'─' * W}")
            print(f"  {'EVAL':}")
            print(
                f"    reward   : {mean_r:+7.3f} ± {std_r:.3f}"
                f"    best: {best_mean_reward:+.3f}"
            )
            print(
                f"    pocket   : {pocket_rate*100:6.1f}%"
                f"              best: {best_pocket_rate*100:.1f}%"
            )
            print(
                f"    clear    : {clear_rate*100:6.1f}%"
                f"    ep_len: {ep_len_mean:.1f} steps"
                f"    buffer: {len(buffer):,}"
            )
            if avg_train:
                print(f"  {'TRAIN':}")
                critic = avg_train.get("critic_loss", float("nan"))
                actor  = avg_train.get("actor_loss",  float("nan"))
                alpha  = avg_train.get("alpha",        float("nan"))
                log_pi = avg_train.get("log_pi",       float("nan"))
                print(
                    f"    critic   : {critic:8.4f}"
                    f"    actor: {actor:+.4f}"
                    f"    alpha: {alpha:.4f}"
                )
                print(f"    log_pi   : {log_pi:+8.4f}    fps:   {fps:.0f}")
            else:
                print(f"  fps: {fps:.0f}  (warming up — no updates yet)")
            print(f"{'─' * W}")

            if not args.no_wandb:
                wandb.log(log_dict, step=total_steps)

            # evaluations.npz (compare.py 호환)
            eval_timesteps.append(total_steps)
            eval_results.append(ep_rewards)
            np.savez(
                os.path.join(exp_dir, "eval", "evaluations.npz"),
                timesteps = np.array(eval_timesteps),
                results   = np.array(eval_results),
            )

    # ── finalise ──────────────────────────────
    elapsed_total = time.time() - t_start
    agent.save(os.path.join(exp_dir, "final_model.pt"))

    # ── [1/2] random baseline ─────────────────
    print(f"\n{'─' * W}")
    print(f"  [1/2] Random baseline (500 episodes)...")
    random_pocketed = 0
    for _ in range(500):
        obs_r, _ = eval_env.reset()
        done = False
        while not done:
            _, _, term, trunc, info = eval_env.step(eval_env.action_space.sample())
            done = term or trunc
        random_pocketed += info.get("total_pocketed", 0) if args.n_balls > 1 else int(info.get("pocketed", False))
    random_rate = random_pocketed / 500 / args.n_balls * 100
    print(f"      random pocket rate: {random_rate:.1f}%")

    # ── [2/2] final eval on best checkpoint ───
    print(f"  [2/2] Final eval — best model (500 episodes)...")
    best_agent = build_agent(args)
    best_agent.load(os.path.join(exp_dir, "best_model", "best_model.pt"))

    final_eval_env = BilliardsEnv(
        n_balls       = args.n_balls,
        max_steps     = args.max_steps,
        step_penalty  = args.step_penalty,
        trunc_penalty = args.trunc_penalty,
    )
    final_eval_env.reset(seed=args.seed)

    n_final = 500
    total_pocketed_final, clears_final = 0, 0
    for _ in range(n_final):
        obs_f, _ = final_eval_env.reset()
        done = False
        while not done:
            action = best_agent.act(obs_f, deterministic=True)
            obs_f, _, term, trunc, info = final_eval_env.step(action)
            done = term or trunc
        total_pocketed_final += info.get("total_pocketed", 0) if args.n_balls > 1 else int(info.get("pocketed", False))
        clears_final += int(info.get("total_pocketed", 0) == args.n_balls) if args.n_balls > 1 else int(info.get("pocketed", False))
    final_eval_env.close()

    trained_rate = total_pocketed_final / n_final / args.n_balls * 100
    clear_rate   = clears_final / n_final * 100
    improvement  = trained_rate - random_rate

    print(f"\n  {'─' * (W - 2)}")
    print(f"  pocket rate : {trained_rate:.1f}%   (random: {random_rate:.1f}%   improvement: {improvement:+.1f}pp)")
    print(f"  clear  rate : {clear_rate:.1f}%")
    print(f"  training time: {elapsed_total/60:.1f} min   fps: {total_steps/elapsed_total:.0f}")
    print(f"  saved → {exp_dir}")

    results = {
        "agent":               args.agent,
        "seed":                args.seed,
        "n_balls":             args.n_balls,
        "max_steps":           args.max_steps,
        "step_penalty":        args.step_penalty,
        "trunc_penalty":       args.trunc_penalty,
        "total_steps":         total_steps,
        "random_pocket_rate":  round(random_rate,   2),
        "trained_pocket_rate": round(trained_rate,  2),
        "clear_rate":          round(clear_rate,    2),
        "improvement_pp":      round(improvement,   2),
        "best_mean_reward":    round(best_mean_reward, 4),
        "training_time_sec":   round(elapsed_total, 1),
        "avg_fps":             round(total_steps / elapsed_total, 0),
        "exp_dir":             exp_dir,
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if not args.no_wandb:
        wandb.log({
            "final/trained_pocket_rate": trained_rate,
            "final/clear_rate":          clear_rate,
            "final/random_pocket_rate":  random_rate,
            "final/improvement_pp":      improvement,
        })
        wandb.finish()

    vec_env.close()
    eval_env.close()


# ──────────────────────────────────────────────
# Eval helper
# ──────────────────────────────────────────────

def _evaluate(agent, env, n_episodes: int):
    n_balls = env.n_balls
    rewards, ep_lens, pocketed, clears = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_r, ep_len = 0.0, 0
        ep_pocketed  = 0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r    += r
            ep_len  += 1
            if n_balls == 1:
                ep_pocketed = int(info.get("pocketed", False))
            else:
                ep_pocketed = info.get("total_pocketed", 0)
        rewards.append(ep_r)
        ep_lens.append(ep_len)
        pocketed.append(ep_pocketed / n_balls)
        clears.append(float(ep_pocketed == n_balls))
    return rewards, ep_lens, float(np.mean(pocketed)), float(np.mean(clears))


if __name__ == "__main__":
    args = parse_args()
    train(args)
