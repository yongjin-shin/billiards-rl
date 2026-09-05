"""
world_model/generate_data_fixeddt.py — Fixed-Δt world model data generator

Shot 하나당 Δt=0.05s 간격으로 공 상태를 샘플링.
각 스텝 (s_t, s_{t+1}, coll_flag, coll_type) 저장.

State dim 14 (normalized):
  [0:2]  cue_x, cue_y   (/ TABLE_W, TABLE_H)
  [2:4]  cue_vx, cue_vy (/ MAX_SPEED)
  [4:7]  cue_wx,wy,wz   (/ MAX_AVEL)
  [7:9]  tgt_x, tgt_y
  [9:11] tgt_vx, tgt_vy
  [11:14] tgt_wx,wy,wz

Collision type: 0=ball_ball 1=linear_cushion 2=circular_cushion 3=pocket -1=none

Usage:
    python world_model/generate_data_fixeddt.py --n-episodes 10000 --tag sac
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.wm_predictor import TABLE_W, TABLE_H

DT        = 0.05          # seconds
T_MAX     = 220           # max steps per episode (~11s)
STATE_DIM = 14
MAX_SPEED = 12.0          # m/s
MAX_AVEL  = 300.0         # rad/s

COLL_STR = {
    "ball_ball":             0,
    "ball_linear_cushion":   1,
    "ball_circular_cushion": 2,
    "ball_pocket":           3,
}


def normalize_ball(pos, vel, avel) -> np.ndarray:
    x  = float(np.clip(pos[0] / TABLE_W, -0.1, 1.1))
    y  = float(np.clip(pos[1] / TABLE_H, -0.1, 1.1))
    vx = float(np.clip(vel[0] / MAX_SPEED, -3.0, 3.0))
    vy = float(np.clip(vel[1] / MAX_SPEED, -3.0, 3.0))
    wx = float(np.clip(avel[0] / MAX_AVEL, -3.0, 3.0))
    wy = float(np.clip(avel[1] / MAX_AVEL, -3.0, 3.0))
    wz = float(np.clip(avel[2] / MAX_AVEL, -3.0, 3.0))
    return np.array([x, y, vx, vy, wx, wy, wz], dtype=np.float32)


def ball_state_vec(state: "BallState") -> np.ndarray:
    rvw = state.rvw
    return normalize_ball(rvw[0, :2], rvw[1, :2], rvw[2, :3])


def extract_episode(system, obs: np.ndarray, action: np.ndarray):
    """
    Returns dict:
        states      (T, 14)   normalized state at each timestep
        coll_flags  (T,)      bool — collision in [t, t+DT)
        coll_types  (T,)      int8 — 0..3 or -1
        length      int       number of valid steps T
        pocketed    bool
    """
    events = system.events
    cue_ball = system.balls["cue"]
    tgt_ball = system.balls["1"]

    # shot 종료 시점 (마지막 이벤트)
    t_end = events[-1].time

    # 타임스탬프 생성
    n_steps = min(int(t_end / DT) + 1, T_MAX - 1)
    timestamps = np.arange(n_steps + 1, dtype=np.float64) * DT
    timestamps = np.clip(timestamps, 0.0, t_end)

    # 공 상태 쿼리
    cue_states = pt.interpolate_ball_states(cue_ball, timestamps, extrapolate=True)
    tgt_states = pt.interpolate_ball_states(tgt_ball, timestamps, extrapolate=True)

    # 충돌 이벤트 목록: (time, type_int)
    coll_events = [
        (e.time, COLL_STR[str(e.event_type.value)])
        for e in events
        if str(e.event_type.value) in COLL_STR
    ]

    # 포켓 여부
    pocketed = any(ct == 3 for _, ct in coll_events)

    # 스텝별 상태 및 충돌 기록
    states     = np.zeros((n_steps, STATE_DIM), dtype=np.float32)
    coll_flags = np.zeros(n_steps, dtype=bool)
    coll_types = np.full(n_steps, -1, dtype=np.int8)

    for i in range(n_steps):
        t0, t1 = timestamps[i], timestamps[i + 1]
        cue_vec = ball_state_vec(cue_states[i])
        tgt_vec = ball_state_vec(tgt_states[i])
        states[i] = np.concatenate([cue_vec, tgt_vec])

        # 이 구간 안의 충돌 (여러 개면 첫 번째만 기록)
        for et, ect in coll_events:
            if t0 <= et < t1:
                coll_flags[i] = True
                coll_types[i] = ect
                break

    return {
        "states":      states,
        "coll_flags":  coll_flags,
        "coll_types":  coll_types,
        "length":      n_steps,
        "pocketed":    pocketed,
        "obs":         obs,
        "action":      action,
    }


def generate(env, policy_fn, n_episodes: int, rng: np.random.Generator,
             verbose: bool = True):
    from simulator import BilliardsEnv

    buf_states     = np.zeros((n_episodes, T_MAX, STATE_DIM), dtype=np.float32)
    buf_coll_flags = np.zeros((n_episodes, T_MAX), dtype=bool)
    buf_coll_types = np.full((n_episodes, T_MAX), -1, dtype=np.int8)
    buf_lengths    = np.zeros(n_episodes, dtype=np.int32)
    buf_pocketed   = np.zeros(n_episodes, dtype=bool)
    buf_obs        = np.zeros((n_episodes, 16), dtype=np.float32)
    buf_actions    = np.zeros((n_episodes, 2), dtype=np.float32)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        action = policy_fn(obs)

        # pooltool 시뮬레이션
        shot = env.system
        pt.simulate(shot, inplace=True)
        env.step(action)

        ep_data = extract_episode(shot, obs, action)
        T = ep_data["length"]

        buf_states[ep, :T]      = ep_data["states"]
        buf_coll_flags[ep, :T]  = ep_data["coll_flags"]
        buf_coll_types[ep, :T]  = ep_data["coll_types"]
        buf_lengths[ep]         = T
        buf_pocketed[ep]        = ep_data["pocketed"]
        buf_obs[ep]             = ep_data["obs"]
        buf_actions[ep]         = ep_data["action"]

        if verbose and (ep + 1) % 500 == 0:
            pct = buf_pocketed[:ep+1].mean() * 100
            print(f"  {ep+1}/{n_episodes}  pocketed={pct:.1f}%")

    return {
        "states":      buf_states,
        "coll_flags":  buf_coll_flags,
        "coll_types":  buf_coll_types,
        "lengths":     buf_lengths,
        "pocketed":    buf_pocketed,
        "obs":         buf_obs,
        "actions":     buf_actions,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",     default="world_model/data_fixeddt")
    p.add_argument("--n-episodes",  type=int, default=10_000)
    p.add_argument("--tag",         default="sac")
    p.add_argument("--sac-model",   default=None)
    p.add_argument("--random",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from simulator import BilliardsEnv
    env = BilliardsEnv(n_balls=1)
    rng = np.random.default_rng(args.seed)

    if args.random:
        def policy_fn(obs):
            return env.action_space.sample()
        tag = f"random_{args.tag}"
    else:
        from stable_baselines3 import SAC
        if args.sac_model is None:
            import glob
            paths = glob.glob("logs/experiments/SAC_5000k_s42_sp0.0_tp0.0_*/best_model/best_model.zip")
            assert paths, "SAC model not found"
            args.sac_model = paths[0]
        print(f"SAC: {args.sac_model}")
        sac = SAC.load(args.sac_model)
        def policy_fn(obs):
            return sac.predict(obs, deterministic=False)[0]
        tag = f"sac_{args.tag}"

    print(f"Generating {args.n_episodes} episodes  DT={DT}s  T_MAX={T_MAX}")
    data = generate(env, policy_fn, args.n_episodes, rng)
    env.close()

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{tag}_{ts}.npz"
    np.savez_compressed(out_dir / fname, **data)
    print(f"Saved → {out_dir / fname}")

    lengths = data["lengths"]
    print(f"Steps/shot: mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  max={lengths.max()}")
    print(f"Pocketed: {data['pocketed'].mean()*100:.1f}%")

    meta_path = out_dir / "metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else []
    meta.append({
        "file": fname, "tag": tag, "n_episodes": args.n_episodes,
        "dt": DT, "state_dim": STATE_DIM,
        "model": args.sac_model, "seed": args.seed,
    })
    json.dump(meta, open(meta_path, "w"), indent=2)


if __name__ == "__main__":
    main()
