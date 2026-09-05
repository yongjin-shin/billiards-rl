"""
world_model/generate_data_v3.py — trajectory dataset v3 for MarkovPredictor

v2 대비 변경:
  - 각 이벤트에 velocity(vx,vy) + angular velocity(wx,wy,wz) 추가
  - event feature: cue_xy(2)+cue_vel(2)+cue_avel(3)+tgt_xy(2)+tgt_vel(2)+tgt_avel(3)+type_oh(10) = 24
  - stick_ball 이후 cue ball initial state → init_cue_state: (N, 5) 별도 저장
  - cue_masks / tgt_masks: v2와 동일 의미 (해당 공이 이벤트에 관여했는지)

Data format v3:
    events    : (N, MAX_EVENTS, 24)  float32
                [:, :,  0: 2] = cue_xy    normalized  [0,1]
                [:, :,  2: 4] = cue_vel   normalized  /MAX_SPEED
                [:, :,  4: 7] = cue_avel  normalized  /MAX_AVEL
                [:, :,  7: 9] = tgt_xy    normalized  [0,1]
                [:, :,  9:11] = tgt_vel   normalized  /MAX_SPEED
                [:, :, 11:14] = tgt_avel  normalized  /MAX_AVEL
                [:, :, 14:  ] = one_hot type (10)
    cue_masks      : (N, MAX_EVENTS)  int8
    tgt_masks      : (N, MAX_EVENTS)  int8
    lengths        : (N,)             int32
    obs            : (N, 16)          float32  raw
    actions        : (N, 2)           float32
    pocketed       : (N,)             bool
    n_bounces      : (N,)             int32
    init_cue_state : (N, 5)           float32  cue ball [vx,vy,wx,wy,wz] after stick_ball
                                               (normalized: vel/MAX_SPEED, avel/MAX_AVEL)

Usage:
    python world_model/generate_data_v3.py --tag random_v3 --n-episodes 25000
    python world_model/generate_data_v3.py --tag sac_v3 \\
        --model logs/experiments/.../best_model.zip --n-episodes 25000
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv
from world_model.wm_predictor import (
    EVENT_TYPES, EVENT2IDX,
    N_EVENT_TYPES, MAX_EVENTS,
    TABLE_W, TABLE_H, MAX_SPEED,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

CUE_ID    = "cue"
TGT_ID    = "1"
MAX_AVEL  = 300.0   # rad/s — 200 에피소드 실측: p99=123, max=247 → 300으로 여유 확보
MAX_SPEED_V3 = 12.0  # m/s  — wm_predictor.MAX_SPEED=8.0은 action 범위; 실측 max=10.5 m/s

EVENT_DIM_V3 = 2 + 2 + 3 + 2 + 2 + 3 + N_EVENT_TYPES  # 24


# ── Ball state 추출 헬퍼 ───────────────────────────────────────────────────────

def get_ball_state(agent):
    """
    agent.initial에서 (xy, vel_xy, avel_xyz) 추출.

    Returns:
        xy   : (float, float)              정규화 전 미터 단위
        vel  : (float, float)              정규화 전 m/s
        avel : (float, float, float)       정규화 전 rad/s
        valid: bool                        agent.initial이 유효한지
    """
    if agent.initial is None:
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0), False

    # position
    if hasattr(agent.initial, "xyz"):
        x, y = float(agent.initial.xyz[0]), float(agent.initial.xyz[1])
    elif hasattr(agent.initial, "state"):
        x, y = float(agent.initial.state.rvw[0, 0]), float(agent.initial.state.rvw[0, 1])
    else:
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0), False

    # linear velocity
    if hasattr(agent.initial, "vel"):
        vx, vy = float(agent.initial.vel[0]), float(agent.initial.vel[1])
    elif hasattr(agent.initial, "state"):
        vx, vy = float(agent.initial.state.rvw[1, 0]), float(agent.initial.state.rvw[1, 1])
    else:
        vx, vy = 0.0, 0.0

    # angular velocity
    if hasattr(agent.initial, "avel"):
        wx = float(agent.initial.avel[0])
        wy = float(agent.initial.avel[1])
        wz = float(agent.initial.avel[2])
    elif hasattr(agent.initial, "state"):
        wx = float(agent.initial.state.rvw[2, 0])
        wy = float(agent.initial.state.rvw[2, 1])
        wz = float(agent.initial.state.rvw[2, 2])
    else:
        wx, wy, wz = 0.0, 0.0, 0.0

    return (x, y), (vx, vy), (wx, wy, wz), True


def normalize_state(xy, vel, avel):
    """물리 단위 → 정규화."""
    x  = float(np.clip(xy[0], 0.0, TABLE_W)) / TABLE_W
    y  = float(np.clip(xy[1], 0.0, TABLE_H)) / TABLE_H
    vx = float(np.clip(vel[0]  / MAX_SPEED_V3, -2.0, 2.0))
    vy = float(np.clip(vel[1]  / MAX_SPEED_V3, -2.0, 2.0))
    wx = float(np.clip(avel[0] / MAX_AVEL,  -2.0, 2.0))
    wy = float(np.clip(avel[1] / MAX_AVEL,  -2.0, 2.0))
    wz = float(np.clip(avel[2] / MAX_AVEL,  -2.0, 2.0))
    return (x, y), (vx, vy), (wx, wy, wz)


# ── 이벤트에서 공 상태 추출 ───────────────────────────────────────────────────

def extract_event_states(event, et_str):
    """
    이벤트에서 (cue_state, tgt_state, cue_mask, tgt_mask) 추출.
    각 state: (xy_norm, vel_norm, avel_norm) 튜플.
    """
    cue_state = ((0., 0.), (0., 0.), (0., 0., 0.))
    tgt_state = ((0., 0.), (0., 0.), (0., 0., 0.))
    cue_mask, tgt_mask = 0, 0

    for agent in event.agents:
        if not (hasattr(agent, "agent_type") and agent.agent_type == "ball"):
            continue
        xy, vel, avel, valid = get_ball_state(agent)
        if not valid:
            continue
        n_xy, n_vel, n_avel = normalize_state(xy, vel, avel)

        if agent.id == CUE_ID:
            cue_state = (n_xy, n_vel, n_avel)
            cue_mask  = 1
        elif agent.id == TGT_ID:
            tgt_state = (n_xy, n_vel, n_avel)
            tgt_mask  = 1

    return cue_state, tgt_state, cue_mask, tgt_mask


# ── Trajectory 추출 ────────────────────────────────────────────────────────────

def extract_trajectory_v3(system):
    """
    physics system에서 v3 format trajectory 추출.

    Returns:
        events         : (MAX_EVENTS, 24)  float32
        cue_masks      : (MAX_EVENTS,)     int8
        tgt_masks      : (MAX_EVENTS,)     int8
        length         : int
        n_bounces      : int
        init_cue_state : (5,) float32  — cue vel/avel after stick_ball (normalized)
    """
    events_enc = np.zeros((MAX_EVENTS, EVENT_DIM_V3), dtype=np.float32)
    cue_masks  = np.zeros(MAX_EVENTS, dtype=np.int8)
    tgt_masks  = np.zeros(MAX_EVENTS, dtype=np.int8)
    n_bounces  = 0
    idx        = 0

    # stick_ball 이후 cue ball state 추출용
    init_cue_vel  = np.zeros(2, dtype=np.float32)
    init_cue_avel = np.zeros(3, dtype=np.float32)

    for e in system.events:
        et = str(e.event_type)

        if et == "none":
            continue

        if et == "stick_ball":
            # cue ball state after stick_ball = 다음 이벤트에서의 initial vel이지만,
            # stick_ball 에서 cue ball initial은 타격 직전 (v=0). 실제 초기 속도는
            # 다음 이벤트에서 추출하므로 여기서는 건너뜀.
            continue

        if idx >= MAX_EVENTS:
            break

        cue_state, tgt_state, cm, tm = extract_event_states(e, et)

        # 첫 이벤트의 cue vel/avel = cue ball이 첫 충돌에 도달했을 때의 속도
        if idx == 0:
            (cx, cy), (cvx, cvy), (cwx, cwy, cwz) = cue_state
            init_cue_vel  = np.array([cvx, cvy], dtype=np.float32)
            init_cue_avel = np.array([cwx, cwy, cwz], dtype=np.float32)

        # one-hot type
        type_idx = EVENT2IDX.get(et, 0)
        one_hot  = np.zeros(N_EVENT_TYPES, dtype=np.float32)
        one_hot[type_idx] = 1.0

        (cx, cy),   (cvx, cvy),   (cwx, cwy, cwz)   = cue_state
        (tx, ty),   (tvx, tvy),   (twx, twy, twz)   = tgt_state

        events_enc[idx,  0] = cx;  events_enc[idx,  1] = cy
        events_enc[idx,  2] = cvx; events_enc[idx,  3] = cvy
        events_enc[idx,  4] = cwx; events_enc[idx,  5] = cwy; events_enc[idx,  6] = cwz
        events_enc[idx,  7] = tx;  events_enc[idx,  8] = ty
        events_enc[idx,  9] = tvx; events_enc[idx, 10] = tvy
        events_enc[idx, 11] = twx; events_enc[idx, 12] = twy; events_enc[idx, 13] = twz
        events_enc[idx, 14:] = one_hot

        cue_masks[idx] = cm
        tgt_masks[idx] = tm

        if "cushion" in et:
            n_bounces += 1

        idx += 1

    init_cue_state = np.concatenate([init_cue_vel, init_cue_avel])  # (5,)
    return events_enc, cue_masks, tgt_masks, idx, n_bounces, init_cue_state


# ── Data generation loop ───────────────────────────────────────────────────────

def generate(env, policy_fn, n_episodes, rng):
    obs_list, action_list = [], []
    events_list, cue_masks_list, tgt_masks_list = [], [], []
    lengths_list, pocketed_list, bounces_list    = [], [], []
    init_cue_state_list = []

    for ep in range(n_episodes):
        seed = int(rng.integers(0, 2**31))
        obs, _  = env.reset(seed=seed)
        action  = policy_fn(obs)

        _, reward, term, trunc, info = env.step(action)

        events_enc, cue_masks, tgt_masks, length, n_bounces, init_cue_state = \
            extract_trajectory_v3(env.system)

        obs_list.append(obs.copy())
        action_list.append(action.copy())
        events_list.append(events_enc)
        cue_masks_list.append(cue_masks)
        tgt_masks_list.append(tgt_masks)
        lengths_list.append(length)
        pocketed_list.append(bool(info.get("pocketed", False)))
        bounces_list.append(n_bounces)
        init_cue_state_list.append(init_cue_state)

        if (ep + 1) % 1000 == 0:
            pr = sum(pocketed_list) / (ep + 1) * 100
            print(f"  [{ep+1:>6}/{n_episodes}]  pocket={pr:.1f}%"
                  f"  avg_len={np.mean(lengths_list):.1f}"
                  f"  avg_bounces={np.mean(bounces_list):.1f}")

    return dict(
        obs            = np.stack(obs_list).astype(np.float32),           # (N, 16)
        actions        = np.stack(action_list).astype(np.float32),        # (N, 2)
        events         = np.stack(events_list).astype(np.float32),        # (N, 32, 24)
        cue_masks      = np.stack(cue_masks_list).astype(np.int8),        # (N, 32)
        tgt_masks      = np.stack(tgt_masks_list).astype(np.int8),        # (N, 32)
        lengths        = np.array(lengths_list,       dtype=np.int32),    # (N,)
        pocketed       = np.array(pocketed_list,      dtype=bool),        # (N,)
        n_bounces      = np.array(bounces_list,       dtype=np.int32),    # (N,)
        init_cue_state = np.stack(init_cue_state_list).astype(np.float32),# (N, 5)
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",        type=str, required=True)
    p.add_argument("--model",      type=str, default=None)
    p.add_argument("--n-episodes", type=int, default=25000)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out-dir",    type=str,
                   default=os.path.join(os.path.dirname(__file__), "data_v3"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = BilliardsEnv(n_balls=1)

    if args.model:
        from stable_baselines3 import SAC
        print(f"  Loading SAC model: {args.model}")
        sb3_model = SAC.load(args.model)
        policy_fn = lambda obs: sb3_model.predict(obs, deterministic=True)[0]
    else:
        policy_fn = lambda obs: env.action_space.sample()
        print("  → random policy")

    print(f"\nGenerating {args.n_episodes} episodes  [tag={args.tag}]")
    print(f"Table: {TABLE_W:.4f} × {TABLE_H:.4f} m")
    print(f"MAX_SPEED_V3={MAX_SPEED_V3}  MAX_AVEL={MAX_AVEL}")
    print(f"Event dim: {EVENT_DIM_V3}\n")

    data = generate(env, policy_fn, args.n_episodes, rng)

    # ── 통계 ──────────────────────────────────────────────────────────────────
    pocket_rate = data["pocketed"].mean() * 100
    print(f"\nPocket rate    : {pocket_rate:.1f}%")
    print(f"Avg length     : {data['lengths'].mean():.1f}")
    print(f"Avg bounces    : {data['n_bounces'].mean():.1f}")
    print(f"cue_masks      : {data['cue_masks'].mean():.3f}")
    print(f"tgt_masks      : {data['tgt_masks'].mean():.3f}")

    # init_cue_vel 분포 확인 (MAX_AVEL 교정용)
    ics = data["init_cue_state"]
    print(f"init_cue vel   : max_abs={np.abs(ics[:, :2]).max():.3f} (정규화 후, >1이면 MAX_SPEED 조정)")
    print(f"init_cue avel  : max_abs={np.abs(ics[:, 2:]).max():.3f} (정규화 후, >1이면 MAX_AVEL 조정)")

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.tag}_{ts}.npz"
    fpath = os.path.join(args.out_dir, fname)
    np.savez_compressed(fpath, **data)

    meta_path = os.path.join(args.out_dir, "metadata.json")
    meta = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.append({
        "file"          : fname,
        "tag"           : args.tag,
        "model"         : args.model,
        "n_episodes"    : args.n_episodes,
        "pocket_rate"   : round(pocket_rate, 2),
        "avg_length"    : round(float(data["lengths"].mean()), 2),
        "avg_bounces"   : round(float(data["n_bounces"].mean()), 2),
        "created_at"    : ts,
        "seed"          : args.seed,
        "format"        : "v3",
        "event_dim"     : EVENT_DIM_V3,
        "max_avel"      : MAX_AVEL,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved → {fpath}")
    env.close()


if __name__ == "__main__":
    main()
