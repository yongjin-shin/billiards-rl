"""
world_model/generate_data_v2.py — trajectory dataset v2 for WMPredictor

v1 대비 변경:
  - 이벤트당 위치를 cue_xy + tgt_xy 두 개로 분리
  - cue_mask / tgt_mask 추가 (어떤 공의 위치가 유효한지)
  - STICK_BALL 은 예측 대상 아님 → events 에서 제외 (Encoder 에 흡수)
  - 좌표를 TABLE_W / TABLE_H 로 정규화하여 저장 ([0,1] 범위)

Data format v2:
    events    : (N, MAX_EVENTS, 14)   float32
                [:, :, 0:2] = cue_xy  normalized
                [:, :, 2:4] = tgt_xy  normalized
                [:, :, 4: ] = one_hot type (10)
    cue_masks : (N, MAX_EVENTS)       int8   (0 or 1)
    tgt_masks : (N, MAX_EVENTS)       int8   (0 or 1)
    lengths   : (N,)                  int32  (STICK_BALL 제외한 실제 이벤트 수)
    obs       : (N, 16)               float32  (raw, model 이 normalize)
    actions   : (N, 2)                float32
    pocketed  : (N,)                  bool
    n_bounces : (N,)                  int32

Usage:
    # SAC model
    python world_model/generate_data_v2.py \\
        --tag sac_abs \\
        --model logs/experiments/.../best_model.zip \\
        --n-episodes 25000

    # random policy
    python world_model/generate_data_v2.py --tag random_abs --n-episodes 25000
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
    N_EVENT_TYPES, EVENT_DIM_V2, MAX_EVENTS,
    TABLE_W, TABLE_H,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

CUE_ID = "cue"
TGT_ID = "1"

# cue / tgt 위치 모두 추출하는 이벤트 타입
BOTH_AGENTS = {"ball_ball"}

# 어느 공인지 agent.id 로 판단하는 이벤트 타입
SINGLE_AGENT = {
    "ball_linear_cushion",
    "ball_circular_cushion",
    "ball_pocket",
    "sliding_rolling",
    "rolling_spinning",
    "rolling_stationary",
    "spinning_stationary",
}

# ── 위치 추출 헬퍼 ─────────────────────────────────────────────────────────────

def get_ball_xy(agent):
    """agent.initial 에서 (x, y) 추출."""
    if agent.initial is None:
        return 0.0, 0.0
    if hasattr(agent.initial, "xyz"):
        return float(agent.initial.xyz[0]), float(agent.initial.xyz[1])
    if hasattr(agent.initial, "state"):
        rvw = agent.initial.state.rvw[0]
        return float(rvw[0]), float(rvw[1])
    return 0.0, 0.0


def extract_event_positions(event, et_str):
    """
    이벤트에서 (cue_xy, tgt_xy, cue_mask, tgt_mask) 추출.

    ball_ball     → cue + tgt 모두
    그 외          → agent.id 로 어느 공인지 판별
    """
    cue_xy, tgt_xy = (0.0, 0.0), (0.0, 0.0)
    cue_mask, tgt_mask = 0, 0

    for agent in event.agents:
        if not (hasattr(agent, "agent_type") and agent.agent_type == "ball"):
            continue
        x, y = get_ball_xy(agent)
        # 물리 오류 방지 clip 후 normalize
        x = float(np.clip(x, 0.0, TABLE_W)) / TABLE_W
        y = float(np.clip(y, 0.0, TABLE_H)) / TABLE_H

        if agent.id == CUE_ID:
            cue_xy, cue_mask = (x, y), 1
        elif agent.id == TGT_ID:
            tgt_xy, tgt_mask = (x, y), 1

    return cue_xy, tgt_xy, cue_mask, tgt_mask


# ── Trajectory 추출 ────────────────────────────────────────────────────────────

def extract_trajectory_v2(system):
    """
    physics system 에서 v2 format trajectory 추출.

    STICK_BALL 은 건너뛰고 (Encoder 에 흡수됨) 이후 이벤트만 저장.
    MAX_EVENTS 초과분은 truncate.

    Returns:
        events    : (MAX_EVENTS, 14)  float32
        cue_masks : (MAX_EVENTS,)     int8
        tgt_masks : (MAX_EVENTS,)     int8
        length    : int    실제 이벤트 수 (STICK_BALL 제외)
        n_bounces : int    쿠션 바운스 횟수
    """
    events_enc = np.zeros((MAX_EVENTS, EVENT_DIM_V2), dtype=np.float32)
    cue_masks  = np.zeros(MAX_EVENTS, dtype=np.int8)
    tgt_masks  = np.zeros(MAX_EVENTS, dtype=np.int8)
    n_bounces  = 0
    idx        = 0

    for e in system.events:
        et = str(e.event_type)

        if et in ("none", "stick_ball"):
            # none: 더미 / stick_ball: Encoder 에 흡수 → 스킵
            continue

        if idx >= MAX_EVENTS:
            break

        cue_xy, tgt_xy, cm, tm = extract_event_positions(e, et)

        # one-hot type
        type_idx = EVENT2IDX.get(et, 0)
        one_hot  = np.zeros(N_EVENT_TYPES, dtype=np.float32)
        one_hot[type_idx] = 1.0

        events_enc[idx, 0] = cue_xy[0]
        events_enc[idx, 1] = cue_xy[1]
        events_enc[idx, 2] = tgt_xy[0]
        events_enc[idx, 3] = tgt_xy[1]
        events_enc[idx, 4:] = one_hot

        cue_masks[idx] = cm
        tgt_masks[idx] = tm

        if "cushion" in et:
            n_bounces += 1

        idx += 1

    return events_enc, cue_masks, tgt_masks, idx, n_bounces


# ── Data generation loop ───────────────────────────────────────────────────────

def generate(env, policy_fn, n_episodes, rng):
    obs_list, action_list = [], []
    events_list, cue_masks_list, tgt_masks_list = [], [], []
    lengths_list, pocketed_list, bounces_list = [], [], []

    for ep in range(n_episodes):
        seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed)
        action  = policy_fn(obs)

        _, reward, term, trunc, info = env.step(action)

        events_enc, cue_masks, tgt_masks, length, n_bounces = \
            extract_trajectory_v2(env.system)

        obs_list.append(obs.copy())
        action_list.append(action.copy())
        events_list.append(events_enc)
        cue_masks_list.append(cue_masks)
        tgt_masks_list.append(tgt_masks)
        lengths_list.append(length)
        pocketed_list.append(bool(info.get("pocketed", False)))
        bounces_list.append(n_bounces)

        if (ep + 1) % 1000 == 0:
            pr = sum(pocketed_list) / (ep + 1) * 100
            print(f"  [{ep+1:>6}/{n_episodes}]  pocket={pr:.1f}%"
                  f"  avg_len={np.mean(lengths_list):.1f}"
                  f"  avg_bounces={np.mean(bounces_list):.1f}")

    return dict(
        obs       = np.stack(obs_list).astype(np.float32),        # (N, 16)  raw
        actions   = np.stack(action_list).astype(np.float32),     # (N, 2)
        events    = np.stack(events_list).astype(np.float32),     # (N, 32, 14)
        cue_masks = np.stack(cue_masks_list).astype(np.int8),     # (N, 32)
        tgt_masks = np.stack(tgt_masks_list).astype(np.int8),     # (N, 32)
        lengths   = np.array(lengths_list,  dtype=np.int32),      # (N,)
        pocketed  = np.array(pocketed_list, dtype=bool),          # (N,)
        n_bounces = np.array(bounces_list,  dtype=np.int32),      # (N,)
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",        type=str, required=True)
    p.add_argument("--model",      type=str, default=None,
                   help="SAC model path (.zip). None = random policy")
    p.add_argument("--n-episodes", type=int, default=25000)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out-dir",    type=str,
                   default=os.path.join(os.path.dirname(__file__), "data_v2"))
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
    print(f"Table: {TABLE_W:.4f} × {TABLE_H:.4f} m  (coords normalized to [0,1])\n")

    data = generate(env, policy_fn, args.n_episodes, rng)

    pocket_rate = data["pocketed"].mean() * 100
    print(f"\nPocket rate : {pocket_rate:.1f}%")
    print(f"Avg length  : {data['lengths'].mean():.1f}  (STICK_BALL 제외)")
    print(f"Avg bounces : {data['n_bounces'].mean():.1f}")
    print(f"cue_masks   : {data['cue_masks'].mean():.3f}  (fraction of steps with cue)")
    print(f"tgt_masks   : {data['tgt_masks'].mean():.3f}  (fraction of steps with tgt)")

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
        "file"        : fname,
        "tag"         : args.tag,
        "model"       : args.model,
        "n_episodes"  : args.n_episodes,
        "pocket_rate" : round(pocket_rate, 2),
        "avg_length"  : round(float(data["lengths"].mean()), 2),
        "avg_bounces" : round(float(data["n_bounces"].mean()), 2),
        "created_at"  : ts,
        "seed"        : args.seed,
        "format"      : "v2",
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved → {fpath}")
    env.close()


if __name__ == "__main__":
    main()
