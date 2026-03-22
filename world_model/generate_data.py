"""
world_model/generate_data.py — trajectory event dataset generator (Exp-15)

n_balls=1 기준:
  - ball_ball 이벤트 이후 target ball("1")의 궤적을 추출
  - SAC 학습 모델 + random policy 두 가지로 데이터 생성
  - 태그로 관리해서 재사용 가능하게 저장

Usage:
    # SAC trained model
    python world_model/generate_data.py \\
        --tag sac_5m_gs4 \\
        --model logs/experiments/SAC_5000k_s42_sp0.0_tp0.0_gs4_20260322_150734/best_model/best_model \\
        --n-episodes 5000

    # random policy
    python world_model/generate_data.py --tag random --n-episodes 5000
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# project root에서 실행 가정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv

# ── 이벤트 타입 정의 ─────────────────────────────────────────────────────────
EVENT_TYPES = [
    "none",
    "stick_ball",
    "ball_ball",
    "ball_linear_cushion",
    "ball_circular_cushion",
    "ball_pocket",
    "sliding_rolling",
    "rolling_spinning",
    "rolling_stationary",
    "spinning_stationary",
]
EVENT2IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
N_EVENT_TYPES = len(EVENT_TYPES)

# 이벤트 1개 = (x, y) + one-hot(10) = 12-dim
EVENT_DIM = 2 + N_EVENT_TYPES
MAX_EVENTS = 32   # 12 쿠션 × 2 + 기타 ≈ 29, 32로 여유 확보


def get_ball_xy(agent):
    """agent.initial 에서 (x, y) 추출. 없으면 (0, 0)."""
    if agent.initial is None:
        return 0.0, 0.0
    if hasattr(agent.initial, "xyz"):
        return float(agent.initial.xyz[0]), float(agent.initial.xyz[1])
    if hasattr(agent.initial, "state"):
        rvw = agent.initial.state.rvw[0]
        return float(rvw[0]), float(rvw[1])
    return 0.0, 0.0


def extract_trajectory(system, target_id="1"):
    """
    stick_ball 부터 전체 궤적 이벤트를 추출.
    cue ball의 접근 경로 + ball_ball 충돌 + target ball 이후 경로 모두 포함.

    각 이벤트에서 위치는:
      - cue 관련 이벤트 → cue ball 위치
      - ball_ball        → target ball 위치 (충돌 지점)
      - target ball 이벤트 → target ball 위치

    위치는 stick_ball 이벤트 위치를 기준으로 상대 좌표로 정규화.
    (절대 좌표가 latent에 인코딩되어 패턴 학습을 방해하는 것을 방지)

    Returns:
        events_enc : (MAX_EVENTS, EVENT_DIM) float32, padded
        length     : int, 실제 이벤트 수
        n_bounces  : int, 쿠션 바운스 횟수 (전체)
    """
    raw = []

    # stick_ball 위치를 기준점으로 먼저 찾기
    ref_x, ref_y = 0.0, 0.0
    for e in system.events:
        if str(e.event_type) == "stick_ball":
            for agent in e.agents:
                if hasattr(agent, "agent_type") and agent.agent_type == "ball":
                    ref_x, ref_y = get_ball_xy(agent)
                    break
            break

    for e in system.events:
        et = str(e.event_type)

        # 더미 이벤트 스킵
        if et == "none":
            continue

        # 이벤트에서 대표 위치 추출:
        # ball_ball → target ball 위치 우선, 없으면 첫 번째 ball agent
        # 나머지  → 첫 번째 ball agent 위치
        x, y = 0.0, 0.0
        if et == "ball_ball":
            for agent in e.agents:
                if agent.id == target_id:
                    x, y = get_ball_xy(agent)
                    break
            else:
                for agent in e.agents:
                    if agent.agent_type == "ball":
                        x, y = get_ball_xy(agent)
                        break
        else:
            for agent in e.agents:
                if hasattr(agent, "agent_type") and agent.agent_type == "ball":
                    x, y = get_ball_xy(agent)
                    break

        # stick_ball 위치 기준 상대 좌표
        raw.append((x - ref_x, y - ref_y, et))

    # 인코딩 (padding)
    events_enc = np.zeros((MAX_EVENTS, EVENT_DIM), dtype=np.float32)
    n = min(len(raw), MAX_EVENTS)
    for i, (x, y, et) in enumerate(raw[:n]):
        events_enc[i, 0] = x
        events_enc[i, 1] = y
        events_enc[i, 2 + EVENT2IDX.get(et, 0)] = 1.0

    n_bounces = sum(1 for _, _, et in raw
                    if "cushion" in et)
    return events_enc, n, n_bounces


def generate(env, policy_fn, n_episodes, rng):
    """n_episodes개의 에피소드를 수집하고 numpy 배열들로 반환."""
    obs_list, action_list, events_list = [], [], []
    lengths_list, rewards_list, pocketed_list, bounces_list = [], [], [], []

    for ep in range(n_episodes):
        seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed)
        action = policy_fn(obs)

        obs_next, reward, term, trunc, info = env.step(action)

        events_enc, length, n_bounces = extract_trajectory(env.system, target_id="1")

        obs_list.append(obs.copy())
        action_list.append(action.copy())
        events_list.append(events_enc)
        lengths_list.append(length)
        rewards_list.append(float(reward))
        pocketed_list.append(bool(info.get("pocketed", False)))
        bounces_list.append(n_bounces)

        if (ep + 1) % 500 == 0:
            pocket_so_far = sum(pocketed_list)
            print(f"  [{ep+1}/{n_episodes}] pocket={pocket_so_far/(ep+1)*100:.1f}%  "
                  f"avg_bounces={np.mean(bounces_list):.1f}")

    return dict(
        obs      = np.stack(obs_list).astype(np.float32),       # (N, 16)
        actions  = np.stack(action_list).astype(np.float32),    # (N, 2)
        events   = np.stack(events_list).astype(np.float32),    # (N, MAX_EVENTS, EVENT_DIM)
        lengths  = np.array(lengths_list, dtype=np.int32),      # (N,)
        rewards  = np.array(rewards_list, dtype=np.float32),    # (N,)
        pocketed = np.array(pocketed_list, dtype=bool),         # (N,)
        n_bounces= np.array(bounces_list, dtype=np.int32),      # (N,)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory dataset for world model")
    parser.add_argument("--tag",        type=str, required=True,
                        help="Dataset tag (e.g. sac_5m_gs4 / random)")
    parser.add_argument("--model",      type=str, default=None,
                        help="SAC model path. None = random policy")
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--out-dir",    type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 환경
    env = BilliardsEnv(n_balls=1)

    # policy 설정
    if args.model:
        from stable_baselines3 import SAC
        print(f"Loading SAC model: {args.model}")
        model = SAC.load(args.model)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
        print("  → deterministic SAC policy")
    else:
        policy_fn = lambda obs: env.action_space.sample()
        print("  → random policy")

    print(f"\nGenerating {args.n_episodes} episodes  [tag={args.tag}]")
    data = generate(env, policy_fn, args.n_episodes, rng)

    pocket_rate = data["pocketed"].mean() * 100
    print(f"\nPocket rate : {pocket_rate:.1f}%")
    print(f"Avg bounces : {data['n_bounces'].mean():.1f}")
    print(f"Max bounces : {data['n_bounces'].max()}")
    print(f"Lengths>0   : {(data['lengths']>0).sum()} / {args.n_episodes}")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.tag}_{ts}.npz"
    fpath = os.path.join(args.out_dir, fname)
    np.savez_compressed(fpath, **data)

    # 메타데이터
    meta_path = os.path.join(args.out_dir, "metadata.json")
    meta = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.append({
        "file"       : fname,
        "tag"        : args.tag,
        "model"      : args.model,
        "n_episodes" : args.n_episodes,
        "pocket_rate": round(pocket_rate, 2),
        "avg_bounces": round(float(data["n_bounces"].mean()), 2),
        "created_at" : ts,
        "seed"       : args.seed,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved → {fpath}")
    print(f"Meta  → {meta_path}")


if __name__ == "__main__":
    main()
