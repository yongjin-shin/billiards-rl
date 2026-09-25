"""
world_model/generate_collision_data.py

각 충돌 이벤트마다 (pre_state, post_state, contact_normal, type) 쌍을 수집.
GNN collision resolver 학습용.

충돌 타입 (4종):
    0 = ball_ball
    1 = ball_linear_cushion
    2 = ball_circular_cushion
    3 = ball_pocket  (post_vel = 0 이므로 학습 제외해도 됨)

Data format (per sample):
    pre_vel    (2,2): [cue_vel, tgt_vel]  — ball_ball 아니면 tgt=0
    pre_avel   (2,3): [cue_avel, tgt_avel]
    post_vel   (2,2)
    post_avel  (2,3)
    pos        (2,2): [cue_pos, tgt_pos]  — normalized [0,1]
    normal     (2,)  : contact normal (unit vector, world coords)
    coll_type  (1,)  : int, 0~3
    has_tgt    (1,)  : bool, ball_ball이면 1

Usage:
    python world_model/generate_collision_data.py --n-episodes 50000
"""

import os, sys, json, argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv

TABLE_W   = 0.9906
TABLE_H   = 1.9812
MAX_SPEED = 12.0
MAX_AVEL  = 300.0

COLL_TYPES = {
    "stick_ball":              4,
    "ball_ball":               0,
    "ball_linear_cushion":     1,
    "ball_circular_cushion":   2,
    "ball_pocket":             3,
}


def _get_ball_pre_post(agent):
    """Ball agent → (pre_vel, pre_avel, post_vel, post_avel, pos) 미터/rad 단위."""
    ini, fin = agent.initial, agent.final
    pre_vel  = np.array(ini.vel[:2],  dtype=np.float64)
    pre_avel = np.array(ini.avel[:3], dtype=np.float64)
    post_vel  = np.array(fin.vel[:2],  dtype=np.float64)
    post_avel = np.array(fin.avel[:3], dtype=np.float64)
    pos = np.array(ini.xyz[:2], dtype=np.float64) if hasattr(ini, "xyz") \
          else np.array(ini.state.rvw[0, :2], dtype=np.float64)
    return pre_vel, pre_avel, post_vel, post_avel, pos


def _contact_normal_linear(agent):
    """Linear cushion → contact normal (2D unit vector, inward)."""
    ini = agent.initial
    n = np.array([ini.lx, ini.ly], dtype=np.float64)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else np.array([1.0, 0.0])


def _contact_normal_circular(agent, ball_pos):
    """Circular cushion → normal from center toward ball."""
    ini = agent.initial
    center = np.array([ini.a, ini.b], dtype=np.float64)
    d = ball_pos - center
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-9 else np.array([1.0, 0.0])


def _contact_normal_pocket(agent, ball_pos):
    """Pocket → normal toward pocket center."""
    ini = agent.initial
    center = np.array([ini.a, ini.b], dtype=np.float64)
    d = center - ball_pos
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-9 else np.array([0.0, -1.0])


def _contact_normal_ball_ball(pos_cue, pos_tgt):
    """Ball-ball → normal from cue toward tgt."""
    d = pos_tgt - pos_cue
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-9 else np.array([1.0, 0.0])


def extract_collisions(system):
    """Shot 하나에서 모든 충돌 이벤트의 pre/post 쌍을 추출."""
    samples = []

    for e in system.events:
        et = str(e.event_type)
        if et not in COLL_TYPES:
            continue

        coll_type = COLL_TYPES[et]

        # ball agent 수집
        balls = {a.id: a for a in e.agents
                 if getattr(a, "agent_type", "") == "ball"}

        cue = balls.get("cue")
        tgt = balls.get("1")

        if cue is None:
            continue  # cue 없는 이벤트는 skip

        cue_pre_vel, cue_pre_avel, cue_post_vel, cue_post_avel, cue_pos = \
            _get_ball_pre_post(cue)

        has_tgt = tgt is not None
        if has_tgt:
            tgt_pre_vel, tgt_pre_avel, tgt_post_vel, tgt_post_avel, tgt_pos = \
                _get_ball_pre_post(tgt)
        else:
            tgt_pre_vel  = np.zeros(2)
            tgt_pre_avel = np.zeros(3)
            tgt_post_vel  = np.zeros(2)
            tgt_post_avel = np.zeros(3)
            tgt_pos       = np.zeros(2)

        # contact normal
        if et == "stick_ball":
            stick = next(a for a in e.agents
                         if getattr(a, "agent_type", "") == "cue")
            phi_rad = float(stick.initial.phi) * np.pi / 180.0
            normal = np.array([np.cos(phi_rad), np.sin(phi_rad)])
        elif et == "ball_ball":
            normal = _contact_normal_ball_ball(cue_pos, tgt_pos)
        elif et == "ball_linear_cushion":
            cush = next(a for a in e.agents
                        if getattr(a, "agent_type", "") == "linear_cushion_segment")
            normal = _contact_normal_linear(cush)
        elif et == "ball_circular_cushion":
            cush = next(a for a in e.agents
                        if getattr(a, "agent_type", "") == "circular_cushion_segment")
            normal = _contact_normal_circular(cush, cue_pos)
        elif et == "ball_pocket":
            pock = next(a for a in e.agents
                        if getattr(a, "agent_type", "") == "pocket")
            normal = _contact_normal_pocket(pock, cue_pos)

        samples.append(dict(
            pre_vel   = np.stack([cue_pre_vel,  tgt_pre_vel],  axis=0),   # (2,2)
            pre_avel  = np.stack([cue_pre_avel, tgt_pre_avel], axis=0),   # (2,3)
            post_vel  = np.stack([cue_post_vel, tgt_post_vel], axis=0),   # (2,2)
            post_avel = np.stack([cue_post_avel, tgt_post_avel], axis=0), # (2,3)
            pos       = np.stack([cue_pos / [TABLE_W, TABLE_H],
                                  tgt_pos / [TABLE_W, TABLE_H]], axis=0), # (2,2) normalized
            normal    = normal.astype(np.float32),                         # (2,)
            coll_type = coll_type,
            has_tgt   = int(has_tgt),
        ))

    return samples


def normalize(samples):
    """velocity/avel 단위 정규화."""
    for s in samples:
        s["pre_vel"]   = (s["pre_vel"]   / MAX_SPEED).astype(np.float32)
        s["pre_avel"]  = (s["pre_avel"]  / MAX_AVEL ).astype(np.float32)
        s["post_vel"]  = (s["post_vel"]  / MAX_SPEED).astype(np.float32)
        s["post_avel"] = (s["post_avel"] / MAX_AVEL ).astype(np.float32)
        s["pos"]       = s["pos"].astype(np.float32)
    return samples


def generate(n_episodes, model_path=None, seed=0):
    rng = np.random.default_rng(seed)
    env = BilliardsEnv(n_balls=1)

    if model_path:
        from stable_baselines3 import SAC
        sb3 = SAC.load(model_path)
        policy_fn = lambda obs: sb3.predict(obs, deterministic=True)[0]
        print(f"SAC policy: {model_path}")
    else:
        policy_fn = lambda obs: env.action_space.sample()
        print("Random policy")

    all_samples = []
    pocketed = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        action = policy_fn(obs)
        _, reward, _, _, info = env.step(action)
        if info.get("pocketed"): pocketed += 1

        samples = extract_collisions(env.system)
        all_samples.extend(normalize(samples))

        if (ep + 1) % 5000 == 0:
            counts = {}
            for s in all_samples:
                counts[s["coll_type"]] = counts.get(s["coll_type"], 0) + 1
            print(f"  [{ep+1}/{n_episodes}]  samples={len(all_samples)}"
                  f"  pocket={pocketed/(ep+1)*100:.1f}%"
                  f"  types={counts}")

    env.close()
    return all_samples


def pack(samples):
    """list of dicts → dict of arrays."""
    return {
        "pre_vel"  : np.stack([s["pre_vel"]   for s in samples]),  # (N,2,2)
        "pre_avel" : np.stack([s["pre_avel"]  for s in samples]),  # (N,2,3)
        "post_vel" : np.stack([s["post_vel"]  for s in samples]),  # (N,2,2)
        "post_avel": np.stack([s["post_avel"] for s in samples]),  # (N,2,3)
        "pos"      : np.stack([s["pos"]       for s in samples]),  # (N,2,2)
        "normal"   : np.stack([s["normal"]    for s in samples]),  # (N,2)
        "coll_type": np.array([s["coll_type"] for s in samples], dtype=np.int8),  # (N,)
        "has_tgt"  : np.array([s["has_tgt"]   for s in samples], dtype=np.int8),  # (N,)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=50000)
    p.add_argument("--model",      type=str, default=None)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out-dir",    type=str,
                   default=os.path.join(os.path.dirname(__file__), "data_collision"))
    p.add_argument("--tag",        type=str, default="")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Generating {args.n_episodes} episodes...")
    samples = generate(args.n_episodes, args.model, args.seed)
    data = pack(samples)

    counts = {}
    for ct in data["coll_type"]:
        counts[int(ct)] = counts.get(int(ct), 0) + 1
    names = ["ball_ball", "linear", "circular", "pocket", "stick_ball"]
    print(f"\nTotal samples: {len(samples)}")
    for i, name in enumerate(names):
        print(f"  {name:20s}: {counts.get(i, 0):6d}")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag  = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out_dir, f"collision{tag}_{ts}.npz")
    np.savez_compressed(path, **data)
    print(f"\nSaved → {path}")

    meta_path = os.path.join(args.out_dir, "metadata.json")
    meta = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.append({"file": os.path.basename(path), "n_samples": len(samples),
                 "n_episodes": args.n_episodes, "type_counts": counts,
                 "created_at": ts})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
