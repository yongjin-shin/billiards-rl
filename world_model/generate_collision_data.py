"""
world_model/generate_collision_data.py

각 충돌 이벤트마다 (pre_state, post_state, contact_normal, type) 쌍을 수집.
GNN collision resolver 학습용.

충돌 타입 (7종):
    0 = ball_ball
    1 = cue_linear_cushion
    2 = cue_circular_cushion
    3 = ball_pocket  (cue 또는 target, post_vel = 0)
    4 = stick_ball
    5 = tgt_linear_cushion   (target ball만 관여)
    6 = tgt_circular_cushion (target ball만 관여)

Data format (per sample):
    pre_vel     (2,2): [protagonist_vel, companion_vel]  — ball_ball 아니면 companion=0
    pre_avel    (2,3): [protagonist_avel, companion_avel]
    post_vel    (2,2)
    post_avel   (2,3)
    pos         (2,2): [protagonist_pos, companion_pos]  — normalized [0,1]
    normal      (2,)  : contact normal (unit vector)
    coll_type   (1,)  : int, 0~6
    has_tgt     (1,)  : bool, ball_ball이면 1
    will_pocket (1,)  : bool, 이 샷에서 target ball이 포켓됐는지

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


def extract_collisions(system, will_pocket: int = 0):
    """Shot 하나에서 모든 충돌 이벤트의 pre/post 쌍을 추출.

    will_pocket: 이 shot에서 target ball이 포켓됐는지 (전 이벤트 공유 레이블).
    """
    samples = []

    for e in system.events:
        et = str(e.event_type)
        if et not in COLL_TYPES:
            continue

        coll_type = COLL_TYPES[et]
        balls = {a.id: a for a in e.agents
                 if getattr(a, "agent_type", "") == "ball"}

        cue = balls.get("cue")
        tgt = balls.get("1")

        # protagonist = slot[0], companion = slot[1]
        if cue is not None:
            protagonist, companion = cue, tgt
        elif tgt is not None:
            # cue ball 없이 target ball만 관여하는 이벤트 (쿠션 / 포켓)
            protagonist, companion = tgt, None
            if   et == "ball_linear_cushion":   coll_type = 5
            elif et == "ball_circular_cushion": coll_type = 6
            elif et == "ball_pocket":           coll_type = 3
            else:                               continue
        else:
            continue

        pro_pre_vel, pro_pre_avel, pro_post_vel, pro_post_avel, pro_pos = \
            _get_ball_pre_post(protagonist)

        has_tgt = (companion is not None) and (et == "ball_ball")
        if has_tgt:
            com_pre_vel, com_pre_avel, com_post_vel, com_post_avel, com_pos = \
                _get_ball_pre_post(companion)
        else:
            com_pre_vel   = np.zeros(2)
            com_pre_avel  = np.zeros(3)
            com_post_vel  = np.zeros(2)
            com_post_avel = np.zeros(3)
            com_pos       = np.zeros(2)

        # contact normal
        if et == "stick_ball":
            stick = next(a for a in e.agents
                         if getattr(a, "agent_type", "") == "cue")
            phi_rad = float(stick.initial.phi) * np.pi / 180.0
            normal = np.array([np.cos(phi_rad), np.sin(phi_rad)])
        elif et == "ball_ball":
            normal = _contact_normal_ball_ball(pro_pos, com_pos)
        elif et == "ball_linear_cushion":
            cush = next(a for a in e.agents
                        if getattr(a, "agent_type", "") == "linear_cushion_segment")
            normal = _contact_normal_linear(cush)
        elif et == "ball_circular_cushion":
            cush = next(a for a in e.agents
                        if getattr(a, "agent_type", "") == "circular_cushion_segment")
            normal = _contact_normal_circular(cush, pro_pos)
        elif et == "ball_pocket":
            pock = next(a for a in e.agents
                        if getattr(a, "agent_type", "") == "pocket")
            normal = _contact_normal_pocket(pock, pro_pos)
        else:
            normal = np.array([1.0, 0.0], dtype=np.float32)

        samples.append(dict(
            pre_vel     = np.stack([pro_pre_vel,  com_pre_vel],  axis=0),
            pre_avel    = np.stack([pro_pre_avel, com_pre_avel], axis=0),
            post_vel    = np.stack([pro_post_vel, com_post_vel], axis=0),
            post_avel   = np.stack([pro_post_avel, com_post_avel], axis=0),
            pos         = np.stack([pro_pos / [TABLE_W, TABLE_H],
                                    com_pos / [TABLE_W, TABLE_H]], axis=0),
            normal      = normal.astype(np.float32),
            coll_type   = coll_type,
            has_tgt     = int(has_tgt),
            will_pocket = will_pocket,
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
        will_pocket = int(bool(info.get("pocketed")))
        if will_pocket: pocketed += 1

        samples = extract_collisions(env.system, will_pocket=will_pocket)
        normalize(samples)
        for i, s in enumerate(samples):
            s["episode_id"] = ep
            s["event_idx"]  = i
        all_samples.extend(samples)

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
        "pre_vel"    : np.stack([s["pre_vel"]     for s in samples]),
        "pre_avel"   : np.stack([s["pre_avel"]    for s in samples]),
        "post_vel"   : np.stack([s["post_vel"]    for s in samples]),
        "post_avel"  : np.stack([s["post_avel"]   for s in samples]),
        "pos"        : np.stack([s["pos"]         for s in samples]),
        "normal"     : np.stack([s["normal"]      for s in samples]),
        "coll_type"  : np.array([s["coll_type"]   for s in samples], dtype=np.int8),
        "has_tgt"    : np.array([s["has_tgt"]     for s in samples], dtype=np.int8),
        "will_pocket": np.array([s["will_pocket"] for s in samples], dtype=np.int8),
        "episode_id" : np.array([s["episode_id"]  for s in samples], dtype=np.int32),
        "event_idx"  : np.array([s["event_idx"]   for s in samples], dtype=np.int8),
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
    names = {0: "ball_ball", 1: "cue_linear", 2: "cue_circular",
             3: "pocket", 4: "stick_ball", 5: "tgt_linear", 6: "tgt_circular"}
    print(f"\nTotal samples: {len(samples)}")
    for i, name in names.items():
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
