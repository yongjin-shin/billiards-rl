"""
exp16_wm/diagnose_q.py — Q-quality diagnostics for Exp-16 (vanilla vs WM traj vs WM rssm).

Asks directly whether the world-model bottleneck ĥ gives the critic a better
handle on "does this shot pocket a ball", instead of inferring it from the
policy's pocket rate. No training — only saved best checkpoints are used.

  D1  pocket discrimination : AUC of Q(s,a) for this shot's real outcome
                              (pocketed_this_step > 0) on a shared (s,a) set per seed
  D2  value accuracy        : corr / MSE between Q(s_t,a_t) and the real discounted
                              return G_t along each model's own policy
  D3  what ĥ holds          : (a) R² of ĥ vs its own target h_real
                              (b) linear-probe CV AUC for the pocket label on
                                  raw [s,a] / vanilla hidden / wm ĥ / real h_real

Usage:
    python -m exp16_wm.diagnose_q --seeds 0 1 2 3 --n-shots 3000
"""

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pooltool as pt
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import BilliardsEnv, _extract_trajectory
from exp16_wm.sac import GeomSAC, VanillaSAC, WMSAC
from exp16_wm.train import ACT_HIGH, ACT_LOW, build_agent
from world_model.event_detector import EventDetector
from world_model.rssm_encode import load_frozen_rssm, real_event_steps
from world_model.rssm_model import EVENT_BALL_BALL, EVENT_POCKET
from world_model.rssm_rollout import RolloutEngine


METHODS = ("vanilla", "traj", "rssm")
# runs that exist only from the G/M experiments (2026-10-06~); pass via --methods
EXTRA_METHODS = ("geom", "hmix")


# ──────────────────────────────────────────────
# Pure metric helpers
# ──────────────────────────────────────────────

def safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC, or nan when only one class is present."""
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=float)))


def r2_score_flat(pred: np.ndarray, target: np.ndarray) -> float:
    """R² over all elements (1 - SSE/SST); nan if the target is constant."""
    pred, target = np.asarray(pred, float).ravel(), np.asarray(target, float).ravel()
    sst = float(((target - target.mean()) ** 2).sum())
    if sst == 0.0:
        return float("nan")
    return 1.0 - float(((pred - target) ** 2).sum()) / sst


def probe_auc(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0,
              nonlinear: bool = False) -> float:
    """Stratified k-fold CV AUC of a standardized logistic-regression (or small MLP) probe."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max() or min(np.bincount(y)) < folds:
        return float("nan")
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        head = (MLPClassifier((128, 128), max_iter=500, early_stopping=True, random_state=seed)
                if nonlinear else LogisticRegression(max_iter=2000, C=1.0))
        clf = make_pipeline(StandardScaler(), head)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return safe_auc(oof, y)


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """G_t = r_t + γ r_{t+1} + ... for one episode."""
    out, g = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


# ──────────────────────────────────────────────
# Model loading / critic access
# ──────────────────────────────────────────────

def find_run(method: str, seed: int, root: str = "logs/experiments") -> str:
    """
    Latest best-ckpt-MA run dir for (method, seed): vanilla/traj/rssm from the
    2026-10-02/03 batch, geom (Exp-16 G) and hmix (Exp-16 M, WM rssm + h mixing).
    """
    pattern = {"vanilla": "exp16_vanilla_multi3", "geom": "exp16_geom_multi3",
               "hmix": "exp16_wm_hmix*_multi3"}.get(method, "exp16_wm_multi3")
    date = "2026-10-0[23]" if method in METHODS else "2026-*"
    cands = []
    for d in sorted(glob.glob(f"{root}/{pattern}_ms5_s{seed}_{date}@*")):
        if not os.path.exists(os.path.join(d, "results.json")):
            continue   # unfinished run
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg.get("best_ckpt_window") is None:
            continue
        if method in ("traj", "rssm") and cfg.get("wm_target") != method:
            continue
        cands.append(d)
    if not cands:
        raise FileNotFoundError(f"no run for {method} seed={seed}")
    return cands[-1]


def load_agent(run_dir: str, ckpt: str = "best_model/best_model.pt") -> tuple[VanillaSAC, dict]:
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    args = SimpleNamespace(**cfg)
    args.wm_target = cfg.get("wm_target", "traj")
    agent = build_agent(args)
    agent.load(os.path.join(run_dir, ckpt))
    agent.critic.eval()
    return agent, cfg


@torch.no_grad()
def critic_q(agent: VanillaSAC, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    o, a = agent._to_tensor(obs), agent._to_tensor(act)
    return agent.critic.q_min(o, a).squeeze(-1).cpu().numpy()


@torch.no_grad()
def critic_features(agent: VanillaSAC, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Representation Q is read from: vanilla Q1 last hidden layer, WM ĥ1 (flattened)."""
    o, a = agent._to_tensor(obs), agent._to_tensor(act)
    if isinstance(agent, WMSAC):
        return agent.critic.M1(o, a).flatten(1).cpu().numpy()
    x = agent.critic.inputs(o, a) if isinstance(agent, GeomSAC) else torch.cat([o, a], dim=-1)
    return agent.critic.Q1[:-1](x).cpu().numpy()


# ──────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────

def make_env(cfg: dict) -> BilliardsEnv:
    env = BilliardsEnv(
        n_balls       = cfg["n_balls"],
        max_steps     = cfg["max_steps"],
        step_penalty  = cfg["step_penalty"],
        trunc_penalty = cfg["trunc_penalty"],
    )
    env.wm_target = "rssm"   # info["h_real"] = R-SSM latent; traj extracted separately
    return env


def behaviour_action(policy_action: np.ndarray, rng: np.random.Generator,
                     p_random: float = 0.4) -> np.ndarray:
    """Mix uniform-random shots with noisy policy shots so both hits and misses appear."""
    if rng.random() < p_random:
        return rng.uniform(ACT_LOW, ACT_HIGH).astype(np.float32)
    sigma = np.array([rng.choice([0.0, 0.05, 0.2]), 0.5])
    return np.clip(policy_action + rng.normal(0.0, sigma), ACT_LOW, ACT_HIGH).astype(np.float32)


def collect_shared_dataset(env: BilliardsEnv, policies: list[Callable[[np.ndarray], np.ndarray]],
                           n_shots: int, seed: int) -> dict[str, np.ndarray]:
    """
    One shot per row: obs, action, pocket label, reward, real h for both WM targets.
    Each episode follows a randomly chosen policy (so no single model owns the
    state distribution) through `behaviour_action`.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {k: [] for k in ("obs", "act", "label", "reward", "h_traj", "h_rssm")}
    env.reset(seed=seed)
    while len(rows["label"]) < n_shots:
        obs, _ = env.reset()
        policy = policies[rng.integers(len(policies))]
        done = False
        while not done and len(rows["label"]) < n_shots:
            a = behaviour_action(policy(obs), rng)
            next_obs, r, term, trunc, info = env.step(a)
            h_traj, _ = _extract_trajectory(env.system, target_id=env._ball_ids[0])
            rows["obs"].append(obs); rows["act"].append(a)
            rows["label"].append(int(info["pocketed_this_step"] > 0))
            rows["reward"].append(r)
            rows["h_traj"].append(h_traj); rows["h_rssm"].append(info["h_real"])
            obs, done = next_obs, term or trunc
    return {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}


def own_policy_rollouts(agent: VanillaSAC, env: BilliardsEnv, n_episodes: int,
                        gamma: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Q(s_t,a_t), G_t, t) along the agent's deterministic policy."""
    env.reset(seed=seed)
    qs, gs, ts = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_obs, ep_act, ep_r = False, [], [], []
        while not done:
            a = agent.act(obs, deterministic=True)
            ep_obs.append(obs); ep_act.append(a)
            obs, r, term, trunc, _ = env.step(a)
            ep_r.append(r); done = term or trunc
        qs.append(critic_q(agent, np.array(ep_obs), np.array(ep_act)))
        gs.append(discounted_returns(ep_r, gamma))
        ts.append(np.arange(len(ep_r)))
    return np.concatenate(qs), np.concatenate(gs), np.concatenate(ts)


def within_step_pearson(q: np.ndarray, g: np.ndarray, t: np.ndarray, min_n: int = 20) -> float:
    """Mean Pearson(Q, G) computed separately at each shot index t (removes horizon effects)."""
    rs = [stats.pearsonr(q[t == k], g[t == k]).statistic
          for k in np.unique(t) if (t == k).sum() >= min_n and np.std(g[t == k]) > 0]
    return float(np.mean(rs)) if rs else float("nan")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def diagnose_seed(seed: int, n_shots: int, n_episodes: int,
                  methods: tuple[str, ...] = METHODS) -> dict:
    agents, cfg = {}, None
    for m in methods:
        agents[m], cfg = load_agent(find_run(m, seed))
    env = make_env(cfg)
    env_plain = make_env(cfg); env_plain.wm_target = "none"

    data = collect_shared_dataset(
        env, [lambda o, ag=ag: ag.act(o, deterministic=True) for ag in agents.values()],
        n_shots, seed=1000 + seed,
    )
    y, sa = data["label"], np.concatenate([data["obs"], data["act"]], axis=1)
    out: dict = {"seed": seed, "n_shots": len(y), "pocket_frac": float(y.mean()),
                 "probe_raw_sa": probe_auc(sa, y),
                 "probe_raw_sa_mlp": probe_auc(sa, y, nonlinear=True),
                 "probe_real_h_traj": probe_auc(data["h_traj"], y),
                 "probe_real_h_rssm": probe_auc(data["h_rssm"], y)}
    for m, ag in agents.items():
        q = critic_q(ag, data["obs"], data["act"])
        out[f"{m}_q_auc"] = safe_auc(q, y)
        out[f"{m}_q_reward_spearman"] = float(stats.spearmanr(q, data["reward"]).statistic)
        out[f"{m}_probe_feat"] = probe_auc(critic_features(ag, data["obs"], data["act"]), y)
        if isinstance(ag, WMSAC):
            target = "traj" if m == "traj" else "rssm"
            h_hat = critic_features(ag, data["obs"], data["act"])
            out[f"{m}_h_r2"] = r2_score_flat(h_hat, data[f"h_{target}"].reshape(len(y), -1))
        qv, gv, tv = own_policy_rollouts(ag, env_plain, n_episodes, cfg["gamma"], seed=2000 + seed)
        out[f"{m}_ret_pearson"] = float(stats.pearsonr(qv, gv).statistic)
        out[f"{m}_ret_spearman"] = float(stats.spearmanr(qv, gv).statistic)
        out[f"{m}_ret_pearson_within_t"] = within_step_pearson(qv, gv, tv)
        out[f"{m}_ret_mse"] = float(((qv - gv) ** 2).mean())
        out[f"{m}_ret_bias"] = float((qv - gv).mean())
    env.close(); env_plain.close()
    return out


# ──────────────────────────────────────────────
# D4 — pre-shot pocket prediction: R-SSM free rollout vs critic
# ──────────────────────────────────────────────

BALL_R = 0.028575


def preshot_balls(system) -> dict[str, tuple[np.ndarray, int]]:
    """Balls still on the table before a shot, as {id: (rvw, state)} with zero velocity."""
    out = {}
    for bid, ball in system.balls.items():
        if ball.state.s == pt.constants.pocketed:
            continue
        rvw = ball.state.rvw.copy()
        rvw[1:] = 0.0
        out[bid] = (rvw, pt.constants.stationary)
    return out


def poststrike_cue_rvw(system) -> np.ndarray:
    """Cue rvw right after the stick hit, read from the shot's `stick_ball` event."""
    ev = next(e for e in system.events if str(e.event_type) == "stick_ball")
    cue = next(a for a in ev.agents if getattr(a, "agent_type", "") == "ball")
    return np.array([cue.initial.state.rvw[0], list(cue.final.vel), list(cue.final.avel)],
                    dtype=np.float64)


def geometric_preshot_score(cue_pos: np.ndarray, cue_vel: np.ndarray,
                            targets: dict[str, np.ndarray], pockets: np.ndarray,
                            r: float = BALL_R) -> float:
    """
    Zero-parameter pre-shot baseline: straight cue ray → first target ball hit →
    that ball leaves along the line of centers → -(closest approach to any pocket).
    No friction, cushions or spin. Misses every ball → -10.
    """
    speed = np.linalg.norm(cue_vel[:2])
    if speed < 1e-9:
        return -10.0
    d = cue_vel[:2] / speed
    best_t, best_pos = np.inf, None
    for pos in targets.values():
        rel = pos[:2] - cue_pos[:2]
        proj = float(rel @ d)
        perp2 = float(rel @ rel) - proj ** 2
        if proj <= 0 or perp2 >= (2 * r) ** 2:
            continue
        t_hit = proj - np.sqrt((2 * r) ** 2 - perp2)
        if t_hit < best_t:
            best_t, best_pos = t_hit, pos[:2]
    if best_pos is None:
        return -10.0
    n = best_pos - (cue_pos[:2] + best_t * d)
    n = n / np.linalg.norm(n)
    rel = pockets - best_pos
    t = np.clip(rel @ n, 0.0, None)
    return -float(np.min(np.linalg.norm(pockets - (best_pos + t[:, None] * n), axis=1)))


@torch.no_grad()
def first_touch_pocket_probs(model, n_balls: int, events: list) -> dict[int, float]:
    """Teacher-force `events` through R-SSM; pocket prob of each ball at its first event."""
    h = model.init_hidden(n_balls)
    probs: dict[int, float] = {}
    for ev in events:
        if ev.event_type == EVENT_BALL_BALL and ev.ball_j is not None:
            h, *_ = model.step_ball_ball(h, ev.ball_i, ev.ball_j, ev.node_i, ev.node_j, ev.edge)
        else:
            h, *_ = model.step_single(h, ev.ball_i, ev.node_i, ev.normal)
        for b in (ev.ball_i, ev.ball_j):
            if b is not None and b not in probs:
                probs[b] = float(model.predict_pocket(h[b].unsqueeze(0)).item())
    return probs


def target_max(probs: dict[int, float], target_idx: list[int]) -> float:
    """Shot-level score = most pocket-likely target ball; untouched balls count as 0."""
    return max([probs.get(i, 0.0) for i in target_idx] + [0.0])


class CueSafeDetector(EventDetector):
    """
    EventDetector builds a pooltool System per query, which needs the cue ball
    present and at least two balls (pooltool's ball-ball search fails on an empty
    pair set). Once a rollout pockets the cue or all but one ball, pad with
    stationary balls parked far off the table — they never take part in an event.
    """

    _FAR = np.array([[-50.0, -50.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def next_event(self, balls: dict):
        padded = dict(balls)
        if "cue" not in padded:
            padded["cue"] = (self._FAR.copy(), pt.constants.stationary)
        if len(padded) < 2:
            far = self._FAR.copy(); far[0, 0] -= 10.0
            padded["_pad"] = (far, pt.constants.stationary)
        return super().next_event(padded)


def collect_d4_dataset(env: BilliardsEnv, policies: list[Callable[[np.ndarray], np.ndarray]],
                       n_shots: int, seed: int, rssm_model) -> dict[str, np.ndarray]:
    """Like collect_shared_dataset, plus R-SSM rollout / teacher-forced / geometric scores per shot."""
    ball_ids = ["cue"] + env._ball_ids
    pockets = np.array([p.center[:2] for p in env.table.pockets.values()])
    rng = np.random.default_rng(seed)
    keys = ("obs", "act", "label", "rollout_n_pocket", "rollout_head", "tf_head", "geom")

    # After a scratch the env rebuilds `system` (ball-in-hand), dropping this shot's
    # events — snapshot the simulated system right before that happens.
    shot_system: dict = {}
    respawn = env._respawn_cue
    def _snapshot_then_respawn() -> None:
        shot_system["sys"] = env.system
        respawn()
    env._respawn_cue = _snapshot_then_respawn

    rows: dict[str, list] = {k: [] for k in keys}
    env.reset(seed=seed)
    while len(rows["label"]) < n_shots:
        obs, _ = env.reset()
        policy = policies[rng.integers(len(policies))]
        done = False
        while not done and len(rows["label"]) < n_shots:
            pre = preshot_balls(env.system)
            a = behaviour_action(policy(obs), rng)
            shot_system.clear()
            next_obs, _, term, trunc, info = env.step(a)
            sim = shot_system.get("sys", env.system)

            cue_rvw = poststrike_cue_rvw(sim)
            balls = dict(pre); balls["cue"] = (cue_rvw, pt.constants.sliding)
            target_idx = [ball_ids.index(b) for b in pre if b != "cue"]
            engine = RolloutEngine(rssm_model, CueSafeDetector(sim.table, sim.cue), ball_ids)
            with torch.no_grad():
                res = engine.run(balls)
            rows["rollout_n_pocket"].append(sum(
                1 for ev in res.event_steps if ev.event_type == EVENT_POCKET and ev.ball_i in target_idx))
            rows["rollout_head"].append(target_max(
                first_touch_pocket_probs(rssm_model, len(ball_ids), res.event_steps), target_idx))
            rows["tf_head"].append(target_max(first_touch_pocket_probs(
                rssm_model, len(ball_ids), real_event_steps(sim, ball_ids)), target_idx))
            rows["geom"].append(geometric_preshot_score(
                cue_rvw[0], cue_rvw[1], {b: v[0][0] for b, v in pre.items() if b != "cue"}, pockets))
            rows["obs"].append(obs); rows["act"].append(a)
            rows["label"].append(int(info["pocketed_this_step"] > 0))
            obs, done = next_obs, term or trunc
    return {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}


def diagnose_seed_d4(seed: int, n_shots: int) -> dict:
    agents, cfg = {}, None
    for m in METHODS:
        agents[m], cfg = load_agent(find_run(m, seed))
    env = make_env(cfg); env.wm_target = "none"
    rssm_model = load_frozen_rssm(env.rssm_checkpoint)
    data = collect_d4_dataset(
        env, [lambda o, ag=ag: ag.act(o, deterministic=True) for ag in agents.values()],
        n_shots, seed=3000 + seed, rssm_model=rssm_model,
    )
    env.close()
    y = data["label"]
    out: dict = {"seed": seed, "n_shots": len(y), "pocket_frac": float(y.mean())}
    for m, ag in agents.items():
        out[f"{m}_q_auc"] = safe_auc(critic_q(ag, data["obs"], data["act"]), y)
    for k in ("rollout_n_pocket", "rollout_head", "tf_head", "geom"):
        out[f"{k}_auc"] = safe_auc(data[k], y)
    out["rollout_pocket_any_acc"] = float(((data["rollout_n_pocket"] > 0) == (y > 0)).mean())
    out["rollout_pocket_any_rate"] = float((data["rollout_n_pocket"] > 0).mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-shots", type=int, default=3000)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--d4", action="store_true", help="run D4 (R-SSM rollout vs critic) instead")
    p.add_argument("--methods", type=str, nargs="+", default=list(METHODS),
                   choices=list(METHODS + EXTRA_METHODS))
    p.add_argument("--out", type=str, default="logs/diagnose_q.json")
    args = p.parse_args()

    results = []
    for s in args.seeds:
        r = (diagnose_seed_d4(s, args.n_shots) if args.d4
             else diagnose_seed(s, args.n_shots, args.n_episodes, tuple(args.methods)))
        print(json.dumps(r, indent=1), flush=True)
        results.append(r)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
