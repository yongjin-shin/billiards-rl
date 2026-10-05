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
from exp16_wm.sac import VanillaSAC, WMSAC
from exp16_wm.train import ACT_HIGH, ACT_LOW, build_agent


METHODS = ("vanilla", "traj", "rssm")


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
    """Latest best-ckpt-MA run dir for (method, seed) — 2026-10-02/03 batch."""
    agent = "vanilla" if method == "vanilla" else "wm"
    cands = []
    for d in sorted(glob.glob(f"{root}/exp16_{agent}_multi3_ms5_s{seed}_2026-10-0[23]@*")):
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg.get("best_ckpt_window") is None:
            continue
        if agent == "wm" and cfg.get("wm_target") != method:
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
    x = torch.cat([o, a], dim=-1)
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

def diagnose_seed(seed: int, n_shots: int, n_episodes: int) -> dict:
    agents, cfg = {}, None
    for m in METHODS:
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
        if m != "vanilla":
            h_hat = critic_features(ag, data["obs"], data["act"])
            out[f"{m}_h_r2"] = r2_score_flat(h_hat, data[f"h_{m}"].reshape(len(y), -1))
        qv, gv, tv = own_policy_rollouts(ag, env_plain, n_episodes, cfg["gamma"], seed=2000 + seed)
        out[f"{m}_ret_pearson"] = float(stats.pearsonr(qv, gv).statistic)
        out[f"{m}_ret_spearman"] = float(stats.spearmanr(qv, gv).statistic)
        out[f"{m}_ret_pearson_within_t"] = within_step_pearson(qv, gv, tv)
        out[f"{m}_ret_mse"] = float(((qv - gv) ** 2).mean())
        out[f"{m}_ret_bias"] = float((qv - gv).mean())
    env.close(); env_plain.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-shots", type=int, default=3000)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--out", type=str, default="logs/diagnose_q.json")
    args = p.parse_args()

    results = []
    for s in args.seeds:
        r = diagnose_seed(s, args.n_shots, args.n_episodes)
        print(json.dumps(r, indent=1), flush=True)
        results.append(r)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
