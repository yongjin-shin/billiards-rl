"""
exp16_wm/goal_gnn.py — A0: the pair structure of the pocket GNN policy (exp16_wm/gnn_policy.py) with arbitrary goal points instead of pockets.

Same building blocks as PairGNNPolicy: a ball node embedding h_i, a (ball i, target j) pair embedding e_ij from a shared MLP on
[h_i, target_j, target_j − ball_i, |·|], a proposal head on e_ij (angle, speed) and a made2 head P(the action reaches the target | e_ij, action).
What differs (experiments.md "A0"):
  - the targets are any points (G per state), not the 6 pocket coordinates; the ball is the cue ball itself;
  - absolute table coordinates and an absolute angle (no "nearest ball is +x" frame — an empty table has no such ball);
  - the action is (cos angle, sin angle, log speed) with no speed floor, and the proposal loss is the one of exp16_wm.cue_goal_head;
  - no ball ↔ ball messages: there is no other ball on the table.
made2 is trained from the recorded shots alone: for a goal (the end of shot i) and the action of another shot j of the same start, the label is
whether shot j also ended within 2 cm of that goal. Inference "verify + refine" scores a grid of actions around the proposal with made2.

Usage:
    python -m exp16_wm.goal_gnn train --data logs/ct0b/collect_200.npz --out logs/a0/run0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.cue_goal_data import TABLE_L, TABLE_W
from exp16_wm.cue_goal_head import ANGLE_TOL_RAD, LOGV_TOL, SCALE, V_CLIP, eval_cases, load_rows, loss_fn, play_error, summarize, targets
from exp16_wm.gnn_policy import _mlp

MADE_TOL = 0.02                                   # made2 label: ended within 2 cm of the goal
REFINE_ANGLES = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.02), 4)          # degrees around the proposal (as in the pocket pipeline)
REFINE_LOGV = np.linspace(-0.02, 0.02, 9)                               # log-speed offsets around the proposal (±2 %)


class GoalPairGNN(nn.Module):
    def __init__(self, hidden: int = 256, enc_layers: int = 4):
        super().__init__()
        H = hidden
        self.ball = _mlp(2 + 1, H, H, enc_layers)                       # cue ball node: position (table fractions) and its distance to the table centre
        self.pair = _mlp(H + 2 + 2 + 1, H, H, enc_layers)               # [h_i, target (fractions), target − ball (m), |·|] -> e_ij
        self.head_a = nn.Linear(H, 3)                                   # proposal: cos, sin (unit-scale) and standardised log speed
        self.head_made2 = _mlp(H + 3 + 2, H, 1, 2)                      # [e_ij, action, (action − proposal) in tolerance units] -> logit
        self.register_buffer("lv_mean", torch.zeros(())); self.register_buffer("lv_std", torch.ones(()))

    def encode(self, ball: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """ball (B, 2) metres, goals (B, G, 2) metres -> pair embeddings e (B, G, H). Any number of goals G."""
        bf = ball / ball.new_tensor(SCALE)
        h = self.ball(torch.cat([bf, (bf - 0.5).norm(dim=-1, keepdim=True)], dim=-1))            # (B, H)
        d = goals - ball[:, None, :]
        gin = torch.cat([h[:, None, :].expand(-1, goals.shape[1], -1), goals / ball.new_tensor(SCALE), d, d.norm(dim=-1, keepdim=True)], dim=-1)
        return self.pair(gin)

    def propose(self, e: torch.Tensor) -> torch.Tensor:
        """(…, 3): cos, sin, log speed."""
        out = self.head_a(e)
        return torch.cat([out[..., :2], out[..., 2:] * self.lv_std + self.lv_mean], dim=-1)

    def made2_logit(self, e: torch.Tensor, action: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """action, prop (…, 3) as (cos, sin, log speed); the difference to the proposal is in units of the tolerances (0.1 deg, 0.5 % speed)."""
        a, p = action, prop.detach()
        cross = p[..., 0] * a[..., 1] - p[..., 1] * a[..., 0]
        dot = p[..., 0] * a[..., 0] + p[..., 1] * a[..., 1]
        d_ang = (torch.atan2(cross, dot) / ANGLE_TOL_RAD).clamp(-50, 50) / 10.0
        d_lv = ((a[..., 2] - p[..., 2]) / LOGV_TOL).clamp(-50, 50) / 10.0
        feats = torch.cat([e, a[..., :2], ((a[..., 2:] - self.lv_mean) / self.lv_std), d_ang[..., None], d_lv[..., None]], dim=-1)
        return self.head_made2(feats).squeeze(-1)


def build_pairs(data: dict, start_ids: np.ndarray, n_rows: int, seed: int) -> dict:
    """
    Training / evaluation rows from the recorded shots of the given starts: goal = end of shot i, proposal target = action of i, made2 candidate = action
    of shot j of the same start (j: the rank-th nearest end to the goal, rank log-uniform up to 256; a quarter are random shots), label = j also ended
    within MADE_TOL of the goal.
    """
    rng = np.random.default_rng(seed)
    per = max(1, n_rows // len(start_ids))
    out = {k: [] for k in ("start", "goal", "phi_i", "v_i", "phi_j", "v_j", "label")}
    for s in start_ids:
        pool = np.nonzero((data["sidx"] == s) & data["ok"])[0]
        ends = data["end"][pool].astype(np.float64)
        i = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        kk = min(256, len(pool))
        _, nb = cKDTree(ends).query(ends[i], k=kk)
        rank = np.minimum((kk ** rng.random(len(i))).astype(int) - 1, kk - 1).clip(0)
        j = nb[np.arange(len(i)), rank]
        far = rng.random(len(i)) < 0.25
        j = np.where(far, rng.integers(0, len(pool), len(i)), j)
        out["start"].append(np.repeat(data["starts"][s][None], len(i), axis=0)); out["goal"].append(ends[i])
        out["phi_i"].append(data["phi"][pool[i]]); out["v_i"].append(data["v"][pool[i]])
        out["phi_j"].append(data["phi"][pool[j]]); out["v_j"].append(data["v"][pool[j]])
        out["label"].append(np.linalg.norm(ends[j] - ends[i], axis=1) <= MADE_TOL)
    return {k: np.concatenate(v) for k, v in out.items()}


def to_tensors(rows: dict) -> dict[str, torch.Tensor]:
    f = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    return {"start": f(rows["start"]), "goal": f(rows["goal"]), "act_i": torch.from_numpy(targets(rows["phi_i"], rows["v_i"])),
            "act_j": torch.from_numpy(targets(rows["phi_j"], rows["v_j"])), "label": f(rows["label"])}


def train(rows: dict, epochs: int, batch: int, lr: float, hidden: int, enc_layers: int, made_weight: float, seed: int, log=print) -> GoalPairGNN:
    torch.manual_seed(seed)
    t = to_tensors(rows)
    model = GoalPairGNN(hidden, enc_layers)
    model.lv_mean.copy_(t["act_i"][:, 2].mean()); model.lv_std.copy_(t["act_i"][:, 2].std())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(t["label"]); steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n); tp = tm = 0.0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            e = model.encode(t["start"][idx], t["goal"][idx][:, None, :])[:, 0]
            prop = model.propose(e)
            lp = loss_fn(prop, t["act_i"][idx])
            lm = F.binary_cross_entropy_with_logits(model.made2_logit(e, t["act_j"][idx], prop), t["label"][idx])
            loss = lp + made_weight * lm
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tp += float(lp.detach()) * len(idx); tm += float(lm.detach()) * len(idx)
        log(f"[a0] epoch {ep + 1}/{epochs} proposal loss {tp / n:.4f}  made2 bce {tm / n:.4f}  {time.time() - t0:.0f}s")
    return model


def _action_np(out: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    o = out.detach().numpy().astype(np.float64)
    return np.degrees(np.arctan2(o[:, 1], o[:, 0])) % 360.0, np.clip(np.exp(o[:, 2]), *V_CLIP)


def propose_actions(model: GoalPairGNN, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        e = model.encode(torch.from_numpy(start.astype(np.float32)), torch.from_numpy(goal.astype(np.float32))[:, None, :])
        return _action_np(model.propose(e)[:, 0])


def refine_actions(model: GoalPairGNN, start: np.ndarray, goal: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Verify + refine: made2 scores every action of a grid around the proposal (31 angles ± 0.3 deg × 9 speeds ± 2 %); the best one is played."""
    phi_out, v_out = np.zeros(len(start)), np.zeros(len(start))
    da = torch.from_numpy(np.radians(REFINE_ANGLES).astype(np.float32)); dv = torch.from_numpy(REFINE_LOGV.astype(np.float32))
    for b in range(0, len(start), chunk):
        s = torch.from_numpy(start[b:b + chunk].astype(np.float32)); g = torch.from_numpy(goal[b:b + chunk].astype(np.float32))
        with torch.no_grad():
            e = model.encode(s, g[:, None, :])[:, 0]
            prop = model.propose(e)
            ang = torch.atan2(prop[:, 1], prop[:, 0])[:, None, None] + da[None, :, None]            # (B, 31, 1)
            lv = prop[:, 2][:, None, None] + dv[None, None, :]                                       # (B, 1, 9)
            cand = torch.stack([torch.cos(ang).expand(-1, -1, len(dv)), torch.sin(ang).expand(-1, -1, len(dv)), lv.expand(-1, len(da), -1)], dim=-1)
            cand = cand.reshape(len(s), -1, 3)                                                       # (B, 279, 3)
            logit = model.made2_logit(e[:, None, :].expand(-1, cand.shape[1], -1), cand, prop[:, None, :].expand(-1, cand.shape[1], -1))
            best = cand[torch.arange(len(s)), logit.argmax(dim=1)]
        phi_out[b:b + chunk], v_out[b:b + chunk] = _action_np(best)
    return phi_out, v_out


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve by the rank statistic (ties get the mean rank)."""
    from scipy.stats import rankdata
    label = label.astype(bool)
    if label.all() or not label.any():
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))


def made2_auc(model: GoalPairGNN, rows: dict) -> float:
    t = to_tensors(rows)
    with torch.no_grad():
        e = model.encode(t["start"], t["goal"][:, None, :])[:, 0]
        logit = model.made2_logit(e, t["act_j"], model.propose(e))
    return auc(logit.numpy(), t["label"].numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--k-bin", type=int, default=0); t.add_argument("--holdout", type=int, default=40)
    t.add_argument("--n-train", type=int, default=1_500_000); t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=4096); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=256); t.add_argument("--enc-layers", type=int, default=4)
    t.add_argument("--made-weight", type=float, default=10.0)
    t.add_argument("--n-goals", type=int, default=100); t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from exp16_wm.code_version import code_version
    os.makedirs(args.out, exist_ok=True)
    data = load_rows(args.data, args.k_bin, args.holdout, args.n_train, args.seed)
    rows = build_pairs(data, data["train_starts"], args.n_train, args.seed)
    print(f"[a0] {len(rows['label']):,} rows from {len(data['train_starts'])} starts; made2 positive share {rows['label'].mean():.3f}", flush=True)
    model = train(rows, args.epochs, args.batch, args.lr, args.hidden, args.enc_layers, args.made_weight, args.seed, log=lambda m: print(m, flush=True))
    torch.save({"state": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "model.pt"))
    result = {"commit": code_version(), "args": vars(args), "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    for name, ids in (("new_starts", data["holdout"]), ("train_starts", data["train_starts"][: len(data["holdout"])])):
        s, g = eval_cases(data["starts"], ids, args.n_goals, 1000 + args.seed)
        r = {"proposal": summarize(play_error(s, g, *propose_actions(model, s, g))),
             "verify_refine": summarize(play_error(s, g, *refine_actions(model, s, g)))}
        r["made2_auc"] = made2_auc(model, build_pairs(data, ids, 200_000, 77 + args.seed))
        result[name] = r
        print(name, json.dumps(r), flush=True)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
