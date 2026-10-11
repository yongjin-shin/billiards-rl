"""
exp16_wm/gnn_policy.py — (ball, pocket)-pair GNN policy for imitation (Exp-16).

Input is the action-frame relative observation (exp16_wm/obs_transform.py, 21-dim:
3 balls × (rx, ry, pocketed) + 6 pockets × (px, py); cue at the origin, cue →
nearest ball along +x). Instead of one flat MLP over those 21 numbers:

  ball node   : [rx, ry, |r|, pocketed, is_ref]   (is_ref = the nearest live ball,
                the direction delta_angle is measured from)
  ball ↔ ball : shared message MLP on [ball_i, ball_k − ball_i, |·|, pocketed_k],
                summed over the other live balls → ball embedding h_i
  ball → pocket pair (i, j): shared MLP on [h_i, pocket_j, pocket_j − ball_i, |·|, side_j]
                → e_ij, masked when ball i is pocketed

Relations are given as plain differences (no billiards formulas) and every pair
shares weights, so what is learned for one (ball, pocket) applies to all 18.
Each pair proposes one action and a selection score; the deterministic action is
the top-scored pair's — several right answers live in different pairs instead of
being averaged into a miss. Trained with relaxed winner-takes-all: the pair
nearest to a pocketing candidate regresses to it (weight 1 − ε, others ε) and
the selection score is trained by cross-entropy to pick that pair.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp16_wm.train import ACT_LOW, ACT_HIGH
from exp16_wm.obs_transform import N_BALLS, N_POCKETS, REL_DIM

N_PAIRS = N_BALLS * N_POCKETS
SIDE_POCKETS = (1, 4)            # obs pocket order: (0,0) (0,½) (0,1) (1,0) (1,½) (1,1)
_NEG = -1e9


def wrap_angle_unit(x: torch.Tensor) -> torch.Tensor:
    """Angle in units of π (a full turn = 2) → the equivalent value in [-1, 1]."""
    return x - 2.0 * torch.round(x / 2.0)


ANGLE_WRAP = True        # the training CLI (bc.py) sets this from --angle-wrap (off by default: worse for the policy)


def action_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a − b for (…, 2) actions in [-1, 1] units; the angle is circular, so its difference is wrapped."""
    d = a - b
    return torch.stack([wrap_angle_unit(d[..., 0]) if ANGLE_WRAP else d[..., 0], d[..., 1]], dim=-1)


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class PairGNNPolicy(nn.Module):
    def __init__(self, hidden: int = 128, wta_eps: float = 0.05, sel_weight: float = 10.0,
                 factorized: bool = False, scratch_head: bool = False, made_head: bool = False):
        super().__init__()
        H = hidden
        self.wta_eps, self.sel_weight = wta_eps, sel_weight
        self.factorized = factorized
        self.scratch_lambda = 0.0         # inference: penalise pairs the scratch head expects to scratch
        self.fact_terms = ("reg", "sel", "scratch")     # factorized loss terms that are switched on (ablations)
        self.mod: tuple[str, str, bool] | None = None   # modular loss (sel_target, reg_target, use_scratch); see _mod_loss
        self.has_scratch = factorized or scratch_head
        self.wta_fallback = True                        # pair supervision: rows with no labelled pair use WTA (False → left out)
        self.has_made = made_head                       # auxiliary only (modular loss): P(pair made) from pair_stats
        self.bb   = _mlp(5 + 2 + 1 + 1, H, H)
        self.ball = _mlp(5 + H, H, H)
        self.pair = _mlp(H + 2 + 2 + 1 + 1, H, H)
        self.head_a = nn.Linear(H, 2)
        self.head_s = nn.Linear(H, 1)             # global mode: pair score; factorized mode: pocket logit within a ball
        if factorized:
            self.head_ball = nn.Linear(2 * H, 1)  # which ball: [h_i, max_j e_ij]
        if self.has_scratch:
            self.head_scratch = nn.Linear(H, 1)   # will this pair's shot pocket the cue ball
        if made_head:
            self.head_made = nn.Linear(H, 1)      # can this pair be made (coarse-scan success rate) — look-ahead auxiliary
        side = torch.zeros(N_POCKETS)
        side[list(SIDE_POCKETS)] = 1.0
        self.register_buffer("side", side)
        self.register_buffer("act_scale", torch.tensor((ACT_HIGH - ACT_LOW) / 2.0, dtype=torch.float32))
        self.register_buffer("act_bias", torch.tensor((ACT_HIGH + ACT_LOW) / 2.0, dtype=torch.float32))

    # ── encoder ───────────────────────────────

    def _encode(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, e, valid = self._encode_full(rel)
        return e, valid

    def _encode_full(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = rel.shape[0]
        balls = rel[:, :3 * N_BALLS].reshape(B, N_BALLS, 3)
        pos, pocketed = balls[..., :2], balls[..., 2]
        live = pocketed < 0.5
        dist = pos.norm(dim=-1)
        ref = torch.where(live, dist, torch.full_like(dist, float("inf"))).argmin(dim=1)
        is_ref = F.one_hot(ref, N_BALLS).float() * live.any(dim=1, keepdim=True).float()
        ball_f = torch.cat([pos, dist[..., None], pocketed[..., None], is_ref[..., None]], dim=-1)

        diff = pos[:, None, :, :] - pos[:, :, None, :]                    # (B, i, k, 2): k relative to i
        bb_in = torch.cat([ball_f[:, :, None, :].expand(B, N_BALLS, N_BALLS, 5), diff,
                           diff.norm(dim=-1, keepdim=True),
                           pocketed[:, None, :, None].expand(B, N_BALLS, N_BALLS, 1)], dim=-1)
        not_self = ~torch.eye(N_BALLS, dtype=torch.bool, device=rel.device)
        nb = (live[:, None, :] & not_self[None]).float()[..., None]
        h = self.ball(torch.cat([ball_f, (self.bb(bb_in) * nb).sum(dim=2)], dim=-1))   # (B, 3, H)

        pockets = rel[:, 3 * N_BALLS:].reshape(B, N_POCKETS, 2)
        pr = pockets[:, None, :, :] - pos[:, :, None, :]                   # (B, 3, 6, 2)
        H = h.shape[-1]
        pair_in = torch.cat([h[:, :, None, :].expand(B, N_BALLS, N_POCKETS, H),
                             pockets[:, None].expand(B, N_BALLS, N_POCKETS, 2), pr,
                             pr.norm(dim=-1, keepdim=True),
                             self.side.view(1, 1, N_POCKETS, 1).expand(B, N_BALLS, N_POCKETS, 1)], dim=-1)
        e = self.pair(pair_in).reshape(B, N_PAIRS, H)
        valid = live[:, :, None].expand(B, N_BALLS, N_POCKETS).reshape(B, N_PAIRS)
        return h, e, valid

    def forward_full(self, rel: torch.Tensor) -> dict[str, torch.Tensor]:
        """Everything the heads produce. Factorized mode adds ball/pocket logits and the scratch logit."""
        h, e, valid = self._encode_full(rel)
        B, H = rel.shape[0], e.shape[-1]
        means = torch.tanh(self.head_a(e))
        out = {"means": means, "valid": valid, "e": e}
        pocket = self.head_s(e).squeeze(-1)
        if self.has_scratch:
            out["scratch_logit"] = self.head_scratch(e).squeeze(-1)
        if self.has_made:
            out["made_logit"] = self.head_made(e).squeeze(-1)
        if not self.factorized:
            out["logits"] = pocket.masked_fill(~valid, _NEG)
            return out
        live = valid.view(B, N_BALLS, N_POCKETS)[:, :, 0]
        e3 = e.view(B, N_BALLS, N_POCKETS, H)
        ball = self.head_ball(torch.cat([h, e3.max(dim=2).values], dim=-1)).squeeze(-1).masked_fill(~live, _NEG)
        pocket = pocket.view(B, N_BALLS, N_POCKETS)
        joint = F.log_softmax(ball, dim=1)[:, :, None] + F.log_softmax(pocket, dim=2)
        out.update(ball_logits=ball, pocket_logits=pocket, live=live,
                   logits=joint.reshape(B, N_PAIRS).masked_fill(~valid, _NEG))
        return out

    def forward(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """rel (B, 21) → per-pair actions in [-1, 1] (B, 18, 2), selection logits (B, 18), valid (B, 18).
        Factorized mode returns the joint log p(ball) + log p(pocket | ball) as the logits."""
        o = self.forward_full(rel)
        return o["means"], o["logits"], o["valid"]

    # ── interface used by exp16_wm/bc.py ──────

    def predict_unit(self, rel: torch.Tensor) -> torch.Tensor:
        """Deterministic action in [-1, 1]: the top-scored pair's proposal. (B, 2)"""
        o = self.forward_full(rel)
        score = o["logits"]
        if self.has_scratch and self.scratch_lambda > 0:
            score = score - self.scratch_lambda * torch.sigmoid(o["scratch_logit"])
        return o["means"][torch.arange(len(score)), score.argmax(dim=1)]

    def train_loss(self, rel: torch.Tensor, teacher_u: torch.Tensor, good_u: torch.Tensor,
                   w: torch.Tensor | None = None, pairs: torch.Tensor | None = None,
                   extras: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        factorized + extras → outcome-grounded loss (_fact_loss).
        pairs None → relaxed winner-takes-all (below). pairs (B, K) int64, −1 = unknown →
        supervised pairs: each labelled candidate regresses its own pair's output and the
        selection head is trained to pick those pairs (fixed targets instead of a moving winner).
        Rows with no labelled pair fall back to winner-takes-all.
        """
        if self.mod is not None:
            return self._mod_loss(rel, good_u, pairs, extras, w)
        if self.factorized:
            if extras is None:
                raise ValueError("factorized PairGNNPolicy needs extras (good_made, good_scratch, pair_made, pair_scratch, pair_n)")
            return self._fact_loss(rel, good_u, extras, w)
        if pairs is None:
            return self._wta_loss(rel, teacher_u, good_u, w)
        rows = (pairs >= 0).any(dim=1)
        if bool(rows.all()):
            return self._pair_loss(rel, good_u, pairs, w)
        if not self.wta_fallback:
            return self._pair_loss(rel[rows], good_u[rows], pairs[rows], w) if bool(rows.any()) else rel.new_zeros(())
        loss, n = rel.new_zeros(()), len(rel)
        if bool(rows.any()):
            loss = loss + self._pair_loss(rel[rows], good_u[rows], pairs[rows], w) * int(rows.sum())
        loss = loss + self._wta_loss(rel[~rows], teacher_u[~rows], good_u[~rows], w) * int((~rows).sum())
        return loss / n

    def _fact_loss(self, rel: torch.Tensor, good_u: torch.Tensor, ex: dict[str, torch.Tensor],
                   w: torch.Tensor | None) -> torch.Tensor:
        """
        Outcome-grounded loss from simulator event logs.
          reg      : every label candidate k that cleanly pocketed regresses *each pair it really made*
                     (ex["good_made"][b,k,p]) onto its action — an action making two pairs trains both heads
          ball CE  : target_i ∝ max_j pair_made[i,j]   (balls with at least one verified-made pair)
          pocket CE: per ball, target_j ∝ pair_made[i,j] (softmax over that ball's 6 pockets)
          scratch  : BCE of the scratch head to pair_scratch / pair_n on pairs that were tried
        """
        w = torch.ones(2, device=rel.device) if w is None else w
        o = self.forward_full(rel)
        means, valid, live = o["means"], o["valid"], o["live"]
        B = rel.shape[0]
        # regression onto the pairs each candidate really made
        gm = ex["good_made"].float()                                            # (B, K2, 18)
        K2 = gm.shape[1]
        g = torch.nan_to_num(good_u[:, :K2])
        cand_ok = (~torch.isnan(good_u[:, :K2, 0])).float() * (1.0 - ex["good_scratch"].float())
        combo = gm * cand_ok[..., None]                                          # (B, K2, 18)
        diff = action_diff(means[:, None, :, :], g[:, :, None, :])               # (B, K2, 18, 2)
        d = ((diff * w) ** 2).sum(-1)
        reg = (d * combo).sum() / combo.sum().clamp(min=1.0)
        # selection: ball head and per-ball pocket softmax, soft targets from intent-verified hits
        made = ex["pair_made"].float().view(B, N_BALLS, N_POCKETS) * live[..., None].float()
        best = made.max(dim=2).values                                             # (B, 3)
        rows = best.sum(dim=1) > 0
        ball_ce = pocket_ce = reg.new_zeros(())
        if bool(rows.any()):
            t_ball = best / best.sum(dim=1, keepdim=True).clamp(min=1e-9)
            logp_b = F.log_softmax(o["ball_logits"], dim=1)
            ball_ce = -(t_ball * logp_b).sum(dim=1)[rows].mean()
            has = best > 0                                                        # (B, 3)
            t_p = made / made.sum(dim=2, keepdim=True).clamp(min=1e-9)
            logp_p = F.log_softmax(o["pocket_logits"], dim=2)
            pocket_ce = (-(t_p * logp_p).sum(dim=2))[has].mean()
        # scratch head
        n = ex["pair_n"].float()
        tried = (n > 0) & valid
        if bool(tried.any()):
            target = ex["pair_scratch"].float() / n.clamp(min=1.0)
            scratch_bce = F.binary_cross_entropy_with_logits(o["scratch_logit"][tried], target[tried])
        else:
            scratch_bce = reg.new_zeros(())
        total = reg if "reg" in self.fact_terms else reg * 0.0
        if "sel" in self.fact_terms:
            total = total + self.sel_weight * (ball_ce + pocket_ce)
        if "scratch" in self.fact_terms:
            total = total + self.sel_weight * scratch_bce
        return total

    def _mod_loss(self, rel: torch.Tensor, good_u: torch.Tensor, pairs: torch.Tensor | None,
                  ex: dict[str, torch.Tensor] | None, w: torch.Tensor | None) -> torch.Tensor:
        """
        Modular loss: each change of the factorized experiment switched on alone (ablation).
          structure  : self.factorized (global 18-way softmax vs ball head × per-ball pocket softmax)
          sel_target : "label" → the labelled candidates' pairs (pairs, counts) | "made" → pair_made over live pairs
          reg_target : "angle" → each labelled candidate regresses its angle-matched pair (pairs)
                       "made"  → each clean candidate regresses every pair it really made (good_made)
                       "made_same" → "made" restricted to the aimed ball (the ball of its angle-matched pair):
                                     drops combination shots, where aiming at ball A pockets ball B
          use_scratch: + sel_weight · BCE(scratch head, pair_scratch / pair_n)
          has_made   : + sel_weight · BCE(made head, pair_made / pair_n)   (auxiliary, not used at inference)
        Selection is one soft CE on the joint pair distribution in both structures (in the factorized
        one this equals ball CE on the target's ball marginal + per-ball pocket CE weighted by it).
        label + angle, global, no scratch == _pair_loss on rows that have a labelled pair; rows with
        no target for a term are left out of that term (instead of the WTA fallback).
        """
        sel_target, reg_target, use_scratch = self.mod
        if (sel_target == "label" or reg_target in ("angle", "made_same")) and pairs is None:
            raise ValueError("modular loss with label selection / angle regression needs pairs (good_pairs)")
        if (sel_target == "made" or reg_target in ("made", "made_same") or use_scratch or self.has_made) and ex is None:
            raise ValueError("modular loss with made targets / scratch needs extras (good_made, pair_made, ...)")
        w = torch.ones(2, device=rel.device) if w is None else w
        o = self.forward_full(rel)
        means, valid = o["means"], o["valid"]
        zero = means.new_zeros(())
        if pairs is not None:
            label = F.one_hot(pairs.clamp(min=0), N_PAIRS).float() * (pairs >= 0)[..., None].float()   # (B, K, 18)
        # regression
        if reg_target == "angle":
            combo, g = label, torch.nan_to_num(good_u[:, :pairs.shape[1]])
        elif reg_target in ("made", "made_same"):
            K2 = ex["good_made"].shape[1]
            cand_ok = (~torch.isnan(good_u[:, :K2, 0])).float() * (1.0 - ex["good_scratch"].float())
            combo, g = ex["good_made"].float() * cand_ok[..., None], torch.nan_to_num(good_u[:, :K2])
            if reg_target == "made_same":
                aimed = pairs[:, :K2]                                                 # (B, K2), −1 = no angle pair
                ball_of_pair = torch.arange(N_PAIRS, device=rel.device) // N_POCKETS
                same = (aimed[..., None] >= 0) & (ball_of_pair == (aimed.clamp(min=0) // N_POCKETS)[..., None])
                combo = combo * same.float()
        else:
            raise ValueError(f"unknown reg_target: {reg_target}")
        d = ((action_diff(means[:, None, :, :], g[:, :, None, :]) * w) ** 2).sum(-1)    # (B, K, 18)
        n = combo.sum(dim=(1, 2))
        rows = n > 0
        reg = ((d * combo).sum(dim=(1, 2)) / n.clamp(min=1.0))[rows].mean() if bool(rows.any()) else zero
        # selection: soft CE on the joint distribution over the 18 pairs
        if sel_target == "label":
            t = label.sum(dim=1)
        elif sel_target == "made":
            t = ex["pair_made"].float() * valid.float()
        else:
            raise ValueError(f"unknown sel_target: {sel_target}")
        mass = t.sum(dim=1)
        rows = mass > 0
        ce = zero
        if bool(rows.any()):
            t = t / mass.clamp(min=1e-9)[:, None]
            ce = -(t * F.log_softmax(o["logits"], dim=1)).sum(dim=1)[rows].mean()
        total = reg + self.sel_weight * ce
        if use_scratch and not self.has_scratch:
            raise ValueError("use_scratch needs PairGNNPolicy(scratch_head=True)")
        if use_scratch or self.has_made:
            n_try = ex["pair_n"].float()
            tried = (n_try > 0) & valid
            if bool(tried.any()):
                for on, key, logit in ((use_scratch, "pair_scratch", "scratch_logit"), (self.has_made, "pair_made", "made_logit")):
                    if on:
                        target = ex[key].float() / n_try.clamp(min=1.0)
                        total = total + self.sel_weight * F.binary_cross_entropy_with_logits(o[logit][tried], target[tried])
        return total

    def _pair_loss(self, rel: torch.Tensor, good_u: torch.Tensor, pairs: torch.Tensor,
                   w: torch.Tensor | None) -> torch.Tensor:
        w = torch.ones(2, device=rel.device) if w is None else w
        means, logits, _ = self(rel)
        B, K = pairs.shape
        has = pairs >= 0
        pidx = pairs.clamp(min=0)
        m = means.gather(1, pidx[..., None].expand(B, K, 2))
        d = ((action_diff(m, torch.nan_to_num(good_u)) * w) ** 2).sum(-1)
        n = has.sum(dim=1).clamp(min=1)
        reg = (d * has).sum(dim=1) / n
        logp = F.log_softmax(logits, dim=1).gather(1, pidx)
        ce = -(logp * has).sum(dim=1) / n
        return (reg + self.sel_weight * ce).mean()

    def _wta_loss(self, rel: torch.Tensor, teacher_u: torch.Tensor, good_u: torch.Tensor,
                  w: torch.Tensor | None = None) -> torch.Tensor:
        """Relaxed winner-takes-all regression to the nearest pocketing candidate + selection CE."""
        w = torch.ones(2, device=rel.device) if w is None else w
        means, logits, valid = self(rel)
        tgt_valid = ~torch.isnan(good_u[..., 0])                          # (B, K)
        has_good = tgt_valid.any(dim=1)
        tgt = torch.nan_to_num(good_u).clone()
        tgt[~has_good, 0] = teacher_u[~has_good]                          # no candidate → teacher action
        tgt_valid = tgt_valid.clone()
        tgt_valid[~has_good, 0] = True

        d = ((action_diff(means[:, :, None, :], tgt[:, None, :, :]) * w) ** 2).sum(-1)       # (B, 18, K)
        d = torch.where(tgt_valid[:, None, :], d, torch.full_like(d, float("inf")))
        dp = torch.where(valid, d.min(dim=2).values, torch.full_like(d[..., 0], float("inf")))
        rows = valid.any(dim=1)
        if not bool(rows.any()):
            return (means * 0).sum()
        dp, valid_r, logits_r = dp[rows], valid[rows], logits[rows]
        win = dp.argmin(dim=1)
        d_win = dp[torch.arange(len(dp)), win]
        others = valid_r & ~F.one_hot(win, N_PAIRS).bool()
        n_oth = others.sum(dim=1).clamp(min=1)
        d_oth = torch.where(others, dp, torch.zeros_like(dp)).sum(dim=1) / n_oth
        reg = (1 - self.wta_eps) * d_win + self.wta_eps * d_oth
        ce = F.cross_entropy(logits_r, win, reduction="none")
        return (reg + self.sel_weight * ce).mean()

    def params(self) -> list[nn.Parameter]:
        return list(self.parameters())

    @torch.no_grad()
    def act(self, rel_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Env action for one (21,) or a batch (B, 21) of relative observations."""
        x = torch.as_tensor(np.atleast_2d(rel_obs), dtype=torch.float32)
        a = (self.predict_unit(x) * self.act_scale + self.act_bias).numpy()
        return a[0] if np.asarray(rel_obs).ndim == 1 else a

    def save(self, path: str) -> None:
        torch.save({"gnn": self.state_dict(), "factorized": self.factorized, "has_scratch": self.has_scratch,
                    "has_made": self.has_made}, path)


assert REL_DIM == 3 * N_BALLS + 2 * N_POCKETS
