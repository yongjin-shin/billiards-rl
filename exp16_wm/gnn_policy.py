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
FOURIER_SCALE = 4.0             # m; the lowest Fourier frequency has period 2·FOURIER_SCALE, longer than any distance on the table
BALL_RADIUS = 0.028575          # m (pooltool default; test_exp16_wm_gnn_policy checks it against the env)
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


def _mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    """layers hidden layers (2 = the original structure, same state_dict keys)."""
    mods: list[nn.Module] = [nn.Linear(i, h), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.ReLU()]
    return nn.Sequential(*mods, nn.Linear(h, o))


LOGVAR_RANGE = 3.0       # uncertainty weighting: log σ² of each loss term is limited to [-3, 3] (weights e^{∓3} ≈ 0.05 … 20)


def speed_bin_centers(n_bins: int) -> torch.Tensor:
    """n_bins centre speeds (m/s), log-spaced from the lowest to the highest speed of the action range.
    With 21 centres the ratio is 16^(1/20) and 0.5, 1.0, 2.0, 4.0, 8.0 m/s are centres (the label speeds sit on 1.0 m/s)."""
    lo, hi = float(ACT_LOW[1]), float(ACT_HIGH[1])
    return lo * (hi / lo) ** (torch.arange(n_bins, dtype=torch.float32) / (n_bins - 1))


def speed_bin_edges(n_bins: int) -> torch.Tensor:
    """n_bins − 1 inner edges: the geometric mean of neighbouring centres (so no label speed of interest lies on an edge)."""
    c = speed_bin_centers(n_bins)
    return torch.sqrt(c[:-1] * c[1:])


def speed_to_bin(speed_ms: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Speeds (m/s) → index of the nearest centre in log space, in [0, n_bins − 1] (out-of-range values go to the end bins)."""
    return torch.bucketize(speed_ms, speed_bin_edges(n_bins).to(speed_ms.device))


def bin_center_ms(idx: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Bin index → centre speed in m/s."""
    return speed_bin_centers(n_bins).to(idx.device)[idx]


class PairGNNPolicy(nn.Module):
    def __init__(self, hidden: int = 128, wta_eps: float = 0.05, sel_weight: float = 10.0,
                 factorized: bool = False, scratch_head: bool = False, made_head: bool = False,
                 made2_head: bool = False, geom_angle: bool = False, made2_geom: bool = False,
                 enc_layers: int = 2, speed_loss: str = "mse", speed_bins: int = 0, uncertainty_weights: bool = False, aux_dim: int = 0,
                 aim_encoder: "PairGNNPolicy | None" = None, fourier: int = 0):
        super().__init__()
        if fourier < 0:
            raise ValueError(f"fourier must be ≥ 0, got {fourier}")
        if aim_encoder is not None and not made2_geom:
            raise ValueError("aim_encoder replaces the exact aim of made2_geom: it needs made2_geom=True")
        if aim_encoder is not None and getattr(aim_encoder, "aux_dim", 0) != 1:
            raise ValueError("aim_encoder must be an aux-only aim model (aux_dim 1)")
        if aux_dim < 0:
            raise ValueError(f"aux_dim must be ≥ 0, got {aux_dim}")
        if uncertainty_weights and (speed_bins or speed_loss != "mse"):
            raise ValueError("uncertainty_weights cannot be combined with speed_loss / speed_bins")
        if speed_loss not in ("mse", "l1"):
            raise ValueError(f"speed_loss must be mse or l1, got {speed_loss!r}")
        if speed_bins < 0 or speed_bins == 1:
            raise ValueError(f"speed_bins must be 0 (off) or ≥ 2, got {speed_bins}")
        H = hidden
        self.speed_loss, self.speed_bins = speed_loss, speed_bins   # speed term of the pair loss: squared / absolute error, or a categorical head over speed bins
        self.speed_sample = False                       # inference with speed_bins: sample the bin from the softmax instead of taking the most probable one
        self.uncertainty_weights = uncertainty_weights  # Kendall et al. 2018: learned log σ² per loss term (angle, speed, selection) in the pair loss
        self.wta_eps, self.sel_weight = wta_eps, sel_weight
        self.factorized = factorized
        self.scratch_lambda = 0.0         # inference: penalise pairs the scratch head expects to scratch
        self.fact_terms = ("reg", "sel", "scratch")     # factorized loss terms that are switched on (ablations)
        self.mod: tuple[str, str, bool] | None = None   # modular loss (sel_target, reg_target, use_scratch); see _mod_loss
        self.has_scratch = factorized or scratch_head
        self.wta_fallback = True                        # pair supervision: rows with no labelled pair use WTA (False → left out)
        self.has_made = made_head                       # auxiliary only (modular loss): P(pair made) from pair_stats
        self.made2_geom = made2_geom                    # diagnostic: made2 head also gets (action angle − geometric aim) / 0.2°
        self.has_made2 = made2_head                     # look-ahead stage 2: P(pair ij made | e_ij, action)
        self.infer = "select"                           # select | verify | refine | verify_refine (needs made2 head)
        self.refine_deg = (-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3)
        self.refine_speed = (0.8, 1.0, 1.25)             # × the proposal's speed
        self.refine_speed_abs: tuple[float, ...] | None = None   # m/s, instead of refine_speed (independent of the proposal)
        self.hidden, self.enc_layers = H, enc_layers
        # input representation: each 2-D coordinate (ball position, ball-ball difference, pocket position, ball→pocket vector)
        # can be extended with Fourier features sin/cos(2^k π x / FOURIER_SCALE), k = 0 … fourier − 1 (Tancik et al. 2020)
        self.fourier = fourier
        d2 = 2 + 4 * fourier                              # width of one encoded 2-D coordinate
        ball_dim = d2 + 3
        self.bb   = _mlp(ball_dim + d2 + 1 + 1, H, H, enc_layers)
        self.ball = _mlp(ball_dim + H, H, H, enc_layers)
        self.pair = _mlp(H + d2 + d2 + 1 + 1, H, H, enc_layers)
        self.geom_angle = geom_angle                     # angle comes from a pair embedding without the neighbour message
        self.head_a = nn.Linear(H, 2)
        if geom_angle:
            self.head_ang = nn.Linear(H, 1)       # angle only; head_a's angle output is then unused (speed still from e)
        if speed_bins:
            self.head_speed_bins = nn.Linear(H, speed_bins)   # logits over log-spaced speed bins; replaces the speed output of head_a
        self.head_s = nn.Linear(H, 1)             # global mode: pair score; factorized mode: pocket logit within a ball
        if factorized:
            self.head_ball = nn.Linear(2 * H, 1)  # which ball: [h_i, max_j e_ij]
        if self.has_scratch:
            self.head_scratch = nn.Linear(H, 1)   # will this pair's shot pocket the cue ball
        if made_head:
            self.head_made = nn.Linear(H, 1)      # can this pair be made (coarse-scan success rate) — look-ahead auxiliary
        if made2_head:
            self.head_made2 = _mlp(H + 4 + int(made2_geom), H, 1)   # [e_ij, a, (a − a_ij)·scale] → will action a pocket pair ij
        self.aux_dim = aux_dim                          # auxiliary geometry targets predicted from each pair embedding (training only)
        if aux_dim:
            self.aux_head = nn.Sequential(nn.Linear(H, 64), nn.ReLU(), nn.Linear(64, aux_dim))
        if uncertainty_weights:
            self.log_var = nn.Parameter(torch.zeros(3))       # [angle, speed, selection]; 0 = the hand-set weights of the plain loss
        side = torch.zeros(N_POCKETS)
        side[list(SIDE_POCKETS)] = 1.0
        self.register_buffer("side", side)
        self.register_buffer("act_scale", torch.tensor((ACT_HIGH - ACT_LOW) / 2.0, dtype=torch.float32))
        self.register_buffer("act_bias", torch.tensor((ACT_HIGH + ACT_LOW) / 2.0, dtype=torch.float32))
        # made2 input: action offset from the pair's proposal in precision units (0.2° = 1, 0.5 m/s = 1) —
        # in [-1, 1] units a 0.1° window is 0.0006 wide, too fine for an MLP to resolve
        half = (ACT_HIGH - ACT_LOW) / 2.0
        self.register_buffer("made2_scale", torch.tensor([np.degrees(half[0]) / 0.2, half[1] / 0.5], dtype=torch.float32),
                             persistent=False)
        # cascade: the made2 head's aim comes from a frozen, separately trained aim encoder instead of the ghost-ball formula
        self.aim_encoder = aim_encoder
        if aim_encoder is not None:
            for prm in aim_encoder.parameters():
                prm.requires_grad_(False)
            aim_encoder.eval()
            self.register_buffer("aim_mean", torch.zeros(1))            # degrees; the aim encoder's target standardisation
            self.register_buffer("aim_std", torch.ones(1))

    def set_aim_stats(self, mean: float, std: float) -> None:
        """The train mean / std (degrees) the aim encoder's target was standardised with."""
        self.aim_mean.fill_(float(mean)); self.aim_std.fill_(float(std))

    def _learned_aim(self, rel: torch.Tensor) -> torch.Tensor:
        """(B, 18) aim (rad, relative frame) predicted by the frozen aim encoder — same quantity as _geometric_aim."""
        with torch.no_grad():
            z = self.aim_encoder.aux_head(self.aim_encoder._encode(rel)[0])[..., 0]
        return torch.deg2rad(z * self.aim_std + self.aim_mean)

    def train(self, mode: bool = True) -> "PairGNNPolicy":
        super().train(mode)
        if getattr(self, "aim_encoder", None) is not None:
            self.aim_encoder.eval()                                          # frozen in every mode
        return self

    # ── encoder ───────────────────────────────

    def _ff(self, x: torch.Tensor) -> torch.Tensor:
        """(…, 2) coordinates in metres → (…, 2 + 4·fourier): the coordinates, then sin and cos at each frequency."""
        if not self.fourier:
            return x
        freq = (2.0 ** torch.arange(self.fourier, dtype=x.dtype, device=x.device)) * (np.pi / FOURIER_SCALE)
        xs = (x[..., None] * freq).flatten(-2)                                  # (…, 2·fourier)
        return torch.cat([x, torch.sin(xs), torch.cos(xs)], dim=-1)

    def _encode(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, e, valid, _ = self._encode_full(rel)
        return e, valid

    def _encode_full(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(h, e, valid, e_geo): e_geo = pair embedding with the ball-ball message zeroed (None unless geom_angle)."""
        B = rel.shape[0]
        balls = rel[:, :3 * N_BALLS].reshape(B, N_BALLS, 3)
        pos, pocketed = balls[..., :2], balls[..., 2]
        live = pocketed < 0.5
        dist = pos.norm(dim=-1)
        ref = torch.where(live, dist, torch.full_like(dist, float("inf"))).argmin(dim=1)
        is_ref = F.one_hot(ref, N_BALLS).float() * live.any(dim=1, keepdim=True).float()
        ball_f = torch.cat([self._ff(pos), dist[..., None], pocketed[..., None], is_ref[..., None]], dim=-1)

        diff = pos[:, None, :, :] - pos[:, :, None, :]                    # (B, i, k, 2): k relative to i
        bb_in = torch.cat([ball_f[:, :, None, :].expand(B, N_BALLS, N_BALLS, ball_f.shape[-1]), self._ff(diff),
                           diff.norm(dim=-1, keepdim=True),
                           pocketed[:, None, :, None].expand(B, N_BALLS, N_BALLS, 1)], dim=-1)
        not_self = ~torch.eye(N_BALLS, dtype=torch.bool, device=rel.device)
        nb = (live[:, None, :] & not_self[None]).float()[..., None]
        h = self.ball(torch.cat([ball_f, (self.bb(bb_in) * nb).sum(dim=2)], dim=-1))   # (B, 3, H)

        pockets = rel[:, 3 * N_BALLS:].reshape(B, N_POCKETS, 2)
        pr = pockets[:, None, :, :] - pos[:, :, None, :]                   # (B, 3, 6, 2)
        H = h.shape[-1]
        def pair_embed(hb: torch.Tensor) -> torch.Tensor:
            pair_in = torch.cat([hb[:, :, None, :].expand(B, N_BALLS, N_POCKETS, H),
                                 self._ff(pockets)[:, None].expand(B, N_BALLS, N_POCKETS, 2 + 4 * self.fourier), self._ff(pr),
                                 pr.norm(dim=-1, keepdim=True),
                                 self.side.view(1, 1, N_POCKETS, 1).expand(B, N_BALLS, N_POCKETS, 1)], dim=-1)
            return self.pair(pair_in).reshape(B, N_PAIRS, H)

        e = pair_embed(h)
        e_geo = pair_embed(self.ball(torch.cat([ball_f, torch.zeros_like(h)], dim=-1))) if self.geom_angle else None
        valid = live[:, :, None].expand(B, N_BALLS, N_POCKETS).reshape(B, N_PAIRS)
        return h, e, valid, e_geo

    def forward_full(self, rel: torch.Tensor) -> dict[str, torch.Tensor]:
        """Everything the heads produce. Factorized mode adds ball/pocket logits and the scratch logit."""
        h, e, valid, e_geo = self._encode_full(rel)
        B, H = rel.shape[0], e.shape[-1]
        means = torch.tanh(self.head_a(e))
        if e_geo is not None:
            means = torch.cat([torch.tanh(self.head_ang(e_geo)), means[..., 1:]], dim=-1)
        if self.speed_bins:
            sp_logits = self.head_speed_bins(e)                                   # (B, 18, bins)
            idx = (torch.multinomial(F.softmax(sp_logits, -1).reshape(-1, self.speed_bins), 1).reshape(sp_logits.shape[:2])
                   if self.speed_sample else sp_logits.argmax(-1))
            sp_unit = (bin_center_ms(idx, self.speed_bins) - self.act_bias[1]) / self.act_scale[1]
            means = torch.cat([means[..., :1], sp_unit[..., None]], dim=-1)       # the proposal's speed is the chosen bin's centre
        out = {"means": means, "valid": valid, "e": e}
        if self.speed_bins:
            out["speed_logits"] = sp_logits
        if self.made2_geom:
            out["aim"] = self._geometric_aim(rel) if self.aim_encoder is None else self._learned_aim(rel)
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

    # ── look-ahead stage 2: action-conditional made head ──

    @staticmethod
    def _geometric_aim(rel: torch.Tensor) -> torch.Tensor:
        """(B, 18) direction (rad, in the relative frame = delta_angle) from the cue to the ghost-ball point of each pair."""
        B = rel.shape[0]
        b = rel[:, :3 * N_BALLS].reshape(B, N_BALLS, 3)[..., :2]
        p = rel[:, 3 * N_BALLS:].reshape(B, N_POCKETS, 2)
        d = p[:, None, :, :] - b[:, :, None, :]                                       # (B, 3, 6, 2) ball → pocket
        ghost = b[:, :, None, :] - 2 * BALL_RADIUS * d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.atan2(ghost[..., 1], ghost[..., 0]).reshape(B, N_PAIRS)

    def made2_logits(self, o: dict[str, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        """
        o = forward_full(rel); a (B, 2) one action for every pair or (B, 18, 2) one per pair, in [-1, 1].
        → (B, 18) logits of "action a pockets pair ij cleanly"; invalid pairs masked.
        """
        e, means = o["e"], o["means"].detach()
        a = a[:, None, :].expand_as(means) if a.dim() == 2 else a
        feats = [e, a, action_diff(a, means) * self.made2_scale]
        if self.made2_geom:
            feats.append(self._geom_feature(o["aim"], a[..., 0]))
        z = self.head_made2(torch.cat(feats, dim=-1)).squeeze(-1)
        return z.masked_fill(~o["valid"], _NEG)

    def _geom_feature(self, aim: torch.Tensor, a_angle_unit: torch.Tensor) -> torch.Tensor:
        """(…, 1): wrap(action angle − aim) in units of 0.2° (a_angle_unit in [-1, 1], 1 = π rad)."""
        d = a_angle_unit * float(self.act_scale[0]) + float(self.act_bias[0]) - aim
        d = torch.atan2(torch.sin(d), torch.cos(d))
        return (torch.rad2deg(d) / 0.2)[..., None]

    def made2_loss(self, rel: torch.Tensor, a: torch.Tensor, made: torch.Tensor,
                   scratch: torch.Tensor) -> torch.Tensor:
        """BCE over the live pairs: target_ij = action a made pair ij and the cue ball stayed up."""
        o = self.forward_full(rel)
        z = self.made2_logits(o, a)
        target = (made & ~scratch[:, None]).float()
        v = o["valid"]
        return F.binary_cross_entropy_with_logits(z[v], target[v])

    def _unit_offsets(self, a: torch.Tensor) -> torch.Tensor:
        """(B, 2) actions → (B, C, 2) refinement candidates: angle ± refine_deg × speed × refine_speed."""
        deg = torch.tensor(self.refine_deg, dtype=a.dtype, device=a.device)
        ang = a[:, None, 0] + deg[None] / 180.0                                    # unit angle: 1 = π rad
        if self.refine_speed_abs is not None:
            v = torch.tensor(self.refine_speed_abs, dtype=a.dtype, device=a.device)[None].expand(len(a), -1)
        else:
            mul = torch.tensor(self.refine_speed, dtype=a.dtype, device=a.device)
            v = (a[:, 1:2] * self.act_scale[1] + self.act_bias[1]) * mul[None]       # m/s
        spd = (v - self.act_bias[1]) / self.act_scale[1]
        grid = torch.stack(torch.broadcast_tensors(ang[:, :, None], spd[:, None, :]), dim=-1)   # (B, A, S, 2)
        return grid.reshape(len(a), -1, 2).clamp(-1.0, 1.0)

    # ── interface used by exp16_wm/bc.py ──────

    def predict_unit(self, rel: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action in [-1, 1]. infer:
          select        : the top selection-scored pair's proposal
          verify        : pair by log p(select) + log P(made | e_ij, a_ij)
          refine        : selection's pair, then the candidate around its proposal with the highest P(made)
          verify_refine : both
        """
        o = self.forward_full(rel)
        score = o["logits"]
        if self.has_scratch and self.scratch_lambda > 0:
            score = score - self.scratch_lambda * torch.sigmoid(o["scratch_logit"])
        if self.infer != "select" and not self.has_made2:
            raise ValueError(f"infer={self.infer} needs made2_head")
        if self.infer in ("verify", "verify_refine"):
            score = F.log_softmax(score, dim=1) + F.logsigmoid(self.made2_logits(o, o["means"]))
        rows = torch.arange(len(score))
        k = score.argmax(dim=1)
        a = o["means"][rows, k]
        if self.infer not in ("refine", "verify_refine"):
            return a
        cand = self._unit_offsets(a)                                                  # (B, C, 2)
        C = cand.shape[1]
        e_k = o["e"][rows, k][:, None].expand(-1, C, -1)
        feats = [e_k, cand, action_diff(cand, a[:, None].expand(-1, C, -1)) * self.made2_scale]
        if self.made2_geom:
            feats.append(self._geom_feature(o["aim"][rows, k][:, None].expand(-1, C), cand[..., 0]))
        z = self.head_made2(torch.cat(feats, dim=-1)).squeeze(-1)
        return cand[rows, z.argmax(dim=1)]

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
        if self.speed_bins or self.speed_loss != "mse" or self.uncertainty_weights:
            if self.mod is not None or self.factorized or pairs is None or self.wta_fallback:
                raise ValueError("speed_loss / speed_bins / uncertainty_weights are implemented for the plain pair-supervised loss only "
                                 "(--pair-supervision --no-wta-fallback, global selection, no modular loss)")
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
        if self.uncertainty_weights:
            return self._pair_loss_kendall(rel, good_u, pairs, w)
        if self.speed_bins or self.speed_loss != "mse":
            return self._pair_loss_speed(rel, good_u, pairs, w)
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

    def aux_loss(self, rel: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary geometry loss: the aux head reads each pair embedding e_ij and regresses the standardised geometric
        targets (B, 18, aux_dim); mean squared error over the live pairs. Training only, nothing at inference.
        """
        if not self.aux_dim:
            raise ValueError("aux_loss needs PairGNNPolicy(aux_dim > 0)")
        e, valid = self._encode(rel)
        err = ((self.aux_head(e) - target) ** 2).mean(dim=-1)
        return (err * valid).sum() / valid.sum().clamp(min=1)

    def _pair_loss_kendall(self, rel: torch.Tensor, good_u: torch.Tensor, pairs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        _pair_loss with learned term weights (Kendall, Gal & Cipolla 2018): Σ_i exp(−s_i) · L_i + s_i over
        L_angle, L_speed (squared errors in precision units) and L_sel (sel_weight · selection CE), s = log σ² clamped to
        ±LOGVAR_RANGE. At s = 0 this is the plain pair loss. At the optimum σ_i² ≈ L_i, so a term with a large (heavy-tailed)
        loss gets a small weight.
        """
        means, logits, _ = self(rel)
        B, K = pairs.shape
        has = pairs >= 0
        pidx = pairs.clamp(min=0)
        m = means.gather(1, pidx[..., None].expand(B, K, 2))
        diff = action_diff(m, torch.nan_to_num(good_u)) * w
        n = has.sum(dim=1).clamp(min=1)
        l_ang = ((diff[..., 0] ** 2 * has).sum(dim=1) / n).mean()
        l_spd = ((diff[..., 1] ** 2 * has).sum(dim=1) / n).mean()
        ce = -(F.log_softmax(logits, dim=1).gather(1, pidx) * has).sum(dim=1) / n
        l_sel = (self.sel_weight * ce).mean()
        s = self.log_var.clamp(-LOGVAR_RANGE, LOGVAR_RANGE)
        return (torch.exp(-s[0]) * l_ang + s[0]) + (torch.exp(-s[1]) * l_spd + s[1]) + (torch.exp(-s[2]) * l_sel + s[2])

    def _pair_loss_speed(self, rel: torch.Tensor, good_u: torch.Tensor, pairs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        _pair_loss with a different speed term. angle: squared error as before. speed:
          speed_loss "l1"   → |Δspeed · w| instead of (Δspeed · w)²  (the model's value is then a median)
          speed_bins > 0    → no regression on speed; sel_weight · cross-entropy of the pair's speed-bin logits on the
                              bin of the label speed (the model's value at inference is a bin centre)
        """
        o = self.forward_full(rel)
        means, logits = o["means"], o["logits"]
        B, K = pairs.shape
        has = pairs >= 0
        pidx = pairs.clamp(min=0)
        m = means.gather(1, pidx[..., None].expand(B, K, 2))
        g = torch.nan_to_num(good_u)
        diff = action_diff(m, g) * w
        d = diff[..., 0] ** 2
        if not self.speed_bins:
            d = d + (diff[..., 1].abs() if self.speed_loss == "l1" else diff[..., 1] ** 2)
        n = has.sum(dim=1).clamp(min=1)
        reg = (d * has).sum(dim=1) / n
        logp = F.log_softmax(logits, dim=1).gather(1, pidx)
        ce = -(logp * has).sum(dim=1) / n
        total = reg + self.sel_weight * ce
        if self.speed_bins:
            sp = o["speed_logits"].gather(1, pidx[..., None].expand(B, K, self.speed_bins))          # (B, K, bins)
            tbin = speed_to_bin(g[..., 1] * self.act_scale[1] + self.act_bias[1], self.speed_bins)
            ce_sp = F.cross_entropy(sp.reshape(-1, self.speed_bins), tbin.reshape(-1), reduction="none").reshape(B, K)
            total = total + self.sel_weight * (ce_sp * has).sum(dim=1) / n
        return total.mean()

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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Checkpoints from the few runs that stored the (constant) made2_scale buffer load too."""
        state_dict = {k: v for k, v in state_dict.items() if k != "made2_scale"}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def load_encoder(self, state_dict) -> list[str]:
        """
        Initialise the encoder (ball message / ball / pair MLPs) and the aux head from another checkpoint; the
        action, selection and made heads (`head_*`) keep their fresh initialisation. Returns the loaded keys.
        A key the checkpoint has but this model lacks, or one with another shape (other width / depth), is an error.
        """
        own = self.state_dict()
        take = {}
        for k, v in state_dict.items():
            if k == "made2_scale" or k.startswith("head_"):
                continue
            if k not in own:
                raise ValueError(f"checkpoint key {k} is not in this model")
            if own[k].shape != v.shape:
                raise ValueError(f"checkpoint key {k} has shape {tuple(v.shape)}, this model {tuple(own[k].shape)}")
            take[k] = v
        if not any(k.startswith(("bb.", "ball.", "pair.")) for k in take):
            raise ValueError("checkpoint has no encoder weights")
        super().load_state_dict(take, strict=False)
        return sorted(take)

    def save(self, path: str) -> None:
        torch.save({"gnn": self.state_dict(), "factorized": self.factorized, "has_scratch": self.has_scratch,
                    "has_made": self.has_made, "has_made2": self.has_made2,
                    "geom_angle": self.geom_angle, "made2_geom": self.made2_geom,
                    "hidden": self.hidden, "enc_layers": self.enc_layers,
                    "speed_loss": self.speed_loss, "speed_bins": self.speed_bins,
                    "uncertainty_weights": self.uncertainty_weights, "aux_dim": self.aux_dim,
                    "made2_aim_learned": self.aim_encoder is not None, "fourier": self.fourier}, path)


assert REL_DIM == 3 * N_BALLS + 2 * N_POCKETS
