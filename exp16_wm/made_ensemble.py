"""
exp16_wm/made_ensemble.py — made head ensemble over independently trained runs (Exp-16, teacher-free).

Runs of the same configuration that differ only in their initialisation (T5: 11 runs, training-run sd of shots ≈ 0.025) choose
differently on the same table. Like the aim-encoder ensemble, averaging their made2 logits over the same candidates should cancel
part of those run-specific errors. Every member must use the same aim (the same frozen aim encoders) and absolute refine speeds,
so all members score exactly the same candidates; the action is the candidate with the highest mean made logit over all pairs
(the ensemble version of infer = all_pairs).
aim="mean" (roadmap EE3, end-to-end members that each predict their own aim): the candidates are built around the circular mean
of the members' aims, and each member scores them with its own made head and its own aim.

Usage:
    python -m exp16_wm.made_ensemble --runs "logs/experiments_teacherfree_T5/*_s0_*" "logs/experiments_teacherfree_T5r/*_i1_*" ... \
        --out logs/eval_paired/mac/ens_T5x3.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp16_wm.gnn_policy import PairGNNPolicy


class MadeEnsemble:
    """Mean made2 logit of several all_pairs models over their (identical) candidates; act() takes relative observations."""

    def __init__(self, models: list[PairGNNPolicy], aim_atol_rad: float = 1e-5, aim: str = "shared"):
        if not models:
            raise ValueError("an ensemble needs at least one model")
        if aim not in ("shared", "mean"):
            raise ValueError(f"aim must be 'shared' or 'mean', got {aim!r}")
        self.aim_mode = aim
        for m in models:
            if not (m.has_made2 and m.made2_geom):
                raise ValueError("every member needs made2_head and made2_geom (an aim)")
            if m.refine_speed_abs is None or tuple(m.refine_speed_abs) != tuple(models[0].refine_speed_abs) \
                    or tuple(m.refine_deg) != tuple(models[0].refine_deg):
                raise ValueError("every member needs the same refine_deg and absolute refine speeds (same candidates)")
        self.models, self.aim_atol = models, aim_atol_rad
        self.act_scale, self.act_bias = models[0].act_scale, models[0].act_bias

    @torch.no_grad()
    def scores(self, rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 18, C) mean made logits (invalid pairs −inf) and (B, 18, C, 2) the candidates in [-1, 1] units."""
        outs = [m.forward_full(rel) for m in self.models]
        if self.aim_mode == "mean":                                 # members' own aims, averaged on the circle
            angles = torch.stack([o["aim"] for o in outs])
            aim = torch.atan2(torch.sin(angles).mean(0), torch.cos(angles).mean(0))
        else:
            aim = outs[0]["aim"]
            for o in outs[1:]:
                d = torch.atan2(torch.sin(o["aim"] - aim), torch.cos(o["aim"] - aim)).abs().max()
                if d > self.aim_atol:
                    raise ValueError(f"members disagree on the aim by {float(d):.2e} rad — not the same aim encoders (use aim='mean')")
        m0 = self.models[0]
        d = aim - m0.act_bias[0]
        aim_u = (torch.atan2(torch.sin(d), torch.cos(d)) / m0.act_scale[0]).clamp(-1.0, 1.0)
        B, P = aim.shape
        cands, zs = [], []
        for j in range(P):
            cand = m0._unit_offsets(torch.stack([aim_u[:, j], outs[0]["means"][:, j, 1]], dim=-1))   # (B, C, 2), speed unused (absolute)
            C = cand.shape[1]
            z = torch.stack([m.head_made2(m._made2_features(o["e"][:, j][:, None].expand(-1, C, -1), cand,
                                                             o["means"][:, j][:, None].expand(-1, C, -1),
                                                             o["aim"][:, j][:, None].expand(-1, C))).squeeze(-1)
                             for m, o in zip(self.models, outs)]).mean(0)
            zs.append(z.masked_fill(~outs[0]["valid"][:, j][:, None], -float("inf"))); cands.append(cand)
        return torch.stack(zs, 1), torch.stack(cands, 1)

    def predict_unit(self, rel: torch.Tensor) -> torch.Tensor:
        z, cand = self.scores(rel)
        B = z.shape[0]
        k = z.reshape(B, -1).argmax(1)
        return cand.reshape(B, -1, 2)[torch.arange(B), k]

    def act(self, rel_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(rel_obs), dtype=torch.float32)
        a = (self.predict_unit(x) * self.act_scale + self.act_bias).numpy()
        return a[0] if np.asarray(rel_obs).ndim == 1 else a


def load_ensemble(run_dirs: list[str], aim: str = "shared") -> MadeEnsemble:
    from exp16_wm import eval_by_balls as E
    ms = []
    for d in run_dirs:
        m = E.load_run(d)
        m.infer, m.refine_deg, m.refine_speed, m.refine_speed_abs = "all_pairs", E.ANG31, (1.0,), E.ABS_SPEEDS
        ms.append(m)
    return MadeEnsemble(ms, aim=aim)


def main() -> None:
    from exp16_wm.bc import TransformedPolicy
    from exp16_wm.paired_eval import EVAL_SEEDS, GAMES_PER_SEED, find_run, machine_info, play_paired, save, summarize, _line
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run dir globs, each matching exactly one run")
    p.add_argument("--out", required=True); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-balls", type=int, default=3)
    p.add_argument("--aim", default="shared", choices=["shared", "mean"], help="mean: members with their own aims (end-to-end runs)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS)); p.add_argument("--games-per-seed", type=int, default=GAMES_PER_SEED)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    runs = [find_run(g) for g in args.runs]
    t0 = time.time()
    from exp16_wm.teacher_data import clear_env_kw
    res = play_paired(TransformedPolicy(load_ensemble(runs, args.aim), "relative"), tuple(args.seeds), args.games_per_seed, clear_env_kw(args.n_balls))
    meta = {"runs": runs, "infer": "made_ensemble", "aim": args.aim, "n_balls": args.n_balls, "eval_seeds": args.seeds, "games_per_seed": args.games_per_seed,
            "threads": args.threads, "seconds": round(time.time() - t0, 1), "machine": machine_info()}
    save(args.out, res, meta)
    print(_line(os.path.basename(args.out), summarize(res)), f"  ({meta['seconds']:.0f}s, {len(runs)} members) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
