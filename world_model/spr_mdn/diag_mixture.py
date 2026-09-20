"""
diag_mixture.py — MixtureHead 진단.

측정:
  - max_k(π_k): 지배 컴포넌트 가중치 (1에 가까우면 confident/unimodal)
  - Var_k(μ_k): 컴포넌트간 μ spread (weight 없이, D차원 평균)
  - h별 변화
  - 충돌 스텝 vs 비충돌 스텝 분리
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.spr_mdn_model import SPRMDNModel, LATENT_DIM, N_COMPONENTS, ACTION_DIM
from world_model.spr_mdn.spr_dataset import SPRDataset, make_balanced_val_eps
from world_model.generate_data_fixeddt import DT

CKPT   = "world_model/results/spr_mdn_v6/best.pt"
DATA   = "world_model/data_fixeddt"
STEPS  = 60
N_BINS = 6   # h bins: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def collect_stats(model, episodes):
    """
    Returns per-step records: list of dicts with
      h, max_pi, var_mu (scalar), is_coll (bool)
    """
    records = []

    for ep_s, ep_f, ep_t, _, ep_a in episodes:
        if len(ep_s) < STEPS + 1:
            continue
        s0  = torch.from_numpy(ep_s[0:1]).float().to(DEVICE)
        act = torch.from_numpy(ep_a[np.newaxis]).float().to(DEVICE)
        a_zeros = torch.zeros(1, model.action_dim, device=DEVICE)

        z_hat = model.encoder(s0)  # (1, D)

        for h in range(STEPS):
            a_tilde = act if h == 0 else a_zeros
            pi, mu, _ = model.mixture_head(z_hat, a_tilde)   # pi(1,K), mu(1,K,D)

            max_pi = pi.max(dim=-1).values.item()             # scalar ∈ (0,1]
            # Var_k(μ_k): variance across K components, mean over D dims
            var_mu = mu.var(dim=1).mean().item()              # scalar

            is_coll = bool(ep_f[h]) if h < len(ep_f) else False

            records.append({"h": h, "max_pi": max_pi, "var_mu": var_mu, "is_coll": is_coll})

            # Advance z_hat (mixture mean, same as rollout_eval)
            z_hat = model.encoder.ln((pi.unsqueeze(-1) * mu).sum(dim=1))

    return records


def summarize(records, label):
    arr_max = np.array([r["max_pi"] for r in records])
    arr_var = np.array([r["var_mu"] for r in records])
    print(f"\n=== {label} (n={len(records)}) ===")
    print(f"  max_pi : mean={arr_max.mean():.3f}  p25={np.percentile(arr_max,25):.3f}  p50={np.percentile(arr_max,50):.3f}  p75={np.percentile(arr_max,75):.3f}  p95={np.percentile(arr_max,95):.3f}")
    print(f"  var_mu : mean={arr_var.mean():.4f}  p25={np.percentile(arr_var,25):.4f}  p50={np.percentile(arr_var,50):.4f}  p75={np.percentile(arr_var,75):.4f}  p95={np.percentile(arr_var,95):.4f}")


def summarize_by_h(records, n_bins=N_BINS):
    bin_size = STEPS // n_bins
    print("\n  h-bin breakdown (all steps):")
    print(f"  {'h-bin':10s}  {'n':>6s}  {'max_pi_mean':>12s}  {'var_mu_mean':>12s}")
    for b in range(n_bins):
        h_lo = b * bin_size
        h_hi = (b + 1) * bin_size
        sub = [r for r in records if h_lo <= r["h"] < h_hi]
        if not sub:
            continue
        mp = np.mean([r["max_pi"] for r in sub])
        vm = np.mean([r["var_mu"] for r in sub])
        print(f"  h={h_lo:2d}-{h_hi-1:2d}      {len(sub):>6d}  {mp:>12.3f}  {vm:>12.4f}")


def main():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = SPRMDNModel(
        latent_dim   = ckpt.get("latent_dim",   LATENT_DIM),
        n_components = ckpt.get("n_components", N_COMPONENTS),
        action_dim   = ckpt.get("action_dim",   ACTION_DIM),
        ema_tau      = ckpt.get("ema_tau",      0.99),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    ds  = SPRDataset(DATA)
    rng = np.random.default_rng(0)
    perm  = rng.permutation(len(ds.episodes))
    n_val = max(200, int(len(ds.episodes) * 0.1))
    val_eps_all = [ds.episodes[i] for i in perm[:n_val]]
    bal_eps     = make_balanced_val_eps(val_eps_all, n_each=250, seed=0)
    valid_eps   = [ep for ep in bal_eps if len(ep[0]) >= STEPS + 1]
    print(f"Episodes: {len(valid_eps)}")

    print("\n--- Collecting per-step stats ---")
    records = collect_stats(model, valid_eps)

    # ── Aggregate ───────────────────────────────────────────────────────
    summarize(records, "ALL steps")
    summarize([r for r in records if not r["is_coll"]], "Non-collision steps")
    summarize([r for r in records if     r["is_coll"]], "Collision steps")

    # ── h-bin breakdown ─────────────────────────────────────────────────
    summarize_by_h(records)

    # ── h-bin for collision steps only ──────────────────────────────────
    coll_records = [r for r in records if r["is_coll"]]
    if coll_records:
        print("\n  h-bin breakdown (collision steps only):")
        print(f"  {'h-bin':10s}  {'n':>6s}  {'max_pi_mean':>12s}  {'var_mu_mean':>12s}")
        bin_size = STEPS // N_BINS
        for b in range(N_BINS):
            h_lo = b * bin_size
            h_hi = (b + 1) * bin_size
            sub = [r for r in coll_records if h_lo <= r["h"] < h_hi]
            if not sub:
                continue
            mp = np.mean([r["max_pi"] for r in sub])
            vm = np.mean([r["var_mu"] for r in sub])
            print(f"  h={h_lo:2d}-{h_hi-1:2d}      {len(sub):>6d}  {mp:>12.3f}  {vm:>12.4f}")
    else:
        print("\n  (no collision steps found in val set)")

    # ── π distribution histogram ────────────────────────────────────────
    all_max_pi = [r["max_pi"] for r in records]
    print("\n  max_π histogram (all steps):")
    hist, edges = np.histogram(all_max_pi, bins=[0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001])
    for lo, hi, cnt in zip(edges[:-1], edges[1:], hist):
        pct = cnt / len(all_max_pi) * 100
        bar = "#" * int(pct / 2)
        print(f"  [{lo:.2f},{hi:.2f})  {cnt:>7d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
