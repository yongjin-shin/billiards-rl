"""
eval_argmax.py — v6 checkpoint을 argmax-mode eval로 재평가.
현재 rollout_eval: z = encoder.ln(Σ_k π_k μ_k)  (mixture mean)
argmax-mode eval:  z = encoder.ln(μ_{argmax(π)})  (single best component)
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from world_model.spr_mdn.spr_mdn_model import SPRMDNModel, LATENT_DIM, N_COMPONENTS, ACTION_DIM
from world_model.spr_mdn.spr_dataset import SPRDataset, make_balanced_val_eps
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H

CKPT   = "world_model/results/spr_mdn_v6/best.pt"
DATA   = "world_model/data_fixeddt"
STEPS  = 60
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_steps = {
    "0.5s": int(0.5 / DT),
    "1.0s": int(1.0 / DT),
    "2.0s": int(2.0 / DT),
    "3.0s": int(3.0 / DT),
}


@torch.no_grad()
def rollout_argmax(model: SPRMDNModel, s0: torch.Tensor,
                   T: int, action: torch.Tensor) -> torch.Tensor:
    """Argmax-component deterministic rollout: z = μ_{argmax(π)}."""
    B      = s0.shape[0]
    device = s0.device
    a_zeros = torch.zeros(B, model.action_dim, device=device)

    z_hat      = model.encoder(s0)
    s_hat_list = [model._decode(z_hat)]

    for h in range(T):
        a_tilde = action if h == 0 else a_zeros
        pi, mu, _ = model.mixture_head(z_hat, a_tilde)
        k_best    = pi.argmax(dim=-1)                          # (B,)
        mu_best   = mu[torch.arange(B, device=device), k_best] # (B, D)
        z_hat     = model.encoder.ln(mu_best)
        s_hat_list.append(model._decode(z_hat))

    return torch.stack(s_hat_list, dim=1)  # (B, T+1, 14)


def eval_mode(model, episodes, label):
    valid = [(ep_s, ep_a) for ep_s, _, _, _, ep_a in episodes if len(ep_s) >= STEPS + 1]

    s0   = torch.from_numpy(np.stack([e[0][0]  for e in valid])).float().to(DEVICE)
    acts = torch.from_numpy(np.stack([e[1]     for e in valid])).float().to(DEVICE)

    # --- mixture mean (current rollout_eval) ---
    s_mean, _ = model.rollout_eval(s0, STEPS, action=acts)
    pr_mean   = s_mean.cpu().numpy()

    # --- argmax mode ---
    s_argmax  = rollout_argmax(model, s0, STEPS, acts)
    pr_argmax = s_argmax.cpu().numpy()

    for tag, pr in [("mean", pr_mean), ("argmax", pr_argmax)]:
        errs_all = []
        cp_errors = {k: [] for k in checkpoint_steps}

        for b_idx, (ep_s, _) in enumerate(valid):
            gt      = ep_s[1:STEPS + 1]
            p       = pr[b_idx, 1:STEPS + 1]
            cue_err = np.sqrt(((p[:, 0] - gt[:, 0]) * TABLE_W) ** 2 +
                              ((p[:, 1] - gt[:, 1]) * TABLE_H) ** 2)
            tgt_err = np.sqrt(((p[:, 7] - gt[:, 7]) * TABLE_W) ** 2 +
                              ((p[:, 8] - gt[:, 8]) * TABLE_H) ** 2)
            step_err = (cue_err + tgt_err) / 2 * 100
            errs_all.append(step_err.mean())
            for lbl, t in checkpoint_steps.items():
                cp_errors[lbl].append(step_err[t - 1])

        mean_err = float(np.mean(errs_all))
        cp_str   = " | ".join(f"{k}={np.mean(v):.1f}cm" for k, v in cp_errors.items())
        print(f"[{label}] {tag:8s}  mean={mean_err:.1f}cm  [{cp_str}]")


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

    print(f"Val episodes: {len(valid_eps)}")
    eval_mode(model, valid_eps, "v6-best")


if __name__ == "__main__":
    main()
