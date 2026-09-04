"""
world_model/compare_recon.py — 실제 trajectory vs VAE reconstruction 비교

Usage:
    python world_model/compare_recon.py \
        --ckpt world_model/checkpoints/vae_z8_20260322_200820.pt \
        --model logs/experiments/.../best_model/best_model \
        --n-samples 3
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.model import TrajectoryVAE, MAX_EVENTS, EVENT_DIM, N_EVENT_TYPES
from world_model.generate_data import extract_trajectory, EVENT_TYPES
from simulator import BilliardsEnv


EVENT_COLORS = {
    "stick_ball":              "cyan",
    "ball_ball":               "red",
    "ball_linear_cushion":     "orange",
    "ball_circular_cushion":   "yellow",
    "ball_pocket":             "lime",
    "sliding_rolling":         "white",
    "rolling_spinning":        "lightblue",
    "rolling_stationary":      "pink",
    "spinning_stationary":     "violet",
    "none":                    "gray",
}


def decode_events(events_enc, length):
    """인코딩된 이벤트 배열을 (x, y, type_str) 리스트로 변환"""
    result = []
    for i in range(min(length, MAX_EVENTS)):
        x, y = events_enc[i, 0], events_enc[i, 1]
        type_idx = int(np.argmax(events_enc[i, 2:2 + N_EVENT_TYPES]))
        et = EVENT_TYPES[type_idx]
        result.append((x, y, et))
    return result


def plot_trajectory(ax, events, title, ref_xy=None):
    """events: list of (x, y, type_str). ref_xy: stick_ball 절대 좌표 (복원용)"""
    ax.set_facecolor("#1a5c1a")
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([]); ax.set_yticks([])

    if len(events) == 0:
        return

    # 상대 좌표 → 절대 좌표 복원 (시각화용)
    rx, ry = (ref_xy if ref_xy is not None else (0.0, 0.0))
    xs = [e[0] + rx for e in events]
    ys = [e[1] + ry for e in events]
    types = [e[2] for e in events]

    # 연결선
    ax.plot(xs, ys, "w-", lw=0.8, alpha=0.4, zorder=1)

    # 이벤트 포인트 (타입별 색상)
    for x, y, et in zip(xs, ys, types):
        color = EVENT_COLORS.get(et, "gray")
        ax.scatter(x, y, c=color, s=40, zorder=3, linewidths=0)

    # 원점(stick_ball) 표시
    ax.scatter([rx], [ry], c="cyan", s=80, marker="*", zorder=5)

    # 이벤트 순서 번호
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(i), fontsize=5, color="white", zorder=6,
                ha="center", va="bottom")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str, required=True)
    parser.add_argument("--model",     type=str, required=True,
                        help="SAC model path")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       type=str,
                        default="world_model/results_z8_b01/recon_compare.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VAE 로드
    ckpt  = torch.load(args.ckpt, map_location=device)
    z_dim = ckpt["z_dim"]
    decoder_type = ckpt.get("decoder_type", "lstm")
    saved_args   = ckpt.get("args", {})
    hidden_enc   = saved_args.get("hidden_enc", 64)
    hidden_dec   = saved_args.get("hidden_dec", 128)
    vae   = TrajectoryVAE(z_dim=z_dim, decoder_type=decoder_type,
                          hidden_enc=hidden_enc, hidden_dec=hidden_dec).to(device)
    vae.load_state_dict(ckpt["state"])
    vae.eval()
    print(f"VAE loaded: z_dim={z_dim}  val_loss={ckpt['val_loss']:.4f}")

    # SAC 로드
    from stable_baselines3 import SAC
    sac = SAC.load(args.model)
    print(f"SAC loaded: {args.model}")

    # 환경
    env = BilliardsEnv(n_balls=1)
    W, L = env.table.w, env.table.l
    rng  = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(args.n_samples, 2,
                             figsize=(8, args.n_samples * 3.5))
    if args.n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(args.n_samples):
        seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed)
        action = sac.predict(obs, deterministic=True)[0]
        obs_next, reward, *_ = env.step(action)

        # 실제 trajectory 추출 (절대좌표)
        events_enc, length, n_bounces = extract_trajectory(env.system, target_id="1")

        # VAE reconstruction (autoregressive — 실제 생성 품질)
        ev_tensor  = torch.tensor(events_enc, dtype=torch.float32).unsqueeze(0).to(device)
        len_tensor = torch.tensor([length], dtype=torch.long).to(device)
        with torch.no_grad():
            x_recon, mu, logvar, z = vae.reconstruct(ev_tensor, len_tensor)
        recon_enc = x_recon.squeeze(0).cpu().numpy()

        # 디코딩
        real_events  = decode_events(events_enc,  length)
        recon_events = decode_events(recon_enc,   length)

        pocketed = any(e[2] == "ball_pocket" for e in real_events)
        label = "POCKET" if pocketed else "MISS"

        # 절대 좌표 범위 설정
        ax_real  = axes[i, 0]
        ax_recon = axes[i, 1]
        for ax in [ax_real, ax_recon]:
            ax.set_xlim(0, W); ax.set_ylim(0, L)

        # 절대좌표이므로 ref_xy=(0,0) (아무것도 더하지 않음)
        plot_trajectory(ax_real,  real_events,
                        f"Sample {i+1} — Real [{label}]\n"
                        f"len={length}  bounces={n_bounces}  "
                        f"a=({action[0]:.2f},{action[1]:.2f})")
        plot_trajectory(ax_recon, recon_events,
                        f"Sample {i+1} — Reconstructed\n"
                        f"z={np.round(mu.squeeze().cpu().numpy(), 2)}")

        print(f"\n[Sample {i+1}] {label}  len={length}  bounces={n_bounces}")
        print(f"  action: delta_angle={action[0]:.3f}  speed={action[1]:.3f}")
        print(f"  z: {mu.squeeze().cpu().numpy().round(3)}")
        print(f"  Real   : {[(round(x,2), round(y,2), t) for x,y,t in real_events]}")
        print(f"  Recon  : {[(round(x,2), round(y,2), t) for x,y,t in recon_events]}")

    # 범례
    legend_elements = [
        plt.scatter([], [], c=c, s=40, label=t)
        for t, c in EVENT_COLORS.items() if t != "none"
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=7, framealpha=0.3)

    plt.suptitle(f"VAE Reconstruction Comparison  (β=0.1, z={z_dim}, relpos)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    plt.close()
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
