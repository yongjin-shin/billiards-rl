"""
world_model/train_cat_gnn.py

CatGNNResolver 학습.

Phase 1 — single-event (기존 flat 데이터):
    python world_model/train_cat_gnn.py --phase 1

Phase 2 — episode chain (episode_id 포함 데이터 필요):
    python world_model/train_cat_gnn.py --phase 2 --max-chain 2
    python world_model/train_cat_gnn.py --phase 2 --max-chain 5 --resume
    python world_model/train_cat_gnn.py --phase 2 --resume          # full chain

Learning rate:
    CosineAnnealingWarmRestarts — T0=50, T_mult=2  (50→100→200 epoch cycles)

Annealing schedules (per epoch):
    β  (KL weight)    : 0 → 1  over kl_warmup  epochs
    τ  (Gumbel temp)  : 1.0 → 0.5 over tau_anneal epochs
    ss (Bengio SS)    : 1.0 → 0.0 over total epochs  (phase 2 only)
"""

import os, sys, glob, argparse, time, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class _Tee:
    """stdout + file 동시 출력."""
    def __init__(self, path: str):
        self._f   = open(path, "a", buffering=1)
        self._out = sys.__stdout__
    def write(self, s):
        self._out.write(s)
        self._f.write(s)
    def flush(self):
        self._out.flush()
        self._f.flush()
    def close(self):
        self._f.close()


def _setup_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = _Tee(log_path)
    print(f"=== log: {log_path} ===")
    print(f"=== started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.cat_gnn_resolver import CatGNNResolver

COLL_NAMES = {
    0: "ball_ball",
    1: "cue_linear",
    2: "cue_circular",
    3: "pocket",
    4: "stick_ball",
    5: "tgt_linear",
    6: "tgt_circular",
}
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

MAX_SPEED = 12.0
MAX_AVEL  = 300.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class SingleEventDataset(Dataset):
    """Phase 1 / eval용 flat dataset."""

    def __init__(self, paths: list[str], exclude_pocket: bool = True,
                 exclude_stick: bool = True):
        arrays = [np.load(p) for p in paths]
        pre_vel     = np.concatenate([a["pre_vel"]    for a in arrays])
        pre_avel    = np.concatenate([a["pre_avel"]   for a in arrays])
        post_vel    = np.concatenate([a["post_vel"]   for a in arrays])
        post_avel   = np.concatenate([a["post_avel"]  for a in arrays])
        pos         = np.concatenate([a["pos"]        for a in arrays])
        normal      = np.concatenate([a["normal"]     for a in arrays])
        coll_type   = np.concatenate([a["coll_type"]  for a in arrays])
        has_tgt     = np.concatenate([a["has_tgt"]    for a in arrays])
        # will_pocket may be absent in older data files — handle per-file
        will_pocket = np.concatenate([
            a["will_pocket"] if "will_pocket" in a
            else np.zeros(len(a["pre_vel"]), dtype=np.int8)
            for a in arrays
        ])

        mask = np.ones(len(coll_type), dtype=bool)
        if exclude_pocket:
            mask &= coll_type != 3
        if exclude_stick:
            mask &= coll_type != 4
        if not mask.all():
            pre_vel, pre_avel   = pre_vel[mask], pre_avel[mask]
            post_vel, post_avel = post_vel[mask], post_avel[mask]
            pos, normal         = pos[mask], normal[mask]
            coll_type, has_tgt  = coll_type[mask], has_tgt[mask]
            will_pocket         = will_pocket[mask]

        self.pre_vel     = torch.from_numpy(pre_vel).float()
        self.pre_avel    = torch.from_numpy(pre_avel).float()
        self.post_vel    = torch.from_numpy(post_vel).float()
        self.post_avel   = torch.from_numpy(post_avel).float()
        self.pos         = torch.from_numpy(pos).float()
        self.normal      = torch.from_numpy(normal).float()
        self.coll_type   = torch.from_numpy(coll_type.astype(np.int64))
        self.has_tgt     = torch.from_numpy(has_tgt.astype(bool))
        self.will_pocket = torch.from_numpy(will_pocket.astype(np.float32))
        n_pos = int(will_pocket.sum())
        n_neg = len(will_pocket) - n_pos
        # pos_weight for BCE: n_neg/n_pos compensates class imbalance (~30x)
        self.pocket_pos_weight = float(n_neg) / max(n_pos, 1)

        for t, name in COLL_NAMES.items():
            n = (coll_type == t).sum()
            print(f"  {name:20s}: {n:6d}  ({n/len(coll_type)*100:.1f}%)")
        print(f"  Total: {len(coll_type)}")
        print(f"  will_pocket: {n_pos} pos / {n_neg} neg  "
              f"(pos_weight={self.pocket_pos_weight:.1f})")

    def __len__(self): return len(self.pre_vel)

    def __getitem__(self, idx):
        return (self.pos[idx], self.pre_vel[idx], self.pre_avel[idx],
                self.normal[idx], self.has_tgt[idx],
                self.post_vel[idx], self.post_avel[idx],
                self.coll_type[idx], self.will_pocket[idx])


class EpisodeChainDataset(Dataset):
    """
    Phase 2용 episode chain dataset.
    데이터에 episode_id, event_idx 컬럼 필요.
    generate_collision_data.py --tag chain 으로 생성.
    """

    def __init__(
        self,
        paths         : list[str],
        max_chain     : int | None = None,
        exclude_pocket: bool       = True,
        exclude_stick : bool       = True,
    ):
        arrays = [np.load(p) for p in paths]
        if "episode_id" not in arrays[0]:
            raise ValueError(
                "episode_id 없음. "
                "python world_model/generate_collision_data.py 로 재생성 필요."
            )

        keys = ["pre_vel", "pre_avel", "post_vel", "post_avel",
                "pos", "normal", "coll_type", "has_tgt", "episode_id", "event_idx"]
        data = {k: np.concatenate([a[k] for a in arrays]) for k in keys}
        data["will_pocket"] = np.concatenate([
            a["will_pocket"] if "will_pocket" in a
            else np.zeros(len(a["pre_vel"]), dtype=np.int8)
            for a in arrays
        ])

        mask = np.ones(len(data["coll_type"]), dtype=bool)
        if exclude_pocket:
            mask &= data["coll_type"] != 3
        if exclude_stick:
            mask &= data["coll_type"] != 4
        if not mask.all():
            data = {k: v[mask] for k, v in data.items()}

        # episode별 그룹핑
        ep_ids  = data["episode_id"]
        unique  = np.unique(ep_ids)
        self.episodes: list[dict] = []

        for ep in unique:
            idx   = np.where(ep_ids == ep)[0]
            order = np.argsort(data["event_idx"][idx])
            idx   = idx[order]
            if max_chain is not None:
                idx = idx[:max_chain]
            if len(idx) == 0:
                continue
            self.episodes.append({
                "pre_vel"    : torch.from_numpy(data["pre_vel"][idx]).float(),
                "pre_avel"   : torch.from_numpy(data["pre_avel"][idx]).float(),
                "post_vel"   : torch.from_numpy(data["post_vel"][idx]).float(),
                "post_avel"  : torch.from_numpy(data["post_avel"][idx]).float(),
                "pos"        : torch.from_numpy(data["pos"][idx]).float(),
                "normal"     : torch.from_numpy(data["normal"][idx]).float(),
                "has_tgt"    : torch.from_numpy(data["has_tgt"][idx].astype(bool)),
                "coll_type"  : torch.from_numpy(data["coll_type"][idx].astype(np.int64)),
                "will_pocket": torch.from_numpy(
                    data["will_pocket"][idx].astype(np.float32)),
            })

        lens = [len(e["pre_vel"]) for e in self.episodes]
        print(f"  {len(self.episodes)} episodes  "
              f"avg {np.mean(lens):.1f} events/ep  "
              f"max {max(lens)}")

    def __len__(self): return len(self.episodes)
    def __getitem__(self, idx): return self.episodes[idx]


def chain_collate_fn(batch: list[dict]) -> dict:
    """Variable-length episodes → zero-padded batch."""
    T = max(ep["pre_vel"].shape[0] for ep in batch)
    B = len(batch)

    out = {
        "pre_vel"    : torch.zeros(B, T, 2, 2),
        "pre_avel"   : torch.zeros(B, T, 2, 3),
        "post_vel"   : torch.zeros(B, T, 2, 2),
        "post_avel"  : torch.zeros(B, T, 2, 3),
        "pos"        : torch.zeros(B, T, 2, 2),
        "normal"     : torch.zeros(B, T, 2),
        "has_tgt"    : torch.zeros(B, T, dtype=torch.bool),
        "coll_type"  : torch.zeros(B, T, dtype=torch.long),
        "will_pocket": torch.zeros(B, T),
        "lengths"    : torch.tensor([ep["pre_vel"].shape[0] for ep in batch]),
    }
    for i, ep in enumerate(batch):
        L = ep["pre_vel"].shape[0]
        for k in ["pre_vel", "pre_avel", "post_vel", "post_avel", "pos"]:
            out[k][i, :L] = ep[k]
        out["normal"][i, :L]      = ep["normal"]
        out["has_tgt"][i, :L]     = ep["has_tgt"]
        out["coll_type"][i, :L]   = ep["coll_type"]
        out["will_pocket"][i, :L] = ep["will_pocket"]
    return out


# ── Loss ──────────────────────────────────────────────────────────────────────

def recon_loss(
    dv       : torch.Tensor,   # (B,2,2)
    da       : torch.Tensor,   # (B,2,3)
    pre_vel  : torch.Tensor,   # (B,2,2)  — SS 적용된 실제 입력
    pre_avel : torch.Tensor,   # (B,2,3)
    post_vel : torch.Tensor,   # (B,2,2)  — 항상 GT
    post_avel: torch.Tensor,   # (B,2,3)
    has_tgt  : torch.Tensor,   # (B,) bool
) -> torch.Tensor:
    """
    MSE on absolute post-state.
    GT target은 항상 post_vel_GT — SS 입력 변화에 무관하게 안정적.
    """
    pred_v = pre_vel  + dv
    pred_a = pre_avel + da

    loss_cue_v = nn.functional.mse_loss(pred_v[:, 0],  post_vel[:, 0])
    loss_cue_a = nn.functional.mse_loss(pred_a[:, 0],  post_avel[:, 0])

    n_tgt      = has_tgt.sum().clamp(min=1)
    loss_tgt_v = ((pred_v[:, 1] - post_vel[:, 1] ).pow(2).mean(-1) * has_tgt).sum() / n_tgt
    loss_tgt_a = ((pred_a[:, 1] - post_avel[:, 1]).pow(2).mean(-1) * has_tgt).sum() / n_tgt

    return loss_cue_v + loss_cue_a + loss_tgt_v + loss_tgt_a


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model  : CatGNNResolver,
    loader : DataLoader,
    device : str,
) -> tuple[float, dict, float, float]:
    """
    Returns (avg_loss, type_rmse, recall, precision)
    """
    model.eval()
    total_loss = 0.0
    type_err   = {t: [] for t in COLL_NAMES}
    tp = fp = fn = tn = 0

    for batch in loader:
        pos, pre_vel, pre_avel, normal, has_tgt, post_vel, post_avel, \
            coll_type, will_pocket = [x.to(device) for x in batch]

        dv, da, pocket_logit = model(pos, pre_vel, pre_avel, normal, has_tgt)
        loss = recon_loss(dv, da, pre_vel, pre_avel, post_vel, post_avel, has_tgt)
        total_loss += loss.item() * len(pos)

        pred_post_v = pre_vel + dv
        err_cue = (pred_post_v[:, 0] - post_vel[:, 0]).pow(2).mean(-1).sqrt()
        for t in type_err:
            m = coll_type == t
            if m.any():
                type_err[t].extend((err_cue[m] * MAX_SPEED).cpu().tolist())

        pred = (pocket_logit > 0)
        lab  = will_pocket.bool()
        tp += (pred &  lab).sum().item()
        fp += (pred & ~lab).sum().item()
        fn += (~pred &  lab).sum().item()
        tn += (~pred & ~lab).sum().item()

    n         = sum(len(v) for v in type_err.values())
    avg       = total_loss / n if n > 0 else 0.0
    type_rmse = {COLL_NAMES[t]: float(np.mean(e)) for t, e in type_err.items() if e}
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return avg, type_rmse, recall, precision


# ── Charts ────────────────────────────────────────────────────────────────────

def save_charts(history: dict, out_dir: str):
    """Plotly 인터랙티브 트레이닝 대시보드를 HTML로 저장."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping charts")
        return

    epochs = history["epochs"]
    if not epochs:
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Train / Val Loss", "LR Schedule",
            "Per-Type Velocity RMSE (m/s)", "Pocket Recall",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # ── Row 1: Loss + LR ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             name="train", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             name="val",   line=dict(color="#ff7f0e")), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=history["lr"],
                             name="lr", line=dict(color="#2ca02c")), row=1, col=2)
    fig.update_yaxes(type="log", title_text="LR", row=1, col=2)

    # ── Row 2: Per-type RMSE + Pocket Recall ─────────────────────────────────
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]
    for idx, (name, rmse_list) in enumerate(history["type_rmse"].items()):
        ep_slice = epochs[:len(rmse_list)]
        fig.add_trace(go.Scatter(x=ep_slice, y=rmse_list,
                                 name=name, line=dict(color=colors[idx % len(colors)])),
                      row=2, col=1)
    fig.update_yaxes(title_text="RMSE m/s", row=2, col=1)

    if history["pocket_acc"]:
        fig.add_trace(go.Scatter(x=epochs, y=[v * 100 for v in history["pocket_acc"]],
                                 name="recall%", line=dict(color="#d62728")),
                      row=2, col=2)
    fig.update_yaxes(title_text="Recall (%)", row=2, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.update_layout(height=700, title_text="GNN Training Dashboard",
                      legend=dict(groupclick="toggleitem"))

    path = os.path.join(out_dir, "training_dashboard.html")
    fig.write_html(path)
    print(f"Charts → {path}")


# ── Phase 1: single-event ─────────────────────────────────────────────────────

def train_p1(args):
    _setup_log(os.path.join(args.out_dir, "train_p1.log"))
    paths = sorted(glob.glob(os.path.join(args.data, args.data_pattern)))
    if not paths:
        raise FileNotFoundError(f"No {args.data_pattern} in {args.data}")
    print(f"Loading {len(paths)} file(s):")
    for p in paths: print(f"  {p}")

    dataset = SingleEventDataset(paths)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    pos_w = torch.tensor(dataset.pocket_pos_weight, device=DEVICE)
    print(f"Train: {n_train}  Val: {n_val}  pocket_pos_weight={pos_w.item():.1f}")

    model = CatGNNResolver(
        hidden=args.hidden, node_dim=args.node_dim, z_dim=args.z_dim,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}  device={DEVICE}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.T0, T_mult=args.T_mult, eta_min=args.lr * 0.01,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_p1.pt")
    best_val, stall = float("inf"), 0

    history: dict = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "type_rmse": defaultdict(list), "lr": [], "pocket_acc": [],
    }

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()

        for batch in train_dl:
            pos, pre_vel, pre_avel, normal, has_tgt, post_vel, post_avel, \
                _, will_pocket = [x.to(DEVICE) for x in batch]

            dv, da, pocket_logit = model(pos, pre_vel, pre_avel, normal, has_tgt)
            l_r = recon_loss(dv, da, pre_vel, pre_avel, post_vel, post_avel, has_tgt)
            l_p = F.binary_cross_entropy_with_logits(
                pocket_logit, will_pocket, pos_weight=pos_w)
            loss = l_r + args.pocket_weight * l_p

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(pos)

        sched.step()
        avg_train = total_loss / n_train

        if ep % args.eval_every == 0 or ep == args.epochs:
            val_loss, type_rmse, recall, precision = evaluate(model, val_dl, DEVICE)
            lr_now   = opt.param_groups[0]["lr"]
            rmse_str = "  ".join(f"{k}={v:.4f}" for k, v in type_rmse.items())
            dt       = time.time() - t0

            history["epochs"].append(ep)
            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr_now)
            history["pocket_acc"].append(recall)
            for name, v in type_rmse.items():
                history["type_rmse"][name].append(v)

            marker = ""
            if val_loss < best_val:
                best_val, stall = val_loss, 0
                torch.save({
                    "epoch": ep, "model": model.state_dict(),
                    "val_loss": val_loss, "type_rmse": type_rmse,
                    "hidden": args.hidden, "node_dim": args.node_dim,
                    "z_dim": args.z_dim,
                }, ckpt_path)
                marker = "*"
            else:
                stall += 1
                if args.patience > 0 and stall >= args.patience:
                    print(f"Early stop at ep{ep}")
                    break

            print(f"ep{ep:4d}  train={avg_train:.5f}  val={val_loss:.5f}  "
                  f"recall={recall*100:.1f}%  prec={precision*100:.1f}%  "
                  f"lr={lr_now:.2e}  [{rmse_str}]  {dt:.1f}s  {marker}")
            save_charts(history, args.out_dir)

    print(f"\nBest val: {best_val:.5f}  →  {ckpt_path}")


# ── Phase 2: episode chain ────────────────────────────────────────────────────

def _chain_forward(
    model   : CatGNNResolver,
    batch   : dict,
    ss      : float,    # Bengio SS ratio (1.0=GT, 0.0=GNN predicted)
    device  : str,
    pos_w   : torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-BPTT chain loss."""
    lengths  = batch["lengths"]
    T        = batch["pre_vel"].shape[1]

    bv  = batch["pre_vel"].to(device)
    ba  = batch["pre_avel"].to(device)
    pv  = batch["post_vel"].to(device)
    pa  = batch["post_avel"].to(device)
    pos = batch["pos"].to(device)
    nrm = batch["normal"].to(device)
    htg = batch["has_tgt"].to(device)

    total  = torch.tensor(0.0, device=device)
    n_step = 0
    prev_v = prev_a = None

    for t in range(T):
        active = (t < lengths).to(device)
        if not active.any():
            break

        if prev_v is None or random.random() < ss:
            in_v, in_a = bv[:, t], ba[:, t]
        else:
            in_v, in_a = prev_v, prev_a

        dv, da, pocket_logit = model(pos[:, t], in_v, in_a, nrm[:, t], htg[:, t])

        wp = batch["will_pocket"][:, t].to(device)
        l_r = recon_loss(dv, da, in_v, in_a, pv[:, t], pa[:, t], htg[:, t])
        l_p = F.binary_cross_entropy_with_logits(pocket_logit, wp, pos_weight=pos_w)

        step_loss = (l_r + l_p) * active.float().mean()
        total     = total + step_loss
        n_step   += 1

        prev_v = in_v + dv
        prev_a = in_a + da

    return total / max(n_step, 1)


def train_p2(args):
    _setup_log(os.path.join(args.out_dir, "train_p2.log"))
    paths = sorted(glob.glob(os.path.join(args.data, args.data_pattern)))

    # Train: chain dataset
    chain_ds = EpisodeChainDataset(paths, max_chain=args.max_chain)
    n_val    = max(1, int(len(chain_ds) * 0.1))
    n_train  = len(chain_ds) - n_val
    train_ds, _ = random_split(chain_ds, [n_train, n_val],
                               generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.chain_batch, shuffle=True,
                          collate_fn=chain_collate_fn, num_workers=0)

    # Val: flat single-event (일관된 metric)
    flat_ds = SingleEventDataset(paths)
    nv2     = max(1, int(len(flat_ds) * 0.1))
    _, val_flat = random_split(flat_ds, [len(flat_ds) - nv2, nv2],
                               generator=torch.Generator().manual_seed(42))
    val_dl  = DataLoader(val_flat, batch_size=args.batch, shuffle=False, num_workers=2)
    pos_w = torch.tensor(flat_ds.pocket_pos_weight, device=DEVICE)
    print(f"Chain train: {n_train} eps  Val (flat): {nv2}  "
          f"pocket_pos_weight={pos_w.item():.1f}")

    # 모델 — Phase 1 ckpt 로드
    model = CatGNNResolver(
        hidden=args.hidden, node_dim=args.node_dim, z_dim=args.z_dim,
    ).to(DEVICE)

    p1_ckpt = os.path.join(args.out_dir, "best_p1.pt")
    if args.resume and os.path.exists(p1_ckpt):
        ck = torch.load(p1_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Loaded P1: val={ck['val_loss']:.5f}")
    elif args.resume:
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {p1_ckpt}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.T0, T_mult=args.T_mult, eta_min=args.lr * 0.003,
    )

    tag       = f"chain{args.max_chain or 'full'}"
    ckpt_path = os.path.join(args.out_dir, f"best_{tag}.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    best_val, stall = float("inf"), 0

    history: dict = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "type_rmse": defaultdict(list), "lr": [], "pocket_acc": [],
    }

    ss_schedule = np.linspace(1.0, 0.0, args.epochs)

    for ep in range(1, args.epochs + 1):
        ss = float(ss_schedule[ep - 1])

        model.train()
        total_loss, t0 = 0.0, time.time()

        for batch in train_dl:
            loss = _chain_forward(model, batch, ss, DEVICE, pos_w=pos_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg_train = total_loss / max(len(train_dl), 1)

        if ep % args.eval_every == 0 or ep == args.epochs:
            val_loss, type_rmse, recall, precision = evaluate(model, val_dl, DEVICE)
            lr_now   = opt.param_groups[0]["lr"]
            rmse_str = "  ".join(f"{k}={v:.4f}" for k, v in type_rmse.items())
            dt       = time.time() - t0

            history["epochs"].append(ep)
            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr_now)
            history["pocket_acc"].append(recall)
            for name, v in type_rmse.items():
                history["type_rmse"][name].append(v)

            marker = ""
            if val_loss < best_val:
                best_val, stall = val_loss, 0
                torch.save({
                    "epoch": ep, "model": model.state_dict(),
                    "val_loss": val_loss, "type_rmse": type_rmse,
                    "hidden": args.hidden, "node_dim": args.node_dim,
                    "z_dim": args.z_dim, "tag": tag,
                }, ckpt_path)
                marker = "*"
            else:
                stall += 1
                if args.patience > 0 and stall >= args.patience:
                    print(f"Early stop at ep{ep}")
                    break

            print(f"ep{ep:4d}  train={avg_train:.5f}  val={val_loss:.5f}  "
                  f"recall={recall*100:.1f}%  prec={precision*100:.1f}%  "
                  f"ss={ss:.2f}  lr={lr_now:.2e}  [{rmse_str}]  {dt:.1f}s  {marker}")
            save_charts(history, args.out_dir)

    print(f"\nBest val: {best_val:.5f}  →  {ckpt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    # phase
    p.add_argument("--phase",       type=int,   default=1, choices=[1, 2])
    # data / output
    p.add_argument("--data",        type=str,   default="world_model/data_collision")
    p.add_argument("--data-pattern",type=str,   default="*.npz",
                   help="glob pattern within --data dir, e.g. 'collision_v3*.npz'")
    p.add_argument("--out-dir",     type=str,   default="world_model/results/cat_gnn")
    # model
    p.add_argument("--hidden",       type=int,   nargs="+", default=[128, 128])
    p.add_argument("--node-dim",     type=int,   default=64)
    p.add_argument("--z-dim",        type=int,   default=64)
    # training
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch",        type=int,   default=1024)
    p.add_argument("--chain-batch",  type=int,   default=128)
    p.add_argument("--max-chain",    type=int,   default=None)
    p.add_argument("--lr",           type=float, default=3e-4)
    # LR schedule
    p.add_argument("--T0",           type=int,   default=50)
    p.add_argument("--T-mult",       type=int,   default=2)
    # loss
    p.add_argument("--pocket-weight",type=float, default=1.0)
    # misc
    p.add_argument("--eval-every",  type=int,   default=5)
    p.add_argument("--patience",    type=int,   default=30)
    p.add_argument("--resume",      action="store_true")
    args = p.parse_args()

    if args.phase == 1:
        train_p1(args)
    else:
        train_p2(args)


if __name__ == "__main__":
    main()
