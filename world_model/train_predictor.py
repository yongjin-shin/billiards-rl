"""
world_model/train_predictor.py — (obs, act) → trajectory predictor 학습

Usage:
    # MLP
    python -m world_model.train_predictor --model mlp --hidden 256 512 256

    # LSTM
    python -m world_model.train_predictor --model lstm --ctx-hidden 128 --lstm-hidden 256 --lstm-layers 1

    # no wandb
    python -m world_model.train_predictor --model lstm --no-wandb
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.predictor import (
    MLPPredictor, LSTMPredictor, predictor_loss,
    EVENT_DIM, MAX_EVENTS,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, tags=None):
        all_npz = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

        meta_path = os.path.join(data_dir, "metadata.json")
        tag_map   = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for entry in json.load(f):
                    tag_map[entry["file"]] = entry["tag"]

        obs_list, act_list, events_list, lengths_list = [], [], [], []
        loaded = []
        for fname in all_npz:
            if tags is not None and tag_map.get(fname) not in tags:
                continue
            d = np.load(os.path.join(data_dir, fname))
            obs_list.append(d["obs"])
            act_list.append(d["actions"])
            events_list.append(d["events"])
            lengths_list.append(d["lengths"])
            loaded.append(fname)

        if not obs_list:
            raise ValueError(f"No data found in {data_dir} (tags={tags})")

        self.obs     = torch.from_numpy(np.concatenate(obs_list)).float()
        self.actions = torch.from_numpy(np.concatenate(act_list)).float()
        self.events  = torch.from_numpy(np.concatenate(events_list)).float()
        self.lengths = torch.from_numpy(np.concatenate(lengths_list)).long()

        print(f"  Files  : {loaded}")
        print(f"  Total  : {len(self.obs):,} episodes")
        print(f"  Avg len: {self.lengths.float().mean():.1f}  "
              f"max={self.lengths.max().item()}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx], self.events[idx], self.lengths[idx]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train (obs, act) → trajectory predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── task ──
    p.add_argument("--model",         type=str,   default="lstm",
                   choices=["mlp", "lstm"])
    p.add_argument("--data",          type=str,   default="world_model/data_abs",
                   help="Path to data directory (contains .npz + metadata.json)")
    p.add_argument("--tags",          type=str,   nargs="+",
                   default=["sac_abs", "random_abs"],
                   help="Dataset tags to load")

    # ── training ──
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--pos-weight",    type=float, default=1.0,
                   help="Position MSE loss weight relative to type CE loss")
    p.add_argument("--device",        type=str,   default="cpu")

    # ── MLP architecture ──
    p.add_argument("--hidden",        type=int,   nargs="+", default=[256, 512, 256],
                   help="(MLP) hidden layer sizes, e.g. --hidden 256 512 256")

    # ── LSTM architecture ──
    p.add_argument("--ctx-hidden",       type=int,   default=128,
                   help="(LSTM) context encoder hidden size")
    p.add_argument("--lstm-hidden",      type=int,   default=256,
                   help="(LSTM) LSTM hidden size")
    p.add_argument("--lstm-layers",      type=int,   default=1,
                   help="(LSTM) number of LSTM layers")

    # ── LSTM training strategy ──
    p.add_argument("--strategy",         type=str,   default="curriculum",
                   choices=["curriculum", "tf_ratio"],
                   help="(LSTM) training strategy")
    # curriculum
    p.add_argument("--curriculum-epochs",type=int,   default=80,
                   help="(LSTM/curriculum) epochs to ramp ar_steps from 0 → MAX_EVENTS")
    # tf_ratio (scheduled sampling)
    p.add_argument("--ss-epochs",        type=int,   default=30,
                   help="(LSTM/tf_ratio) scheduled sampling anneal epochs")
    p.add_argument("--ss-min-ratio",     type=float, default=0.0,
                   help="(LSTM/tf_ratio) min teacher forcing ratio after annealing")

    # ── logging ──
    p.add_argument("--wandb-project", type=str,   default="billiards-wm-predictor")
    p.add_argument("--no-wandb",      action="store_true")

    return p.parse_args()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(args, obs_dim: int) -> torch.nn.Module:
    if args.model == "mlp":
        return MLPPredictor(
            obs_dim     = obs_dim,
            act_dim     = 2,
            hidden_dims = args.hidden,
        )
    else:
        return LSTMPredictor(
            obs_dim     = obs_dim,
            act_dim     = 2,
            ctx_hidden  = args.ctx_hidden,
            lstm_hidden = args.lstm_hidden,
            lstm_layers = args.lstm_layers,
        )


# ── Experiment directory ──────────────────────────────────────────────────────

def make_exp_dir(args) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.model == "mlp":
        arch = "h" + "_".join(str(h) for h in args.hidden)
    else:
        strat = f"cur{args.curriculum_epochs}" if args.strategy == "curriculum" \
                else f"tf{args.ss_epochs}"
        arch = f"ctx{args.ctx_hidden}_h{args.lstm_hidden}_l{args.lstm_layers}_{strat}"
    name = f"pred_{args.model}_{arch}_s{args.seed}_{ts}"
    path = os.path.join("world_model", "checkpoints", name)
    os.makedirs(path, exist_ok=True)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    W      = 70

    # ── data ──
    print(f"\n{'═' * W}")
    print(f"  Loading data: {args.data}  tags={args.tags}")
    dataset = TrajectoryDataset(args.data, tags=args.tags)
    obs_dim = dataset.obs.shape[1]

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── model ──
    model    = build_model(args, obs_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5,
    )

    # ── experiment dir & config ──
    exp_dir  = make_exp_dir(args)
    run_name = os.path.basename(exp_dir)

    config = {
        "model":        args.model,
        "tags":         args.tags,
        "seed":         args.seed,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "pos_weight":   args.pos_weight,
        "device":       args.device,
        "obs_dim":      obs_dim,
        "n_params":     n_params,
        "n_train":      n_train,
        "n_val":        n_val,
        "exp_dir":      exp_dir,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.model == "mlp":
        config["hidden"] = args.hidden
    else:
        config.update({
            "ctx_hidden":        args.ctx_hidden,
            "lstm_hidden":       args.lstm_hidden,
            "lstm_layers":       args.lstm_layers,
            "strategy":          args.strategy,
            "curriculum_epochs": args.curriculum_epochs,
            "ss_epochs":         args.ss_epochs,
            "ss_min_ratio":      args.ss_min_ratio,
        })

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── wandb ──
    if not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = config,
            tags    = [f"model:{args.model}", f"seed:{args.seed}"],
        )

    # ── header ──
    print(f"{'═' * W}")
    print(f"  World Model Predictor  model={args.model}  seed={args.seed}")
    print(f"  obs_dim={obs_dim}  act_dim=2  → event_dim={EVENT_DIM}  max_events={MAX_EVENTS}")
    if args.model == "mlp":
        print(f"  hidden={args.hidden}")
    else:
        print(f"  ctx_hidden={args.ctx_hidden}  lstm_hidden={args.lstm_hidden}"
              f"  lstm_layers={args.lstm_layers}  strategy={args.strategy}")
        if args.strategy == "curriculum":
            print(f"  curriculum: ar_steps 0 → {MAX_EVENTS} over {args.curriculum_epochs} epochs")
        else:
            print(f"  tf_ratio: 1.0 → {args.ss_min_ratio} over {args.ss_epochs} epochs")
    print(f"  n_params={n_params:,}  lr={args.lr}  batch={args.batch_size}"
          f"  epochs={args.epochs}  pos_weight={args.pos_weight}")
    print(f"  train={n_train:,}  val={n_val:,}")
    print(f"  exp_dir: {exp_dir}")
    print(f"{'═' * W}\n")

    best_val = float("inf")
    t_start  = time.time()

    for epoch in range(1, args.epochs + 1):

        # ── 학습 전략 파라미터 계산 ──
        if args.model == "lstm":
            if args.strategy == "curriculum":
                # ar_steps: 0 → MAX_EVENTS 선형 증가
                progress = min(1.0, (epoch - 1) / max(1, args.curriculum_epochs - 1))
                ar_steps = int(round(progress * MAX_EVENTS))
                tf_ratio = None
            else:
                # tf_ratio: 1.0 → ss_min_ratio 선형 감소
                progress = min(1.0, (epoch - 1) / max(1, args.ss_epochs - 1))
                tf_ratio = 1.0 - progress * (1.0 - args.ss_min_ratio)
                ar_steps = None

        # ── train ──
        model.train()
        train_total = train_pos = train_type = 0.0

        for obs, act, events, lengths in train_loader:
            obs, act, events = obs.to(device), act.to(device), events.to(device)
            lengths = lengths.to(device)

            if args.model == "mlp":
                pred = model(obs, act)
            elif args.strategy == "curriculum":
                pred = model(obs, act, x_teacher=events, ar_steps=ar_steps)
            else:
                pred = model(obs, act, x_teacher=events, tf_ratio=tf_ratio)

            loss, pos_l, type_l = predictor_loss(events, pred, lengths, args.pos_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs           = obs.size(0)
            train_total += loss.item()  * bs
            train_pos   += pos_l.item() * bs
            train_type  += type_l.item()* bs

        train_total /= n_train
        train_pos   /= n_train
        train_type  /= n_train

        # ── val — 완전 AR inference (추론 조건과 동일) ──
        model.eval()
        val_total = val_pos = val_type = 0.0

        with torch.no_grad():
            for obs, act, events, lengths in val_loader:
                obs, act, events = obs.to(device), act.to(device), events.to(device)
                lengths = lengths.to(device)

                pred = model(obs, act) if args.model == "mlp" \
                       else model(obs, act)   # LSTM: x_teacher=None → 완전 AR

                loss, pos_l, type_l = predictor_loss(events, pred, lengths, args.pos_weight)
                bs          = obs.size(0)
                val_total  += loss.item()  * bs
                val_pos    += pos_l.item() * bs
                val_type   += type_l.item()* bs

        val_total /= n_val
        val_pos   /= n_val
        val_type  /= n_val

        scheduler.step(val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── best checkpoint ──
        if val_total < best_val:
            best_val = val_total
            torch.save({
                "epoch":    epoch,
                "model":    args.model,
                "state":    model.state_dict(),
                "val_loss": best_val,
                "args":     vars(args),
            }, os.path.join(exp_dir, "best.pt"))

        # ── wandb log ──
        log_dict = {
            "train/loss":      train_total,
            "train/pos_loss":  train_pos,
            "train/type_loss": train_type,
            "val/loss":        val_total,
            "val/pos_loss":    val_pos,
            "val/type_loss":   val_type,
            "val/best_loss":   best_val,
            "train/lr":        current_lr,
        }
        if args.model == "lstm":
            if args.strategy == "curriculum":
                log_dict["train/ar_steps"]      = ar_steps
                log_dict["train/ar_steps_frac"] = ar_steps / MAX_EVENTS
            else:
                log_dict["train/tf_ratio"] = tf_ratio

        if not args.no_wandb:
            wandb.log(log_dict, step=epoch)

        # ── console ──
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            if args.model == "lstm":
                if args.strategy == "curriculum":
                    strategy_str = f"  ar={ar_steps}/{MAX_EVENTS}"
                else:
                    strategy_str = f"  tf={tf_ratio:.2f}"
            else:
                strategy_str = ""
            print(
                f"  [{epoch:>4d}/{args.epochs}]"
                f"  train={train_total:.4f}"
                f" (pos={train_pos:.4f} type={train_type:.4f})"
                f"  val={val_total:.4f}"
                f" (pos={val_pos:.4f} type={val_type:.4f})"
                f"  best={best_val:.4f}"
                f"{strategy_str}"
                f"  lr={current_lr:.1e}"
                f"  {elapsed:.0f}s"
            )

    # ── final ──
    elapsed_total = time.time() - t_start
    torch.save({
        "epoch":    args.epochs,
        "model":    args.model,
        "state":    model.state_dict(),
        "val_loss": val_total,
        "args":     vars(args),
    }, os.path.join(exp_dir, "final.pt"))

    print(f"\n{'─' * W}")
    print(f"  Done  ({elapsed_total/60:.1f} min)")
    print(f"  best val_loss = {best_val:.4f}  →  {exp_dir}/best.pt")
    print(f"{'─' * W}")

    if not args.no_wandb:
        wandb.log({"final/best_val_loss": best_val})
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
