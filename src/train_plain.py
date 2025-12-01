#!/usr/bin/env python
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .utils import *
from .data import get_loaders
from .data_custom import get_custom_loaders
from .models import *



# ---------------------------- Train/Eval --------------------------
@T.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum / max(total,1), correct / max(total,1)

def train_plain_epoch(model, opt, loader, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with T.no_grad():
            loss_sum += loss.item() * yb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return loss_sum / max(total,1), correct / max(total,1)

# ----------------------------- Main -------------------------------
def parse_args():
    p = argparse.ArgumentParser("Train plain CNNs on distilled subsets (no VAE, no HASeparator).")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","cifar10","custom"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    p.add_argument("--seed", type=int, default=42)

    # model
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--arch", type=str, default="simple", choices=["conv","mlp","raw","simple","tiny"], help="mlp=SDVAE_Embedder, conv=LatentConvEmbedder, raw=flatten, simple=SIMPLE_Embedder, tiny=TinyLatentEmbedder")
    p.add_argument("--keep-pct", type=float, default=100.0, help="Randomly keep this percentage of the training set (e.g., 10, 20, ... 90), 100.0 means use full dataset.")
    p.add_argument("--distill", action="store_true" )
    # opt
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # distilled subset
    p.add_argument("--subset-file", type=str, default="", help="runs/.../subsets/keep_idx_XX.txt")
    # custom dataset
    p.add_argument("--img-root", type=str, default="")
    p.add_argument("--labels-csv", type=str, default="")
    p.add_argument("--val-split", type=float, default=0.2)

    # run naming
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # out dir
    run_id = make_plain_id(args)
    out_dir = prepare_plain_dir(run_id)
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / "config.json")

    # loaders
    if args.dataset == "custom":
        if not args.img_root or not args.labels_csv:
            raise SystemExit("--dataset custom requires --img-root and --labels-csv")
        train_ds, test_ds, train_loader_full, test_loader, train_loader_noshuf, num_classes = \
            get_custom_loaders(args.img_root, args.labels_csv, args.image_size, args.batch_size,
                               args.num_workers, device, test_size=args.val_split, seed=args.seed)
        in_ch = 3
    else:
        train_ds, test_ds, train_loader_full, test_loader, _ = \
            get_loaders(args.dataset, args.image_size, args.batch_size, args.num_workers, device)
        num_classes = 10 if args.dataset in ["mnist","cifar10"] else 2
        in_ch = 1 if args.dataset == "mnist" else 3

    # distilled subset (IMPORTANT: indices are relative to the train split used during your DD run)
    if args.subset_file:
        keep_idx = np.loadtxt(args.subset_file, dtype=int).reshape(-1)
        sub_train = Subset(train_ds, keep_idx.tolist())
        pin = (device.type == "cuda")
        train_loader = DataLoader(sub_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin)
        print(f"Using distilled subset: {len(sub_train)} samples | Test size: {len(test_ds)}")
    elif args.distill:
        if args.keep_pct < 100.0:
            n = len(train_ds)
            k_keep = max(1, int(round(args.keep_pct * n / 100.0)))
            rng = np.random.default_rng(args.seed)
            keep_idx = rng.choice(n, size=k_keep, replace=False)
            sub_train = Subset(train_ds, keep_idx.tolist())
            train_loader = DataLoader(
                sub_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=pin,
            )
            subset_file = out_dir / "subsets" / f"keep_idx_{pct}.txt"
            np.savetxt(subset_file, np.array(keep_idx, dtype=np.int64), fmt="%d")
            print(
                f"Using random subset with keep-pct={args.keep_pct:.1f}%: "
                f"{len(sub_train)}/{n} samples | Test size: {len(test_ds)}"
            )
    else:
        train_loader = train_loader_full
        print(f"Using FULL train split: {len(train_ds)} | Test size: {len(test_ds)}")

    # model
    if args.arch == "raw":
        backbone  = RawLatentEmbedder(args.proj_dim, base_dim).to(device)
        model = PlainFromLatentBackbone(backbone, args.proj_dim, num_classes, in_ch).to(device)

    elif args.arch == "mlp":
        backbone = SDVAE_Embedder(args.proj_dim, base_dim).to(device)
        model = PlainFromLatentBackbone(backbone, args.proj_dim, num_classes, in_ch).to(device)

    elif args.arch == "conv":
        backbone = LatentConvEmbedder(proj_dim=max(args.proj_dim, 256)).to(device)
        model = PlainFromLatentBackbone(backbone, args.proj_dim, num_classes, in_ch).to(device)

    elif args.arch == "simple":
        backbone = SIMPLE_Embedder(proj_dim=args.proj_dim, channels=64).to(device)
        model = PlainFromLatentBackbone(backbone, args.proj_dim, num_classes, in_ch).to(device)

    else:
        backbone = TinyLatentEmbedder(proj_dim=args.proj_dim, channels=6).to(device)
        model = PlainFromLatentBackbone(backbone, args.proj_dim, num_classes, in_ch).to(device)

    # optim
    opt = T.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    timer = Stopwatch(); timer.start()
    best_acc = -1.0
    log_csv = out_dir / "logs" / "train_log.csv"
    best_ep = -1
    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","test_loss","test_acc","best_test_acc"])
        for ep in range(1, args.epochs+1):
            tr_l, tr_a = train_plain_epoch(model, opt, train_loader, device)
            te_l, te_a = evaluate(model, test_loader, device)
            if te_a > best_acc:
                best_acc = te_a
                best_ep = ep
                T.save({"model": model.state_dict(), "epoch": ep, "test_acc": float(best_acc)},
                       out_dir / "ckpts" / "best_plain_cnn.ckpt")
            print(f"Epoch {ep:03d} | train {tr_l:.4f}/{tr_a:.4f} | test {te_l:.4f}/{te_a:.4f} | best {best_acc:.4f}")
            w.writerow([ep, f"{tr_l:.6f}", f"{tr_a:.6f}", f"{te_l:.6f}", f"{te_a:.6f}", f"{best_acc:.6f}"])
    timer.stop()
    
    save_json({
        "best_test_acc": float(best_acc),
        "best_epoch": int(best_ep),
        "wall_sec": timer.acc
    }, out_dir / "timing.json")

    print(f"[OK] Done. Best test acc = {best_acc:.4f}")
    print(f"Logs: {log_csv}")
    print(f"Best ckpt: {out_dir/'ckpts'/'best_plain_cnn.ckpt'}")

if __name__ == "__main__":
    main()
