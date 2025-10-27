#!/usr/bin/env python
import argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchinfo import summary
from .utils import set_seed, resolve_device, make_run_id, prepare_run_dir, save_json
from .models import load_vae, encode_to_latent, LatentConvEmbedder, SIMPLE_Embedder,TinyLatentEmbedder
from .data_custom import PNGBinaryDataset, build_transforms


def train_epoch_plain(vae, embedder, clf, opt, loader, device, vae_dtype):
    embedder.train(); clf.train()
    total = correct = 0; loss_sum = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        with torch.no_grad():
            z = encode_to_latent(vae, xb, device, vae_dtype)
        feat = embedder(z)
        logits = clf(feat)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch_plain(vae, embedder, clf, loader, device, vae_dtype):
    embedder.eval(); clf.eval()
    total = correct = 0; loss_sum = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        z = encode_to_latent(vae, xb, device, vae_dtype)
        feat = embedder(z)
        logits = clf(feat)
        loss = F.cross_entropy(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total


def select_uniform_by_n(df_train, k_total=800, nbins=20, seed=42):

    rng = np.random.default_rng(seed)
    n_vals = pd.to_numeric(df_train["n"], errors="coerce")
    mask = n_vals.notna()
    df_train = df_train.loc[mask].copy()
    n_vals = n_vals.loc[mask].to_numpy()

    bins = np.linspace(n_vals.min(), n_vals.max(), nbins+1)
    cats = pd.cut(n_vals, bins=bins, include_lowest=True, right=False)

    per_bin = max(1, int(np.floor(k_total / nbins)))
    chosen = []
    for cat in cats.unique().sort_values():
        idx_bin = np.where(cats == cat)[0]
        if len(idx_bin) == 0:
            continue
        take = min(per_bin, len(idx_bin))
        pick = rng.choice(idx_bin, size=take, replace=False)
        chosen.extend(pick.tolist())

    if len(chosen) < k_total:
        remaining = np.setdiff1d(np.arange(len(df_train)), np.array(chosen, dtype=int))
        need = min(k_total - len(chosen), len(remaining))
        if need > 0:
            extra = rng.choice(remaining, size=need, replace=False)
            chosen.extend(extra.tolist())

    chosen_idx = df_train.index.to_numpy()[np.array(chosen, dtype=int)]
    return np.array(chosen_idx, dtype=int)


def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--img-root", type=str, required=True, help="Folder με PNGs")
    ap.add_argument("--labels-csv", type=str, required=True, help="CSV με fname,label,n,...")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-split", type=float, default=0.2)
    # selection
    ap.add_argument("--k-total", type=int, default=800, help="σύνολο δειγμάτων για training")
    ap.add_argument("--nbins", type=int, default=20, help="bins για ομοιόμορφη κάλυψη του n")
    # model
    ap.add_argument("--vae", type=str, default="sd15", choices=["sd15","taesd"])
    ap.add_argument("--sd15-path", type=str, default="./_sd15_vae")
    ap.add_argument("--taesd-path", type=str, default="./_taesd")
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # system
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_id = f"naiveuniform_k={args.k_total}_bins={args.nbins}"
    out_dir = prepare_run_dir(run_id)
    save_json(vars(args), out_dir / "config.json")


    df = pd.read_csv(args.labels_csv)

    def norm_label(v):
        if isinstance(v,str):
            s=v.strip().lower()
            if s=="true": return 1
            if s=="false": return 0
        return int(v)
    labels = df["label"].apply(norm_label).astype(int).to_numpy()

    idx_all = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx_all, test_size=args.val_split, random_state=args.seed, stratify=labels)


    df_train = df.iloc[idx_tr].copy()
    chosen_global_idx = select_uniform_by_n(df_train, k_total=args.k_total, nbins=args.nbins, seed=args.seed)

    pos_in_train = {g:i for i,g in enumerate(idx_tr)}
    chosen_train_positions = [pos_in_train[int(g)] for g in chosen_global_idx if int(g) in pos_in_train]
    chosen_train_positions = np.array(chosen_train_positions, dtype=int)

    (out_dir / "subsets").mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "subsets" / f"naive_uniform_keep_idx_{args.k_total}.txt",
               chosen_global_idx, fmt="%d")


    train_tf, test_tf = build_transforms(args.image_size)
    base_ds = PNGBinaryDataset(args.img_root, args.labels_csv, transform=None)

    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, base, indices, tf):
            self.base=base; self.indices=indices; self.tf=tf
        def __len__(self): return len(self.indices)
        def __getitem__(self, k):
            i = self.indices[k]
            p, y = self.base.items[i]
            img = (Path(p)).open("rb")
            from PIL import Image; im = Image.open(img).convert("RGB")
            return self.tf(im), torch.tensor(y, dtype=torch.long)


    train_ds = _Wrap(base_ds, idx_tr[chosen_train_positions], train_tf)
    test_ds  = _Wrap(base_ds, idx_te, test_tf)

    pin = (device.type=="cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    print(f"Naive-uniform subset size: {len(train_ds)} | Test size: {len(test_ds)}")

    # VAE (frozen)
    vae_dtype = torch.float16 if (device.type=="cuda" and args.vae=="sd15") else torch.float32
    repo = args.sd15_path if args.vae=="sd15" else args.taesd_path
    vae = load_vae(args.vae, repo, device, vae_dtype)

    # Conv embedder + plain linear classifier
    embedder = TinyLatentEmbedder(proj_dim=args.proj_dim).to(device)
    summary(embedder, input_size=(1, 4, 8, 8))
    clf = nn.Linear(args.proj_dim, 2).to(device)

    opt = torch.optim.AdamW(list(embedder.parameters())+list(clf.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    # train loop
    best_te = -1.0; best_ep = -1
    ckpt_dir = out_dir / "ckpts"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","test_loss","test_acc"])
        for e in range(1, args.epochs+1):
            tr_loss, tr_acc = train_epoch_plain(vae, embedder, clf, opt, train_loader, device, vae_dtype)
            te_loss, te_acc = eval_epoch_plain(vae, embedder, clf, test_loader, device, vae_dtype)
            w.writerow([e, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{te_loss:.6f}", f"{te_acc:.6f}"])
            print(f"Epoch {e:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | test {te_loss:.4f}/{te_acc:.4f}")
            if te_acc > best_te:
                best_te, best_ep = te_acc, e
                torch.save({
                    "embedder": embedder.state_dict(),
                    "clf": clf.state_dict(),
                    "epoch": e,
                    "test_acc": te_acc
                }, ckpt_dir / "best.ckpt")
    print(f"BEST test acc: {best_te:.4f} (epoch {best_ep})")
    print(f"[OK] Saved: ckpts/best.ckpt, train_log.csv and subsets/naive_uniform_keep_idx_{args.k_total}.txt in {out_dir}")

if __name__ == "__main__":
    main()
