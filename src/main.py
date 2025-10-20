#!/usr/bin/env python
import csv
import numpy as np
import argparse, json
from pathlib import Path
import torch as T
from torch.utils.data import DataLoader
from .models import load_vae
from .utils import set_seed, resolve_device, make_run_id, prepare_run_dir, save_json, Stopwatch, UpdateCounter, epochs_for_fraction
from .data import get_loaders
from .models import load_vae, SDVAE_Embedder, HASeparator, LatentConvEmbedder
from .train import train_epoch, eval_epoch
from .metrics import plot_tsne, embedding_metrics
from .distill import (
    collect_embed_and_logits, collect_vae_features,
    kmeans_assign_dist_whiten, make_distilled_indices_balanced,
    select_kcenter_cosine_balanced, select_kcenter_cosine_global,
    select_margin_mix_balanced
)
from .metrics import plot_tsne, embedding_metrics
from .train import eval_epoch
from .models import encode_to_latent
from .distill import collect_embed_and_logits, collect_vae_features

def parse_args():
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist","cifar10"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--vae", type=str, default="sd15", choices=["sd15","taesd"])
    p.add_argument("--sd15-path", type=str, default="./_sd15_vae")
    p.add_argument("--embedder", type=str, default="mlp", choices=["mlp","conv"],
                   help="mlp = SDVAE_Embedder (flatten+MLP), conv = LatentConvEmbedder (ResNet-style)")

    p.add_argument("--taesd-path", type=str, default="./_taesd")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # viz
    p.add_argument("--max-tsne-train", type=int, default=5000)
    p.add_argument("--max-tsne-test", type=int, default=5000)
    # distill
    p.add_argument("--run-distill", action="store_true")
    p.add_argument("--dd-method", type=str, default="kcenter_cosine",
                   choices=["kmeans_euclid","kcenter_cosine","margin_mix"])
    p.add_argument("--dd-criterion", type=str, default="nearest",
                   choices=["nearest","furthest","random"])
    p.add_argument("--dd-k", type=int, default=10)
    p.add_argument("--epochs-per-dd", type=int, default=10)
    # if you want fixed epoch training for all DD set the following to "fixed"
    p.add_argument("--dd-epoch-mode", type=str, default="scaled", choices=["scaled","fixed"])
    p.add_argument("--hard-fraction", type=float, default=0.5)
    p.add_argument("--dd-supervision", type=str, default="groundtruth",
                   choices=["groundtruth","pseudo","unsupervised"])
    # reproducibility / names
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--load-ckpt", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # prepare run dir
    run_id = make_run_id(args)
    out_dir = prepare_run_dir(run_id)
    save_json(vars(args), out_dir / "config.json")

    # data
    train_ds, test_ds, train_loader, test_loader, train_loader_noshuf = get_loaders(
        args.dataset, args.image_size, args.batch_size, args.num_workers, device
    )

    # vae
    vae_dtype = T.float16 if (device.type=="cuda" and args.vae=="sd15") else T.float32
    repo = args.sd15_path if args.vae=="sd15" else args.taesd_path

    vae = load_vae(args.vae, repo, device, vae_dtype)

    # model
    LATENT_HW = args.image_size // 8
    base_dim = 4 * LATENT_HW * LATENT_HW
    eff_proj_dim = args.proj_dim if args.embedder == "mlp" else max(args.proj_dim, 256)

    if args.embedder == "conv":
        embedder = LatentConvEmbedder(proj_dim=eff_proj_dim).to(device)
    else:
        embedder = SDVAE_Embedder(eff_proj_dim, base_dim).to(device)

    head = HASeparator(input_dim=args.proj_dim, num_classes=10, margin=0.4, scale=30.0).to(device)
    opt = T.optim.AdamW(list(embedder.parameters())+list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # optional preload
    if args.load_ckpt:
        ckpt = T.load(args.load_ckpt, map_location=device)
        embedder.load_state_dict(ckpt["embedder"])
        head.load_state_dict(ckpt["head"])
        print(f"Loaded checkpoint: {args.load_ckpt}")

    # baseline training
    best_acc=-1.0; best_state=None
    timer = Stopwatch(); timer.start()
    updates = UpdateCounter()
    if not args.skip_train:
        for e in range(1, args.epochs+1):
            tr_loss, tr_acc = train_epoch(vae, embedder, head, opt, train_loader, device, vae_dtype, counter=updates)
            te_loss, te_acc = eval_epoch(vae, embedder, head, test_loader, device, vae_dtype)
            print(f"Epoch {e:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")
            if te_acc > best_acc:
                best_acc = te_acc; best_state=(embedder.state_dict(), head.state_dict(), e, te_acc)
                T.save({"embedder":best_state[0],"head":best_state[1],"epoch":e,"test_acc":te_acc}, out_dir / "ckpts" / "best.ckpt")
    else:
        print("Skipping baseline training (using loaded weights).")
    timer.stop()

    # reload best
    if best_state is not None:
        embedder.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        print(f"Reloaded BEST baseline weights (epoch {best_state[2]}, test_acc={best_state[3]:.4f})")
    else:
        # if loaded from file
        pass

    # embeddings + metrics + plots
    @T.no_grad()
    def collect(ldr, maxn):
        E=[]; L=[];
        embedder.eval(); head.eval()
        seen=0
        for xb,yb in ldr:
            xb=xb.to(device)
            z=encode_to_latent(vae, xb, device, vae_dtype)
            feat=embedder(z)
            _, emb, _ = head(feat, labels=None)
            E.append(emb.cpu()); L.append(yb)
            seen += yb.size(0)
            if maxn and seen>=maxn: break
        return T.cat(E).numpy(), T.cat(L).numpy()

    E_tr, L_tr = collect(train_loader, args.max_tsne_train)
    E_te, L_te = collect(test_loader,  args.max_tsne_test)
    plot_tsne(E_tr, L_tr, out_dir / "plots" / "tsne_train.png", "t-SNE Train")
    plot_tsne(E_te, L_te, out_dir / "plots" / "tsne_test.png", "t-SNE Test")
    sil, intra, margin = embedding_metrics(E_te, L_te)
    print("silhouette (cosine):", sil, "intra (1-cos):", intra, "inter-margin:", margin)

    # timing + updates dump
    timing = {
        "baseline": {
            "wall_sec": timer.acc,
            "updates": updates.steps,
            "best_test_acc": float(best_acc)
        }
    }
    save_json(timing, out_dir / "timing.json")

    # distillation sweep
    if args.run_distill:
        # collect cosine features + logits (for kcenter/margin_mix)
        
        E_all=L_all=IDX_all=LOG_all=None
        if args.dd_method in ["kcenter_cosine","margin_mix"]:
            E_all, Y_all, IDX_all, LOG_all = collect_embed_and_logits(
                vae, embedder, head, train_loader_noshuf, device, vae_dtype
            )

        # prepare CSV
        dd_csv = out_dir / "dd_curve.csv"
        with open(dd_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pct","method","supervision","kept","epochs","best_epoch","best_test_acc","wall_sec","updates"])

            schedule = [10,20,30,40,50,60,70,80,90]
            for pct in schedule:
                # choose keep_idx
                sup = args.dd_supervision
                if args.dd_method == "kmeans_euclid":
                    throw_embed = SDVAE_Embedder(args.proj_dim, base_dim).to(device)
                    F_all, YY, IDX = collect_vae_features(vae, throw_embed, train_loader_noshuf, device, vae_dtype)
                    _, d2 = kmeans_assign_dist_whiten(F_all, k=args.dd_k, seed=args.seed)
                    keep_idx = make_distilled_indices_balanced(IDX, d2, YY, pct, args.dd_criterion, seed=args.seed, num_classes=10)
                elif args.dd_method == "kcenter_cosine":
                    if sup == "unsupervised":
                        keep_idx = select_kcenter_cosine_global(E_all, IDX_all, pct, seed=args.seed)
                    else:
                        Y_used = Y_all if sup=="groundtruth" else LOG_all.argmax(axis=1)
                        keep_idx = select_kcenter_cosine_balanced(E_all, Y_used, IDX_all, pct, num_classes=10, seed=args.seed)
                else:  # margin_mix
                    Y_used = Y_all if sup=="groundtruth" else LOG_all.argmax(axis=1)
                    keep_idx = select_margin_mix_balanced(E_all, Y_used, IDX_all, LOG_all, pct,
                                                          hard_fraction=args.hard_fraction, num_classes=10, seed=args.seed)

                # save subset indices
                subset_file = out_dir / "subsets" / f"keep_idx_{pct}.txt"
                np.savetxt(subset_file, np.array(keep_idx, dtype=np.int64), fmt="%d")

                if args.embedder == "conv":
                    model = LatentConvEmbedder(proj_dim=eff_proj_dim).to(device)
                else:
                    model = SDVAE_Embedder(eff_proj_dim, base_dim).to(device)
                head2 = HASeparator(input_dim=args.proj_dim, num_classes=10, margin=0.4, scale=28.0).to(device)
                eff_lr = args.lr if (len(keep_idx)/len(train_ds))>0.3 else args.lr*0.7
                opt2 = T.optim.AdamW(list(model.parameters())+list(head2.parameters()), lr=eff_lr, weight_decay=args.weight_decay)

                sub_ds = T.utils.data.Subset(train_ds, keep_idx)
                sub_loader = T.utils.data.DataLoader(sub_ds, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

                # epochs choice
                keep_frac = len(keep_idx)/len(train_ds)
                dd_epochs = epochs_for_fraction(args.epochs_per_dd, keep_frac, mode=args.dd_epoch_mode)

                # train/eval with timing + updates
                t = Stopwatch(); t.start()
                uc = UpdateCounter()
                best_te=-1.0; best_ep=-1; best_state=None
                for ep in range(1, dd_epochs+1):
                    tl, ta = train_epoch(vae, model, head2, opt2, sub_loader, device, vae_dtype, counter=uc)
                    _, te = eval_epoch(vae, model, head2, test_loader, device, vae_dtype)
                    if te > best_te:
                        best_te, best_ep = te, ep
                        best_state = (model.state_dict(), head2.state_dict())
                t.stop()
                if best_state:
                    T.save({"embedder":best_state[0],"head":best_state[1],"epoch":best_ep,"test_acc":best_te},
                               out_dir / "ckpts" / f"dd_{pct}_best.ckpt")

                print(f"[DD/{args.dd_method}:{sup}] pct={pct}% kept={len(keep_idx)} epochs={dd_epochs} | BEST {best_te:.4f} (ep {best_ep})")
                w.writerow([pct, args.dd_method, sup, len(keep_idx), dd_epochs, best_ep, f"{best_te:.6f}", f"{t.acc:.3f}", uc.steps])

if __name__ == "__main__":
    main()
