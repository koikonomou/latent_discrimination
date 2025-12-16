#!/usr/bin/env python
import csv
import numpy as np
import argparse, json
from pathlib import Path
import torch as T
from torchinfo import summary
from torch.utils.data import DataLoader
from .utils import set_seed, resolve_device, make_run_id, prepare_run_dir, save_json, Stopwatch, UpdateCounter, epochs_for_fraction, collect_embeddings, collect
from .data import get_loaders
from .models import load_vae, SDVAE_Embedder, HASeparator, LatentConvEmbedder, SIMPLE_Embedder, TinyLatentEmbedder, ResNet18LatentEmbedder
from .train import train_epoch, eval_epoch
from .metrics import plot_tsne, embedding_metrics
from .distill import *

from .metrics import plot_tsne, embedding_metrics
from .train import eval_epoch
from .models import encode_to_latent
from .data_custom import get_custom_loaders


def parse_args():
    p = argparse.ArgumentParser()
    # basic
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist","fashion_mnist","cifar10","cifar100","tinyimagenet","custom"])
    # 28 for mnist, 32 for cifar10
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # pretrained vae model
    p.add_argument("--vae", type=str, default="sd15", choices=["sd15","taesd"])
    p.add_argument("--sd15-path", type=str, default="./_sd15_vae")

    p.add_argument("--taesd-path", type=str, default="./_taesd")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # viz
    p.add_argument("--max-tsne-train", type=int, default=5000)
    p.add_argument("--max-tsne-test", type=int, default=5000)
    
    # distill
    p.add_argument("--run-distill", action="store_true")
    p.add_argument("--dd-method", type=str, default="kcenter_cosine", choices=["kmeans_euclid","kcenter_cosine","margin_mix","select_random"])
    p.add_argument("--dd-criterion", type=str, default="nearest", choices=["nearest","furthest","random"])

    p.add_argument("--dd-k", type=int, default=10)
    p.add_argument("--epochs-per-dd", type=int, default=10)
    p.add_argument("--dd-epoch-mode", type=str, default="scaled", choices=["scaled","fixed"])
    p.add_argument("--hard-fraction", type=float, default=0.5)
    p.add_argument("--dd-supervision", type=str, default="groundtruth", choices=["groundtruth","pseudo","unsupervised"], help="unsupervised is for class agnostic approach ") 
    p.add_argument("--oreset-value", type=str, default="percentage", choices=["percentage","absolute_value"])
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--train-coreset", action="store_true", default=False, help="If set, enables coreset selection training")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--load-ckpt", type=str, default="")
    p.add_argument("--embedder", type=str, default="conv", choices=["conv","mlp","raw","simple", "tiny","resnet"],help="mlp = SDVAE_Embedder , conv = LatentConvEmbedder, raw = Latent space")
    # synthetic dataset
    p.add_argument("--img-root", type=str, default="", help="Root folder with PNGs (for --dataset custom)")
    p.add_argument("--labels-csv", type=str, default="", help="CSV with filename,label (for --dataset custom)")
    p.add_argument("--val-split", type=float, default=0.2, help="Holdout fraction for custom dataset")
  
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) 
    print(f"Current default CUDA device: {torch.cuda.current_device()}")

    run_id = make_run_id(args)
    out_dir = prepare_run_dir(run_id)
    save_json(vars(args), out_dir / "logs" / "config_baseline.json")


    if args.dataset == "custom":
        if not args.img_root or not args.labels_csv:
            raise SystemExit("--dataset custom requires --img-root and --labels-csv")
        (train_ds, test_ds, train_loader, test_loader, train_loader_noshuf, num_classes) = get_custom_loaders(args.img_root, args.labels_csv, args.image_size, args.batch_size, args.num_workers, device, test_size=args.val_split, seed=args.seed)
    else:
        # existing path for mnist/cifar10
        train_ds, test_ds, train_loader, test_loader, train_loader_noshuf = get_loaders(args.dataset, args.image_size, args.batch_size, args.num_workers, device)
    


    vae_dtype = T.float16 #if (args.device=="auto" and args.vae=="sd15") else T.float32
    repo = args.sd15_path if args.vae=="sd15" else args.taesd_path
#    #TODO: FIX THIS
    if args.vae=="sd15" :
        repo = args.sd15_path
    elif args.vae=="taesd":
        repo = args.taesd_path 
    else:
        raise ValueError("No autoencoder model detected")

    vae = load_vae(args.vae, repo, device, vae_dtype)


    LATENT_HW = args.image_size // 8
    base_dim = 4 * LATENT_HW * LATENT_HW

    if args.embedder == "raw":
        embedder = RawLatentEmbedder(args.proj_dim).to(device)
    elif args.embedder == "mlp":
        embedder = SDVAE_Embedder(args.proj_dim, base_dim).to(device)
    elif args.embedder == "simple":
        embedder = SIMPLE_Embedder(proj_dim=args.proj_dim, channels=64).to(device)
    elif args.embedder == "tiny":
        embedder = TinyLatentEmbedder(proj_dim=args.proj_dim).to(device)
    elif args.embedder == "resnet":
        embedder = ResNet18LatentEmbedder(proj_dim=args.proj_dim, pretrained=False).to(device)
    else:
        embedder = LatentConvEmbedder(proj_dim=max(args.proj_dim, 128)).to(device)

    summary(embedder, input_size=(1, 4, 8, 8))

    head = HASeparator(input_dim=args.proj_dim, num_classes=args.num_classes, margin=0.1, scale=5.0).to(device)
    opt = T.optim.AdamW(list(embedder.parameters())+list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in embedder.parameters())
    trainable_params = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
                
    if args.load_ckpt:
        ckpt = T.load(args.load_ckpt, map_location=device)
        embedder.load_state_dict(ckpt["embedder"])
        head.load_state_dict(ckpt["head"])
        print(f"Loaded checkpoint: {args.load_ckpt}")

    # baseline training
    best_acc=-1.0;  best_ep = -1; best_state=None
    timer = Stopwatch(); timer.start()
    updates = UpdateCounter()
    log_csv = out_dir / "logs" / "baseline_train_log.csv"
    if not args.skip_train:
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","test_loss","test_acc","best_test_acc"])
            for e in range(1, args.epochs+1):
                tr_loss, tr_acc = train_epoch(vae, embedder, head, opt, train_loader, device, vae_dtype, counter=updates)
                te_loss, te_acc = eval_epoch(vae, embedder, head, test_loader, device, vae_dtype)
                print(f"Epoch {e:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")
                if te_acc > best_acc:
                    best_ep = e
                    best_acc = te_acc; best_state=(embedder.state_dict(), head.state_dict(), e, te_acc)
                    T.save({"embedder":best_state[0],"head":best_state[1],"epoch":e,"test_acc":te_acc}, out_dir / "ckpts" / "best.ckpt")
                w.writerow([e, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{te_loss:.6f}", f"{te_acc:.6f}", f"{best_acc:.6f}"])
    else:
        print("Skipping baseline training (using loaded weights).")
    timer.stop()

    save_json({
        "best_test_acc": float(best_acc),
        "best_epoch": int(best_ep),
        "wall_sec": timer.acc,
        "total_params": total_params,
        "trainable_params": trainable_params
    }, out_dir / "logs" / "params_baseline.json")

    if best_state is not None:
        embedder.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        print(f"Reloaded BEST baseline weights (epoch {best_state[2]}, test_acc={best_state[3]:.4f})")
    else:
        pass


    E_tr, L_tr = collect(train_loader, vae, args.max_tsne_train, embedder, head, device=device, vae_dtype=vae_dtype)
    E_t, L_te = collect(test_loader, vae, args.max_tsne_test, embedder, head, device=device, vae_dtype=vae_dtype)
    plot_tsne(E_tr, L_tr, out_dir / "plots" / "tsne_train.png", "t-SNE Train")
    plot_tsne(E_te, L_te, out_dir / "plots" / "tsne_test.png", "t-SNE Test")
    sil, intra, margin = embedding_metrics(E_te, L_te)
    print("silhouette (cosine):", sil, "intra (1-cos):", intra, "inter-margin:", margin)


    # distillation
    if args.run_distill:
        # collect cosine features + logits (for kcenter/margin_mix)
        
        E_all=L_all=IDX_all=LOG_all=None
        E_all, Y_all, IDX_all, LOG_all = collect_embed_and_logits(vae, embedder, head, train_loader_noshuf, device, vae_dtype)


        # schedule = [10,20,30,40,50,60,70,80,90]
        percentage = [0.1,0.5,1.0,5.0,10,20,30,40,50,60,70,80,90]
        absolute_values= [10,50]

        if args.coreset_value == "percentage":
            schedule_list = percentages 
            is_prercentage = True
        else:
            scedule_list = absolute_values
            is_percentage = False
        print(f"Running schedule with:{scedule_list}")
        for pct in schedule_list: 
            if args.dd_method == "kcenter_cosine":
                keep_idx = select_kcenter_cosine_balanced(E_all, Y_all, IDX_all, pct, percentage=args.percentage, num_classes=args.num_classes, seed=args.seed)
            elif args.dd_method == "select_random":
                keep_idx = select_random(E_all, Y_all, IDX_all, pct, percentage=args.percentage, num_classes=args.num_classes,seed=args.seed)

            subset_file = out_dir / "subsets" / f"keep_idx_{pct}.txt"
            np.savetxt(subset_file, np.array(keep_idx, dtype=np.int64), fmt="%d")
            if args.train_coreset:
                print("Coreset training enabled!")
                if args.embedder == "conv":
                    model = LatentConvEmbedder(proj_dim=max(args.proj_dim, 256)).to(device)
                    dd_proj_dim = max(args.proj_dim, 256)
                elif args.embedder == "raw":
                    model = SDVAE_Embedder(256, base_dim).to(device)
                    dd_proj_dim = 256
                elif args.embedder == "resnet":
                    model = ResNet18LatentEmbedder(proj_dim=args.proj_dim, pretrained=False).to(device)
                    dd_proj_dim = args.proj_dim
                else:
                    model = SDVAE_Embedder(args.proj_dim, base_dim).to(device)
                    dd_proj_dim = args.proj_dim

                head2 = HASeparator(input_dim=dd_proj_dim, num_classes=args.num_classes, margin=0.4, scale=28.0).to(device)
                eff_lr = args.lr if (len(keep_idx)/len(train_ds))>0.3 else args.lr*0.7
                opt2 = T.optim.AdamW(list(model.parameters())+list(head2.parameters()), lr=eff_lr, weight_decay=args.weight_decay)

                sub_ds = T.utils.data.Subset(train_ds, keep_idx)
                sub_loader = T.utils.data.DataLoader(sub_ds, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_workers, pin_memory=(device))

                keep_frac = len(keep_idx)/len(train_ds)
                dd_epochs = epochs_for_fraction(args.epochs_per_dd, keep_frac, mode=args.dd_epoch_mode)

                t = Stopwatch(); t.start()
                uc = UpdateCounter()
                log_csv = out_dir / "logs" / f"train_{pct}_log.csv"
                best_te=-1.0; best_ep=-1; best_state=None
                with open(log_csv, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","test_loss","test_acc","best_test_acc"])
                    for ep in range(1, dd_epochs+1):
                        tr_l, tr_a = train_epoch(vae, model, head2, opt2, sub_loader, device, vae_dtype, counter=uc)
                        te_l, te_a = eval_epoch(vae, model, head2, test_loader, device, vae_dtype)
                        if te_a > best_te:
                            best_te, best_ep = te_a, ep
                            best_state = (model.state_dict(), head2.state_dict())
                        w.writerow([ep, f"{tr_l:.6f}", f"{tr_a:.6f}", f"{te_l:.6f}", f"{te_a:.6f}", f"{best_te:.6f}"])

                t.stop()
                save_json({
                    "coreset": pct,
                    "dataset": args.dataset,
                    "best_test_acc": float(best_te),
                    "best_epoch": int(best_ep),
                    "wall_sec": t.acc
                }, out_dir / "logs" / f"results_{pct}.json")
                save_json(vars(args), out_dir / "logs" / f"config_{pct}.json")
                if best_state:
                    T.save({"embedder":best_state[0],"head":best_state[1],"epoch":best_ep,"test_acc":best_te},
                               out_dir / "ckpts" / f"dd_{pct}_best.ckpt")
                if best_state is not None:
                    # load best weights
                    model.load_state_dict(best_state[0])
                    head2.load_state_dict(best_state[1])
                    model.eval(); head2.eval()

                    E_tr_dd, L_tr_dd = collect_embeddings(vae, model, head2, train_loader, device, vae_dtype, args.max_tsne_train)
                    E_te_dd, L_te_dd = collect_embeddings(vae, model, head2, test_loader, device, vae_dtype, args.max_tsne_test)

                    plot_tsne(E_tr_dd, L_tr_dd,out_dir / "plots" / f"tsne_train_dd_{pct}.png", f"t-SNE Train DD {pct}%")
                    plot_tsne(E_te_dd, L_te_dd,out_dir / "plots" / f"tsne_test_dd_{pct}.png", f"t-SNE Test DD {pct}%")

                print(f"[DD/{args.dd_method}:{sup}] pct={pct}% kept={len(keep_idx)} epochs={dd_epochs} | BEST {best_te:.4f} (ep {best_ep})")
                summary_csv  = out_dir / "logs" / f"summary_{pct}.csv"
                with open(summary_csv, "w", newline="") as f_summary:
                    w_summary = csv.writer(f_summary)
                    w_summary.writerow(["pct", "dd_method", "sup", "kept", "epochs", "best_ep", "best_test_acc", "wall_sec", "steps"])
                    w_summary.writerow([pct, args.dd_method, sup, len(keep_idx), dd_epochs, best_ep, f"{best_te:.6f}", f"{t.acc:.3f}", uc.steps])

if __name__ == "__main__":
    main()
