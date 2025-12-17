#!/usr/bin/env python
import argparse, time, csv
from pathlib import Path
import numpy as np
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision as tv
import random, torch
from .utils import save_json
from .model_baselines import create_model
from os import listdir
from os.path import isfile, join

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10","mnist"], default="cifar10")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--subset-folder", type=str, required=True, help="runs/.../subsets/")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--pretrained", action="store_true", help="Start from ImageNet pretrained weights")
    p.add_argument("--amp", action="store_true", help="Use mixed precision")
    p.add_argument("--arch", type=str, required=True, default="resnet50",help="Choose baseline (see model_baselines.py)")
    return p.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)



def get_loaders(dataset, data_root, subset_path, batch_size, num_workers):
    if subset_path and Path(subset_path).exists():
        subset_idx = np.loadtxt(subset_path, dtype=np.int64).tolist()
    else:
        subset_idx = None 

    if dataset == "cifar10":
        normalize = tv.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            normalize])
        test_tf = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            normalize])
        train_full = tv.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_set   = tv.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
        num_classes = 10

    else:  # MNIST
        normalize = tv.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        to3 = lambda x: x.expand(3, *x.shape[1:])
        train_tf = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(to3),
            normalize])
        test_tf = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(to3),
            normalize])

        train_full = tv.datasets.MNIST(root=data_root, train=True, download=True, transform=train_tf)
        test_set   = tv.datasets.MNIST(root=data_root, train=False, download=True, transform=test_tf)
        num_classes = 10
    
    if subset_idx is not None:
        train_set = Subset(train_full, subset_idx)
    else:
        train_set = train_full
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes, len(train_set), len(test_set)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) 
    print(f"Current default CUDA device: {torch.cuda.current_device()}")
    coreset_size = 0
    coreset_file = [f for f in listdir(args.subset_folder) if isfile(join(args.subset_folder,f))]
    for c_file in coreset_file:
        match = re.search(r"idx_(\d+)\.txt", c_file)
        full_subset_path = os.path.join(args.subset_folder, c_file)
        if match:
            coreset_size = match.group(1) 
            print(f"Detected Coreset size: {coreset_size} from file: {c_file}")
        else:
            continue
        train_loader, test_loader, num_classes, n_train, n_test = get_loaders(args.dataset, args.data_root, full_subset_path, args.batch_size, args.num_workers)
        print(f"Subset size: {n_train} | Test size: {n_test}")
        subset_path = Path(args.subset_folder)
        exp_root = subset_path.parents[1]
        model, tag = create_model(args.arch, num_classes=num_classes, pretrained=args.pretrained)
        counter = 1

        baseline_root = exp_root / "baselines" / tag
        if os.path.exists(baseline_root):
            baseline_root = exp_root/ "baselines"/ f"{tag}_{counter:02d}"
            counter +=1
        default_dir = baseline_root

        out_dir = Path(args.out) if args.out else default_dir

        (out_dir/"ckpts").mkdir(parents=True, exist_ok=True)
        (out_dir/"logs").mkdir(parents=True, exist_ok=True)
        save_json(vars(args), out_dir / "logs" / f"config_coreset_{coreset_size}.json")

        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.1)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    

        scaler = torch.amp.GradScaler(enabled=args.amp)

        csv_path = out_dir/"logs"/f"metrics_{coreset_size}.csv"
        with open(csv_path, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["epoch", "train_loss","train_acc","test_loss","test_acc","lr"])
        
        start_time = time.time()
        best_acc = 0.0
        best_epoch = -1
        best_ckpt = out_dir/"ckpts"/f"best_{args.arch}_{coreset_size}.ckpt"

        for ep in range(1, args.epochs+1):
            model.train()
            tot, correct, loss_sum = 0, 0, 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                tot += yb.size(0)
            train_loss = loss_sum / tot
            train_acc = correct / tot

            model.eval()
            tot, correct, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    if args.amp:
                        with torch.cuda.amp.autocast():
                            logits = model(xb)
                            loss = criterion(logits, yb)
                    else:
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    loss_sum += loss.item() * yb.size(0)
                    correct += (logits.argmax(1) == yb).sum().item()
                    tot += yb.size(0)
            test_loss = loss_sum / tot
            test_acc = correct / tot

            sched.step()
            curr_lr = sched.get_last_lr()[0]

            with open(csv_path, "a", newline="") as fcsv:
                w = csv.writer(fcsv)
                w.writerow([ep, f"{train_loss:.6f}", f"{train_acc:.6f}",
                               f"{test_loss:.6f}", f"{test_acc:.6f}", f"{curr_lr:.8f}"])

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = ep
                torch.save({"model": model.state_dict(), "epoch": ep, "test_acc": best_acc},
                           best_ckpt)
                print(f"Epoch {ep:03d} | train {train_loss:.4f}/{train_acc:.4f} | "
                      f"test {test_loss:.4f}/{test_acc:.4f}  <-- NEW BEST (acc {best_acc:.4f})")
            else:
                print(f"Epoch {ep:03d} | train {train_loss:.4f}/{train_acc:.4f} | "
                      f"test {test_loss:.4f}/{test_acc:.4f} | best {best_acc:.4f} (ep {best_epoch})")

            wall_sec = time.time() - start_time

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Save JSON
            params_json_path = out_dir / "logs" / f"params_baseline_{coreset_size}.json"
            params_json_path.parent.mkdir(parents=True, exist_ok=True)
            save_json({
                    "arch": args.arch,
                    "dataset" : args.dataset,
                    "coreset": coreset_size,
                    "total_epochs": args.epochs,
                    "best_test_acc": float(best_acc),
                    "best_epoch": int(best_epoch),
                    "wall_sec": wall_sec,
                    "total_params": total_params,
                    "trainable_params": trainable_params
                }, params_json_path)

            print(f"Best test acc: {best_acc:.4f} at epoch {best_epoch} | ckpt: {best_ckpt}")
            print(f"Metrics CSV saved to: {csv_path}")
            print(f"Params JSON saved to: {params_json_path}")
            print(f"Training took {wall_sec:.2f} seconds")
if __name__ == "__main__":
    main()
