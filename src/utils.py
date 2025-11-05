import os, json, time, math, argparse, datetime, hashlib
from pathlib import Path
import numpy as np
import torch
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        return torch.cuda.set_device(device)
    if arg == "cpu":
        device = torch.device("cpu")
        return torch.cuda.set_device(device)

def make_run_id(args: argparse.Namespace) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        ts,
        f"dataset={args.dataset}",
        f"method={args.dd_method}" if args.run_distill else "method=baseline",
    ]
    if args.run_name:
        parts.insert(1, args.run_name)
    return "_".join(parts)

def prepare_run_dir(run_id: str) -> Path:
    out = Path("runs") / run_id
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "subsets").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out

def make_plain_id(args: argparse.Namespace) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name, extension = os.path.splitext(args.subset_file)
    if extension.lower() == '.txt':
        num = name[-2:]
    else:
        return "ERROR in subset filename: Not a .txt file or format is incorrect"
    parts = [
        ts,
        f"dataset={args.dataset}",
        f"DD={num}",
    ]
    if args.run_name:
        parts.insert(1, args.run_name)
    return "_".join(parts)

def prepare_plain_dir(run_id: str) -> Path:
    out = Path("runs") / run_id
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out

def save_json(d: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

class Stopwatch:
    def __init__(self): self.t0=None; self.acc=0.0
    def start(self): self.t0=time.time()
    def stop(self): self.acc += max(0.0, time.time()-self.t0); self.t0=None

class UpdateCounter:
    def __init__(self): self.steps=0
    def add(self, n): self.steps += int(n)

def epochs_for_fraction(base_epochs, keep_frac, mode="scaled"):
    if mode == "fixed":
        return max(1, int(base_epochs))
    return max(1, int(np.ceil(base_epochs / max(keep_frac, 1e-8))))
