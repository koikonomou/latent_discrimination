import os, re, json, time, math, argparse, datetime, hashlib
from pathlib import Path
import numpy as np
import torch as T
import random
from .models import encode_to_latent

@T.no_grad()
def collect_embeddings(vae, embedder, head, loader, device="cuda:0", vae_dtype=T.float16, maxn=None):
    E, L = [], []
    embedder.eval(); head.eval()
    seen = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        z = encode_to_latent(vae, xb, device, vae_dtype)   # (B,4,8,8)
        feat = embedder(z)                                 # (B, proj_dim)
        _, emb, _ = head(feat, labels=None)                # normalized features
        E.append(emb.cpu()); L.append(yb.cpu())
        seen += yb.size(0)
        if maxn and seen >= maxn:
            break
    return T.cat(E).numpy(), T.cat(L).numpy()

@T.no_grad()
def collect(ldr, vae, maxn, embedder, head, device="cuda:0", vae_dtype=T.float16):
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    
def resolve_device(arg: str) -> T.device:
    if arg == "auto":
        device = T.device("cuda:2" if T.cuda.is_available() else "cpu")
    elif arg == "cpu":
        device = T.device("cpu")
    else:
        raise ValueError(f"Unknown device: {arg}")
    
    if device.type == "cuda":
        T.cuda.set_device(device)  # optional, sets default CUDA device
    return device


def make_run_id(args: argparse.Namespace) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d")
    parts = [
        ts,
        f"dataset_{args.dataset}",
        f"method_{args.dd_method}" if args.run_distill else "method_baseline",
    ]
    if args.run_name:
        parts.insert(1, args.run_name)
    base_id = "_".join(parts)
    run_id = base_id
    counter = 1
    run_dir = Path("runs")
    run_dir.mkdir(parents=True,exist_ok=True)

    while run_dir/run_id in run_dir.iterdir():
        run_id =f"{base_id}_{counter:02d}"
        counter+=1
    return run_id

def prepare_run_dir(run_id: str) -> Path:
    out = Path("runs") / run_id
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "subsets").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out



def make_plain_id(args: argparse.Namespace) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [ts]

    if args.run_name:
        parts.append(args.run_name)

    parts.append(f"dataset={args.dataset}")

    # Case 1: explicit subset file
    if args.subset_file:
        base = os.path.basename(args.subset_file)
        name, ext = os.path.splitext(base)
        if ext.lower() != ".txt":
            raise ValueError(f"--subset-file must be a .txt file, got: {args.subset_file}")

        # try to pull trailing digits as subset id, e.g. keep_idx_10 -> '10'
        m = re.search(r"(\d+)$", name)
        if m:
            num = m.group(1)
        else:
            num = "custom"

        parts.append(f"DD={num}")

    elif args.distill and args.keep_pct < 100.0:
        parts.append(f"DD={int(args.keep_pct)}")

    # Case 3: full dataset
    else:
        parts.append(f"DD=full")

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
    def __init__(self):
        self.steps = 0
    def step(self, n=1):
        self.steps += int(n)

def epochs_for_fraction(base_epochs, keep_frac, mode="scaled"):
    if mode == "fixed":
        return max(1, int(base_epochs))
    return max(1, int(np.ceil(base_epochs / max(keep_frac, 1e-8))))
