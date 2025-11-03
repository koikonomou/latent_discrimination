# src/data_custom.py
from pathlib import Path
import csv
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

def _normalize_label(val, invert: bool = False) -> int:
    # Accept 0/1, True/False,
    if isinstance(val, (int, np.integer)):
        y = int(val)
    else:
        s = str(val).strip().lower()
        if s in {"true"}:
            y = 1
        elif s in {"false"}:
            y = 0
        else:
            raise ValueError(f"Unrecognized label: {val!r} (expected True/False)")
    return 1 - y if invert else y

class PNGBinaryDataset(Dataset):
    """
    CSV expected header with at least: fname, label
    Extra columns (e.g., 'shape','n') are ignored.
    """
    def __init__(self, img_root: str, csv_path: str, transform=None, invert_labels: bool = False):
        self.img_root = Path(img_root)
        self.items: List[Tuple[Path,int]] = []
        self.transform = transform

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = [h.strip().lower() for h in header]
            try:
                i_fname = cols.index("fname")
                i_label = cols.index("label")
            except ValueError as e:
                raise RuntimeError(f"CSV must contain columns 'fname' and 'label'. Found: {cols}") from e

            for row in reader:
                if not row or len(row) <= max(i_fname, i_label):
                    continue
                fn = row[i_fname].strip()
                lab = row[i_label].strip()
                p = Path(fn)
                if not p.is_absolute():
                    p = self.img_root / p
                y = _normalize_label(lab, invert=invert_labels)
                self.items.append((p, y))

        if len(self.items) == 0:
            raise RuntimeError(f"No items read from CSV: {csv_path}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)

def build_transforms(image_size: int):
    # VAE expects [-1,1] after Normalize([0.5],[0.5])
    train_tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    test_tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return train_tf, test_tf

def get_custom_loaders(img_root: str, csv_path: str, image_size: int, batch_size: int,
                       num_workers: int, device, test_size=0.2, seed=42,
                       invert_labels: bool = False):
    train_tf, test_tf = build_transforms(image_size)
    full = PNGBinaryDataset(img_root, csv_path, transform=None, invert_labels=invert_labels)

    # stratified split
    labels = np.array([y for _, y in full.items], dtype=np.int64)
    idx = np.arange(len(full))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)

    class _Wrap(Dataset):
        def __init__(self, base, indices, tf):
            self.base = base; self.indices = indices; self.tf = tf
        def __len__(self): return len(self.indices)
        def __getitem__(self, k):
            i = self.indices[k]
            p, y = self.base.items[i]
            img = Image.open(p).convert("RGB")
            return self.tf(img), torch.tensor(y, dtype=torch.long)

    train_ds = _Wrap(full, idx_tr, train_tf)
    test_ds  = _Wrap(full, idx_te,  test_tf)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    train_loader_noshuf = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin)

    num_classes = 2
    return train_ds, test_ds, train_loader, test_loader, train_loader_noshuf, num_classes
