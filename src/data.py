from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

def get_loaders(dataset: str, image_size: int, batch_size: int, num_workers: int, device: torch.device):
    pin = device.type == "cuda"

    if dataset.lower() == "mnist":
        from sklearn.datasets import fetch_openml
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        print("Fetching MNIST from OpenML…")
        Xy = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = Xy["data"].reshape(-1, 28, 28).astype("uint8")
        y = Xy["target"].astype("int64")
        X_tr, y_tr = X[:60000], y[:60000]
        X_te, y_te = X[60000:], y[60000:]

        class NPDataset(Dataset):
            def __init__(self, X, y): self.X, self.y = X, y
            def __len__(self): return len(self.y)
            def __getitem__(self, i):
                img = Image.fromarray(self.X[i], mode="L")
                return tf(img), torch.tensor(self.y[i], dtype=torch.long)

        train_ds, test_ds = NPDataset(X_tr, y_tr), NPDataset(X_te, y_te)

    elif dataset.lower() == "cifar10":
        tf_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
        ])

        tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        print("Fetching CIFAR-10 via torchvision…")
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
        test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    train_loader_noshuf = DataLoader(train_ds, batch_size=256, shuffle=False,
                                     num_workers=0, pin_memory=pin)
    return train_ds, test_ds, train_loader, test_loader, train_loader_noshuf
