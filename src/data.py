from pathlib import Path
from PIL import Image
import numpy as np
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.datasets import fetch_openml
from .utils import resolve_device



TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _prepare_tiny_imagenet(root: Path) -> Path:
    """
    Download + extract Tiny-ImageNet-200 into `root`
    and reorganize val/ into class subfolders.
    Returns the path to root/'tiny-imagenet-200'.
    """
    root.mkdir(parents=True, exist_ok=True)
    tiny_root = root / "tiny-imagenet-200"

    # Already prepared
    if tiny_root.exists():
        return tiny_root

    zip_path = root / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        print(f"Downloading Tiny-ImageNet from {TINY_IMAGENET_URL} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print("Extracting Tiny-ImageNet...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print("Extraction complete.")

    # Reorganize val/ so that ImageFolder can read it
    val_dir = tiny_root / "val"
    anno_path = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"

    if anno_path.exists() and images_dir.exists():
        print("Reorganizing Tiny-ImageNet val/ into class subfolders...")
        with open(anno_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, cls = parts[0], parts[1]
                cls_dir = val_dir / cls
                cls_dir.mkdir(exist_ok=True)
                src = images_dir / fname
                dst = cls_dir / fname
                if src.exists():
                    shutil.move(str(src), str(dst))

        # remove now empty images/ and annotation file
        try:
            shutil.rmtree(images_dir)
        except FileNotFoundError:
            pass
        try:
            anno_path.unlink()
        except FileNotFoundError:
            pass

        print("Reorganization done.")

    return tiny_root

def get_loaders(dataset: str, image_size: int, batch_size: int, num_workers: int, device):
    pin = device

    if dataset.lower() == "mnist":
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

    elif dataset.lower() == "fashion_mnist":
        tf_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        print("Fetching Fashion-MNIST via torchvision…")
        train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf_train)
        test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf_test)

    elif dataset.lower() == "cifar10":
        tf_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
        ])

        tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        print("Fetching CIFAR-10 via torchvision…")
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
        test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)

    
    elif dataset.lower() == "cifar100":
        tf_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
        ])

        tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        print("Fetching CIFAR-100 via torchvision…")
        train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=tf_train)
        test_ds  = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf_test)
    
    elif dataset.lower() =="tinyimagenet":
        # Assumes Tiny-ImageNet in standard folder structure:
        #   ./data/tiny-imagenet-200/train/<class>/images/*.JPEG
        #   ./data/tiny-imagenet-200/val/<class>/images/*.JPEG
        # If your val/ is still in the original flat layout with val_annotations.txt,
        # you’ll need one more preprocessing step to reorganize it into class folders.
        data_root = Path("./data")
        tiny_root = _prepare_tiny_imagenet(data_root)
        train_dir = data_root / "train"
        val_dir   = data_root / "val"

        tf_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

        tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

        print("Fetching Tiny-ImageNet via ImageFolder…")
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=tf_train)
        test_ds  = datasets.ImageFolder(root=str(val_dir), transform=tf_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    train_loader_noshuf = DataLoader(train_ds, batch_size=256, shuffle=False,num_workers=0, pin_memory=pin)
    
    return train_ds, test_ds, train_loader, test_loader, train_loader_noshuf
