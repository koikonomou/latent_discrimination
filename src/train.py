import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .models import encode_to_latent

def ha_loss(logits, labels, penalties, lambda_pen=1.0):
    ce = F.cross_entropy(logits, labels)
    if penalties is None:
        return ce
    cos_pen = penalties.sum(dim=1).mean()     return ce + lambda_pen * cos_pen



def train_epoch(vae, embedder, head, opt, loader, device, vae_dtype, counter=None):
    embedder.train(); head.train()
    total_loss = 0.0; correct = 0; n = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        z = encode_to_latent(vae, xb, device, vae_dtype)   # (B,4,8,8)
        feat = embedder(z)                                 # (B, proj_dim)

        logits, _, penalties = head(feat, labels=yb)

        loss = ha_loss(logits, yb, penalties, lambda_pen=1.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        n += yb.size(0)

        if counter is not None:
            counter.step()

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def eval_epoch(vae, embedder, head, loader, device, vae_dtype):
    embedder.eval(); head.eval()
    total_loss = 0.0; correct = 0; n = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        z = encode_to_latent(vae, xb, device, vae_dtype)
        feat = embedder(z)
        logits, _, penalties = head(feat, labels=yb)

        loss = ha_loss(logits, yb, penalties, lambda_pen=1.0)

        total_loss += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        n += yb.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)
