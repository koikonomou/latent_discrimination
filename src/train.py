import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .models import encode_to_latent

def train_epoch(vae, embedder, head, opt, loader, device, vae_dtype, counter=None):
    embedder.train(); head.train()
    total=correct=0; loss_sum=0.0
    for xb, yb in loader:
        xb=xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True).long()
        with torch.no_grad():
            z = encode_to_latent(vae, xb, device, vae_dtype)
        feat = embedder(z)
        logits, emb, penalties = head(feat, labels=yb)
        loss = F.cross_entropy(logits, yb)
        if penalties is not None:
            loss = loss + 0.01 * penalties.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        if counter is not None: counter.add(1)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1); correct += (pred==yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(vae, embedder, head, loader, device, vae_dtype):
    embedder.eval(); head.eval()
    total=correct=0; loss_sum=0.0
    for xb, yb in loader:
        xb=xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True).long()
        z = encode_to_latent(vae, xb, device, vae_dtype)
        feat = embedder(z)
        logits, emb, _ = head(feat, labels=None)
        loss = F.cross_entropy(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1); correct += (pred==yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total
