import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from .models import encode_to_latent


def l2_normalize_np(X, eps=1e-8):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6780070
def farthest_first_cosine(Ec, k, seed=42):
    if Ec.shape[0] == 0: return np.array([], dtype=int)
    k = min(k, Ec.shape[0])
    sims = Ec @ Ec.T
    start = int(np.argmax(sims.mean(axis=1)))
    chosen = [start]
    min_d = 1.0 - sims[start]
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        new_d = 1.0 - (Ec @ Ec[nxt:nxt+1].T).ravel()
        min_d = np.minimum(min_d, new_d)
    return np.array(chosen, dtype=int)

def per_sample_margin(logits, y):
    lt = logits[np.arange(len(y)), y]
    masked = logits.copy(); masked[np.arange(len(y)), y] = -np.inf
    maxo = masked.max(axis=1)
    return (lt - maxo)


@torch.no_grad()
def collect_embed_and_logits(vae, embedder, head, loader_no_shuffle, device, vae_dtype):
    embedder.eval(); head.eval()
    Es, Ys, IDX, LOG = [], [], [], []
    base = 0
    for xb, yb in loader_no_shuffle:
        xb = xb.to(device)
        z = encode_to_latent(vae, xb, device, vae_dtype)
        feat = embedder(z)
        logits, emb, _ = head(feat, labels=None)
        Es.append(emb.cpu().numpy()); LOG.append(logits.cpu().numpy())
        Ys.append(yb.numpy()); bsz = yb.size(0)
        IDX.append(np.arange(base, base+bsz)); base += bsz
    E = l2_normalize_np(np.concatenate(Es, axis=0))
    L = np.concatenate(Ys, axis=0)
    I = np.concatenate(IDX, axis=0)
    G = np.concatenate(LOG, axis=0)
    return E, L, I, G

@torch.no_grad()
def collect_vae_features(vae, embedder_throw, loader_no_shuffle, device, vae_dtype):
    Fs, Ys, IDX = [], [], []
    base=0
    for xb, yb in loader_no_shuffle:
        xb = xb.to(device)
        z = encode_to_latent(vae, xb, device, vae_dtype)
        f = z.reshape(z.size(0), -1).cpu().numpy()
        Fs.append(f); Ys.append(yb.numpy())
        bsz=yb.size(0); IDX.append(np.arange(base, base+bsz)); base+=bsz
    F_all = np.concatenate(Fs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)
    IDX_all = np.concatenate(IDX, axis=0)
    return F_all, Y_all, IDX_all

## THiS is for the test with the original 
def kmeans_assign_dist_whiten(F_all, k, seed=42):
    scaler = StandardScaler()
    Fw = scaler.fit_transform(F_all.astype(np.float32))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    assign = km.fit_predict(Fw)
    centroids = km.cluster_centers_[assign]
    d2 = ((Fw - centroids)**2).sum(axis=1)
    return assign, d2

def make_distilled_indices_balanced(all_indices, distances, labels, pct, criterion, seed=42, num_classes=10):
    all_indices = np.asarray(all_indices); distances = np.asarray(distances); labels = np.asarray(labels)
    keep = []; rng = np.random.default_rng(seed)
    for c in range(num_classes):
        m = labels == c
        idx_c = all_indices[m]; d_c = distances[m]; n = len(idx_c)
        drop = int(round(pct * n / 100.0))
        if drop <= 0: keep.extend(idx_c.tolist()); continue
        if criterion == "nearest":   order = np.argsort(d_c)
        elif criterion == "furthest": order = np.argsort(-d_c)
        else: order = rng.permutation(n)
        drop_set = set(idx_c[order[:drop]].tolist())
        keep.extend([int(i) for i in idx_c if int(i) not in drop_set])
    return keep

def select_kcenter_cosine_balanced(E, Y, IDX, pct, percentage=False, num_classes=10, seed=42):
    keep=[]
    for c in range(num_classes):
        m = (Y==c); Ec = E[m]; Ic = IDX[m]
        if percentage==True:
            k_keep = int(round((100 - pct) * len(Ic) / 100.0))
        else:
            k_keep = min(pct, len(Ic))
        if k_keep <= 0: continue
        if k_keep >= len(Ic): keep.extend(Ic.tolist()); continue
        sel_rel = farthest_first_cosine(Ec, k_keep, seed=seed)
        keep.extend(Ic[sel_rel].tolist())
    return keep

def select_kcenter_cosine_global(E, IDX, pct, seed=42):
    keep_total = int(round((100 - pct) * len(IDX) / 100.0))
    sel_rel = farthest_first_cosine(E, keep_total, seed=seed)
    return IDX[sel_rel].tolist()

def select_random(E, Y, IDX, pct, percentage=False, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)

    keep = []
    for c in range(num_classes):
        m = (Y==c); Ec=E[m]; Ic = IDX[m]
        if percentage==True:
            k_keep = int(round((100 - pct) * len(Ic) / 100.0))
        else:
            k_keep = min(pct, len(Ic))
        if k_keep<=0: continue
        if k_keep>= len(Ic): keep.extend(Ic.tolist()); continue
        chosen = rng.choice(Ic, size=k_keep, replace=False)
        keep.extend(chosen.tolist())
    return keep

