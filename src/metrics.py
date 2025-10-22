import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pathlib import Path


def make_tsne(**kwargs):
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        kwargs.setdefault("max_iter", 1000)
    else:
        kwargs.setdefault("n_iter", 1000)
    return TSNE(**kwargs)

def _auto_perplexity(n_samples: int, default: int = 35) -> int | None:
    """
    Pick a safe perplexity for t-SNE given n_samples.
    Returns None if n_samples is too small to run t-SNE robustly.
    """
    if n_samples < 10:
        return None
    # keep at least 5, cap at default, and ensure < n_samples
    return max(5, min(default, (n_samples - 1) // 3))

def plot_tsne(E: np.ndarray, labels: np.ndarray, out_path, title: str):
    n = E.shape[0]
    p = _auto_perplexity(n, default=35)
    if p is None:
        # Not enough points to run t-SNE sensibly; just skip plotting
        print(f"[t-SNE] Skipping plot: only {n} samples.")
        return

    tsne = make_tsne(n_components=2, init="random", learning_rate="auto", perplexity=p)
    Z = tsne.fit_transform(E)

    plt.figure(figsize=(7, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.7, label=str(c))
    plt.legend(markerscale=2.5, fontsize=8, loc="best", frameon=False)
    plt.title(f"{title} (perplexity={p}, n={n})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def embedding_metrics(E_te, L_te):
    sil = silhouette_score(E_te, L_te, metric="cosine")
    Et = torch.tensor(E_te); Et = F.normalize(Et, dim=1)
    Lt = torch.tensor(L_te)
    means = torch.stack([Et[Lt==c].mean(0) for c in torch.unique(Lt)])
    means = F.normalize(means, dim=1)
    intra = torch.stack([1 - (Et[Lt==c] @ means[c]).mean() for c in range(means.size(0))]).mean().item()
    S = Et @ means.T
    own = torch.stack([(S[Lt==c, c]).mean() for c in range(means.size(0))])
    other_max = torch.stack([S[Lt==c][:, [j for j in range(means.size(0)) if j!=c]].max(dim=1).values.mean()
                             for c in range(means.size(0))])
    margin = (own - other_max).mean().item()
    return sil, intra, margin
