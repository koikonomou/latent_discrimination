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
    kwargs.setdefault("random_state", 42)
    if "max_iter" in sig.parameters:
        kwargs.setdefault("max_iter", 1000)
    else:
        kwargs.setdefault("n_iter", 1000)
    return TSNE(**kwargs)

def plot_tsne(E, L, out_path: Path, title: str):
    Z = make_tsne(n_components=2, init="random", learning_rate="auto", perplexity=35).fit_transform(E)
    plt.figure(figsize=(7,6))
    for c in np.unique(L):
        m = (L==c)
        plt.scatter(Z[m,0], Z[m,1], s=6, alpha=0.7, label=str(c))
    plt.legend(markerscale=3, fontsize=8, loc="best", frameon=False)
    plt.title(title); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()

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
