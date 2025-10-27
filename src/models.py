import torch, torch.nn as nn, torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny

VAE_SCALE = 0.18215

class HASeparator(nn.Module):
    def __init__(self, input_dim, num_classes, margin=0.4, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, spk_embeddings, labels=None):
        x = F.normalize(spk_embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=0)
        logits = self.scale * (x @ w)
        penalties = None
        if labels is not None:
            labels = labels.long()
            gt_w = w[:, labels].t().unsqueeze(-1)
            dw = gt_w - w.unsqueeze(0)
            ndw = F.normalize(dw, p=2, dim=1)
            win = torch.einsum('bi,bic->bc', x, ndw)
            penalties = self.margin - torch.clamp(win, max=self.margin)
        return logits, x, penalties

class SDVAE_Embedder(nn.Module):
    def __init__(self, proj_dim=128, base_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(base_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )
    def forward(self, z_map):
        return self.proj(z_map.reshape(z_map.size(0), -1))

class TinyLatentEmbedder(nn.Module):
    """
    Ultra-tiny embedder for SD-VAE latents (B,4,8,8):
      1x1 Conv (4→C) → ReLU → GAP → Linear (C→proj_dim)
    With C=6 and proj_dim=8 → ~80 params.
    """
    def __init__(self, proj_dim=8, channels=6, conv_bias=True, linear_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(4, channels, kernel_size=1, bias=conv_bias)  # params: 4*C (+C if bias)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)                                # no params
        self.fc   = nn.Linear(channels, proj_dim, bias=linear_bias)        # params: C*D (+D if bias)

    def forward(self, z):  # z: (B,4,H,W) e.g., (B,4,8,8)
        h = self.act(self.conv(z))
        h = self.pool(h).flatten(1)  # (B,C)
        return self.fc(h)

class SIMPLE_Embedder(nn.Module):
    def __init__(self, proj_dim=128, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, proj_dim)

    def forward(self, z):
        h = self.stem(z)
        h = self.pool(h).flatten(1)
        return self.head(h)

class ResidBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))

class LatentConvEmbedder(nn.Module):
    """Input: z in R^{B,4,8,8}; Output: L2-unconstrained feature in R^{B,proj_dim}"""
    def __init__(self, proj_dim=256):
        super().__init__()
        ch = 128
        self.stem = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )
        self.block1 = ResidBlock(ch)
        self.block2 = ResidBlock(ch)
        self.block3 = ResidBlock(ch)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, proj_dim),
        )
    def forward(self, z):
        h = self.stem(z)
        h = self.block1(h); h = self.block2(h); h = self.block3(h)
        return self.head(h)

def load_vae(backend: str, repo_path: str, device, dtype):
    if backend == "sd15":
        vae = AutoencoderKL.from_pretrained(repo_path, torch_dtype=dtype).to(device)
    elif backend == "taesd":
        vae = AutoencoderTiny.from_pretrained(repo_path).to(device)
    else:
        raise ValueError("backend must be 'sd15' or 'taesd'")
    for p in vae.parameters(): p.requires_grad_(False)
    vae.eval()
    return vae

@torch.no_grad()
def encode_to_latent(vae, x, device, vae_dtype):
    x = x.to(device)
    if x.dtype != vae_dtype: x = x.to(vae_dtype)
    out = vae.encode(x)
    if hasattr(out, "latent_dist"):
        z = out.latent_dist.mean * VAE_SCALE
    elif hasattr(out, "latent"):
        z = out.latent * VAE_SCALE
    else:
        z = out[0] * VAE_SCALE
    return z.to(torch.float32)

class RawLatentEmbedder(nn.Module):
    """Identity: flatten VAE latent (B,4,8,8) -> (B,256). No learnable params."""
    def __init__(self, proj_dim=256):
        super().__init__()
        assert proj_dim == 256, "Raw latent has 256 dims (4×8×8)."
    def forward(self, z):
        return z.reshape(z.size(0), -1)
