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

class LatentConvEmbedder(nn.Module):
    """
    Input: z [B, 4, H/8, W/8]  (e.g., 4x8x8 for 64x64)
    """
    def __init__(self, proj_dim=256, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.GroupNorm(16, 128), nn.SiLU(),  # 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B,256,1,1]
        )
        self.proj = nn.Linear(256, proj_dim)

    def forward(self, z):
        h = self.net(z).flatten(1)   # [B,256]
        return self.proj(h)


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
