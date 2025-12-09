"""
This code include all the models for the training process.
The classes ImageToLatentAdapter and PlainFromLatentBackbone are used to transform the plain dataset to (B,4,8,8) similar to the latent dims so we don't have to change the inital models.
The HASeparator approach is developed based on this paper :
Kansizoglou, Ioannis, et al. "Haseparator: Hyperplane-assisted softmax." 2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2020.
All the other embedders are used for testing.
"""
import os
import torch, torch.nn as nn, torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny
from huggingface_hub import snapshot_download
from torchvision import models as tv_models

# This param is used in Stable Diffusion to normalize latent vectors
VAE_SCALE = 0.18215

class ResNet18LatentEmbedder(nn.Module):
    """
    Use a ResNet18-style backbone on VAE latents (B,4,H,W) and output a proj_dim embedding.

    - Input:  z (B,4,H,W)  e.g. (B,4,8,8) from SD-VAE
    - Output: (B, proj_dim)
    """
    def __init__(self, proj_dim=128, pretrained=False):
        super().__init__()
        # base ResNet18
        base = tv_models.resnet18(weights=None if not pretrained else tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv from 3→4 channels
        old_conv = base.conv1
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if pretrained:
            # Copy weights for first 3 channels, init 4th as mean of them
            with torch.no_grad():
                self.conv1.weight[:, :3] = old_conv.weight
                self.conv1.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
            if self.conv1.bias is not None:
                nn.init.zeros_(self.conv1.bias)

        # Reuse the rest of ResNet18
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool  # GlobalAvgPool → (B, 512, 1, 1)

        # Replace classifier with embedding head
        self.fc = nn.Linear(512, proj_dim)

    def forward(self, z):
        # z: (B,4,H,W) from VAE
        x = self.conv1(z)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)              # (B,512,1,1)
        x = torch.flatten(x, 1)          # (B,512)
        x = self.fc(x)                   # (B,proj_dim)
        return x

class ImageToLatentAdapter(nn.Module):
    """
    Map image (B, Cin, H, W) to 'latent-like' (B, 4, 8, 8) without using the VAE.
    - AdaptiveAvgPool2d -> (8,8)
    - 1x1 conv to go Cin->{4} channels (Cin can be 1 or 3)
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.map  = nn.Conv2d(in_ch, 4, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.pool(x)           # (B, Cin, 8, 8)
        z = self.map(x)            # (B, 4,   8, 8)
        return z

class PlainFromLatentBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, in_ch: int):
        super().__init__()
        self.adapt = ImageToLatentAdapter(in_ch)
        self.backbone = backbone
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):          # x: images (B, Cin, H, W)
        z = self.adapt(x)          # (B,4,8,8)
        f = self.backbone(z)       # (B, feat_dim)
        return self.fc(f)          # (B, num_classes)


class HASeparator(nn.Module):
    def __init__(self, input_dim, num_classes, margin=0.4, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embed, labels=None):
        x = F.normalize(embed, p=2, dim=1)
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
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.head(h)

class RawLatentEmbedder(nn.Module):
    """Identity: flatten VAE latent (B,4,8,8) -> (B,256). No learnable params."""
    def __init__(self, proj_dim=256):
        super().__init__()
        assert proj_dim == 256, "Raw latent has 256 dims (4×8×8)."
    def forward(self, z):
        return z.reshape(z.size(0), -1)

def load_vae(backend: str, repo_path: str, device, dtype):
    if backend == "sd15":
        if not os.path.isdir('repo_path'):
            snapshot_download(
                repo_id="stabilityai/sd-vae-ft-mse",
                local_dir="./_sd15_vae",
                allow_patterns=["diffusion_pytorch_model.safetensors", "config.json"],
                resume_download=True,             # resumes partials
                max_workers=8                     # parallel chunks
            )
            print("SD vae downloaded at -> ./_sd15_vae")
            vae = AutoencoderKL.from_pretrained(repo_path, torch_dtype=dtype).to(device)
        else:
            vae = AutoencoderKL.from_pretrained(repo_path, torch_dtype=dtype).to(device)
    elif backend == "taesd":
        if not os.path.isdir('repo_path'):
            snapshot_download(
                repo_id="madebyollin/taesd",
                local_dir="./_taesd",
                allow_patterns=["diffusion_pytorch_model.safetensors", "config.json"],
                resume_download=True,             # resumes partials
                max_workers=8                     # parallel chunks
            )
            print("Taesd vae downloaded at -> ./_taesd")
            vae = AutoencoderTiny.from_pretrained(repo_path).to(device)
        else:
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

