# Stable Diffusion VAE + HASeparator on MNIST

This repo trains a small projection head and an HASeparator on latents extracted from the Stable Diffusion v1.5 VAE, then visualizes the discriminated latent space with t‑SNE.

## 1) Create the Conda env

```bash
conda env create -n vae python=3.10
conda activate vae

```

Install the package python requirements:
```bash
pip install -r requirements.txt
```

## 2) Hugging Face auth (needed to pull SD v1.5 VAE)

The Stable VAE and Tiny one that we use in this repo doen't require a login howerever for different VAE login is required.

```bash
huggingface-cli login
# or set env var just for the run:
# export HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## 3) Run training

For MNIST baseline run:
```bash
python -m src.main --dataset mnist --vae sd15 --embedder mlp --proj-dim 128 --epochs 10  --batch-size 64 --image-size 64 --num-workers 4 --device auto --seed 42
```

For CIFAR10 run 
```bash
python -m src.main --dataset cifar10 --vae sd15 --embedder conv --proj-dim 256 --epochs 150 --batch-size 128 --image-size 64 --num-workers 4 --device auto --seed 42
```

If you want to test the distillation (sweep at 10-90%) run:
```bash

python -m src.main --dataset mnist --vae sd15 --embedder mlp --proj-dim 128 --epochs 1 --skip-train --load-ckpt runs/<...>/ckpts/best.ckpt --run-distill --dd-method kcenter_cosine --epochs-per-dd 10 --dd-epoch-mode scaled --batch-size 128 --image-size 64 --num-workers 4 --device auto --seed 42
```


Key outputs:
- `runs/ckpts` – best model weights (best.ckpt)
- `runs/plots`
- `runs/subsets`
- `runs/config.json` 

