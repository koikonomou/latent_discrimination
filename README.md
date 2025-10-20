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

```bash
huggingface-cli login
# or set env var just for the run:
# export HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

You must accept the license for `runwayml/stable-diffusion-v1-5` on the model page once on your HF account.

## 3) Run training

```bash
python train_haseparator_mnist_sdvae.py   --epochs 10   --batch-size 256   --image-size 64   --num-workers 4   --device auto
```

Key outputs:
- `checkpoints/best.ckpt` – best model weights (embedder + head)
- `artifacts/tsne_train.png` – t-SNE of train embeddings
- `artifacts/tsne_test.png` – t-SNE of test embeddings
- `artifacts/train_log.csv` – per-epoch metrics

You can resume, change hyperparams, or run `--dry-run` to just build a batch and exit (sanity check).

## 4) Tips
- If VRAM is tight, lower `--batch-size` or force `--device cpu`.
- First run will download the VAE (~335MB); later runs use the local cache.
- For reproducible color labels in plots, we use Matplotlib’s default cycle.
