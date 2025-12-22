# Stable Diffusion VAE + HASeparator on MNIST

This repo trains a small projection head and an HASeparator on latents extracted from the Stable Diffusion v1.5 VAE, then visualizes the discriminated latent space with t‑SNE.

Install poetry to run for the following package. 

## Information

You can build this package by installing the poetry or with conda enviroment by installing the necessary packages included in the requirements.txt . This package automatically donwload 2 vae files from Hugging Face with no auth required. 


## 1) Install requirements

Install poetry from here [https://python-poetry.org/docs/#installation]
Inside the latent_discrimination package you can install the requirements :

```bash
 poetry add $(cat requirements.txt)
```

Otherwise with conda create a python 3.10 environment and then install the requirements with:
Install the package python requirements:

```bash
pip install -r requirements.txt
```

## 3) Run training

You can run the training with 2 dataset mnist and cifar10. Choose the proper one with --dataset option {mnist, cifar10} at the training script

For MNIST baseline run with poetry with the following command:

```bash
poetry run python -m src.main --dataset mnist 
```
There are more arguments regarding the seeds, epochs as well as option for distillation with training or by loading pretrained weights and more. Refer to src/main.py for more options.

If you want to train and test the distillation (sweep at 10-90%) run:
```bash

python -m src.main --dataset mnist --vae sd15 --embedder mlp --proj-dim 128 --epochs 5  --run-distill --epochs-per-dd 10 --dd-epoch-mode scaled --batch-size 128 --image-size 64 --num-workers 4 --device auto --seed 42
```
The --dd-epoch-mode offers the option to train the distilled dataset for the same lr and epochs as the baseline model with the arg {fixed} or at a scaled factor based on the distillation percentage with the arg {scaled}


The bash script train.sh runs the main training and evaluation for the cifar10 dataset.
Run:
```bash
chmod +x train.sh
./train.sh
```
To print all the results with mean and std run from the parent folder:
```bash
poetry run python print_results.py 
```


