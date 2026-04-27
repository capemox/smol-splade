# SAE-SPLADE with ettin-17m

Trains an SAE-SPLADE sparse retrieval model using [jhu-clsp/ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) as the backbone encoder. Based on the paper *From Tokens to Concepts: Leveraging SAE for SPLADE* (SIGIR 2026).

## Setup

```bash
uv sync
```

## Run

```bash
# Stage 1: SAE pretraining (~160k steps)
uv run train.py sae

# Stage 2: SPLADE finetuning (~240k steps, requires stage 1 to complete first)
uv run train.py splade

# Resume from a checkpoint
uv run train.py sae --resume checkpoints/sae/step_10000.pt

# Use a custom config
uv run train.py sae --config my_config.yaml
```

## Monitor

```bash
tensorboard --logdir checkpoints/
```

Then open `http://localhost:6006`.

## Training pipeline

1. **SAE pretraining**: Trains a TopK Sparse Autoencoder over ettin-17m token embeddings on the MS MARCO document corpus. Only the SAE weights are updated; the backbone is frozen.

2. **SPLADE finetuning**: Trains a SPLADE-style dual encoder where the vocabulary is the SAE's latent dictionary. Uses KL + MSE distillation from ColBERTv2 if a teacher file is provided, otherwise falls back to in-batch cross-entropy.

## Key hyperparameters (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.sae_width` | 65536 | SAE dictionary size |
| `model.k` | 8 | TopK activations per token |
| `sae.batch_size` | 16 | Docs per step (stage 1) |
| `splade.batch_size` | 8 | Queries per step (stage 2) |
| `splade.lambda_d` | 0.04 | Document FLOPs regularisation |
| `splade.lambda_q` | 0.06 | Query FLOPs regularisation |

## ColBERTv2 distillation (optional)

Download teacher scores and set `splade.distil_data_path` in `config.yaml`:

```bash
wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json
```
