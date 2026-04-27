# SAE-SPLADE with ettin-17m

Trains an SAE-SPLADE sparse retrieval model using [ettin-17m](https://huggingface.co/Couchbase/ettin-17m) as the backbone encoder. Based on the paper *From Tokens to Concepts: Leveraging SAE for SPLADE* (SIGIR 2026).

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Configure

Before running, edit `src/experiments/ettin/normal.yaml` and set:

1. **`train_config.sae.hf_id`** — HuggingFace model ID for ettin-17m (e.g. `"Couchbase/ettin-17m"`)
2. **`train_config.sae.freeze_regex`** — Regex to freeze the backbone during SAE pretraining. `"model\\."` works for most HF models; adjust to match ettin-17m's module naming
3. **`train_config.sae_splade.distil_data_path`** — Path to ColBERTv2 distillation samples:
   ```bash
   wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json
   ```
4. **[Experimaestro workspace](https://experimaestro-python.readthedocs.io/en/latest/settings/)** — where experiment outputs are saved

## Run

```bash
# Full training run
uv run experimaestro run-experiment src/experiments/ettin/normal.yaml \
    --workdir /your/working/directory

# Debug (small batch/steps to verify the pipeline)
uv run experimaestro run-experiment src/experiments/ettin/debug.yaml \
    --workdir /your/working/directory

# Dry run (check config without launching tasks)
uv run experimaestro run-experiment src/experiments/ettin/normal.yaml \
    --workdir /your/working/directory --run-mode dry-run
```

## Training pipeline

1. **SAE pretraining** (`~160k steps`): Trains a TopK Sparse Autoencoder over ettin-17m token embeddings on the MS MARCO document corpus. Only the SAE weights are updated; the backbone is frozen via `freeze_regex`.

2. **SAE-SPLADE finetuning** (`~240k steps`): Trains a SPLADE-style dual encoder where the vocabulary is the SAE's latent dictionary rather than the tokeniser vocabulary. Uses KL + MSE distillation from ColBERTv2.

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sae_width` | 65536 | SAE dictionary size |
| `k` (SAE) | 8 | TopK sparsity during SAE pretraining |
| `override_k` (SPLADE) | 8 | TopK sparsity during SPLADE finetuning |
| `d_flops` | 0.04 | Document FLOPs regularisation coefficient |
| `q_flops` | 0.06 | Query FLOPs regularisation coefficient |

## Source layout

```
src/
├── experiments/ettin/     # Experiment entry points (YAML + Python)
├── sae/                   # SAE model, adapters, hooks, Triton kernels
├── letor/                 # Trainers and validation listeners
├── dataset/               # Data loaders and samplers
├── text/                  # ColBERT-style tokenisers
└── utils/                 # Triton sparse kernels and dataset helpers
```
