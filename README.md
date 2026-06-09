# SAE-SMO-SPLADE

This repo is currently narrowed to shallow/pruned query encoders for sparse retrieval.
The document encoder stays frozen; the query encoder keeps selected layers from
that same SPLADE-style model so query and document vectors remain dot-product
compatible in the same sparse vocabulary space.

The active training stages are:

| Stage | Purpose |
|---|---|
| `splade_shallow` | Build a shallow query encoder from `naver/splade-v3`. |
| `lion_shallow` | Build a shallow query encoder from `hzeng/Lion-SP-1B-llama3-marco-mntp`. |

Older tokenizer/vocabulary transplant, projection, SAE, direct-align, and
ablation configs have been removed from the active config. If those experiments
are needed again, recover them from branch history.

## Setup

```bash
uv sync
```

## Configure Layers

Edit `config.yaml`.

To keep the first N layers, set `n_layers`:

```yaml
splade_shallow:
  n_layers: 3
```

To choose exact layers, set `layer_indices` instead. This covers first-N,
last-N, spaced, and hand-picked selections:

```yaml
splade_shallow:
  layer_indices: [0, 6, 11]
```

For Lion, negative indices are supported by the model constructor:

```yaml
lion_shallow:
  layer_indices: [0, 3, 6, 10, -1]
```

## Factorization

Use the same stage and toggle the lexical matrix factorization in config:

```yaml
splade_shallow:
  factorize_embeddings: true
  factorized_embedding_dim: 256
  factorization_init: "svd"
```

Set `factorize_embeddings: false` for the non-factorized baseline.

## Train

```bash
uv run train.py splade_shallow --config config.yaml
uv run train.py lion_shallow --config config.yaml
```

Resume from a checkpoint:

```bash
uv run train.py splade_shallow --config config.yaml --resume checkpoints_splade-v3/splade_shallow/align_step_10000.pt
```

Initialize from a checkpoint but reset the step counter:

```bash
uv run train.py splade_shallow --config config.yaml --init-from path/to/checkpoint.pt
```

## Evaluate

NanoBEIR evaluation during training uses the `eval` section in `config.yaml`.
Standalone scripts now accept the same two stages:

```bash
uv run scripts/eval_beir.py --stage splade_shallow --checkpoint checkpoints_splade-v3/splade_shallow/align_final.pt
uv run scripts/eval_msmarco.py --stage splade_shallow --checkpoint checkpoints_splade-v3/splade_shallow/align_final.pt
```

Build indexes for the frozen document encoder selected by the stage:

```bash
uv run scripts/build_beir_index.py --stage splade_shallow
uv run scripts/build_msmarco_index.py --stage splade_shallow
```

## CoCondenser VM Experiment

The complete 15-run pruning experiment for
`naver/splade-cocondenser-ensembledistil` has a resumable one-command launcher:

```bash
./run_cocondenser_vm.sh
```

Run it inside `tmux` on a CUDA VM with at least 200 GiB of free disk. It builds
one shared MS MARCO index, trains first/spaced/last 1-5 layer query encoders for
30,000 steps with MSE and effective batch size 32, selects one winner per depth
on MS MARCO dev NDCG@10, builds one shared set of 13 BEIR indexes, and evaluates
the five winners plus the full model. Re-running the command resumes or skips
completed work. Results are written to `results/cocondenser_vm.md`.
