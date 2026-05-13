# Running Training and MS MARCO Eval on Modal

This repo can run on Modal without uploading trained checkpoints first. The
recommended workflow is:

1. Start a Modal GPU shell with this repo mounted.
2. Copy the repo into a persistent Modal Volume.
3. Train from inside the volume so checkpoints persist.
4. Build a Lion-specific MS MARCO index in the same volume.
5. Run full MS MARCO eval against the saved checkpoint.

The Lion models use a 128K Llama vocabulary, so they cannot reuse the existing
SPLADE-v3 MS MARCO index. Build a separate index for Lion stages.

## Local Modal Setup

Install and authenticate Modal on your local machine:

```bash
pip install modal
modal setup
```

Create one persistent volume for repo outputs, checkpoints, indexes, and Hugging
Face cache:

```bash
modal volume create --version=2 sae-smo-splade-vol
```

Start an interactive GPU shell. `L40S` or `A100` is a good default for the Lion
runs; `A10G` may work but will be slower and tighter on memory.

```bash
modal shell \
  --gpu L40S \
  --volume sae-smo-splade-vol \
  --add-local . \
  --add-python 3.11 \
  --cmd bash
```

Modal mounts the local repo at `/mnt/sae-smo-splade` and the volume at
`/mnt/sae-smo-splade-vol`.

## Prepare the Workspace

Inside the Modal shell:

```bash
set -euo pipefail

export VOL=/mnt/sae-smo-splade-vol
export WORK=$VOL/work/sae-smo-splade
export HF_HOME=$VOL/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p "$VOL/work" "$HF_HOME"
apt-get update && apt-get install -y rsync
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude data/msmarco_lion_index \
  /mnt/sae-smo-splade/ "$WORK/"

cd "$WORK"
pip install uv
uv sync
```

If the Lion model or datasets require Hugging Face credentials, authenticate in
the shell before training:

```bash
huggingface-cli login
```

## Smoke Test

Before launching a long job, verify the repo and model-loading paths:

```bash
uv run smoke_test.py
uv run scripts/build_msmarco_index.py --help
uv run scripts/eval_msmarco.py --help
```

For a tiny end-to-end index build smoke test:

```bash
uv run scripts/build_msmarco_index.py \
  --stage lion_shallow_factorized_align \
  --config config.yaml \
  --index_dir "$VOL/indexes/msmarco_lion_index_smoke" \
  --batch_size 8 \
  --shard_size 100 \
  --limit 200
```

## Train the Lion Factorized Query Model

Run the current first-layers Lion factorized setup:

```bash
uv run train.py lion_shallow_factorized_align --config config.yaml
sync "$VOL"
```

By default this writes checkpoints under:

```text
checkpoints_lion_shallow/lion_shallow_factorized_align/
```

Because the repo is copied into the Modal Volume, those checkpoints persist at:

```text
/mnt/sae-smo-splade-vol/work/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_factorized_align/
```

The checkpoint used for full eval is usually:

```text
checkpoints_lion_shallow/lion_shallow_factorized_align/align_final.pt
```

## Build the Full Lion MS MARCO Index

Build this once per frozen Lion document encoder. It is independent of the query
checkpoint, so the same index can evaluate multiple query checkpoints from the
same Lion doc model.

```bash
uv run scripts/build_msmarco_index.py \
  --stage lion_shallow_factorized_align \
  --config config.yaml \
  --index_dir "$VOL/indexes/msmarco_lion_index" \
  --batch_size 64 \
  --shard_size 50000
sync "$VOL"
```

Notes:

- The full corpus has about 8.84M passages.
- Budget at least 50-100 GB of volume space for the Lion sparse index and cache.
- If you hit GPU OOM during indexing, lower `--batch_size` to `32`, then `16`.
- The script is resumable at shard boundaries. Re-running the same command will
  continue from completed shards.

## Run Full MS MARCO Eval

After training and index building:

```bash
uv run scripts/eval_msmarco.py \
  --stage lion_shallow_factorized_align \
  --checkpoint checkpoints_lion_shallow/lion_shallow_factorized_align/align_final.pt \
  --config config.yaml \
  --index_dir "$VOL/indexes/msmarco_lion_index" \
  --encode_batch_size 8 \
  --query_batch_size 128 \
  --densify_chunk 1024
sync "$VOL"
```

If eval OOMs during scoring, reduce one or more of:

```bash
--encode_batch_size 4
--query_batch_size 64
--densify_chunk 512
```

Expected final output looks like:

```text
==================================================
MSMARCO Dev  (... queries, ...-passage on-disk index)
  NDCG@10 : ...
  MRR@10  : ...
  Checkpoint : checkpoints_lion_shallow/lion_shallow_factorized_align/align_final.pt
==================================================
```

## Evaluating Another Lion Checkpoint

Once `$VOL/indexes/msmarco_lion_index` exists, skip the index-build step and run:

```bash
uv run scripts/eval_msmarco.py \
  --stage lion_shallow_factorized_align \
  --checkpoint path/to/another_checkpoint.pt \
  --config config.yaml \
  --index_dir "$VOL/indexes/msmarco_lion_index" \
  --encode_batch_size 8 \
  --query_batch_size 128 \
  --densify_chunk 1024
```

Only rebuild the index when the frozen document encoder changes.

## Optional: Run a SPLADE-v3 Stage Instead

For SPLADE-v3 factorized spaced training:

```bash
uv run train.py splade_shallow_factorized_spaced_align --config config.yaml
```

Build the SPLADE-v3 MS MARCO index:

```bash
uv run scripts/build_msmarco_index.py \
  --stage splade_shallow_factorized_spaced_align \
  --config config.yaml \
  --index_dir "$VOL/indexes/msmarco_splade_v3_index" \
  --batch_size 64 \
  --shard_size 50000
```

Evaluate:

```bash
uv run scripts/eval_msmarco.py \
  --stage splade_shallow_factorized_spaced_align \
  --checkpoint checkpoints_splade-v3/splade_shallow_factorized_spaced_align/align_final.pt \
  --config config.yaml \
  --index_dir "$VOL/indexes/msmarco_splade_v3_index" \
  --encode_batch_size 16 \
  --query_batch_size 256 \
  --densify_chunk 4096
```

## Modal References

- Modal shell CLI: https://modal.com/docs/reference/cli/shell
- Modal volume CLI: https://modal.com/docs/reference/cli/volume
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal model-weight storage guide: https://modal.com/docs/guide/model-weights
