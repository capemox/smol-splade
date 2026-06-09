#!/usr/bin/env bash
# Sequential: full SPLADE baseline → full Lion baseline → static cosine retrain+eval
set -e
cd "$(dirname "$0")/.."

DATASETS="nfcorpus scifact arguana scidocs fiqa"
mkdir -p runs/baselines runs/static_loss_ablation

echo "============================================================"
echo "STEP 1/3: Full SPLADE-v3 BEIR baseline (doc_only)"
echo "============================================================"
uv run python3 scripts/eval_beir.py \
  --stage splade_shallow \
  --doc_only \
  --datasets $DATASETS \
  2>&1 | tee runs/baselines/splade_full_beir.log
echo ">>> Full SPLADE-v3 BEIR done."

echo "============================================================"
echo "STEP 2/3: Full Lion BEIR baseline (doc_only)"
echo "============================================================"
uv run python3 scripts/eval_beir.py \
  --stage lion_shallow \
  --doc_only \
  --encode_batch_size 4 \
  --doc_batch_size 8 \
  --datasets $DATASETS \
  2>&1 | tee runs/baselines/lion_full_beir.log
echo ">>> Full Lion BEIR done."

echo "============================================================"
echo "STEP 3/3: Static cosine — retrain 10k steps + BEIR eval"
echo "============================================================"
uv run python3 train.py splade_static \
  --config runs/static_loss_ablation/config_cosine.yaml \
  2>&1 | tee runs/static_loss_ablation/cosine_train.log
uv run python3 scripts/eval_beir.py \
  --stage splade_static \
  --config runs/static_loss_ablation/config_cosine.yaml \
  --checkpoint checkpoints_splade-v3/splade_static_loss_cosine/best_NanoMSMARCO.pt \
  --datasets $DATASETS \
  2>&1 | tee runs/static_loss_ablation/cosine_beir.log
echo ">>> Static cosine done."

echo "============================================================"
echo "ALL DONE"
echo "============================================================"
