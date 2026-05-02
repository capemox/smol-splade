#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root: bash scripts/download_data.sh
mkdir -p data

echo "Downloading ColBERTv2 64-way distillation file (~1.5GB) ..."
wget -c \
  "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json" \
  -O data/colbertv2_msmarco_64way.json

echo "Done. File saved to data/colbertv2_msmarco_64way.json"
