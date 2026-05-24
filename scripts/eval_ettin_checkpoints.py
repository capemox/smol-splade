"""
Evaluate all checkpoints from an Ettin SPLADE training run with
SparseNanoBEIREvaluator. Prints a comparison table at the end.

Usage:
    python scripts/eval_ettin_checkpoints.py \\
        --base_dir /vol/ettin_splade/ettin-encoder-150m__B_higher_reg_fast
"""

import argparse
import logging
from pathlib import Path

import torch
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def find_checkpoints(base_dir: Path):
    """Discover checkpoint-N subdirectories and final/ if present."""
    ckpts = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    final = base_dir / "final"
    if final.exists() and final.is_dir():
        ckpts.append(final)
    return ckpts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Not found: {base_dir}")

    ckpts = find_checkpoints(base_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoint-* dirs in {base_dir}")
    logging.info(f"Found {len(ckpts)} checkpoints to evaluate")
    for c in ckpts:
        logging.info(f"  {c.name}")

    evaluator = SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus"],
        batch_size=args.batch_size,
    )

    results = []
    for ckpt_dir in ckpts:
        logging.info(f"\n{'='*60}\nEvaluating: {ckpt_dir.name}\n{'='*60}")
        try:
            model = SparseEncoder(str(ckpt_dir))
            scores = evaluator(model)
            results.append({
                "checkpoint": ckpt_dir.name,
                "ms_ndcg": scores.get("NanoMSMARCO_dot_ndcg@10", float("nan")),
                "ms_mrr":  scores.get("NanoMSMARCO_dot_mrr@10",  float("nan")),
                "nf_ndcg": scores.get("NanoNFCorpus_dot_ndcg@10", float("nan")),
                "q_dims":  scores.get("NanoMSMARCO_query_active_dims",  float("nan")),
                "d_dims":  scores.get("NanoMSMARCO_corpus_active_dims", float("nan")),
            })
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed to eval {ckpt_dir.name}: {e}")
            results.append({"checkpoint": ckpt_dir.name, "error": str(e)})

    # Print comparison
    print("\n" + "=" * 80)
    print(f"{'Checkpoint':<22} {'MS_NDCG':<10} {'MS_MRR':<10} {'NF_NDCG':<10} {'q_dims':<10} {'d_dims':<10}")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['checkpoint']:<22} ERROR: {r['error'][:50]}")
        else:
            print(f"{r['checkpoint']:<22} {r['ms_ndcg']:<10.4f} {r['ms_mrr']:<10.4f} "
                  f"{r['nf_ndcg']:<10.4f} {r['q_dims']:<10.1f} {r['d_dims']:<10.1f}")

    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["ms_ndcg"])
        print(f"\nBest by NanoMSMARCO NDCG@10: {best['checkpoint']} ({best['ms_ndcg']:.4f})")


if __name__ == "__main__":
    main()
