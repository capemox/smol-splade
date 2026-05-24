"""
Train a SPLADE model from an Ettin encoder using sentence-transformers.

Follows the Ettin model-card SPLADE recipe (SpladeLoss with
SparseMultipleNegativesRankingLoss). Two deliberate deviations from the model
card snippet, for downstream MS MARCO performance:

  1. Trains on MS MARCO triplets, not Natural Questions, because that's where
     we evaluate downstream.
  2. Trains for ~30k optimizer steps (vs. the snippet's ~6k), which is closer
     to published SPLADE training budgets.

Outputs a SparseEncoder at <output_dir>/final. The underlying HuggingFace
MaskedLM weights are at <output_dir>/final/0_MLMTransformer and can be loaded
directly with AutoModelForMaskedLM (which is what the repo's FrozenSPLADE
wrapper expects).

Usage:
    python scripts/train_ettin_splade.py \\
        --model_id jhu-clsp/ettin-encoder-150m \\
        --output_dir /vol/ettin_splade/ettin-encoder-150m \\
        --max_steps 30000 \\
        --batch_size 32
"""

import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="jhu-clsp/ettin-encoder-150m")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=30_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Effective batch = batch_size * grad_accum.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--query_reg_weight", type=float, default=5e-5)
    p.add_argument("--doc_reg_weight", type=float, default=3e-5)
    p.add_argument("--save_steps", type=int, default=5_000)
    p.add_argument("--eval_steps", type=int, default=5_000)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="Memory-saving. Enabled by default.")
    p.add_argument("--dataset_size", type=int, default=500_000,
                   help="Number of MS MARCO triplets to sample. Full set is ~10M.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Ettin and wrap as a SparseEncoder.
    logging.info(f"Loading encoder: {args.model_id}")
    model = SparseEncoder(
        args.model_id,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{args.model_id.split('/')[-1]}-splade-msmarco",
        ),
    )
    model.max_seq_length = 192

    # 2. MS MARCO triplets.
    logging.info("Loading MS MARCO triplet-hard data")
    full = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    full = full.select(range(min(args.dataset_size, len(full))))
    splits = full.train_test_split(test_size=1_000, seed=12)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    logging.info(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    # 3. SpladeLoss — recipe loss + FLOPs regularizers.
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=args.query_reg_weight,
        document_regularizer_weight=args.doc_reg_weight,
    )

    # 4. NanoBEIR mini-eval during training.
    nano_evaluator = SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus"],
        batch_size=args.batch_size,
    )

    # 5. Training args.
    training_args = SparseEncoderTrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        report_to="none",
        dataloader_num_workers=2,
        run_name=output_dir.name,
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=nano_evaluator,
    )

    logging.info(
        f"Training: steps={args.max_steps} batch={args.batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps} "
        f"lr={args.learning_rate} q_reg={args.query_reg_weight} d_reg={args.doc_reg_weight}"
    )
    trainer.train()

    logging.info("Final NanoBEIR eval:")
    nano_evaluator(model)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    logging.info(f"Saved final SparseEncoder to {final_dir}")
    logging.info(
        f"HuggingFace MLM weights at {final_dir}/0_MLMTransformer — "
        "load with AutoModelForMaskedLM as the doc encoder."
    )


if __name__ == "__main__":
    main()
