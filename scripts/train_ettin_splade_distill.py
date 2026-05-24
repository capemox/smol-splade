"""
Train an Ettin SPLADE model using distillation from precomputed teacher
margins — the SPLADE-v3 style recipe.

Differences from train_ettin_splade.py:
  - Loss: SpladeLoss wrapping SparseMarginMSELoss (uses teacher's margin scores
    from a precomputed cross-encoder), not just in-batch InfoNCE.
  - Dataset: a sentence-transformers MS MARCO config that includes a teacher
    `score` column (the cross-encoder margin between positive and negative).
  - Slightly tuned dataloader / precision settings for speed.

Usage:
    python scripts/train_ettin_splade_distill.py \\
        --model_id jhu-clsp/ettin-encoder-150m \\
        --output_dir /vol/ettin_splade/ettin-encoder-150m__distill \\
        --max_steps 30000 \\
        --batch_size 64
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMarginMSELoss, SpladeLoss
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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--query_reg_weight", type=float, default=5e-5)
    p.add_argument("--doc_reg_weight", type=float, default=3e-5)
    p.add_argument("--save_steps", type=int, default=5_000)
    p.add_argument("--eval_steps", type=int, default=15_000)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--max_seq_length", type=int, default=192)
    p.add_argument("--dataset_size", type=int, default=500_000)
    p.add_argument(
        "--dataset_id",
        default="sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
        help="HF dataset with precomputed teacher margins. Must have columns "
             "(query, positive, negative, score) — score is the teacher's margin.",
    )
    p.add_argument(
        "--dataset_config", default=None,
        help="Optional HF dataset config name. Leave None for default config."
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Faster matmul on Ada/Hopper GPUs. Tiny effective precision cost.
    torch.set_float32_matmul_precision("high")

    # 1. Load Ettin as a SparseEncoder.
    logging.info(f"Loading encoder: {args.model_id}")
    model = SparseEncoder(
        args.model_id,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{args.model_id.split('/')[-1]}-splade-distill",
        ),
    )
    model.max_seq_length = args.max_seq_length

    # 2. Load a dataset that includes precomputed teacher margin scores.
    logging.info(f"Loading distillation dataset: {args.dataset_id} ({args.dataset_config or 'default'})")
    if args.dataset_config:
        full = load_dataset(args.dataset_id, args.dataset_config, split="train")
    else:
        full = load_dataset(args.dataset_id, split="train")

    # Sanity-check the dataset format expected by SparseMarginMSELoss.
    required_cols = {"query", "positive", "negative", "score"}
    missing = required_cols - set(full.column_names)
    if missing:
        raise RuntimeError(
            f"Dataset {args.dataset_id} is missing columns {missing}. "
            f"Available columns: {full.column_names}. "
            f"SparseMarginMSELoss needs (query, positive, negative, score) where "
            f"score is the teacher's pos-vs-neg margin."
        )

    full = full.select(range(min(args.dataset_size, len(full))))
    splits = full.train_test_split(test_size=1_000, seed=12)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    logging.info(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")
    logging.info(f"Dataset columns: {train_dataset.column_names}")

    # 3. SPLADE-v3 style loss: MarginMSE on teacher scores + FLOPs regularizers.
    loss = SpladeLoss(
        model=model,
        loss=SparseMarginMSELoss(model=model),
        query_regularizer_weight=args.query_reg_weight,
        document_regularizer_weight=args.doc_reg_weight,
    )

    # 4. In-training NanoBEIR mini-eval to track progress.
    nano_evaluator = SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus"],
        batch_size=args.batch_size,
    )

    # 5. Training args. bf16, no grad checkpointing, more dataloader workers.
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
        tf32=True,
        gradient_checkpointing=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
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
        f"max_seq_length={args.max_seq_length} "
        f"lr={args.learning_rate} q_reg={args.query_reg_weight} d_reg={args.doc_reg_weight} "
        f"loss=MarginMSE+SpladeLoss(FLOPs)"
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
