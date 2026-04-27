#!/usr/bin/env python
"""Train SAE-SPLADE with ettin-17m.

Two stages:

  Stage 1 — SAE pretraining
    Trains a TopK Sparse Autoencoder on backbone token embeddings from the
    MS MARCO document corpus.  The backbone is frozen; only the SAE weights
    (W_enc, b_enc, W_dec, b_dec) are updated.

  Stage 2 — SPLADE finetuning
    Loads the pretrained SAE and trains the full dual encoder for retrieval
    using KL/MSE distillation from ColBERTv2 (or CE loss if no teacher file).

Usage:
    # Install dependencies
    uv sync

    # Stage 1: SAE pretraining
    uv run train.py sae

    # Stage 2: SPLADE finetuning (after stage 1 completes)
    uv run train.py splade

    # Use a different config file
    uv run train.py sae --config my_config.yaml

    # Resume SAE pretraining from a checkpoint
    uv run train.py sae --resume checkpoints/sae/step_50000.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: SAE pretraining
# ──────────────────────────────────────────────────────────────────────────────

def train_sae(cfg: dict, resume: str | None = None):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter

    from model import TopKSAE, SAEPretrainModel, sae_loss
    from data import make_sae_loader

    mc = cfg["model"]
    sc = cfg["sae"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(mc["hf_id"])
    sae = TopKSAE(
        hidden_size=_get_hidden_size(mc["hf_id"]),
        sae_width=mc["sae_width"],
        k=mc["k"],
        aux_k=mc["aux_k"],
        dead_steps_threshold=mc["dead_steps_threshold"],
        normalize_input=mc["normalize_input"],
    )
    model = SAEPretrainModel(mc["hf_id"], sae, freeze_backbone=sc["freeze_backbone"])
    model.to(device)

    # ── Optionally initialise bias normalisation ───────────────────────
    if mc["normalize_input"]:
        print("Computing SAE input normalisation statistics …")
        _init_sae_normalisation(model, tokenizer, sc, device)

    # ── Optimiser ─────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(
        params, lr=sc["lr"], weight_decay=sc["weight_decay"], eps=6e-10, betas=(0.9, 0.999)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=sc["warmup_steps"],
        num_training_steps=sc["max_steps"],
    )
    scaler = GradScaler(enabled=sc["fp16"] and device.type == "cuda")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.sae.load_state_dict(ckpt["sae"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    # ── Setup ─────────────────────────────────────────────────────────
    out_dir = Path(sc["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    data_iter = make_sae_loader(
        sc["corpus_dataset"],
        sc["corpus_text_field"],
        tokenizer,
        sc["batch_size"],
        sc["doc_max_length"],
        device,
    )

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    for step in range(start_step, sc["max_steps"]):
        batch = next(data_iter)

        optimizer.zero_grad()
        with autocast(enabled=sc["fp16"] and device.type == "cuda"):
            output = model(**batch)
            loss, metrics = sae_loss(output, sc["rcst_coeff"], sc["aux_coeff"])

        scaler.scale(loss).backward()

        # Remove parallel gradient component from W_dec before the step
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        model.sae.remove_parallel_gradient()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Normalise W_dec columns after each step
        model.sae.post_step()

        if (step + 1) % sc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"[SAE] step {step+1:>7} | "
                f"loss {metrics['loss']:.4f} | rcst {metrics['rcst']:.4f} | "
                f"aux {metrics['aux']:.4f} | sparsity {metrics['sparsity']:.1f} | "
                f"dead {metrics['dead_ratio']:.3f} | lr {lr:.2e} | "
                f"{elapsed:.0f}s"
            )
            for k, v in metrics.items():
                writer.add_scalar(f"sae/{k}", v, step + 1)
            writer.add_scalar("sae/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % sc["save_every"] == 0:
            _save_sae(model.sae, optimizer, scheduler, step + 1, out_dir / f"step_{step+1}.pt")

    _save_sae(model.sae, optimizer, scheduler, sc["max_steps"], out_dir / "sae_final.pt")
    print(f"SAE pretraining complete. Saved to {out_dir / 'sae_final.pt'}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: SPLADE finetuning
# ──────────────────────────────────────────────────────────────────────────────

def train_splade(cfg: dict, resume: str | None = None):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter

    from model import TopKSAE, SAESPLADEModel, splade_loss
    from data import (
        TevatronMSMARCODataset,
        ColBERTDistillationDataset,
        build_corpus_lookup,
        build_query_lookup,
        collate_splade_batch,
    )

    mc = cfg["model"]
    sp = cfg["splade"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load pretrained SAE ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(mc["hf_id"])
    hidden_size = _get_hidden_size(mc["hf_id"])
    sae = TopKSAE(
        hidden_size=hidden_size,
        sae_width=mc["sae_width"],
        k=mc["k"],
        aux_k=mc["aux_k"],
        dead_steps_threshold=mc["dead_steps_threshold"],
        normalize_input=mc["normalize_input"],
    )
    ckpt_path = sp["sae_checkpoint"]
    print(f"Loading SAE from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sae.load_state_dict(ckpt["sae"])

    model = SAESPLADEModel(mc["hf_id"], sae, scale=True)
    model.to(device)

    # ── Dataset ────────────────────────────────────────────────────────
    if sp.get("distil_data_path"):
        corpus = build_corpus_lookup(cfg["sae"]["corpus_dataset"], cfg["sae"].get("corpus_text_field", "text"))
        queries = build_query_lookup(sp["queries_dataset"])
        dataset: object = ColBERTDistillationDataset(
            sp["distil_data_path"], corpus, queries, nway=sp["nway"]
        )
    else:
        dataset = TevatronMSMARCODataset(nway=sp["nway"])

    def data_iter():
        buf: list = []
        while True:
            for item in dataset:
                buf.append(item)
                if len(buf) == sp["batch_size"]:
                    yield collate_splade_batch(
                        buf, tokenizer, sp["query_max_length"], sp["doc_max_length"], device
                    )
                    buf = []

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sp["lr"], weight_decay=sp["weight_decay"], eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, sp["warmup_steps"], sp["max_steps"]
    )
    scaler = GradScaler(enabled=sp["fp16"] and device.type == "cuda")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    # ── Setup ─────────────────────────────────────────────────────────
    out_dir = Path(sp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    batches = data_iter()

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    for step in range(start_step, sp["max_steps"]):
        # Ramp FLOPs regularisation linearly from 0 to 1
        flops_scale = min(1.0, step / max(sp["flops_warmup_steps"], 1))

        q_ids, q_mask, d_ids, d_mask, teacher_scores = next(batches)
        optimizer.zero_grad()

        with autocast(enabled=sp["fp16"] and device.type == "cuda"):
            loss, metrics = splade_loss(
                model, q_ids, q_mask, d_ids, d_mask, teacher_scores,
                lambda_d=sp["lambda_d"],
                lambda_q=sp["lambda_q"],
                flops_scale=flops_scale,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.sae.post_step()

        if (step + 1) % sp["log_every"] == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"[SPLADE] step {step+1:>7} | "
                f"loss {metrics['loss']:.4f} | retr {metrics['retr']:.4f} | "
                f"flops {metrics['flops']:.4f} | "
                f"d_nnz {metrics['avg_d_nnz']:.1f} | q_nnz {metrics['avg_q_nnz']:.1f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s"
            )
            for k, v in metrics.items():
                writer.add_scalar(f"splade/{k}", v, step + 1)
            writer.add_scalar("splade/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % sp["save_every"] == 0:
            _save_splade(model, optimizer, scheduler, step + 1, out_dir / f"step_{step+1}.pt")

    _save_splade(model, optimizer, scheduler, sp["max_steps"], out_dir / "splade_final.pt")
    print(f"SPLADE finetuning complete. Saved to {out_dir / 'splade_final.pt'}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_hidden_size(hf_id: str) -> int:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_id)
    size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
    if size is None:
        raise ValueError(f"Cannot determine hidden_size from {hf_id} config")
    print(f"Backbone hidden_size: {size}")
    return size


def _init_sae_normalisation(model, tokenizer, sc: dict, device):
    """Run a sample of documents through the backbone to initialise SAE bias."""
    from datasets import load_dataset
    import torch

    vecs = []
    ds = load_dataset(sc["corpus_dataset"], split="train", streaming=True)
    for item in ds:
        text = item.get(sc["corpus_text_field"]) or item.get("passage") or item.get("contents", "")
        if not text:
            continue
        enc = tokenizer(text, max_length=sc["doc_max_length"], truncation=True, return_tensors="pt")
        with torch.no_grad():
            tokens = model.get_token_vecs(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        vecs.append(tokens.cpu())
        if sum(v.shape[0] for v in vecs) >= 8192:
            break

    all_vecs = torch.cat(vecs, dim=0)[:8192]
    model.sae.init_normalisation(all_vecs.to(device))
    print(f"SAE normalisation: mean_norm={model.sae.mean_norm.item():.4f}")


def _save_sae(sae, optimizer, scheduler, step, path):
    torch.save({
        "sae": sae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, path)
    print(f"  Saved SAE checkpoint → {path}")


def _save_splade(model, optimizer, scheduler, step, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, path)
    print(f"  Saved SPLADE checkpoint → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SAE-SPLADE with ettin-17m")
    parser.add_argument("stage", choices=["sae", "splade"], help="Training stage to run")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage == "sae":
        train_sae(cfg, resume=args.resume)
    else:
        # Expose SAE config to SPLADE training for corpus loading
        cfg["splade"]["_sae_cfg"] = cfg["sae"]
        train_splade(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
