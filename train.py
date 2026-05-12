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
    out_dir = _model_ckpt_root(mc["hf_id"]) / sc["output_dir"]
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
    from eval import evaluate_nanobeir
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
    ckpt_path = _model_ckpt_root(mc["hf_id"]) / sp["sae_checkpoint"]
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
    out_dir = _model_ckpt_root(mc["hf_id"]) / sp["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    batches = data_iter()

    # ── Initial evaluation (step 0) ───────────────────────────────────
    if cfg.get("eval", {}).get("datasets"):
        print("[SPLADE] Initial eval …")
        evaluate_nanobeir(model, tokenizer, cfg, device, writer=writer, step=0)

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
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        model.sae.remove_parallel_gradient()
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
            if cfg.get("eval", {}).get("datasets"):
                print(f"[SPLADE] Eval at step {step+1} …")
                evaluate_nanobeir(model, tokenizer, cfg, device, writer=writer, step=step + 1)

    _save_splade(model, optimizer, scheduler, sp["max_steps"], out_dir / "splade_final.pt")
    print(f"SPLADE finetuning complete. Saved to {out_dir / 'splade_final.pt'}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Asymmetric finetuning
# ──────────────────────────────────────────────────────────────────────────────

def train_asymmetric(cfg: dict, resume: str | None = None):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter

    from model import TopKSAE, SAESPLADEModel, FrozenDocSPLADE, asymmetric_splade_loss
    from eval import evaluate_asymmetric
    from data import (
        TevatronMSMARCODataset,
        ColBERTDistillationDataset,
        build_corpus_lookup,
        build_query_lookup,
        collate_asymmetric_batch,
    )

    mc = cfg["model"]
    ac = cfg["asymmetric"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"[ASYM] Config: lambda_q={ac['lambda_q']} | flops_warmup_steps={ac['flops_warmup_steps']} | "
        f"k={mc['k']} | k_init={ac.get('k_init', mc['k'])} | k_anneal_steps={ac.get('k_anneal_steps', 1)} | "
        f"lambda_d={ac.get('lambda_d', 0.0)}"
    )

    # ── Frozen doc encoder ────────────────────────────────────────────
    print(f"Loading frozen doc SPLADE: {ac['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(ac["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()
    vocab_size = doc_splade.vocab_size
    print(f"Doc SPLADE vocab_size (= query SAE width): {vocab_size}")

    # ── Query SAE-SPLADE (sae_width must equal vocab_size) ────────────
    query_tokenizer = AutoTokenizer.from_pretrained(mc["hf_id"])
    hidden_size = _get_hidden_size(mc["hf_id"])
    sae = TopKSAE(
        hidden_size=hidden_size,
        sae_width=vocab_size,
        k=mc["k"],
        aux_k=mc["aux_k"],
        dead_steps_threshold=mc["dead_steps_threshold"],
        normalize_input=mc["normalize_input"],
    )
    ckpt_path = _model_ckpt_root(mc["hf_id"]) / ac["sae_checkpoint"]
    print(f"Loading SAE checkpoint from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sae.load_state_dict(ckpt["sae"])

    query_model = SAESPLADEModel(mc["hf_id"], sae, scale=True)
    query_model.to(device)

    # ── Dataset ────────────────────────────────────────────────────────
    if ac.get("distil_data_path"):
        corpus = build_corpus_lookup(cfg["sae"]["corpus_dataset"], cfg["sae"].get("corpus_text_field", "text"))
        queries = build_query_lookup(ac.get("queries_dataset", "Tevatron/msmarco-passage"))
        dataset: object = ColBERTDistillationDataset(
            ac["distil_data_path"], corpus, queries, nway=ac["nway"]
        )
    else:
        dataset = TevatronMSMARCODataset(nway=ac["nway"])

    def data_iter():
        buf: list = []
        while True:
            for item in dataset:
                buf.append(item)
                if len(buf) == ac["batch_size"]:
                    yield collate_asymmetric_batch(
                        buf, query_tokenizer, ac["query_max_length"], device
                    )
                    buf = []

    # ── Optimiser (query_model only — doc_splade is frozen) ───────────
    optimizer = torch.optim.AdamW(
        query_model.parameters(), lr=ac["lr"], weight_decay=ac["weight_decay"], eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, ac["warmup_steps"], ac["max_steps"]
    )
    scaler = GradScaler(enabled=ac["fp16"] and device.type == "cuda")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    # ── Setup ─────────────────────────────────────────────────────────
    out_dir = _model_ckpt_root(mc["hf_id"]) / ac["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    batches = data_iter()

    # ── k-annealing schedule ──────────────────────────────────────────
    k_target = mc["k"]
    k_init = ac.get("k_init", k_target)
    k_anneal_steps = ac.get("k_anneal_steps", 1)

    def get_k(step: int) -> int:
        if k_init <= k_target:
            return k_target
        frac = min(1.0, step / k_anneal_steps)
        return max(k_target, round(k_init + (k_target - k_init) * frac))

    # ── Initial evaluation ────────────────────────────────────────────
    if cfg.get("eval", {}).get("datasets"):
        print("[ASYM] Initial eval — doc_doc ceiling + query_doc baseline …")
        evaluate_asymmetric(
            query_model, query_tokenizer, doc_splade, cfg, device,
            writer=writer, step=0, run_doc_doc=True, override_k=get_k(0),
        )

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    for step in range(start_step, ac["max_steps"]):
        flops_scale = min(1.0, step / max(ac["flops_warmup_steps"], 1))
        override_k = get_k(step)

        q_ids, q_mask, doc_texts, teacher_scores = next(batches)

        # Encode documents with frozen doc SPLADE (no grad, own tokenizer)
        doc_vecs = doc_splade.encode(doc_texts, ac["doc_max_length"])  # [B*nway, vocab_size]

        optimizer.zero_grad()
        with autocast(enabled=ac["fp16"] and device.type == "cuda"):
            loss, metrics = asymmetric_splade_loss(
                query_model, q_ids, q_mask, doc_vecs, teacher_scores,
                lambda_q=ac["lambda_q"],
                flops_scale=flops_scale,
                override_k=override_k,
            )

        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        query_model.sae.remove_parallel_gradient()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        query_model.sae.post_step()

        if (step + 1) % ac["log_every"] == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"[ASYM] step {step+1:>7} | "
                f"loss {metrics['loss']:.4f} | retr {metrics['retr']:.4f} | "
                f"flops {metrics['flops']:.4f} | "
                f"q_nnz {metrics['avg_q_nnz']:.1f} | k {override_k} | "
                f"lr {lr:.2e} | {elapsed:.0f}s"
            )
            for k, v in metrics.items():
                writer.add_scalar(f"asymmetric/{k}", v, step + 1)
            writer.add_scalar("asymmetric/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % ac["save_every"] == 0:
            _save_splade(query_model, optimizer, scheduler, step + 1, out_dir / f"step_{step+1}.pt")
            if cfg.get("eval", {}).get("datasets"):
                print(f"[ASYM] Eval at step {step+1} …")
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False,
                    override_k=get_k(step + 1),
                )

    _save_splade(query_model, optimizer, scheduler, ac["max_steps"], out_dir / "asymmetric_final.pt")
    print(f"Asymmetric training complete. Saved to {out_dir / 'asymmetric_final.pt'}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Projected query encoder (small backbone + projection + frozen MLM head)
# ──────────────────────────────────────────────────────────────────────────────

def train_projected(cfg, resume=None):
    """Train a small query encoder that projects into a frozen doc-SPLADE's vocabulary space.

    Two phases:
      1. Alignment warm-up: train only the projection layer with MSE loss against
         the doc SPLADE output on the same corpus texts.  Establishes a good
         initialisation before ranking fine-tuning.
      2. Ranking fine-tuning: KL distillation + FLOPs regularisation with the full
         backbone + projection trainable and the doc SPLADE frozen.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from model import (
        FrozenDocSPLADE,
        ProjectedQuerySPLADE,
        asymmetric_splade_loss,
        projected_alignment_loss,
    )
    from eval import evaluate_asymmetric
    from data import (
        ColBERTDistillationDataset,
        TevatronMSMARCODataset,
        build_corpus_lookup,
        build_query_lookup,
        collate_asymmetric_batch,
        make_alignment_loader,
    )

    pc = cfg["projected"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"[PROJ] query_backbone={pc['query_hf_id']} | "
        f"doc_splade={pc['doc_splade_hf_id']} | "
        f"alignment_steps={pc.get('alignment_steps', 0)} | "
        f"lambda_q={pc['lambda_q']} | flops_warmup={pc.get('flops_warmup_steps', 1)}"
    )

    # ── Models ────────────────────────────────────────────────────────
    print(f"Loading frozen doc SPLADE: {pc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(pc["doc_splade_hf_id"])
    doc_splade.to(device)

    print(f"Building projected query encoder …")
    query_model = ProjectedQuerySPLADE(pc["query_hf_id"], pc["doc_splade_hf_id"])
    query_model.to(device)

    query_tokenizer = AutoTokenizer.from_pretrained(pc["query_hf_id"])
    proj_size = sum(p.numel() for p in query_model.proj.parameters())
    backbone_size = sum(p.numel() for p in query_model.backbone.parameters())
    print(f"  backbone params: {backbone_size:,} | projection params: {proj_size:,}")

    # ── Phase 1: Alignment warm-up ────────────────────────────────────
    alignment_steps = pc.get("alignment_steps", 0)
    if alignment_steps > 0:
        print(f"[PROJ] Phase 1 — alignment warm-up ({alignment_steps} steps) …")
        align_opt = torch.optim.AdamW(
            query_model.proj.parameters(),
            lr=pc.get("alignment_lr", 1e-3),
        )
        align_loader = make_alignment_loader(
            cfg["sae"]["corpus_dataset"],
            cfg["sae"].get("corpus_text_field", "text"),
            query_tokenizer,
            pc.get("batch_size", 8),
            pc["doc_max_length"],
            device,
        )

        query_model.backbone.requires_grad_(False)
        query_model.train()
        t0 = time.time()

        for step in range(alignment_steps):
            q_ids, q_mask, texts = next(align_loader)
            loss = projected_alignment_loss(query_model, doc_splade, q_ids, q_mask, texts, pc["doc_max_length"])
            align_opt.zero_grad()
            loss.backward()
            align_opt.step()

            if (step + 1) % pc.get("log_every", 200) == 0:
                print(f"[PROJ][ALIGN] step {step+1:>6} | cosine_loss {loss.item():.4f} | {time.time()-t0:.0f}s")
                t0 = time.time()

        query_model.backbone.requires_grad_(True)
        print("[PROJ] Alignment warm-up complete.")

    # ── Dataset ───────────────────────────────────────────────────────
    if pc.get("distil_data_path"):
        corpus = build_corpus_lookup(cfg["sae"]["corpus_dataset"], cfg["sae"].get("corpus_text_field", "text"))
        queries = build_query_lookup(pc.get("queries_dataset", "Tevatron/msmarco-passage"))
        dataset: object = ColBERTDistillationDataset(pc["distil_data_path"], corpus, queries, nway=pc["nway"])
    else:
        dataset = TevatronMSMARCODataset(nway=pc["nway"])

    def data_iter():
        buf: list = []
        while True:
            for item in dataset:
                buf.append(item)
                if len(buf) == pc["batch_size"]:
                    yield collate_asymmetric_batch(buf, query_tokenizer, pc["query_max_length"], device)
                    buf = []

    # ── Optimiser (backbone + projection; MLM head is frozen) ─────────
    optimizer = torch.optim.AdamW(
        [p for p in query_model.parameters() if p.requires_grad],
        lr=pc["lr"], weight_decay=pc["weight_decay"], eps=1e-8,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, pc["warmup_steps"], pc["max_steps"])
    scaler = GradScaler(enabled=pc["fp16"] and device.type == "cuda")

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    out_dir = _model_ckpt_root(pc["query_hf_id"]) / pc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    batches = data_iter()

    # ── Initial eval (doc-doc ceiling + query-doc baseline) ───────────
    if cfg.get("eval", {}).get("datasets"):
        print("[PROJ] Initial eval — doc_doc ceiling + query_doc baseline …")
        evaluate_asymmetric(
            query_model, query_tokenizer, doc_splade, cfg, device,
            writer=writer, step=0, run_doc_doc=True, override_k=0,
            section="projected",
        )

    # ── Phase 2: Ranking fine-tuning ─────────────────────────────────
    print(f"[PROJ] Phase 2 — ranking fine-tuning ({pc['max_steps']} steps) …")
    query_model.train()
    t0 = time.time()

    for step in range(start_step, pc["max_steps"]):
        flops_scale = min(1.0, step / max(pc.get("flops_warmup_steps", 1), 1))

        q_ids, q_mask, doc_texts, teacher_scores = next(batches)
        doc_vecs = doc_splade.encode(doc_texts, pc["doc_max_length"])

        optimizer.zero_grad()
        with autocast(enabled=pc["fp16"] and device.type == "cuda"):
            loss, metrics = asymmetric_splade_loss(
                query_model, q_ids, q_mask, doc_vecs, teacher_scores,
                lambda_q=pc["lambda_q"],
                flops_scale=flops_scale,
                override_k=0,
            )

        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(query_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % pc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"[PROJ] step {step+1:>7} | "
                f"loss {metrics['loss']:.4f} | retr {metrics['retr']:.4f} | "
                f"flops {metrics['flops']:.4f} | q_nnz {metrics['avg_q_nnz']:.1f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s"
            )
            for k, v in metrics.items():
                writer.add_scalar(f"projected/{k}", v, step + 1)
            writer.add_scalar("projected/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % pc["save_every"] == 0:
            _save_splade(query_model, optimizer, scheduler, step + 1, out_dir / f"step_{step+1}.pt")
            if cfg.get("eval", {}).get("datasets"):
                print(f"[PROJ] Eval at step {step+1} …")
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="projected",
                )

    _save_splade(query_model, optimizer, scheduler, pc["max_steps"], out_dir / "projected_final.pt")
    print(f"Projected training complete. Saved to {out_dir / 'projected_final.pt'}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: Vocab-transplant query encoder
# ──────────────────────────────────────────────────────────────────────────────

def _run_tokensurgeon(
    query_hf_id: str,
    donor_hf_id: str,
    out_path: str,
    k: int = 64,
):
    """Transplant the donor's tokenizer and vocabulary onto the query MLM model.

    For each token in the donor vocabulary:
    - Exact match in query vocabulary → copy query embedding directly.
    - No match → approximate using cosine-NN interpolation: find the k nearest
      neighbours of the donor token embedding among the shared tokens (in donor
      embedding space), then apply those neighbour weights to the corresponding
      query embeddings.  This preserves the relational geometry of the donor
      vocabulary in the query model's hidden space.

    The transplanted model is saved to ``out_path`` together with the donor
    tokenizer so it can be loaded as AutoModelForMaskedLM.
    """
    import torch.nn.functional as F
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"[TRANSPLANT] {query_hf_id}  ←vocab—  {donor_hf_id}")
    print(f"[TRANSPLANT] Loading tokenizers …")
    query_tok = AutoTokenizer.from_pretrained(query_hf_id)
    donor_tok = AutoTokenizer.from_pretrained(donor_hf_id)
    query_vocab: dict = query_tok.get_vocab()   # str → int
    donor_vocab: dict = donor_tok.get_vocab()   # str → int
    donor_vocab_size = len(donor_vocab)

    print(f"[TRANSPLANT] Query vocab: {len(query_vocab):,} | Donor vocab: {donor_vocab_size:,}")
    print(f"[TRANSPLANT] Loading models …")
    query_model = AutoModelForMaskedLM.from_pretrained(query_hf_id)
    donor_model = AutoModelForMaskedLM.from_pretrained(donor_hf_id)

    orig_embed  = query_model.get_input_embeddings().weight.data.float()  # [Q_V, H_q]
    donor_embed = donor_model.get_input_embeddings().weight.data.float()  # [D_V, H_d]
    del donor_model  # free memory
    H_q = orig_embed.shape[1]

    # ── Identify shared tokens (exact string match) ───────────────────
    shared_tokens = list(set(query_vocab) & set(donor_vocab))
    print(f"[TRANSPLANT] Shared tokens: {len(shared_tokens):,} / {donor_vocab_size:,}  "
          f"({100*len(shared_tokens)/donor_vocab_size:.1f}%)")

    shared_q_idx = torch.tensor([query_vocab[t] for t in shared_tokens])
    shared_d_idx = torch.tensor([donor_vocab[t] for t in shared_tokens])
    query_shared  = orig_embed[shared_q_idx]                         # [|S|, H_q]
    donor_shared  = donor_embed[shared_d_idx]                        # [|S|, H_d]
    donor_shared_norm = F.normalize(donor_shared, dim=-1)            # [|S|, H_d]

    # ── Build new embedding matrix [D_V, H_q] ────────────────────────
    new_embed = torch.zeros(donor_vocab_size, H_q, dtype=orig_embed.dtype)
    for t in shared_tokens:
        new_embed[donor_vocab[t]] = orig_embed[query_vocab[t]]

    # ── Approximate non-shared tokens ────────────────────────────────
    non_shared = [t for t in donor_vocab if t not in set(shared_tokens)]
    if non_shared:
        print(f"[TRANSPLANT] Approximating {len(non_shared):,} non-shared tokens (k={k}) …")
        actual_k = min(k, len(shared_tokens))
        ns_d_idx  = torch.tensor([donor_vocab[t] for t in non_shared])
        targets   = donor_embed[ns_d_idx]                            # [N, H_d]
        targets_norm = F.normalize(targets, dim=-1)

        chunk_size = 512
        for i in range(0, len(non_shared), chunk_size):
            chunk_norm = targets_norm[i : i + chunk_size]            # [C, H_d]
            sims = chunk_norm @ donor_shared_norm.T                  # [C, |S|]
            topk_vals, topk_idx = sims.topk(actual_k, dim=-1)       # [C, k]
            # Distance-proportional weights: w ∝ 1/(1 − sim + ε)
            weights = 1.0 / (1.0 - topk_vals.clamp(max=0.9999) + 1e-6)
            weights = weights / weights.sum(dim=-1, keepdim=True)    # [C, k]
            # Interpolate in query space
            approx = (weights.unsqueeze(-1) * query_shared[topk_idx]).sum(dim=1)  # [C, H_q]
            for j, t in enumerate(non_shared[i : i + chunk_size]):
                new_embed[donor_vocab[t]] = approx[j]

    # ── Resize model and install new embedding ────────────────────────
    query_model.resize_token_embeddings(donor_vocab_size)
    query_model.get_input_embeddings().weight.data.copy_(new_embed)

    # ── Align special-token IDs with the donor tokenizer ─────────────
    # The query model's config still holds ettin's original special-token IDs,
    # which may be >= donor_vocab_size and would cause nn.Embedding to crash
    # on reload.  Replace them with the donor's values.
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id",
                 "unk_token_id", "mask_token_id"):
        setattr(query_model.config, attr, getattr(donor_tok, attr, None))

    # ── Save transplanted model + donor tokenizer ─────────────────────
    Path(out_path).mkdir(parents=True, exist_ok=True)
    query_model.save_pretrained(out_path)
    donor_tok.save_pretrained(out_path)
    print(f"[TRANSPLANT] Saved transplanted model → {out_path}")


def _run_tokensurgeon_lion(
    query_hf_id: str,
    lion_hf_id: str,
    out_path: str,
    k: int = 64,
):
    """Vocab transplant from a Lion-SP LoRA adapter (Llama-3) onto a small MLM.

    Same algorithm as ``_run_tokensurgeon`` but loads the donor model via the
    PEFT library (LoRA adapter on LlamaForCausalLM) instead of AutoModelForMaskedLM.
    Only the embedding table is extracted from the donor; no forward pass is run.
    """
    import json
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer, LlamaForCausalLM
    from peft import PeftModel, LoraConfig

    print(f"[TRANSPLANT-LION] {query_hf_id}  ←vocab—  {lion_hf_id}")

    # ── Load tokenizers ───────────────────────────────────────────────
    print("[TRANSPLANT-LION] Loading tokenizers …")
    query_tok = AutoTokenizer.from_pretrained(query_hf_id)
    donor_tok = AutoTokenizer.from_pretrained(lion_hf_id)
    if donor_tok.pad_token_id is None:
        donor_tok.pad_token_id = donor_tok.eos_token_id
    query_vocab: dict = query_tok.get_vocab()
    donor_vocab: dict = donor_tok.get_vocab()
    donor_vocab_size = len(donor_vocab)
    print(f"[TRANSPLANT-LION] Query vocab: {len(query_vocab):,} | Donor vocab: {donor_vocab_size:,}")

    # ── Load query model (BERT-style MLM) ────────────────────────────
    print("[TRANSPLANT-LION] Loading query model …")
    query_model = AutoModelForMaskedLM.from_pretrained(query_hf_id)
    orig_embed = query_model.get_input_embeddings().weight.data.float()  # [Q_V, H_q]
    H_q = orig_embed.shape[1]

    # ── Extract Lion donor embeddings (no forward pass needed) ────────
    print("[TRANSPLANT-LION] Loading Lion donor model to extract embeddings …")
    adapter_cfg_path = hf_hub_download(lion_hf_id, "adapter_config.json")
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    base_model_path = adapter_cfg["base_model_name_or_path"]
    print(f"[TRANSPLANT-LION] Lion base model: {base_model_path}")

    base = LlamaForCausalLM.from_pretrained(base_model_path)
    lora_cfg = LoraConfig.from_pretrained(lion_hf_id)
    peft_model = PeftModel.from_pretrained(base, lion_hf_id, config=lora_cfg, is_trainable=False)
    merged = peft_model.merge_and_unload()
    donor_embed = merged.get_input_embeddings().weight.data.float().cpu()  # [128K, H_d]
    del merged, peft_model, base
    print(f"[TRANSPLANT-LION] Donor embedding table: {donor_embed.shape}")

    # ── Identify shared tokens (exact string match) ───────────────────
    shared_tokens = list(set(query_vocab) & set(donor_vocab))
    print(f"[TRANSPLANT-LION] Shared tokens: {len(shared_tokens):,} / {donor_vocab_size:,}  "
          f"({100*len(shared_tokens)/donor_vocab_size:.1f}%)")

    shared_q_idx = torch.tensor([query_vocab[t] for t in shared_tokens])
    shared_d_idx = torch.tensor([donor_vocab[t] for t in shared_tokens])
    query_shared = orig_embed[shared_q_idx]                          # [|S|, H_q]
    donor_shared = donor_embed[shared_d_idx]                         # [|S|, H_d]
    donor_shared_norm = F.normalize(donor_shared, dim=-1)

    # ── Build new embedding matrix [D_V, H_q] ────────────────────────
    new_embed = torch.zeros(donor_vocab_size, H_q, dtype=orig_embed.dtype)
    for t in shared_tokens:
        new_embed[donor_vocab[t]] = orig_embed[query_vocab[t]]

    # ── Approximate non-shared tokens via kNN in donor space ─────────
    non_shared = [t for t in donor_vocab if t not in set(shared_tokens)]
    if non_shared:
        print(f"[TRANSPLANT-LION] Approximating {len(non_shared):,} non-shared tokens (k={k}) …")
        actual_k = min(k, len(shared_tokens))
        ns_d_idx = torch.tensor([donor_vocab[t] for t in non_shared])
        targets = donor_embed[ns_d_idx]
        targets_norm = F.normalize(targets, dim=-1)

        chunk_size = 512
        for i in range(0, len(non_shared), chunk_size):
            chunk_norm = targets_norm[i : i + chunk_size]
            sims = chunk_norm @ donor_shared_norm.T
            topk_vals, topk_idx = sims.topk(actual_k, dim=-1)
            weights = 1.0 / (1.0 - topk_vals.clamp(max=0.9999) + 1e-6)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            approx = (weights.unsqueeze(-1) * query_shared[topk_idx]).sum(dim=1)
            for j, t in enumerate(non_shared[i : i + chunk_size]):
                new_embed[donor_vocab[t]] = approx[j]

    # ── Resize model and install new embedding ────────────────────────
    query_model.resize_token_embeddings(donor_vocab_size)
    query_model.get_input_embeddings().weight.data.copy_(new_embed)

    # ── Align special-token IDs with donor ────────────────────────────
    # The tokenizer (saved alongside) carries the donor's correct IDs. The
    # model config's special-token IDs are only needed for forward-time
    # behaviour (padding_idx in some embedding layers, ignore_index in some
    # losses). Donor IDs are typically very large (Llama-3 pad=128001) and
    # exceed embedding ranges in some backbones — RoBERTa's position
    # embeddings have only `max_position_embeddings≈514` rows but use
    # `padding_idx=config.pad_token_id`, so a large pad_token_id breaks
    # model construction at load time. Skip any ID that's >= the new vocab
    # size or would obviously overflow positional padding.
    max_pos = getattr(query_model.config, "max_position_embeddings", None)
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id",
                 "unk_token_id", "mask_token_id"):
        new_id = getattr(donor_tok, attr, None)
        if new_id is None:
            continue
        # Skip if it would crash an embedding layer that uses this id as
        # padding_idx (RoBERTa does this with pad_token_id specifically).
        if attr == "pad_token_id" and max_pos is not None and new_id >= max_pos:
            print(
                f"[TRANSPLANT-LION] Keeping original pad_token_id "
                f"({getattr(query_model.config, 'pad_token_id', None)}); donor "
                f"value {new_id} would exceed max_position_embeddings={max_pos} "
                "and break position-embedding construction."
            )
            continue
        setattr(query_model.config, attr, new_id)

    # ── Save ──────────────────────────────────────────────────────────
    Path(out_path).mkdir(parents=True, exist_ok=True)
    query_model.save_pretrained(out_path)
    donor_tok.save_pretrained(out_path)
    print(f"[TRANSPLANT-LION] Saved transplanted model → {out_path}")


def train_vocab_transplant(cfg: dict, resume: str | None = None):
    """Train a vocab-transplanted small MLM encoder as a SPLADE query model.

    Step 1 — Vocab transplant (skipped if transplant_dir already populated):
      Runs tokensurgeon-style embedding transfer to give the query encoder
      the doc SPLADE's tokenizer and vocabulary.

    Step 2 — Ranking fine-tuning:
      Distils from ColBERTv2 teacher scores (or CE loss) against the frozen
      doc SPLADE.  The full transplanted model is trainable (body + MLM head).
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE, vocab_transplant_joint_loss, projected_alignment_loss
    from eval import evaluate_asymmetric
    from data import (
        ColBERTDistillationDataset,
        TevatronMSMARCODataset,
        build_corpus_lookup,
        build_query_lookup,
        collate_asymmetric_batch,
        make_alignment_loader,
    )

    vc = cfg["vocab_transplant"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"[VT] query={vc['query_hf_id']} | doc={vc['doc_splade_hf_id']} | "
        f"lambda_q={vc['lambda_q']} | flops_warmup={vc.get('flops_warmup_steps', 1)}"
    )

    # ── Step 1: vocab transplant ──────────────────────────────────────
    transplant_dir = str(_model_ckpt_root(vc["query_hf_id"]) / vc["transplant_dir"])
    if not Path(transplant_dir, "config.json").exists():
        _run_tokensurgeon(
            query_hf_id=vc["query_hf_id"],
            donor_hf_id=vc["doc_splade_hf_id"],
            out_path=transplant_dir,
            k=vc.get("tokensurgeon_k", 64),
        )
    else:
        print(f"[VT] Transplanted model already exists at {transplant_dir}, skipping.")

    # ── Step 2: load models ───────────────────────────────────────────
    print(f"[VT] Loading frozen doc SPLADE: {vc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(vc["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()

    # ── Early LoRA validation — fail before training starts ───────────
    if vc.get("lora_doc_encoder", False):
        import torch.nn as _nn
        target_mods = vc.get("lora_target_modules", ["query", "value"])
        all_module_names = [name for name, _ in doc_splade.mlm.named_modules()]
        missing = [t for t in target_mods if not any(n.endswith(t) for n in all_module_names)]
        if missing:
            linear_leaves = sorted(
                name.split(".")[-1]
                for name, mod in doc_splade.mlm.named_modules()
                if isinstance(mod, _nn.Linear)
            )
            raise ValueError(
                f"[VT] lora_target_modules {missing} not found in {vc['doc_splade_hf_id']}.\n"
                f"Available Linear leaf names: {sorted(set(linear_leaves))}\n"
                f"Update lora_target_modules in config.yaml."
            )
        print(f"[VT] LoRA target modules validated: {target_mods}")

    print(f"[VT] Loading transplanted query model from {transplant_dir} …")
    query_model = VocabTransplantQuerySPLADE(transplant_dir)
    query_model.to(device)

    # The transplanted model uses the donor's tokenizer
    query_tokenizer = AutoTokenizer.from_pretrained(transplant_dir)
    n_params = sum(p.numel() for p in query_model.parameters())
    print(f"[VT] Query model params: {n_params:,} | vocab_size: {query_model.vocab_size}")

    # ── Dataset ───────────────────────────────────────────────────────
    if vc.get("distil_data_path"):
        corpus = build_corpus_lookup(
            cfg["sae"]["corpus_dataset"], cfg["sae"].get("corpus_text_field", "text")
        )
        queries = build_query_lookup(vc.get("queries_dataset", "Tevatron/msmarco-passage"))
        dataset: object = ColBERTDistillationDataset(
            vc["distil_data_path"], corpus, queries, nway=vc["nway"]
        )
    else:
        dataset = TevatronMSMARCODataset(nway=vc["nway"])

    def data_iter():
        buf: list = []
        while True:
            for item in dataset:
                buf.append(item)
                if len(buf) == vc["batch_size"]:
                    q_texts = [it["query"] for it in buf]
                    q_ids, q_mask, doc_texts, _ = collate_asymmetric_batch(
                        buf, query_tokenizer, vc["query_max_length"], device
                    )
                    yield q_ids, q_mask, q_texts, doc_texts
                    buf = []

    # ── Optimiser (full model trainable) ──────────────────────────────
    optimizer = torch.optim.AdamW(
        query_model.parameters(), lr=vc["lr"], weight_decay=vc["weight_decay"], eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, vc["warmup_steps"], vc["max_steps"])
    scaler = GradScaler(enabled=vc["fp16"] and device.type == "cuda")

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    out_dir = _model_ckpt_root(vc["query_hf_id"]) / vc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    batches = data_iter()

    # ── Phase 1: cosine alignment warmup ──────────────────────────────
    alignment_steps = vc.get("alignment_steps", 0)
    if alignment_steps > 0 and start_step == 0:
        print(f"[VT] Phase 1: Cosine alignment warm-up ({alignment_steps} steps) …")
        align_optimizer = torch.optim.AdamW(
            query_model.parameters(), lr=vc.get("alignment_lr", 5e-4), weight_decay=vc["weight_decay"]
        )
        align_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            align_optimizer, T_max=alignment_steps, eta_min=1e-5
        )
        # Use training queries (not corpus passages) so alignment targets the actual
        # query distribution the model will see at inference time.
        align_loader = make_alignment_loader(
            vc.get("queries_dataset", "Tevatron/msmarco-passage"),
            "query",
            query_tokenizer,
            vc["batch_size"],
            vc["query_max_length"],
            device,
        )
        query_model.train()
        t0 = time.time()
        import torch.nn.functional as _F
        for astep in range(alignment_steps):
            a_ids, a_mask, texts = next(align_loader)
            align_optimizer.zero_grad()
            with autocast(enabled=vc["fp16"] and device.type == "cuda"):
                q_vecs = query_model.encode(a_ids, a_mask)
                d_vecs = doc_splade.encode(texts, vc["query_max_length"], no_grad=True)
                align_loss = (1.0 - _F.cosine_similarity(q_vecs, d_vecs.to(q_vecs.dtype))).mean()
            q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
            scaler.scale(align_loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(align_optimizer)
            torch.nn.utils.clip_grad_norm_(list(query_model.parameters()), 1.0)
            scaler.step(align_optimizer)
            scaler.update()
            align_scheduler.step()

            if (astep + 1) % vc["log_every"] == 0:
                elapsed = time.time() - t0
                align_lr = align_optimizer.param_groups[0]["lr"]
                print(
                    f"[VT-align] step {astep+1:>6} | align_loss {align_loss.item():.4f} | "
                    f"q_nnz {q_nnz:.1f} | lr {align_lr:.2e} | {elapsed:.0f}s"
                )
                writer.add_scalar("vocab_transplant/align_loss", align_loss.item(), astep + 1)
                writer.add_scalar("vocab_transplant/align_q_nnz", q_nnz, astep + 1)
                writer.add_scalar("vocab_transplant/align_lr", align_lr, astep + 1)
                t0 = time.time()

            if (astep + 1) % vc["save_every"] == 0:
                align_ckpt = out_dir / f"align_step_{astep+1}.pt"
                torch.save({"model": query_model.state_dict(), "step": astep + 1}, align_ckpt)
                print(f"  Saved alignment checkpoint → {align_ckpt}")
                if cfg.get("eval", {}).get("datasets"):
                    print(f"[VT-align] Eval at alignment step {astep+1} …")
                    query_model.eval()
                    evaluate_asymmetric(
                        query_model, query_tokenizer, doc_splade, cfg, device,
                        writer=writer, step=astep + 1, run_doc_doc=False, override_k=0,
                        section="vocab_transplant",
                    )
                    query_model.train()

        del align_optimizer, align_loader
        print("[VT] Phase 1 complete.")

    # ── Initial eval ──────────────────────────────────────────────────
    if cfg.get("eval", {}).get("datasets"):
        print("[VT] Initial eval …")
        evaluate_asymmetric(
            query_model, query_tokenizer, doc_splade, cfg, device,
            writer=writer, step=0, run_doc_doc=True, override_k=0,
            section="vocab_transplant",
        )

    # ── Training loop ─────────────────────────────────────────────────
    print(f"[VT] Ranking fine-tuning ({vc['max_steps']} steps) …")
    query_model.train()
    t0 = time.time()
    lora_enabled = False
    lora_optimizer = None

    for step in range(start_step, vc["max_steps"]):
        # ── LoRA activation: enable doc encoder LoRA after warmup ─────
        if (
            vc.get("lora_doc_encoder", False)
            and not lora_enabled
            and step >= vc.get("lora_warmup_steps", 10_000)
        ):
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=vc.get("lora_r", 8),
                lora_alpha=vc.get("lora_alpha", 16),
                lora_dropout=vc.get("lora_dropout", 0.1),
                target_modules=vc.get("lora_target_modules", ["query", "value"]),
                bias="none",
            )
            doc_splade.mlm = get_peft_model(doc_splade.mlm, lora_cfg)
            doc_splade.mlm.print_trainable_parameters()
            lora_params = [p for p in doc_splade.parameters() if p.requires_grad]
            # Separate optimizer for LoRA — constant LR, no scheduler.
            # Adding params to the existing optimizer would break the scheduler
            # (it was initialised for a fixed number of param groups).
            lora_optimizer = torch.optim.AdamW(
                lora_params, lr=vc.get("lora_lr", vc["lr"]), weight_decay=vc["weight_decay"]
            )
            doc_splade.train()
            lora_enabled = True
            print(f"[VT] LoRA enabled on doc encoder at step {step}")

        flops_scale = min(1.0, step / max(vc.get("flops_warmup_steps", 1), 1))

        q_ids, q_mask, q_texts, doc_texts = next(batches)

        # Doc encoding always runs in fp32 — BERT is not fp16-stable under gradients.
        # Gradient flows when LoRA is active; no_grad otherwise.
        doc_vecs = doc_splade.encode(doc_texts, vc["doc_max_length"], no_grad=not lora_enabled)

        # Run doc encoder on queries for the alignment term (always frozen, fp32).
        teacher_q_vecs = doc_splade.encode(q_texts, vc["query_max_length"], no_grad=True)

        optimizer.zero_grad()
        if lora_enabled:
            lora_optimizer.zero_grad()
        with autocast(enabled=vc["fp16"] and device.type == "cuda"):
            loss, metrics = vocab_transplant_joint_loss(
                query_model, q_ids, q_mask, doc_vecs, teacher_q_vecs,
                lambda_q=vc["lambda_q"],
                flops_scale=flops_scale,
                align_coeff=vc.get("align_coeff", 0.3),
            )

        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            if lora_enabled:
                scaler.unscale_(lora_optimizer)
        trainable = list(query_model.parameters())
        if lora_enabled:
            trainable += lora_params
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        if lora_enabled:
            scaler.step(lora_optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % vc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            lora_tag = " [+LoRA]" if lora_enabled else ""
            print(
                f"[VT{lora_tag}] step {step+1:>7} | "
                f"loss {metrics['loss']:.4f} | rank {metrics['ranking']:.4f} | "
                f"align {metrics['align']:.4f} | flops {metrics['flops']:.4f} | "
                f"q_nnz {metrics['avg_q_nnz']:.1f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            for k_name, v in metrics.items():
                writer.add_scalar(f"vocab_transplant/{k_name}", v, step + 1)
            writer.add_scalar("vocab_transplant/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % vc["save_every"] == 0:
            _save_splade(query_model, optimizer, scheduler, step + 1, out_dir / f"step_{step+1}.pt")
            if lora_enabled:
                lora_path = out_dir / f"lora_doc_step_{step+1}"
                doc_splade.mlm.save_pretrained(str(lora_path))
                print(f"  Saved LoRA adapter → {lora_path}")
            if cfg.get("eval", {}).get("datasets"):
                print(f"[VT] Eval at step {step+1} …")
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="vocab_transplant",
                )

    _save_splade(query_model, optimizer, scheduler, vc["max_steps"], out_dir / "vocab_transplant_final.pt")
    if lora_enabled:
        doc_splade.mlm.save_pretrained(str(out_dir / "lora_doc_final"))
        print(f"  Saved final LoRA adapter → {out_dir / 'lora_doc_final'}")
    print(f"Vocab-transplant training complete. Saved to {out_dir / 'vocab_transplant_final.pt'}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5b: Vocab-transplant — alignment-only
# ──────────────────────────────────────────────────────────────────────────────

def train_vocab_transplant_align(cfg: dict, resume: str | None = None):
    """Cosine-alignment-only variant of vocab_transplant.

    Self-contained — does not share code with ``train_vocab_transplant``.
    Intentionally kept separate so it can diverge freely (alternative losses,
    schedulers, samplers, etc.) without disturbing the original two-phase
    flow.

    Reads from its own ``vocab_transplant_align`` config section, kept
    independent of ``vocab_transplant`` so the two flows can diverge freely.
    Required keys:
      - alignment_steps, alignment_lr, batch_size, query_max_length
      - query_hf_id, doc_splade_hf_id, transplant_dir, output_dir
      - log_every, save_every, fp16, weight_decay
      - queries_dataset (optional, defaults to Tevatron/msmarco-passage)
      - tokensurgeon_k (optional, defaults to 64)

    Saves intermediate ``align_step_N.pt`` checkpoints and a final
    ``align_final.pt`` to ``checkpoints_<query_model>/<output_dir>/``.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer
    import torch.nn.functional as F

    from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE
    from eval import evaluate_asymmetric
    from data import make_alignment_loader

    if "vocab_transplant_align" not in cfg:
        raise SystemExit(
            "[VT-align] config.yaml is missing a `vocab_transplant_align:` "
            "section. Add one (see config.yaml for the template)."
        )
    vc = cfg["vocab_transplant_align"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alignment_steps = vc.get("alignment_steps", 0)
    if alignment_steps <= 0:
        raise SystemExit(
            "[VT-align] alignment_steps must be > 0 in config.yaml "
            "vocab_transplant section."
        )

    print(
        f"[VT-align] query={vc['query_hf_id']} | doc={vc['doc_splade_hf_id']} | "
        f"alignment_steps={alignment_steps} | alignment_lr={vc.get('alignment_lr', 5e-4)}"
    )

    # ── Step 1: vocab transplant (skipped if already populated) ───────
    transplant_dir = str(_model_ckpt_root(vc["query_hf_id"]) / vc["transplant_dir"])
    if not Path(transplant_dir, "config.json").exists():
        _run_tokensurgeon(
            query_hf_id=vc["query_hf_id"],
            donor_hf_id=vc["doc_splade_hf_id"],
            out_path=transplant_dir,
            k=vc.get("tokensurgeon_k", 64),
        )
    else:
        print(f"[VT-align] Transplanted model already exists at {transplant_dir}, skipping.")

    # ── Step 2: load models ───────────────────────────────────────────
    print(f"[VT-align] Loading frozen doc SPLADE: {vc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(vc["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()

    print(f"[VT-align] Loading transplanted query model from {transplant_dir} …")
    query_model = VocabTransplantQuerySPLADE(transplant_dir)
    query_model.to(device)
    query_tokenizer = AutoTokenizer.from_pretrained(transplant_dir)
    n_params = sum(p.numel() for p in query_model.parameters())
    print(f"[VT-align] Query model params: {n_params:,} | vocab_size: {query_model.vocab_size}")

    # ── Resume (model state only; optimizer/scheduler reset) ──────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[VT-align] Resumed model state from {resume} at step {start_step}")

    # ── Optimiser / scheduler / scaler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        query_model.parameters(),
        lr=vc.get("alignment_lr", 5e-4),
        weight_decay=vc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=alignment_steps,
        eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=vc["fp16"] and device.type == "cuda")

    # ── Output dir & loader ───────────────────────────────────────────
    out_dir = _model_ckpt_root(vc["query_hf_id"]) / vc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    align_loader = make_alignment_loader(
        vc.get("queries_dataset", "Tevatron/msmarco-passage"),
        "query",
        query_tokenizer,
        vc["batch_size"],
        vc["query_max_length"],
        device,
    )

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    print(f"[VT-align] Training cosine alignment ({alignment_steps - start_step} steps remaining) …")

    for astep in range(start_step, alignment_steps):
        a_ids, a_mask, texts = next(align_loader)
        optimizer.zero_grad()
        with autocast(enabled=vc["fp16"] and device.type == "cuda"):
            q_vecs = query_model.encode(a_ids, a_mask)
            d_vecs = doc_splade.encode(texts, vc["query_max_length"], no_grad=True)
            align_loss = (1.0 - F.cosine_similarity(q_vecs, d_vecs.to(q_vecs.dtype))).mean()
        q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
        scaler.scale(align_loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(query_model.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (astep + 1) % vc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[VT-align] step {astep+1:>6} | align_loss {align_loss.item():.4f} | "
                f"q_nnz {q_nnz:.1f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar("vocab_transplant_align/loss", align_loss.item(), astep + 1)
            writer.add_scalar("vocab_transplant_align/q_nnz", q_nnz, astep + 1)
            writer.add_scalar("vocab_transplant_align/lr", lr, astep + 1)
            t0 = time.time()

        if (astep + 1) % vc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{astep+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": astep + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            if cfg.get("eval", {}).get("datasets"):
                print(f"[VT-align] Eval at step {astep+1} …")
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=astep + 1, run_doc_doc=False, override_k=0,
                    section="vocab_transplant_align",
                )
                query_model.train()

    # ── Final save ────────────────────────────────────────────────────
    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": alignment_steps}, final_path)
    print(f"[VT-align] Done. Final checkpoint → {final_path}")
    writer.close()


def train_lion_transplant_align(cfg: dict, resume: str | None = None):
    """Vocab transplant + cosine alignment using Lion-SP-1B as the frozen doc encoder.

    Mirrors ``train_vocab_transplant_align`` but uses a Lion-SP LoRA model
    (bidirectional Llama-3, 128K vocab) instead of naver/splade-v3.  Because
    Llama-3 BPE shares very few exact tokens with ettin's BERT wordpiece vocab,
    most donor embeddings are approximated by kNN interpolation, leaving the
    transplanted model in a weaker initial state than the BERT→BERT case.  A
    KD warmup phase is therefore included before cosine alignment.

    Reads from the ``lion_transplant_align`` config section.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer
    import torch.nn.functional as F

    from model import FrozenLionSPLADE, VocabTransplantQuerySPLADE
    from eval import evaluate_asymmetric
    from data import make_alignment_loader

    if "lion_transplant_align" not in cfg:
        raise SystemExit(
            "[lion-align] config.yaml is missing a `lion_transplant_align:` section."
        )
    lc = cfg["lion_transplant_align"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alignment_steps = lc.get("alignment_steps", 0)
    if alignment_steps <= 0:
        raise SystemExit("[lion-align] alignment_steps must be > 0.")

    kd_warmup_steps  = lc.get("kd_warmup_steps", 10_000)
    kd_temperature   = lc.get("kd_temperature", 4.0)
    lambda_q         = lc.get("lambda_q", 0.0)
    flops_warmup     = lc.get("flops_warmup_steps", 0)
    grad_accum       = lc.get("gradient_accumulation_steps", 1)
    total_steps = kd_warmup_steps + alignment_steps

    print(
        f"[lion-align] query={lc['query_hf_id']} | lion={lc['lion_hf_id']} | "
        f"kd_warmup={kd_warmup_steps} | alignment={alignment_steps} | "
        f"lr={lc.get('alignment_lr', 5e-4)} | lambda_q={lambda_q}"
    )

    # ── Step 1: vocab transplant (cached after first run) ─────────────
    transplant_dir = str(_model_ckpt_root(lc["query_hf_id"]) / lc["transplant_dir"])
    if not Path(transplant_dir, "config.json").exists():
        _run_tokensurgeon_lion(
            query_hf_id=lc["query_hf_id"],
            lion_hf_id=lc["lion_hf_id"],
            out_path=transplant_dir,
            k=lc.get("tokensurgeon_k", 64),
        )
    else:
        print(f"[lion-align] Transplanted model already exists at {transplant_dir}, skipping.")

    # ── Step 2: load models ───────────────────────────────────────────
    print(f"[lion-align] Loading frozen Lion-SP doc encoder: {lc['lion_hf_id']} …")
    lion_doc = FrozenLionSPLADE(lc["lion_hf_id"])
    lion_doc.to(device)
    lion_doc.eval()

    print(f"[lion-align] Loading transplanted query model from {transplant_dir} …")
    query_model = VocabTransplantQuerySPLADE(transplant_dir)
    query_model.to(device)
    query_tokenizer = AutoTokenizer.from_pretrained(transplant_dir)
    if query_tokenizer.pad_token_id is None:
        query_tokenizer.pad_token_id = query_tokenizer.eos_token_id
    n_params = sum(p.numel() for p in query_model.parameters())
    print(f"[lion-align] Query params: {n_params:,} | vocab_size: {query_model.vocab_size}")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[lion-align] Resumed from {resume} at step {start_step}")

    # ── Optimiser / scheduler / scaler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        query_model.parameters(),
        lr=lc.get("alignment_lr", 5e-4),
        weight_decay=lc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=lc["fp16"] and device.type == "cuda")

    # ── Output dir & loader ───────────────────────────────────────────
    out_dir = _model_ckpt_root(lc["query_hf_id"]) / lc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    align_loader = make_alignment_loader(
        lc.get("queries_dataset", "Tevatron/msmarco-passage"),
        "query",
        query_tokenizer,
        lc["batch_size"],
        lc["query_max_length"],
        device,
    )

    # ── Calibrate the MLM head bias so initial q_logits ≈ 0 ──────────
    # Different backbones produce wildly different post-transplant q_logit
    # distributions (ettin-150m: mean=+0.40, 59% positive; bert-base: mean=-2.70,
    # 0.87% positive). A heavily negative starting mean puts the model in the
    # dead-relu basin from step 0 — the cosine alignment can't escape because
    # the relu has no gradient on negative logits.
    #
    # Fix: measure mean(q_logits) on a small batch of real training queries,
    # then shift the MLM bias by -mean. This re-centers any backbone to the
    # same healthy starting point. Skipped on resume.
    if start_step == 0 and lc.get("calibrate_bias", True):
        with torch.no_grad():
            n_cal = lc.get("calibrate_batches", 16)
            means = []
            query_model.eval()
            for _ in range(n_cal):
                ids, mask, _ = next(align_loader)
                raw = query_model.mlm(input_ids=ids, attention_mask=mask).logits
                logits = (raw + (1.0 - mask.unsqueeze(-1).float()) * -1e6).max(dim=1).values
                means.append(logits.float().mean().item())
            shift = sum(means) / len(means)
            V = query_model.vocab_size
            bias_params = [
                (n, p) for n, p in query_model.mlm.named_parameters()
                if p.dim() == 1 and p.shape[0] == V
            ]
            if bias_params:
                for name, par in bias_params:
                    par.data -= shift
                print(
                    f"[lion-align] Calibrated MLM bias: shifted by {-shift:+.3f} "
                    f"({len(bias_params)} param(s): {[n for n, _ in bias_params]})"
                )
            else:
                print(
                    "[lion-align] WARNING: no vocab-sized bias parameter found; "
                    "skipping bias calibration. Initial q_logit mean was "
                    f"{shift:.3f} — if training stalls, the backbone may need "
                    "a custom calibration path."
                )
            query_model.train()

    # ── Phase 0: MLM warmup ──────────────────────────────────────────
    # Re-train the body+head on standard masked-LM in the donor's vocabulary.
    # The transplant gave the model new input embeddings (and tied output
    # embeddings) but the body learned its hidden representations against
    # the *old* vocab. For backbones whose head-body coherence doesn't survive
    # the swap (bert-base-uncased), a few thousand steps of MLM in the new
    # vocab re-establishes the relationship and gives alignment a healthy
    # starting point. Default off (0) so this is opt-in per-backbone.
    #
    # Skipped on resume — the saved checkpoint already includes the warmed body.
    mlm_warmup_steps = lc.get("mlm_warmup_steps", 0)
    if mlm_warmup_steps > 0 and start_step == 0:
        mlm_mask_rate = lc.get("mlm_mask_rate", 0.15)
        mlm_lr        = lc.get("mlm_warmup_lr", lc.get("alignment_lr", 5e-4))

        # Repurpose a Llama-3 reserved special token as our [MASK]. These
        # never appear in normal text, so they make a clean MASK marker
        # (vs. the previous "90% random" strategy, which had no signal that
        # a position needed prediction). Default: <|reserved_special_token_0|>
        # at id 128002.
        mask_token_id = lc.get("mlm_mask_token_id", 128002)

        # The transplant left cls.predictions.bias as a 128k vector where
        # the first 30522 entries are bert's original biases for bert's
        # token IDs (now reinterpreted as Llama-3 IDs — random misalignment)
        # and the rest are zero. Re-zero the whole bias so MLM trains it
        # from scratch with proper Llama-3 token statistics.
        for name, par in query_model.mlm.named_parameters():
            if name == "cls.predictions.bias" and par.dim() == 1 and par.shape[0] == query_model.vocab_size:
                par.data.zero_()
                print(f"[lion-align] Zeroed {name} (was bert-mismapped after transplant).")
                break

        print(
            f"[lion-align] Phase 0: MLM warmup ({mlm_warmup_steps} steps, "
            f"mask_rate={mlm_mask_rate}, lr={mlm_lr}, "
            f"mask_token_id={mask_token_id}) …"
        )

        # MLM benefits from longer contexts; use corpus passages, not queries.
        mlm_loader = make_alignment_loader(
            cfg["sae"]["corpus_dataset"],
            cfg["sae"].get("corpus_text_field", "text"),
            query_tokenizer,
            lc["batch_size"],
            lc["doc_max_length"],
            device,
        )
        mlm_optimizer = torch.optim.AdamW(
            query_model.parameters(), lr=mlm_lr, weight_decay=lc["weight_decay"]
        )
        mlm_scaler = GradScaler(enabled=lc["fp16"] and device.type == "cuda")
        V_qm    = query_model.vocab_size
        pad_id  = query_tokenizer.pad_token_id

        query_model.train()
        t_mlm = time.time()
        for mstep in range(mlm_warmup_steps):
            ids, attn_mask, _ = next(mlm_loader)
            # Standard BERT 80/10/10 masking strategy:
            #   80% replaced with the MASK token  (model knows: predict here)
            #   10% replaced with a random token  (regularises against blind copying)
            #   10% kept unchanged                 (regularises against blind reliance on MASK)
            rand   = torch.rand_like(ids, dtype=torch.float)
            keep   = (ids == pad_id) | (attn_mask == 0)
            mlm_pos = (rand < mlm_mask_rate) & ~keep

            rand2  = torch.rand_like(ids, dtype=torch.float)
            mask_action   = mlm_pos & (rand2 < 0.8)                 # → [MASK]
            random_action = mlm_pos & (rand2 >= 0.8) & (rand2 < 0.9)  # → random token
            # remaining 10% of mlm_pos: unchanged

            random_tokens = torch.randint(0, V_qm, ids.shape, device=device)
            input_ids_mlm = ids.clone()
            input_ids_mlm[mask_action]   = mask_token_id
            input_ids_mlm[random_action] = random_tokens[random_action]

            labels_mlm = ids.clone()
            labels_mlm[~mlm_pos] = -100  # ignore non-mask positions in CE

            mlm_optimizer.zero_grad()
            with autocast(enabled=lc["fp16"] and device.type == "cuda"):
                logits_mlm = query_model.mlm(
                    input_ids=input_ids_mlm, attention_mask=attn_mask
                ).logits  # [B, L, V]
                mlm_loss = F.cross_entropy(
                    logits_mlm.reshape(-1, V_qm), labels_mlm.reshape(-1),
                    ignore_index=-100,
                )

            mlm_scaler.scale(mlm_loss).backward()
            if mlm_scaler.is_enabled():
                mlm_scaler.unscale_(mlm_optimizer)
            # Looser grad clip than the alignment phase: the post-transplant
            # model has the *head* in essentially fresh-init state for the new
            # vocab, so initial total grad norms are ~100. Clipping to 1.0
            # (alignment-phase default) starves training by ~100x. Configurable
            # via mlm_grad_clip; 10.0 is a safer default for re-pretraining.
            mlm_grad_clip = lc.get("mlm_grad_clip", 10.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(query_model.parameters()), mlm_grad_clip,
            ).item()
            mlm_scaler.step(mlm_optimizer)
            mlm_scaler.update()

            if (mstep + 1) % lc["log_every"] == 0:
                elapsed = time.time() - t_mlm
                # MLM accuracy on masked positions — quick health check.
                with torch.no_grad():
                    pred = logits_mlm.argmax(-1)
                    correct = ((pred == labels_mlm) & mlm_pos).float().sum()
                    total = mlm_pos.float().sum().clamp(min=1.0)
                    acc = (correct / total).item()
                print(
                    f"[lion-align/mlm] step {mstep+1:>6} | loss {mlm_loss.item():.4f} "
                    f"| acc {acc:.3f} | gnorm {grad_norm:.1f} "
                    f"| masked/batch {int(mlm_pos.sum().item())} | {elapsed:.0f}s"
                )
                writer.add_scalar("lion_transplant_align/mlm/loss", mlm_loss.item(), mstep + 1)
                writer.add_scalar("lion_transplant_align/mlm/acc", acc, mstep + 1)
                t_mlm = time.time()

        del mlm_loader, mlm_optimizer, mlm_scaler
        import gc; gc.collect(); torch.cuda.empty_cache()
        print("[lion-align] Phase 0 complete; alignment optimiser starts fresh from here.")

    # ── Initial eval: doc-doc ceiling (Lion both sides) ──────────────
    if start_step == 0 and cfg.get("eval", {}).get("datasets"):
        print("[lion-align] Initial eval — doc_doc ceiling + query_doc baseline …")
        import gc; gc.collect(); torch.cuda.empty_cache()
        evaluate_asymmetric(
            query_model, query_tokenizer, lion_doc, cfg, device,
            writer=writer, step=0, run_doc_doc=True, override_k=0,
            section="lion_transplant_align",
        )
        # Eval encodes thousands of docs through Lion-SP-1B in fp16; the caching
        # allocator holds that memory by default and the training loop OOMs when
        # it tries to allocate its own activations on top. Force a release here.
        import gc; gc.collect(); torch.cuda.empty_cache()

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    if kd_warmup_steps > 0 and start_step < kd_warmup_steps:
        print(f"[lion-align] Phase 1: KD warmup ({kd_warmup_steps - start_step} steps, T={kd_temperature}) …")
    print(f"[lion-align] Phase 2: cosine alignment ({alignment_steps} steps) …")

    accum_loss = 0.0
    q_vecs_log = None
    optimizer.zero_grad()
    for step in range(start_step, total_steps):
        a_ids, a_mask, texts = next(align_loader)
        is_accum_boundary = (step + 1) % grad_accum == 0 or (step + 1) == total_steps

        in_kd_phase = step < kd_warmup_steps

        with autocast(enabled=lc["fp16"] and device.type == "cuda"):
            # Single forward + single Lion forward, both phases.
            _raw = query_model.mlm(input_ids=a_ids, attention_mask=a_mask).logits
            _mask = a_mask.unsqueeze(-1).float()
            q_logits = (_raw + (1.0 - _mask) * -1e6).max(dim=1).values
            d_logits = lion_doc.encode_logits(texts, lc["query_max_length"]).to(q_logits.dtype)
            q_vecs = torch.log1p(torch.relu(q_logits))

            if in_kd_phase:
                # Plain softmax KD: warm the model into Lion's logit-shape basin.
                # Note: shift-invariant — the model's logit offset is unconstrained,
                # which is why we don't end the KD phase using post-relu metrics.
                T = kd_temperature
                align_loss = F.kl_div(
                    F.log_softmax(q_logits / T, dim=-1),
                    F.softmax(d_logits / T, dim=-1),
                    reduction="batchmean",
                ) * T ** 2
            else:
                # Two alignment-loss formulations are available, switched by
                # the `align_loss_kind` config knob:
                #
                # "cos_kl" (default for ettin-150m): cosine on SPLADE vectors
                # + softmax-KL on raw logits. Works when the body's natural
                # post-transplant logits already line up roughly with Lion's
                # active set (true for ModernBERT/ettin). Fails on bert-base
                # / roberta-base because cos has zero gradient through the
                # relu and KL is shift-invariant — once q_logits drift to
                # all-negative, every gradient on the offset axis is zero
                # and the model is stuck at q_nnz=0, loss=1.
                #
                # "bce_logits" (recommended for bert/roberta): BCE-with-logits
                # against the BINARY support of Lion (target = (d_logits > 0)).
                # Operates on raw q_logits → no relu trap. Not shift-invariant
                # → real force on the offset axis. pos_weight balances the
                # ~300:128k class imbalance. Crucially uses BINARY targets
                # (0 or 1), not sigmoid(d_logits), to avoid the soft-target
                # arithmetic blowup we hit earlier where borderline tokens
                # got upweighted into runaway density.
                kind = lc.get("align_loss_kind", "cos_kl")
                if kind == "bce_logits" or kind == "bce_cos":
                    target = (d_logits > 0).float()
                    pw_override = lc.get("bce_pos_weight")
                    if pw_override is None:
                        with torch.no_grad():
                            n_pos = target.sum().clamp(min=1.0)
                            n_neg = (1.0 - target).sum()
                            pos_weight = (n_neg / n_pos).clamp(max=2000.0).to(q_logits.dtype)
                    else:
                        pos_weight = torch.tensor(
                            float(pw_override), device=q_logits.device, dtype=q_logits.dtype
                        )
                    bce_loss = F.binary_cross_entropy_with_logits(
                        q_logits, target, pos_weight=pos_weight
                    )
                    if kind == "bce_cos":
                        # BCE handles the binary support (escapes dead-relu);
                        # cos refines magnitudes & tightens the support. Once
                        # BCE has pulled q_vec away from zero, cos's gradient
                        # through the relu is alive and useful.
                        cos_loss = (
                            1.0 - F.cosine_similarity(
                                q_vecs, torch.log1p(torch.relu(d_logits))
                            )
                        ).mean()
                        cos_coeff = lc.get("cos_coeff", 1.0)
                        align_loss = bce_loss + cos_coeff * cos_loss
                    else:
                        align_loss = bce_loss
                else:  # "cos_kl"
                    cos_loss = (
                        1.0 - F.cosine_similarity(q_vecs, torch.log1p(torch.relu(d_logits)))
                    ).mean()
                    T = kd_temperature
                    kl_loss = F.kl_div(
                        F.log_softmax(q_logits / T, dim=-1),
                        F.softmax(d_logits / T, dim=-1),
                        reduction="batchmean",
                    ) * T ** 2
                    kl_coeff = lc.get("kl_coeff", 1.0)
                    align_loss = cos_loss + kl_coeff * kl_loss

            # FLOPs in both phases. Squared-mean form: gradient ∝ activation,
            # gives stable soft sparsity. Without this in the KD phase, the
            # warmup ends with q_nnz ~90k (very dense) and the next phase has
            # to do drastic sparsification, which destabilises training.
            if lambda_q > 0.0:
                flops_scale = min(1.0, step / max(flops_warmup, 1)) if flops_warmup > 0 else 1.0
                flops_loss = (q_vecs.mean(dim=0) ** 2).sum()
                align_loss = align_loss + flops_scale * lambda_q * flops_loss

        q_vecs_log = q_vecs.detach()
        # Lion's SPLADE vector for the same query texts — for d_nnz / d_flops
        # logging only (no gradient, computed alongside d_logits at no extra cost).
        d_vecs_log = torch.log1p(torch.relu(d_logits.detach()))
        accum_loss += align_loss.item()
        scaler.scale(align_loss / grad_accum).backward()

        if is_accum_boundary:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(query_model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            accum_loss = 0.0

        if (step + 1) % lc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "kd" if in_kd_phase else "cos"
            q_nnz = (q_vecs_log > 0).float().mean(0).sum().item()
            d_nnz = (d_vecs_log > 0).float().mean(0).sum().item()
            flops_val = q_vecs_log.abs().sum(dim=-1).mean().item()
            d_flops_val = d_vecs_log.abs().sum(dim=-1).mean().item()
            print(
                f"[lion-align/{phase}] step {step+1:>6} | loss {align_loss.item():.4f} | "
                f"q_nnz {q_nnz:.1f} (lion {d_nnz:.1f}) | "
                f"flops {flops_val:.1f} (lion {d_flops_val:.1f}) | "
                f"lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar(f"lion_transplant_align/{phase}/loss", align_loss.item(), step + 1)
            writer.add_scalar("lion_transplant_align/q_nnz", q_nnz, step + 1)
            writer.add_scalar("lion_transplant_align/d_nnz", d_nnz, step + 1)
            writer.add_scalar("lion_transplant_align/flops_q", flops_val, step + 1)
            writer.add_scalar("lion_transplant_align/flops_d", d_flops_val, step + 1)
            writer.add_scalar("lion_transplant_align/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % lc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            if not in_kd_phase and cfg.get("eval", {}).get("datasets"):
                print(f"[lion-align] Eval at step {step+1} …")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, lion_doc, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="lion_transplant_align",
                )
                query_model.train()
                # Eval encodes thousands of docs through Lion-SP-1B in fp16; the
                # caching allocator holds that memory by default, so the next
                # training step OOMs trying to allocate activations on top.
                gc.collect()
                torch.cuda.empty_cache()

    # ── Final save ────────────────────────────────────────────────────
    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": total_steps}, final_path)
    print(f"[lion-align] Done. Final checkpoint → {final_path}")
    writer.close()


def train_random_init_align(cfg: dict, resume: str | None = None):
    """Ablation: cosine alignment with a randomly-initialized query encoder.

    Same architecture as vocab_transplant_align (ettin backbone + donor MLM head
    in donor vocab space) but the embedding table is randomly initialized instead
    of being seeded by tokensurgeon.  Isolates whether tokensurgeon's kNN
    embedding transfer is load-bearing, or whether the right vocabulary size
    and architecture are sufficient on their own.

    Because the embeddings start random, the model needs a KD warmup phase
    (temperature-scaled KL on pre-relu logits) to prevent dead-dim collapse
    before switching to cosine alignment.

    Reads from the ``random_init_align`` config section.  The randomly-initialized
    model is cached in ``init_cache_dir`` so subsequent runs skip the build step.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import torch.nn.functional as F

    from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE
    from eval import evaluate_asymmetric
    from data import make_alignment_loader

    if "random_init_align" not in cfg:
        raise SystemExit(
            "[random-init-align] config.yaml is missing a `random_init_align:` section."
        )
    rc = cfg["random_init_align"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alignment_steps = rc.get("alignment_steps", 0)
    if alignment_steps <= 0:
        raise SystemExit("[random-init-align] alignment_steps must be > 0.")

    kd_warmup_steps = rc.get("kd_warmup_steps", 10_000)
    kd_temperature  = rc.get("kd_temperature", 4.0)
    total_steps = kd_warmup_steps + alignment_steps

    print(
        f"[random-init-align] query={rc['query_hf_id']} | doc={rc['doc_splade_hf_id']} | "
        f"kd_warmup={kd_warmup_steps} | alignment={alignment_steps} | "
        f"lr={rc.get('alignment_lr', 5e-4)}"
    )

    # ── Step 1: build random-init model (cached after first run) ──────
    cache_dir = str(_model_ckpt_root(rc["query_hf_id"]) / rc["init_cache_dir"])
    if not Path(cache_dir, "config.json").exists():
        print(f"[random-init-align] Building random-init model in donor vocab space …")
        raw = AutoModelForMaskedLM.from_pretrained(rc["query_hf_id"])
        donor_tok = AutoTokenizer.from_pretrained(rc["doc_splade_hf_id"])
        donor_vocab_size = len(donor_tok.get_vocab())
        raw.resize_token_embeddings(donor_vocab_size)
        # Re-init so the ablation starts from pure noise, not ettin's rows 0..30521
        # (which are semantically wrong for splade-v3 vocab IDs anyway).
        torch.nn.init.normal_(raw.get_input_embeddings().weight, mean=0.0, std=0.02)
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id",
                     "unk_token_id", "mask_token_id"):
            setattr(raw.config, attr, getattr(donor_tok, attr, None))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        raw.save_pretrained(cache_dir)
        donor_tok.save_pretrained(cache_dir)
        print(f"[random-init-align] Cached → {cache_dir}")
    else:
        print(f"[random-init-align] Random-init model already cached at {cache_dir}, skipping.")

    # ── Step 2: load models ───────────────────────────────────────────
    print(f"[random-init-align] Loading frozen doc SPLADE: {rc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(rc["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()

    print(f"[random-init-align] Loading query model from {cache_dir} …")
    query_model = VocabTransplantQuerySPLADE(cache_dir)
    query_model.to(device)
    query_tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    n_params = sum(p.numel() for p in query_model.parameters())
    print(f"[random-init-align] Query params: {n_params:,} | vocab_size: {query_model.vocab_size}")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[random-init-align] Resumed from {resume} at step {start_step}")

    # ── Optimiser / scheduler / scaler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        query_model.parameters(),
        lr=rc.get("alignment_lr", 5e-4),
        weight_decay=rc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=rc["fp16"] and device.type == "cuda")

    # ── Output dir & loader ───────────────────────────────────────────
    out_dir = _model_ckpt_root(rc["query_hf_id"]) / rc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    align_loader = make_alignment_loader(
        rc.get("queries_dataset", "Tevatron/msmarco-passage"),
        "query",
        query_tokenizer,
        rc["batch_size"],
        rc["query_max_length"],
        device,
    )

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    if kd_warmup_steps > 0 and start_step < kd_warmup_steps:
        print(f"[random-init-align] Phase 1: KD warmup ({kd_warmup_steps - start_step} steps, T={kd_temperature}) …")
    print(f"[random-init-align] Phase 2: cosine alignment ({alignment_steps} steps) …")

    for step in range(start_step, total_steps):
        a_ids, a_mask, texts = next(align_loader)
        optimizer.zero_grad()

        in_kd_phase = step < kd_warmup_steps

        with autocast(enabled=rc["fp16"] and device.type == "cuda"):
            if in_kd_phase:
                _raw = query_model.mlm(input_ids=a_ids, attention_mask=a_mask).logits
                _mask = a_mask.unsqueeze(-1).float()
                q_logits = (_raw + (1.0 - _mask) * -1e6).max(dim=1).values
                d_logits = doc_splade.encode_logits(texts, rc["query_max_length"]).to(q_logits.dtype)
                T = kd_temperature
                align_loss = F.kl_div(
                    F.log_softmax(q_logits / T, dim=-1),
                    F.softmax(d_logits / T, dim=-1),
                    reduction="batchmean",
                ) * T ** 2
                with torch.no_grad():
                    q_vecs = query_model.encode(a_ids, a_mask)
            else:
                q_vecs = query_model.encode(a_ids, a_mask)
                d_vecs = doc_splade.encode(texts, rc["query_max_length"], no_grad=True)
                align_loss = (1.0 - F.cosine_similarity(q_vecs, d_vecs.to(q_vecs.dtype))).mean()

        q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
        scaler.scale(align_loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(query_model.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % rc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "kd" if in_kd_phase else "cos"
            print(
                f"[random-init-align/{phase}] step {step+1:>6} | loss {align_loss.item():.4f} | "
                f"q_nnz {q_nnz:.1f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar(f"random_init_align/{phase}/loss", align_loss.item(), step + 1)
            writer.add_scalar("random_init_align/q_nnz", q_nnz, step + 1)
            writer.add_scalar("random_init_align/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % rc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            if not in_kd_phase and cfg.get("eval", {}).get("datasets"):
                print(f"[random-init-align] Eval at step {step+1} …")
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="random_init_align",
                )
                query_model.train()

    # ── Final save ────────────────────────────────────────────────────
    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": total_steps}, final_path)
    print(f"[random-init-align] Done. Final checkpoint → {final_path}")
    writer.close()


def train_doc_head_align(cfg: dict, resume: str | None = None):
    """KD warmup + cosine alignment using the doc SPLADE's MLM head as vocabulary decoder.

    Architecture: backbone (ettin-17m) → linear projection (hidden→splade_hidden)
    → doc SPLADE MLM head → SPLADE vector in doc vocab space.

    The MLM head is frozen by default (freeze_doc_head=true): only the backbone
    and projection are trained.  This tests whether the pre-trained head's
    structure alone bootstraps alignment without tokensurgeon.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer
    import torch.nn.functional as F

    from model import FrozenDocSPLADE, ProjectedQuerySPLADE
    from eval import evaluate_asymmetric
    from data import make_alignment_loader

    if "doc_head_align" not in cfg:
        raise SystemExit(
            "[doc-head-align] config.yaml is missing a `doc_head_align:` section."
        )
    dc = cfg["doc_head_align"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alignment_steps = dc.get("alignment_steps", 0)
    if alignment_steps <= 0:
        raise SystemExit("[doc-head-align] alignment_steps must be > 0.")

    kd_warmup_steps = dc.get("kd_warmup_steps", 10_000)
    kd_temperature  = dc.get("kd_temperature", 4.0)
    freeze_doc_head = dc.get("freeze_doc_head", True)
    total_steps = kd_warmup_steps + alignment_steps

    print(
        f"[doc-head-align] query={dc['query_hf_id']} | doc={dc['doc_splade_hf_id']} | "
        f"kd_warmup={kd_warmup_steps} | alignment={alignment_steps} | "
        f"freeze_doc_head={freeze_doc_head} | lr={dc.get('alignment_lr', 5e-4)}"
    )

    # ── Load models ───────────────────────────────────────────────────
    print(f"[doc-head-align] Loading frozen doc SPLADE: {dc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(dc["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()

    print(f"[doc-head-align] Building query encoder …")
    query_model = ProjectedQuerySPLADE(dc["query_hf_id"], dc["doc_splade_hf_id"])
    if freeze_doc_head:
        for p in query_model.mlm_head.parameters():
            p.requires_grad_(False)
    query_model.to(device)

    query_tokenizer = AutoTokenizer.from_pretrained(dc["query_hf_id"])

    trainable = sum(p.numel() for p in query_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in query_model.parameters())
    print(f"[doc-head-align] Params — total: {total_params:,} | trainable: {trainable:,} | "
          f"vocab_size: {query_model.vocab_size}")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[doc-head-align] Resumed from {resume} at step {start_step}")

    # ── Optimiser / scheduler / scaler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, query_model.parameters()),
        lr=dc.get("alignment_lr", 5e-4),
        weight_decay=dc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=dc["fp16"] and device.type == "cuda")

    # ── Output dir & loader ───────────────────────────────────────────
    out_dir = _model_ckpt_root(dc["query_hf_id"]) / dc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    align_loader = make_alignment_loader(
        dc.get("queries_dataset", "Tevatron/msmarco-passage"),
        "query",
        query_tokenizer,
        dc["batch_size"],
        dc["query_max_length"],
        device,
    )

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    if kd_warmup_steps > 0 and start_step < kd_warmup_steps:
        print(f"[doc-head-align] Phase 1: KD warmup ({kd_warmup_steps - start_step} steps, T={kd_temperature}) …")
    print(f"[doc-head-align] Phase 2: cosine alignment ({alignment_steps} steps) …")

    for step in range(start_step, total_steps):
        a_ids, a_mask, texts = next(align_loader)
        optimizer.zero_grad()

        in_kd_phase = step < kd_warmup_steps

        with autocast(enabled=dc["fp16"] and device.type == "cuda"):
            if in_kd_phase:
                q_logits = query_model.encode_logits(a_ids, a_mask)
                d_logits = doc_splade.encode_logits(texts, dc["query_max_length"]).to(q_logits.dtype)
                T = kd_temperature
                align_loss = F.kl_div(
                    F.log_softmax(q_logits / T, dim=-1),
                    F.softmax(d_logits / T, dim=-1),
                    reduction="batchmean",
                ) * T ** 2
                with torch.no_grad():
                    q_vecs = query_model.encode(a_ids, a_mask)
            else:
                q_vecs = query_model.encode(a_ids, a_mask)
                d_vecs = doc_splade.encode(texts, dc["query_max_length"], no_grad=True)
                align_loss = (1.0 - F.cosine_similarity(q_vecs, d_vecs.to(q_vecs.dtype))).mean()

        q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
        scaler.scale(align_loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in query_model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % dc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "kd" if in_kd_phase else "cos"
            print(
                f"[doc-head-align/{phase}] step {step+1:>6} | loss {align_loss.item():.4f} | "
                f"q_nnz {q_nnz:.1f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar(f"doc_head_align/{phase}/loss", align_loss.item(), step + 1)
            writer.add_scalar("doc_head_align/q_nnz", q_nnz, step + 1)
            writer.add_scalar("doc_head_align/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % dc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            if not in_kd_phase and cfg.get("eval", {}).get("datasets"):
                print(f"[doc-head-align] Eval at step {step+1} …")
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="doc_head_align",
                )
                query_model.train()

    # ── Final save ────────────────────────────────────────────────────
    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": total_steps}, final_path)
    print(f"[doc-head-align] Done. Final checkpoint → {final_path}")
    writer.close()


def train_direct_align(cfg: dict, resume: str | None = None):
    """Cosine alignment against a frozen doc SPLADE — no vocab transplant.

    For query encoders that already share the doc SPLADE's tokenizer
    (vocab_size=30522, bert-base-uncased).  Skips tokensurgeon entirely;
    the model is loaded directly from HuggingFace.

    Training has two phases:
      1. KD warmup (kd_warmup_steps): temperature-scaled KL divergence on
         pre-relu max-pooled logits. Provides gradient to all vocabulary
         dimensions, preventing the dead-dim collapse that happens when
         cold-starting a BERT model without tokensurgeon initialisation.
      2. Cosine alignment (remaining steps): standard 1 - cosine_similarity
         on SPLADE sparse vectors, same as vocab_transplant_align.

    Set kd_warmup_steps=0 to skip phase 1 (not recommended).
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer
    import torch.nn.functional as F

    from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE
    from eval import evaluate_asymmetric
    from data import make_alignment_loader

    if "direct_align" not in cfg:
        raise SystemExit(
            "[direct_align] config.yaml is missing a `direct_align:` section."
        )
    dc = cfg["direct_align"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alignment_steps = dc.get("alignment_steps", 0)
    if alignment_steps <= 0:
        raise SystemExit("[direct_align] alignment_steps must be > 0.")

    kd_warmup_steps = dc.get("kd_warmup_steps", 0)
    kd_temperature  = dc.get("kd_temperature", 4.0)
    total_steps = kd_warmup_steps + alignment_steps

    print(
        f"[direct_align] query={dc['query_hf_id']} | doc={dc['doc_splade_hf_id']} | "
        f"kd_warmup={kd_warmup_steps} | alignment={alignment_steps} | "
        f"lr={dc.get('alignment_lr', 5e-4)}"
    )

    # ── Load models ───────────────────────────────────────────────────
    print(f"[direct_align] Loading frozen doc SPLADE: {dc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(dc["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()

    print(f"[direct_align] Loading query model: {dc['query_hf_id']} …")
    query_model = VocabTransplantQuerySPLADE(dc["query_hf_id"])
    query_model.to(device)
    query_tokenizer = AutoTokenizer.from_pretrained(dc["query_hf_id"])

    # Fail fast if vocab sizes don't match — alignment won't make sense otherwise.
    doc_vocab = doc_splade.tokenizer.vocab_size
    if query_model.vocab_size != doc_vocab:
        raise SystemExit(
            f"[direct_align] Vocab size mismatch: query model has "
            f"{query_model.vocab_size} tokens, doc SPLADE has {doc_vocab}. "
            f"Use vocab_transplant_align instead, or pick a model with "
            f"vocab_size={doc_vocab} (e.g. google/bert_uncased_L-4_H-256_A-4)."
        )

    n_params = sum(p.numel() for p in query_model.parameters())
    print(f"[direct_align] Query model params: {n_params:,} | vocab_size: {query_model.vocab_size}")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[direct_align] Resumed from {resume} at step {start_step}")

    # ── Optimiser / scheduler / scaler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        query_model.parameters(),
        lr=dc.get("alignment_lr", 5e-4),
        weight_decay=dc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=dc["fp16"] and device.type == "cuda")

    # ── Output dir & loader ───────────────────────────────────────────
    out_dir = _model_ckpt_root(dc["query_hf_id"]) / dc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    align_loader = make_alignment_loader(
        dc.get("queries_dataset", "Tevatron/msmarco-passage"),
        "query",
        query_tokenizer,
        dc["batch_size"],
        dc["query_max_length"],
        device,
    )

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    if kd_warmup_steps > 0 and start_step < kd_warmup_steps:
        print(f"[direct_align] Phase 1: KD warmup ({kd_warmup_steps - start_step} steps, T={kd_temperature}) …")
    print(f"[direct_align] Phase 2: cosine alignment ({alignment_steps} steps) …")

    for step in range(start_step, total_steps):
        a_ids, a_mask, texts = next(align_loader)
        optimizer.zero_grad()

        in_kd_phase = step < kd_warmup_steps

        with autocast(enabled=dc["fp16"] and device.type == "cuda"):
            if in_kd_phase:
                # Phase 1: KL divergence on pre-relu logits — gradient flows to
                # all vocab dims regardless of current activation state.
                _raw = query_model.mlm(input_ids=a_ids, attention_mask=a_mask).logits
                _mask = a_mask.unsqueeze(-1).float()
                q_logits = (_raw + (1.0 - _mask) * -1e6).max(dim=1).values
                d_logits = doc_splade.encode_logits(texts, dc["query_max_length"]).to(q_logits.dtype)
                T = kd_temperature
                align_loss = F.kl_div(
                    F.log_softmax(q_logits / T, dim=-1),
                    F.softmax(d_logits / T, dim=-1),
                    reduction="batchmean",
                ) * T ** 2
                with torch.no_grad():
                    q_vecs = query_model.encode(a_ids, a_mask)
            else:
                # Phase 2: cosine alignment on sparse SPLADE vectors.
                q_vecs = query_model.encode(a_ids, a_mask)
                d_vecs = doc_splade.encode(texts, dc["query_max_length"], no_grad=True)
                align_loss = (1.0 - F.cosine_similarity(q_vecs, d_vecs.to(q_vecs.dtype))).mean()

        q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
        scaler.scale(align_loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(query_model.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % dc["log_every"] == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "kd" if in_kd_phase else "cos"
            print(
                f"[direct_align/{phase}] step {step+1:>6} | loss {align_loss.item():.4f} | "
                f"q_nnz {q_nnz:.1f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar(f"direct_align/{phase}/loss", align_loss.item(), step + 1)
            writer.add_scalar("direct_align/q_nnz", q_nnz, step + 1)
            writer.add_scalar("direct_align/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % dc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            # Only run NDCG eval during the cosine phase — KD logits aren't sparse vectors yet.
            if not in_kd_phase and cfg.get("eval", {}).get("datasets"):
                print(f"[direct_align] Eval at step {step+1} …")
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="direct_align",
                )
                query_model.train()

    # ── Final save ────────────────────────────────────────────────────
    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": total_steps}, final_path)
    print(f"[direct_align] Done. Final checkpoint → {final_path}")
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _model_ckpt_root(hf_id: str) -> Path:
    return Path(f"checkpoints_{hf_id.split('/')[-1]}")


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

def train_splade_shallow_align(
    cfg: dict,
    resume: str | None = None,
    init_from: str | None = None,
    section: str = "splade_shallow_align",
    factorized: bool = False,
):
    """Layer-truncated SPLADE query encoder, distilled against the full doc encoder.

    Both encoders share tokenizer, embeddings, and MLM head. Only the body is
    shallower on the query side. This avoids every cross-architecture mismatch
    that hits the vocab-transplant approaches (vocab overlap, BPE convention,
    head alignment): the query model is literally the doc model with later
    layers chopped off.

    Two-phase training:
      Phase 0 — frozen warmup: only the kept body layers update. Embeddings and
        head stay pinned at the doc encoder's values, giving the body a stable
        target distribution to adapt to.
      Phase 1 — full fine-tuning: everything trainable.

    Loss is teacher-vector MSE on SPLADE query vectors, optionally augmented
    with a contrastive CE term against frozen teacher passage vectors. The
    contrastive term uses Tevatron positives and hard negatives, so it adds a
    ranking signal without making the full doc encoder trainable. Reads its
    config from the ``splade_shallow_align`` section.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn.functional as F

    from model import FrozenDocSPLADE, ShallowFactorizedSpladeQuery, ShallowSpladeQuery
    from eval import evaluate_asymmetric
    from data import make_ranking_distill_loader

    if section not in cfg:
        raise SystemExit(f"[shallow] config.yaml is missing a `{section}:` section.")
    sc = cfg[section]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_layers            = int(sc["n_layers"])
    alignment_steps     = int(sc.get("alignment_steps", 50_000))
    freeze_warmup_steps = int(sc.get("freeze_warmup_steps", 10_000))
    align_lr            = float(sc.get("alignment_lr", 5e-4))
    weight_decay        = float(sc.get("weight_decay", 0.01))
    fp16                = bool(sc.get("fp16", True))
    lambda_q              = float(sc.get("lambda_q", 0.0))
    lambda_q_warmup_steps = int(sc.get("lambda_q_warmup_steps", 0))
    use_contrastive       = bool(sc.get("use_contrastive", sc.get("contrastive_coeff", 0.0) > 0.0))
    contrastive_coeff     = float(sc.get("contrastive_coeff", 0.0)) if use_contrastive else 0.0
    contrastive_temperature = float(sc.get("contrastive_temperature", 1.0))
    log_ranking_metrics   = bool(sc.get("log_ranking_metrics", use_contrastive))
    freeze_head_after_warmup = bool(sc.get("freeze_head_after_warmup", False))
    nway                  = int(sc.get("nway", 8))
    grad_accum          = int(sc.get("gradient_accumulation_steps", 1))
    if alignment_steps <= 0:
        raise SystemExit("[shallow] alignment_steps must be > 0.")
    if freeze_warmup_steps < 0 or freeze_warmup_steps > alignment_steps:
        raise SystemExit("[shallow] freeze_warmup_steps must be in [0, alignment_steps].")

    if contrastive_temperature <= 0.0:
        raise SystemExit("[shallow] contrastive_temperature must be > 0.")
    print(
        f"[shallow] doc={sc['doc_splade_hf_id']} | n_layers={n_layers} "
        f"| freeze_warmup={freeze_warmup_steps} | alignment={alignment_steps} "
        f"| lr={align_lr} | nway={nway} | lambda_q={lambda_q} "
        f"| use_contrastive={use_contrastive} | contrastive_coeff={contrastive_coeff} "
        f"| log_ranking_metrics={log_ranking_metrics} | grad_accum={grad_accum} "
        f"| freeze_head_after_warmup={freeze_head_after_warmup}"
    )

    # ── Models ────────────────────────────────────────────────────────
    print(f"[shallow] Loading frozen doc SPLADE: {sc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(sc["doc_splade_hf_id"]); doc_splade.to(device); doc_splade.eval()

    print(f"[shallow] Building shallow query model from same checkpoint, keeping {n_layers} layers …")
    layer_indices = sc.get("layer_indices")
    if layer_indices is not None:
        layer_indices = [int(idx) for idx in layer_indices]
        print(f"[shallow] Using explicit doc-layer indices for query body: {layer_indices}")
    if factorized:
        factor_dim = int(sc.get("factorized_embedding_dim", 128))
        factor_init = sc.get("factorization_init", "svd")
        query_kwargs = {
            "n_layers": n_layers,
            "factorized_embedding_dim": factor_dim,
            "init": factor_init,
        }
        if layer_indices is not None:
            query_kwargs["layer_indices"] = layer_indices
        query_model = ShallowFactorizedSpladeQuery(sc["doc_splade_hf_id"], **query_kwargs)
        print(
            f"[shallow] Installed factorized lexical matrix: dim={factor_dim} "
            f"init={factor_init} factorized_params={query_model.factorized_param_count()/1e6:.1f}M"
        )
    else:
        query_kwargs = {"n_layers": n_layers}
        if layer_indices is not None:
            query_kwargs["layer_indices"] = layer_indices
        query_model = ShallowSpladeQuery(sc["doc_splade_hf_id"], **query_kwargs)
    query_model.to(device)
    query_tokenizer = query_model.tokenizer
    n_total = sum(p.numel() for p in query_model.parameters())
    n_trainable_full = n_total
    print(
        f"[shallow] Query model params (after truncation): {n_total/1e6:.1f}M "
        f"(was {query_model.original_n_layers}-layer original); will freeze "
        "embeddings + head during warmup."
    )

    # ── Resume / init ─────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"[shallow] Resumed from {resume} at step {start_step}")
    elif init_from:
        ckpt = torch.load(init_from, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        print(f"[shallow] Initialised weights from {init_from} (step counter reset to 0)")

    # ── Initial trainability based on phase at start_step ─────────────
    in_warmup = start_step < freeze_warmup_steps
    if in_warmup or freeze_head_after_warmup:
        query_model.freeze_for_warmup()
        phase_msg = "Phase 0 (frozen warmup)" if in_warmup else "Phase 1 with head kept frozen"
        print(f"[shallow] {phase_msg}. Trainable params: "
              f"{query_model.trainable_param_count()/1e6:.1f}M / {n_trainable_full/1e6:.1f}M total.")
    else:
        query_model.unfreeze_all()
        print(f"[shallow] Resuming directly into Phase 1 (everything trainable).")

    # ── Optimiser / scheduler / scaler ────────────────────────────────
    # Initial optimiser only has the warmup-phase parameters; we'll rebuild it
    # at the phase boundary so the new (now trainable) tensors have AdamW state.
    optimizer = torch.optim.AdamW(
        [p for p in query_model.parameters() if p.requires_grad],
        lr=align_lr, weight_decay=weight_decay,
    )
    # CosineAnnealingLR needs initial_lr in param_groups when last_epoch > -1
    # (i.e. when resuming mid-run without a saved optimizer state).
    if start_step > 0:
        for pg in optimizer.param_groups:
            pg["initial_lr"] = align_lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=alignment_steps, eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=fp16 and device.type == "cuda")

    out_dir = _model_ckpt_root(sc["doc_splade_hf_id"]) / sc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    ranking_loader = make_ranking_distill_loader(
        nway=nway, batch_size=sc["batch_size"],
    )

    # ── Initial eval ──────────────────────────────────────────────────
    if start_step == 0 and cfg.get("eval", {}).get("datasets"):
        print("[shallow] Initial eval — doc_doc ceiling + query_doc baseline …")
        import gc; gc.collect(); torch.cuda.empty_cache()
        evaluate_asymmetric(
            query_model, query_tokenizer, doc_splade, cfg, device,
            writer=writer, step=0, run_doc_doc=True, override_k=0,
            section=section,
        )
        gc.collect(); torch.cuda.empty_cache()

    # ── Training loop ─────────────────────────────────────────────────
    query_model.train()
    t0 = time.time()
    print(f"[shallow] Training loop ({alignment_steps - start_step} steps remaining) …")
    optimizer.zero_grad()

    for step in range(start_step, alignment_steps):
        # ── Phase boundary: unfreeze everything ───────────────────────
        if in_warmup and step >= freeze_warmup_steps:
            current_lr = optimizer.param_groups[0]["lr"]
            body_params = list(query_model.mlm.bert.encoder.layer.parameters())
            if freeze_head_after_warmup:
                query_model.freeze_for_warmup()
                optimizer = torch.optim.AdamW(
                    body_params, lr=current_lr, weight_decay=weight_decay,
                )
                head_lr_scale = 0.0
            else:
                query_model.unfreeze_all()
                head_lr_scale = float(sc.get("head_lr_scale", 0.1))
                # Two param groups: body layers at full LR, embeddings + MLM head at
                # a fraction. This prevents co-adaptation instability right after unfreeze.
                body_ids = {id(p) for p in body_params}
                head_params = [p for p in query_model.parameters() if id(p) not in body_ids]
                optimizer = torch.optim.AdamW([
                    {"params": body_params, "lr": current_lr},
                    {"params": head_params, "lr": current_lr * head_lr_scale},
                ], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=alignment_steps - freeze_warmup_steps, eta_min=1e-5,
            )
            optimizer.zero_grad()
            in_warmup = False
            print(
                f"[shallow] Step {step}: switched to Phase 1. "
                f"body_lr={current_lr:.2e} head_lr={current_lr * head_lr_scale:.2e}"
            )

        do_log = (step + 1) % sc["log_every"] == 0
        q_texts, p_texts = next(ranking_loader)
        B_actual = len(q_texts)
        q_enc = query_tokenizer(
            q_texts, max_length=sc["query_max_length"],
            truncation=True, padding=True, return_tensors="pt",
        )
        a_ids = q_enc["input_ids"].to(device)
        a_mask = q_enc["attention_mask"].to(device)

        with torch.no_grad():
            # Teacher query vecs needed every step for the loss.
            t_q_vecs = doc_splade.encode(q_texts, sc["query_max_length"]).float()  # [B, V]
            # Passage vecs are needed for contrastive training and optional p@1 logging.
            # Encode in chunks to avoid OOM when training batch_size is large.
            need_passage_vecs = contrastive_coeff > 0.0 or (do_log and log_ranking_metrics)
            if need_passage_vecs:
                enc_bs = sc.get("eval_batch_size", 32)
                p_vecs = torch.cat([
                    doc_splade.encode(p_texts[i:i+enc_bs], sc["doc_max_length"]).float()
                    for i in range(0, len(p_texts), enc_bs)
                ], dim=0)
                p_vecs_3d = p_vecs.view(B_actual, nway, -1)
            if do_log and log_ranking_metrics:
                t_scores = (t_q_vecs.unsqueeze(1) * p_vecs_3d).sum(-1)  # [B, nway]

        with autocast(enabled=fp16 and device.type == "cuda"):
            _raw = query_model.mlm(input_ids=a_ids, attention_mask=a_mask).logits  # [B, L, V]
            _m = a_mask.unsqueeze(-1).float()
            q_logits = (_raw + (1.0 - _m) * -1e6).max(dim=1).values  # [B, V]

            # STE: forward uses real SPLADE vecs (scale matches teacher),
            # backward treats ReLU as identity so gradient reaches dead dims.
            q_relu = q_logits + (F.relu(q_logits) - q_logits).detach()
            q_vecs = torch.log1p(q_relu)

            # Direct vector MSE against teacher SPLADE query vecs.
            # Sum over vocab dims, mean over batch — keeps per-dim gradient at O(1/B)
            # instead of O(1/(B*V)), which is strong enough to revive dead dims from zero.
            rank_loss = ((q_vecs - t_q_vecs.to(q_vecs.dtype)) ** 2).sum(dim=-1).mean()
            align_loss = rank_loss
            if contrastive_coeff > 0.0:
                # In-batch contrastive loss: each query's positive passage is
                # the first passage in its nway block; all other passages in
                # the batch are negatives. Passage vectors stay frozen teacher
                # outputs, keeping the extra signal query-side only.
                passage_vecs = p_vecs.to(q_vecs.dtype)
                scores = q_vecs @ passage_vecs.T
                scores = scores / contrastive_temperature
                labels = torch.arange(B_actual, device=scores.device) * nway
                contrastive_loss = F.cross_entropy(scores, labels)
                align_loss = align_loss + contrastive_coeff * contrastive_loss
            else:
                contrastive_loss = q_vecs.new_tensor(0.0)
            if lambda_q > 0.0:
                lq_scale = min(1.0, step / lambda_q_warmup_steps) if lambda_q_warmup_steps > 0 else 1.0
                align_loss = align_loss + lq_scale * lambda_q * q_vecs.sum(-1).mean()

        is_accum_boundary = (step + 1) % grad_accum == 0 or (step + 1) == alignment_steps
        scaler.scale(align_loss / grad_accum).backward()
        if is_accum_boundary:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in query_model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad()

        if do_log:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "warm" if step < freeze_warmup_steps else "full"
            with torch.no_grad():
                q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
                t_nnz = (t_q_vecs > 0).float().mean(0).sum().item()
                flops = q_vecs.detach().abs().sum(-1).mean().item()
                if log_ranking_metrics:
                    p_f = p_vecs_3d.to(q_vecs.dtype)
                    s_scores = (q_vecs.unsqueeze(1) * p_f).sum(-1)
                    t_rank1 = t_scores.argmax(-1).eq(0).float().mean().item()
                    s_rank1 = s_scores.argmax(-1).eq(0).float().mean().item()
                    rank_msg = f" | p@1 s={s_rank1:.2f} t={t_rank1:.2f}"
                else:
                    s_rank1 = None
                    rank_msg = ""
            print(
                f"[shallow/{phase}] step {step+1:>6} | loss {align_loss.item():.4f} "
                f"(rank {rank_loss.item():.3f} ctr {contrastive_loss.item():.3f}) | "
                f"q_nnz {q_nnz:.1f} (teacher {t_nnz:.1f}) | flops {flops:.1f}"
                f"{rank_msg} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar(f"{section}/{phase}/loss", align_loss.item(), step + 1)
            writer.add_scalar(f"{section}/{phase}/rank_loss", rank_loss.item(), step + 1)
            writer.add_scalar(f"{section}/{phase}/contrastive_loss", contrastive_loss.item(), step + 1)
            writer.add_scalar(f"{section}/q_nnz", q_nnz, step + 1)
            writer.add_scalar(f"{section}/teacher_nnz", t_nnz, step + 1)
            if s_rank1 is not None:
                writer.add_scalar(f"{section}/precision_at_1", s_rank1, step + 1)
            writer.add_scalar(f"{section}/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % sc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            if cfg.get("eval", {}).get("datasets"):
                print(f"[shallow] Eval at step {step+1} …")
                import gc; gc.collect(); torch.cuda.empty_cache()
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section=section,
                )
                query_model.train()
                gc.collect(); torch.cuda.empty_cache()

    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": alignment_steps}, final_path)
    print(f"[shallow] Done. Final checkpoint → {final_path}")
    writer.close()


def train_splade_shallow_factorized_align(cfg: dict, resume: str | None = None, init_from: str | None = None):
    return train_splade_shallow_align(
        cfg,
        resume=resume,
        init_from=init_from,
        section="splade_shallow_factorized_align",
        factorized=True,
    )


def train_splade_shallow_factorized_spaced_align(cfg: dict, resume: str | None = None, init_from: str | None = None):
    return train_splade_shallow_align(
        cfg,
        resume=resume,
        init_from=init_from,
        section="splade_shallow_factorized_spaced_align",
        factorized=True,
    )


def train_splade_shallow_align_distill(cfg: dict, resume: str | None = None, init_from: str | None = None):
    """Layer-truncated SPLADE query encoder trained with cross-encoder MarginMSE.

    This mirrors ``train_splade_shallow_align`` architecturally, but replaces
    teacher-vector MSE + contrastive CE with a traditional reranker distillation
    objective:

      teacher_margin = CE(q, positive) - CE(q, negative)
      student_margin = dot(q_student, d_positive_full_splade) - dot(q_student, d_negative_full_splade)

    The document encoder remains frozen. Only the shallow query encoder learns.
    Reads its config from ``splade_shallow_align_distill``.
    """
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from model import FrozenDocSPLADE, ShallowSpladeQuery
    from eval import evaluate_asymmetric
    from data import make_ranking_distill_loader

    if "splade_shallow_align_distill" not in cfg:
        raise SystemExit("[shallow-distill] config.yaml is missing a `splade_shallow_align_distill:` section.")
    sc = cfg["splade_shallow_align_distill"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_layers              = int(sc["n_layers"])
    alignment_steps       = int(sc.get("alignment_steps", 50_000))
    freeze_warmup_steps   = int(sc.get("freeze_warmup_steps", 5_000))
    head_lr_scale         = float(sc.get("head_lr_scale", 0.1))
    align_lr              = float(sc.get("alignment_lr", 2e-4))
    weight_decay          = float(sc.get("weight_decay", 0.01))
    fp16                  = bool(sc.get("fp16", True))
    lambda_q              = float(sc.get("lambda_q", 0.0))
    lambda_q_warmup_steps = int(sc.get("lambda_q_warmup_steps", 0))
    query_anchor_coeff    = float(sc.get("query_anchor_coeff", 0.0))
    teacher_margin_scale  = float(sc.get("teacher_margin_scale", 1.0))
    teacher_margin_clip   = float(sc.get("teacher_margin_clip", 0.0))
    nway                  = int(sc.get("nway", 8))
    grad_accum            = int(sc.get("gradient_accumulation_steps", 1))
    teacher_batch_size    = int(sc.get("teacher_batch_size", 32))
    teacher_max_length    = int(sc.get("teacher_max_length", 512))
    teacher_hf_id         = sc["cross_encoder_hf_id"]

    if alignment_steps <= 0:
        raise SystemExit("[shallow-distill] alignment_steps must be > 0.")
    if freeze_warmup_steps < 0 or freeze_warmup_steps > alignment_steps:
        raise SystemExit("[shallow-distill] freeze_warmup_steps must be in [0, alignment_steps].")
    if nway < 2:
        raise SystemExit("[shallow-distill] nway must be >= 2 for margin MSE.")

    print(
        f"[shallow-distill] doc={sc['doc_splade_hf_id']} | teacher={teacher_hf_id} "
        f"| n_layers={n_layers} | freeze_warmup={freeze_warmup_steps} "
        f"| alignment={alignment_steps} | lr={align_lr} | nway={nway} "
        f"| lambda_q={lambda_q} | anchor={query_anchor_coeff} "
        f"| margin_scale={teacher_margin_scale} | grad_accum={grad_accum}"
    )

    print(f"[shallow-distill] Loading frozen doc SPLADE: {sc['doc_splade_hf_id']} …")
    doc_splade = FrozenDocSPLADE(sc["doc_splade_hf_id"]); doc_splade.to(device); doc_splade.eval()

    print(f"[shallow-distill] Loading cross-encoder teacher: {teacher_hf_id} …")
    ce_tokenizer = AutoTokenizer.from_pretrained(teacher_hf_id)
    ce_model = AutoModelForSequenceClassification.from_pretrained(teacher_hf_id)
    ce_model.to(device); ce_model.eval()
    for p in ce_model.parameters():
        p.requires_grad_(False)

    print(f"[shallow-distill] Building shallow query model from same checkpoint, keeping {n_layers} layers …")
    query_model = ShallowSpladeQuery(sc["doc_splade_hf_id"], n_layers=n_layers); query_model.to(device)
    query_tokenizer = query_model.tokenizer
    n_total = sum(p.numel() for p in query_model.parameters())
    print(
        f"[shallow-distill] Query model params: {n_total/1e6:.1f}M "
        f"({n_layers}/{query_model.original_n_layers} layers kept)."
    )

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"[shallow-distill] Resumed from {resume} at step {start_step}")
    elif init_from:
        ckpt = torch.load(init_from, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        print(f"[shallow-distill] Initialised weights from {init_from} (step counter reset to 0)")

    in_warmup = start_step < freeze_warmup_steps
    if in_warmup:
        query_model.freeze_for_warmup()
        print(
            f"[shallow-distill] Phase 0 (frozen warmup). Trainable params: "
            f"{query_model.trainable_param_count()/1e6:.1f}M / {n_total/1e6:.1f}M"
        )
    else:
        query_model.unfreeze_all()
        print("[shallow-distill] Resuming directly into Phase 1 (everything trainable).")

    optimizer = torch.optim.AdamW(
        [p for p in query_model.parameters() if p.requires_grad],
        lr=align_lr, weight_decay=weight_decay,
    )
    if start_step > 0:
        for pg in optimizer.param_groups:
            pg["initial_lr"] = align_lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=alignment_steps, eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    scaler = GradScaler(enabled=fp16 and device.type == "cuda")

    out_dir = _model_ckpt_root(sc["doc_splade_hf_id"]) / sc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    ranking_loader = make_ranking_distill_loader(nway=nway, batch_size=sc["batch_size"])

    def score_cross_encoder(q_texts: list[str], p_texts: list[str]) -> torch.Tensor:
        """Return teacher scores shaped [B, nway]."""
        pairs = []
        for i, query in enumerate(q_texts):
            start = i * nway
            for passage in p_texts[start:start + nway]:
                pairs.append((query, passage))

        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), teacher_batch_size):
                batch_pairs = pairs[i:i + teacher_batch_size]
                q_batch = [pair[0] for pair in batch_pairs]
                p_batch = [pair[1] for pair in batch_pairs]
                enc = ce_tokenizer(
                    q_batch,
                    p_batch,
                    max_length=teacher_max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with autocast(enabled=fp16 and device.type == "cuda"):
                    logits = ce_model(**enc).logits
                if logits.ndim == 2 and logits.shape[-1] > 1:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.view(-1)
                scores.append(batch_scores.float())
        return torch.cat(scores, dim=0).view(len(q_texts), nway)

    if start_step == 0 and cfg.get("eval", {}).get("datasets"):
        print("[shallow-distill] Initial eval — doc_doc ceiling + query_doc baseline …")
        import gc; gc.collect(); torch.cuda.empty_cache()
        evaluate_asymmetric(
            query_model, query_tokenizer, doc_splade, cfg, device,
            writer=writer, step=0, run_doc_doc=True, override_k=0,
            section="splade_shallow_align_distill",
        )
        gc.collect(); torch.cuda.empty_cache()

    query_model.train()
    t0 = time.time()
    print(f"[shallow-distill] Training loop ({alignment_steps - start_step} steps remaining) …")
    optimizer.zero_grad()

    for step in range(start_step, alignment_steps):
        if in_warmup and step >= freeze_warmup_steps:
            query_model.unfreeze_all()
            current_lr = optimizer.param_groups[0]["lr"]
            body_params = list(query_model.mlm.bert.encoder.layer.parameters())
            body_ids = {id(p) for p in body_params}
            head_params = [p for p in query_model.parameters() if id(p) not in body_ids]
            optimizer = torch.optim.AdamW([
                {"params": body_params, "lr": current_lr},
                {"params": head_params, "lr": current_lr * head_lr_scale},
            ], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=alignment_steps - freeze_warmup_steps, eta_min=1e-5,
            )
            optimizer.zero_grad()
            in_warmup = False
            print(
                f"[shallow-distill] Step {step}: switched to Phase 1. "
                f"body_lr={current_lr:.2e} head_lr={current_lr * head_lr_scale:.2e}"
            )

        do_log = (step + 1) % sc["log_every"] == 0
        q_texts, p_texts = next(ranking_loader)
        B_actual = len(q_texts)

        q_enc = query_tokenizer(
            q_texts, max_length=sc["query_max_length"],
            truncation=True, padding=True, return_tensors="pt",
        )
        a_ids = q_enc["input_ids"].to(device)
        a_mask = q_enc["attention_mask"].to(device)

        with torch.no_grad():
            teacher_scores = score_cross_encoder(q_texts, p_texts)
            teacher_margins_raw = teacher_scores[:, :1] - teacher_scores[:, 1:]
            teacher_margins = teacher_margins_raw * teacher_margin_scale
            if teacher_margin_clip > 0.0:
                teacher_margins = teacher_margins.clamp(-teacher_margin_clip, teacher_margin_clip)
            if query_anchor_coeff > 0.0:
                t_q_vecs = doc_splade.encode(q_texts, sc["query_max_length"]).float()
            enc_bs = sc.get("eval_batch_size", 32)
            p_vecs = torch.cat([
                doc_splade.encode(p_texts[i:i + enc_bs], sc["doc_max_length"]).float()
                for i in range(0, len(p_texts), enc_bs)
            ], dim=0)
            p_vecs_3d = p_vecs.view(B_actual, nway, -1)

        with autocast(enabled=fp16 and device.type == "cuda"):
            raw = query_model.mlm(input_ids=a_ids, attention_mask=a_mask).logits
            mask = a_mask.unsqueeze(-1).float()
            q_logits = (raw + (1.0 - mask) * -1e6).max(dim=1).values

            q_relu = q_logits + (F.relu(q_logits) - q_logits).detach()
            q_vecs = torch.log1p(q_relu)

            passage_vecs = p_vecs_3d.to(q_vecs.dtype)
            student_scores = (q_vecs.unsqueeze(1) * passage_vecs).sum(-1)
            student_margins = student_scores[:, :1] - student_scores[:, 1:]
            margin_loss = F.mse_loss(student_margins.float(), teacher_margins.float())
            train_loss = margin_loss
            if query_anchor_coeff > 0.0:
                anchor_loss = ((q_vecs - t_q_vecs.to(q_vecs.dtype)) ** 2).sum(dim=-1).mean()
                train_loss = train_loss + query_anchor_coeff * anchor_loss
            else:
                anchor_loss = q_vecs.new_tensor(0.0)

            if lambda_q > 0.0:
                lq_scale = min(1.0, step / lambda_q_warmup_steps) if lambda_q_warmup_steps > 0 else 1.0
                train_loss = train_loss + lq_scale * lambda_q * q_vecs.sum(-1).mean()

        is_accum_boundary = (step + 1) % grad_accum == 0 or (step + 1) == alignment_steps
        scaler.scale(train_loss / grad_accum).backward()
        if is_accum_boundary:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in query_model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad()

        if do_log:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "warm" if step < freeze_warmup_steps else "full"
            with torch.no_grad():
                q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
                flops = q_vecs.detach().abs().sum(-1).mean().item()
                s_rank1 = student_scores.detach().argmax(-1).eq(0).float().mean().item()
                t_rank1 = teacher_scores.argmax(-1).eq(0).float().mean().item()
                margin_mae = (student_margins.float() - teacher_margins.float()).abs().mean().item()
                teacher_margin_abs = teacher_margins.float().abs().mean().item()
                student_margin_abs = student_margins.float().abs().mean().item()
            print(
                f"[shallow-distill/{phase}] step {step+1:>6} | loss {train_loss.item():.4f} "
                f"(margin {margin_loss.item():.3f} anchor {anchor_loss.item():.3f} "
                f"mae {margin_mae:.3f} | |m| s={student_margin_abs:.2f} t={teacher_margin_abs:.2f}) | "
                f"q_nnz {q_nnz:.1f} | flops {flops:.1f} | "
                f"p@1 s={s_rank1:.2f} t={t_rank1:.2f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar(f"splade_shallow_align_distill/{phase}/loss", train_loss.item(), step + 1)
            writer.add_scalar(f"splade_shallow_align_distill/{phase}/margin_loss", margin_loss.item(), step + 1)
            writer.add_scalar(f"splade_shallow_align_distill/{phase}/anchor_loss", anchor_loss.item(), step + 1)
            writer.add_scalar("splade_shallow_align_distill/margin_mae", margin_mae, step + 1)
            writer.add_scalar("splade_shallow_align_distill/student_margin_abs", student_margin_abs, step + 1)
            writer.add_scalar("splade_shallow_align_distill/teacher_margin_abs", teacher_margin_abs, step + 1)
            writer.add_scalar("splade_shallow_align_distill/q_nnz", q_nnz, step + 1)
            writer.add_scalar("splade_shallow_align_distill/precision_at_1", s_rank1, step + 1)
            writer.add_scalar("splade_shallow_align_distill/teacher_precision_at_1", t_rank1, step + 1)
            writer.add_scalar("splade_shallow_align_distill/lr", lr, step + 1)
            t0 = time.time()

        if (step + 1) % sc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            if cfg.get("eval", {}).get("datasets"):
                print(f"[shallow-distill] Eval at step {step+1} …")
                import gc; gc.collect(); torch.cuda.empty_cache()
                query_model.eval()
                evaluate_asymmetric(
                    query_model, query_tokenizer, doc_splade, cfg, device,
                    writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                    section="splade_shallow_align_distill",
                )
                query_model.train()
                gc.collect(); torch.cuda.empty_cache()

    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": alignment_steps}, final_path)
    print(f"[shallow-distill] Done. Final checkpoint → {final_path}")
    writer.close()


def train_lion_shallow_align(cfg: dict, resume: str | None = None):
    """3-layer (or N-layer) Lion-SP-1B query encoder distilled against the full Lion doc encoder.

    Same recipe as train_splade_shallow_align but for Lion-SP-1B (Llama-3 decoder-only):
      - Phase 0: freeze embeddings + LM head, train only the kept body layers.
      - Phase 1: everything trains; head+embeddings at head_lr_scale * body_lr.
      - Loss: direct MSE on SPLADE vectors + L1 sparsity (lambda_q), STE through relu.

    Reads from the ``lion_shallow_align`` config section.
    """
    import gc
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn.functional as F

    from model import FrozenLionSPLADE, ShallowLionQuery
    from eval import evaluate_asymmetric
    from data import make_ranking_distill_loader

    if "lion_shallow_align" not in cfg:
        raise SystemExit("[lion-shallow] config.yaml is missing a `lion_shallow_align:` section.")
    sc = cfg["lion_shallow_align"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_layers            = int(sc["n_layers"])
    alignment_steps     = int(sc.get("alignment_steps", 50_000))
    freeze_warmup_steps = int(sc.get("freeze_warmup_steps", 5_000))
    head_lr_scale       = float(sc.get("head_lr_scale", 0.1))
    align_lr            = float(sc.get("alignment_lr", 2e-4))
    weight_decay        = float(sc.get("weight_decay", 0.01))
    fp16                = bool(sc.get("fp16", True))
    lambda_q              = float(sc.get("lambda_q", 0.0))
    lambda_q_warmup_steps = int(sc.get("lambda_q_warmup_steps", 0))
    nway                  = int(sc.get("nway", 8))
    grad_accum          = int(sc.get("gradient_accumulation_steps", 1))

    print(
        f"[lion-shallow] lion={sc['lion_hf_id']} | n_layers={n_layers} "
        f"| freeze_warmup={freeze_warmup_steps} | alignment={alignment_steps} "
        f"| lr={align_lr} | head_lr_scale={head_lr_scale} "
        f"| nway={nway} | lambda_q={lambda_q} | grad_accum={grad_accum}"
    )

    print(f"[lion-shallow] Loading frozen Lion doc encoder …")
    lion_doc = FrozenLionSPLADE(sc["lion_hf_id"]); lion_doc.to(device); lion_doc.eval()

    print(f"[lion-shallow] Building {n_layers}-layer shallow Lion query encoder …")
    query_model = ShallowLionQuery(sc["lion_hf_id"], n_layers=n_layers)
    query_model.to(device)
    query_tokenizer = query_model.tokenizer
    n_total = sum(p.numel() for p in query_model.parameters())
    print(
        f"[lion-shallow] Query model: {n_total/1e6:.1f}M params "
        f"({n_layers}/{query_model.original_n_layers} layers kept)."
    )

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        query_model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"[lion-shallow] Resumed from {resume} at step {start_step}")

    in_warmup = start_step < freeze_warmup_steps
    if in_warmup:
        query_model.freeze_for_warmup()
        print(
            f"[lion-shallow] Phase 0 (frozen warmup). Trainable: "
            f"{query_model.trainable_param_count()/1e6:.1f}M / {n_total/1e6:.1f}M"
        )
    else:
        query_model.unfreeze_all()
        print(f"[lion-shallow] Resuming directly into Phase 1.")

    optimizer = torch.optim.AdamW(
        [p for p in query_model.parameters() if p.requires_grad],
        lr=align_lr, weight_decay=weight_decay,
    )
    if start_step > 0:
        for pg in optimizer.param_groups:
            pg["initial_lr"] = align_lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=alignment_steps, eta_min=1e-5,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    # bfloat16 has the same exponent range as float32; no loss scaling needed.
    scaler = GradScaler(enabled=False)
    use_bf16 = fp16 and device.type == "cuda"

    out_dir = Path("checkpoints_lion_shallow") / sc.get("output_dir", "lion_shallow_align")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")

    ranking_loader = make_ranking_distill_loader(nway=nway, batch_size=sc["batch_size"])

    # Initial eval
    print(f"[lion-shallow] Initial eval …")
    query_model.eval()
    evaluate_asymmetric(
        query_model, query_tokenizer, lion_doc, cfg, device,
        writer=writer, step=0, run_doc_doc=True, override_k=0,
        section="lion_shallow_align",
    )
    gc.collect(); torch.cuda.empty_cache()

    query_model.train()
    t0 = time.time()
    print(f"[lion-shallow] Training loop ({alignment_steps - start_step} steps remaining) …")
    optimizer.zero_grad()

    for step in range(start_step, alignment_steps):
        # ── Phase boundary ────────────────────────────────────────────
        if in_warmup and step >= freeze_warmup_steps:
            # embed_tokens and lm_head are tied (262M shared params). Unfreezing
            # them adds ~1GB of AdamW states which OOMs on 8GB GPUs. Only add the
            # final layer norm — it's tiny and lets the body-to-head interface adapt.
            for p in query_model.model.model.norm.parameters():
                p.requires_grad_(True)
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer = torch.optim.AdamW(
                [p for p in query_model.parameters() if p.requires_grad],
                lr=current_lr, weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=alignment_steps - freeze_warmup_steps, eta_min=1e-5,
            )
            optimizer.zero_grad()
            in_warmup = False
            print(f"[lion-shallow] Step {step}: switched to Phase 1 (body + norm). lr={current_lr:.2e}")

        do_log = (step + 1) % sc["log_every"] == 0
        q_texts, p_texts = next(ranking_loader)
        B_actual = len(q_texts)
        q_enc = query_tokenizer(
            q_texts, max_length=sc["query_max_length"],
            truncation=True, padding=True, return_tensors="pt",
        )
        a_ids = q_enc["input_ids"].to(device)
        a_mask = q_enc["attention_mask"].to(device)

        with torch.no_grad():
            t_q_vecs = lion_doc.encode(q_texts, sc["query_max_length"]).float()
            if do_log:
                enc_bs = sc.get("eval_batch_size", 8)
                p_vecs = torch.cat([
                    lion_doc.encode(p_texts[i:i+enc_bs], sc["doc_max_length"]).float()
                    for i in range(0, len(p_texts), enc_bs)
                ], dim=0)
                p_vecs_3d = p_vecs.view(B_actual, nway, -1)
                t_scores = (t_q_vecs.unsqueeze(1) * p_vecs_3d).sum(-1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _raw = query_model.model(
                input_ids=a_ids, attention_mask=a_mask, use_cache=False
            ).logits  # [B, L, V]
            scale = query_model.hidden_size ** -0.25
            _m = a_mask.unsqueeze(-1).float()
            q_logits = (_raw * scale + (1.0 - _m) * -1e6).max(dim=1).values  # [B, V]

            # STE: forward = real SPLADE vecs, backward = identity through relu
            q_relu = q_logits + (F.relu(q_logits) - q_logits).detach()
            q_vecs = torch.log1p(q_relu)

            rank_loss = ((q_vecs - t_q_vecs.to(q_vecs.dtype)) ** 2).sum(dim=-1).mean()
            align_loss = rank_loss
            if lambda_q > 0.0:
                lq_scale = min(1.0, step / lambda_q_warmup_steps) if lambda_q_warmup_steps > 0 else 1.0
                align_loss = align_loss + lq_scale * lambda_q * q_vecs.sum(-1).mean()

        is_accum_boundary = (step + 1) % grad_accum == 0 or (step + 1) == alignment_steps
        scaler.scale(align_loss / grad_accum).backward()
        if is_accum_boundary:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in query_model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad()

        if do_log:
            with torch.no_grad():
                q_nnz = (q_vecs.detach() > 0).float().mean(0).sum().item()
                t_nnz = (t_q_vecs > 0).float().mean(0).sum().item()
                flops = q_vecs.detach().sum(-1).mean().item()
                lr = optimizer.param_groups[0]["lr"]
                phase = "warm" if in_warmup else "full"
                if do_log and p_texts:
                    s_scores = (q_vecs.detach().float().unsqueeze(1) * p_vecs_3d.to(q_vecs.dtype)).sum(-1)
                    s_rank1 = (s_scores.argmax(-1) == 0).float().mean().item()
                    t_rank1 = (t_scores.argmax(-1) == 0).float().mean().item()
                else:
                    s_rank1 = t_rank1 = float("nan")
            elapsed = time.time() - t0; t0 = time.time()
            print(
                f"[lion-shallow/{phase}] step {step+1:>6} | loss {align_loss.item():.4f} "
                f"(rank {rank_loss.item():.3f}) | q_nnz {q_nnz:.1f} (teacher {t_nnz:.1f}) "
                f"| flops {flops:.1f} | p@1 s={s_rank1:.2f} t={t_rank1:.2f} "
                f"| lr {lr:.2e} | {elapsed:.0f}s"
            )
            writer.add_scalar("lion_shallow_align/loss", align_loss.item(), step + 1)
            writer.add_scalar("lion_shallow_align/rank_loss", rank_loss.item(), step + 1)
            writer.add_scalar("lion_shallow_align/q_nnz", q_nnz, step + 1)
            writer.add_scalar("lion_shallow_align/teacher_nnz", t_nnz, step + 1)
            writer.add_scalar("lion_shallow_align/lr", lr, step + 1)

        if (step + 1) % sc["save_every"] == 0:
            ckpt_path = out_dir / f"align_step_{step+1}.pt"
            torch.save({"model": query_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
            query_model.eval()
            evaluate_asymmetric(
                query_model, query_tokenizer, lion_doc, cfg, device,
                writer=writer, step=step + 1, run_doc_doc=False, override_k=0,
                section="lion_shallow_align",
            )
            query_model.train()
            gc.collect(); torch.cuda.empty_cache()

    final_path = out_dir / "align_final.pt"
    torch.save({"model": query_model.state_dict(), "step": alignment_steps}, final_path)
    print(f"[lion-shallow] Done. Final checkpoint → {final_path}")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train SAE-SPLADE with ettin-17m")
    parser.add_argument(
        "stage",
        choices=[
            "sae", "splade", "asymmetric", "projected",
            "vocab_transplant", "vocab_transplant_align", "lion_transplant_align",
            "random_init_align", "doc_head_align", "direct_align",
            "splade_shallow_align", "splade_shallow_factorized_align",
            "splade_shallow_factorized_spaced_align",
            "splade_shallow_align_distill", "lion_shallow_align",
        ],
        help="Training stage to run.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--init-from", default=None, dest="init_from",
                        help="Load model weights from checkpoint but reset step counter to 0")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage == "sae":
        train_sae(cfg, resume=args.resume)
    elif args.stage == "splade":
        cfg["splade"]["_sae_cfg"] = cfg["sae"]
        train_splade(cfg, resume=args.resume)
    elif args.stage == "asymmetric":
        train_asymmetric(cfg, resume=args.resume)
    elif args.stage == "projected":
        train_projected(cfg, resume=args.resume)
    elif args.stage == "vocab_transplant_align":
        train_vocab_transplant_align(cfg, resume=args.resume)
    elif args.stage == "lion_transplant_align":
        train_lion_transplant_align(cfg, resume=args.resume)
    elif args.stage == "random_init_align":
        train_random_init_align(cfg, resume=args.resume)
    elif args.stage == "doc_head_align":
        train_doc_head_align(cfg, resume=args.resume)
    elif args.stage == "direct_align":
        train_direct_align(cfg, resume=args.resume)
    elif args.stage == "splade_shallow_align":
        train_splade_shallow_align(cfg, resume=args.resume, init_from=args.init_from)
    elif args.stage == "splade_shallow_factorized_align":
        train_splade_shallow_factorized_align(cfg, resume=args.resume, init_from=args.init_from)
    elif args.stage == "splade_shallow_factorized_spaced_align":
        train_splade_shallow_factorized_spaced_align(cfg, resume=args.resume, init_from=args.init_from)
    elif args.stage == "splade_shallow_align_distill":
        train_splade_shallow_align_distill(cfg, resume=args.resume, init_from=args.init_from)
    elif args.stage == "lion_shallow_align":
        train_lion_shallow_align(cfg, resume=args.resume)
    else:
        train_vocab_transplant(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
