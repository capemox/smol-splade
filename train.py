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
    ckpt_path = ac["sae_checkpoint"]
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
    out_dir = Path(ac["output_dir"])
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

    out_dir = Path(pc["output_dir"])
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

    from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE, vocab_transplant_splade_loss, projected_alignment_loss
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
    transplant_dir = vc["transplant_dir"]
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
                    yield collate_asymmetric_batch(
                        buf, query_tokenizer, vc["query_max_length"], device
                    )
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

    out_dir = Path(vc["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tensorboard")
    batches = data_iter()

    # ── Phase 1: cosine alignment warmup ──────────────────────────────
    alignment_steps = vc.get("alignment_steps", 0)
    if alignment_steps > 0 and start_step == 0:
        print(f"[VT] Phase 1: Cosine alignment warm-up ({alignment_steps} steps) …")
        align_optimizer = torch.optim.AdamW(
            query_model.parameters(), lr=vc.get("alignment_lr", 1e-3), weight_decay=vc["weight_decay"]
        )
        align_loader = make_alignment_loader(
            cfg["sae"]["corpus_dataset"],
            cfg["sae"].get("corpus_text_field", "text"),
            query_tokenizer,
            vc["batch_size"],
            vc["doc_max_length"],
            device,
        )
        query_model.train()
        t0 = time.time()
        for astep in range(alignment_steps):
            a_ids, a_mask, texts = next(align_loader)
            align_optimizer.zero_grad()
            with autocast(enabled=vc["fp16"] and device.type == "cuda"):
                align_loss = projected_alignment_loss(
                    query_model, doc_splade, a_ids, a_mask, texts, vc["doc_max_length"]
                )
            scaler.scale(align_loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(align_optimizer)
            torch.nn.utils.clip_grad_norm_(list(query_model.parameters()), 1.0)
            scaler.step(align_optimizer)
            scaler.update()

            if (astep + 1) % vc["log_every"] == 0:
                elapsed = time.time() - t0
                print(
                    f"[VT-align] step {astep+1:>6} | align_loss {align_loss.item():.4f} | {elapsed:.0f}s"
                )
                writer.add_scalar("vocab_transplant/align_loss", align_loss.item(), astep + 1)
                t0 = time.time()

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

        q_ids, q_mask, doc_texts, teacher_scores = next(batches)

        # Doc encoding always runs in fp32 — BERT is not fp16-stable under gradients.
        # Gradient flows when LoRA is active; no_grad otherwise.
        doc_vecs = doc_splade.encode(doc_texts, vc["doc_max_length"], no_grad=not lora_enabled)

        optimizer.zero_grad()
        if lora_enabled:
            lora_optimizer.zero_grad()
        with autocast(enabled=vc["fp16"] and device.type == "cuda"):
            loss, metrics = vocab_transplant_splade_loss(
                query_model, q_ids, q_mask, doc_vecs, teacher_scores,
                lambda_q=vc["lambda_q"],
                flops_scale=flops_scale,
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
                f"loss {metrics['loss']:.4f} | retr {metrics['retr']:.4f} | "
                f"flops {metrics['flops']:.4f} | q_nnz {metrics['avg_q_nnz']:.1f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s"
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
    parser.add_argument("stage", choices=["sae", "splade", "asymmetric", "projected", "vocab_transplant"], help="Training stage to run")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
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
    else:
        train_vocab_transplant(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
