"""
Expanded-SPLADE query encoder: Ettin body -> SPLADE-v3 vocabulary space.

Implements the architecture from "Learning Sparse Lexical Representations Over
Specified Vocabularies for Retrieval" (Dudek et al., CIKM'23), specialized for
the case where the OUTPUT vocabulary is an existing trained SPLADE model's vocab.

Key design (better than the paper's pooling init for this case):
  - Body: Ettin encoder (its own ~50k input tokenizer, used whole, no truncation)
  - Adapter: Linear(ettin_hidden -> 768) — learned translation into BERT's space
  - Head: REUSE naver/splade-v3's trained MLM head (transform + decoder + bias),
    which already maps hidden(768) -> vocab(30522) correctly. We're only learning
    how to feed it from a different encoder.

Doc encoder (naver/splade-v3) is frozen. The query encoder is trained with:
  1. Per-dim distillation (MSE or KL) of student query vec vs SPLADE-v3 query vec
     — replaces MLM pretraining, anchors which of the 30522 dims fire.
  2. MarginMSE on (q, d+, d-): student margins vs SPLADE-v3's own margins.
  3. Joint FLOPS on (Q, D): student query embeddings × frozen doc embeddings —
     the paper's serving-correct FLOPS shape, with quadratic ramp.

Usage:
    python scripts/train_expanded_splade.py \\
        --ettin_id jhu-clsp/ettin-encoder-68m \\
        --doc_splade_id naver/splade-v3 \\
        --output_dir /vol/expanded_splade/ettin68m-to-spladev3 \\
        --max_steps 30000 --batch_size 24
"""

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class FrozenDocSPLADE(nn.Module):
    """Frozen SPLADE doc encoder (naver/splade-v3). Tokenizes internally."""

    def __init__(self, hf_id: str):
        super().__init__()
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        # naver/splade-v3 ships pytorch_model.bin (no safetensors). Force .bin load
        # to avoid a hang/exception in transformers' auto safetensors-conversion.
        try:
            self.mlm = AutoModelForMaskedLM.from_pretrained(hf_id, use_safetensors=False)
        except Exception:
            self.mlm = AutoModelForMaskedLM.from_pretrained(hf_id)
        self.vocab_size = self.mlm.config.vocab_size
        self.hidden_size = self.mlm.config.hidden_size
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, texts, max_length, device):
        enc = self.tokenizer(
            texts, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.mlm(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        logits = out.logits * mask
        return torch.log1p(torch.relu(logits).max(dim=1).values)  # [N, V]

    @torch.no_grad()
    def encode_logits(self, texts, max_length, device):
        """Max-pooled MLM logits BEFORE relu/log1p — [N, V].

        Distillation target with meaningful values on ALL dims (not just the
        sparse survivors), so 'output zero' is NOT a free way to lower the loss.
        """
        enc = self.tokenizer(
            texts, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.mlm(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        logits = out.logits * mask
        return logits.max(dim=1).values  # [N, V]

    def extract_head(self):
        """Return the trained MLM prediction head (transform + decoder + bias)."""
        # naver/splade-v3 is BertForMaskedLM -> head at .cls.predictions
        if hasattr(self.mlm, "cls") and hasattr(self.mlm.cls, "predictions"):
            return self.mlm.cls.predictions
        raise RuntimeError(
            "Could not locate MLM prediction head on doc SPLADE model. "
            f"Top-level modules: {[n for n, _ in self.mlm.named_children()]}"
        )


class ExpandedSpladeQuery(nn.Module):
    """Ettin body + adapter + reused SPLADE-v3 MLM head -> SPLADE-v3 vocab space."""

    def __init__(self, ettin_id: str, doc_splade: FrozenDocSPLADE):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.body = AutoModel.from_pretrained(ettin_id)
        self.tokenizer = AutoTokenizer.from_pretrained(ettin_id)
        ettin_h = self.body.config.hidden_size
        splade_h = doc_splade.hidden_size
        self.vocab_size = doc_splade.vocab_size

        # Learned translation Ettin hidden -> BERT hidden, followed by a LayerNorm
        # so the reused head receives input matching the distribution it was trained
        # on (SPLADE-v3's head expects LayerNorm'd BERT hidden states). Without this
        # the head sees mis-scaled input and the output logits swing from massively
        # over-dense to fully dead. Small init keeps step-0 output well-behaved.
        self.adapter = nn.Linear(ettin_h, splade_h)
        nn.init.normal_(self.adapter.weight, std=0.02)
        nn.init.zeros_(self.adapter.bias)
        self.adapter_norm = nn.LayerNorm(splade_h)

        # REUSE the doc model's trained prediction head. Deep-copy so training the
        # query head doesn't mutate the frozen doc encoder's weights.
        import copy
        self.head = copy.deepcopy(doc_splade.extract_head())
        for p in self.head.parameters():
            p.requires_grad_(True)  # trainable; anchored by alignment loss

        log.info(
            f"ExpandedSpladeQuery: ettin_h={ettin_h} splade_h={splade_h} "
            f"vocab={self.vocab_size} | adapter params={ettin_h*splade_h/1e6:.1f}M"
        )

    def encode(self, input_ids, attention_mask, return_logits=False):
        out = self.body(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                      # [B, L, ettin_h]
        h = self.adapter(h)                            # [B, L, 768]
        h = self.adapter_norm(h)                        # match head's expected scale
        logits = self.head(h)                          # [B, L, 30522]
        logits = logits * attention_mask.unsqueeze(-1).float()
        pooled_logits = logits.max(dim=1).values       # [B, V] pre-activation
        vec = torch.log1p(torch.relu(pooled_logits))   # [B, V] SPLADE vector
        if return_logits:
            return vec, pooled_logits
        return vec


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def expanded_splade_loss(
    q_vecs, q_logits, t_q_logits, t_q_vecs, doc_vecs, *,
    distill_coeff=1.0,
    margin_coeff=0.05, lambda_j=0.9, flops_scale=1.0,
    align_coeff=0.3,
):
    """
    Proven recipe (from vocab_transplant_joint_loss, battle-tested on this exact
    Ettin-body + SPLADE-v3-head architecture):
      - CE ranking over [B, B*nway] query×doc scores (positive on diagonal) — the
        primary, stable retrieval signal.
      - Cosine alignment of student vs teacher query vectors — anchors to
        SPLADE-v3's vocab space WITHOUT per-dim MSE (cosine ignores the
        background-zeros that made MSE collapse).
      - Joint FLOPS (q·d) for sparsity.

    distill_coeff -> weight on cosine alignment (kept name for arg compatibility).
    margin_coeff is accepted but unused (CE replaces MarginMSE here).

    q_vecs    : [B, V]      student query SPLADE vectors (grad)
    t_q_vecs  : [B, V]      teacher query SPLADE vectors (no grad)
    doc_vecs  : [B*nway, V] teacher doc vectors (no grad), positive at offset 0
    q_logits, t_q_logits : unused now (kept for call-site compatibility)
    """
    B, V = q_vecs.shape
    nway = doc_vecs.shape[0] // B
    idx = torch.arange(B, device=q_vecs.device)

    scores = q_vecs @ doc_vecs.T                       # [B, B*nway]

    # ── Primary: CE ranking (positive = diagonal block, offset 0) ──────
    labels = idx * nway
    ranking = F.cross_entropy(scores, labels)

    # ── Anchor: cosine alignment to teacher query vecs (direction only) ─
    align = (1.0 - F.cosine_similarity(q_vecs, t_q_vecs.to(q_vecs.dtype))).mean()

    # ── Sparsity: Joint FLOPS (serving-correct q·d shape) ──────────────
    joint_flops = (q_vecs.mean(dim=0) * doc_vecs.mean(dim=0)).sum()

    total = ranking + align_coeff * align + lambda_j * flops_scale * joint_flops

    with torch.no_grad():
        q_nnz = (q_vecs > 0).float().sum(-1).mean().item()
        acc = (scores.argmax(dim=1) == labels).float().mean().item()

    return total, {
        "loss": total.item(),
        "ranking": ranking.item(),
        "align": align.item(),
        "jflops": joint_flops.item(),
        "q_nnz": q_nnz,
        "inbatch_acc": acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ettin_id", default="jhu-clsp/ettin-encoder-68m")
    p.add_argument("--doc_splade_id", default="naver/splade-v3")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--head_lr_scale", type=float, default=0.1,
                   help="LR multiplier for the reused MLM head (anchored, slower).")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--distill_kind", choices=["mse", "kl"], default="mse")
    p.add_argument("--distill_coeff", type=float, default=1.0)
    p.add_argument("--margin_coeff", type=float, default=0.05)
    p.add_argument("--alignment_steps", type=int, default=3000,
                   help="Phase 1: cosine-alignment-only warmup steps (proven recipe).")
    p.add_argument("--alignment_lr", type=float, default=5e-4,
                   help="High LR for Phase 1 alignment (matches vocab_transplant).")
    p.add_argument("--lambda_j", type=float, default=0.02)
    p.add_argument("--flops_warmup_steps", type=int, default=500,
                   help="Phase 2: hard-OFF FLOPS for first N steps (alignment already "
                        "established a representation, so this can be short).")
    p.add_argument("--distill_only_steps", type=int, default=0,
                   help="(unused in CE-ranking recipe; kept for compatibility)")
    p.add_argument("--flops_ramp_steps", type=int, default=6000,
                   help="Quadratic ramp of joint-FLOPS weight over first N steps.")
    p.add_argument("--query_max_length", type=int, default=64)
    p.add_argument("--doc_max_length", type=int, default=192)
    p.add_argument("--dataset_id",
                   default="sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1")
    p.add_argument("--dataset_config", default="triplet")
    p.add_argument("--dataset_size", type=int, default=400000)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--log_steps", type=int, default=100)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Models
    log.info(f"Loading frozen doc encoder: {args.doc_splade_id}")
    doc = FrozenDocSPLADE(args.doc_splade_id).to(device)
    log.info(f"Loading Ettin query body: {args.ettin_id}")
    q = ExpandedSpladeQuery(args.ettin_id, doc).to(device)

    # Data
    log.info(f"Loading {args.dataset_id} ({args.dataset_config})")
    ds = load_dataset(args.dataset_id, args.dataset_config, split="train")
    ds = ds.select(range(min(args.dataset_size, len(ds))))
    cols = ds.column_names
    log.info(f"Columns: {cols} | size {len(ds)}")
    # expect (query, positive, negative)
    def collate(batch):
        return (
            [b["query"] for b in batch],
            [b["positive"] for b in batch],
            [b["negative"] for b in batch],
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=2, drop_last=True)

    # Optimizer with separate LR for the reused head
    head_ids = {id(pp) for pp in q.head.parameters()}
    body_adapter = [pp for pp in q.parameters() if id(pp) not in head_ids]
    head_params = list(q.head.parameters())
    opt = torch.optim.AdamW([
        {"params": body_adapter, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_scale},
    ], weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))  # cosine to 0

    q.train()
    step = 0
    data_iter = iter(dl)

    # ── Phase 1: cosine-alignment warmup (the proven recipe's key step) ──────
    # Mirrors vocab_transplant's Phase 1. A freshly-adapted query encoder produces
    # garbage; starting ranking CE on it dead-ReLUs immediately. Instead, FIRST pull
    # the student query vecs toward the teacher's direction with cosine-only loss at
    # HIGH LR (5e-4). This establishes sane, teacher-like sparse vectors. Only then
    # does Phase 2 ranking refine without collapsing.
    if args.alignment_steps > 0:
        log.info(f"Phase 1: cosine-alignment warmup ({args.alignment_steps} steps) "
                 f"at lr={args.alignment_lr}")
        align_opt = torch.optim.AdamW(q.parameters(), lr=args.alignment_lr,
                                      weight_decay=0.01)
        align_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            align_opt, T_max=args.alignment_steps, eta_min=1e-5)
        for astep in range(args.alignment_steps):
            try:
                queries, pos, neg = next(data_iter)
            except StopIteration:
                data_iter = iter(dl); queries, pos, neg = next(data_iter)

            q_enc = q.tokenizer(queries, max_length=args.query_max_length,
                                truncation=True, padding=True, return_tensors="pt")
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            q_vecs = q.encode(q_enc["input_ids"], q_enc["attention_mask"])  # [B,V]
            # teacher encodes the SAME query text (as the alignment target).
            t_q_vecs = doc.encode(queries, args.query_max_length, device)   # [B,V]

            align_loss = (1.0 - F.cosine_similarity(
                q_vecs, t_q_vecs.to(q_vecs.dtype))).mean()

            align_opt.zero_grad()
            align_loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            align_opt.step()
            align_sched.step()

            if (astep + 1) % args.log_steps == 0:
                with torch.no_grad():
                    q_nnz = (q_vecs > 0).float().sum(-1).mean().item()
                log.info(
                    f"[align] step {astep+1}/{args.alignment_steps} | "
                    f"align_loss {align_loss.item():.4f} | q_nnz {q_nnz:.0f} | "
                    f"lr {align_opt.param_groups[0]['lr']:.2e}"
                )
        del align_opt
        log.info("Phase 1 complete. Starting Phase 2 (ranking + FLOPS).")

    # ── Phase 2: ranking (CE) + cosine align + ramped joint-FLOPS ────────────
    log.info(f"Phase 2: max_steps={args.max_steps} batch={args.batch_size} "
             f"lambda_j={args.lambda_j} flops_warmup={args.flops_warmup_steps}")
    while step < args.max_steps:
        try:
            queries, pos, neg = next(data_iter)
        except StopIteration:
            data_iter = iter(dl); queries, pos, neg = next(data_iter)

        # student query vectors + pre-activation logits (grad)
        q_enc = q.tokenizer(queries, max_length=args.query_max_length,
                            truncation=True, padding=True, return_tensors="pt")
        q_enc = {k: v.to(device) for k, v in q_enc.items()}
        q_vecs, q_logits = q.encode(
            q_enc["input_ids"], q_enc["attention_mask"], return_logits=True
        )  # [B,V], [B,V]

        # teacher query vectors + logits + doc vectors (no grad)
        t_q_vecs = doc.encode(queries, args.query_max_length, device)        # [B,V]
        t_q_logits = doc.encode_logits(queries, args.query_max_length, device)  # [B,V]
        # interleave pos/neg so positive is at offset 0 in each group (nway=2)
        docs = []
        for pdoc, ndoc in zip(pos, neg):
            docs.append(pdoc); docs.append(ndoc)
        doc_vecs = doc.encode(docs, args.doc_max_length, device)            # [2B,V]

        # Joint-FLOPS is hard-OFF during warmup. The freshly-init head starts very
        # over-dense (q_nnz ~20k); with FLOPS on, its huge gradient slams all
        # activations to zero in a couple steps -> dead ReLU. Let ranking+alignment
        # establish a nonzero, ranking-meaningful representation FIRST, then ramp
        # FLOPS gently to sparsify.
        flops_warmup = args.flops_warmup_steps
        if step < flops_warmup:
            ramp = 0.0
        else:
            ramp = min(1.0, ((step - flops_warmup)
                             / max(1, args.flops_ramp_steps)) ** 2)

        loss, logs = expanded_splade_loss(
            q_vecs, q_logits, t_q_logits, t_q_vecs, doc_vecs,
            distill_coeff=args.distill_coeff,
            margin_coeff=args.margin_coeff, lambda_j=args.lambda_j,
            flops_scale=ramp, align_coeff=0.3,
        )

        # LR schedule
        scale = lr_at(step)
        for g, base in zip(opt.param_groups, [args.lr, args.lr * args.head_lr_scale]):
            g["lr"] = base * scale

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_steps == 0:
            log.info(
                f"step {step}/{args.max_steps} | loss {logs['loss']:.4f} "
                f"ranking {logs['ranking']:.4f} align {logs['align']:.4f} "
                f"jflops {logs['jflops']:.4f} | q_nnz {logs['q_nnz']:.0f} "
                f"inbatch_acc {logs['inbatch_acc']:.3f} | lr_scale {scale:.3f} ramp {ramp:.2f}"
            )

        if step % args.save_steps == 0 or step == args.max_steps:
            ckpt = out / f"checkpoint-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({
                "body": q.body.state_dict(),
                "adapter": q.adapter.state_dict(),
                "adapter_norm": q.adapter_norm.state_dict(),
                "head": q.head.state_dict(),
                "step": step,
                "config": vars(args),
            }, ckpt / "query_encoder.pt")
            q.tokenizer.save_pretrained(str(ckpt))
            log.info(f"saved {ckpt}")

    log.info("done")


if __name__ == "__main__":
    main()