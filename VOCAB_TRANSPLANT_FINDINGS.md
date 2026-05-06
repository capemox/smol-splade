# Vocab-Transplant Asymmetric SPLADE: Findings & Publishability Assessment

## Core Idea

Train a small, efficient query encoder that produces sparse SPLADE-compatible vectors
by "transplanting" the vocabulary of a large frozen doc SPLADE onto the small model.
After transplant the query encoder operates in the same vocabulary space as the doc
encoder — their vectors can be dot-producted directly without any projection layer.

The goal is asymmetric retrieval where:
- **Doc encoder**: `naver/splade-v3` (large, frozen, used as-is at inference)
- **Query encoder**: `jhu-clsp/ettin-encoder-17m` or `ettin-encoder-68m` (small, trained fresh)

---

## Motivation

Standard SPLADE models use the same encoder for both queries and documents. This is
expensive at query time because:
1. You need to run the full model (e.g., DistilBERT-based) on every query at serving time.
2. Sparse vectors carry different structural biases for queries vs. documents, but the
   model is forced to produce both from the same weights.

The hypothesis: a smaller query encoder fine-tuned specifically for the query distribution
can match or beat a symmetric large SPLADE on NDCG@10, while being 4–10x cheaper to serve.

---

## Problem Statement

Two sub-problems to solve:

**Problem 1 — Vocabulary mismatch**: If we take a small model (ettin-17m has its own
tokenizer and embedding space) and a large SPLADE (naver/splade-v3, DistilBERT-based),
their output vectors are in different vocabulary spaces. You cannot dot-product them.

**Problem 2 — Query-doc asymmetry in SPLADE**: SPLADE-v3 is trained as a document encoder.
When you run it on short query texts it produces near-zero sparse vectors, because SPLADE
scoring functions learned from doc-length texts at training time. This makes it a bad
"teacher" for query-side self-distillation.

---

## Approaches Tried (in order)

### Approach 1 — Symmetric SAE-SPLADE (baseline)
Train a TopK Sparse Autoencoder on top of ettin-17m backbone embeddings, then fine-tune
end-to-end as SPLADE using ColBERTv2 hard negatives.

- Both query and doc go through the same ettin-17m + SAE pipeline.
- Compatible: same vocab, same SAE width.
- Works but doesn't solve the serving-cost problem.

### Approach 2 — Asymmetric SAE-SPLADE
Use a frozen `naver/splade-v3` doc encoder and train only the ettin-17m SAE as the query
encoder. The SAE width must exactly match the doc encoder's vocab size (30522 for DistilBERT).

Key innovation here: k-annealing. Start with k=256 (dense) so the SAE latents can
align with the 30522-dim vocabulary, then linearly decay to k=32 over 30k steps.
Rationale: at k=32 the gradient from doc-dot-product is vanishing sparse, so you need
high k initially to get a strong enough signal.

Findings: achieves retrieval but the SAE bottleneck limits quality vs. symmetric SPLADE.

### Approach 3 — Projected Query Encoder
Small backbone (ettin-17m) → Linear projection → frozen MLM head from naver/splade-v3.

Architecture:
```
backbone → Linear(backbone_H → splade_H) → GELU → LayerNorm → Linear(splade_H → splade_H)
         → frozen MLM head → SPLADE max-pool
```

Phase 1 alignment warmup trains only the projection layer on MSE loss against the
doc SPLADE output on the same corpus texts.

Problem: even with alignment the quality gap to symmetric SPLADE is large, and the
projection layer is a hard representational bottleneck.

### Approach 4 — Vocab-Transplant (current, primary approach)
Instead of projecting, surgically transfer the doc SPLADE's tokenizer and embedding
matrix onto the small backbone. After this "transplant," the small model operates
natively in the doc encoder's vocabulary space — no projection needed.

### Approach 5 — Direct Alignment (bert-small, no transplant)
bert-small (`google/bert_uncased_L-4_H-512_A-8`, ~29M) already shares DistilBERT's
vocabulary, so no transplant is needed. Trained with KD warmup (10k steps, T=4.0)
to prevent dead-dim collapse, followed by cosine alignment (50k steps).

Motivation: removes tokensurgeon from the equation entirely. If this matches
vocab_transplant_align then the transplant adds nothing beyond getting the vocab size right.

Result: underperforms vocab_transplant_align with ettin-17m. The BERT architecture
also appears less efficient for this setup than the ettin family.

### Approach 6 — Random-Init Alignment (tokensurgeon ablation)
Same architecture as vocab_transplant_align (ettin backbone, resized to 30522 tokens,
donor tokenizer) but the embedding table is randomly initialized with N(0, 0.02) instead
of being seeded by tokensurgeon's kNN interpolation. KD warmup (10k steps) + cosine
alignment (50k steps) with otherwise identical hyperparameters.

Motivation: isolates whether tokensurgeon's embedding initialization is the critical
ingredient, or whether just having the right vocabulary size and architecture is enough.

Result: **fails**. Even with KD warmup preventing dead-dim collapse, the model does not
reach vocab_transplant_align quality. Tokensurgeon initialization is load-bearing.

### Approach 7 — Doc-Head Alignment (frozen MLM head from doc SPLADE)
ettin-17m backbone → linear projection (256→768) → frozen naver/splade-v3 MLM head.
The pre-trained head already knows which vocabulary dimensions to activate; only the
backbone and projection are trained. KD warmup (10k steps) + cosine alignment (50k steps).

Motivation: test whether borrowing the doc encoder's pre-trained output head (rather
than a randomly initialized one) provides enough signal without the full transplant.

Result: **fails**. Having the correct MLM head in isolation is not sufficient — without
tokensurgeon initializing the embedding table, the backbone cannot learn to produce
hidden states the frozen head can interpret well.

---

## Vocab Transplant: Technical Implementation

### Step 1 — Tokensurgeon-style Embedding Transfer

For each token in the donor vocab (naver/splade-v3 = DistilBERT, 30,522 tokens):

1. **Exact match** (token exists in both vocabs): copy the query model's embedding directly.
2. **No match** (donor-only token): approximate using cosine-NN interpolation.
   - Find k=64 nearest neighbors of the donor token's embedding among shared tokens
     (in donor embedding space).
   - Compute distance-proportional weights: `w ∝ 1 / (1 − cosine_sim + ε)`.
   - Apply those weights to the corresponding query embeddings (in query embedding space).
   - This preserves the relational geometry of the donor vocabulary in the query model's
     hidden space.

```python
# Distance-proportional weights
weights = 1.0 / (1.0 - topk_vals.clamp(max=0.9999) + 1e-6)
weights = weights / weights.sum(dim=-1, keepdim=True)
approx = (weights.unsqueeze(-1) * query_shared[topk_idx]).sum(dim=1)
```

3. Resize the query model's embedding matrix to `donor_vocab_size` (30,522).
4. Replace special-token IDs in config to match the donor tokenizer's values.
5. Save the transplanted model with the donor tokenizer — it can be loaded as
   `AutoModelForMaskedLM` and produces logits in the donor vocabulary space.

**Key property**: after transplant, running SPLADE max-pool on the query model's logits
gives a vector in the exact same 30,522-dimensional space as `naver/splade-v3`'s output.
No adapter, no projection — pure dot-product compatible.

Overlap observed: ettin-17m shares ~50–60% of its tokens with DistilBERT's vocabulary
(subword tokenizers derived from similar training data). The non-shared ~40% are
approximated by interpolation.

### Step 2 — Cosine Alignment Warmup

Before ranking fine-tuning, align the query encoder's output distribution to match
the doc SPLADE's output on the same query texts:

```
align_loss = mean(1 - cosine_similarity(query_model(query), doc_splade(query)))
```

Key design choices:
- **Uses training queries** (not corpus passages): targets the actual query distribution
  seen at inference time. Earlier experiments on passages gave ~0.61 NDCG quickly but
  did not transfer well to the query-doc retrieval task.
- **Cosine annealing LR**: `alignment_lr → 1e-5` over `alignment_steps` to avoid
  getting stuck in the plateau region (loss ≈ 0.2–0.3 with constant LR).
- Saves checkpoints every `save_every` steps so alignment quality can be evaluated
  independently on NanoBEIR (NDCG@10 vs doc-doc ceiling).

### Step 3 — Joint Ranking Fine-Tuning

Loss = CE ranking + cosine alignment + squared FLOPs:

```python
# CE: query vs. hard-negative docs (ColBERTv2 64-way)
ranking_loss = F.cross_entropy(q_vecs @ doc_vecs.T, labels)

# Cosine alignment: keep query encoder anchored to doc SPLADE's space
align_loss = (1.0 - F.cosine_similarity(q_vecs, doc_splade(queries))).mean()

# Squared FLOPs: differentiable sparsity (gradient ∝ current activation magnitude)
flops = flops_scale * lambda_q * (q_vecs.mean(dim=0) ** 2).sum()

total = ranking_loss + align_coeff * align_loss + flops
```

**Why CE and not KL self-distillation?**
KL distillation from naver/splade-v3 teacher scores on queries fails because splade-v3
is a doc encoder — on 5–8 word query texts it produces near-zero sparse vectors, making
the softmax nearly uniform. The KL loss has no gradient to follow. CE loss on hard-negative
ranking is the correct signal.

**Why squared FLOPs and not L1?**
L1 applies uniform gradient regardless of activation magnitude — kills small, learning
activations as aggressively as large ones. Squared FLOPs gradient scales with activation
magnitude: large activations are penalized strongly while small ones stabilize.

---

## Experimental Findings

### Evaluation Setup

- **NanoBEIR**: 50 queries per dataset (NanoMSMARCO, NanoNFCorpus). Very noisy — ±0.03
  confidence intervals from query sample size alone. Used for quick iteration only.
- **MSMARCO Dev (full)**: 6,980 queries, 8.8M passages. Reliable signal but slow (~hours
  on a single GPU). Subset mode available: all relevant docs + random distractors → inflated
  by ~2–3 points absolute but preserves relative ordering.
- **Baseline (doc-doc ceiling)**: naver/splade-v3 encodes both queries and docs.
  This is the performance upper bound for this asymmetric setup.

### Key Results

| Configuration | NanoMSMARCO NDCG@10 | Notes |
|---|---|---|
| doc-doc ceiling (splade-v3 both sides) | ~0.64 | upper bound for asymmetric setup |
| vocab_transplant_align (ettin-17m, tokensurgeon + cosine) | ~0.61 | **primary result** |
| direct_align (bert-small, shared vocab, KD + cosine) | worse | BERT arch less efficient here |
| random_init_align (ettin-17m, random embed + KD + cosine) | fails | tokensurgeon init is load-bearing |
| doc_head_align (ettin + proj + frozen doc head, KD + cosine) | fails | frozen head insufficient alone |
| Joint CE + alignment fine-tuning | ~0.55–0.60 | does not improve over alignment-only |

**Main finding**: cosine alignment warmup with tokensurgeon-initialized embeddings achieves
~0.61 NDCG@10 — approaching the ~0.64 doc-doc ceiling. Ranking fine-tuning does not improve
this further.

**Ablation conclusion**: the three ablations (direct_align, random_init_align, doc_head_align)
all fail to match vocab_transplant_align. Tokensurgeon's kNN embedding interpolation is the
single critical ingredient. Correct vocabulary size alone (random_init) is insufficient.
Pre-trained output head alone (doc_head_align) is insufficient. Shared vocabulary via BERT
(direct_align) is insufficient. The embedding initialization quality is the binding constraint.

**Ranking fine-tuning hypothesis**: after cosine alignment the query encoder sits in a local
minimum where CE gradient is weak — outputs are highly correlated with the doc encoder's,
and in-batch hard negatives don't provide strong enough signal to push further.

### Failed Approaches

- **KL self-distillation from splade-v3 on queries**: produces near-zero teacher distributions
  → uniform softmax → no useful gradient.
- **LoRA on frozen doc encoder**: tried to unfreeze doc encoder with LoRA adapters to
  let both sides adapt together. Did not improve and added training complexity.
- **nway=33** (more hard negatives per query): no improvement, likely because alignment
  ceiling is the binding constraint.
- **ettin-68m** (4x larger query encoder): similar results to 17m, suggesting capacity
  is not the bottleneck.
- **Flops warmup delay** (40k steps): letting the model rank freely before adding sparsity
  pressure did not help.
- **random_init_align**: correct vocab size + architecture but random embedding init → fails.
  KD warmup prevents dead-dim collapse but the model never reaches vocab_transplant quality.
- **doc_head_align**: pre-trained frozen doc SPLADE head + linear projection → fails.
  The head provides the right output space but without tokensurgeon the backbone cannot
  learn to drive it properly.
- **direct_align (bert-small)**: shared vocab, no transplant needed, KD warmup + cosine
  → underperforms. BERT architecture appears less suited than ettin here; also confirms
  that just sharing vocabulary without transplant is not the same as tokensurgeon init.

---

## Open Questions / Next Experiments

1. **Is 0.61 actually near the ceiling for this setup?** The gap between query-doc and
   doc-doc is only ~3–4 points. Maybe the alignment warmup already found the optimum
   and ranking fine-tuning adds nothing because there's nothing left to find. Full MSMARCO
   dev eval needed to confirm.

2. **Scale up the doc encoder.** All experiments so far use naver/splade-v3 (DistilBERT,
   ~66M) as the frozen doc encoder. The asymmetric setup should benefit from a stronger doc
   encoder. Candidate: `hzeng/Lion-SP-8B-llama3-marco-mntp` (8B Llama-3, MS MARCO trained,
   SIGIR 2025). Caveat: LLM-based SPLADE models use large tokenizer vocabularies (32K+)
   which changes the transplant and dot-product space significantly.
   Note: no SPLADE models between ~110M and ~600M appear to exist publicly.

3. **Is tokensurgeon interpolation quality the binding constraint?** The kNN interpolation
   for non-shared tokens is an approximation. Random-init ablation confirms init matters;
   the question is whether *better* interpolation (larger k, different weighting) would
   push further. Measure embedding distance for non-shared tokens post-transplant vs.
   the random baseline to quantify the gap.

4. **Why does ranking fine-tuning fail?** The CE loss does not improve over alignment-only
   across all architectures tried. This needs a gradient analysis — is the CE gradient
   genuinely zero after alignment, or is it being overwhelmed by the FLOPs regularizer?

---

## Publishability Assessment

### What's potentially novel

**The core combination** — vocabulary transplant + asymmetric sparse retrieval — does not
appear to have been published directly. Specifically:

1. Using tokensurgeon/embedding interpolation to make a small query MLM share a large
   doc SPLADE's vocabulary space, enabling zero-projection asymmetric sparse retrieval.
2. The finding that a small vocab-transplanted model trained only with cosine alignment
   (no ranking signal) achieves competitive NDCG with the full doc encoder used
   symmetrically. This is surprising and worth reporting.
3. The KL self-distillation failure mode for query-side SPLADE training: doc encoders
   produce degenerate distributions on short queries — this is underappreciated.

### Related work to check

These papers likely cover adjacent territory — read them before claiming novelty:

- **SPLADE-v2** (Formal et al., 2022): already distinguishes query/doc representations
  to some extent using separate FLOP regularization coefficients.
- **SPLADE-v3** / **SPLADE-3** (Lassance & Piwowarski, 2024): uses `naver/splade-v3`
  as the doc encoder. Check if they train a lighter query encoder.
- **SPLADEv2 with asymmetric distillation**: some SPLADE variants use a smaller query
  encoder from the start (same vocabulary, different capacity).
- **Tokensurgeon** (Baumann et al., 2024): the embedding transfer method used here.
  It was designed for cross-lingual transfer, not asymmetric retrieval.
- **TILDE / TILDEV2** (Zhuang & Zuccon, 2021): precompute token scores offline;
  query processing is cheap. Different approach to same serving-cost problem.
- **CoCondenser / Efficient SPLADE**: various compression efforts — check if any use
  vocabulary transfer.
- **uniCOIL / DeepImpact / SLIM**: other learned sparse retrieval models; most use
  the same vocab for Q and D by design.
- **LexMAE** (Shen et al., 2023): learned sparse lexical matching; may have asymmetric mode.

### Strongest claim (if results hold)

A vocab-transplanted model trained only with cosine alignment to the doc SPLADE —
without any ranking supervision — achieves ~95% of the symmetric doc-doc performance
on MSMARCO NDCG@10, at 4–10x lower serving cost.

The ablation suite strengthens this: three independent failure modes (random embedding
init, shared vocabulary without transplant, pre-trained head without transplant) all
confirm that tokensurgeon's kNN embedding interpolation is the mechanism driving the
result — not just vocabulary size alignment or architecture choices. This makes the
finding more crisply attributable and harder to dismiss as a hyperparameter accident.

### Weakest points

- Results are currently only on NanoBEIR (50 queries). Full MSMARCO dev needed before
  any claim is credible.
- The ranking fine-tuning appears not to improve over alignment alone, which is either
  a genuine finding or a training failure. Until this is resolved it's unclear whether
  the method "works" or is stuck.
- Ettin-17m / 68m are not widely-known baselines. The comparison would be stronger
  with DistilBERT or MiniLM as the query backbone.

---

## Implementation Reference

| Component | File | Entry point |
|---|---|---|
| Vocab transplant (tokensurgeon) | `train.py` | `_run_tokensurgeon()` |
| VocabTransplantQuerySPLADE model | `src/model.py` | `VocabTransplantQuerySPLADE` |
| Joint loss (CE + align + FLOPs) | `src/model.py` | `vocab_transplant_joint_loss()` |
| Training loop | `train.py` | `train_vocab_transplant()` |
| NanoBEIR eval | `src/eval.py` | `evaluate_asymmetric()` |
| Full MSMARCO dev eval | `scripts/eval_msmarco.py` | `--checkpoint`, `--doc_only` |

Run vocab-transplant training:
```bash
uv run train.py vocab_transplant --config config.yaml
```

Run full MSMARCO eval on an alignment checkpoint:
```bash
uv run scripts/eval_msmarco.py \
    --checkpoint checkpoints_ettin-encoder-68m/vocab_transplant/align_step_10000.pt \
    --max_corpus_size 200000
```

Run doc-only upper bound:
```bash
uv run scripts/eval_msmarco.py --doc_only
```
