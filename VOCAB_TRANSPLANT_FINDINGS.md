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

### Approach 8 — Shallow SPLADE Query Alignment
Instead of crossing vocabularies or architectures, build the query encoder by loading
`naver/splade-v3` and truncating the body to the first `n_layers` while keeping the
same tokenizer, embedding table, and MLM head. The frozen document encoder remains
the full `naver/splade-v3`; only the shallow query side is trained.

This is currently the most reliable SPLADE-v3-side setup. The working recipe uses:
- a 5k-step warmup where embeddings/head stay frozen and only the retained body
  layers train;
- phase-1 unfreezing with embeddings/head at a lower LR (`head_lr_scale=0.1`);
- teacher-vector MSE against the full SPLADE query vector with an STE through ReLU;
- query L1 sparsity;
- a contrastive CE term over Tevatron positives plus hard negatives. The student
  query scores frozen full-SPLADE passage vectors, so the document encoder stays fixed
  while the query side gets direct retrieval-margin supervision.

Result: **works well and trains stably**. The contrastive term is important because
pure teacher-vector matching only asks the shallow model to imitate query vectors,
whereas the contrastive term also teaches which frozen passage vectors should be
separated for each query. This is the best path so far for a practical lightweight
SPLADE-v3 query encoder.

### Approach 9 — Shallow SPLADE + Cross-Encoder MarginMSE Distillation
We tried a more traditional SPLADE-v3-style distillation objective in
`splade_shallow_align_distill`: keep the same shallow query architecture and frozen
full SPLADE document encoder, but replace the contrastive/vector-MSE recipe with
cross-encoder margin distillation:

```
CE(q, positive) - CE(q, negative)
  ≈ dot(q_shallow, d_positive_full_splade)
    - dot(q_shallow, d_negative_full_splade)
```

The first implementation used `cross-encoder/ms-marco-MiniLM-L6-v2` as a simple
single teacher. SPLADE-v3 uses stronger teacher distillation, including
SparseDistillKLDivLoss and SparseMarginMSELoss with an ensemble of rerankers, but
we started with MarginMSE only for simplicity. We added the same shallow-query
training mechanics that were necessary for the stable contrastive run: 5k frozen
warmup, lower LR for embeddings/head after unfreeze, frozen full-SPLADE passage
vectors, and query sparsity.

Result: **decent retrieval but unstable sparse representations**. A representative
10k-step run reached NanoMSMARCO NDCG@10 around 0.65 and NanoNFCorpus around 0.31,
which is competitive with the stable contrastive run. However, `q_nnz` varied
wildly across logged batches. Depending on scaling and sparsity settings it could
collapse to ~2-5 active dimensions, or swing into tens/hundreds of active
dimensions, even when evaluation looked good.

Likely reason: MarginMSE is under-constrained for this asymmetric sparse setup.
Cross-encoder logits have arbitrary scale, while frozen-SPLADE dot products are
unbounded and can be matched by many different query-vector shapes. The model can
satisfy a batch's positive-vs-negative margins by changing query mass and activating
idiosyncratic dimensions, without preserving a stable SPLADE-like sparse support.
Small per-step batches make this look even noisier in logs, but the nnz/flops
swings are real. Adding a full-SPLADE query-vector anchor, clipping/rescaling CE
margins, and tuning `lambda_q` improved the retrieval/scale tradeoff but did not
remove the support instability.

Conclusion: write this off for now as an unstable variant unless we revisit it with
a stronger/listwise teacher, precomputed teacher scores, larger effective batches,
or an explicit support/mass regularizer. The stable `splade_shallow_align`
contrastive recipe remains the preferred setup.

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
| splade_shallow_align (3-layer query + contrastive CE) | ~0.65 | stable lightweight SPLADE query path |
| splade_shallow_align_distill (cross-encoder MarginMSE) | ~0.65 | decent eval, but unstable q_nnz/flops |
| splade_shallow_factorized_spaced_align | ~0.69 | strongest factorized run; full MS MARCO NDCG@10 0.4438 vs 0.4657 doc-only ceiling |

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
- **splade_shallow_align_distill**: cross-encoder MarginMSE on frozen full-SPLADE passage
  vectors → decent NanoBEIR retrieval, but unstable sparse support. The loss can match
  CE margins with arbitrary query-vector mass/support, so q_nnz and FLOPs vary too much
  for a reliable sparse retriever.

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

   **Update**: this is now done with `lion_transplant_align`
   (`Lion-SP-1B-llama3-marco-mntp`, 128K Llama-3 BPE vocabulary). After resolving
   several distinct training-dynamics issues (see Lion-SP-1B Variant section
   below), it reaches NDCG@10=0.6410 on NanoMSMARCO vs 0.6510 doc-doc ceiling.
   8B doc encoder is still TODO.

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

### Current stable path: shallow alignment

The most stable and useful configs now are `splade_shallow_align` and
`lion_shallow_align`. They avoid the brittle cross-vocabulary transplant problem
by constructing the query encoder from the same checkpoint as the document
encoder, then truncating the body to the first `n_layers`. The query side keeps
the document tokenizer, embedding table, and output head, so query and document
vectors live in the same sparse vocabulary space with no projection layer or
tokensurgeon approximation.

Both configs use a two-phase schedule: first freeze embeddings/head and train
only the retained body layers, then fine-tune the lightweight query side. This
has been stable during training and gives good results, unlike several earlier
transplant and direct-alignment ablations that collapsed or plateaued.

For `splade_shallow_align`, the active loss is teacher-vector MSE on SPLADE query
vectors with an STE through ReLU, query L1 sparsity, and an optional contrastive
CE term over Tevatron positives plus hard negatives. The contrastive term scores
student queries against frozen full-SPLADE passage vectors, adding retrieval
margin supervision while keeping the document encoder fixed.

This contrastive setup has now been tested and is the stable SPLADE-v3-side
lightweight-query recipe. The separate `splade_shallow_align_distill` experiment,
which used cross-encoder MarginMSE instead of contrastive CE, produced competitive
NanoBEIR scores but unstable `q_nnz`/FLOPs. The likely issue is not retrieval signal
quality alone; it is that raw cross-encoder margins do not define a unique sparse
query vector. They constrain score differences, not which SPLADE vocabulary
dimensions should be active or how much query mass should be used.

The next size-reduction experiment is `splade_shallow_factorized_align`: keep the
proven shallow alignment recipe, but replace the tied `[vocab, hidden]` embedding
/ MLM-head matrix with ALBERT-style shared factors. With SPLADE-v3 dimensions this
turns the lexical parameter floor from roughly `30,522 * 768` parameters into
`30,522 * factorized_embedding_dim + factorized_embedding_dim * 768` while
preserving the same output vocabulary. The default factorized dimension is 128 and
the factors are initialized by SVD from the original tied matrix.

Factorized runs exposed a separate issue from ordinary shallow alignment. With
contrastive training enabled and the shared factors unfrozen after warmup, `q_nnz`
kept climbing even when activation mass stayed modest. That means the model was
creating many tiny positive dimensions: the L1 sparsity term penalizes total
mass, while `q_nnz` counts any dimension barely above zero.

The strongest factorized variant so far keeps the factorized embedding / SPLADE
head frozen for the full run and trains only the retained transformer body. This
worked surprisingly well, suggesting that the SVD-initialized lexical factors are
already a good enough approximation of the SPLADE-v3 lexical basis, and that most
of the useful adaptation should happen in the shallow body rather than by moving
the shared lexical factors.

A follow-up isolation run disabled contrastive training and unfroze the
factorized embedding/head after 5k steps with a much lower LR
(`head_lr_scale: 0.01`, roughly `2e-6` when the body LR is `2e-4`). That setup
does work and gives good results, but it is still weaker than keeping the
factorized SPLADE head frozen throughout. So contrastive loss was not the whole
problem; moving the factorized lexical basis itself appears to be risky even at a
small learning rate.

Because contrastive training encodes `batch_size * nway` frozen SPLADE passages
on every microstep, factorized runs currently use `use_contrastive: false` and
`log_ranking_metrics: false` for cheap iteration. That recovers the older
teacher-query MSE setup; periodic NanoBEIR eval still measures retrieval quality.

New follow-up: `splade_shallow_factorized_spaced_align` keeps the same frozen
factorized lexical factors/head and non-contrastive loss, but builds the 3-layer
query body from doc layers `[0, 6, 11]` instead of the first three contiguous
layers. This tests whether a shallow query benefits from retaining one early,
one middle, and the final SPLADE-adapted transformer block.

This is now the strongest factorized SPLADE-v3-side result. At 50k steps it
reached about `0.6928` NanoMSMARCO NDCG@10 and `0.3333` NanoNFCorpus NDCG@10.
On full MS MARCO dev with the 8.8M-passage index, the final checkpoint reached:

| Configuration | MSMARCO Dev NDCG@10 | MSMARCO Dev MRR@10 | Notes |
|---|---:|---:|---|
| `naver/splade-v3` doc-only ceiling | 0.4657 | 0.3989 | Same SPLADE-v3 model encodes queries and docs. |
| `splade_shallow_factorized_spaced_align` | 0.4438 | 0.3787 | 3-layer `[0, 6, 11]` factorized query, frozen lexical factors/head. |

The factorized spaced query keeps about `95.3%` of the SPLADE-v3 doc-only
NDCG@10 and about `94.9%` of its MRR@10 while using a much smaller query-side
model. The remaining gap is roughly `-0.0219` NDCG@10 and `-0.0202` MRR@10.

### Weakest points

- The strongest factorized shallow result now has a full MS MARCO dev number,
  but older vocab-transplant and Lion results are still mostly NanoBEIR-only.
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
    --stage splade_shallow_factorized_spaced_align \
    --checkpoint checkpoints_splade-v3/splade_shallow_factorized_spaced_align/align_final.pt \
    --config config.yaml \
    --index_dir data/msmarco_index
```

Run doc-only upper bound:
```bash
uv run scripts/eval_msmarco.py \
    --stage splade_shallow_align \
    --doc_only \
    --config config.yaml \
    --index_dir data/msmarco_index
```

---

## Lion-SP-1B Variant (`lion_transplant_align`)

### Setup

Same idea as `vocab_transplant_align` but the frozen doc encoder is
`hzeng/Lion-SP-1B-llama3-marco-mntp` (1B Llama-3, bidirectional, MS MARCO trained,
SIGIR 2025) instead of `naver/splade-v3`. The query encoder is
`jhu-clsp/ettin-encoder-150m` transplanted onto Lion's 128K Llama-3 BPE vocabulary.

This stresses the transplant pipeline considerably:
- Vocabulary overlap between ettin (~50K wordpiece) and Llama-3 (128K BPE) is **single-digit
  percent** — ~95% of donor tokens are kNN-interpolated, vs. ~40% for the splade-v3 case.
- Lion's SPLADE outputs are very sparse for short queries (~190 nnz vs. splade-v3's
  several hundred), so the alignment target lives in a much smaller subspace of the 128K
  output dimension.

### Dead Ends (and Why)

Several "obvious" fixes failed in instructive ways. Recording them so we don't retry.

**Pure cosine alignment on `q_vecs`** (initial implementation): collapses to `q_nnz=0,
loss=1.0` within ~3000 steps. Cosine is scale-invariant, so the model can drift to
arbitrarily small magnitudes; once `q_vecs ≡ 0` the relu has zero gradient and the
state is absorbing. Cosine returns 0 for a zero vector by PyTorch convention, so
`align_loss` saturates at 1.0 and gradient flow stops entirely.

**L1 FLOPs (`q_vecs.abs().sum(-1).mean()`)**: same collapse mode, accelerated. L1's
gradient is uniform across active elements regardless of magnitude — pushes everything
toward zero at the same rate, blowing past the desired sparse equilibrium. The
codebase already documents this in `vocab_transplant_splade_loss` ("L1 was too
aggressive"); the Lion variant was inconsistent and used L1 anyway.

**Squared-mean FLOPs (`(q_vecs.mean(0) ** 2).sum()`)**: necessary but insufficient.
Gradient ∝ activation magnitude is the right primitive (kills loud activations
strongly, leaves small ones to stabilise) but doesn't fix the underlying cosine
instability — collapse still happens, just slower.

**Sum-MSE on `q_vecs` as a magnitude anchor**
(`(q_vecs - d_vecs).pow(2).sum(-1).mean()`): made things *worse*. Initial MSE was
~19,000 (huge) so it dominated the loss, overshot, and any loss computed on `q_vecs`
inherits the dead-relu trap once `q_vecs ≡ 0`. Magnitude anchors only help if they
operate on a quantity that has gradient through the dead-relu state.

**Softmax-KL on raw logits + cosine on `q_vecs`** (the "obvious" KD-style fix):
seemed to work at first — `q_nnz` stayed alive in 50–500 range for a 5000-step smoke,
no immediate collapse. But the failure mode showed up at the KD → cos phase boundary
in a real 50k-step run:

- KD phase reached `loss=0.007, q_nnz≈90k` — i.e. KL drove the *shape* of `softmax(q_logits/T)`
  to match Lion's, but the *offset* drifted upward freely (softmax is shift-invariant).
  All logits ended up positive → q_vecs dense, not sparse.
- At the phase transition, cos saw a huge mismatch (dense `q_vecs` vs sparse `d_vecs`)
  and overcorrected, sparsifying so aggressively that all logits went negative.
- Once `q_vecs ≡ 0`: cos has zero gradient through dead relu, KL is *also* satisfied
  (shift-invariance → can claim alignment), FLOPs is zero. Total loss stuck at ~1.0
  with zero parameter gradient. Dead.

The KL "safety net" gave up the offset axis entirely. That's the trap: any loss on a
softmax-normalized quantity is shift-invariant on its inputs.

**Bumping `lambda_q` to fix density**: backfires. `lambda_q=0.005` collapses `q_nnz`
to ~7 within 1000 steps; `lambda_q=0.01` collapses faster. Tightening sparsity
regularisation accelerates the transition into the dead-relu basin.

### Working Formulation

Two-phase, both phases use the same single forward pass through ettin and Lion:

**Phase 1 — KD warmup** (`kd_warmup_steps`, default 0): softmax-KL on
max-pooled raw logits at temperature T=4. Warms the model into Lion's logit-shape
basin without yet committing to the absolute logit values. Squared-mean FLOPs is
**now also active during this phase** (previously off) so the warmup ends with
sparse `q_vecs` instead of dense ~90k.

**Phase 2 — Cosine alignment + softmax-KL safety net**:

```python
cos_loss = (1.0 - F.cosine_similarity(q_vecs, torch.log1p(torch.relu(d_logits)))).mean()
kl_loss  = T**2 * F.kl_div(log_softmax(q_logits/T), softmax(d_logits/T))
align_loss = cos_loss + kl_coeff * kl_loss
```

Cosine drives the SPLADE-vector direction. The KL term operates on raw signed
logits, so its gradient survives even when `q_vecs ≡ 0` (the relu kills the
cosine gradient but not KL). FLOPs (always on, both phases) anchors the absolute
scale to keep softmax-KL's shift-invariance from drifting the offset.

**Why not BCE-with-logits** (an earlier attempt): BCE-with-logits against
`sigmoid(d_logits)` looks attractive — it's not shift-invariant, operates on
raw logits, and encodes Lion's sparsity directly. But it plateaued in practice:

- *Without* `pos_weight`: ~190 active vs ~128,000 inactive targets per query
  (670:1 imbalance) → inactive class dominates gradient → model learns "make
  everything negative" → `q_nnz` plateaus at ~5–10, NDCG caps around 0.42.
- *With* dynamic `pos_weight = n_neg / n_pos ≈ 425`: the formula is designed
  for binary targets, but soft sigmoid targets in the [0.05, 0.5] range
  (tokens with d_logit slightly negative) get scaled by `pw·t` ≈ 50–200,
  flipping their equilibrium to σ(q) ≈ 1. Result: `q_nnz` blew up to 128,256
  (every dim active), NDCG = 0.

So while BCE-with-logits has nice theoretical properties at the boundary
(relu, shift), the soft-target loss landscape is treacherous at this vocab
scale and class imbalance. Cosine + KL turned out to be more stable in
practice, which matched the user's prior runs.

Squared-mean FLOPs stays available but as a fine-tuning knob, not the primary
sparsity mechanism:

```yaml
lambda_q: 0.001    # try 0.0 first; bump only if q_nnz drifts above target
```

### Results (50k-step run, ettin-encoder-150m → Lion-SP-1B)

Final checkpoint, NDCG@10 vs the Lion-Lion symmetric ceiling:

| dataset | step 40k | step 50k | Lion-Lion ceiling | gap |
|---|---|---|---|---|
| NanoMSMARCO  | 0.6129 | **0.6410** | 0.6510 | -0.010 |
| NanoNFCorpus | 0.3105 | **0.3211** | 0.3480 | -0.027 |

The asymmetric setup (ettin-150m on queries, Lion-1B on docs) lands within
~0.01 of the symmetric Lion-Lion ceiling on NanoMSMARCO. Full MSMARCO dev
eval needed to confirm (NanoBEIR is 50 queries → ±0.05 noise floor) but the
gap to ceiling is small enough that this looks like a real result, not a
measurement artefact.

Per-batch sparsity from training logs:

|        | q (ettin-150m) | d (Lion-1B on the same query texts) |
|---|---|---|
| nnz    | 150–220 | 250–350 |
| L1 mass | 35–45  | 55–65   |

Query nnz is meaningfully lower than doc nnz — the standard SPLADE asymmetry
emerged on its own from the cosine alignment dynamics, no separate `lambda_q`
vs `lambda_d` tuning required.

### Operational Gotcha — Eval Memory Leak

`evaluate_asymmetric` encodes thousands of corpus docs through Lion-SP-1B in fp16.
PyTorch's caching allocator holds those activations even after the eval tensors are
freed, so the next training step OOMs trying to allocate its own activations on top.
Fix: explicit `gc.collect(); torch.cuda.empty_cache()` *after* both the initial eval
and every periodic eval (`train.py:1508` and `train.py:1625`).

If OOM still happens at periodic evals: drop `eval_batch_size` (currently 4) — Lion
creates `[B, L, 128K]` fp16 activations and the peak during eval dictates whether
resumed training fits.

### Implementation Reference

| Component | File | Entry point |
|---|---|---|
| Lion vocab transplant (tokensurgeon over PEFT) | `train.py` | `_run_tokensurgeon_lion()` |
| FrozenLionSPLADE doc encoder | `src/model.py` | `FrozenLionSPLADE` |
| Training loop (KD + BCE + FLOPs) | `train.py` | `train_lion_transplant_align()` |

Run:
```bash
uv run train.py lion_transplant_align --config config.yaml
```

Healthy log signature (post-warmup phase tagged `cos`):
```
[lion-align/cos] step  N | loss <decreasing> | q_nnz 50–300 (lion 250–350) | flops … | lr …
```

`q_nnz` should track `lion d_nnz` within an order of magnitude. If you
see one of these, something's wrong:

```
[lion-align/cos] step  N | loss 1.0xxx | q_nnz 0.0 (lion ~300) | …
```
Dead relu basin. Cosine drove magnitudes to zero, KL safety net wasn't strong
enough. Bump `kl_coeff` or shorten `flops_warmup_steps`.

```
[lion-align/cos] step  N | loss …      | q_nnz 128256 (lion ~300) | …
```
Density runaway — every dim active, retrieval is meaningless. Check whether
something is upweighting the positive class (e.g., a stray BCE pos_weight)
or whether `lambda_q` is too low to provide any sparsity pressure.

### Backbone Transferability — bert-base-uncased (negative result)

The 50k ettin-150m run (NDCG@10=0.6410, ~ceiling) made the recipe look
backbone-agnostic. It isn't. Replacing the query encoder with
`google-bert/bert-base-uncased` produced a chain of failures, none of which
were the *same* failure as the ettin debugging earlier in this doc.

#### What we tried, in order

1. **Vanilla cos+KL alignment** (the configuration that worked for ettin).
   Collapsed to `q_nnz=0, loss=1.01` by step 3000.
2. **MSE-on-raw-logits anchor**. Made it worse (collapsed faster). MSE pulls
   `q_logits` toward `d_logits`, which is mostly *very negative* for Lion's
   sparse SPLADE — exactly the wrong direction.
3. **Softplus instead of relu in the SPLADE encoding**. Hard relu's "dead
   gradient" trap was a suspect. Softplus didn't help — the issue isn't the
   relu, it's earlier.
4. **MLM-head bias calibration**. We measured it: bert-base post-transplant
   produces `q_logits` with mean=-2.7, frac>0=0.87% (vs ettin's mean=+0.4,
   frac>0=58.8%). Calibration shifts the bias by `-mean(q_logits)` so the
   initial distribution centres at 0. Helped at *step 0* but the model drifts
   back into the dead basin within a few hundred steps of alignment.
5. **MLM warmup phase** (re-train body+head on standard masked-LM in Llama-3
   vocab before alignment). 10k steps, lr=1e-4, mask token = a Llama-3
   reserved special token, BERT-style 80/10/10 masking, `cls.predictions.bias`
   zeroed (it was bert-mismapped). MLM loss dropped from `log(128k)≈11.76` to
   ~8 in the first 50 steps then **plateaued there**. Loss=8 ≈ unigram-only
   prediction (knowing what tokens are common overall, no context).
6. Investigating: a single-example overfit drops loss 9.3 → 0.7 in 30 steps
   at lr=1e-4. So forward, backward, optimizer all work. The problem is
   *specifically* the multi-batch case — the model can memorise but can't
   generalise.

#### What's actually wrong

The transplant's quality is the binding constraint. Specifically:

- Llama-3 tokenizes English with `Ġ`-prefixed BPE pieces (`Ġcat`, `Ġsat`, …).
  BERT's WordPiece doesn't use `Ġ` anywhere in its vocab.
- `_run_tokensurgeon_lion` does **exact string matching** to find shared tokens.
  For Llama-3 ↔ bert-base WordPiece, near-zero content tokens match — almost
  every token in real English text gets a kNN-interpolated embedding rather
  than a directly-copied one.
- bert-base's body was trained on real BERT-WordPiece embeddings. With ~95%
  of input embeddings interpolated, the body's hidden states become OOD. It
  can produce broadly-correct unigram-level predictions through the head
  (after the bias is calibrated), but it can't extract context-specific
  information from these distorted hidden states.
- ettin/ModernBERT survived this because its body is more robust (newer
  architecture, longer pretraining, better data). bert-base doesn't.

This is not a hyperparameter issue. The binding constraint is the embedding
table, and tokensurgeon's exact-string + cosine-NN strategy is too lossy for
backbones whose tokenizer disagrees with the donor's at the substring level.

#### What might fix it (not yet tried)

1. **BPE-decomposition transplant**: instead of cosine-NN in donor space for
   non-shared tokens, decompose each Llama-3 BPE token into BERT WordPiece
   subwords and average those embeddings. Should give bert-base much better
   coverage. The WECHSEL paper (Minixhofer et al.) does this kind of
   subword-aware cross-lingual transfer.
2. **Real pretraining-scale MLM warmup**: 100k+ steps. Train the body to
   interpret interpolated embeddings.
3. **roberta-base instead of bert-base**: RoBERTa uses GPT-2-style BPE with
   the same `Ġ` prefix convention as Llama-3. String overlap should be
   dramatically higher. This is the cleanest test of the
   "BPE-overlap vs. body-robustness" hypothesis. ← currently being tested.

#### Diagnostic helpers we added (kept in the codebase)

These all live in `train_lion_transplant_align` and are skipped on resume:

- **MLM head bias calibration** (always on, `calibrate_bias: true` config knob,
  `train.py:1498`). Measures `mean(q_logits)` over a batch of training queries
  and shifts the head bias by `-mean`. Cheap; no downside.
- **MLM warmup phase** (off by default, `mlm_warmup_steps: 0`,
  `train.py:1540`). Standard masked-LM in the donor vocab. Recommend 0 for
  ettin-style backbones, ≥10k for bert-style ones (though even 10k didn't
  rescue bert-base — see above).
- **`d_nnz` logging** alongside `q_nnz` so you can compare the model's
  output sparsity to Lion's target live, `train.py:1700+`.
