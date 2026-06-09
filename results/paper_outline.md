# Paper Outline: Asymmetric Depth Pruning for Efficient Sparse Query Encoding

---

## Pending Before Camera-Ready

> These are the only experiments left to run (others are complete or intentionally deferred).

| # | Experiment | Status | Notes |
|---|---|---|---|
| 1 | Full SPLADE-v3 BEIR baseline | **TODO** | `eval_beir.py --doc_only --stage splade_shallow` |
| 2 | Full Lion BEIR baseline | **TODO** | `eval_beir.py --doc_only --stage lion_shallow` |
| 3 | Static cosine at 10k steps | **TODO** | retrain (crashed at step 3,600) + BEIR eval |
| 4 | BM25 BEIR baseline | **DEFERRED** | no pyserini/rank_bm25 installed |

Intentionally deferred: full BEIR, MS MARCO dev MRR@10, throughput/latency measurements.

Already confirmed complete: full SPLADE-v3 and Lion shallow layer sweeps (1–5 layers × first/spaced/last), head-unfreezing ablation, factorization grid, loss ablations (3-layer and static), static model training.

---

## 1. Abstract

One paragraph. Problem statement, approach, key quantitative claims. Suggested structure:

> SPLADE-style sparse retrievers produce high-quality lexical vectors but run a full transformer on every user query. We show that the query encoder can be asymmetrically pruned to 3 layers — or reduced to a completely static per-vocabulary-token weight table — while keeping the document encoder and index unchanged. On BEIR-small, a 3-layer SPLADE-v3 query encoder trained by direct vector MSE distillation retains 98% of the full-model BEIR quality with 4× fewer query-time layers. A zero-layer static encoder (30k parameters, no transformer forward pass) retains ~95% of 3-layer quality, providing an extreme efficiency anchor. Key ablations show frozen lexical heads, direct vector distillation, and spacing of retained layers are the critical design choices; unfreezing, factorization, and ranking-score supervision all degrade quality.

---

## 2. Introduction

### 2.1 The Query Encoding Bottleneck in Sparse Retrieval
SPLADE-style models produce high-quality sparse lexical vectors but run a full transformer at query time on every user query. Documents are indexed once; queries pay the encoder cost on every request.

### 2.2 Asymmetric Pruning: Shallow Query, Full Document
Key insight: if the query encoder remains in the same lexical sparse vector space as the document encoder, no re-indexing is required. We can prune only the query-time path.

### 2.3 Contributions
1. A layer-selection and distillation recipe for building shallow sparse query encoders from pretrained SPLADE-style models, with a frozen lexical head and direct vector MSE distillation.
2. A zero-layer static sparse query encoder (one learnable weight per vocabulary token) as an extreme-efficiency anchor that needs no per-query transformer computation and retains ~95% of 3-layer BEIR quality.
3. Systematic ablations over layer count (0–5 layers), layer position (first/spaced/last), head freezing, factorized lexical matrices, and distillation losses across two model families (SPLADE-v3 and Lion-SP-1B).
4. Evidence that the optimal loss function is architecture-dependent: for 3-layer encoders, direct vector MSE dominates; for the static (0-layer) encoder, cosine distance is slightly better — but in both cases, ranking-score supervision (KL/MarginMSE) consistently underperforms.

---

## 3. Background and Related Work

### 3.1 Sparse Lexical Retrieval
SPLADE, uniCOIL, SPARTA, BM25. Focus on SPLADE-v3 and Lion-SP as the two bases used.

### 3.2 Knowledge Distillation for Retrieval
TAS-B, ColBERT distillation, MarginMSE, KL-divergence over scores. Contrast with our approach of direct vector-level distillation.

### 3.3 Model Compression for Retrieval
DistilSPLADE, quantization (SPLADE-v3 int8), width pruning (attention head pruning). Distinguish from depth/layer pruning.

### 3.4 Layer Pruning in Transformer Models
DistilBERT (task-agnostic), Shortened LLaMA, LaCo. Relevance: we do task-specific layer selection with distillation recovery, applied to a sparse retrieval objective.

### 3.5 Asymmetric Retrieval Architectures
ColBERT (asymmetric dense), DPR, prior asymmetric sparse work if any. Note our setup is strictly asymmetric in depth, not modality.

### 3.6 Static / Bag-of-Words Query Representations
Learned sparse retrieval without a query encoder (e.g., learned unigram weights). Situate the static model as the extreme compression limit in our framework: the query encoder degenerates to a static vocabulary lookup.

---

## 4. Method

### 4.1 Asymmetric Retrieval Setup
Full document encoder (frozen) builds the index. Shallow or static query encoder runs at serving time. Both live in the same 128k-dimensional sparse lexical space; no index modification needed.

### 4.2 Shallow Query Encoder Construction
Layer selection strategies: first-N, last-N, equally spaced. The selected layers inherit all weights from the pretrained full model.

### 4.3 Zero-Layer Static Query Encoder
Degenerate limit: no transformer layers. The query encoder is a 30,522-dimensional learnable weight vector. At inference, tokenize the query, binary presence mask, then element-wise multiply by learned weights. No matrix multiplications, no attention, no position encodings. Despite extreme parameter reduction, the model lives in the same vocabulary space and can be trained by the same distillation objective.

### 4.4 Distillation Objective
Direct vector MSE (default) or cosine distance to the full model's query SPLADE vector. Teacher is the frozen document encoder applied to queries. Sparsity regularization (FLOPS loss) on the student output. Ranking-score supervision (KL/MarginMSE from ColBERT) also evaluated.

### 4.5 Lexical Head Treatment
The tied embedding/LM-head matrix maps hidden states to vocabulary logits. We freeze it after a short warmup. Rationale: preserving the pretrained vocabulary geometry maintains compatibility with document index vectors.

### 4.6 Training Details
Warmup schedule, learning rates, batch size, gradient accumulation. Hardware and wall-clock context. Note kd/margin_mse variants require additional passage encoding per step (~15 min/1k steps on 8GB GPU).

---

## 5. Experimental Setup

### 5.1 Base Models
SPLADE-v3 (12-layer BERT-base encoder), Lion-SP-1B (16-layer Llama-3.2-1B decoder). Different architectures, sizes, and sparsity distributions.

### 5.2 Training Data
MS MARCO passage pairs. Teacher query vectors generated with the full model. ColBERT distillation scores from `colbertv2_msmarco_64way.json` for ranking-supervision variants.

### 5.3 Evaluation Benchmarks
- In-domain: MS MARCO dev (MRR@10) — via NanoMSMARCO during training
- Cross-domain: BEIR subset — nfcorpus, scifact, arguana, scidocs, fiqa (NDCG@10, MRR@10)
- Full BEIR: deferred to final experiments

### 5.4 Baselines
- Full SPLADE-v3 query encoder against full SPLADE-v3 document index (upper bound)
- Full Lion-SP-1B (upper bound for Lion family)
- BM25 (sparse lexical reference)
- These upper-bound numbers need to be measured explicitly for the camera-ready table.

### 5.5 Checkpoint Selection
Best NanoMSMARCO NDCG@10 during training used to select checkpoint for BEIR evaluation.

---

## 6. Main Results

### 6.1 Quality–Efficiency Tradeoff: Full Table
Main comparison table spanning the full range from zero-layer to full model:

| Model | Layers | Params (query) | BEIR-small avg NDCG@10 | Notes |
|---|---:|---|---:|---|
| BM25 | — | — | TBD | sparse baseline |
| Full SPLADE-v3 | 12 | ~110M | TBD | upper bound |
| SPLADE-v3 spaced3 | 3 | ~33M | 0.3509 | best current config |
| SPLADE-v3 first3 | 3 | ~33M | 0.3483 | |
| SPLADE-v3 static | 0 | 30,522 | 0.3378 (cosine) / 0.3326 (mse) | no transformer |
| Full Lion-SP-1B | 16 | ~1B | TBD | upper bound |
| Lion spaced4 | 4 | ~250M | 0.3163 | best Lion config |

### 6.2 Quality Retained vs. Encoder Cost
Once efficiency measurements are available: quality retained (%) vs. query throughput (queries/sec) or transformer GFLOPs. One figure showing the Pareto curve for each model family. The static model anchors the extreme-right (cheapest) end.

---

## 7. Ablations

### 7.1 Layer Count (Including Zero-Layer Limit)
NDCG@10 vs. number of retained layers for SPLADE-v3 (0, 2, 3, 4 layers) and Lion (3, 4, 5 layers). The zero-layer static result (0.3326–0.3378) extends this curve to its extreme. Shows:
- Static (0-layer): ~0.333 NDCG@10
- 3-layer SPLADE spaced: 0.3509 NDCG@10
- Full SPLADE: TBD (significantly higher)

Diminishing-return behavior and where the quality plateau is reached. Notably, the jump from 0 to 3 layers is substantially larger than any gain within 3–5 layers, motivating 3 layers as the practical minimum for SPLADE.

### 7.2 Layer Position
First-N vs. equally spaced vs. last-N at fixed layer count. Spaced dominates; last-N is consistently weakest. Full table across all first/spaced/last × layer-count combinations for both model families. (See `results/splade_shallow_layer_sweep.md`, `results/lion_shallow_layer_sweep.md`.)

### 7.3 Lexical Head Freezing
Frozen vs. unfrozen after warmup, for SPLADE spaced3 and Lion first3. Frozen head is consistently better. Unfreezing drops SPLADE spaced3 by 0.0078 NDCG and Lion first3 by 0.0151. Explanation: unfreezing shifts vocabulary geometry, breaking alignment with the fixed document index vectors. (See `results/head_unfrozen_ablation.md`.)

### 7.4 Distillation Loss
Compared across two architectures:

**3-layer SPLADE-v3 first3** (from `results/splade_first3_loss_ablation.md`):

| Loss | BEIR avg NDCG@10 | Delta |
|---|---:|---:|
| mse | 0.3430 | baseline |
| margin_mse (ColBERT) | 0.3178 | -0.0252 |
| kd (ColBERT) | 0.2661 | -0.0769 |
| cosine | 0.2116 | -0.1314 |

**Static (0-layer) model** (from `results/static_loss_ablation.md`):

| Loss | BEIR avg NDCG@10 | Delta vs mse |
|---|---:|---:|
| cosine | 0.3378 | +0.0052 |
| mse | 0.3326 | baseline |
| kd (ColBERT) | 0.3288 | -0.0038 |
| margin_mse (ColBERT) | 0.3248 | -0.0078 |

Key findings:
1. **Ranking supervision consistently hurts** across both architectures — ColBERT KD and MarginMSE underperform direct vector distillation in every configuration tested.
2. **Cosine reversal**: for the 3-layer model, cosine is catastrophically bad (-0.1314); for the static model, cosine slightly outperforms MSE (+0.0052). Likely explanation: the 3-layer model's hidden states need calibrated magnitude to produce well-scaled sparse outputs; the static model has no such hidden-state dependency, so direction alignment is sufficient.
3. Direct vector MSE is the safe default for any shallow model with transformer layers; cosine may be appropriate for the static limit.

### 7.5 Factorized Lexical Matrix
Quality vs. compression tradeoff across factor dimensions (64–512 for SPLADE, 256–2048 for Lion). Factorization never improves quality; best SPLADE at d256 loses 0.0029 NDCG; Lion requires d2048 to get within 0.0041 of baseline. Frame as a second compression axis (parameter count, memory) rather than a quality improvement. (See `results/factorized_first3_grid.md`.)

---

## 8. Analysis

### 8.1 Query Throughput and Latency
**PENDING — needs measurement.** Queries/sec, mean latency, peak GPU memory for each configuration. Absolute numbers and speedup relative to full encoder. The static model should be especially fast: no matrix multiplications in the query encoder path.

### 8.2 Why Does Freezing the Head Help?
Cosine similarity between shallow and full model query SPLADE vectors, with and without head freezing. Supports the vocabulary geometry hypothesis. Can be measured from saved checkpoints.

### 8.3 SPLADE-v3 vs. Lion: Why the Performance Gap?
SPLADE first3 achieves 0.3483 vs. Lion first3 at 0.3117. Possible explanations: encoder vs. decoder architecture, tied embeddings, model size, pretraining corpus, sparsity distribution. This section can be speculative if direct evidence is not available.

### 8.4 What Does the Static Model Learn?
The static model has no contextual processing — its query vector is a fixed weighted sum of token indicators. Interesting to inspect: which tokens get high vs. low weights, how the weight distribution relates to IDF, whether the learned weights are sparse. This section could be a short qualitative analysis of the learned vocabulary weights.

---

## 9. Conclusion

Summary of findings. Practical recommendations:
- **3-layer spaced SPLADE-v3 with frozen head and MSE distillation** as a drop-in replacement for full query encoding (best quality–efficiency tradeoff found).
- **Static (0-layer) model** as the extreme efficiency choice when transformer cost is completely unacceptable, at a modest quality penalty.
- Avoid unfreezing the lexical head, factorizing the lexical matrix as a quality improvement, or using ranking-score supervision in place of vector distillation.

Limitations: evaluation is on a BEIR subset only; efficiency numbers not yet measured; no repeated seeds; full-model baselines not yet directly measured in this setup.

Future directions: full BEIR; larger model families; adaptive early exit (query-complexity-aware depth); hybrid loss (MSE + ranking signal); weight analysis of static model; deployment against a real sparse index.

---

## Appendix

### A. Hyperparameters
Full training configuration tables for SPLADE and Lion experiments.

### B. Per-Dataset BEIR Results
Full breakdown of NDCG@10 and MRR@10 per dataset for all variants. (Available in per-result `.md` files in `results/`.)

### C. Training Curves
NanoMSMARCO validation curves for key configurations.

### D. Static Model Weight Analysis
(If included as a qualitative section) Sorted vocabulary weights, comparison to IDF, discussion of what "importance" the model learns to assign to different query terms.

---

## Remaining Experiments Before Camera-Ready

See the **Pending** table at the top for what's still to run. Status summary:

- **Done**: full SPLADE-v3 and Lion layer sweeps (all 1–5 × first/spaced/last), head-unfreezing ablation, factorization grid (SPLADE d64–512, Lion d256–2048), loss ablations on 3-layer and static models (all 4 variants), static model training (mse/cosine/kd/margin_mse with BEIR results).
- **TODO**: full SPLADE-v3 and Lion BEIR baselines (doc_only eval), static cosine retrain to 10k steps.
- **DEFERRED / BM25**: no pyserini or rank_bm25 installed; add when library is available.
- **Intentionally later**: full BEIR, MS MARCO dev, throughput and latency measurements.
