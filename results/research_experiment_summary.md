# Research Experiment Summary: Shallow Sparse Query Encoders

Generated: 2026-05-22

## Executive Summary

This project studies whether SPLADE-style sparse retrievers can be made cheaper at query time by pruning only the query encoder while keeping the document encoder fixed. The central design is asymmetric: documents are encoded once with the original full sparse model, while queries are encoded with a shallow model created by selecting a subset of transformer layers from the same model family. Because the shallow query encoder retains the same tokenizer and lexical output space, its query vectors can be scored directly against the existing document index with the standard sparse dot product.

The current experimental evidence supports a simple baseline:

- Use a shallow query encoder derived from `naver/splade-v3`.
- Keep the lexical embedding/language-model head frozen after the warmup phase.
- Do not factorize the lexical matrix unless compression is required.
- Use direct vector MSE distillation against the full SPLADE query representation.
- For 3-layer SPLADE, evenly spaced layers `[0, 6, 11]` are the best current BEIR-small configuration, with average `NDCG@10/MRR@10 = 0.3509/0.4067`.

The Lion sparse model family was also tested with 3, 4, and 5 selected layers. The best Lion configuration so far is the 4-layer spaced variant `[0, 5, 10, 15]`, with average `0.3163/0.3611` on the same BEIR-small subset. Lion improves slightly from 3 to 4 layers, but remains below SPLADE-v3 in these experiments.

The main ablations so far are consistent: unfreezing the lexical head hurts, factorization does not improve quality, and alternative losses underperform direct vector MSE.

## Project Aim

The working research question is:

> Can a sparse lexical retriever preserve most of the retrieval quality of a full SPLADE-style model while using a much smaller query encoder, without rebuilding or changing the document index?

The motivation is query-time efficiency. SPLADE-style models are attractive because they produce sparse lexical vectors that can be indexed and searched efficiently, but the transformer encoder remains a significant query-time cost. If the document encoder is left unchanged and only the query encoder is pruned, then:

- existing document indexes can remain valid;
- retrieval remains interpretable in the lexical vocabulary space;
- serving can potentially become cheaper because queries pass through fewer transformer layers;
- the approach can be evaluated cleanly as a drop-in replacement for full query encoding.

The project is therefore not currently focused on tokenizer transplant, vocabulary mapping, SAE projection, or dense-to-sparse conversion. Those older directions have been intentionally removed from the active config surface. The active research direction is pruning-based shallow query encoding.

## Method

Each active experiment starts from a pretrained sparse retrieval model and constructs a shallow query encoder by selecting transformer layers to keep. The document encoder remains the original full model.

Current active stages:

| Stage | Base model | Purpose |
|---|---|---|
| `splade_shallow` | `naver/splade-v3` | SPLADE-v3 shallow query encoder experiments |
| `lion_shallow` | `hzeng/Lion-SP-1B-llama3-marco-mntp` | Lion sparse model shallow query encoder experiments |

The main exposed configuration variables are intentionally minimal:

| Config choice | Meaning |
|---|---|
| `layer_indices` / layer count | Which transformer layers are retained in the query encoder |
| `factorize_embeddings` / factorization dimension | Whether the lexical output matrix is replaced by a low-rank factorized form |

The primary training setup uses:

- selected-layer shallow query encoder;
- full original model as the teacher;
- frozen document encoder;
- lexical embedding/head frozen after the initial warmup;
- non-factorized lexical matrix unless explicitly testing factorization;
- NanoMSMARCO validation during training;
- BEIR-small evaluation after training using the best NanoMSMARCO checkpoint.

## Evaluation Protocol

The best checkpoint for each run is selected by NanoMSMARCO validation. That checkpoint is then evaluated on a small BEIR subset:

- `nfcorpus`
- `scifact`
- `arguana`
- `scidocs`
- `fiqa`

The reported aggregate is the unweighted average across these five datasets. Metrics are reported as `NDCG@10/MRR@10`.

Document indexes are expected to be built for the corresponding full document model. The shallow query encoder is evaluated against that full-model document index, which is important for testing the intended deployment setup.

## Current Best Results

| Family | Best current config | Layers | Factorized | Head after warmup | Loss | Best NanoMSMARCO | BEIR-small avg |
|---|---|---:|---|---|---|---:|---:|
| SPLADE-v3 | spaced 3 layers | `[0, 6, 11]` | no | frozen | vector MSE | 0.7070 @ 50000 | 0.3509/0.4067 |
| Lion | spaced 4 layers | `[0, 5, 10, 15]` | no | frozen | vector MSE | 0.6895 @ 50000 | 0.3163/0.3611 |

SPLADE-v3 is currently the stronger base for this pruning setup. The best Lion variant is competitive within the Lion sweep but substantially behind SPLADE-v3 on the BEIR-small average.

## Experiment Group 1: SPLADE Layer Selection

Aim: test whether a 3-layer shallow SPLADE-v3 query encoder works better when keeping the first layers, evenly spaced layers, or the last layers.

Setup:

- Stage: `splade_shallow`
- Base model: `naver/splade-v3`
- Factorized embeddings: no
- Embedding/head matrix frozen after warmup: yes
- BEIR-small datasets: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`
- Source result file: `results/splade_shallow_layer_sweep.md`

| Run | Layers | Best NanoMSMARCO | BEIR avg NDCG@10 | BEIR avg MRR@10 |
|---|---:|---:|---:|---:|
| first3 | `[0, 1, 2]` | 0.7139 @ 30000 | 0.3483 | 0.4050 |
| spaced3 | `[0, 6, 11]` | 0.7070 @ 50000 | 0.3509 | 0.4067 |
| last3 | `[9, 10, 11]` | 0.6966 @ 50000 | 0.3473 | 0.4019 |

Per-dataset results:

| Run | nfcorpus | scifact | arguana | scidocs | fiqa | Average |
|---|---:|---:|---:|---:|---:|---:|
| first3 NDCG@10 | 0.3301 | 0.6276 | 0.3181 | 0.1498 | 0.3158 | 0.3483 |
| spaced3 NDCG@10 | 0.3327 | 0.6289 | 0.3208 | 0.1539 | 0.3183 | 0.3509 |
| last3 NDCG@10 | 0.3333 | 0.6244 | 0.3198 | 0.1505 | 0.3083 | 0.3473 |

Interpretation:

The SPLADE 3-layer result is robust across layer choices. The spread between first3, spaced3, and last3 is small, but spaced3 is the current best BEIR-small configuration. First3 has the best NanoMSMARCO validation score, which suggests NanoMSMARCO and BEIR-small are directionally useful but not perfectly aligned.

## Experiment Group 2: Lion Layer Count and Layer Selection

Aim: test whether Lion shallow query encoders benefit from keeping 3, 4, or 5 layers, and whether first, spaced, or last layer selection works best.

Setup:

- Stage: `lion_shallow`
- Base model: `hzeng/Lion-SP-1B-llama3-marco-mntp`
- Factorized embeddings: no
- Embedding/head matrix frozen after warmup: yes
- BEIR-small datasets: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`
- Source result file: `results/lion_shallow_layer_sweep.md`

| Run | Layers | Best NanoMSMARCO | BEIR avg NDCG@10 | BEIR avg MRR@10 |
|---|---:|---:|---:|---:|
| first3 | `[0, 1, 2]` | 0.6587 @ 40000 | 0.3117 | 0.3581 |
| spaced3 | `[0, 8, 15]` | 0.6888 @ 30000 | 0.3084 | 0.3537 |
| last3 | `[13, 14, 15]` | 0.6859 @ 50000 | 0.3068 | 0.3558 |
| first4 | `[0, 1, 2, 3]` | 0.6796 @ 40000 | 0.3126 | 0.3593 |
| spaced4 | `[0, 5, 10, 15]` | 0.6895 @ 50000 | 0.3163 | 0.3611 |
| last4 | `[12, 13, 14, 15]` | 0.6780 @ 50000 | 0.3070 | 0.3538 |
| first5 | `[0, 1, 2, 3, 4]` | 0.6751 @ 40000 | 0.3125 | 0.3576 |
| spaced5 | `[0, 4, 8, 11, 15]` | 0.6920 @ 50000 | 0.3141 | 0.3595 |
| last5 | `[11, 12, 13, 14, 15]` | 0.6797 @ 40000 | 0.3042 | 0.3508 |

Per-dataset NDCG@10:

| Run | nfcorpus | scifact | arguana | scidocs | fiqa | Average |
|---|---:|---:|---:|---:|---:|---:|
| first3 | 0.2929 | 0.5710 | 0.2948 | 0.1310 | 0.2690 | 0.3117 |
| spaced3 | 0.2881 | 0.5646 | 0.3004 | 0.1266 | 0.2623 | 0.3084 |
| last3 | 0.2863 | 0.5564 | 0.2938 | 0.1295 | 0.2681 | 0.3068 |
| first4 | 0.2948 | 0.5715 | 0.2956 | 0.1311 | 0.2700 | 0.3126 |
| spaced4 | 0.2918 | 0.5794 | 0.3053 | 0.1311 | 0.2737 | 0.3163 |
| last4 | 0.2899 | 0.5515 | 0.2984 | 0.1268 | 0.2682 | 0.3070 |
| first5 | 0.2954 | 0.5700 | 0.2983 | 0.1283 | 0.2703 | 0.3125 |
| spaced5 | 0.2921 | 0.5743 | 0.3001 | 0.1318 | 0.2720 | 0.3141 |
| last5 | 0.2939 | 0.5496 | 0.2878 | 0.1267 | 0.2632 | 0.3042 |

Interpretation:

Lion benefits modestly from moving from 3 to 4 retained layers, but 5 layers does not improve the BEIR-small average. The best Lion BEIR result is spaced4, while the best NanoMSMARCO score is spaced5. Last-layer-only selection is consistently weak. Compared with SPLADE-v3, Lion remains lower on every aggregate tested so far.

## Experiment Group 3: Head Unfreezing

Aim: test whether training the lexical embedding/head with a small learning rate after 5k warmup steps improves the best shallow configurations.

Setup:

- Factorized embeddings: no
- Warmup behavior: embedding/head trains during the first 5k steps, then continues at a reduced head learning-rate scale
- `freeze_head_after_warmup`: false
- `head_lr_scale`: 0.1 unless overridden
- BEIR-small datasets: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`
- Source result file: `results/head_unfrozen_ablation.md`

| Run | Stage | Layers | Best NanoMSMARCO | Frozen baseline avg | Unfrozen avg | NDCG delta |
|---|---|---:|---:|---:|---:|---:|
| splade_spaced3_head_unfrozen | `splade_shallow` | `[0, 6, 11]` | 0.7081 @ 10000 | 0.3509/0.4067 | 0.3431/0.3986 | -0.0078 |
| lion_first3_head_unfrozen | `lion_shallow` | `[0, 1, 2]` | 0.6330 @ 50000 | 0.3117/0.3581 | 0.2966/0.3417 | -0.0151 |
| lion_spaced4_head_unfrozen | `lion_shallow` | `[0, 5, 10, 15]` | n/a | 0.3163/0.3611 | n/a | n/a |

Per-dataset NDCG@10 for completed runs:

| Run | nfcorpus | scifact | arguana | scidocs | fiqa | Average |
|---|---:|---:|---:|---:|---:|---:|
| SPLADE spaced3 head-unfrozen | 0.3259 | 0.6200 | 0.3126 | 0.1495 | 0.3073 | 0.3431 |
| Lion first3 head-unfrozen | 0.2725 | 0.5408 | 0.2862 | 0.1270 | 0.2564 | 0.2966 |

Interpretation:

Unfreezing the lexical head after warmup hurt both completed runs. This suggests that preserving the original lexical projection is important for compatibility with the full document index. It also keeps the training problem simpler: the shallow query encoder learns to reproduce the full model's query vectors without shifting the vocabulary-space geometry.

The larger Lion head-unfrozen variants were not completed due to GPU memory limits on the available 8GB hardware, even with small microbatches and gradient checkpointing. This is also practically relevant: full-head Lion unfreezing appears expensive enough that it may not be a good default ablation unless larger hardware is available.

## Experiment Group 4: Factorized Lexical Matrix

Aim: test whether replacing the lexical matrix with an SVD-initialized low-rank factorization improves compression or quality for first3 SPLADE and first3 Lion while keeping the embedding/head frozen after warmup.

Setup:

- SPLADE layers: `[0, 1, 2]`
- Lion layers: `[0, 1, 2]`
- Factorization initialization: SVD
- Factorized embedding/head matrix frozen after warmup: yes
- BEIR-small datasets: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`
- Source result file: `results/factorized_first3_grid.md`

### SPLADE Factorization

Frozen non-factorized SPLADE first3 baseline: `0.3483/0.4050`.

| Run | Factor dim | Best NanoMSMARCO | BEIR avg | NDCG delta vs baseline |
|---|---:|---:|---:|---:|
| splade_first3_factorized_d64 | 64 | 0.6915 @ 20000 | 0.3379/0.3893 | -0.0104 |
| splade_first3_factorized_d128 | 128 | 0.7066 @ 50000 | 0.3433/0.3940 | -0.0050 |
| splade_first3_factorized_d256 | 256 | 0.6960 @ 50000 | 0.3454/0.3992 | -0.0029 |
| splade_first3_factorized_d512 | 512 | 0.7055 @ 20000 | 0.3428/0.3979 | -0.0055 |

SPLADE factorization nearly recovers the baseline at dimension 256, but does not improve over the non-factorized model. If factorization is used, it should currently be framed as a compression/serving tradeoff rather than a quality improvement.

### Lion Factorization

Frozen non-factorized Lion first3 baseline: `0.3117/0.3581`.

| Run | Factor dim | Best NanoMSMARCO | BEIR avg | NDCG delta vs baseline |
|---|---:|---:|---:|---:|
| lion_first3_factorized_d256 | 256 | 0.6566 @ 50000 | 0.2722/0.3130 | -0.0395 |
| lion_first3_factorized_d512 | 512 | 0.6718 @ 50000 | 0.2889/0.3334 | -0.0228 |
| lion_first3_factorized_d1024 | 1024 | 0.6564 @ 20000 | 0.2799/0.3226 | -0.0318 |
| lion_first3_factorized_d2048 | 2048 | 0.6825 @ 40000 | 0.3076/0.3547 | -0.0041 |

Lion factorization is much more sensitive to rank. Only dimension 2048 comes close to the non-factorized first3 baseline. Lower ranks cause substantial degradation.

Interpretation:

Factorization does not look like a promising accuracy-improving direction in the current setup. It may still be useful if the paper wants to discuss a second compression axis: transformer depth reduction plus lexical-matrix rank reduction. That claim would need to be supported by parameter, memory, latency, and quality tradeoff plots.

## Experiment Group 5: Loss Function Ablation

Aim: test whether the training loss is limiting shallow SPLADE-v3 first3 quality.

Setup:

- Stage: `splade_shallow`
- Base model: `naver/splade-v3`
- Layers: `[0, 1, 2]`
- Factorized embeddings: no
- Embedding/head frozen after warmup: yes
- Non-factorized first3 baseline from the original layer sweep: `0.3483/0.4050`
- ColBERT score supervision file: `data/colbertv2_msmarco_64way.json`
- BEIR-small datasets: `nfcorpus`, `scifact`, `arguana`, `scidocs`, `fiqa`
- Source result file: `results/splade_first3_loss_ablation.md`

| Run | Loss | Supervision target | Best NanoMSMARCO | BEIR avg | NDCG delta vs baseline |
|---|---|---|---:|---:|---:|
| mse | `mse` | full SPLADE query vector | 0.7079 @ 30000 | 0.3430/0.3953 | -0.0053 |
| cosine | `cosine` | full SPLADE query vector direction | 0.3375 @ 10000 | 0.2116/0.2421 | -0.1367 |
| kd_colbert | `kd` | ColBERT n-way passage scores | 0.5834 @ 30000 | 0.2661/0.3032 | -0.0822 |
| margin_mse_colbert | `margin_mse` | ColBERT positive-negative score margins | 0.6890 @ 50000 | 0.3178/0.3675 | -0.0305 |

Per-dataset NDCG@10:

| Run | nfcorpus | scifact | arguana | scidocs | fiqa | Average |
|---|---:|---:|---:|---:|---:|---:|
| mse | 0.3278 | 0.6100 | 0.3199 | 0.1502 | 0.3070 | 0.3430 |
| cosine | 0.2118 | 0.4725 | 0.1894 | 0.0819 | 0.1022 | 0.2116 |
| kd_colbert | 0.2532 | 0.5183 | 0.2526 | 0.1143 | 0.1922 | 0.2661 |
| margin_mse_colbert | 0.2939 | 0.5786 | 0.2963 | 0.1405 | 0.2798 | 0.3178 |

Interpretation:

Direct vector MSE remains the best tested loss. Cosine loss is much worse, likely because preserving only vector direction is insufficient for sparse lexical retrieval where magnitude and activation calibration matter. ColBERT score supervision also underperforms direct vector distillation. MarginMSE is much better than KL-style KD over ColBERT scores, but still clearly below MSE.

For now, the training objective should remain direct MSE to the full sparse query vector. A future paper version could test hybrid objectives, but the current ablation does not justify replacing MSE.

## Cross-Ablation Findings

### Finding 1: Shallow SPLADE-v3 is viable

The SPLADE-v3 shallow 3-layer query encoder produces stable BEIR-small performance across first, spaced, and last layer selections. The best current configuration, spaced3, reaches `0.3509/0.4067` on BEIR-small. The small spread across layer choices suggests that much of the query-side sparse behavior can be recovered with only three retained layers when trained by vector distillation.

### Finding 2: Layer spacing is useful but not universally dominant

For SPLADE-v3, spaced3 is best by BEIR-small average, but first3 is best on NanoMSMARCO. For Lion, spaced4 is best by BEIR-small average and spaced5 is best on NanoMSMARCO. This makes layer spacing a strong candidate default, but also shows that validation choice matters.

### Finding 3: Freezing the lexical head is currently preferable

Unfreezing the lexical head after warmup reduced BEIR-small quality for both completed runs:

- SPLADE spaced3: `0.3509 -> 0.3431`
- Lion first3: `0.3117 -> 0.2966`

The likely explanation is that the frozen head preserves compatibility with the full-model document index. If the query head drifts, the query vectors may remain trainable on the in-domain objective while becoming less aligned with the fixed document vectors.

### Finding 4: Factorization is a compression knob, not a quality improvement

SPLADE factorization at dimension 256 nearly matches the non-factorized first3 baseline, but no factorized setting beats it. Lion requires a much larger factor dimension, 2048, to get close to the non-factorized first3 baseline. This makes factorization potentially useful for deployment tradeoffs, but not currently central to the retrieval-quality story.

### Finding 5: MSE is the strongest tested loss

The loss ablation gives a clear ordering:

`MSE > MarginMSE with ColBERT score supervision > KD with ColBERT score supervision > Cosine`

This supports a simple explanation: the shallow model is best trained to reproduce the full sparse vector directly. Retrieval-score supervision may discard too much lexical calibration information, and cosine loss ignores important magnitude information.

## Suggested Paper Framing

A clean paper framing would be:

> We study asymmetric pruning for sparse lexical retrieval: keep the expensive full document encoder offline, but replace the online query encoder with a shallow layer-selected student that remains in the same lexical sparse vector space. This preserves index compatibility while reducing query-time transformer depth.

Possible contributions:

1. A simple layer-selection method for constructing shallow sparse query encoders from pretrained SPLADE-style models.
2. An asymmetric retrieval setup where the full document index is reused and only the query encoder is compressed.
3. An empirical study of layer placement, layer count, lexical-head freezing, low-rank lexical factorization, and distillation losses.
4. Evidence that direct sparse-vector distillation with a frozen lexical head is a strong default for preserving retrieval quality.

The strongest currently supported claim is narrow but useful:

> For SPLADE-v3 on the tested BEIR-small subset, a 3-layer query encoder trained by direct vector MSE can preserve useful sparse retrieval behavior against the unchanged full-model document index, and common alternatives such as head unfreezing, low-rank factorization, and score-only distillation do not improve quality in this setup.

## Claims Not Yet Fully Supported

The current experiments are promising, but several claims should not yet be made strongly:

- Full BEIR generalization has not been established; only the smaller BEIR subset has been evaluated.
- Query-time speedup has not yet been measured in a formal table.
- Parameter count, FLOPs, activation memory, and query throughput are not yet reported.
- Indexing cost is unchanged by design, but index reuse and index compatibility should be demonstrated explicitly.
- The experiments do not yet include repeated seeds, so small differences between layer choices should be treated cautiously.
- The current comparison is mostly internal; external baselines such as full SPLADE, DistilSPLADE-style models, uniCOIL, BM25, and dense retrievers should be added depending on the target venue.
- Larger Lion head-unfreezing runs were blocked by memory, so the Lion unfreezing conclusion is partial.

## Recommended Next Experiments

The next round should focus on making the story paper-ready rather than adding many new knobs.

1. Add efficiency measurements for the best configurations:
   - query latency;
   - encoder FLOPs;
   - trainable and total parameter counts;
   - peak GPU memory;
   - average query vector nonzero count;
   - retrieval latency against the existing sparse index.

2. Evaluate the best configurations more broadly:
   - full BEIR, if compute permits;
   - MS MARCO dev;
   - TREC DL 2019/2020 if available;
   - the same BEIR-small subset with repeated seeds for the top configurations.

3. Add full-model baselines in the same table:
   - full SPLADE-v3 query encoder against full SPLADE-v3 document index;
   - full Lion query encoder against full Lion document index;
   - BM25 as a sparse lexical baseline;
   - any relevant distilled sparse retriever baseline.

4. Expand the best SPLADE-v3 family only if needed:
   - 2-layer and 4-layer SPLADE-v3 variants;
   - more systematic layer sets around the current spaced3 winner;
   - compare first-N vs spaced-N under identical compute.

5. Keep loss work limited unless there is a concrete hypothesis:
   - MSE is currently the default;
   - possible future tests are MSE plus score loss, MSE plus sparsity/activation calibration, or margin loss with the full SPLADE teacher rather than ColBERT supervision.

6. Treat factorization as an efficiency ablation:
   - report memory and parameter savings;
   - use quality-efficiency curves;
   - do not frame it as an accuracy improvement unless future results change.

## Candidate Main Table for a Draft

| Model family | Query encoder | Layers | Head | Factor dim | Loss | BEIR-small avg | Notes |
|---|---|---:|---|---:|---|---:|---|
| SPLADE-v3 | shallow first3 | `[0, 1, 2]` | frozen | none | MSE | 0.3483/0.4050 | Best NanoMSMARCO among SPLADE layer sweep |
| SPLADE-v3 | shallow spaced3 | `[0, 6, 11]` | frozen | none | MSE | 0.3509/0.4067 | Best SPLADE BEIR-small |
| SPLADE-v3 | shallow last3 | `[9, 10, 11]` | frozen | none | MSE | 0.3473/0.4019 | Similar but slightly weaker |
| Lion | shallow first3 | `[0, 1, 2]` | frozen | none | MSE | 0.3117/0.3581 | Best Lion 3-layer BEIR-small |
| Lion | shallow spaced4 | `[0, 5, 10, 15]` | frozen | none | MSE | 0.3163/0.3611 | Best Lion BEIR-small |
| Lion | shallow spaced5 | `[0, 4, 8, 11, 15]` | frozen | none | MSE | 0.3141/0.3595 | Best Lion NanoMSMARCO |

## Candidate Ablation Table for a Draft

| Ablation | Best setting tested | Result | Practical conclusion |
|---|---|---:|---|
| SPLADE layer choice | spaced3 `[0, 6, 11]` | 0.3509/0.4067 | Use spaced3 as current SPLADE default |
| Lion layer choice/count | spaced4 `[0, 5, 10, 15]` | 0.3163/0.3611 | 4 spaced layers are best among Lion variants |
| Head unfreezing | frozen head | SPLADE unfrozen drops by 0.0078 NDCG | Keep lexical head frozen |
| SPLADE factorization | no factorization | best factorized d256 still -0.0029 | Avoid factorization unless compression is needed |
| Lion factorization | no factorization | best factorized d2048 still -0.0041 | Low-rank Lion needs high rank to recover quality |
| Loss | MSE | best tested loss | Use direct vector distillation |

## Open Research Questions

- Why is SPLADE-v3 substantially more robust than Lion in this shallow-query setup?
- Is the performance gap due to architecture, vocabulary/head behavior, pretraining, sparsity distribution, or the training data used for distillation?
- Would a 4-layer SPLADE-v3 sweep improve over the current 3-layer spaced result enough to justify the extra query cost?
- Can score supervision help when combined with MSE, rather than replacing it?
- Does the shallow query encoder preserve term-level interpretability in the same way as the full model?
- How much of the observed quality comes from retaining the lexical head versus retaining early/middle transformer layers?
- Is NanoMSMARCO the right checkpoint-selection signal for BEIR transfer, or should model selection use a broader validation mixture?

## Source Result Files

- `results/splade_shallow_layer_sweep.md`
- `results/lion_shallow_layer_sweep.md`
- `results/head_unfrozen_ablation.md`
- `results/factorized_first3_grid.md`
- `results/splade_first3_loss_ablation.md`

