# Ablation Study — Consolidated Results

## Models

| Model | Arch | Layers | Vocab | q_nnz (approx) |
|---|---|---:|---:|---:|
| SPLADE v3 (`naver/splade-v3`) | BERT-base | 12 | 30,522 | ~30–60 |
| Lion (`hzeng/Lion-SP-1B-llama3-marco-mntp`) | LLaMA 3 1B | 16 | 128,256 | ~200–300 |

**Ceilings (full model, doc encoder used for both query and doc)**

| Model | MS-MARCO NDCG@10 | MS-MARCO MRR@10 |
|---|---:|---:|
| SPLADE v3 | 0.4657 | 0.3989 |
| Lion | 0.4758 | 0.4085 |

Eval corpus: 8,841,823 MS-MARCO passages. BEIR datasets (5-set): nfcorpus, scifact, arguana, scidocs, fiqa.

---

## 1. Layer Sweep

Training objective: MSE distillation from frozen full doc encoder. Head frozen after warmup.

### 1a. SPLADE v3 — MS-MARCO Dev (full 8.8M index)

| Variant | Layers | NDCG@10 | % of full | MRR@10 | % of full |
|---|---|---:|---:|---:|---:|
| **SPLADE v3 (full)** | all 12 | **0.4657** | 100.0% | **0.3989** | 100.0% |
| first1 | `[0]` | 0.4356 | 93.5% | 0.3716 | 93.2% |
| first2 | `[0,1]` | 0.4404 | 94.6% | 0.3755 | 94.1% |
| first3 | `[0,1,2]` | 0.4447 | 95.5% | 0.3795 | 95.1% |
| first4 | `[0,1,2,3]` | 0.4446 | 95.5% | 0.3790 | 95.0% |
| first5 | `[0,1,2,3,4]` | 0.4485 | 96.3% | 0.3840 | 96.3% |
| last1 | `[11]` | 0.4383 | 94.1% | 0.3744 | 93.9% |
| last2 | `[10,11]` | 0.4361 | 93.7% | 0.3720 | 93.3% |
| last3 | `[9,10,11]` | 0.4463 | 95.8% | 0.3824 | 95.9% |
| last4 | `[8,9,10,11]` | 0.4462 | 95.8% | 0.3815 | 95.6% |
| last5 | `[7,8,9,10,11]` | 0.4469 | 95.9% | 0.3831 | 96.0% |
| spaced1 | `[6]` | 0.4397 | 94.4% | 0.3760 | 94.3% |
| spaced2 | `[0,11]` | 0.4453 | 95.6% | 0.3809 | 95.5% |
| spaced3 | `[0,6,11]` | 0.4475 | 96.1% | 0.3830 | 96.0% |
| spaced4 | `[0,4,7,11]` | 0.4480 | 96.2% | 0.3829 | 96.0% |
| spaced5 | `[0,3,6,8,11]` | 0.4423 | 95.0% | 0.3772 | 94.6% |

### 1b. SPLADE v3 — BEIR avg (NDCG@10 / MRR@10)

5-dataset avg (nfcorpus, scifact, arguana, scidocs, fiqa) for all variants; 13-dataset avg for spaced sweep (see `results/beir_splade_spaced_eval.md` for per-dataset breakdown).

| Variant | Layers | NDCG@10 (5-ds) | MRR@10 (5-ds) | NDCG@10 (13-ds) | MRR@10 (13-ds) |
|---|---|---:|---:|---:|---:|
| **SPLADE v3 (full)** | all 12 | — | — | **0.4882** | **0.6474** |
| first1 | `[0]` | 0.3444 | 0.3980 | — | — |
| first2 | `[0,1]` | 0.3465 | 0.3999 | — | — |
| first3 | `[0,1,2]` | 0.3483 | 0.4050 | — | — |
| first4 | `[0,1,2,3]` | 0.3462 | 0.4014 | — | — |
| first5 | `[0,1,2,3,4]` | 0.3503 | 0.4066 | — | — |
| last1 | `[11]` | 0.3446 | 0.3980 | — | — |
| last2 | `[10,11]` | 0.3359 | 0.3900 | — | — |
| last3 | `[9,10,11]` | 0.3473 | 0.4019 | — | — |
| last4 | `[8,9,10,11]` | 0.3460 | 0.3980 | — | — |
| last5 | `[7,8,9,10,11]` | 0.3460 | 0.4000 | — | — |
| spaced1 | `[6]` | 0.3433 | 0.3983 | 0.4533 | 0.6082 |
| spaced2 | `[0,11]` | 0.3499 | 0.4048 | — | — |
| spaced3 | `[0,6,11]` | 0.3509 | 0.4067 | 0.4627 | 0.6219 |
| spaced4 | `[0,4,7,11]` | 0.3513 | 0.4056 | — | — |
| spaced5 | `[0,3,6,8,11]` | 0.3467 | 0.4002 | 0.4607 | 0.6213 |

### 1c. Lion — MS-MARCO Dev (full 8.84M index, spaced variants only)

> **Note on Lion layer sensitivity**: Lion recovers only 76–87% of full performance with 1–5 layers, compared to 93–96% for SPLADE v3. The likely cause is that Lion produces much denser sparse vectors (q_nnz ~200–300) relative to SPLADE (q_nnz ~30–60). Higher q_nnz means the model needs to learn to "compress" relevance signals into many co-active dimensions — a harder task that benefits more from additional transformer layers. The monotonic improvement from spaced1→spaced5 (+9.8 NDCG points) is steeper than SPLADE's equivalent range (~2.5 points), consistent with this hypothesis.

| Variant | Layers | NDCG@10 | % of full | MRR@10 | % of full |
|---|---|---:|---:|---:|---:|
| **Lion (full)** | all 16 | **0.4758** | 100.0% | **0.4085** | 100.0% |
| spaced1 | `[0]` | 0.3650 | 76.7% | 0.3065 | 75.0% |
| spaced2 | `[0,15]` | 0.3830 | 80.5% | 0.3230 | 79.1% |
| spaced3 | `[0,8,15]` | 0.3921 | 82.4% | 0.3310 | 81.0% |
| spaced4 | `[0,5,10,15]` | 0.4049 | 85.1% | 0.3437 | 84.1% |
| spaced5 | `[0,4,8,11,15]` | 0.4117 | 86.5% | 0.3499 | 85.7% |

### 1d. Lion — BEIR 5-dataset avg (all 15 variants)

| Variant | Layers | NDCG@10 | MRR@10 |
|---|---|---:|---:|
| first1 | `[0]` | 0.3070 | 0.3527 |
| first2 | `[0,1]` | 0.3091 | 0.3508 |
| first3 | `[0,1,2]` | 0.3117 | 0.3581 |
| first4 | `[0,1,2,3]` | 0.3126 | 0.3593 |
| first5 | `[0,1,2,3,4]` | 0.3125 | 0.3576 |
| last1 | `[15]` | 0.2881 | 0.3320 |
| last2 | `[14,15]` | 0.2981 | 0.3416 |
| last3 | `[13,14,15]` | 0.3068 | 0.3558 |
| last4 | `[12,13,14,15]` | 0.3070 | 0.3538 |
| last5 | `[11,12,13,14,15]` | 0.3042 | 0.3508 |
| spaced1 | `[8]` | 0.2870 | 0.3298 |
| spaced2 | `[0,15]` | 0.3031 | 0.3479 |
| spaced3 | `[0,8,15]` | 0.3084 | 0.3537 |
| spaced4 | `[0,5,10,15]` | 0.3163 | 0.3611 |
| spaced5 | `[0,4,8,11,15]` | 0.3141 | 0.3595 |

---

## 2. Loss Ablation

### 2a. SPLADE v3 first3 — BEIR + MS-MARCO

Baseline (default MSE, frozen head): BEIR avg 0.3483 / MS-MARCO NDCG@10 0.4447.

| Loss | BEIR NDCG@10 | BEIR MRR@10 | MS-MARCO NDCG@10 | MS-MARCO MRR@10 |
|---|---:|---:|---:|---:|
| mse (baseline) | 0.3430 | 0.3953 | 0.4455 | 0.3811 |
| cosine | 0.2116 | 0.2421 | 0.0174 | 0.0120 |
| kd (ColBERT) | 0.2661 | 0.3032 | 0.3148 | 0.2651 |
| margin_mse (ColBERT) | 0.3178 | 0.3675 | 0.3635 | 0.3067 |

MSE distillation dominates. Cosine loss collapses on MS-MARCO (magnitude information is critical for SPLADE). Ranking-supervision losses (KD, MarginMSE) both underperform direct vector distillation.

### 2b. SPLADE v3 static (0-layer) — BEIR + MS-MARCO

Architecture: single learnable weight per vocab token (30,522 trainable params). Baseline for "how much is token identity alone worth."

| Loss | BEIR NDCG@10 | BEIR MRR@10 | MS-MARCO NDCG@10 | MS-MARCO MRR@10 |
|---|---:|---:|---:|---:|
| mse | 0.3326 | 0.3861 | 0.3999 | 0.3401 |
| cosine | 0.3378 | 0.3929 | 0.4122 | 0.3512 |
| kd (ColBERT) | 0.3288 | 0.3817 | 0.3834 | 0.3247 |
| margin_mse (ColBERT) | 0.3248 | 0.3778 | 0.3608 | 0.3036 |

Unlike the 3-layer model, cosine marginally outperforms MSE at static depth (no hidden-state magnitude to preserve). Ranking losses still hurt — replicates the first3 finding.

### 2c. Lion static (0-layer) — BEIR

| Loss | BEIR NDCG@10 | BEIR MRR@10 |
|---|---:|---:|
| kd (ColBERT) | 0.3009 | 0.3520 |
| margin_mse (ColBERT) | 0.3033 | 0.3534 |

Both are competitive with Lion first3 frozen (0.3117) despite 0 transformer layers — suggests the static token-weight baseline is unusually strong for Lion, likely due to Lion's vocabulary (128K tokens) encoding more semantic specificity per token.

---

## 3. Head Unfrozen Ablation

Freeze the lexical projection head for the first 5k steps (warmup), then continue training at `head_lr_scale=0.1`. Compared to fully frozen head.

| Run | Model | Layers | Frozen BEIR | Unfrozen BEIR | Delta |
|---|---|---|---:|---:|---:|
| SPLADE spaced3 | SPLADE v3 | `[0,6,11]` | 0.3509 | 0.3431 | −0.0078 |
| Lion first3 | Lion | `[0,1,2]` | 0.3117 | 0.2966 | −0.0151 |
| Lion spaced4 | Lion | `[0,5,10,15]` | 0.3163 | n/a | — |

Unfreezing the head hurts in both cases. Freezing the head after warmup is the correct choice.

---

## 4. Factorized Projection Ablation

Replace the full lexical head (hidden_dim → vocab_size) with a low-rank factorized matrix (hidden_dim → d → vocab_size), initialized via SVD of the original weight. Head frozen after warmup.

Baseline (full-rank, frozen): SPLADE first3 = 0.3483, Lion first3 = 0.3117.

| Run | Factor dim | BEIR NDCG@10 | BEIR MRR@10 | Delta vs. full-rank |
|---|---:|---:|---:|---:|
| **SPLADE first3** | | | | |
| d=64 | 64 | 0.3379 | 0.3893 | −0.0104 |
| d=128 | 128 | 0.3433 | 0.3940 | −0.0050 |
| d=256 | 256 | 0.3454 | 0.3992 | −0.0029 |
| d=512 | 512 | 0.3428 | 0.3979 | −0.0055 |
| **Lion first3** | | | | |
| d=256 | 256 | 0.2722 | 0.3130 | −0.0395 |
| d=512 | 512 | 0.2889 | 0.3334 | −0.0228 |
| d=1024 | 1024 | 0.2799 | 0.3226 | −0.0318 |
| d=2048 | 2048 | 0.3076 | 0.3547 | −0.0041 |

SPLADE: factorization is nearly lossless at d=256 (−0.003). Lion: much larger penalty at all factor dims; even d=2048 loses −0.004. The asymmetry is expected — Lion's 128K-vocab head has far more singular vectors needed to represent the active token set.
