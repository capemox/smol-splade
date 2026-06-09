# SPLADE Shallow Layer Sweep

Updated: 2026-06-02 04:19:44

## Setup

- Stage: `splade_shallow`
- Factorized embeddings: `false`
- Embedding/head matrix frozen after warmup: `freeze_head_after_warmup: true`
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Variants

| Run | Layers | Best NanoMSMARCO | Output dir | Best checkpoint |
|---|---:|---:|---|---|
| first1 | `[0]` | 0.6973 @ step 30000 | `splade_shallow_first1` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first1/best_NanoMSMARCO.pt` |
| spaced1 | `[6]` | 0.6951 @ step 50000 | `splade_shallow_spaced1` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_spaced1/best_NanoMSMARCO.pt` |
| last1 | `[11]` | 0.7031 @ step 40000 | `splade_shallow_last1` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_last1/best_NanoMSMARCO.pt` |
| first2 | `[0, 1]` | 0.7083 @ step 30000 | `splade_shallow_first2` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first2/best_NanoMSMARCO.pt` |
| spaced2 | `[0, 11]` | 0.7220 @ step 50000 | `splade_shallow_spaced2` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_spaced2/best_NanoMSMARCO.pt` |
| last2 | `[10, 11]` | 0.6947 @ step 10000 | `splade_shallow_last2` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_last2/best_NanoMSMARCO.pt` |
| first3 | `[0, 1, 2]` | 0.7139 @ step 30000 | `splade_shallow_first3` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3/best_NanoMSMARCO.pt` |
| spaced3 | `[0, 6, 11]` | 0.7070 @ step 50000 | `splade_shallow_spaced3` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_spaced3/best_NanoMSMARCO.pt` |
| last3 | `[9, 10, 11]` | 0.6966 @ step 50000 | `splade_shallow_last3` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_last3/best_NanoMSMARCO.pt` |
| first4 | `[0, 1, 2, 3]` | 0.7089 @ step 20000 | `splade_shallow_first4` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first4/best_NanoMSMARCO.pt` |
| spaced4 | `[0, 4, 7, 11]` | 0.7061 @ step 50000 | `splade_shallow_spaced4` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_spaced4/best_NanoMSMARCO.pt` |
| last4 | `[8, 9, 10, 11]` | 0.7194 @ step 20000 | `splade_shallow_last4` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_last4/best_NanoMSMARCO.pt` |
| first5 | `[0, 1, 2, 3, 4]` | 0.7083 @ step 50000 | `splade_shallow_first5` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first5/best_NanoMSMARCO.pt` |
| spaced5 | `[0, 3, 6, 8, 11]` | 0.6954 @ step 20000 | `splade_shallow_spaced5` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_spaced5/best_NanoMSMARCO.pt` |
| last5 | `[7, 8, 9, 10, 11]` | 0.7023 @ step 30000 | `splade_shallow_last5` | `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_last5/best_NanoMSMARCO.pt` |

## BEIR Results

### first1

- Description: First 1 layer
- Layers: `[0]`
- Best NanoMSMARCO NDCG@10: `0.6973 @ step 30000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first1_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first1_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3260 | 0.5430 |
| scifact | 0.6254 | 0.5975 |
| arguana | 0.3183 | 0.2149 |
| scidocs | 0.1468 | 0.2578 |
| fiqa | 0.3052 | 0.3766 |
| Average | 0.3444 | 0.3980 |

### spaced1

- Description: Middle layer
- Layers: `[6]`
- Best NanoMSMARCO NDCG@10: `0.6951 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced1_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced1_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3306 | 0.5508 |
| scifact | 0.6160 | 0.5919 |
| arguana | 0.3166 | 0.2133 |
| scidocs | 0.1470 | 0.2598 |
| fiqa | 0.3061 | 0.3756 |
| Average | 0.3433 | 0.3983 |

### last1

- Description: Last 1 layer
- Layers: `[11]`
- Best NanoMSMARCO NDCG@10: `0.7031 @ step 40000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last1_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last1_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3248 | 0.5361 |
| scifact | 0.6271 | 0.6010 |
| arguana | 0.3164 | 0.2132 |
| scidocs | 0.1470 | 0.2602 |
| fiqa | 0.3078 | 0.3793 |
| Average | 0.3446 | 0.3980 |

### first2

- Description: First 2 layers
- Layers: `[0, 1]`
- Best NanoMSMARCO NDCG@10: `0.7083 @ step 30000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first2_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first2_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3286 | 0.5460 |
| scifact | 0.6216 | 0.5901 |
| arguana | 0.3202 | 0.2158 |
| scidocs | 0.1485 | 0.2598 |
| fiqa | 0.3135 | 0.3876 |
| Average | 0.3465 | 0.3999 |

### spaced2

- Description: Equally spaced 2 layers
- Layers: `[0, 11]`
- Best NanoMSMARCO NDCG@10: `0.7220 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced2_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced2_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3325 | 0.5526 |
| scifact | 0.6237 | 0.5932 |
| arguana | 0.3226 | 0.2183 |
| scidocs | 0.1507 | 0.2669 |
| fiqa | 0.3199 | 0.3931 |
| Average | 0.3499 | 0.4048 |

### last2

- Description: Last 2 layers
- Layers: `[10, 11]`
- Best NanoMSMARCO NDCG@10: `0.6947 @ step 10000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last2_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last2_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3275 | 0.5445 |
| scifact | 0.5999 | 0.5745 |
| arguana | 0.3106 | 0.2087 |
| scidocs | 0.1439 | 0.2526 |
| fiqa | 0.2974 | 0.3697 |
| Average | 0.3359 | 0.3900 |

### first3

- Description: First 3 layers
- Layers: `[0, 1, 2]`
- Best NanoMSMARCO NDCG@10: `0.7139 @ step 30000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first3_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first3_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3301 | 0.5530 |
| scifact | 0.6276 | 0.5979 |
| arguana | 0.3181 | 0.2152 |
| scidocs | 0.1498 | 0.2662 |
| fiqa | 0.3158 | 0.3927 |
| Average | 0.3483 | 0.4050 |

### spaced3

- Description: Equally spaced 3 layers
- Layers: `[0, 6, 11]`
- Best NanoMSMARCO NDCG@10: `0.7070 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced3_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced3_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3327 | 0.5514 |
| scifact | 0.6289 | 0.6012 |
| arguana | 0.3208 | 0.2157 |
| scidocs | 0.1539 | 0.2712 |
| fiqa | 0.3183 | 0.3938 |
| Average | 0.3509 | 0.4067 |

### last3

- Description: Last 3 layers
- Layers: `[9, 10, 11]`
- Best NanoMSMARCO NDCG@10: `0.6966 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last3_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last3_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3333 | 0.5530 |
| scifact | 0.6244 | 0.5956 |
| arguana | 0.3198 | 0.2148 |
| scidocs | 0.1505 | 0.2674 |
| fiqa | 0.3083 | 0.3786 |
| Average | 0.3473 | 0.4019 |

### first4

- Description: First 4 layers
- Layers: `[0, 1, 2, 3]`
- Best NanoMSMARCO NDCG@10: `0.7089 @ step 20000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first4_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first4_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3305 | 0.5532 |
| scifact | 0.6174 | 0.5890 |
| arguana | 0.3204 | 0.2163 |
| scidocs | 0.1494 | 0.2633 |
| fiqa | 0.3131 | 0.3853 |
| Average | 0.3462 | 0.4014 |

### spaced4

- Description: Equally spaced 4 layers
- Layers: `[0, 4, 7, 11]`
- Best NanoMSMARCO NDCG@10: `0.7061 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced4_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced4_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3295 | 0.5509 |
| scifact | 0.6361 | 0.6038 |
| arguana | 0.3213 | 0.2168 |
| scidocs | 0.1522 | 0.2668 |
| fiqa | 0.3172 | 0.3893 |
| Average | 0.3513 | 0.4056 |

### last4

- Description: Last 4 layers
- Layers: `[8, 9, 10, 11]`
- Best NanoMSMARCO NDCG@10: `0.7194 @ step 20000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last4_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last4_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3270 | 0.5407 |
| scifact | 0.6329 | 0.6002 |
| arguana | 0.3163 | 0.2119 |
| scidocs | 0.1471 | 0.2605 |
| fiqa | 0.3065 | 0.3768 |
| Average | 0.3460 | 0.3980 |

### first5

- Description: First 5 layers
- Layers: `[0, 1, 2, 3, 4]`
- Best NanoMSMARCO NDCG@10: `0.7083 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first5_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/first5_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3315 | 0.5525 |
| scifact | 0.6252 | 0.5985 |
| arguana | 0.3245 | 0.2191 |
| scidocs | 0.1495 | 0.2659 |
| fiqa | 0.3208 | 0.3970 |
| Average | 0.3503 | 0.4066 |

### spaced5

- Description: Equally spaced 5 layers
- Layers: `[0, 3, 6, 8, 11]`
- Best NanoMSMARCO NDCG@10: `0.6954 @ step 20000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced5_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/spaced5_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3258 | 0.5404 |
| scifact | 0.6273 | 0.5966 |
| arguana | 0.3189 | 0.2146 |
| scidocs | 0.1497 | 0.2650 |
| fiqa | 0.3117 | 0.3841 |
| Average | 0.3467 | 0.4002 |

### last5

- Description: Last 5 layers
- Layers: `[7, 8, 9, 10, 11]`
- Best NanoMSMARCO NDCG@10: `0.7023 @ step 30000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last5_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_shallow_layer_sweep/last5_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3355 | 0.5569 |
| scifact | 0.6230 | 0.5906 |
| arguana | 0.3128 | 0.2097 |
| scidocs | 0.1475 | 0.2598 |
| fiqa | 0.3112 | 0.3828 |
| Average | 0.3460 | 0.4000 |

