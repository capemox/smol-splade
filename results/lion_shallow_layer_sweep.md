# Lion Shallow Layer Sweep

Updated: 2026-06-02 08:33:48

## Setup

- Stage: `lion_shallow`
- Factorized embeddings: `false`
- Embedding/head matrix frozen after warmup: `freeze_head_after_warmup: true`
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Variants

| Run | Layers | Best NanoMSMARCO | Output dir | Best checkpoint |
|---|---:|---:|---|---|
| first1 | `[0]` | 0.6748 @ step 40000 | `lion_shallow_first1` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first1/best_NanoMSMARCO.pt` |
| spaced1 | `[8]` | 0.6576 @ step 30000 | `lion_shallow_spaced1` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_spaced1/best_NanoMSMARCO.pt` |
| last1 | `[15]` | 0.6328 @ step 50000 | `lion_shallow_last1` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_last1/best_NanoMSMARCO.pt` |
| first2 | `[0, 1]` | 0.6838 @ step 50000 | `lion_shallow_first2` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first2/best_NanoMSMARCO.pt` |
| spaced2 | `[0, 15]` | 0.6562 @ step 30000 | `lion_shallow_spaced2` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_spaced2/best_NanoMSMARCO.pt` |
| last2 | `[14, 15]` | 0.6954 @ step 40000 | `lion_shallow_last2` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_last2/best_NanoMSMARCO.pt` |
| first3 | `[0, 1, 2]` | 0.6587 @ step 40000 _(loaded from prior run)_ | `lion_shallow_first3` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first3/best_NanoMSMARCO.pt` |
| spaced3 | `[0, 8, 15]` | 0.6888 @ step 30000 _(loaded from prior run)_ | `lion_shallow_spaced3` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_spaced3/best_NanoMSMARCO.pt` |
| last3 | `[13, 14, 15]` | 0.6859 @ step 50000 _(loaded from prior run)_ | `lion_shallow_last3` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_last3/best_NanoMSMARCO.pt` |
| first4 | `[0, 1, 2, 3]` | 0.6796 @ step 40000 _(loaded from prior run)_ | `lion_shallow_first4` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first4/best_NanoMSMARCO.pt` |
| spaced4 | `[0, 5, 10, 15]` | 0.6895 @ step 50000 _(loaded from prior run)_ | `lion_shallow_spaced4` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_spaced4/best_NanoMSMARCO.pt` |
| last4 | `[12, 13, 14, 15]` | 0.6780 @ step 50000 _(loaded from prior run)_ | `lion_shallow_last4` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_last4/best_NanoMSMARCO.pt` |
| first5 | `[0, 1, 2, 3, 4]` | 0.6751 @ step 40000 _(loaded from prior run)_ | `lion_shallow_first5` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first5/best_NanoMSMARCO.pt` |
| spaced5 | `[0, 4, 8, 11, 15]` | 0.6920 @ step 50000 _(loaded from prior run)_ | `lion_shallow_spaced5` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_spaced5/best_NanoMSMARCO.pt` |
| last5 | `[11, 12, 13, 14, 15]` | 0.6797 @ step 40000 _(loaded from prior run)_ | `lion_shallow_last5` | `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_last5/best_NanoMSMARCO.pt` |

## BEIR Results

### first1

- Description: First 1 layer
- Layers: `[0]`
- Best NanoMSMARCO NDCG@10: `0.6748 @ step 40000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first1_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first1_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2851 | 0.4692 |
| scifact | 0.5647 | 0.5352 |
| arguana | 0.2953 | 0.2016 |
| scidocs | 0.1292 | 0.2331 |
| fiqa | 0.2606 | 0.3244 |
| Average | 0.3070 | 0.3527 |

### spaced1

- Description: Middle layer
- Layers: `[8]`
- Best NanoMSMARCO NDCG@10: `0.6576 @ step 30000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced1_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced1_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2668 | 0.4475 |
| scifact | 0.5343 | 0.5070 |
| arguana | 0.2787 | 0.1877 |
| scidocs | 0.1169 | 0.2141 |
| fiqa | 0.2383 | 0.2925 |
| Average | 0.2870 | 0.3298 |

### last1

- Description: Last 1 layer
- Layers: `[15]`
- Best NanoMSMARCO NDCG@10: `0.6328 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last1_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last1_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2760 | 0.4578 |
| scifact | 0.5290 | 0.5023 |
| arguana | 0.2761 | 0.1851 |
| scidocs | 0.1176 | 0.2142 |
| fiqa | 0.2420 | 0.3004 |
| Average | 0.2881 | 0.3320 |

### first2

- Description: First 2 layers
- Layers: `[0, 1]`
- Best NanoMSMARCO NDCG@10: `0.6838 @ step 50000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first2_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first2_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2876 | 0.4656 |
| scifact | 0.5576 | 0.5183 |
| arguana | 0.3029 | 0.2055 |
| scidocs | 0.1288 | 0.2323 |
| fiqa | 0.2690 | 0.3323 |
| Average | 0.3091 | 0.3508 |

### spaced2

- Description: Equally spaced 2 layers
- Layers: `[0, 15]`
- Best NanoMSMARCO NDCG@10: `0.6562 @ step 30000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced2_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced2_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2902 | 0.4761 |
| scifact | 0.5601 | 0.5290 |
| arguana | 0.2846 | 0.1905 |
| scidocs | 0.1233 | 0.2250 |
| fiqa | 0.2572 | 0.3189 |
| Average | 0.3031 | 0.3479 |

### last2

- Description: Last 2 layers
- Layers: `[14, 15]`
- Best NanoMSMARCO NDCG@10: `0.6954 @ step 40000`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last2_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last2_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2834 | 0.4662 |
| scifact | 0.5537 | 0.5213 |
| arguana | 0.2735 | 0.1830 |
| scidocs | 0.1210 | 0.2207 |
| fiqa | 0.2587 | 0.3170 |
| Average | 0.2981 | 0.3416 |

### first3

- Description: First 3 layers
- Layers: `[0, 1, 2]`
- Best NanoMSMARCO NDCG@10: `0.6587 @ step 40000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first3_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first3_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2929 | 0.4888 |
| scifact | 0.5710 | 0.5333 |
| arguana | 0.2948 | 0.1996 |
| scidocs | 0.1310 | 0.2364 |
| fiqa | 0.2690 | 0.3325 |
| Average | 0.3117 | 0.3581 |

### spaced3

- Description: Equally spaced 3 layers
- Layers: `[0, 8, 15]`
- Best NanoMSMARCO NDCG@10: `0.6888 @ step 30000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced3_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced3_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2881 | 0.4807 |
| scifact | 0.5646 | 0.5319 |
| arguana | 0.3004 | 0.2026 |
| scidocs | 0.1266 | 0.2282 |
| fiqa | 0.2623 | 0.3252 |
| Average | 0.3084 | 0.3537 |

### last3

- Description: Last 3 layers
- Layers: `[13, 14, 15]`
- Best NanoMSMARCO NDCG@10: `0.6859 @ step 50000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last3_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last3_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2863 | 0.4788 |
| scifact | 0.5564 | 0.5323 |
| arguana | 0.2938 | 0.1977 |
| scidocs | 0.1295 | 0.2381 |
| fiqa | 0.2681 | 0.3322 |
| Average | 0.3068 | 0.3558 |

### first4

- Description: First 4 layers
- Layers: `[0, 1, 2, 3]`
- Best NanoMSMARCO NDCG@10: `0.6796 @ step 40000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first4_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first4_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2948 | 0.4915 |
| scifact | 0.5715 | 0.5386 |
| arguana | 0.2956 | 0.1996 |
| scidocs | 0.1311 | 0.2355 |
| fiqa | 0.2700 | 0.3316 |
| Average | 0.3126 | 0.3593 |

### spaced4

- Description: Equally spaced 4 layers
- Layers: `[0, 5, 10, 15]`
- Best NanoMSMARCO NDCG@10: `0.6895 @ step 50000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced4_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced4_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2918 | 0.4822 |
| scifact | 0.5794 | 0.5428 |
| arguana | 0.3053 | 0.2069 |
| scidocs | 0.1311 | 0.2354 |
| fiqa | 0.2737 | 0.3385 |
| Average | 0.3163 | 0.3611 |

### last4

- Description: Last 4 layers
- Layers: `[12, 13, 14, 15]`
- Best NanoMSMARCO NDCG@10: `0.6780 @ step 50000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last4_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last4_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2899 | 0.4896 |
| scifact | 0.5515 | 0.5213 |
| arguana | 0.2984 | 0.1998 |
| scidocs | 0.1268 | 0.2284 |
| fiqa | 0.2682 | 0.3298 |
| Average | 0.3070 | 0.3538 |

### first5

- Description: First 5 layers
- Layers: `[0, 1, 2, 3, 4]`
- Best NanoMSMARCO NDCG@10: `0.6751 @ step 40000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first5_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/first5_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2954 | 0.4887 |
| scifact | 0.5700 | 0.5367 |
| arguana | 0.2983 | 0.2010 |
| scidocs | 0.1283 | 0.2287 |
| fiqa | 0.2703 | 0.3331 |
| Average | 0.3125 | 0.3576 |

### spaced5

- Description: Equally spaced 5 layers
- Layers: `[0, 4, 8, 11, 15]`
- Best NanoMSMARCO NDCG@10: `0.6920 @ step 50000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced5_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/spaced5_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2921 | 0.4798 |
| scifact | 0.5743 | 0.5435 |
| arguana | 0.3001 | 0.2038 |
| scidocs | 0.1318 | 0.2355 |
| fiqa | 0.2720 | 0.3347 |
| Average | 0.3141 | 0.3595 |

### last5

- Description: Last 5 layers
- Layers: `[11, 12, 13, 14, 15]`
- Best NanoMSMARCO NDCG@10: `0.6797 @ step 40000` _(loaded from prior run)_
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last5_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_shallow_layer_sweep/last5_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2939 | 0.4892 |
| scifact | 0.5496 | 0.5177 |
| arguana | 0.2878 | 0.1937 |
| scidocs | 0.1267 | 0.2294 |
| fiqa | 0.2632 | 0.3239 |
| Average | 0.3042 | 0.3508 |

