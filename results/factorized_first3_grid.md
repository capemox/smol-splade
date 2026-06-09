# Factorized First3 Shallow Grid

Updated: 2026-05-18 22:59:51

## Setup

- SPLADE layers: `[0, 1, 2]`
- Lion layers: `[0, 1, 2]`
- Factorization init: `svd`
- Factorized embedding/head matrix frozen after warmup: `freeze_head_after_warmup: true`
- Training target: full document encoder SPLADE vectors; BEIR eval uses the pre-built document indexes when available.
- Frozen first3 baselines: SPLADE `0.3483/0.4050`, Lion `0.3117/0.3581`
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Summary

| Run | Stage | Factor dim | Best NanoMSMARCO | Frozen first3 BEIR avg | Factorized BEIR avg | Delta | Output dir |
|---|---|---:|---:|---:|---:|---:|---|
| splade_first3_factorized_d64 | `splade_shallow` | 64 | 0.6915 @ step 20000 | 0.3483/0.4050 | 0.3379/0.3893 | -0.0104 | `splade_shallow_first3_factorized_d64` |
| splade_first3_factorized_d128 | `splade_shallow` | 128 | 0.7066 @ step 50000 | 0.3483/0.4050 | 0.3433/0.3940 | -0.0050 | `splade_shallow_first3_factorized_d128` |
| splade_first3_factorized_d256 | `splade_shallow` | 256 | 0.6960 @ step 50000 | 0.3483/0.4050 | 0.3454/0.3992 | -0.0029 | `splade_shallow_first3_factorized_d256` |
| splade_first3_factorized_d512 | `splade_shallow` | 512 | 0.7055 @ step 20000 | 0.3483/0.4050 | 0.3428/0.3979 | -0.0055 | `splade_shallow_first3_factorized_d512` |
| lion_first3_factorized_d256 | `lion_shallow` | 256 | 0.6566 @ step 50000 | 0.3117/0.3581 | 0.2722/0.3130 | -0.0395 | `lion_shallow_first3_factorized_d256` |
| lion_first3_factorized_d512 | `lion_shallow` | 512 | 0.6718 @ step 50000 | 0.3117/0.3581 | 0.2889/0.3334 | -0.0228 | `lion_shallow_first3_factorized_d512` |
| lion_first3_factorized_d1024 | `lion_shallow` | 1024 | 0.6564 @ step 20000 | 0.3117/0.3581 | 0.2799/0.3226 | -0.0318 | `lion_shallow_first3_factorized_d1024` |
| lion_first3_factorized_d2048 | `lion_shallow` | 2048 | 0.6825 @ step 40000 | 0.3117/0.3581 | 0.3076/0.3547 | -0.0041 | `lion_shallow_first3_factorized_d2048` |

## Details

### splade_first3_factorized_d64

- Description: SPLADE v3 first 3 layers, SVD factorized lexical matrix dim 64
- Stage: `splade_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `64`
- Best NanoMSMARCO NDCG@10: `0.6915 @ step 20000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_factorized_d64/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d64_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d64_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3182 | 0.5290 |
| scifact | 0.6146 | 0.5820 |
| arguana | 0.3112 | 0.2101 |
| scidocs | 0.1415 | 0.2527 |
| fiqa | 0.3038 | 0.3725 |
| Average | 0.3379 | 0.3893 |

### splade_first3_factorized_d128

- Description: SPLADE v3 first 3 layers, SVD factorized lexical matrix dim 128
- Stage: `splade_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `128`
- Best NanoMSMARCO NDCG@10: `0.7066 @ step 50000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_factorized_d128/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d128_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d128_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3231 | 0.5325 |
| scifact | 0.6180 | 0.5857 |
| arguana | 0.3186 | 0.2157 |
| scidocs | 0.1515 | 0.2670 |
| fiqa | 0.3054 | 0.3692 |
| Average | 0.3433 | 0.3940 |

### splade_first3_factorized_d256

- Description: SPLADE v3 first 3 layers, SVD factorized lexical matrix dim 256
- Stage: `splade_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `256`
- Best NanoMSMARCO NDCG@10: `0.6960 @ step 50000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_factorized_d256/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d256_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d256_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3296 | 0.5402 |
| scifact | 0.6254 | 0.5969 |
| arguana | 0.3140 | 0.2112 |
| scidocs | 0.1506 | 0.2668 |
| fiqa | 0.3074 | 0.3808 |
| Average | 0.3454 | 0.3992 |

### splade_first3_factorized_d512

- Description: SPLADE v3 first 3 layers, SVD factorized lexical matrix dim 512
- Stage: `splade_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `512`
- Best NanoMSMARCO NDCG@10: `0.7055 @ step 20000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_factorized_d512/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d512_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/splade_first3_factorized_d512_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3280 | 0.5506 |
| scifact | 0.6148 | 0.5847 |
| arguana | 0.3163 | 0.2130 |
| scidocs | 0.1483 | 0.2627 |
| fiqa | 0.3068 | 0.3788 |
| Average | 0.3428 | 0.3979 |

### lion_first3_factorized_d256

- Description: Lion first 3 layers, SVD factorized lexical matrix dim 256
- Stage: `lion_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `256`
- Best NanoMSMARCO NDCG@10: `0.6566 @ step 50000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first3_factorized_d256/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d256_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d256_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2474 | 0.4063 |
| scifact | 0.4805 | 0.4581 |
| arguana | 0.2793 | 0.1866 |
| scidocs | 0.1150 | 0.2123 |
| fiqa | 0.2387 | 0.3018 |
| Average | 0.2722 | 0.3130 |

### lion_first3_factorized_d512

- Description: Lion first 3 layers, SVD factorized lexical matrix dim 512
- Stage: `lion_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `512`
- Best NanoMSMARCO NDCG@10: `0.6718 @ step 50000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first3_factorized_d512/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d512_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d512_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2682 | 0.4435 |
| scifact | 0.5165 | 0.4918 |
| arguana | 0.2840 | 0.1911 |
| scidocs | 0.1230 | 0.2235 |
| fiqa | 0.2529 | 0.3172 |
| Average | 0.2889 | 0.3334 |

### lion_first3_factorized_d1024

- Description: Lion first 3 layers, SVD factorized lexical matrix dim 1024
- Stage: `lion_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `1024`
- Best NanoMSMARCO NDCG@10: `0.6564 @ step 20000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first3_factorized_d1024/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d1024_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d1024_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2620 | 0.4373 |
| scifact | 0.5062 | 0.4819 |
| arguana | 0.2723 | 0.1820 |
| scidocs | 0.1132 | 0.2063 |
| fiqa | 0.2460 | 0.3056 |
| Average | 0.2799 | 0.3226 |

### lion_first3_factorized_d2048

- Description: Lion first 3 layers, SVD factorized lexical matrix dim 2048
- Stage: `lion_shallow`
- Layers: `[0, 1, 2]`
- Factor dim: `2048`
- Best NanoMSMARCO NDCG@10: `0.6825 @ step 40000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first3_factorized_d2048/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d2048_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/factorized_first3_grid/lion_first3_factorized_d2048_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2907 | 0.4857 |
| scifact | 0.5587 | 0.5287 |
| arguana | 0.2975 | 0.2017 |
| scidocs | 0.1261 | 0.2292 |
| fiqa | 0.2648 | 0.3284 |
| Average | 0.3076 | 0.3547 |

