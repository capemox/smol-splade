# SPLADE First3 Loss Ablation

Updated: 2026-05-20 20:43:07

## Setup

- Stage: `splade_shallow`
- Model: `naver/splade-v3`
- Layers: `[0, 1, 2]`
- Factorized embeddings: `false`
- Embedding/head frozen after warmup: `freeze_head_after_warmup: true`
- ColBERT score supervision: `data/colbertv2_msmarco_64way.json` for KD and MarginMSE
- Non-factorized first3 baseline BEIR avg: `0.3483/0.4050`
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Summary

| Run | Loss | Best NanoMSMARCO | Baseline BEIR avg | Loss BEIR avg | Delta | Output dir |
|---|---|---:|---:|---:|---:|---|
| mse | `mse` | 0.7079 @ step 30000 | 0.3483/0.4050 | 0.3430/0.3953 | -0.0053 | `splade_shallow_first3_loss_mse` |
| cosine | `cosine` | 0.3375 @ step 10000 | 0.3483/0.4050 | 0.2116/0.2421 | -0.1367 | `splade_shallow_first3_loss_cosine` |
| kd_colbert | `kd` | 0.5834 @ step 30000 | 0.3483/0.4050 | 0.2661/0.3032 | -0.0822 | `splade_shallow_first3_loss_kd_colbert` |
| margin_mse_colbert | `margin_mse` | 0.6890 @ step 50000 | 0.3483/0.4050 | 0.3178/0.3675 | -0.0305 | `splade_shallow_first3_loss_margin_mse_colbert` |

## Details

### mse

- Description: Current vector MSE against full SPLADE query vectors
- Loss: `mse`
- Best NanoMSMARCO NDCG@10: `0.7079 @ step 30000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_loss_mse/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/mse_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/mse_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3278 | 0.5392 |
| scifact | 0.6100 | 0.5825 |
| arguana | 0.3199 | 0.2168 |
| scidocs | 0.1502 | 0.2653 |
| fiqa | 0.3070 | 0.3727 |
| Average | 0.3430 | 0.3953 |

### cosine

- Description: Cosine distance against full SPLADE query vectors
- Loss: `cosine`
- Best NanoMSMARCO NDCG@10: `0.3375 @ step 10000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_loss_cosine/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/cosine_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/cosine_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2118 | 0.3639 |
| scifact | 0.4725 | 0.4457 |
| arguana | 0.1894 | 0.1228 |
| scidocs | 0.0819 | 0.1555 |
| fiqa | 0.1022 | 0.1227 |
| Average | 0.2116 | 0.2421 |

### kd_colbert

- Description: KL distillation over nway passage scores from ColBERT supervision
- Loss: `kd`
- Best NanoMSMARCO NDCG@10: `0.5834 @ step 30000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_loss_kd_colbert/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/kd_colbert_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/kd_colbert_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2532 | 0.4175 |
| scifact | 0.5183 | 0.4827 |
| arguana | 0.2526 | 0.1661 |
| scidocs | 0.1143 | 0.2106 |
| fiqa | 0.1922 | 0.2393 |
| Average | 0.2661 | 0.3032 |

### margin_mse_colbert

- Description: MarginMSE on positive-negative passage score margins from ColBERT supervision
- Loss: `margin_mse`
- Best NanoMSMARCO NDCG@10: `0.6890 @ step 50000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_first3_loss_margin_mse_colbert/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/margin_mse_colbert_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/splade_first3_loss_ablation/margin_mse_colbert_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2939 | 0.4989 |
| scifact | 0.5786 | 0.5492 |
| arguana | 0.2963 | 0.1991 |
| scidocs | 0.1405 | 0.2507 |
| fiqa | 0.2798 | 0.3397 |
| Average | 0.3178 | 0.3675 |

