# Static SPLADE Loss Ablation

Updated: 2026-06-05

## Setup

- Architecture: static (0-layer), one learnable weight per vocabulary token (30,522 params total)
- Model: `naver/splade-v3` (frozen doc encoder as teacher)
- mse/cosine training steps: 10,000 (cosine crashed at step 3,600; best checkpoint at step 3,000)
- kd training steps: 3,000 (best checkpoint at step 1,000; killed early due to ~15 min/1k step overhead)
- margin_mse training steps: 2,000 (best checkpoint at step 500)
- ColBERT score supervision: `data/colbertv2_msmarco_64way.json` for kd/margin_mse
- mse baseline BEIR avg NDCG@10/MRR@10: `0.3326/0.3861`
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Summary

| Variant | Loss | Best NanoMSMARCO | BEIR avg NDCG@10/MRR@10 | Delta NDCG | Output dir |
|---|---|---:|---:|---:|---|
| mse | `mse` | n/a | 0.3326/0.3861 | +0.0000 | `splade_static_loss_mse` |
| cosine | `cosine` | 0.6427 @ step 3000 | 0.3378/0.3929 | +0.0052 | `splade_static_loss_cosine` |
| kd | `kd` | n/a (step 1000) | 0.3288/0.3817 | -0.0038 | `splade_static_loss_kd` |
| margin_mse | `margin_mse` | n/a (step 500) | 0.3248/0.3778 | -0.0078 | `splade_static_loss_margin_mse` |

## Key Finding

Unlike the 3-layer shallow model (where `MSE >> cosine`), for the static (0-layer) model **cosine slightly outperforms MSE**. The difference is small (+0.0052 NDCG), and cosine training was cut short at step 3,000 (best checkpoint), so it may not hold at 10k steps. However, the result is directionally consistent with the intuition that a static vocabulary-weight model has no hidden-state magnitude calibration to preserve, making direction alignment sufficient.

The finding that **ranking supervision hurts** (kd < mse, margin_mse < mse) replicates across both the 3-layer and static models, strengthening the case for direct vector distillation as the preferred objective.

## Details

### mse

- Description: Vector MSE against full SPLADE-v3 teacher query vectors (baseline)
- Loss: `mse`
- Best NanoMSMARCO NDCG@10: `n/a`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_static/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/mse_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/mse_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3160 | 0.5273 |
| scifact | 0.6231 | 0.5890 |
| arguana | 0.2910 | 0.1945 |
| scidocs | 0.1453 | 0.2662 |
| fiqa | 0.2877 | 0.3538 |
| Average | 0.3326 | 0.3861 |

### cosine

- Description: Cosine distance against full SPLADE-v3 teacher query vectors
- Loss: `cosine`
- Best NanoMSMARCO NDCG@10: `0.6427 @ step 3000` (process crashed at step 3,600)
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_static_loss_cosine/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/cosine_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/cosine_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3224 | 0.5389 |
| scifact | 0.6267 | 0.5950 |
| arguana | 0.2972 | 0.1994 |
| scidocs | 0.1487 | 0.2706 |
| fiqa | 0.2941 | 0.3605 |
| Average | 0.3378 | 0.3929 |

### kd

- Description: KL distillation over nway passage scores from ColBERT supervision
- Loss: `kd`
- Best NanoMSMARCO NDCG@10: `n/a` (best checkpoint at step 1,000; killed at step 3,000 due to overhead)
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_static_loss_kd/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/kd_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/kd_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3159 | 0.5289 |
| scifact | 0.6248 | 0.5894 |
| arguana | 0.2846 | 0.1901 |
| scidocs | 0.1442 | 0.2653 |
| fiqa | 0.2745 | 0.3345 |
| Average | 0.3288 | 0.3817 |

### margin_mse

- Description: MarginMSE on positive-negative passage score margins from ColBERT supervision
- Loss: `margin_mse`
- Best NanoMSMARCO NDCG@10: `n/a` (best checkpoint at step 500; only 2,000 steps run)
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_static_loss_margin_mse/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/margin_mse_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/static_loss_ablation/margin_mse_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3139 | 0.5280 |
| scifact | 0.6214 | 0.5897 |
| arguana | 0.2799 | 0.1868 |
| scidocs | 0.1420 | 0.2602 |
| fiqa | 0.2670 | 0.3244 |
| Average | 0.3248 | 0.3778 |
