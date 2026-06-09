# Lion Static Loss Ablation

Updated: 2026-06-07 02:48:44

## Setup

- Architecture: static (0-layer), one learnable weight per Lion vocabulary token
- Model: `hzeng/Lion-SP-1B-llama3-marco-mntp` (frozen doc encoder as teacher)
- Training steps: 10,000
- ColBERT score supervision: `data/colbertv2_msmarco_64way.json` for kd/margin_mse
- mse baseline BEIR avg NDCG@10/MRR@10: pending
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Summary

| Variant | Loss | Best NanoMSMARCO | BEIR avg NDCG@10/MRR@10 | Delta NDCG | Output dir |
|---|---|---:|---:|---:|---|
| kd | `kd` | 0.6167 @ step 2000 | 0.3009/0.3520 | n/a | `lion_static_loss_kd` |
| margin_mse | `margin_mse` | 0.6257 @ step 1000 | 0.3033/0.3534 | n/a | `lion_static_loss_margin_mse` |

## Details

### kd

- Description: KL distillation over nway passage scores from ColBERT supervision
- Loss: `kd`
- Best NanoMSMARCO NDCG@10: `0.6167 @ step 2000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_static/lion_static_loss_kd/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_static_loss_ablation/kd_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_static_loss_ablation/kd_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2822 | 0.4897 |
| scifact | 0.5965 | 0.5656 |
| arguana | 0.2654 | 0.1774 |
| scidocs | 0.1366 | 0.2454 |
| fiqa | 0.2241 | 0.2819 |
| Average | 0.3009 | 0.3520 |

### margin_mse

- Description: MarginMSE on positive-negative passage score margins from ColBERT
- Loss: `margin_mse`
- Best NanoMSMARCO NDCG@10: `0.6257 @ step 1000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_static/lion_static_loss_margin_mse/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/lion_static_loss_ablation/margin_mse_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/lion_static_loss_ablation/margin_mse_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2840 | 0.4884 |
| scifact | 0.5979 | 0.5668 |
| arguana | 0.2696 | 0.1796 |
| scidocs | 0.1373 | 0.2464 |
| fiqa | 0.2277 | 0.2856 |
| Average | 0.3033 | 0.3534 |

