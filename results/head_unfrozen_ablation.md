# Head Unfrozen Shallow Ablation

Updated: 2026-05-18 13:51:12

## Setup

- Selection basis: best prior frozen-head BEIR small average for SPLADE, Lion 3-layer, Lion 4-layer, and Lion 5-layer.
- Factorized embeddings: `false`
- Warmup behavior: embedding/head train during the first 5k steps, then continue training at `head_lr_scale`.
- `freeze_head_after_warmup`: `false`
- `head_lr_scale`: `0.1` unless overridden in the base config
- Lion microbatch: `batch_size: 1`, `gradient_accumulation_steps: 32` to fit the full lexical matrix on the available GPU while preserving the previous effective batch size.
- Lion gradient checkpointing: `true` for the full-head ablation to reduce activation memory.
- BEIR datasets: `nfcorpus, scifact, arguana, scidocs, fiqa`

## Summary

| Run | Stage | Layers | Best NanoMSMARCO | Frozen BEIR avg | Unfrozen BEIR avg | Delta | Output dir |
|---|---|---:|---:|---:|---:|---:|---|
| splade_spaced3_head_unfrozen | `splade_shallow` | `[0, 6, 11]` | 0.7081 @ step 10000 | 0.3509/0.4067 | 0.3431/0.3986 | -0.0078 | `splade_shallow_spaced3_head_unfrozen` |
| lion_first3_head_unfrozen | `lion_shallow` | `[0, 1, 2]` | 0.6330 @ step 50000 | 0.3117/0.3581 | 0.2966/0.3417 | -0.0151 | `lion_shallow_first3_head_unfrozen` |
| lion_spaced4_head_unfrozen | `lion_shallow` | `[0, 5, 10, 15]` | n/a | 0.3163/0.3611 | n/a | n/a | `lion_shallow_spaced4_head_unfrozen` |

## Details

### splade_spaced3_head_unfrozen

- Description: Best prior SPLADE shallow config by BEIR average: spaced 3 layers
- Stage: `splade_shallow`
- Layers: `[0, 6, 11]`
- Best NanoMSMARCO NDCG@10: `0.7081 @ step 10000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_splade-v3/splade_shallow_spaced3_head_unfrozen/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/head_unfrozen_ablation/splade_spaced3_head_unfrozen_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/head_unfrozen_ablation/splade_spaced3_head_unfrozen_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.3259 | 0.5490 |
| scifact | 0.6200 | 0.5908 |
| arguana | 0.3126 | 0.2111 |
| scidocs | 0.1495 | 0.2610 |
| fiqa | 0.3073 | 0.3809 |
| Average | 0.3431 | 0.3986 |

### lion_first3_head_unfrozen

- Description: Best prior Lion 3-layer config by BEIR average: first 3 layers
- Stage: `lion_shallow`
- Layers: `[0, 1, 2]`
- Best NanoMSMARCO NDCG@10: `0.6330 @ step 50000`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_first3_head_unfrozen/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/head_unfrozen_ablation/lion_first3_head_unfrozen_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/head_unfrozen_ablation/lion_first3_head_unfrozen_beir.log`

| Dataset | NDCG@10 | MRR@10 |
|---|---:|---:|
| nfcorpus | 0.2725 | 0.4579 |
| scifact | 0.5408 | 0.5089 |
| arguana | 0.2862 | 0.1930 |
| scidocs | 0.1270 | 0.2316 |
| fiqa | 0.2564 | 0.3170 |
| Average | 0.2966 | 0.3417 |

### lion_spaced4_head_unfrozen

- Description: Best prior Lion 4-layer config by BEIR average: spaced 4 layers
- Stage: `lion_shallow`
- Layers: `[0, 5, 10, 15]`
- Best NanoMSMARCO NDCG@10: `n/a`
- Checkpoint: `/home/capemox/projects/sae-smo-splade/checkpoints_lion_shallow/lion_shallow_spaced4_head_unfrozen/best_NanoMSMARCO.pt`
- Train log: `/home/capemox/projects/sae-smo-splade/runs/head_unfrozen_ablation/lion_spaced4_head_unfrozen_train.log`
- BEIR log: `/home/capemox/projects/sae-smo-splade/runs/head_unfrozen_ablation/lion_spaced4_head_unfrozen_beir.log`

_No BEIR metrics recorded yet._

