# SPLADE v3 Shallow Models — MS-MARCO Dev Evaluation

Updated: 2026-06-08 15:34:00

- Index: `data/msmarco_index/` (8,841,823 passages, `naver/splade-v3` doc encoder)
- Query models: `best_NanoMSMARCO.pt` checkpoint from each training run
- Metrics: NDCG@10 and MRR@10 on MS-MARCO dev (~6,980 queries)
- **Ceiling**: full `naver/splade-v3` (doc encoder on both queries and docs) — NDCG@10 0.4657, MRR@10 0.3989

## Layer Sweep

| Variant | NDCG@10 | % of full | MRR@10 | % of full |
|---|---:|---:|---:|---:|
| **splade-v3 (full)** | **0.4657** | **100.0%** | **0.3989** | **100.0%** |
| first1 | 0.4356 | 93.5% | 0.3716 | 93.2% |
| first2 | 0.4404 | 94.6% | 0.3755 | 94.1% |
| first3 | 0.4447 | 95.5% | 0.3795 | 95.1% |
| first4 | 0.4446 | 95.5% | 0.3790 | 95.0% |
| first5 | 0.4485 | 96.3% | 0.3840 | 96.3% |
| last1 | 0.4383 | 94.1% | 0.3744 | 93.9% |
| last2 | 0.4361 | 93.7% | 0.3720 | 93.3% |
| last3 | 0.4463 | 95.8% | 0.3824 | 95.9% |
| last4 | 0.4462 | 95.8% | 0.3815 | 95.6% |
| last5 | 0.4469 | 95.9% | 0.3831 | 96.0% |
| spaced1 | 0.4397 | 94.4% | 0.3760 | 94.3% |
| spaced2 | 0.4453 | 95.6% | 0.3809 | 95.5% |
| spaced3 | 0.4475 | 96.1% | 0.3830 | 96.0% |
| spaced4 | 0.4480 | 96.2% | 0.3829 | 96.0% |
| spaced5 | 0.4423 | 95.0% | 0.3772 | 94.6% |

## Loss Ablation (first3)

| Variant | NDCG@10 | % of full | MRR@10 | % of full |
|---|---:|---:|---:|---:|
| **splade-v3 (full)** | **0.4657** | **100.0%** | **0.3989** | **100.0%** |
| first3_loss_mse | 0.4455 | 95.7% | 0.3811 | 95.5% |
| first3_loss_cosine | 0.0174 | 3.7% | 0.0120 | 3.0% |
| first3_loss_kd_colbert | 0.3148 | 67.6% | 0.2651 | 66.5% |
| first3_loss_margin_mse_colbert | 0.3635 | 78.1% | 0.3067 | 76.9% |
