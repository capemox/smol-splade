# SPLADE v3 Shallow — BEIR Evaluation (All 13 Datasets)

Updated: 2026-06-09 14:32:25

- Model: `naver/splade-v3`
- Query encoders: spaced 1/3/5 layers (best_NanoMSMARCO checkpoint) + full doc encoder ceiling
- Doc indexes: pre-built with `naver/splade-v3` doc encoder, all 13 standard BEIR datasets
- Metric: NDCG@10 (primary), MRR@10

## NDCG@10 per Dataset

| Dataset | spaced1 | spaced3 | spaced5 | splade_v3_full |
|---|---:|---:|---:|---:|
| nfcorpus | 0.3306 | 0.3326 | 0.3258 | 0.3423 |
| scifact | 0.6160 | 0.6297 | 0.6273 | 0.6370 |
| arguana | 0.3166 | 0.3206 | 0.3189 | 0.3210 |
| scidocs | 0.1470 | 0.1541 | 0.1497 | 0.1575 |
| fiqa | 0.3061 | 0.3183 | 0.3117 | 0.3459 |
| trec-covid | 0.7062 | 0.7172 | 0.6912 | 0.7509 |
| webis-touche2020 | 0.2819 | 0.2901 | 0.2955 | 0.2961 |
| quora | 0.7630 | 0.7738 | 0.7758 | 0.8138 |
| nq | 0.5353 | 0.5474 | 0.5424 | 0.5766 |
| dbpedia-entity | 0.4116 | 0.4233 | 0.4216 | 0.4494 |
| hotpotqa | 0.6324 | 0.6501 | 0.6540 | 0.6830 |
| fever | 0.6811 | 0.6947 | 0.7043 | 0.7645 |
| climate-fever | 0.1655 | 0.1638 | 0.1714 | 0.2088 |
| **Average** | **0.4533** | **0.4627** | **0.4607** | **0.4882** |

## MRR@10 per Dataset

| Dataset | spaced1 | spaced3 | spaced5 | splade_v3_full |
|---|---:|---:|---:|---:|
| nfcorpus | 0.5508 | 0.5517 | 0.5404 | 0.5689 |
| scifact | 0.5919 | 0.6023 | 0.5966 | 0.6065 |
| arguana | 0.2133 | 0.2155 | 0.2146 | 0.2129 |
| scidocs | 0.2598 | 0.2719 | 0.2650 | 0.2777 |
| fiqa | 0.3756 | 0.3938 | 0.3841 | 0.4223 |
| trec-covid | 0.9085 | 0.9025 | 0.9035 | 0.9217 |
| webis-touche2020 | 0.5640 | 0.5458 | 0.6113 | 0.5504 |
| quora | 0.7419 | 0.7546 | 0.7580 | 0.7997 |
| nq | 0.4897 | 0.4999 | 0.4944 | 0.5308 |
| dbpedia-entity | 0.7090 | 0.7246 | 0.7291 | 0.7528 |
| hotpotqa | 0.8078 | 0.8262 | 0.8293 | 0.8638 |
| fever | 0.6726 | 0.6875 | 0.6969 | 0.7644 |
| climate-fever | 0.2306 | 0.2328 | 0.2359 | 0.2912 |
| **Average** | **0.5473** | **0.5545** | **0.5584** | **0.5818** |

## Variant Details

- **spaced1**: layers `[6]`, checkpoint: `checkpoints_splade-v3/splade_shallow_spaced1/best_NanoMSMARCO.pt`
- **spaced3**: layers `[0, 6, 11]`, checkpoint: `checkpoints_splade-v3/splade_shallow_spaced3/best_NanoMSMARCO.pt`
- **spaced5**: layers `[0, 3, 6, 8, 11]`, checkpoint: `checkpoints_splade-v3/splade_shallow_spaced5/best_NanoMSMARCO.pt`
- **splade_v3_full**: layers `all 12`, checkpoint: `full doc encoder (ceiling)`
