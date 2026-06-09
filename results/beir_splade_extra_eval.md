# SPLADE v3 Shallow — BEIR Evaluation (spaced2, spaced4, first5)

Date: 2026-06-09

**Model:** `naver/splade-v3`  
**Doc encoder:** frozen `naver/splade-v3` (pre-built indexes, same for all variants)  
**Query encoders:** shallow variants trained via MSE distillation  

| Variant | Layers | Checkpoint |
|---|---|---|
| spaced2 | `[0, 11]` | `splade_shallow_spaced2/best_NanoMSMARCO.pt` |
| spaced4 | `[0, 4, 7, 11]` | `splade_shallow_spaced4/best_NanoMSMARCO.pt` |
| first5 | `[0, 1, 2, 3, 4]` | `splade_shallow_first5/best_NanoMSMARCO.pt` |

---

## NDCG@10

| Dataset | spaced2 | spaced4 | first5 |
|---|---:|---:|---:|
| nfcorpus | 0.3325 | 0.3295 | 0.3315 |
| scifact | 0.6237 | 0.6361 | 0.6252 |
| arguana | 0.3226 | 0.3213 | 0.3245 |
| scidocs | 0.1507 | 0.1522 | 0.1495 |
| fiqa | 0.3199 | 0.3172 | 0.3208 |
| trec-covid | 0.6996 | 0.7190 | 0.7138 |
| webis-touche2020 | 0.2844 | 0.2908 | 0.3068 |
| quora | 0.7779 | 0.7836 | 0.7822 |
| nq | 0.5472 | 0.5510 | 0.5493 |
| dbpedia-entity | 0.4251 | 0.4285 | 0.4252 |
| hotpotqa | 0.6491 | 0.6512 | 0.6472 |
| fever | 0.7012 | 0.7065 | 0.7010 |
| climate-fever | 0.1636 | 0.1675 | 0.1596 |
| **Average** | **0.4613** | **0.4657** | **0.4644** |

## MRR@10

| Dataset | spaced2 | spaced4 | first5 |
|---|---:|---:|---:|
| nfcorpus | 0.5526 | 0.5509 | 0.5525 |
| scifact | 0.5932 | 0.6038 | 0.5985 |
| arguana | 0.2183 | 0.2168 | 0.2191 |
| scidocs | 0.2669 | 0.2668 | 0.2659 |
| fiqa | 0.3931 | 0.3893 | 0.3970 |
| trec-covid | 0.9207 | 0.9029 | 0.9529 |
| webis-touche2020 | 0.5276 | 0.5381 | 0.5730 |
| quora | 0.7593 | 0.7656 | 0.7646 |
| nq | 0.5013 | 0.5048 | 0.5030 |
| dbpedia-entity | 0.7397 | 0.7361 | 0.7305 |
| hotpotqa | 0.8263 | 0.8286 | 0.8247 |
| fever | 0.6953 | 0.7013 | 0.6947 |
| climate-fever | 0.2280 | 0.2358 | 0.2227 |
| **Average** | **0.5556** | **0.5570** | **0.5615** |
