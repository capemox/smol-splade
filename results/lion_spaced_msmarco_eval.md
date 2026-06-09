# Lion Shallow Spaced Models — MS-MARCO Dev Evaluation

- Architecture: `lion_shallow`, MSE alignment loss, spaced layer selection
- Model: `hzeng/Lion-SP-1B-llama3-marco-mntp` (16 layers, LLaMA 3 1B)
- Corpus: 8,841,823-passage MS-MARCO (full, pre-built Lion index on Modal L40)
- Index: `/vol/indexes/msmarco_lion_index`

## MS-MARCO Dev Results

| Variant | Layers | NDCG@10 | % of full | MRR@10 | % of full | Best NanoMSMARCO |
|---|---|---:|---:|---:|---:|---:|
| **Lion (full)** | all 16 | **0.4758** | 100.0% | **0.4085** | 100.0% | — |
| spaced1 | `[0]` | 0.3650 | 76.7% | 0.3065 | 75.0% | 0.6576 @ step 30000 |
| spaced2 | `[0, 15]` | 0.3830 | 80.5% | 0.3230 | 79.1% | 0.6562 @ step 30000 |
| spaced3 | `[0, 8, 15]` | 0.3921 | 82.4% | 0.3310 | 81.0% | 0.6852 @ step 30000 |
| spaced4 | `[0, 5, 10, 15]` | 0.4049 | 85.1% | 0.3437 | 84.1% | 0.6864 @ step 30000 |
| spaced5 | `[0, 4, 8, 11, 15]` | 0.4117 | 86.5% | 0.3499 | 85.7% | 0.6816 @ step 40000 |
