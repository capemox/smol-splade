# SAE-SMO-SPLADE

Sparse retrieval experiments for training cheap query encoders that stay
dot-product compatible with stronger frozen SPLADE-style document encoders.
The repo started from SAE-SPLADE on `jhu-clsp/ettin-encoder-17m`, then grew into
a set of asymmetric retrieval experiments: vocabulary transplant, direct/shared
vocabulary baselines, doc-head ablations, Lion-SP-1B variants, and the current
shallow-query alignment recipes.

The original motivation is serving cost. Standard SPLADE usually runs the same
encoder on queries and documents. Here, document vectors are produced offline by
a strong frozen document encoder, while a smaller query encoder is trained to
emit vectors in the same sparse vocabulary space.

## Current Status

The most stable contributor path is now:

- `splade_shallow_align`: query encoder is the first `n_layers` of
  `naver/splade-v3`, with the same tokenizer, embedding table, and MLM head as
  the frozen document encoder. This is the stable non-factorized baseline.
- `splade_shallow_factorized_spaced_align`: current strongest factorized setup.
  It uses ALBERT-style tied lexical factors, keeps those factors/head frozen,
  disables contrastive passage encoding for cheaper training, and composes the
  3-layer query body from doc layers 1, 7, and 12.
- `splade_shallow_factorized_align`: contiguous 3-layer factorized variant used
  for isolation runs around frozen vs. unfrozen lexical factors.
- `splade_shallow_align_distill`: the same shallow SPLADE architecture, but
  trained with cross-encoder MarginMSE distillation instead of the current
  teacher-vector MSE plus contrastive CE setup.
- `lion_shallow_align`: the same truncation recipe for
  `hzeng/Lion-SP-1B-llama3-marco-mntp`.
- `lion_shallow_factorized_spaced_align`: Lion analog of the factorized spaced
  SPLADE recipe, using Lion layers 1, 7, and the final layer. This experiment
  unfreezes the factorized lexical factors/head after warmup at a low LR.
- `lion_shallow_factorized_align`: same factorized Lion setup, but using the
  first 3 Lion layers contiguously.

These avoid the brittle cross-vocabulary transplant problem. The query side is
literally a shallow copy of the document model, so query and document vectors
remain in the same sparse vocabulary space without tokensurgeon, projection, or
adapter layers. The non-factorized baseline uses a two-phase schedule: freeze
embeddings/head while the retained body layers adapt, then unfreeze the
lightweight query side with a lower head/embedding LR. The best factorized run
instead keeps the SVD-initialized lexical factors/head frozen for the full run.

Current full-index comparison on MS MARCO dev:

| Configuration | NDCG@10 | MRR@10 | Notes |
|---|---:|---:|---|
| `naver/splade-v3` doc-only ceiling | 0.4657 | 0.3989 | Same SPLADE-v3 model encodes queries and docs. |
| `splade_shallow_factorized_spaced_align` | 0.4438 | 0.3787 | Factorized 3-layer `[0, 6, 11]` query with frozen lexical factors/head. |

The factorized spaced query keeps about 95% of the full SPLADE-v3 doc-only
quality on MS MARCO dev while substantially reducing query-side parameters.

The older vocabulary-transplant line is still important because it produced the
main research finding: tokensurgeon-style kNN embedding initialization is
load-bearing for cross-vocabulary asymmetric SPLADE. Random initialization,
shared vocabulary without transplant, and a frozen doc head all failed to match
the transplanted model.

## Setup

```bash
uv sync
```

Training and evaluation expect GPU access for the larger stages. The Lion
variants are memory-heavy because Lion emits `[batch, length, 128K]` logits.

## Quick Start

Run the current stable SPLADE shallow-query recipe:

```bash
uv run train.py splade_shallow_align --config config.yaml
```

Run the factorized shallow-query variant:

```bash
uv run train.py splade_shallow_factorized_align --config config.yaml
```

Run the spaced-layer factorized variant:

```bash
uv run train.py splade_shallow_factorized_spaced_align --config config.yaml
```

Run the Lion shallow-query recipe:

```bash
uv run train.py lion_shallow_align --config config.yaml
```

Run the Lion factorized spaced-layer variant:

```bash
uv run train.py lion_shallow_factorized_spaced_align --config config.yaml
```

Run the Lion factorized first-3-layer variant:

```bash
uv run train.py lion_shallow_factorized_align --config config.yaml
```

Run the cross-encoder MarginMSE distillation variant:

```bash
uv run train.py splade_shallow_align_distill --config config.yaml
```

Run the main vocab-transplant alignment experiment:

```bash
uv run train.py vocab_transplant_align --config config.yaml
```

Run the Lion vocab-transplant alignment experiment:

```bash
uv run train.py lion_transplant_align --config config.yaml
```

Resume from a checkpoint:

```bash
uv run train.py splade_shallow_align --config config.yaml --resume checkpoints_splade-v3/splade_shallow_align/align_step_10000.pt
```

Initialize a shallow SPLADE run from weights but reset the step counter:

```bash
uv run train.py splade_shallow_align --config config.yaml --init-from path/to/checkpoint.pt
```

Monitor training:

```bash
uv run tensorboard --logdir .
```

Then open `http://localhost:6006`.

## Training Stages

The CLI stages in `train.py` are:

| Stage | Purpose |
|---|---|
| `sae` | Pretrain a TopK SAE on backbone token embeddings from MS MARCO documents. |
| `splade` | Fine-tune a symmetric SAE-SPLADE dual encoder. |
| `asymmetric` | Train an SAE-SPLADE query encoder against a frozen `naver/splade-v3` doc encoder. |
| `projected` | Train a small backbone plus projection into the frozen doc SPLADE MLM head. |
| `vocab_transplant` | Tokensurgeon transplant plus joint CE/alignment/FLOPs training. |
| `vocab_transplant_align` | Tokensurgeon transplant plus cosine alignment only. |
| `random_init_align` | Ablation: donor vocab size with random embeddings instead of tokensurgeon. |
| `doc_head_align` | Ablation: small backbone plus frozen doc MLM head. |
| `direct_align` | Ablation: BERT-family query model already sharing DistilBERT/SPLADE vocab. |
| `lion_transplant_align` | Vocab transplant to Lion-SP-1B's 128K Llama-3 vocabulary. |
| `splade_shallow_align` | Stable shallow query encoder for `naver/splade-v3`. |
| `splade_shallow_factorized_align` | Shallow SPLADE query encoder with ALBERT-style tied lexical factors. |
| `splade_shallow_factorized_spaced_align` | Factorized shallow query using doc layers 1, 7, and 12. |
| `splade_shallow_align_distill` | Shallow SPLADE query encoder trained with cross-encoder MarginMSE distillation. |
| `lion_shallow_align` | Stable shallow query encoder for Lion-SP-1B. |
| `lion_shallow_factorized_align` | Factorized shallow Lion query using the first 3 layers. |
| `lion_shallow_factorized_spaced_align` | Factorized shallow Lion query using layers 1, 7, and the final layer. |

Checkpoints are written under `checkpoints_<model-name>/<stage>/`.

## Current Stable Recipes

### `splade_shallow_align`

The query encoder is made by truncating the full SPLADE document encoder to the
first `n_layers`, while keeping the same tokenizer, embedding table, and output
head. This gives a smaller query model whose outputs are still in exactly the
same vocabulary space as the document vectors.

The loss is teacher-vector MSE on SPLADE query vectors with an STE through ReLU,
query L1 sparsity, and an optional contrastive CE term over Tevatron positives
plus hard negatives. The contrastive term scores the shallow query against
frozen full-SPLADE passage vectors, adding a retrieval-margin signal while the
document encoder stays fixed.

Important config keys:

| Key | Current value | Meaning |
|---|---:|---|
| `splade_shallow_align.doc_splade_hf_id` | `naver/splade-v3` | Frozen document encoder. |
| `splade_shallow_align.n_layers` | `3` | Number of document-model layers kept for queries. |
| `splade_shallow_align.freeze_warmup_steps` | `5000` | Train retained body layers before unfreezing embeddings/head. |
| `splade_shallow_align.alignment_steps` | `50000` | Total shallow alignment steps. |
| `splade_shallow_align.contrastive_coeff` | `0.1` | Weight for positive-vs-hard-negative CE. |
| `splade_shallow_align.lambda_q` | `0.0001` | Query sparsity pressure. |

### `splade_shallow_factorized_align`

This is the contiguous-layer factorized version of `splade_shallow_align`. The
tied `[vocab, hidden]` lexical matrix is replaced by two shared factors
`[vocab, factorized_embedding_dim]` and `[factorized_embedding_dim, hidden]`.
The MLM head computes `hidden -> B.T -> A.T`, so output vectors remain in the
same SPLADE vocabulary. Current runs use teacher-vector MSE plus query sparsity
with contrastive training disabled for cheaper iteration. Unfreezing the
factorized lexical factors/head at very low LR works, but it has been weaker
than keeping the factors/head frozen throughout.

Important config keys:

| Key | Current value | Meaning |
|---|---:|---|
| `splade_shallow_factorized_align.factorized_embedding_dim` | `128` | Bottleneck/rank dimension for the tied input/output lexical matrix. |
| `splade_shallow_factorized_align.factorization_init` | `svd` | Initialize the factors from the original SPLADE-v3 tied matrix. |
| `splade_shallow_factorized_align.batch_size` | `16` | Microbatch size; with accumulation 2 gives effective batch 32. |
| `splade_shallow_factorized_align.use_contrastive` | `false` | Runs the cheaper teacher-query MSE recipe without training-time passage encodes. |
| `splade_shallow_factorized_align.log_ranking_metrics` | `false` | Skips p@1 logging passage encodes when contrastive is disabled. |
| `splade_shallow_factorized_align.freeze_warmup_steps` | `5000` | Same warmup as the proven shallow run. |
| `splade_shallow_factorized_align.head_lr_scale` | `0.01` | Very low LR for factorized lexical factors/head after unfreeze. |
| `splade_shallow_factorized_align.freeze_head_after_warmup` | `false` | Unfreeze shared factorized lexical factors after warmup; contrastive remains off for isolation. |
| `splade_shallow_factorized_align.lambda_q` | `0.0002` | Slightly stronger query sparsity pressure for the factorized head. |

### `splade_shallow_factorized_spaced_align`

This experiment keeps the factorized embedding/head setup but composes the
3-layer query body from the document encoder's first, seventh, and final BERT
layers (`layer_indices: [0, 6, 11]`). The factorized embedding/SPLADE head stays
frozen throughout and contrastive training is disabled, matching the strongest
factorized-head behavior seen so far.

This is the current strongest factorized SPLADE-v3-side checkpoint. The 50k-step
run reached about `0.6928` NanoMSMARCO NDCG@10 and `0.3333` NanoNFCorpus
NDCG@10. On full MS MARCO dev with the 8.8M-passage index, the final checkpoint
reached `0.4438` NDCG@10 and `0.3787` MRR@10, compared with `0.4657` and
`0.3989` for the SPLADE-v3 doc-only ceiling.

### `lion_shallow_align`

This applies the same shallow-query idea to
`hzeng/Lion-SP-1B-llama3-marco-mntp`. It is the preferred way to work with Lion
now because it avoids most transplant instability.

Important config keys:

| Key | Current value | Meaning |
|---|---:|---|
| `lion_shallow_align.lion_hf_id` | `hzeng/Lion-SP-1B-llama3-marco-mntp` | Frozen Lion document encoder. |
| `lion_shallow_align.n_layers` | `3` | Lion layers kept for queries. |
| `lion_shallow_align.batch_size` | `4` | Small because Lion logits are large. |
| `lion_shallow_align.gradient_accumulation_steps` | `8` | Effective batch size 32. |
| `lion_shallow_align.eval_batch_size` | `4` | Keep low to avoid eval OOM. |
| `lion_shallow_align.lambda_q` | `0.001` | Query sparsity pressure. |

### `lion_shallow_factorized_spaced_align`

This is the Lion-SP-1B analog of `splade_shallow_factorized_spaced_align`. The
query body uses Lion layers `[0, 6, -1]`, and the large Llama
`embed_tokens`/`lm_head` lexical matrix is replaced with shared ALBERT-style
factors. The default initializer is exact SVD (`svd`) over Lion's full lexical
matrix, matching the SPLADE-v3 factorized setup more closely than randomized
low-rank SVD.

Only the retained body layers train during warmup. After warmup, the final norm
and factorized lexical factors/head are also trainable, with the lexical factors
using `head_lr_scale: 0.01`.

### `lion_shallow_factorized_align`

This uses the same factorized Lion hyperparameters as
`lion_shallow_factorized_spaced_align`, but keeps the first 3 Lion layers
contiguously instead of selecting `[0, 6, -1]`.

## Experiment History

### 1. Symmetric SAE-SPLADE

The baseline trains a TopK Sparse Autoencoder over ettin token embeddings, then
fine-tunes an SAE-SPLADE dual encoder. Query and document vectors use the same
ettin + SAE pipeline, so compatibility is straightforward. This works, but it
does not reduce query serving cost as much as an asymmetric setup.

```bash
uv run train.py sae --config config.yaml
uv run train.py splade --config config.yaml
```

For asymmetric SAE pretraining, `model.sae_width` must match
`naver/splade-v3`'s vocabulary size, `30522`. The regular wider SAE setting
cannot be reused for that mode.

### 2. Asymmetric SAE-SPLADE

This trains only an ettin SAE query encoder against a frozen
`naver/splade-v3` document encoder. The SAE width must equal the doc encoder
vocabulary size. A k-annealing schedule starts dense (`k_init=256`) and decays
to the target `k=32`, because sparse doc-dot-product gradients are too weak at
low `k` early in training. It retrieves, but the SAE bottleneck limits quality.

### 3. Projected Query Encoder

This uses:

```text
small backbone -> linear projection -> frozen naver/splade-v3 MLM head -> SPLADE pooling
```

The projection bridges hidden-size differences and the frozen MLM head gives the
right vocabulary. The quality gap stayed large; the projection is a hard
representational bottleneck.

### 4. Vocabulary Transplant

This is the main cross-vocabulary experiment. The query model's embedding table
is resized to the donor document encoder's vocabulary and initialized with a
tokensurgeon-style transfer:

1. Exact-match tokens copy the query model's original embedding.
2. Donor-only tokens are approximated by kNN interpolation through shared tokens.
3. The query model is saved with the donor tokenizer and donor special-token IDs.

After transplant, SPLADE max pooling on the query model produces vectors in the
same vocabulary space as the frozen document encoder. No projection is needed.

For `naver/splade-v3`, ettin shares roughly 50-60% of tokens with the donor
vocabulary, so about 40% are interpolated. The alignment-only version reached
about `0.61` NanoMSMARCO NDCG@10 versus about `0.64` for the doc-doc ceiling.
Ranking fine-tuning did not improve over alignment-only.

### 5. Direct Alignment

`google/bert_uncased_L-4_H-512_A-8` already shares DistilBERT's vocabulary, so
no transplant is needed. A KD warmup followed by cosine alignment underperformed
the transplanted ettin model. This suggests that simply sharing vocabulary is
not enough.

### 6. Random-Init Alignment

This keeps the vocab-transplant architecture and donor vocabulary size but
randomly initializes the embedding table instead of using tokensurgeon. It
fails to reach vocab-transplant quality even with KD warmup. This is the
cleanest ablation showing that the interpolation initialization matters.

### 7. Doc-Head Alignment

This uses an ettin backbone, a projection to the SPLADE hidden size, and the
frozen `naver/splade-v3` MLM head. It also fails. A pretrained output head alone
does not give the backbone usable hidden states for the donor vocabulary.

### 8. Lion-SP-1B Vocab Transplant

`lion_transplant_align` repeats the transplant idea with
`hzeng/Lion-SP-1B-llama3-marco-mntp` as the frozen doc encoder and
`jhu-clsp/ettin-encoder-150m` as the query encoder. This is harder because Lion
uses a 128K Llama-3 BPE vocabulary and exact overlap with ettin is only
single-digit percent. About 95% of donor tokens are interpolated.

The working formulation uses cosine alignment plus a raw-logit softmax-KL
safety net and squared-mean FLOPs. The KL term keeps gradients alive when ReLU
would otherwise enter the zero-vector basin, while FLOPs anchors the logit
offset that KL alone leaves shift-invariant.

Best recorded 50k-step NanoBEIR results:

| Dataset | Step 40k | Step 50k | Lion-Lion ceiling | Gap |
|---|---:|---:|---:|---:|
| NanoMSMARCO | 0.6129 | 0.6410 | 0.6510 | -0.010 |
| NanoNFCorpus | 0.3105 | 0.3211 | 0.3480 | -0.027 |

These are promising but still NanoBEIR-only. Full MS MARCO dev evaluation is
needed before treating them as reliable.

## Important Findings

- The core publishable idea is vocabulary transplant plus asymmetric sparse
  retrieval: a small query MLM is retargeted to the frozen document SPLADE's
  vocabulary so query and document vectors can be dot-producted directly.
- Tokensurgeon-style kNN embedding interpolation is the critical ingredient in
  the transplant line. Correct vocabulary size alone is insufficient.
- KL self-distillation from `naver/splade-v3` on short queries is weak because
  SPLADE-v3 behaves like a document encoder; on short query texts it can produce
  near-zero sparse vectors and nearly uniform softmax targets.
- Ranking CE after cosine alignment has not improved over alignment-only in the
  tested transplant setups.
- Squared FLOPs is generally safer than L1 FLOPs for these experiments because
  its gradient scales with activation magnitude. L1 can kill small useful
  activations too aggressively.
- NanoBEIR is useful for iteration but noisy: each dataset has around 50
  queries, so full MS MARCO dev is required for credible claims.

## Known Failure Modes

- Pure cosine alignment can collapse to `q_nnz=0`. Once vectors are zero, ReLU
  blocks gradient and cosine loss stays near 1.0.
- Softmax-KL on logits is shift-invariant. Without a sparsity or magnitude
  anchor, logits can drift positive and make every vocabulary dimension active.
- L1 FLOPs can accelerate collapse by applying uniform pressure to all active
  dimensions.
- BCE-with-logits on Lion's sparse soft targets was unstable: no `pos_weight`
  drove everything inactive, while dynamic `pos_weight` made everything active.
- BERT-base to Lion transplant failed because exact string overlap between
  WordPiece and Llama-3 BPE is extremely low. The model could learn unigram-like
  behavior but not context-specific predictions from the interpolated embedding
  table.

Healthy Lion vocab-transplant logs after warmup should look like:

```text
[lion-align/cos] step N | loss <decreasing> | q_nnz 50-300 (lion 250-350) | ...
```

Bad signs:

```text
[lion-align/cos] step N | loss 1.0xxx | q_nnz 0.0 (lion ~300) | ...
[lion-align/cos] step N | loss ...    | q_nnz 128256 (lion ~300) | ...
```

The first is dead-ReLU collapse. The second is density runaway.

## Evaluation

NanoBEIR runs during training using the datasets in `config.yaml`:

- `zeta-alpha-ai/NanoMSMARCO`
- `zeta-alpha-ai/NanoNFCorpus`

Full MS MARCO dev evaluation:

```bash
uv run scripts/eval_msmarco.py \
  --stage splade_shallow_factorized_spaced_align \
  --checkpoint checkpoints_splade-v3/splade_shallow_factorized_spaced_align/align_final.pt \
  --config config.yaml \
  --index_dir data/msmarco_index
```

Doc-only upper bound:

```bash
uv run scripts/eval_msmarco.py \
  --stage splade_shallow_align \
  --doc_only \
  --config config.yaml \
  --index_dir data/msmarco_index
```

BEIR index/eval helpers:

```bash
uv run scripts/build_beir_index.py
uv run scripts/eval_beir.py
```

MS MARCO index helpers:

```bash
uv run scripts/build_msmarco_index.py
uv run scripts/eval_msmarco.py --help
```

## Operational Notes

- Lion eval can OOM after periodic evaluation because PyTorch's caching
  allocator holds memory from corpus encoding. The training code explicitly runs
  `gc.collect()` and `torch.cuda.empty_cache()` after Lion eval. If OOM persists,
  lower `eval_batch_size`.
- `lion_transplant_align.batch_size` is intentionally small and relies on
  gradient accumulation.
- Transplant artifacts are cached under each checkpoint root. If you change the
  transplant algorithm or donor/query pair, make sure the old cache is not being
  reused accidentally.
- `hf-transfer` is included in dependencies; enabling it can speed large
  Hugging Face downloads.

## Code Map

| Component | File / entry point |
|---|---|
| CLI dispatch and training loops | `train.py` |
| TopK SAE | `src/model.py::TopKSAE` |
| Symmetric SAE-SPLADE | `src/model.py::SAESPLADEModel` |
| Vocab-transplant query model | `src/model.py::VocabTransplantQuerySPLADE` |
| Frozen SPLADE doc encoder | `src/model.py::FrozenDocSPLADE` |
| Frozen Lion doc encoder | `src/model.py::FrozenLionSPLADE` |
| Shallow SPLADE query | `src/model.py::ShallowSpladeQuery` |
| Shallow Lion query | `src/model.py::ShallowLionQuery` |
| Vocab transplant builder | `train.py::_run_tokensurgeon` |
| Lion transplant builder | `train.py::_run_tokensurgeon_lion` |
| Joint vocab-transplant loss | `src/model.py::vocab_transplant_joint_loss` |
| NanoBEIR/asymmetric eval | `src/eval.py::evaluate_asymmetric` |
| Data loaders | `src/data.py` |
| Full MS MARCO eval | `scripts/eval_msmarco.py` |

## Open Questions

1. Run full MS MARCO dev for the older vocab-transplant and Lion checkpoints.
2. Test whether the shallow-query recipes preserve the Lion NanoBEIR gains on
   larger evaluations.
3. Investigate why ranking CE fails to improve over alignment-only after cosine
   alignment.
4. Improve cross-tokenizer transplant quality for low-overlap pairs, especially
   BERT WordPiece to Llama-3 BPE. A BPE/subword-decomposition transplant similar
   in spirit to WECHSEL is a plausible next step.
5. Try stronger public document encoders where available. The previous note was
   to test Lion-SP-8B; Lion-SP-1B has been tested, but 8B remains TODO.

## Research Context

Related work to check before making novelty claims:

- SPLADE-v2 and SPLADE-v3 / SPLADE-3
- Tokensurgeon / embedding transfer methods
- TILDE and TILDEv2
- Efficient SPLADE, CoCondenser, and other sparse retriever compression work
- uniCOIL, DeepImpact, SLIM, and LexMAE

The strongest current claim, if full evaluations hold, is that a
vocab-transplanted or shallow small query encoder can approach the symmetric
doc-doc SPLADE ceiling while using a substantially cheaper query-time model.
