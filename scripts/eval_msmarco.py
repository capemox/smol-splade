#!/usr/bin/env python
"""Full MSMARCO dev evaluation for VocabTransplant query models.

Streams the 8.8M-passage corpus in batches — no full corpus load into RAM.
Processes queries in sub-batches so only a small slice of query vecs lives
on GPU at any time. Reports NDCG@10 and MRR@10 on the standard dev set.

Usage:
    uv run scripts/eval_msmarco.py --checkpoint checkpoints/vocab_transplant/step_10000.pt
    uv run scripts/eval_msmarco.py --checkpoint checkpoints/vocab_transplant/align_step_10000.pt
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dev_queries_and_qrels():
    """Load MSMARCO dev queries and qrels from BeIR datasets on HuggingFace."""
    from datasets import load_dataset

    print("Loading MSMARCO dev qrels (BeIR/msmarco-qrels, validation) ...")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")
    qrels: dict = {}
    dev_qids: set = set()
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row.get("score", 1))
        qrels.setdefault(qid, {})[did] = score
        dev_qids.add(qid)
    print(f"  {len(qrels)} queries with relevance judgments")

    print("Loading MSMARCO queries (BeIR/msmarco, queries split) ...")
    query_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    query_ids, query_texts = [], []
    for row in query_ds:
        qid = str(row["_id"])
        if qid in dev_qids:
            query_ids.append(qid)
            query_texts.append(row["text"] or "")
    print(f"  {len(query_ids)} dev queries matched")

    return query_ids, query_texts, qrels


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_queries(query_model, tokenizer, query_texts, max_length, batch_size, device):
    """Encode all dev queries; returns fp16 CPU tensor [Q, vocab]."""
    all_vecs = []
    query_model.eval()
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i : i + batch_size]
            enc = tokenizer(
                batch, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
            )
            vecs = query_model.encode(
                enc["input_ids"].to(device), enc["attention_mask"].to(device)
            )
            all_vecs.append(vecs.cpu().half())
            print(f"  queries: {min(i+batch_size, len(query_texts))}/{len(query_texts)}", end="\r")
    print()
    result = torch.cat(all_vecs, dim=0)
    del all_vecs
    return result


def encode_queries_with_doc_splade(doc_splade, query_texts, max_length, batch_size):
    """Encode queries using the frozen doc SPLADE (ceiling / doc-only benchmark); returns fp16."""
    all_vecs = []
    for i in range(0, len(query_texts), batch_size):
        batch = query_texts[i : i + batch_size]
        vecs = doc_splade.encode(batch, max_length, no_grad=True)
        all_vecs.append(vecs.cpu().half())
        print(f"  queries (doc_splade): {min(i+batch_size, len(query_texts))}/{len(query_texts)}", end="\r")
    print()
    result = torch.cat(all_vecs, dim=0)
    del all_vecs
    return result


# ── Corpus subset ─────────────────────────────────────────────────────────────

def resolve_corpus_subset(
    max_corpus_size: int,
    corpus_dataset: str,
    text_field: str,
    qrels: dict,
) -> tuple:
    """Return (corpus_ids, corpus_texts) for a subset of at most max_corpus_size passages.

    Always includes every passage relevant to a dev query; the remainder is
    reservoir-sampled distractors. Streams from HuggingFace so peak RAM is
    O(max_corpus_size), not O(full corpus). Result is cached to
    data/msmarco_dev_subset_N.pkl so subsequent calls are instant.
    """
    import pickle
    import random
    from datasets import load_dataset

    cache = Path("data") / f"msmarco_dev_subset_{max_corpus_size}.pkl"
    if cache.exists():
        print(f"Loading corpus subset from cache {cache} ...")
        with cache.open("rb") as f:
            return pickle.load(f)

    required_ids = {did for rel in qrels.values() for did in rel}
    n_distractor_slots = max(0, max_corpus_size - len(required_ids))
    print(
        f"Streaming {corpus_dataset} to build {max_corpus_size:,}-passage subset "
        f"({len(required_ids):,} required + {n_distractor_slots:,} reservoir distractors) ..."
    )

    relevant: dict = {}   # docid -> text; always kept
    reservoir: list = []  # [(docid, text)]; reservoir-sampled distractors
    n_distractor_seen = 0

    ds = load_dataset(corpus_dataset, split="train", streaming=True)
    for item in ds:
        docid = str(item.get("docid", ""))
        text = item.get(text_field) or item.get("passage") or item.get("contents", "")
        if docid in required_ids:
            relevant[docid] = text
        else:
            n_distractor_seen += 1
            if len(reservoir) < n_distractor_slots:
                reservoir.append((docid, text))
            elif n_distractor_slots > 0:
                j = random.randrange(n_distractor_seen)
                if j < n_distractor_slots:
                    reservoir[j] = (docid, text)

    missing = len(required_ids) - len(relevant)
    if missing:
        print(f"  Warning: {missing:,} relevant doc IDs not found in corpus")

    combined = list(relevant.items()) + reservoir
    random.shuffle(combined)
    corpus_ids = [d for d, _ in combined]
    corpus_texts = [t for _, t in combined]

    print(f"  {len(relevant):,} relevant + {len(reservoir):,} distractors = {len(corpus_ids):,} total")
    Path("data").mkdir(exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump((corpus_ids, corpus_texts), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached → {cache}")
    return corpus_ids, corpus_texts


# ── Retrieval ─────────────────────────────────────────────────────────────────

def list_retrieve(query_vecs_cpu, doc_splade, corpus_ids, corpus_texts, doc_max_length, doc_batch_size, query_batch_size, topk, device):
    """Retrieve from an in-memory corpus list (subset mode)."""
    from tqdm import tqdm

    Q = len(query_vecs_cpu)
    top_scores = torch.full((Q, topk), float("-inf"))
    top_doc_idx = torch.zeros((Q, topk), dtype=torch.long)

    pbar = tqdm(total=len(corpus_texts), unit="doc", unit_scale=True, dynamic_ncols=True)

    for doc_start in range(0, len(corpus_texts), doc_batch_size):
        doc_end = min(doc_start + doc_batch_size, len(corpus_texts))
        buf = corpus_texts[doc_start:doc_end]
        B = len(buf)

        doc_vecs_cpu = doc_splade.encode(buf, doc_max_length, no_grad=True).cpu().half()

        for q_start in range(0, Q, query_batch_size):
            q_end = min(q_start + query_batch_size, Q)
            q_gpu = query_vecs_cpu[q_start:q_end].half().to(device)
            d_gpu = doc_vecs_cpu.to(device)
            batch_scores = (q_gpu @ d_gpu.T).cpu().float()
            del q_gpu, d_gpu

            combined = torch.cat([top_scores[q_start:q_end], batch_scores], dim=1)
            pos = torch.arange(doc_start, doc_end, dtype=torch.long)
            combined_idx = torch.cat([top_doc_idx[q_start:q_end], pos.unsqueeze(0).expand(q_end - q_start, -1)], dim=1)
            new_top = combined.topk(topk, dim=1)
            top_scores[q_start:q_end] = new_top.values
            top_doc_idx[q_start:q_end] = combined_idx.gather(1, new_top.indices)

        pbar.update(B)

    pbar.close()
    return [[corpus_ids[i] for i in row.tolist()] for row in top_doc_idx]


# ── Indexed retrieval (full corpus, on-disk SPLADE index) ─────────────────────

def _load_manifest(index_dir: Path, expected_doc_splade: str, expected_vocab: int) -> dict:
    """Load and validate the index manifest. Errors are user-actionable."""
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"\nNo SPLADE index found at {index_dir}.\n"
            f"Build it first (one-time, ~1-2h on an 8GB GPU):\n"
            f"    uv run scripts/build_msmarco_index.py\n"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("doc_splade_hf_id") != expected_doc_splade:
        raise SystemExit(
            f"\nIndex at {index_dir} was built for doc model "
            f"{manifest.get('doc_splade_hf_id')!r}, but config specifies "
            f"{expected_doc_splade!r}. Delete {index_dir} and rebuild."
        )
    if manifest.get("vocab_size") != expected_vocab:
        raise SystemExit(
            f"\nIndex vocab_size {manifest.get('vocab_size')} != model vocab_size "
            f"{expected_vocab}. Delete {index_dir} and rebuild."
        )
    return manifest


def _shard_files(index_dir: Path) -> list:
    return sorted(index_dir.glob("shard_*.npz"))


def indexed_retrieve(
    query_vecs_cpu,
    index_dir: Path,
    vocab_size: int,
    densify_chunk: int,
    topk: int,
    device: torch.device,
):
    """Retrieve top-k from on-disk sparse SPLADE index.

    VRAM at any point:
        queries dense fp16        Q × vocab × 2 B  (≈426 MB for 6980 × 30522)
        densified doc chunk fp16  densify_chunk × vocab × 2 B
        scores fp16               Q × densify_chunk × 2 B
    """
    from tqdm import tqdm

    Q = len(query_vecs_cpu)
    # Queries live on GPU as dense fp16 — small (~430MB) and reused for every chunk.
    # Caller should already pass fp16 and have freed the fp32 copy.
    q_gpu = query_vecs_cpu.to(device) if query_vecs_cpu.dtype == torch.float16 else query_vecs_cpu.half().to(device)

    top_scores = torch.full((Q, topk), float("-inf"), device=device, dtype=torch.float16)
    # Track docids as integer slot positions into a flat list we accumulate.
    top_doc_slot = torch.zeros((Q, topk), dtype=torch.long, device=device)
    docid_table: list = []  # slot index → docid string

    shards = _shard_files(index_dir)
    if not shards:
        raise SystemExit(f"No shard files in {index_dir}; rebuild the index.")

    total_docs = 0
    for sp in shards:
        with np.load(sp, allow_pickle=True) as z:
            total_docs += int(len(z["docids"]))

    pbar = tqdm(total=total_docs, unit="doc", unit_scale=True, dynamic_ncols=True, desc="scoring")

    for sp in shards:
        with np.load(sp, allow_pickle=True) as z:
            indices = z["indices"]            # int32, concat
            values = z["values"]              # fp16, concat
            offsets = z["offsets"]            # int64, length B+1
            shard_docids = z["docids"].tolist()
        B = len(shard_docids)

        # Slot offset for this shard in the global docid_table
        shard_slot_base = len(docid_table)
        docid_table.extend(shard_docids)

        # Process the shard in densify_chunk-sized GPU windows
        for c_start in range(0, B, densify_chunk):
            c_end = min(c_start + densify_chunk, B)
            cb = c_end - c_start

            # Build a dense fp16 [cb, vocab] tile on GPU from this slice's CSR rows
            row_starts = offsets[c_start:c_end]
            row_ends = offsets[c_start + 1 : c_end + 1]
            tile_indices = indices[row_starts[0] : row_ends[-1]]
            tile_values = values[row_starts[0] : row_ends[-1]]
            tile_row_lengths = (row_ends - row_starts).astype(np.int64, copy=False)

            d_dense = torch.zeros((cb, vocab_size), dtype=torch.float16, device=device)
            if tile_indices.size > 0:
                # Construct row index per nonzero entry, then scatter into the dense tile.
                row_ids = np.repeat(np.arange(cb, dtype=np.int64), tile_row_lengths)
                ri = torch.from_numpy(row_ids).to(device)
                ci = torch.from_numpy(tile_indices.astype(np.int64, copy=False)).to(device)
                vv = torch.from_numpy(tile_values).to(device)
                d_dense[ri, ci] = vv

            # Score: [Q, V] @ [V, cb] -> [Q, cb]
            batch_scores = q_gpu @ d_dense.T  # fp16
            del d_dense

            # Slot positions in docid_table for this chunk
            slot_pos = torch.arange(
                shard_slot_base + c_start,
                shard_slot_base + c_end,
                dtype=torch.long, device=device,
            )

            combined = torch.cat([top_scores, batch_scores], dim=1)
            combined_slots = torch.cat([
                top_doc_slot,
                slot_pos.unsqueeze(0).expand(Q, -1),
            ], dim=1)
            new_top = combined.topk(topk, dim=1)
            top_scores = new_top.values
            top_doc_slot = combined_slots.gather(1, new_top.indices)

            del batch_scores, combined, combined_slots

        pbar.update(B)

    pbar.close()

    # Materialise final ranked docid lists on CPU
    slot_idx_cpu = top_doc_slot.cpu().numpy()
    return [[docid_table[i] for i in row] for row in slot_idx_cpu]


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked, qrels, query_ids, k=10):
    scores = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        if not rel:
            continue
        dcg = sum(
            rel.get(did, 0) / math.log2(i + 2) for i, did in enumerate(ranked[qi][:k])
        )
        idcg = sum(
            r / math.log2(i + 2)
            for i, r in enumerate(sorted(rel.values(), reverse=True)[:k])
        )
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(ranked, qrels, query_ids, k=10):
    scores = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        rr = 0.0
        for i, did in enumerate(ranked[qi][:k]):
            if rel.get(did, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def _infer_factor_dim(state: dict, config_fallback: int) -> int:
    """Read factorized_embedding_dim from checkpoint weights rather than config."""
    for k, v in state.items():
        if "lexical_embeddings.weight" in k:
            return v.shape[1]
    return config_fallback


def _load_shallow_query_model(stage: str, sc: dict, checkpoint: str, device):
    """Load a shallow query model from a checkpoint.

    Loads the checkpoint before constructing the model so architecture
    hyperparameters (e.g. factorized_embedding_dim) are read from the saved
    weights rather than from config — avoids shape mismatches when config drifts.
    """
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"]

    if stage == "splade_shallow_align":
        from model import ShallowSpladeQuery
        hf_id = sc["doc_splade_hf_id"]
        print(f"Loading ShallowSpladeQuery ({sc['n_layers']} layers) from {checkpoint} ...")
        model = ShallowSpladeQuery(
            hf_id,
            sc["n_layers"],
            layer_indices=sc.get("layer_indices"),
        )
    elif stage in ("splade_shallow_factorized_align", "splade_shallow_factorized_spaced_align"):
        from model import ShallowFactorizedSpladeQuery
        hf_id = sc["doc_splade_hf_id"]
        factor_dim = _infer_factor_dim(state, sc.get("factorized_embedding_dim", 128))
        print(
            f"Loading ShallowFactorizedSpladeQuery ({sc['n_layers']} layers, "
            f"factor_dim={factor_dim}, "
            f"layers={sc.get('layer_indices', list(range(sc['n_layers'])))}"
            f") from {checkpoint} ..."
        )
        model = ShallowFactorizedSpladeQuery(
            hf_id,
            sc["n_layers"],
            factorized_embedding_dim=factor_dim,
            init="random",
            layer_indices=sc.get("layer_indices"),
        )
    elif stage == "lion_shallow_align":
        from model import ShallowLionQuery
        hf_id = sc["lion_hf_id"]
        print(f"Loading ShallowLionQuery ({sc['n_layers']} layers) from {checkpoint} ...")
        model = ShallowLionQuery(
            hf_id,
            sc["n_layers"],
            layer_indices=sc.get("layer_indices"),
        )
    elif stage in ("lion_shallow_factorized_align", "lion_shallow_factorized_spaced_align"):
        from model import ShallowFactorizedLionQuery
        hf_id = sc["lion_hf_id"]
        factor_dim = _infer_factor_dim(state, sc.get("factorized_embedding_dim", 128))
        print(
            f"Loading ShallowFactorizedLionQuery ({sc['n_layers']} layers, "
            f"factor_dim={factor_dim}, "
            f"layers={sc.get('layer_indices', list(range(sc['n_layers'])))}"
            f") from {checkpoint} ..."
        )
        model = ShallowFactorizedLionQuery(
            hf_id,
            sc["n_layers"],
            factorized_embedding_dim=factor_dim,
            init="random",
            layer_indices=sc.get("layer_indices"),
        )
    else:
        raise ValueError(f"Unsupported shallow stage: {stage}")

    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Step: {ckpt.get('step', 'unknown')}")
    return model, model.tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Full MSMARCO dev eval for VocabTransplant / shallow models"
    )
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file (not needed with --doc_only)")
    parser.add_argument("--doc_only", action="store_true", help="Benchmark doc encoder on both queries and docs (upper-bound ceiling)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage", default="vocab_transplant",
        choices=[
            "vocab_transplant",
            "splade_shallow_align",
            "splade_shallow_factorized_align",
            "splade_shallow_factorized_spaced_align",
            "lion_shallow_align",
            "lion_shallow_factorized_align",
            "lion_shallow_factorized_spaced_align",
        ],
        help="Model type to evaluate (selects config section and model class)",
    )
    parser.add_argument(
        "--doc_batch_size", type=int, default=128,
        help="Passages encoded per GPU call (lower = less VRAM per call)",
    )
    parser.add_argument(
        "--query_batch_size", type=int, default=500,
        help="Queries scored per GPU call (lower = less VRAM, more iterations)",
    )
    parser.add_argument(
        "--encode_batch_size", type=int, default=64,
        help="Query encoding batch size",
    )
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--max_corpus_size", type=int, default=0,
        help="Subsample corpus to this many passages (0 = full 8.8M via on-disk index). "
             "Always includes all relevant passages; rest are random distractors. "
             "Cached to data/msmarco_dev_subset_N.pkl after first build.",
    )
    parser.add_argument(
        "--index_dir", default="data/msmarco_index",
        help="Path to prebuilt SPLADE index (used when --max_corpus_size 0). "
             "Build with: uv run scripts/build_msmarco_index.py",
    )
    parser.add_argument(
        "--densify_chunk", type=int, default=4096,
        help="Docs densified per GPU tile during indexed scoring "
             "(lower = less VRAM, more iterations)",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.doc_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --doc_only is set")

    query_ids, query_texts, qrels = load_dev_queries_and_qrels()

    # ── Stage-specific model + config setup ───────────────────────────────────
    if args.stage in (
        "splade_shallow_align",
        "splade_shallow_factorized_align",
        "splade_shallow_factorized_spaced_align",
        "lion_shallow_align",
        "lion_shallow_factorized_align",
        "lion_shallow_factorized_spaced_align",
    ):
        sc = cfg[args.stage]
        query_max_length = sc["query_max_length"]
        doc_max_length = sc["doc_max_length"]

        if args.stage in (
            "splade_shallow_align",
            "splade_shallow_factorized_align",
            "splade_shallow_factorized_spaced_align",
        ):
            from model import FrozenDocSPLADE
            doc_hf_id = sc["doc_splade_hf_id"]
            print(f"Loading frozen doc SPLADE: {doc_hf_id} ...")
            doc_splade = FrozenDocSPLADE(doc_hf_id)
            # Can use full index (same splade-v3 vocab)
            has_index = args.max_corpus_size == 0
        else:
            from model import FrozenLionSPLADE
            doc_hf_id = sc["lion_hf_id"]
            print(f"Loading frozen Lion doc encoder: {doc_hf_id} ...")
            doc_splade = FrozenLionSPLADE(doc_hf_id)
            # Lion needs a separate 128K-vocab index. If none is present, fall
            # back to subset mode rather than accidentally using the SPLADE-v3
            # 30K-vocab index.
            has_index = args.max_corpus_size == 0
            if has_index and not (Path(args.index_dir) / "manifest.json").exists():
                print(
                    "  [lion] No pre-built Lion index found. Defaulting to 200k subset. "
                    "Build one with scripts/build_msmarco_index.py --stage "
                    f"{args.stage} --index_dir {args.index_dir}"
                )
                has_index = False
                args.max_corpus_size = 200_000

        doc_splade.to(device).eval()

        if args.doc_only:
            print(f"Encoding {len(query_texts)} dev queries with doc encoder (ceiling) ...")
            query_vecs = encode_queries_with_doc_splade(
                doc_splade, query_texts, query_max_length, args.encode_batch_size,
            )
            ckpt_label = f"{doc_hf_id} (doc_only ceiling)"
        else:
            query_model, query_tokenizer = _load_shallow_query_model(
                args.stage, sc, args.checkpoint, device,
            )
            print(f"Encoding {len(query_texts)} dev queries ...")
            query_vecs = encode_queries(
                query_model, query_tokenizer, query_texts,
                query_max_length, args.encode_batch_size, device,
            )
            query_model.cpu()
            torch.cuda.empty_cache()
            del query_model
            gc.collect()
            ckpt_label = args.checkpoint

        corpus_dataset = cfg["sae"]["corpus_dataset"]
        text_field = cfg["sae"].get("corpus_text_field", "text")

        if has_index:
            index_dir = Path(args.index_dir)
            manifest = _load_manifest(
                index_dir,
                expected_doc_splade=doc_hf_id,
                expected_vocab=doc_splade.vocab_size,
            )
            vocab_size = int(manifest["vocab_size"])
            doc_splade.cpu()
            del doc_splade
            torch.cuda.empty_cache()
            # Convert to fp16 and free the fp32 copy before retrieval to save ~850MB RAM.
            query_vecs = query_vecs.half()
            gc.collect()
            ranked = indexed_retrieve(
                query_vecs, index_dir, vocab_size,
                args.densify_chunk, args.topk, device,
            )
            n_indexed = sum(
                int(np.load(sp, allow_pickle=True)["offsets"].shape[0] - 1)
                for sp in _shard_files(index_dir)
            )
            corpus_label = f"{n_indexed:,}-passage on-disk index"
        else:
            corpus_ids, corpus_texts = resolve_corpus_subset(
                args.max_corpus_size, corpus_dataset, text_field, qrels
            )

            # Convert to fp16 to halve query-vec RAM (critical for large-vocab models like Lion)
            query_vecs = query_vecs.half()
            gc.collect()
            print(f"Retrieving from {len(corpus_texts):,}-passage subset ...")
            ranked = list_retrieve(
                query_vecs, doc_splade, corpus_ids, corpus_texts,
                doc_max_length, args.doc_batch_size, args.query_batch_size,
                args.topk, device,
            )
            corpus_label = f"{len(corpus_texts):,}-passage subset"

    else:
        # ── Original vocab_transplant path ────────────────────────────────────
        vc = cfg["vocab_transplant"]
        from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE
        from transformers import AutoTokenizer

        print(f"Loading frozen doc SPLADE: {vc['doc_splade_hf_id']} ...")
        doc_splade = FrozenDocSPLADE(vc["doc_splade_hf_id"])
        doc_splade.to(device)
        doc_splade.eval()

        if args.doc_only:
            print(f"Encoding {len(query_texts)} dev queries with doc SPLADE (ceiling) ...")
            query_vecs = encode_queries_with_doc_splade(
                doc_splade, query_texts, vc["query_max_length"], args.encode_batch_size,
            )
            ckpt_label = f"{vc['doc_splade_hf_id']} (doc_only ceiling)"
        else:
            transplant_dir = str(
                Path(f"checkpoints_{vc['query_hf_id'].split('/')[-1]}")
                / vc["transplant_dir"]
            )
            if not Path(transplant_dir, "config.json").exists():
                raise SystemExit(
                    f"Transplant directory not found: {transplant_dir}\n"
                    f"Expected layout (matches train.py): "
                    f"checkpoints_<query_model>/{vc['transplant_dir']}/config.json\n"
                    f"Check vocab_transplant.query_hf_id and vocab_transplant.transplant_dir in config.yaml."
                )
            print(f"Loading query model from {args.checkpoint} (architecture: {transplant_dir}) ...")
            query_model = VocabTransplantQuerySPLADE(transplant_dir)
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            query_model.load_state_dict(ckpt["model"])
            query_model.to(device)
            query_model.eval()
            query_tokenizer = AutoTokenizer.from_pretrained(transplant_dir)
            print(f"  Checkpoint step: {ckpt.get('step', 'unknown')}")

            print(f"Encoding {len(query_texts)} dev queries ...")
            query_vecs = encode_queries(
                query_model, query_tokenizer, query_texts,
                vc["query_max_length"], args.encode_batch_size, device,
            )
            query_model.cpu()
            torch.cuda.empty_cache()
            del query_model
            gc.collect()
            ckpt_label = args.checkpoint

        corpus_dataset = cfg["sae"]["corpus_dataset"]
        text_field = cfg["sae"].get("corpus_text_field", "text")

        if args.max_corpus_size > 0:
            corpus_ids, corpus_texts = resolve_corpus_subset(
                args.max_corpus_size, corpus_dataset, text_field, qrels
            )

            query_vecs = query_vecs.half()
            gc.collect()
            print(f"Retrieving from {len(corpus_texts):,}-passage subset ...")
            ranked = list_retrieve(
                query_vecs, doc_splade, corpus_ids, corpus_texts,
                vc["doc_max_length"], args.doc_batch_size, args.query_batch_size,
                args.topk, device,
            )
            corpus_label = f"{len(corpus_texts):,}-passage subset"
        else:
            index_dir = Path(args.index_dir)
            manifest = _load_manifest(
                index_dir,
                expected_doc_splade=vc["doc_splade_hf_id"],
                expected_vocab=doc_splade.vocab_size,
            )
            vocab_size = int(manifest["vocab_size"])
            doc_splade.cpu()
            del doc_splade
            torch.cuda.empty_cache()
            query_vecs = query_vecs.half()
            gc.collect()
            ranked = indexed_retrieve(
                query_vecs, index_dir, vocab_size,
                args.densify_chunk, args.topk, device,
            )
            n_indexed = sum(
                int(np.load(sp, allow_pickle=True)["offsets"].shape[0] - 1)
                for sp in _shard_files(index_dir)
            )
            corpus_label = f"{n_indexed:,}-passage on-disk index"

    n10 = ndcg_at_k(ranked, qrels, query_ids, k=10)
    m10 = mrr_at_k(ranked, qrels, query_ids, k=10)

    print(f"\n{'='*50}")
    print(f"MSMARCO Dev  ({len(query_ids)} queries, {corpus_label})")
    print(f"  NDCG@10 : {n10:.4f}")
    print(f"  MRR@10  : {m10:.4f}")
    print(f"  Checkpoint : {ckpt_label}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
