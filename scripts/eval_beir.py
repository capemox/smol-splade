#!/usr/bin/env python
"""BEIR evaluation for shallow/pruned query models.

For each dataset, encodes queries with either a trained query checkpoint or the
frozen doc SPLADE (ceiling), then retrieves against:
  - a pre-built on-disk CSR index (fast; build with build_beir_index.py), or
  - in-memory doc encoding as a fallback (convenient for small corpora).

Reports NDCG@10 and MRR@10 per dataset and averaged.

Usage:
    # Doc-only ceiling (doc encoder on both sides):
    uv run scripts/eval_beir.py --doc_only

    # Evaluate a query checkpoint:
    uv run scripts/eval_beir.py --stage splade_shallow --checkpoint checkpoints_splade-v3/splade_shallow/align_final.pt

    # Specific datasets only:
    uv run scripts/eval_beir.py --checkpoint <ckpt> --datasets nfcorpus scifact fiqa

    # Lion:
    uv run scripts/eval_beir.py --stage lion_shallow --checkpoint checkpoints_lion_shallow/lion_shallow/align_final.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_DATASETS = [
    # Small BEIR datasets (≤60K docs) — suitable for in-memory or indexed eval
    "nfcorpus",  # ~3.6K docs, 323 queries
    "scifact",   # ~5K docs,   300 queries
    "arguana",   # ~8.6K docs, 1,406 queries
    "scidocs",   # ~25K docs,  1,000 queries
    "fiqa",      # ~57K docs,  648 queries
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    return (title + " " + text).strip() if title else text


def load_beir_queries_and_qrels(name: str):
    from datasets import load_dataset

    # qrels are a separate HuggingFace dataset: BeIR/<name>-qrels
    # Most datasets use the "test" split; a few use "validation" (e.g. quora)
    qrels_ds = None
    for split in ("test", "validation"):
        try:
            qrels_ds = load_dataset(f"BeIR/{name}-qrels", split=split)
            break
        except Exception:
            continue
    if qrels_ds is None:
        raise RuntimeError(
            f"Could not load qrels for BeIR/{name}-qrels (tried splits: test, validation)"
        )

    qrels: dict = {}
    dev_qids: set = set()
    for row in qrels_ds:
        qid = str(row.get("query-id") or row.get("query_id", ""))
        did = str(row.get("corpus-id") or row.get("corpus_id", ""))
        score = int(row.get("score", 1))
        qrels.setdefault(qid, {})[did] = score
        dev_qids.add(qid)

    query_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    query_ids, query_texts = [], []
    for row in query_ds:
        qid = str(row["_id"])
        if qid in dev_qids:
            query_ids.append(qid)
            query_texts.append((row.get("text") or "").strip())

    return query_ids, query_texts, qrels


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_with_query_model(model, tokenizer, texts, max_length, batch_size, device):
    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
            )
            vecs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            all_vecs.append(vecs.cpu().float())
            print(f"  queries: {min(i+batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return torch.cat(all_vecs, dim=0)


def encode_with_doc_splade(doc_splade, texts, max_length, batch_size):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = doc_splade.encode(batch, max_length, no_grad=True)
        all_vecs.append(vecs.cpu().float())
        print(f"  queries (doc_splade): {min(i+batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return torch.cat(all_vecs, dim=0)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _shard_files(index_dir: Path) -> list:
    return sorted(index_dir.glob("shard_*.npz"))


def indexed_retrieve(query_vecs_cpu, index_dir: Path, vocab_size: int,
                     densify_chunk: int, topk: int, device: torch.device):
    """Score queries against a pre-built on-disk CSR index (same format as MSMARCO index)."""
    from tqdm import tqdm

    Q = len(query_vecs_cpu)
    q_gpu = query_vecs_cpu.half().to(device)
    top_scores   = torch.full((Q, topk), float("-inf"), device=device, dtype=torch.float16)
    top_doc_slot = torch.zeros((Q, topk), dtype=torch.long, device=device)
    docid_table: list = []

    shards = _shard_files(index_dir)
    total_docs = sum(
        int(np.load(sp, allow_pickle=True)["offsets"].shape[0] - 1) for sp in shards
    )
    pbar = tqdm(total=total_docs, unit="doc", unit_scale=True,
                dynamic_ncols=True, desc="  scoring")

    for sp in shards:
        with np.load(sp, allow_pickle=True) as z:
            indices      = z["indices"]
            values       = z["values"]
            offsets      = z["offsets"]
            shard_docids = z["docids"].tolist()
        B = len(shard_docids)
        shard_slot_base = len(docid_table)
        docid_table.extend(shard_docids)

        for c_start in range(0, B, densify_chunk):
            c_end = min(c_start + densify_chunk, B)
            cb = c_end - c_start

            row_starts = offsets[c_start : c_end]
            row_ends   = offsets[c_start + 1 : c_end + 1]
            tile_indices = indices[row_starts[0] : row_ends[-1]]
            tile_values  = values[row_starts[0] : row_ends[-1]]
            tile_row_lengths = (row_ends - row_starts).astype(np.int64, copy=False)

            d_dense = torch.zeros((cb, vocab_size), dtype=torch.float16, device=device)
            if tile_indices.size > 0:
                row_ids = np.repeat(np.arange(cb, dtype=np.int64), tile_row_lengths)
                ri = torch.from_numpy(row_ids).to(device)
                ci = torch.from_numpy(tile_indices.astype(np.int64, copy=False)).to(device)
                vv = torch.from_numpy(tile_values).to(device)
                d_dense[ri, ci] = vv

            batch_scores = q_gpu @ d_dense.T
            del d_dense

            slot_pos = torch.arange(
                shard_slot_base + c_start, shard_slot_base + c_end,
                dtype=torch.long, device=device,
            )
            combined       = torch.cat([top_scores, batch_scores], dim=1)
            combined_slots = torch.cat([top_doc_slot, slot_pos.unsqueeze(0).expand(Q, -1)], dim=1)
            new_top        = combined.topk(topk, dim=1)
            top_scores     = new_top.values
            top_doc_slot   = combined_slots.gather(1, new_top.indices)
            del batch_scores, combined, combined_slots

        pbar.update(B)

    pbar.close()
    slot_idx_cpu = top_doc_slot.cpu().numpy()
    return [[docid_table[i] for i in row] for row in slot_idx_cpu], total_docs


def inmemory_retrieve(query_vecs_cpu, doc_splade, corpus_ids, corpus_texts,
                      doc_max_length: int, doc_batch_size: int, topk: int,
                      device: torch.device):
    """Encode corpus on-the-fly and retrieve. Suitable for small corpora."""
    from tqdm import tqdm

    Q = len(query_vecs_cpu)
    top_scores  = torch.full((Q, topk), float("-inf"))
    top_doc_idx = torch.zeros((Q, topk), dtype=torch.long)

    pbar = tqdm(total=len(corpus_texts), unit="doc", unit_scale=True,
                dynamic_ncols=True, desc="  encode+score")

    for doc_start in range(0, len(corpus_texts), doc_batch_size):
        doc_end  = min(doc_start + doc_batch_size, len(corpus_texts))
        buf      = corpus_texts[doc_start:doc_end]
        doc_vecs = doc_splade.encode(buf, doc_max_length, no_grad=True).cpu().float()

        batch_scores = query_vecs_cpu @ doc_vecs.T
        combined     = torch.cat([top_scores, batch_scores], dim=1)
        pos          = torch.arange(doc_start, doc_end, dtype=torch.long)
        combined_idx = torch.cat([top_doc_idx, pos.unsqueeze(0).expand(Q, -1)], dim=1)
        new_top      = combined.topk(topk, dim=1)
        top_scores   = new_top.values
        top_doc_idx  = combined_idx.gather(1, new_top.indices)

        pbar.update(len(buf))

    pbar.close()
    return [[corpus_ids[i] for i in row.tolist()] for row in top_doc_idx], len(corpus_ids)


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked, qrels, query_ids, k=10):
    scores = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        if not rel:
            continue
        dcg  = sum(rel.get(did, 0) / math.log2(i + 2) for i, did in enumerate(ranked[qi][:k]))
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(ranked, qrels, query_ids, k=10):
    scores = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        rr = next(
            (1.0 / (i + 1) for i, did in enumerate(ranked[qi][:k]) if rel.get(did, 0) > 0),
            0.0,
        )
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


# ── Model loading helpers ─────────────────────────────────────────────────────

def _infer_factor_dim(state: dict, config_fallback: int) -> int:
    for k, v in state.items():
        if "lexical_embeddings.weight" in k:
            return v.shape[1]
    return config_fallback


def _infer_layer_indices(state: dict) -> list[int] | None:
    import re
    patterns = [
        re.compile(r"encoder\.layer\.(\d+)\."),   # BERT / SPLADE
        re.compile(r"\.layers\.(\d+)\."),           # LLaMA / Lion
    ]
    for pat in patterns:
        indices = sorted({int(m.group(1)) for k in state for m in [pat.search(k)] if m})
        if indices:
            return indices
    return None


def _load_shallow_query_model(stage: str, sc: dict, checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"]

    if stage == "splade_static":
        from model import StaticSpladeQuery
        print(f"Loading StaticSpladeQuery from {checkpoint} ...")
        model = StaticSpladeQuery(sc["doc_splade_hf_id"])
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"  Step: {ckpt.get('step', 'unknown')}")
        return model, model.tokenizer

    if stage == "lion_static":
        from model import StaticLionQuery
        print(f"Loading StaticLionQuery from {checkpoint} ...")
        model = StaticLionQuery(sc["lion_hf_id"])
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"  Step: {ckpt.get('step', 'unknown')}")
        return model, model.tokenizer

    layer_indices = _infer_layer_indices(state)
    if layer_indices is not None:
        n_layers = len(layer_indices)
    else:
        layer_indices = sc.get("layer_indices")
        n_layers = len(layer_indices) if layer_indices is not None else sc["n_layers"]
    factorized = bool(sc.get("factorize_embeddings", False))

    if stage == "splade_shallow":
        if factorized:
            from model import ShallowFactorizedSpladeQuery
            factor_dim = _infer_factor_dim(state, sc.get("factorized_embedding_dim", 128))
            print(f"Loading ShallowFactorizedSpladeQuery ({n_layers} layers, factor_dim={factor_dim}) from {checkpoint} ...")
            model = ShallowFactorizedSpladeQuery(
                sc["doc_splade_hf_id"], n_layers,
                factorized_embedding_dim=factor_dim, init="random",
                layer_indices=layer_indices,
            )
        else:
            from model import ShallowSpladeQuery
            print(f"Loading ShallowSpladeQuery ({n_layers} layers) from {checkpoint} ...")
            model = ShallowSpladeQuery(
                sc["doc_splade_hf_id"], n_layers,
                layer_indices=layer_indices,
            )
    elif stage == "lion_shallow":
        if factorized:
            from model import ShallowFactorizedLionQuery
            factor_dim = _infer_factor_dim(state, sc.get("factorized_embedding_dim", 128))
            print(f"Loading ShallowFactorizedLionQuery ({n_layers} layers, factor_dim={factor_dim}) from {checkpoint} ...")
            model = ShallowFactorizedLionQuery(
                sc["lion_hf_id"], n_layers,
                factorized_embedding_dim=factor_dim, init="random",
                layer_indices=layer_indices,
            )
        else:
            from model import ShallowLionQuery
            print(f"Loading ShallowLionQuery ({n_layers} layers) from {checkpoint} ...")
            model = ShallowLionQuery(
                sc["lion_hf_id"], n_layers,
                layer_indices=layer_indices,
            )
    else:
        raise ValueError(f"Unsupported shallow stage: {stage}")

    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Step: {ckpt.get('step', 'unknown')}")
    return model, model.tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pt checkpoint file (not needed with --doc_only)",
    )
    parser.add_argument(
        "--doc_only", action="store_true",
        help="Use doc encoder for both query and doc encoding (upper-bound ceiling)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage",
        default="splade_shallow",
        choices=["splade_shallow", "lion_shallow", "splade_static", "lion_static"],
        help="Config section to load doc encoder and query model from",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="BEIR dataset names (loaded as BeIR/<name>)")
    parser.add_argument("--index_dir", default="data/beir_index",
                        help="Base dir for pre-built per-dataset indexes "
                             "(built with build_beir_index.py)")
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument(
        "--doc_batch_size", type=int, default=128,
        help="Doc encoding batch size for in-memory fallback (when no index found)",
    )
    parser.add_argument("--densify_chunk", type=int, default=4096,
                        help="Docs densified per GPU tile during indexed scoring")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    if not args.doc_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --doc_only is set")

    import yaml
    cfg = yaml.safe_load(open(args.config))
    sc = cfg[args.stage]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.stage.startswith("lion_"):
        from model import FrozenLionSPLADE
        doc_hf_id = sc["lion_hf_id"]
        print(f"Loading frozen Lion doc encoder: {doc_hf_id} ...")
        doc_splade = FrozenLionSPLADE(doc_hf_id)
        # Lion (1B params) is left on CPU until actually needed. Moving it to the
        # GPU unconditionally fills all VRAM and leaves nothing for indexed-retrieval
        # dense chunks (~2 GB per 4096-doc tile at vocab_size=128256).
        _doc_splade_on_device = False
    else:
        from model import FrozenDocSPLADE
        doc_hf_id = sc["doc_splade_hf_id"]
        print(f"Loading frozen doc SPLADE: {doc_hf_id} ...")
        doc_splade = FrozenDocSPLADE(doc_hf_id)
        doc_splade.to(device)
        _doc_splade_on_device = True
    doc_splade.eval()

    query_model     = None
    query_tokenizer = None
    if not args.doc_only:
        query_model, query_tokenizer = _load_shallow_query_model(
            args.stage, sc, args.checkpoint, device
        )

    base_index_dir = Path(args.index_dir)
    all_results: dict = {}

    for name in args.datasets:
        print(f"\n{'─'*60}")
        print(f"[{name}]")

        print(f"  Loading queries and qrels ...")
        try:
            query_ids, query_texts, qrels = load_beir_queries_and_qrels(name)
        except Exception as e:
            print(f"  SKIP — could not load dataset: {e}")
            continue
        print(f"  {len(query_ids)} queries, {len(qrels)} with relevance judgments")

        # Encode queries
        if args.doc_only:
            if not _doc_splade_on_device:
                doc_splade.to(device)
                _doc_splade_on_device = True
            print(f"  Encoding queries with doc encoder (ceiling) ...")
            query_vecs = encode_with_doc_splade(
                doc_splade, query_texts, sc["query_max_length"], args.encode_batch_size
            )
        else:
            print(f"  Encoding queries with query model ...")
            query_vecs = encode_with_query_model(
                query_model, query_tokenizer, query_texts,
                sc["query_max_length"], args.encode_batch_size, device,
            )

        # Retrieve: prefer pre-built index, fall back to in-memory
        index_dir     = base_index_dir / args.stage / name
        manifest_path = index_dir / "manifest.json"

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("doc_splade_hf_id") != doc_hf_id:
                print(
                    f"  WARNING: index was built for {manifest['doc_splade_hf_id']!r} "
                    f"but config has {doc_hf_id!r}. Skipping {name}.\n"
                    f"  Delete {index_dir} and rebuild with build_beir_index.py --stage {args.stage}."
                )
                continue
            vocab_size   = int(manifest["vocab_size"])
            n_index_docs = int(manifest.get("target_total_docs", 0))
            print(f"  Retrieving from pre-built index ({n_index_docs:,} docs) ...")
            ranked, n_docs = indexed_retrieve(
                query_vecs, index_dir, vocab_size,
                args.densify_chunk, args.topk, device,
            )
            corpus_label = f"index ({n_docs:,} docs)"
        else:
            print(f"  No index found at {index_dir} — encoding corpus on-the-fly.")
            print(
                f"  (Build an index for faster repeated eval: "
                f"uv run scripts/build_beir_index.py --stage {args.stage} --datasets {name})"
            )
            if not _doc_splade_on_device:
                doc_splade.to(device)
                _doc_splade_on_device = True
            from datasets import load_dataset
            corpus_ds    = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
            corpus_ids   = [str(r["_id"]) for r in corpus_ds]
            corpus_texts = [_corpus_text(r) for r in corpus_ds]
            ranked, n_docs = inmemory_retrieve(
                query_vecs, doc_splade, corpus_ids, corpus_texts,
                sc["doc_max_length"], args.doc_batch_size, args.topk, device,
            )
            corpus_label = f"in-memory ({n_docs:,} docs)"

        n10 = ndcg_at_k(ranked, qrels, query_ids, k=10)
        m10 = mrr_at_k(ranked, qrels, query_ids, k=10)
        all_results[name] = {"ndcg@10": n10, "mrr@10": m10}
        print(f"  NDCG@10: {n10:.4f}  MRR@10: {m10:.4f}  [{corpus_label}]")

    if not all_results:
        print("\nNo datasets evaluated.")
        return

    # Summary table
    ckpt_label = (
        f"{doc_hf_id} (doc_only ceiling)"
        if args.doc_only else args.checkpoint
    )
    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_label}")
    print(f"Stage      : {args.stage}")
    print(f"{'─'*60}")
    print(f"{'Dataset':<20} {'NDCG@10':>10} {'MRR@10':>10}")
    print(f"{'─'*42}")
    for ds, m in all_results.items():
        print(f"{ds:<20} {m['ndcg@10']:>10.4f} {m['mrr@10']:>10.4f}")
    avg_ndcg = sum(m["ndcg@10"] for m in all_results.values()) / len(all_results)
    avg_mrr  = sum(m["mrr@10"]  for m in all_results.values()) / len(all_results)
    print(f"{'─'*42}")
    print(f"{'Average':<20} {avg_ndcg:>10.4f} {avg_mrr:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
