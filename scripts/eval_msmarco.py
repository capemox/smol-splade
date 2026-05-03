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
import math
import sys
import time
from pathlib import Path

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
    """Encode all dev queries with the query model; returns fp32 CPU tensor [Q, vocab]."""
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
            all_vecs.append(vecs.cpu().float())
            print(f"  queries: {min(i+batch_size, len(query_texts))}/{len(query_texts)}", end="\r")
    print()
    return torch.cat(all_vecs, dim=0)


def encode_queries_with_doc_splade(doc_splade, query_texts, max_length, batch_size):
    """Encode queries using the frozen doc SPLADE (ceiling / doc-only benchmark)."""
    all_vecs = []
    for i in range(0, len(query_texts), batch_size):
        batch = query_texts[i : i + batch_size]
        vecs = doc_splade.encode(batch, max_length, no_grad=True)
        all_vecs.append(vecs.cpu().float())
        print(f"  queries (doc_splade): {min(i+batch_size, len(query_texts))}/{len(query_texts)}", end="\r")
    print()
    return torch.cat(all_vecs, dim=0)


# ── Corpus subset ─────────────────────────────────────────────────────────────

def build_or_load_subset(full_corpus: dict, qrels: dict, max_corpus_size: int) -> tuple:
    """Return (corpus_ids, corpus_texts) for a subset of at most max_corpus_size passages.

    All passages that are relevant to at least one dev query are always included.
    The remainder is filled with randomly sampled distractors from the full corpus.
    The result is pickled to data/ so subsequent runs are instant.
    """
    import pickle, random
    from pathlib import Path

    cache = Path("data") / f"msmarco_dev_subset_{max_corpus_size}.pkl"
    if cache.exists():
        print(f"Loading corpus subset from {cache} ...")
        with cache.open("rb") as f:
            return pickle.load(f)

    print(f"Building {max_corpus_size:,}-passage subset (includes all relevant docs) ...")
    relevant_ids = {did for rel in qrels.values() for did in rel}
    relevant_ids = {did for did in relevant_ids if did in full_corpus}

    distractor_pool = [k for k in full_corpus if k not in relevant_ids]
    n_distractors = max(0, max_corpus_size - len(relevant_ids))
    distractors = random.sample(distractor_pool, min(n_distractors, len(distractor_pool)))

    subset_ids = list(relevant_ids) + distractors
    random.shuffle(subset_ids)
    subset_texts = [full_corpus[k] for k in subset_ids]

    print(f"  {len(relevant_ids):,} relevant + {len(distractors):,} distractors = {len(subset_ids):,} total")
    with cache.open("wb") as f:
        pickle.dump((subset_ids, subset_texts), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached → {cache}")
    return subset_ids, subset_texts


# ── Retrieval ─────────────────────────────────────────────────────────────────

def streaming_retrieve(
    query_vecs_cpu,
    doc_splade,
    corpus_dataset,
    text_field,
    doc_max_length,
    doc_batch_size,
    query_batch_size,
    topk,
    device,
):
    """Stream corpus one batch at a time; score against query sub-batches.

    VRAM at any point:
        doc_splade model      ~440 MB
        query sub-batch       query_batch_size × 30522 × 2 bytes fp16
        doc batch             doc_batch_size   × 30522 × 2 bytes fp16
    RAM at any point:
        query_vecs_cpu        6980 × 30522 × 4 bytes fp32  ~854 MB
        one doc batch buffer  doc_batch_size strings + encoded fp16 tensor
    """
    from datasets import load_dataset

    Q = len(query_vecs_cpu)
    top_scores = torch.full((Q, topk), float("-inf"))
    # Store sequential corpus positions (= string doc IDs for Tevatron corpus)
    top_doc_pos = torch.zeros((Q, topk), dtype=torch.long)

    def process_batch(buf_texts, doc_start):
        nonlocal top_scores, top_doc_pos
        B = len(buf_texts)

        # Encode on GPU, immediately pull back to CPU as fp16 to free VRAM
        doc_vecs_cpu = doc_splade.encode(buf_texts, doc_max_length, no_grad=True).cpu().half()

        for q_start in range(0, Q, query_batch_size):
            q_end = min(q_start + query_batch_size, Q)

            # Brief GPU residency: one query sub-batch + one doc batch
            q_gpu = query_vecs_cpu[q_start:q_end].half().to(device)   # [Qb, vocab]
            d_gpu = doc_vecs_cpu.to(device)                            # [B, vocab]
            batch_scores = (q_gpu @ d_gpu.T).cpu().float()             # [Qb, B]
            del q_gpu, d_gpu

            combined = torch.cat([top_scores[q_start:q_end], batch_scores], dim=1)
            doc_pos = torch.arange(doc_start, doc_start + B, dtype=torch.long)
            combined_pos = torch.cat([
                top_doc_pos[q_start:q_end],
                doc_pos.unsqueeze(0).expand(q_end - q_start, -1),
            ], dim=1)

            new_top = combined.topk(topk, dim=1)
            top_scores[q_start:q_end] = new_top.values
            top_doc_pos[q_start:q_end] = combined_pos.gather(1, new_top.indices)

    from tqdm import tqdm

    print(f"Streaming corpus from {corpus_dataset} ...")
    ds = load_dataset(corpus_dataset, split="train", streaming=True)

    buf_texts: list = []
    doc_position = 0
    CORPUS_SIZE = 8_841_823  # known size; lets tqdm show accurate ETA

    pbar = tqdm(total=CORPUS_SIZE, unit="doc", unit_scale=True, dynamic_ncols=True)

    for item in ds:
        text = item.get(text_field) or item.get("passage") or item.get("contents", "")
        buf_texts.append(text)
        doc_position += 1

        if len(buf_texts) == doc_batch_size:
            process_batch(buf_texts, doc_position - doc_batch_size)
            buf_texts = []
            pbar.update(doc_batch_size)

    if buf_texts:
        process_batch(buf_texts, doc_position - len(buf_texts))
        pbar.update(len(buf_texts))

    pbar.close()

    # Tevatron docids are str(sequential_position), matching BeIR qrels corpus-ids
    ranked = [[str(i) for i in row.tolist()] for row in top_doc_pos]
    return ranked


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

def main():
    parser = argparse.ArgumentParser(
        description="Full MSMARCO dev eval for VocabTransplant models"
    )
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file (not needed with --doc_only)")
    parser.add_argument("--doc_only", action="store_true", help="Benchmark doc encoder on both queries and docs (upper-bound ceiling)")
    parser.add_argument("--config", default="config.yaml")
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
        help="Subsample corpus to this many passages (0 = full 8.8M). "
             "Always includes all relevant passages; rest are random distractors. "
             "Cached to data/msmarco_dev_subset_N.pkl after first build.",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    vc = cfg["vocab_transplant"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.doc_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --doc_only is set")

    from model import FrozenDocSPLADE, VocabTransplantQuerySPLADE
    from transformers import AutoTokenizer

    print(f"Loading frozen doc SPLADE: {vc['doc_splade_hf_id']} ...")
    doc_splade = FrozenDocSPLADE(vc["doc_splade_hf_id"])
    doc_splade.to(device)
    doc_splade.eval()

    query_ids, query_texts, qrels = load_dev_queries_and_qrels()

    if args.doc_only:
        print(f"Encoding {len(query_texts)} dev queries with doc SPLADE (ceiling) ...")
        query_vecs = encode_queries_with_doc_splade(
            doc_splade, query_texts, vc["query_max_length"], args.encode_batch_size,
        )
        ckpt_label = f"{vc['doc_splade_hf_id']} (doc_only ceiling)"
    else:
        transplant_dir = vc["transplant_dir"]
        print(f"Loading query model from {args.checkpoint} ...")
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
        ckpt_label = args.checkpoint

    corpus_dataset = cfg["sae"]["corpus_dataset"]
    text_field = cfg["sae"].get("corpus_text_field", "text")

    if args.max_corpus_size > 0:
        # Subset mode: load full corpus pickle, subsample, retrieve from list
        pickle_path = Path("data") / (corpus_dataset.replace("/", "__") + ".pkl")
        import pickle
        print(f"Loading corpus pickle for subset build ...")
        with pickle_path.open("rb") as f:
            full_corpus = pickle.load(f)
        corpus_ids, corpus_texts = build_or_load_subset(full_corpus, qrels, args.max_corpus_size)
        del full_corpus

        print(f"Retrieving from {len(corpus_texts):,}-passage subset ...")
        ranked = list_retrieve(
            query_vecs, doc_splade, corpus_ids, corpus_texts,
            vc["doc_max_length"], args.doc_batch_size, args.query_batch_size,
            args.topk, device,
        )
        corpus_label = f"{len(corpus_texts):,}-passage subset"
    else:
        # Full corpus: stream from HuggingFace dataset
        ranked = streaming_retrieve(
            query_vecs, doc_splade, corpus_dataset, text_field,
            vc["doc_max_length"], args.doc_batch_size, args.query_batch_size,
            args.topk, device,
        )
        corpus_label = "8.8M full corpus"

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
