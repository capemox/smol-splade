"""NanoBEIR evaluation for SAE-SPLADE."""

import math
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


def load_nanobeir(dataset_name: str):
    """Load corpus, queries, qrels from a zeta-alpha-ai/Nano* dataset."""
    from datasets import load_dataset

    corpus_ds = load_dataset(dataset_name, "corpus", split="train")
    query_ds = load_dataset(dataset_name, "queries", split="train")
    qrels_ds = load_dataset(dataset_name, "qrels", split="train")

    corpus_ids = [str(x) for x in corpus_ds["_id"]]
    corpus_texts = [t or "" for t in corpus_ds["text"]]

    query_ids = [str(x) for x in query_ds["_id"]]
    query_texts = [t or "" for t in query_ds["text"]]

    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query_id") or row.get("query-id", ""))
        did = str(row.get("doc_id") or row.get("corpus-id") or row.get("corpus_id", ""))
        score = int(row.get("score") or row.get("relevance") or 1)  # implicit 1 if no score column
        qrels.setdefault(qid, {})[did] = score

    return corpus_ids, corpus_texts, query_ids, query_texts, qrels


def _encode_texts(model, tokenizer, texts: List[str], max_length: int, batch_size: int, device) -> torch.Tensor:
    """Encode texts to SPLADE vectors; returns float32 tensor on CPU."""
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            vecs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            all_vecs.append(vecs.cpu().float())
    return torch.cat(all_vecs, dim=0)


def _ndcg_at_k(ranked_doc_ids: List[List[str]], qrels: Dict[str, Dict[str, int]], query_ids: List[str], k: int = 10) -> float:
    scores = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        if not rel:
            continue
        dcg = sum(rel.get(did, 0) / math.log2(i + 2) for i, did in enumerate(ranked_doc_ids[qi][:k]))
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_nanobeir(model, tokenizer, cfg: dict, device, writer=None, step: int = 0) -> Dict[str, float]:
    """Evaluate on NanoBEIR datasets. Logs NDCG@10 to tensorboard if writer is given."""
    ec = cfg.get("eval", {})
    datasets = ec.get("datasets", ["zeta-alpha-ai/NanoMSMARCO"])
    batch_size = ec.get("batch_size", 32)
    doc_max_length = cfg["splade"]["doc_max_length"]
    query_max_length = cfg["splade"]["query_max_length"]

    model.eval()
    results: Dict[str, float] = {}

    for ds_name in datasets:
        short = ds_name.split("/")[-1]
        print(f"  [{short}] loading …")
        corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_nanobeir(ds_name)

        print(f"  [{short}] encoding {len(corpus_texts)} docs …")
        corpus_vecs = _encode_texts(model, tokenizer, corpus_texts, doc_max_length, batch_size, device)

        print(f"  [{short}] encoding {len(query_texts)} queries …")
        query_vecs = _encode_texts(model, tokenizer, query_texts, query_max_length, batch_size, device)

        scores = query_vecs @ corpus_vecs.T                              # [Q, D]
        ranked_indices = scores.argsort(dim=-1, descending=True).tolist()
        ranked_doc_ids = [[corpus_ids[i] for i in row] for row in ranked_indices]

        ndcg = _ndcg_at_k(ranked_doc_ids, qrels, query_ids, k=10)
        results[short] = ndcg
        print(f"  [{short}] NDCG@10 = {ndcg:.4f}")

        if writer is not None:
            writer.add_scalar(f"eval/{short}/ndcg@10", ndcg, step)

    model.train()
    return results
