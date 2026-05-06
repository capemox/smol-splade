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


def _encode_texts(model, tokenizer, texts: List[str], max_length: int, batch_size: int, device, override_k: int = 0) -> torch.Tensor:
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
            vecs = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device), override_k=override_k)
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


def _encode_texts_frozen(doc_splade, texts: List[str], max_length: int, batch_size: int) -> torch.Tensor:
    """Encode texts with FrozenDocSPLADE (handles its own tokenization); returns float32 on CPU."""
    # Pre-allocate to avoid the 2× peak that torch.cat causes (copy of list + output tensor).
    first = doc_splade.encode(texts[:batch_size], max_length).cpu().float()
    out = torch.zeros(len(texts), first.shape[-1], dtype=torch.float32)
    out[: len(first)] = first
    del first
    for i in range(batch_size, len(texts), batch_size):
        vecs = doc_splade.encode(texts[i : i + batch_size], max_length).cpu().float()
        out[i : i + len(vecs)] = vecs
        del vecs
    return out


def evaluate_asymmetric(
    query_model,
    query_tokenizer,
    doc_splade,
    cfg: dict,
    device,
    writer=None,
    step: int = 0,
    run_doc_doc: bool = True,
    override_k: int = 0,
    section: str = "asymmetric",
) -> Dict[str, Dict[str, float]]:
    """Evaluate asymmetric retrieval on NanoBEIR datasets.

    Always runs query-doc eval (query_model encodes queries, doc_splade encodes docs).
    When ``run_doc_doc=True``, also runs the doc-doc upper-bound eval (doc_splade
    encodes both sides) — useful once at the start to establish the ceiling.

    Returns ``{dataset_short_name: {"query_doc": float, "doc_doc": float}}``.
    """
    ec = cfg.get("eval", {})
    datasets = ec.get("datasets", ["zeta-alpha-ai/NanoMSMARCO"])
    batch_size = ec.get("batch_size", 32)
    ac = cfg[section]
    batch_size = ac.get("eval_batch_size", batch_size)
    doc_max = ac["doc_max_length"]
    query_max = ac["query_max_length"]

    query_model.eval()
    results: Dict[str, Dict[str, float]] = {}

    import gc

    for ds_name in datasets:
        gc.collect()
        torch.cuda.empty_cache()

        short = ds_name.split("/")[-1]
        print(f"  [{short}] loading …")
        corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_nanobeir(ds_name)

        print(f"  [{short}] encoding {len(corpus_texts)} docs (doc_splade) …")
        corpus_vecs = _encode_texts_frozen(doc_splade, corpus_texts, doc_max, batch_size)

        row: Dict[str, float] = {}

        if run_doc_doc:
            print(f"  [{short}] encoding {len(query_texts)} queries (doc_doc ceiling) …")
            q_vecs_dd = _encode_texts_frozen(doc_splade, query_texts, query_max, batch_size)
            ranked_dd = [
                [corpus_ids[i] for i in r]
                for r in (q_vecs_dd @ corpus_vecs.T).argsort(dim=-1, descending=True).tolist()
            ]
            del q_vecs_dd
            ndcg_dd = _ndcg_at_k(ranked_dd, qrels, query_ids, k=10)
            row["doc_doc"] = ndcg_dd
            print(f"  [{short}] NDCG@10  doc_doc (ceiling) : {ndcg_dd:.4f}")
            if writer is not None:
                writer.add_scalar(f"eval/{short}/doc_doc_ceiling", ndcg_dd, step)

        k_label = f"k={override_k}" if override_k > 0 else (f"k={query_model.sae.k}" if hasattr(query_model, "sae") else "dense")
        print(f"  [{short}] encoding {len(query_texts)} queries (query_model, {k_label}) …")
        q_vecs_qd = _encode_texts(query_model, query_tokenizer, query_texts, query_max, batch_size, device, override_k=override_k)
        ranked_qd = [
            [corpus_ids[i] for i in r]
            for r in (q_vecs_qd @ corpus_vecs.T).argsort(dim=-1, descending=True).tolist()
        ]
        del corpus_vecs, q_vecs_qd
        ndcg_qd = _ndcg_at_k(ranked_qd, qrels, query_ids, k=10)
        row["query_doc"] = ndcg_qd
        print(f"  [{short}] NDCG@10  query_doc ({k_label})  : {ndcg_qd:.4f}")
        if writer is not None:
            writer.add_scalar(f"eval/{short}/query_doc", ndcg_qd, step)

        results[short] = row

    query_model.train()
    return results


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
