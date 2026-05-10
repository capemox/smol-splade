"""Data loading utilities for SAE pretraining and SPLADE finetuning."""

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_tokenizer(hf_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(hf_id)


def tokenize(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}


# ──────────────────────────────────────────────────────────────────────────────
# SAE pretraining: MSMARCO corpus stream
# ──────────────────────────────────────────────────────────────────────────────

class CorpusStreamDataset(IterableDataset):
    """Infinitely streams documents from a HuggingFace corpus dataset.

    Wraps around after each full pass.  Use ``datasets.load_dataset`` with
    ``streaming=True`` to avoid downloading the full corpus.

    Expected HF dataset columns: one of ``text``, ``passage``, or ``contents``.
    Override ``text_field`` in config if your corpus uses a different name.
    """

    def __init__(self, dataset_name: str, text_field: str = "text"):
        self.dataset_name = dataset_name
        self.text_field = text_field

    def _iter_dataset(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, split="train", streaming=True)
        for item in ds:
            text = item.get(self.text_field) or item.get("passage") or item.get("contents", "")
            if text:
                yield text

    def __iter__(self) -> Iterator[str]:
        while True:
            yield from self._iter_dataset()


def make_sae_loader(
    dataset_name: str,
    text_field: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Yields tokenised document batches for SAE pretraining."""
    from datasets import load_dataset

    def generate():
        while True:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            batch: List[str] = []
            for item in ds:
                text = item.get(text_field) or item.get("passage") or item.get("contents", "")
                if not text:
                    continue
                batch.append(text)
                if len(batch) == batch_size:
                    enc = tokenizer(
                        batch,
                        max_length=max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    yield {k: v.to(device) for k, v in enc.items()
                           if k in ("input_ids", "attention_mask")}
                    batch = []

    return generate()


# ──────────────────────────────────────────────────────────────────────────────
# SPLADE finetuning: Tevatron hard-negative dataset
# ──────────────────────────────────────────────────────────────────────────────

class TevatronMSMARCODataset(IterableDataset):
    """Streams (query, pos, neg*) triplets from Tevatron/msmarco-passage.

    Each record in the HF dataset has:
      ``query_id``, ``query``, ``positive_passages``, ``negative_passages``
    where passages are ``{"docid": ..., "title": ..., "text": ...}``.
    """

    def __init__(self, nway: int = 8, shuffle_buffer: int = 10_000):
        self.nway = nway          # 1 pos + (nway-1) neg
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[Dict]:
        from datasets import load_dataset
        ds = load_dataset("Tevatron/msmarco-passage", split="train", streaming=True)
        buf: List = []
        for item in ds:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                random.shuffle(buf)
                yield from self._emit(buf)
                buf = []
        if buf:
            random.shuffle(buf)
            yield from self._emit(buf)

    def _emit(self, buf):
        for item in buf:
            query = item["query"]
            pos_list = item.get("positive_passages", [])
            neg_list = item.get("negative_passages", [])
            if not pos_list:
                continue
            pos = random.choice(pos_list)
            negs = neg_list[:self.nway - 1]
            if len(negs) < self.nway - 1:
                continue
            yield {
                "query": query,
                "passages": [pos] + negs,   # first is positive
                "teacher_scores": None,
            }


# ──────────────────────────────────────────────────────────────────────────────
# SPLADE finetuning: ColBERTv2 distillation file
# ──────────────────────────────────────────────────────────────────────────────

class ColBERTDistillationDataset(IterableDataset):
    """Streams training triplets from a ColBERTv2 examples.json file.

    File format: one JSON array per line::

        [qid, [pid, score], [pid, score], ...]

    Requires a pre-built corpus lookup ``{docid: text}`` and query lookup
    ``{qid: text}``.  These are loaded once by :func:`build_corpus_lookup`
    and :func:`build_query_lookup`.
    """

    def __init__(
        self,
        path: str,
        corpus: Dict[str, str],
        queries: Dict[str, str],
        nway: int = 8,
    ):
        self.path = Path(path)
        self.corpus = corpus
        self.queries = queries
        self.nway = nway

    def __iter__(self) -> Iterator[Dict]:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                qid = str(sample[0])
                query = self.queries.get(qid)
                if query is None:
                    continue

                passages, scores = [], []
                # First entry is positive
                for pid, score in sample[1:self.nway + 1]:
                    doc = self.corpus.get(str(pid))
                    if doc is None:
                        break
                    passages.append({"text": doc})
                    scores.append(float(score))
                if len(passages) < self.nway:
                    continue

                yield {
                    "query": query,
                    "passages": passages,
                    "teacher_scores": scores,
                }


def build_corpus_lookup(corpus_dataset: str, text_field: str = "text") -> Dict[str, str]:
    """Load MS MARCO corpus into memory as ``{docid: text}``.

    ~8.8 M passages, expect ~3-4 GB RAM.  Result is pickled to disk so
    subsequent runs load in seconds instead of re-streaming 8.8M records.
    """
    import pickle
    cache_path = Path("data") / (corpus_dataset.replace("/", "__") + ".pkl")
    if cache_path.exists():
        print(f"Loading corpus from cache {cache_path} ...")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    from datasets import load_dataset
    print(f"Loading corpus from {corpus_dataset} into memory ...")
    corpus: Dict[str, str] = {}
    ds = load_dataset(corpus_dataset, split="train", streaming=True)
    for i, item in enumerate(ds):
        pid = str(item.get("docid") or item.get("id") or i)
        text = item.get(text_field) or item.get("passage") or item.get("contents", "")
        corpus[pid] = text
        if (i + 1) % 1_000_000 == 0:
            print(f"  loaded {i+1:,} passages …")
    print(f"Corpus loaded: {len(corpus):,} passages")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Corpus cached → {cache_path}")
    return corpus


def build_query_lookup(queries_dataset: str) -> Dict[str, str]:
    """Load MSMARCO train queries into memory as ``{qid: text}``."""
    from datasets import load_dataset
    print(f"Loading queries from {queries_dataset} …")
    queries: Dict[str, str] = {}
    ds = load_dataset(queries_dataset, split="train")
    for item in ds:
        qid = str(item.get("query_id") or item.get("id"))
        text = item.get("query") or item.get("question", "")
        queries[qid] = text
    print(f"Queries loaded: {len(queries):,}")
    return queries


# ──────────────────────────────────────────────────────────────────────────────
# Alignment warm-up loader
# ──────────────────────────────────────────────────────────────────────────────

def make_ranking_distill_loader(
    nway: int,
    batch_size: int,
) -> Iterator:
    """Yields (query_texts, passage_texts_flat) for ranking distillation.

    Streams Tevatron/msmarco-passage hard negatives. passage_texts_flat has
    length B*nway; the first entry for each query is its positive passage.
    Teacher scores are computed on-the-fly in the training loop.
    """
    def generate():
        batch_q: List[str] = []
        batch_p: List[str] = []
        while True:
            for item in TevatronMSMARCODataset(nway=nway):
                batch_q.append(item["query"])
                batch_p.extend(p.get("text", "") for p in item["passages"])
                if len(batch_q) == batch_size:
                    yield list(batch_q), list(batch_p)
                    batch_q.clear()
                    batch_p.clear()

    return generate()


def make_alignment_loader(
    dataset_name: str,
    text_field: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Iterator:
    """Yields ``(input_ids, attention_mask, texts)`` triples for alignment warm-up.

    Streams corpus text, tokenises with the query tokenizer, and also returns the
    original strings so the doc SPLADE can tokenise them independently.
    """
    from datasets import load_dataset

    def generate():
        while True:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            batch: List[str] = []
            for item in ds:
                text = item.get(text_field) or item.get("passage") or item.get("contents", "")
                if not text:
                    continue
                batch.append(text)
                if len(batch) == batch_size:
                    enc = tokenizer(
                        batch,
                        max_length=max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    yield enc["input_ids"].to(device), enc["attention_mask"].to(device), batch
                    batch = []

    return generate()


# ──────────────────────────────────────────────────────────────────────────────
# Batch collation for SPLADE
# ──────────────────────────────────────────────────────────────────────────────

def collate_asymmetric_batch(
    items: List[Dict],
    query_tokenizer: AutoTokenizer,
    query_max_length: int,
    device: torch.device,
):
    """Collate for asymmetric training: tokenise only queries; return doc texts as strings.

    The doc texts are left un-tokenised so that :class:`FrozenDocSPLADE` can
    handle them with its own tokenizer.

    Returns:
        q_ids, q_mask   — query token tensors ``[B, Lq]``
        doc_texts       — flat list of passage strings ``[B * nway]``
        teacher_scores  — ``[B, nway]`` float tensor or None
    """
    queries = [it["query"] for it in items]
    doc_texts = [p.get("text", "") for it in items for p in it["passages"]]

    q_enc = query_tokenizer(
        queries,
        max_length=query_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    q_ids = q_enc["input_ids"].to(device)
    q_mask = q_enc["attention_mask"].to(device)

    teacher_scores = None
    if items[0]["teacher_scores"] is not None:
        teacher_scores = torch.tensor(
            [it["teacher_scores"] for it in items], dtype=torch.float32, device=device
        )

    return q_ids, q_mask, doc_texts, teacher_scores


def collate_splade_batch(
    items: List[Dict],
    tokenizer: AutoTokenizer,
    query_max_length: int,
    doc_max_length: int,
    device: torch.device,
):
    """Tokenise a list of (query, passages) items and return tensors.

    Returns:
        q_ids, q_mask       — query tokens  ``[B, Lq]``
        d_ids, d_mask       — document tokens ``[B*nway, Ld]``
        teacher_scores      — ``[B, nway]`` float tensor or None
    """
    queries = [it["query"] for it in items]
    # Flatten all passages
    passages = [p.get("text", "") for it in items for p in it["passages"]]

    q_enc = tokenizer(
        queries,
        max_length=query_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    d_enc = tokenizer(
        passages,
        max_length=doc_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    q_ids = q_enc["input_ids"].to(device)
    q_mask = q_enc["attention_mask"].to(device)
    d_ids = d_enc["input_ids"].to(device)
    d_mask = d_enc["attention_mask"].to(device)

    teacher_scores = None
    if items[0]["teacher_scores"] is not None:
        teacher_scores = torch.tensor(
            [it["teacher_scores"] for it in items], dtype=torch.float32, device=device
        )

    return q_ids, q_mask, d_ids, d_mask, teacher_scores
