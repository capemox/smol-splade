#!/usr/bin/env python
"""Build sparse SPLADE doc indexes for BEIR datasets.

Encodes each corpus with the frozen doc SPLADE (bf16 autocast on GPU) and
writes CSR-format shards to ``data/beir_index/<dataset>/``, using the same
format as the MSMARCO index so eval_beir.py can reuse the same scorer.

Building once lets you evaluate multiple query checkpoints without re-encoding.
Resumable: re-running picks up at the next shard boundary.

Usage:
    uv run scripts/build_beir_index.py
    uv run scripts/build_beir_index.py --datasets nfcorpus scifact fiqa
    uv run scripts/build_beir_index.py --datasets quora --batch_size 128
    uv run scripts/build_beir_index.py --limit 500   # smoke test
"""

import argparse
import contextlib
import json
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


def manifest_key_val(doc_splade_hf_id, doc_max_length, vocab_size, key):
    return {"doc_splade_hf_id": doc_splade_hf_id,
            "doc_max_length": doc_max_length,
            "vocab_size": int(vocab_size)}[key]


def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return (title + " " + text).strip() if title else text


def _shard_path(index_dir: Path, shard_idx: int) -> Path:
    return index_dir / f"shard_{shard_idx:05d}.npz"


def _write_shard(index_dir, shard_idx, docids, indices_chunks, values_chunks, nnz_per_doc):
    indices = np.concatenate(indices_chunks).astype(np.int32, copy=False) if indices_chunks else np.empty(0, dtype=np.int32)
    values  = np.concatenate(values_chunks).astype(np.float16, copy=False) if values_chunks else np.empty(0, dtype=np.float16)
    offsets = np.concatenate([[0], np.cumsum(nnz_per_doc)]).astype(np.int64, copy=False)
    np.savez(
        _shard_path(index_dir, shard_idx),
        indices=indices, values=values, offsets=offsets,
        docids=np.array(docids, dtype=object),
    )


def _scan_existing_shards(index_dir: Path, shard_size: int) -> tuple:
    next_idx = 0
    docs_done = 0
    while _shard_path(index_dir, next_idx).exists():
        with np.load(_shard_path(index_dir, next_idx), allow_pickle=True) as z:
            n = len(z["docids"])
        if n != shard_size:
            _shard_path(index_dir, next_idx).unlink()
            return next_idx, docs_done
        docs_done += n
        next_idx += 1
    return next_idx, docs_done


def build_dataset_index(
    name: str,
    doc_splade_hf_id: str,
    doc_splade,
    doc_max_length: int,
    index_dir: Path,
    batch_size: int,
    shard_size: int,
    use_bf16: bool,
    limit: int,
):
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"[{name}]")

    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    vocab_size = doc_splade.vocab_size

    # ── Fast-path: skip corpus download if index is already complete ──────────
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        for key in ("doc_splade_hf_id", "doc_max_length", "vocab_size"):
            if prev.get(key) != manifest_key_val(doc_splade_hf_id, doc_max_length, vocab_size, key):
                raise SystemExit(
                    f"  Manifest mismatch on '{key}': have {prev[key]!r}, "
                    f"want {manifest_key_val(doc_splade_hf_id, doc_max_length, vocab_size, key)!r}. "
                    f"Delete {index_dir} to rebuild."
                )
        prev_shard_size = int(prev.get("shard_size", shard_size))
        if _shard_path(index_dir, 0).exists():
            shard_size = prev_shard_size
        prev_target = int(prev.get("target_total_docs", 0))
        if prev_target > 0:
            next_shard_idx, docs_done = _scan_existing_shards(index_dir, shard_size)
            effective_target = min(limit, prev_target) if limit > 0 else prev_target
            if docs_done >= effective_target:
                print(f"  Index complete ({docs_done:,} docs). Skipping.")
                return

    # ── Load corpus (only reached when index is absent or incomplete) ──────────
    print(f"  Loading corpus from BeIR/{name} ...")
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    corpus_ids   = [str(row["_id"]) for row in corpus_ds]
    corpus_texts = [_corpus_text(row) for row in corpus_ds]
    full_size = len(corpus_ids)
    target_total = min(limit, full_size) if limit > 0 else full_size
    print(f"  {full_size:,} documents{f' (limited to {target_total:,})' if limit > 0 else ''}")

    manifest = {
        "dataset": name,
        "doc_splade_hf_id": doc_splade_hf_id,
        "doc_max_length": doc_max_length,
        "vocab_size": int(vocab_size),
        "shard_size": shard_size,
        "target_total_docs": target_total,
    }
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        if prev.get("shard_size") and _shard_path(index_dir, 0).exists():
            shard_size = int(prev["shard_size"])
            manifest["shard_size"] = shard_size
    manifest_path.write_text(json.dumps(manifest, indent=2))

    next_shard_idx, docs_done = _scan_existing_shards(index_dir, shard_size)
    if docs_done >= target_total:
        print(f"  Index complete ({docs_done:,} docs). Skipping.")
        return
    if docs_done > 0:
        print(f"  Resuming from doc {docs_done:,} (shard {next_shard_idx})")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    shard_idx = next_shard_idx
    shard_docids:  list = []
    shard_indices: list = []
    shard_values:  list = []
    shard_nnz:     list = []
    total_nnz = 0

    pbar = tqdm(total=target_total - docs_done, unit="doc", unit_scale=True,
                dynamic_ncols=True, desc=f"  {name}")
    t0 = time.time()

    for batch_start in range(docs_done, target_total, batch_size):
        batch_end = min(batch_start + batch_size, target_total)
        texts = corpus_texts[batch_start:batch_end]
        dids  = corpus_ids[batch_start:batch_end]

        with torch.no_grad(), autocast_ctx:
            vecs = doc_splade.encode(texts, doc_max_length, no_grad=True).float().cpu()

        for j, did in enumerate(dids):
            row = vecs[j]
            nz = torch.nonzero(row, as_tuple=False).squeeze(1)
            if nz.numel() == 0:
                shard_indices.append(np.empty(0, dtype=np.int32))
                shard_values.append(np.empty(0, dtype=np.float16))
                shard_nnz.append(0)
            else:
                shard_indices.append(nz.numpy().astype(np.int32, copy=False))
                shard_values.append(row[nz].numpy().astype(np.float16, copy=False))
                shard_nnz.append(int(nz.numel()))
                total_nnz += int(nz.numel())
            shard_docids.append(did)

            if len(shard_docids) >= shard_size:
                _write_shard(index_dir, shard_idx, shard_docids,
                             shard_indices, shard_values, shard_nnz)
                shard_idx += 1
                shard_docids, shard_indices, shard_values, shard_nnz = [], [], [], []

        pbar.update(len(texts))

    if shard_docids:
        _write_shard(index_dir, shard_idx, shard_docids,
                     shard_indices, shard_values, shard_nnz)

    pbar.close()
    seen = target_total - docs_done
    elapsed = time.time() - t0
    avg_nnz = total_nnz / max(seen, 1)
    print(
        f"  Done. {target_total:,} docs indexed. "
        f"Avg {avg_nnz:.1f} nonzero terms/doc. {elapsed/60:.1f} min."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage",
        default="splade_shallow",
        choices=["splade_shallow", "lion_shallow"],
        help="Config section whose frozen doc encoder builds the index",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="BEIR dataset names (loaded as BeIR/<name> from HuggingFace)",
    )
    parser.add_argument("--index_dir", default="data/beir_index",
                        help="Base directory; each dataset gets its own subdirectory")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Passages per GPU forward pass")
    parser.add_argument("--shard_size", type=int, default=50_000,
                        help="Documents per shard file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many docs per dataset (0 = full). For smoke tests.")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 autocast (fall back to fp32)")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    sc = cfg[args.stage]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.stage.startswith("lion_"):
        from model import FrozenLionSPLADE
        doc_hf_id = sc["lion_hf_id"]
        print(f"Loading frozen Lion SPLADE: {doc_hf_id} ...")
        doc_splade = FrozenLionSPLADE(doc_hf_id)
    else:
        from model import FrozenDocSPLADE
        doc_hf_id = sc["doc_splade_hf_id"]
        print(f"Loading frozen doc SPLADE: {doc_hf_id} ...")
        doc_splade = FrozenDocSPLADE(doc_hf_id)
    doc_splade.to(device)
    doc_splade.eval()

    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    base_index_dir = Path(args.index_dir)

    for name in args.datasets:
        build_dataset_index(
            name=name,
            doc_splade_hf_id=doc_hf_id,
            doc_splade=doc_splade,
            doc_max_length=sc["doc_max_length"],
            index_dir=base_index_dir / args.stage / name,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            use_bf16=use_bf16,
            limit=args.limit,
        )

    print(f"\nAll done. Indexes written to {base_index_dir}/")


if __name__ == "__main__":
    main()
