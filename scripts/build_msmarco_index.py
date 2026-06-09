#!/usr/bin/env python
"""Build a sparse document index for the full MSMARCO passage corpus.

Streams Tevatron/msmarco-passage-corpus, encodes each passage with the frozen
doc SPLADE (bf16 autocast on GPU), keeps only nonzero terms, and writes
CSR-format shards to ``data/msmarco_index/``.

The index is checkpoint-independent: it depends only on the frozen document
encoder specified by the selected config section. Any number of
query checkpoints can be evaluated against the same index without re-encoding.

Resumable: if the script is interrupted, re-running picks up at the next shard
boundary. A manifest at ``data/msmarco_index/manifest.json`` records the model
config so eval can sanity-check compatibility.

Usage:
    uv run scripts/build_msmarco_index.py
    uv run scripts/build_msmarco_index.py --batch_size 64 --shard_size 50000
    uv run scripts/build_msmarco_index.py --limit 10000   # smoke test
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

CORPUS_SIZE = 8_841_823  # known size of Tevatron/msmarco-passage-corpus


def iter_corpus_from_pkl(pkl_path: Path, text_field: str, skip: int):
    """Yield (text, docid) from a local corpus pickle.

    Pickle is expected to be ``dict[docid_str -> text_str]`` (Tevatron format
    used elsewhere in this repo). Insertion order is preserved by Python 3.7+
    dicts and matches the original streaming order, so resume positions remain
    consistent across pickle and streaming sources.
    """
    import pickle
    size_gb = pkl_path.stat().st_size / 1e9
    print(f"Loading corpus pickle {pkl_path} ({size_gb:.1f} GB) ...")
    with pkl_path.open("rb") as f:
        corpus = pickle.load(f)
    print(f"  Loaded {len(corpus):,} passages into RAM")
    items = iter(corpus.items())
    for _ in range(skip):
        next(items, None)
    for did, val in items:
        if isinstance(val, str):
            text = val
        elif isinstance(val, dict):
            text = val.get(text_field) or val.get("passage") or val.get("contents", "")
        else:
            text = str(val)
        yield text, str(did)


def iter_corpus_from_stream(corpus_dataset: str, text_field: str, skip: int):
    """Yield (text, docid) from the streaming HF dataset (online fallback)."""
    from datasets import load_dataset
    ds = load_dataset(corpus_dataset, split="train", streaming=True)
    if skip > 0:
        ds = ds.skip(skip)
    for i, item in enumerate(ds):
        text = item.get(text_field) or item.get("passage") or item.get("contents", "")
        did = str(item.get("docid", str(skip + i)))
        yield text, did


def shard_path(index_dir: Path, shard_idx: int) -> Path:
    return index_dir / f"shard_{shard_idx:05d}.npz"


def write_shard(
    index_dir: Path,
    shard_idx: int,
    docids: list,
    indices_chunks: list,
    values_chunks: list,
    nnz_per_doc: list,
) -> None:
    """Write one CSR shard to disk."""
    indices = np.concatenate(indices_chunks).astype(np.int32, copy=False) if indices_chunks else np.empty(0, dtype=np.int32)
    values = np.concatenate(values_chunks).astype(np.float16, copy=False) if values_chunks else np.empty(0, dtype=np.float16)
    offsets = np.concatenate([[0], np.cumsum(nnz_per_doc)]).astype(np.int64, copy=False)
    docids_arr = np.array(docids, dtype=object)
    np.savez(
        shard_path(index_dir, shard_idx),
        indices=indices,
        values=values,
        offsets=offsets,
        docids=docids_arr,
    )


def scan_existing_shards(index_dir: Path, shard_size: int) -> tuple:
    """Return (next_shard_idx, docs_already_processed) by scanning shard files.

    Drops any short shard mid-sequence (interrupted run) so we resume cleanly.
    """
    next_idx = 0
    docs_done = 0
    while shard_path(index_dir, next_idx).exists():
        with np.load(shard_path(index_dir, next_idx), allow_pickle=True) as z:
            n = len(z["docids"])
        if n != shard_size:
            shard_path(index_dir, next_idx).unlink()
            return next_idx, docs_done
        docs_done += n
        next_idx += 1
    return next_idx, docs_done


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage",
        default="splade_shallow",
        choices=["splade_shallow", "lion_shallow"],
        help="Config section whose frozen document encoder should build the index",
    )
    parser.add_argument(
        "--index_dir", default="data/msmarco_index",
        help="Directory to write CSR shards into",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Passages encoded per GPU forward (8GB VRAM: 64 with bf16 is comfortable)",
    )
    parser.add_argument(
        "--shard_size", type=int, default=50_000,
        help="Documents per shard file",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Stop after this many docs (0 = full corpus). For smoke tests.",
    )
    parser.add_argument(
        "--no_bf16", action="store_true",
        help="Disable bf16 autocast (fall back to fp32 — needs more VRAM)",
    )
    parser.add_argument(
        "--corpus_pkl", default=None,
        help="Path to local corpus pickle (dict[docid->text]). If unset, uses "
             "data/<corpus_dataset_underscored>.pkl when present, else streams "
             "from HF Hub.",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sc = cfg[args.stage]
    corpus_dataset = cfg.get("data", {}).get("corpus_dataset", "Tevatron/msmarco-passage-corpus")
    text_field = cfg.get("data", {}).get("corpus_text_field", "text")

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
    vocab_size = doc_splade.vocab_size
    doc_max_length = int(sc["doc_max_length"])

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Manifest carries config used to build the index. Eval cross-checks this
    # so a stale index (built against a different doc model) is rejected.
    manifest_path = index_dir / "manifest.json"
    target_total = args.limit if args.limit > 0 else CORPUS_SIZE
    manifest = {
        "doc_splade_hf_id": doc_hf_id,
        "doc_max_length": doc_max_length,
        "vocab_size": int(vocab_size),
        "shard_size": int(args.shard_size),
        "corpus_dataset": corpus_dataset,
        "text_field": text_field,
        "target_total_docs": int(target_total),
    }
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        for key in ("doc_splade_hf_id", "doc_max_length", "vocab_size", "corpus_dataset"):
            if prev.get(key) != manifest[key]:
                raise SystemExit(
                    f"Existing manifest at {manifest_path} disagrees on '{key}': "
                    f"have {prev.get(key)!r}, want {manifest[key]!r}. "
                    f"Delete {index_dir} to rebuild from scratch."
                )
        # Carry forward shard_size from disk if shards already exist; we can't
        # change shard size mid-build.
        if prev.get("shard_size") and (index_dir / "shard_00000.npz").exists():
            args.shard_size = int(prev["shard_size"])
            manifest["shard_size"] = args.shard_size
    manifest_path.write_text(json.dumps(manifest, indent=2))

    next_shard_idx, docs_done = scan_existing_shards(index_dir, args.shard_size)
    if docs_done > 0:
        print(f"Resuming: {docs_done:,} docs already in {next_shard_idx} shards")

    if docs_done >= target_total:
        print(f"Index already complete ({docs_done:,} >= {target_total:,}). Nothing to do.")
        return

    # ── Resolve corpus source (offline pickle preferred over streaming) ──────
    from tqdm import tqdm

    default_pkl = Path("data") / (corpus_dataset.replace("/", "__") + ".pkl")
    if args.corpus_pkl:
        pkl_path = Path(args.corpus_pkl)
        if not pkl_path.exists():
            raise SystemExit(f"--corpus_pkl {pkl_path} does not exist")
    elif default_pkl.exists():
        pkl_path = default_pkl
    else:
        pkl_path = None

    if pkl_path is not None:
        print(f"Source: pickle {pkl_path} (offline)")
        source = iter_corpus_from_pkl(pkl_path, text_field, skip=docs_done)
    else:
        print(f"Source: streaming {corpus_dataset} from doc {docs_done:,}")
        source = iter_corpus_from_stream(corpus_dataset, text_field, skip=docs_done)

    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    shard_idx = next_shard_idx
    shard_docids: list = []
    shard_indices: list = []
    shard_values: list = []
    shard_nnz: list = []

    buf_texts: list = []
    buf_docids: list = []

    pbar = tqdm(
        total=target_total - docs_done,
        unit="doc", unit_scale=True, dynamic_ncols=True,
        desc="encoding",
    )
    total_nnz = 0
    t0 = time.time()

    def flush_encode():
        nonlocal total_nnz
        if not buf_texts:
            return
        with torch.no_grad(), autocast_ctx:
            vecs = doc_splade.encode(buf_texts, doc_max_length, no_grad=True)
        # bf16/fp16 → fp32 on CPU for sparsification; values re-cast to fp16 on save.
        vecs = vecs.float().cpu()
        for j, did in enumerate(buf_docids):
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

    def flush_shard_if_full():
        nonlocal shard_idx, shard_docids, shard_indices, shard_values, shard_nnz
        if len(shard_docids) >= args.shard_size:
            write_shard(
                index_dir, shard_idx,
                shard_docids, shard_indices, shard_values, shard_nnz,
            )
            shard_idx += 1
            shard_docids = []
            shard_indices = []
            shard_values = []
            shard_nnz = []

    seen = 0
    for text, did in source:
        if seen + docs_done >= target_total:
            break
        buf_texts.append(text)
        buf_docids.append(did)
        seen += 1

        if len(buf_texts) >= args.batch_size:
            flush_encode()
            buf_texts.clear()
            buf_docids.clear()
            pbar.update(args.batch_size)
            flush_shard_if_full()

    if buf_texts:
        flush_encode()
        pbar.update(len(buf_texts))
        buf_texts.clear()
        buf_docids.clear()

    # Final partial shard (legitimate at end-of-corpus or --limit boundary)
    if shard_docids:
        write_shard(
            index_dir, shard_idx,
            shard_docids, shard_indices, shard_values, shard_nnz,
        )
        shard_idx += 1

    pbar.close()
    elapsed = time.time() - t0
    final_total = docs_done + seen
    avg_nnz = (total_nnz / max(seen, 1)) if seen else 0.0
    print(
        f"Done. {final_total:,} docs across {shard_idx} shards. "
        f"Avg {avg_nnz:.1f} nonzero terms/doc. Build took {elapsed/60:.1f} min."
    )


if __name__ == "__main__":
    main()
