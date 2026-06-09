#!/usr/bin/env python
"""Run BEIR eval for SPLADE v3 spaced2, spaced4, first5 across all 13 datasets.

Results written to results/beir_splade_extra_eval.md after each variant.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

DATASETS = [
    "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
    "trec-covid", "webis-touche2020", "quora",
    "nq", "dbpedia-entity", "hotpotqa", "fever", "climate-fever",
]

VARIANTS = [
    {
        "name": "spaced2",
        "layers": "[0, 11]",
        "checkpoint": "checkpoints_splade-v3/splade_shallow_spaced2/best_NanoMSMARCO.pt",
    },
    {
        "name": "spaced4",
        "layers": "[0, 4, 7, 11]",
        "checkpoint": "checkpoints_splade-v3/splade_shallow_spaced4/best_NanoMSMARCO.pt",
    },
    {
        "name": "first5",
        "layers": "[0, 1, 2, 3, 4]",
        "checkpoint": "checkpoints_splade-v3/splade_shallow_first5/best_NanoMSMARCO.pt",
    },
]

DS_HDR_PAT = re.compile(r"^\[([^\]]+)\]\s*$")
SCORE_PAT  = re.compile(r"NDCG@10:\s*([0-9.]+)\s+MRR@10:\s*([0-9.]+)")


def _parse_log(text: str) -> dict[str, tuple[float, float]]:
    parsed: dict[str, list] = {}
    current_ds = None
    for line in text.splitlines():
        m_hdr = DS_HDR_PAT.match(line.strip())
        if m_hdr:
            current_ds = m_hdr.group(1)
            continue
        if current_ds:
            m_sc = SCORE_PAT.search(line)
            if m_sc:
                parsed[current_ds] = (float(m_sc.group(1)), float(m_sc.group(2)))
                current_ds = None
    return parsed


def run_variant(variant: dict, log_path: Path) -> dict[str, tuple[float, float]]:
    cmd = [
        sys.executable, "scripts/eval_beir.py",
        "--stage", "splade_shallow",
        "--config", "config.yaml",
        "--index_dir", "data/beir_index",
        "--datasets", *DATASETS,
        "--encode_batch_size", "64",
        "--densify_chunk", "4096",
        "--checkpoint", variant["checkpoint"],
    ]

    print(f"\n{'='*60}")
    print(f"[{variant['name']}] layers={variant['layers']}")
    print(f"  cmd: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    with log_path.open("w") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        proc.wait()

    return _parse_log(log_path.read_bytes().decode("utf-8", errors="ignore"))


def write_report(path: Path, all_results: dict[str, dict]) -> None:
    variants_done = list(all_results.keys())
    lines = [
        "# SPLADE v3 Shallow — BEIR Evaluation (spaced2, spaced4, first5)",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Model: `naver/splade-v3`",
        "- Query encoders: spaced2/4 and first5 layers (best_NanoMSMARCO checkpoint)",
        "- Doc indexes: pre-built with `naver/splade-v3` doc encoder, all 13 standard BEIR datasets",
        "- Metric: NDCG@10 (primary), MRR@10",
        "",
        "## NDCG@10 per Dataset",
        "",
        "| Dataset |" + "".join(f" {v} |" for v in variants_done),
        "|---|" + "---:|" * len(variants_done),
    ]
    for ds in DATASETS:
        row = f"| {ds} |"
        for v in variants_done:
            res = all_results[v].get(ds)
            row += f" {res[0]:.4f} |" if res else " — |"
        lines.append(row)
    avg_row = "| **Average** |"
    for v in variants_done:
        s = [all_results[v][ds][0] for ds in DATASETS if ds in all_results[v]]
        avg_row += f" **{sum(s)/len(s):.4f}** |" if s else " — |"
    lines.append(avg_row)

    lines += ["", "## MRR@10 per Dataset", "",
              "| Dataset |" + "".join(f" {v} |" for v in variants_done),
              "|---|" + "---:|" * len(variants_done)]
    for ds in DATASETS:
        row = f"| {ds} |"
        for v in variants_done:
            res = all_results[v].get(ds)
            row += f" {res[1]:.4f} |" if res else " — |"
        lines.append(row)
    avg_row2 = "| **Average** |"
    for v in variants_done:
        s = [all_results[v][ds][1] for ds in DATASETS if ds in all_results[v]]
        avg_row2 += f" **{sum(s)/len(s):.4f}** |" if s else " — |"
    lines.append(avg_row2)

    lines += ["", "## Variant Details", ""]
    for v in VARIANTS:
        if v["name"] in variants_done:
            lines.append(f"- **{v['name']}**: layers `{v['layers']}`, checkpoint: `{v['checkpoint']}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


def main() -> int:
    root = Path.cwd()
    log_dir = root / "runs/beir_splade_extra"
    results_path = root / "results/beir_splade_extra_eval.md"

    all_results: dict[str, dict] = {}
    for variant in VARIANTS:
        log_path = log_dir / f"{variant['name']}.log"
        t0 = time.time()
        results = run_variant(variant, log_path)
        elapsed = (time.time() - t0) / 60
        print(f"  done in {elapsed:.1f} min, got {len(results)} datasets")
        all_results[variant["name"]] = results
        write_report(results_path, all_results)

    print(f"\nAll done. Results at {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
