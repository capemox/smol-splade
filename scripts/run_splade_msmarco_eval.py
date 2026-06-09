#!/usr/bin/env python
"""Evaluate all SPLADE v3 shallow models on full MS-MARCO dev.

Runs 15 layer-sweep variants + 4 loss-ablation variants sequentially against
the pre-built 8.8M-passage index at data/msmarco_index/.

Resumable: skips any variant whose log already ends with a results line.
Results written to results/splade_msmarco_eval.md after each variant.

Usage:
    uv run python3 scripts/run_splade_msmarco_eval.py
    uv run python3 scripts/run_splade_msmarco_eval.py --only first3 last3
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LAYER_SWEEP = [
    "first1", "first2", "first3", "first4", "first5",
    "last1",  "last2",  "last3",  "last4",  "last5",
    "spaced1", "spaced2", "spaced3", "spaced4", "spaced5",
]

VARIANTS = (
    [
        {
            "name": v,
            "group": "layer_sweep",
            "checkpoint": f"checkpoints_splade-v3/splade_shallow_{v}/best_NanoMSMARCO.pt",
        }
        for v in LAYER_SWEEP
    ]
    + [
        {
            "name": "first3_loss_mse",
            "group": "loss_ablation",
            "checkpoint": "checkpoints_splade-v3/splade_shallow_first3_loss_mse/best_NanoMSMARCO.pt",
        },
        {
            "name": "first3_loss_cosine",
            "group": "loss_ablation",
            "checkpoint": "checkpoints_splade-v3/splade_shallow_first3_loss_cosine/best_NanoMSMARCO.pt",
        },
        {
            "name": "first3_loss_kd_colbert",
            "group": "loss_ablation",
            "checkpoint": "checkpoints_splade-v3/splade_shallow_first3_loss_kd_colbert/best_NanoMSMARCO.pt",
        },
        {
            "name": "first3_loss_margin_mse_colbert",
            "group": "loss_ablation",
            "checkpoint": "checkpoints_splade-v3/splade_shallow_first3_loss_margin_mse_colbert/best_NanoMSMARCO.pt",
        },
    ]
)

NDCG_PAT = re.compile(r"NDCG@10\s*:\s*([0-9.]+)")
MRR_PAT  = re.compile(r"MRR@10\s*:\s*([0-9.]+)")


def is_done(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text()
    return bool(NDCG_PAT.search(text) and MRR_PAT.search(text))


def parse_results(log_path: Path) -> tuple[float, float] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m_n = NDCG_PAT.search(text)
    m_m = MRR_PAT.search(text)
    if m_n and m_m:
        return float(m_n.group(1)), float(m_m.group(1))
    return None


def run_eval(variant: dict, log_path: Path, config: str) -> int:
    cmd = [
        sys.executable, "scripts/eval_msmarco.py",
        "--stage", "splade_shallow",
        "--checkpoint", variant["checkpoint"],
        "--config", config,
        "--index_dir", "data/msmarco_index",
    ]
    print(f"\n{'='*60}")
    print(f"[{variant['name']}] {' '.join(cmd)}")
    print(f"  log → {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return proc.wait()


def write_report(results_path: Path, rows: list[dict]) -> None:
    layer_rows = [r for r in rows if r["group"] == "layer_sweep"]
    loss_rows  = [r for r in rows if r["group"] == "loss_ablation"]

    lines = [
        "# SPLADE v3 Shallow Models — MS-MARCO Dev Evaluation",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Index: `data/msmarco_index/` (8,841,823 passages, `naver/splade-v3` doc encoder)",
        "- Query models: `best_NanoMSMARCO.pt` checkpoint from each training run",
        "- Metrics: NDCG@10 and MRR@10 on MS-MARCO dev (~6,980 queries)",
        "",
        "## Layer Sweep",
        "",
        "| Variant | NDCG@10 | MRR@10 |",
        "|---|---:|---:|",
    ]
    for r in layer_rows:
        res = r.get("results")
        if res:
            lines.append(f"| {r['name']} | {res[0]:.4f} | {res[1]:.4f} |")
        else:
            lines.append(f"| {r['name']} | — | — |")

    lines += [
        "",
        "## Loss Ablation (first3)",
        "",
        "| Variant | NDCG@10 | MRR@10 |",
        "|---|---:|---:|",
    ]
    for r in loss_rows:
        res = r.get("results")
        if res:
            lines.append(f"| {r['name']} | {res[0]:.4f} | {res[1]:.4f} |")
        else:
            lines.append(f"| {r['name']} | — | — |")

    lines.append("")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {results_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--work_dir", default="runs/splade_msmarco_eval")
    parser.add_argument("--results", default="results/splade_msmarco_eval.md")
    parser.add_argument("--only", nargs="+", default=None, metavar="NAME")
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results

    variants = VARIANTS
    if args.only:
        names = set(args.only)
        variants = [v for v in VARIANTS if v["name"] in names]
        if not variants:
            raise SystemExit(f"No matching variants for: {args.only}")

    rows = []
    for variant in variants:
        log_path = work_dir / f"{variant['name']}.log"
        row = dict(variant, results=None)

        if is_done(log_path):
            print(f"[skip] {variant['name']}: already complete")
            row["results"] = parse_results(log_path)
        else:
            ckpt = root / variant["checkpoint"]
            if not ckpt.exists():
                print(f"[WARN] Missing checkpoint: {ckpt} — skipping", file=sys.stderr)
                rows.append(row)
                continue
            t0 = time.time()
            code = run_eval(variant, log_path, args.config)
            elapsed = time.time() - t0
            if code != 0:
                print(f"[ERROR] {variant['name']} exited {code}", file=sys.stderr)
                rows.append(row)
                write_report(results_path, rows)
                return code
            row["results"] = parse_results(log_path)
            print(f"  [{variant['name']}] done in {elapsed/60:.1f} min → {row['results']}")

        rows.append(row)
        write_report(results_path, rows)

    print(f"\nAll done. Results at {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
