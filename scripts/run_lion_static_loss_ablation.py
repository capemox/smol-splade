#!/usr/bin/env python
"""Run loss-function ablations for the Lion static (0-layer) query encoder.

Variants: mse | cosine | kd | margin_mse
Writes results to results/lion_static_loss_ablation.md after each variant.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


DEFAULT_DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]

BASELINE_BEIR = {"ndcg@10": None, "mrr@10": None}  # filled in once mse runs

VARIANTS = [
    {
        "name": "mse",
        "description": "Vector MSE against full Lion teacher query vectors (baseline)",
        "align_loss_kind": "mse",
    },
    {
        "name": "cosine",
        "description": "Cosine distance against full Lion teacher query vectors",
        "align_loss_kind": "cosine",
    },
    {
        "name": "kd",
        "description": "KL distillation over nway passage scores from ColBERT supervision",
        "align_loss_kind": "kd",
    },
    {
        "name": "margin_mse",
        "description": "MarginMSE on positive-negative passage score margins from ColBERT",
        "align_loss_kind": "margin_mse",
    },
]


def run_command(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    print(" ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("$ " + " ".join(cmd) + "\n")
        return 0
    with log_path.open("w") as log:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def variant_config(base_cfg: dict, variant: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    sc = cfg["lion_static"]
    sc["align_loss_kind"] = variant["align_loss_kind"]
    sc["output_dir"] = f"lion_static_loss_{variant['name']}"
    if variant["align_loss_kind"] == "kd":
        sc["kd_temperature"] = float(sc.get("kd_temperature", 1.0))
    if variant["align_loss_kind"] in {"kd", "margin_mse"}:
        sc["batch_size"] = 16
        sc["doc_enc_batch_size"] = 16  # keep per-chunk allocation below fragmentation threshold
        sc["distil_data_path"] = sc.get("distil_data_path", "data/colbertv2_msmarco_64way.json")
        sc["corpus_dataset"] = sc.get("corpus_dataset", "Tevatron/msmarco-passage-corpus")
        sc["corpus_text_field"] = sc.get("corpus_text_field", "text")
        sc["queries_dataset"] = sc.get("queries_dataset", "Tevatron/msmarco-passage")
    return cfg


def parse_beir_summary(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^([a-zA-Z0-9_-]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        dataset, ndcg, mrr = m.groups()
        if dataset.lower() == "average":
            dataset = "Average"
        rows[dataset] = {"ndcg@10": float(ndcg), "mrr@10": float(mrr)}
    return rows


def parse_nanoms_best(text: str) -> dict | None:
    best = None
    current_step = None
    step_pat = re.compile(r"^\[lion-static\] Eval at step (\d+)")
    score_pat = re.compile(
        r"^\s*\[NanoMSMARCO\] NDCG@10\s+query_doc \(dense\)\s+:\s+([0-9.]+)"
    )
    for line in text.splitlines():
        sm = step_pat.match(line)
        if sm:
            current_step = int(sm.group(1))
            continue
        sc = score_pat.match(line)
        if sc and current_step is not None:
            score = float(sc.group(1))
            if best is None or score > best["score"]:
                best = {"step": current_step, "score": score}
    return best


def metric_delta(row: dict, key: str, baseline: dict[str, float | None]) -> str:
    if baseline.get(key) is None:
        return "n/a"
    average = (row.get("metrics") or {}).get("Average")
    if not average:
        return "n/a"
    return f"{average[key] - baseline[key]:+.4f}"


def write_report(path: Path, datasets: list[str], rows: list[dict], baseline: dict) -> None:
    baseline_str = (
        f"`{baseline['ndcg@10']:.4f}/{baseline['mrr@10']:.4f}`"
        if baseline.get("ndcg@10") is not None
        else "pending"
    )
    lines = [
        "# Lion Static Loss Ablation",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Architecture: static (0-layer), one learnable weight per Lion vocabulary token",
        "- Model: `hzeng/Lion-SP-1B-llama3-marco-mntp` (frozen doc encoder as teacher)",
        "- Training steps: 10,000",
        "- ColBERT score supervision: `data/colbertv2_msmarco_64way.json` for kd/margin_mse",
        f"- mse baseline BEIR avg NDCG@10/MRR@10: {baseline_str}",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        "",
        "## Summary",
        "",
        "| Variant | Loss | Best NanoMSMARCO | BEIR avg NDCG@10/MRR@10 | Delta NDCG | Output dir |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms else "n/a"
        )
        average = (row.get("metrics") or {}).get("Average")
        average_text = (
            f"{average['ndcg@10']:.4f}/{average['mrr@10']:.4f}" if average else "n/a"
        )
        lines.append(
            f"| {row['name']} | `{row['align_loss_kind']}` | {best_text} | "
            f"{average_text} | {metric_delta(row, 'ndcg@10', baseline)} | `{row['output_dir']}` |"
        )

    lines.extend(["", "## Details", ""])
    for row in rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms else "n/a"
        )
        lines.extend([
            f"### {row['name']}",
            "",
            f"- Description: {row['description']}",
            f"- Loss: `{row['align_loss_kind']}`",
            f"- Best NanoMSMARCO NDCG@10: `{best_text}`",
            f"- Checkpoint: `{row['checkpoint']}`",
            f"- Train log: `{row['train_log']}`",
            f"- BEIR log: `{row['eval_log']}`",
            "",
        ])
        if row.get("metrics"):
            lines.extend(["| Dataset | NDCG@10 | MRR@10 |", "|---|---:|---:|"])
            for dataset, metrics in row["metrics"].items():
                lines.append(f"| {dataset} | {metrics['ndcg@10']:.4f} | {metrics['mrr@10']:.4f} |")
        else:
            lines.append("_No BEIR metrics recorded yet._")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--work-dir", default="runs/lion_static_loss_ablation")
    parser.add_argument("--results", default="results/lion_static_loss_ablation.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    base_cfg = yaml.safe_load(Path(args.config).read_text())

    variants = VARIANTS
    if args.only:
        requested = set(args.only)
        known = {v["name"] for v in VARIANTS}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"Unknown variants: {', '.join(sorted(unknown))}")
        variants = [v for v in VARIANTS if v["name"] in requested]

    baseline: dict[str, float | None] = {"ndcg@10": None, "mrr@10": None}
    rows: list[dict] = []

    for variant in variants:
        cfg = variant_config(base_cfg, variant)
        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        output_dir = cfg["lion_static"]["output_dir"]
        train_log  = work_dir / f"{variant['name']}_train.log"
        eval_log   = work_dir / f"{variant['name']}_beir.log"
        checkpoint = root / "checkpoints_lion_static" / output_dir / "best_NanoMSMARCO.pt"

        row = {
            "name": variant["name"],
            "description": variant["description"],
            "align_loss_kind": variant["align_loss_kind"],
            "output_dir": output_dir,
            "checkpoint": checkpoint,
            "train_log": train_log,
            "eval_log": eval_log,
            "best_nanoms": None,
            "metrics": {},
        }

        if not args.skip_train:
            if checkpoint.exists() and eval_log.exists() and "Average" in eval_log.read_text():
                print(f"[skip] {variant['name']}: checkpoint + eval already exist")
            else:
                code = run_command(
                    [sys.executable, "train.py", "lion_static", "--config", str(cfg_path)],
                    train_log,
                    args.dry_run,
                )
                if train_log.exists():
                    row["best_nanoms"] = parse_nanoms_best(train_log.read_text())
                if code != 0:
                    rows.append(row)
                    write_report(results_path, args.datasets, rows, baseline)
                    return code
        if train_log.exists():
            row["best_nanoms"] = parse_nanoms_best(train_log.read_text())

        if not args.skip_eval:
            if eval_log.exists() and "Average" in eval_log.read_text():
                print(f"[skip] {variant['name']}: BEIR eval already complete")
            else:
                if not checkpoint.exists() and not args.dry_run:
                    print(f"Missing checkpoint: {checkpoint}", file=sys.stderr)
                    rows.append(row)
                    write_report(results_path, args.datasets, rows, baseline)
                    return 1
                code = run_command(
                    [
                        sys.executable, "scripts/eval_beir.py",
                        "--stage", "lion_static",
                        "--config", str(cfg_path),
                        "--checkpoint", str(checkpoint),
                        "--doc_batch_size", "16",
                        "--datasets", *args.datasets,
                    ],
                    eval_log,
                    args.dry_run,
                )
                if code != 0:
                    if eval_log.exists():
                        row["metrics"] = parse_beir_summary(eval_log.read_text())
                    rows.append(row)
                    write_report(results_path, args.datasets, rows, baseline)
                    return code
        if eval_log.exists():
            row["metrics"] = parse_beir_summary(eval_log.read_text())

        # Use mse result as baseline for delta column
        if variant["name"] == "mse":
            avg = (row.get("metrics") or {}).get("Average")
            if avg:
                baseline = {"ndcg@10": avg["ndcg@10"], "mrr@10": avg["mrr@10"]}

        rows.append(row)
        write_report(results_path, args.datasets, rows, baseline)

    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
