#!/usr/bin/env python
"""Run SPLADE v3 first-3-layer loss-function ablations."""

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
BASELINE_BEIR = {"ndcg@10": 0.3483, "mrr@10": 0.4050}

VARIANTS = [
    {
        "name": "mse",
        "description": "Current vector MSE against full SPLADE query vectors",
        "align_loss_kind": "mse",
    },
    {
        "name": "cosine",
        "description": "Cosine distance against full SPLADE query vectors",
        "align_loss_kind": "cosine",
    },
    {
        "name": "kd_colbert",
        "description": "KL distillation over nway passage scores from ColBERT supervision",
        "align_loss_kind": "kd",
    },
    {
        "name": "margin_mse_colbert",
        "description": "MarginMSE on positive-negative passage score margins from ColBERT supervision",
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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def variant_config(base_cfg: dict, variant: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    sc = cfg["splade_shallow"]
    sc["layer_indices"] = [0, 1, 2]
    sc.pop("n_layers", None)
    sc["factorize_embeddings"] = False
    sc["freeze_head_after_warmup"] = True
    sc["align_loss_kind"] = variant["align_loss_kind"]
    sc["use_contrastive"] = False
    sc["contrastive_coeff"] = 0.0
    sc["log_ranking_metrics"] = variant["align_loss_kind"] in {"kd", "margin_mse"}
    sc["output_dir"] = f"splade_shallow_first3_loss_{variant['name']}"
    if variant["align_loss_kind"] == "kd":
        sc["kd_temperature"] = float(sc.get("kd_temperature", 1.0))
    if variant["align_loss_kind"] in {"kd", "margin_mse"}:
        sc["batch_size"] = 4
        sc["gradient_accumulation_steps"] = 8
        sc["distil_data_path"] = sc.get("distil_data_path", "data/colbertv2_msmarco_64way.json")
        sc["corpus_dataset"] = sc.get("corpus_dataset", "Tevatron/msmarco-passage-corpus")
        sc["corpus_text_field"] = sc.get("corpus_text_field", "text")
        sc["queries_dataset"] = sc.get("queries_dataset", "Tevatron/msmarco-passage")
    return cfg


def parse_beir_summary(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^([a-zA-Z0-9_-]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        dataset, ndcg, mrr = match.groups()
        if dataset.lower() == "average":
            dataset = "Average"
        rows[dataset] = {"ndcg@10": float(ndcg), "mrr@10": float(mrr)}
    return rows


def parse_nanoms_best(text: str) -> dict[str, float | int] | None:
    best: dict[str, float | int] | None = None
    current_step: int | None = None
    step_pattern = re.compile(r"^\[shallow\] Eval at step (\d+) ")
    score_pattern = re.compile(
        r"^\s*\[NanoMSMARCO\] NDCG@10\s+query_doc \(dense\)\s+:\s+([0-9.]+)"
    )
    for line in text.splitlines():
        step_match = step_pattern.match(line)
        if step_match:
            current_step = int(step_match.group(1))
            continue
        score_match = score_pattern.match(line)
        if score_match and current_step is not None:
            score = float(score_match.group(1))
            if best is None or score > best["score"]:
                best = {"step": current_step, "score": score}
    return best


def metric_delta(row: dict, key: str) -> str:
    average = (row.get("metrics") or {}).get("Average")
    if not average:
        return "n/a"
    return f"{average[key] - BASELINE_BEIR[key]:+.4f}"


def write_report(path: Path, datasets: list[str], rows: list[dict]) -> None:
    lines = [
        "# SPLADE First3 Loss Ablation",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Stage: `splade_shallow`",
        "- Model: `naver/splade-v3`",
        "- Layers: `[0, 1, 2]`",
        "- Factorized embeddings: `false`",
        "- Embedding/head frozen after warmup: `freeze_head_after_warmup: true`",
        "- ColBERT score supervision: `data/colbertv2_msmarco_64way.json` for KD and MarginMSE",
        f"- Non-factorized first3 baseline BEIR avg: `{BASELINE_BEIR['ndcg@10']:.4f}/{BASELINE_BEIR['mrr@10']:.4f}`",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        "",
        "## Summary",
        "",
        "| Run | Loss | Best NanoMSMARCO | Baseline BEIR avg | Loss BEIR avg | Delta | Output dir |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms
            else "n/a"
        )
        average = (row.get("metrics") or {}).get("Average")
        average_text = (
            f"{average['ndcg@10']:.4f}/{average['mrr@10']:.4f}"
            if average
            else "n/a"
        )
        lines.append(
            f"| {row['name']} | `{row['align_loss_kind']}` | {best_text} | "
            f"{BASELINE_BEIR['ndcg@10']:.4f}/{BASELINE_BEIR['mrr@10']:.4f} | "
            f"{average_text} | {metric_delta(row, 'ndcg@10')} | `{row['output_dir']}` |"
        )

    lines.extend(["", "## Details", ""])
    for row in rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms
            else "n/a"
        )
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- Description: {row['description']}",
                f"- Loss: `{row['align_loss_kind']}`",
                f"- Best NanoMSMARCO NDCG@10: `{best_text}`",
                f"- Checkpoint: `{row['checkpoint']}`",
                f"- Train log: `{row['train_log']}`",
                f"- BEIR log: `{row['eval_log']}`",
                "",
            ]
        )
        if row.get("metrics"):
            lines.extend(["| Dataset | NDCG@10 | MRR@10 |", "|---|---:|---:|"])
            for dataset, metrics in row["metrics"].items():
                lines.append(
                    f"| {dataset} | {metrics['ndcg@10']:.4f} | {metrics['mrr@10']:.4f} |"
                )
        else:
            lines.append("_No BEIR metrics recorded yet._")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--work-dir", default="runs/splade_first3_loss_ablation")
    parser.add_argument("--results", default="results/splade_first3_loss_ablation.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip-index", action="store_true")
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
        known = {variant["name"] for variant in VARIANTS}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"Unknown variants for --only: {', '.join(sorted(unknown))}")
        variants = [variant for variant in VARIANTS if variant["name"] in requested]

    if not args.skip_index:
        index_cfg = variant_config(base_cfg, VARIANTS[0])
        index_cfg_path = work_dir / "config_index.yaml"
        index_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        index_cfg_path.write_text(yaml.safe_dump(index_cfg, sort_keys=False))
        code = run_command(
            [
                sys.executable,
                "scripts/build_beir_index.py",
                "--config",
                str(index_cfg_path),
                "--stage",
                "splade_shallow",
                "--datasets",
                *args.datasets,
            ],
            work_dir / "build_beir_index.log",
            args.dry_run,
        )
        if code != 0:
            return code

    rows: list[dict] = []
    for variant in variants:
        cfg = variant_config(base_cfg, variant)
        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        output_dir = cfg["splade_shallow"]["output_dir"]
        train_log = work_dir / f"{variant['name']}_train.log"
        eval_log = work_dir / f"{variant['name']}_beir.log"
        checkpoint = root / "checkpoints_splade-v3" / output_dir / "best_NanoMSMARCO.pt"
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
            code = run_command(
                [sys.executable, "train.py", "splade_shallow", "--config", str(cfg_path)],
                train_log,
                args.dry_run,
            )
            if train_log.exists():
                row["best_nanoms"] = parse_nanoms_best(train_log.read_text())
            if code != 0:
                rows.append(row)
                write_report(results_path, args.datasets, rows)
                return code
        elif train_log.exists():
            row["best_nanoms"] = parse_nanoms_best(train_log.read_text())

        if not args.skip_eval:
            if not checkpoint.exists() and not args.dry_run:
                print(f"Missing best checkpoint: {checkpoint}", file=sys.stderr)
                rows.append(row)
                write_report(results_path, args.datasets, rows)
                return 1
            code = run_command(
                [
                    sys.executable,
                    "scripts/eval_beir.py",
                    "--stage",
                    "splade_shallow",
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--datasets",
                    *args.datasets,
                ],
                eval_log,
                args.dry_run,
            )
            if eval_log.exists():
                row["metrics"] = parse_beir_summary(eval_log.read_text())
            if code != 0:
                rows.append(row)
                write_report(results_path, args.datasets, rows)
                return code
        elif eval_log.exists():
            row["metrics"] = parse_beir_summary(eval_log.read_text())

        rows.append(row)
        write_report(results_path, args.datasets, rows)

    write_report(results_path, args.datasets, rows)
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
