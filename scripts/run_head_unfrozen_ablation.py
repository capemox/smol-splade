#!/usr/bin/env python
"""Run the best shallow configs with the embedding/head unfrozen after warmup."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


DEFAULT_DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]

VARIANTS = [
    {
        "name": "splade_spaced3_head_unfrozen",
        "stage": "splade_shallow",
        "description": "Best prior SPLADE shallow config by BEIR average: spaced 3 layers",
        "layer_indices": [0, 6, 11],
        "output_dir": "splade_shallow_spaced3_head_unfrozen",
        "checkpoint_root": "checkpoints_splade-v3",
        "baseline_beir": {"ndcg@10": 0.3509, "mrr@10": 0.4067},
    },
    {
        "name": "lion_first3_head_unfrozen",
        "stage": "lion_shallow",
        "description": "Best prior Lion 3-layer config by BEIR average: first 3 layers",
        "layer_indices": [0, 1, 2],
        "output_dir": "lion_shallow_first3_head_unfrozen",
        "checkpoint_root": "checkpoints_lion_shallow",
        "baseline_beir": {"ndcg@10": 0.3117, "mrr@10": 0.3581},
    },
    {
        "name": "lion_spaced4_head_unfrozen",
        "stage": "lion_shallow",
        "description": "Best prior Lion 4-layer config by BEIR average: spaced 4 layers",
        "layer_indices": [0, 5, 10, 15],
        "output_dir": "lion_shallow_spaced4_head_unfrozen",
        "checkpoint_root": "checkpoints_lion_shallow",
        "baseline_beir": {"ndcg@10": 0.3163, "mrr@10": 0.3611},
    },
    {
        "name": "lion_spaced5_head_unfrozen",
        "stage": "lion_shallow",
        "description": "Best prior Lion 5-layer config by BEIR average: spaced 5 layers",
        "layer_indices": [0, 4, 8, 11, 15],
        "output_dir": "lion_shallow_spaced5_head_unfrozen",
        "checkpoint_root": "checkpoints_lion_shallow",
        "baseline_beir": {"ndcg@10": 0.3141, "mrr@10": 0.3595},
    },
]


def run_command(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    print(" ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("$ " + " ".join(cmd) + "\n")
        return 0

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def variant_config(base_cfg: dict, variant: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    sc = cfg[variant["stage"]]
    sc["layer_indices"] = list(variant["layer_indices"])
    sc.pop("n_layers", None)
    sc["factorize_embeddings"] = False
    sc["freeze_head_after_warmup"] = False
    sc["head_lr_scale"] = float(sc.get("head_lr_scale", 0.1))
    sc["output_dir"] = variant["output_dir"]
    if variant["stage"] == "lion_shallow":
        # Full Lion lexical-matrix training is tight on 8 GB GPUs. Keep the
        # effective batch size at 32 while reducing activation memory.
        sc["batch_size"] = 1
        sc["gradient_accumulation_steps"] = 32
        sc["gradient_checkpointing"] = True
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
    step_patterns = (
        re.compile(r"^\[shallow\] Eval at step (\d+) "),
        re.compile(r"^\[lion-shallow/(?:warm|full)\] step\s+(\d+)\s+\|"),
    )
    score_pattern = re.compile(
        r"^\s*\[NanoMSMARCO\] NDCG@10\s+query_doc \(dense\)\s+:\s+([0-9.]+)"
    )
    for line in text.splitlines():
        for step_pattern in step_patterns:
            step_match = step_pattern.match(line)
            if step_match:
                current_step = int(step_match.group(1))
                break
        else:
            score_match = score_pattern.match(line)
            if score_match and current_step is not None:
                score = float(score_match.group(1))
                if best is None or score > best["score"]:
                    best = {"step": current_step, "score": score}
    return best


def metric_delta(row: dict, key: str) -> str:
    metrics = row.get("metrics") or {}
    baseline = row.get("baseline_beir") or {}
    if "Average" not in metrics or key not in baseline:
        return "n/a"
    delta = metrics["Average"][key] - baseline[key]
    return f"{delta:+.4f}"


def write_report(path: Path, datasets: list[str], variant_rows: list[dict]) -> None:
    lines = [
        "# Head Unfrozen Shallow Ablation",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Selection basis: best prior frozen-head BEIR small average for SPLADE, Lion 3-layer, Lion 4-layer, and Lion 5-layer.",
        "- Factorized embeddings: `false`",
        "- Warmup behavior: embedding/head train during the first 5k steps, then continue training at `head_lr_scale`.",
        "- `freeze_head_after_warmup`: `false`",
        "- `head_lr_scale`: `0.1` unless overridden in the base config",
        "- Lion microbatch: `batch_size: 1`, `gradient_accumulation_steps: 32` to fit the full lexical matrix on the available GPU while preserving the previous effective batch size.",
        "- Lion gradient checkpointing: `true` for the full-head ablation to reduce activation memory.",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        "",
        "## Summary",
        "",
        "| Run | Stage | Layers | Best NanoMSMARCO | Frozen BEIR avg | Unfrozen BEIR avg | Delta | Output dir |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in variant_rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms
            else "n/a"
        )
        baseline = row.get("baseline_beir") or {}
        baseline_text = (
            f"{baseline['ndcg@10']:.4f}/{baseline['mrr@10']:.4f}"
            if baseline
            else "n/a"
        )
        metrics = row.get("metrics") or {}
        average = metrics.get("Average")
        average_text = (
            f"{average['ndcg@10']:.4f}/{average['mrr@10']:.4f}"
            if average
            else "n/a"
        )
        lines.append(
            f"| {row['name']} | `{row['stage']}` | `{row['layers']}` | {best_text} | "
            f"{baseline_text} | {average_text} | {metric_delta(row, 'ndcg@10')} | "
            f"`{row['output_dir']}` |"
        )

    lines.extend(["", "## Details", ""])
    for row in variant_rows:
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
                f"- Stage: `{row['stage']}`",
                f"- Layers: `{row['layers']}`",
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


def build_stage_indexes(
    base_cfg: dict,
    variants: list[dict],
    work_dir: Path,
    datasets: list[str],
    dry_run: bool,
) -> int:
    seen_stages: set[str] = set()
    for variant in variants:
        stage = variant["stage"]
        if stage in seen_stages:
            continue
        seen_stages.add(stage)

        index_cfg = variant_config(base_cfg, variant)
        index_cfg_path = work_dir / f"config_index_{stage}.yaml"
        index_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        index_cfg_path.write_text(yaml.safe_dump(index_cfg, sort_keys=False))

        build_cmd = [
            sys.executable,
            "scripts/build_beir_index.py",
            "--config",
            str(index_cfg_path),
            "--stage",
            stage,
            "--batch_size",
            str(index_cfg[stage].get("eval_batch_size", 4)),
            "--datasets",
            *datasets,
        ]
        code = run_command(build_cmd, work_dir / f"build_beir_index_{stage}.log", dry_run)
        if code != 0:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--work-dir", default="runs/head_unfrozen_ablation")
    parser.add_argument("--results", default="results/head_unfrozen_ablation.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[variant["name"] for variant in VARIANTS],
        help="Run only these variants; existing logs for other variants are still included in the report.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    base_cfg = yaml.safe_load(Path(args.config).read_text())

    if not args.skip_index:
        code = build_stage_indexes(base_cfg, VARIANTS, work_dir, args.datasets, args.dry_run)
        if code != 0:
            return code

    variant_rows: list[dict] = []
    selected_names = set(args.only or [variant["name"] for variant in VARIANTS])
    for variant in VARIANTS:
        cfg = variant_config(base_cfg, variant)
        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        train_log = work_dir / f"{variant['name']}_train.log"
        eval_log = work_dir / f"{variant['name']}_beir.log"
        checkpoint = (
            root
            / variant["checkpoint_root"]
            / variant["output_dir"]
            / "best_NanoMSMARCO.pt"
        )
        row = {
            "name": variant["name"],
            "description": variant["description"],
            "stage": variant["stage"],
            "layers": variant["layer_indices"],
            "output_dir": variant["output_dir"],
            "checkpoint": checkpoint,
            "train_log": train_log,
            "eval_log": eval_log,
            "baseline_beir": variant["baseline_beir"],
            "best_nanoms": None,
            "metrics": {},
        }

        should_run = variant["name"] in selected_names

        if should_run and not args.skip_train:
            train_cmd = [
                sys.executable,
                "train.py",
                variant["stage"],
                "--config",
                str(cfg_path),
            ]
            code = run_command(train_cmd, train_log, dry_run=args.dry_run)
            if train_log.exists():
                row["best_nanoms"] = parse_nanoms_best(train_log.read_text())
            if code != 0:
                variant_rows.append(row)
                write_report(results_path, args.datasets, variant_rows)
                return code
        elif train_log.exists():
            row["best_nanoms"] = parse_nanoms_best(train_log.read_text())

        if should_run and not args.skip_eval:
            if not checkpoint.exists() and not args.dry_run:
                print(f"Missing best checkpoint: {checkpoint}", file=sys.stderr)
                variant_rows.append(row)
                write_report(results_path, args.datasets, variant_rows)
                return 1
            eval_cmd = [
                sys.executable,
                "scripts/eval_beir.py",
                "--stage",
                variant["stage"],
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(checkpoint),
                "--datasets",
                *args.datasets,
            ]
            code = run_command(eval_cmd, eval_log, dry_run=args.dry_run)
            if eval_log.exists():
                row["metrics"] = parse_beir_summary(eval_log.read_text())
            if code != 0:
                variant_rows.append(row)
                write_report(results_path, args.datasets, variant_rows)
                return code
        elif eval_log.exists():
            row["metrics"] = parse_beir_summary(eval_log.read_text())

        variant_rows.append(row)
        write_report(results_path, args.datasets, variant_rows)

    write_report(results_path, args.datasets, variant_rows)
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
