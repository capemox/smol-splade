#!/usr/bin/env python
"""Run first-3-layer factorized shallow ablations for SPLADE and Lion."""

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
DEFAULT_SPLADE_DIMS = [64, 128, 256, 512]
DEFAULT_LION_DIMS = [256, 512, 1024, 2048]

BASELINES = {
    "splade_shallow": {"ndcg@10": 0.3483, "mrr@10": 0.4050},
    "lion_shallow": {"ndcg@10": 0.3117, "mrr@10": 0.3581},
}


def build_variants(splade_dims: list[int], lion_dims: list[int]) -> list[dict]:
    variants: list[dict] = []
    for dim in splade_dims:
        variants.append(
            {
                "name": f"splade_first3_factorized_d{dim}",
                "stage": "splade_shallow",
                "description": f"SPLADE v3 first 3 layers, SVD factorized lexical matrix dim {dim}",
                "layer_indices": [0, 1, 2],
                "factor_dim": dim,
                "output_dir": f"splade_shallow_first3_factorized_d{dim}",
                "checkpoint_root": "checkpoints_splade-v3",
            }
        )
    for dim in lion_dims:
        variants.append(
            {
                "name": f"lion_first3_factorized_d{dim}",
                "stage": "lion_shallow",
                "description": f"Lion first 3 layers, SVD factorized lexical matrix dim {dim}",
                "layer_indices": [0, 1, 2],
                "factor_dim": dim,
                "output_dir": f"lion_shallow_first3_factorized_d{dim}",
                "checkpoint_root": "checkpoints_lion_shallow",
            }
        )
    return variants


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
    sc["factorize_embeddings"] = True
    sc["factorized_embedding_dim"] = int(variant["factor_dim"])
    sc["factorization_init"] = "svd"
    sc["freeze_head_after_warmup"] = True
    sc["output_dir"] = variant["output_dir"]
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
    average = (row.get("metrics") or {}).get("Average")
    baseline = BASELINES.get(row["stage"], {})
    if not average or key not in baseline:
        return "n/a"
    return f"{average[key] - baseline[key]:+.4f}"


def write_report(path: Path, datasets: list[str], variant_rows: list[dict]) -> None:
    lines = [
        "# Factorized First3 Shallow Grid",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- SPLADE layers: `[0, 1, 2]`",
        "- Lion layers: `[0, 1, 2]`",
        "- Factorization init: `svd`",
        "- Factorized embedding/head matrix frozen after warmup: `freeze_head_after_warmup: true`",
        "- Training target: full document encoder SPLADE vectors; BEIR eval uses the pre-built document indexes when available.",
        f"- Frozen first3 baselines: SPLADE `{BASELINES['splade_shallow']['ndcg@10']:.4f}/{BASELINES['splade_shallow']['mrr@10']:.4f}`, Lion `{BASELINES['lion_shallow']['ndcg@10']:.4f}/{BASELINES['lion_shallow']['mrr@10']:.4f}`",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        "",
        "## Summary",
        "",
        "| Run | Stage | Factor dim | Best NanoMSMARCO | Frozen first3 BEIR avg | Factorized BEIR avg | Delta | Output dir |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in variant_rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms
            else "n/a"
        )
        baseline = BASELINES[row["stage"]]
        baseline_text = f"{baseline['ndcg@10']:.4f}/{baseline['mrr@10']:.4f}"
        average = (row.get("metrics") or {}).get("Average")
        average_text = (
            f"{average['ndcg@10']:.4f}/{average['mrr@10']:.4f}"
            if average
            else "n/a"
        )
        lines.append(
            f"| {row['name']} | `{row['stage']}` | {row['factor_dim']} | {best_text} | "
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
                f"- Factor dim: `{row['factor_dim']}`",
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
    parser.add_argument("--work-dir", default="runs/factorized_first3_grid")
    parser.add_argument("--results", default="results/factorized_first3_grid.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--splade-dims", nargs="+", type=int, default=DEFAULT_SPLADE_DIMS)
    parser.add_argument("--lion-dims", nargs="+", type=int, default=DEFAULT_LION_DIMS)
    parser.add_argument("--only", nargs="+", default=None, help="Run only these variant names")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    base_cfg = yaml.safe_load(Path(args.config).read_text())

    variants = build_variants(args.splade_dims, args.lion_dims)
    if args.only:
        requested = set(args.only)
        known = {variant["name"] for variant in variants}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"Unknown variants for --only: {', '.join(sorted(unknown))}")
        variants = [variant for variant in variants if variant["name"] in requested]

    if not args.skip_index:
        code = build_stage_indexes(base_cfg, variants, work_dir, args.datasets, args.dry_run)
        if code != 0:
            return code

    variant_rows: list[dict] = []
    for variant in variants:
        cfg = variant_config(base_cfg, variant)
        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        train_log = work_dir / f"{variant['name']}_train.log"
        eval_log = work_dir / f"{variant['name']}_beir.log"
        checkpoint = root / variant["checkpoint_root"] / variant["output_dir"] / "best_NanoMSMARCO.pt"

        row = {
            "name": variant["name"],
            "description": variant["description"],
            "stage": variant["stage"],
            "layers": variant["layer_indices"],
            "factor_dim": variant["factor_dim"],
            "output_dir": variant["output_dir"],
            "checkpoint": checkpoint,
            "train_log": train_log,
            "eval_log": eval_log,
            "best_nanoms": None,
            "metrics": {},
        }

        if not args.skip_train:
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

        if not args.skip_eval:
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
