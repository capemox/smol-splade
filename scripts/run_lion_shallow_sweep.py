#!/usr/bin/env python
"""Run non-factorized Lion shallow layer-selection experiments."""

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
LION_N_LAYERS = 16


def spaced_layers(count: int, total: int = LION_N_LAYERS) -> list[int]:
    if count <= 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def build_variants() -> list[dict]:
    variants: list[dict] = []
    for count in (3, 4, 5):
        selections = [
            ("first", f"First {count} layers", list(range(count))),
            ("spaced", f"Equally spaced {count} layers", spaced_layers(count)),
            ("last", f"Last {count} layers", list(range(LION_N_LAYERS - count, LION_N_LAYERS))),
        ]
        for kind, description, layer_indices in selections:
            variants.append(
                {
                    "name": f"{kind}{count}",
                    "description": description,
                    "layer_indices": layer_indices,
                    "output_dir": f"lion_shallow_{kind}{count}",
                }
            )
    return variants


VARIANTS = build_variants()


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
    sc = cfg["lion_shallow"]
    sc["layer_indices"] = list(variant["layer_indices"])
    sc.pop("n_layers", None)
    sc["factorize_embeddings"] = False
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
    step_pattern = re.compile(r"^\[lion-shallow/(?:warm|full)\] step\s+(\d+)\s+\|")
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


def write_report(path: Path, datasets: list[str], variant_rows: list[dict]) -> None:
    lines = [
        "# Lion Shallow Layer Sweep",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Stage: `lion_shallow`",
        "- Factorized embeddings: `false`",
        "- Embedding/head matrix frozen after warmup: `freeze_head_after_warmup: true`",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        "",
        "## Variants",
        "",
        "| Run | Layers | Best NanoMSMARCO | Output dir | Best checkpoint |",
        "|---|---:|---:|---|---|",
    ]
    for row in variant_rows:
        best_nanoms = row.get("best_nanoms")
        best_text = (
            f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
            if best_nanoms
            else "n/a"
        )
        lines.append(
            f"| {row['name']} | `{row['layers']}` | {best_text} | "
            f"`{row['output_dir']}` | `{row['checkpoint']}` |"
        )

    lines.extend(["", "## BEIR Results", ""])
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
                f"- Layers: `{row['layers']}`",
                f"- Best NanoMSMARCO NDCG@10: `{best_text}`",
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
    parser.add_argument("--work-dir", default="runs/lion_shallow_layer_sweep")
    parser.add_argument("--results", default="results/lion_shallow_layer_sweep.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    base_cfg = yaml.safe_load(Path(args.config).read_text())

    variant_rows: list[dict] = []

    if not args.skip_index:
        index_cfg = variant_config(base_cfg, VARIANTS[0])
        index_cfg_path = work_dir / "config_index.yaml"
        index_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        index_cfg_path.write_text(yaml.safe_dump(index_cfg, sort_keys=False))
        index_cmd = [
            sys.executable,
            "scripts/build_beir_index.py",
            "--config",
            str(index_cfg_path),
            "--stage",
            "lion_shallow",
            "--batch_size",
            str(index_cfg["lion_shallow"].get("eval_batch_size", 4)),
            "--datasets",
            *args.datasets,
        ]
        code = run_command(index_cmd, work_dir / "build_beir_index.log", dry_run=args.dry_run)
        if code != 0:
            return code

    for variant in VARIANTS:
        cfg = variant_config(base_cfg, variant)
        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        train_log = work_dir / f"{variant['name']}_train.log"
        eval_log = work_dir / f"{variant['name']}_beir.log"
        checkpoint = (
            root
            / "checkpoints_lion_shallow"
            / variant["output_dir"]
            / "best_NanoMSMARCO.pt"
        )
        row = {
            "name": variant["name"],
            "description": variant["description"],
            "layers": variant["layer_indices"],
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
                "lion_shallow",
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
                "lion_shallow",
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
