#!/usr/bin/env python
"""SPLADE shallow layer sweep: 1–5 layers × first/spaced/last positions.

Skips variants where the best checkpoint and eval log already exist.
Writes results to results/splade_shallow_layer_sweep.md after each variant.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

SPLADE_N_LAYERS = 12
DEFAULT_DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]


def spaced_layers(count: int, total: int = SPLADE_N_LAYERS) -> list[int]:
    if count == 1:
        return [round((total - 1) / 2)]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def build_variants() -> list[dict]:
    variants: list[dict] = []
    for count in range(1, 6):
        s = "s" if count > 1 else ""
        selections = [
            ("first", f"First {count} layer{s}", list(range(count))),
            (
                "spaced",
                "Middle layer" if count == 1 else f"Equally spaced {count} layers",
                spaced_layers(count),
            ),
            (
                "last",
                f"Last {count} layer{s}",
                list(range(SPLADE_N_LAYERS - count, SPLADE_N_LAYERS)),
            ),
        ]
        for kind, description, layer_indices in selections:
            variants.append(
                {
                    "name": f"{kind}{count}",
                    "description": description,
                    "layer_indices": layer_indices,
                    "output_dir": f"splade_shallow_{kind}{count}",
                }
            )
    return variants


VARIANTS = build_variants()


def already_done(checkpoint: Path, eval_log: Path) -> bool:
    return checkpoint.exists() and eval_log.exists() and "Average" in eval_log.read_text()


def run_command(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    print(" ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("$ " + " ".join(cmd) + "\n")
        return 0
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def variant_config(base_cfg: dict, variant: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    sc = cfg["splade_shallow"]
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
    step_pat = re.compile(r"^\[shallow\] Eval at step (\d+) ")
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


def write_report(path: Path, datasets: list[str], variant_rows: list[dict]) -> None:
    lines = [
        "# SPLADE Shallow Layer Sweep",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Stage: `splade_shallow`",
        "- Factorized embeddings: `false`",
        "- Embedding/head matrix frozen after warmup: `freeze_head_after_warmup: true`",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        "",
        "## Variants",
        "",
        "| Run | Layers | Best NanoMSMARCO | Output dir | Best checkpoint |",
        "|---|---:|---:|---|---|",
    ]
    root = Path.cwd()
    for row in variant_rows:
        best = row.get("best_nanoms")
        best_text = f"{best['score']:.4f} @ step {best['step']}" if best else "n/a"
        lines.append(
            f"| {row['name']} | `{row['layers']}` | {best_text} | "
            f"`{row['output_dir']}` | `{row['checkpoint']}` |"
        )

    lines.extend(["", "## BEIR Results", ""])
    for row in variant_rows:
        best = row.get("best_nanoms")
        best_text = f"{best['score']:.4f} @ step {best['step']}" if best else "n/a"
        lines.extend([
            f"### {row['name']}",
            "",
            f"- Description: {row['description']}",
            f"- Layers: `{row['layers']}`",
            f"- Best NanoMSMARCO NDCG@10: `{best_text}`",
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
    parser.add_argument("--work-dir", default="runs/splade_shallow_layer_sweep")
    parser.add_argument("--results", default="results/splade_shallow_layer_sweep.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    base_cfg = yaml.safe_load(Path(args.config).read_text())

    variant_rows: list[dict] = []

    for variant in VARIANTS:
        cfg = variant_config(base_cfg, variant)
        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        train_log = work_dir / f"{variant['name']}_train.log"
        eval_log = work_dir / f"{variant['name']}_beir.log"
        checkpoint = (
            root / "checkpoints_splade-v3" / variant["output_dir"] / "best_NanoMSMARCO.pt"
        )

        row: dict = {
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

        if already_done(checkpoint, eval_log):
            print(f"[skip] {variant['name']} — checkpoint and eval log already exist")
            row["best_nanoms"] = parse_nanoms_best(train_log.read_text()) if train_log.exists() else None
            row["metrics"] = parse_beir_summary(eval_log.read_text())
            variant_rows.append(row)
            write_report(results_path, args.datasets, variant_rows)
            continue

        train_cmd = [
            sys.executable, "train.py", "splade_shallow", "--config", str(cfg_path),
        ]
        code = run_command(train_cmd, train_log, dry_run=args.dry_run)
        if train_log.exists():
            row["best_nanoms"] = parse_nanoms_best(train_log.read_text())
        if code != 0:
            variant_rows.append(row)
            write_report(results_path, args.datasets, variant_rows)
            print(f"Training failed for {variant['name']} (exit {code})", file=sys.stderr)
            return code

        if not checkpoint.exists() and not args.dry_run:
            print(f"Missing checkpoint after training: {checkpoint}", file=sys.stderr)
            variant_rows.append(row)
            write_report(results_path, args.datasets, variant_rows)
            return 1

        eval_cmd = [
            sys.executable,
            "scripts/eval_beir.py",
            "--stage", "splade_shallow",
            "--config", str(cfg_path),
            "--checkpoint", str(checkpoint),
            "--datasets", *args.datasets,
        ]
        code = run_command(eval_cmd, eval_log, dry_run=args.dry_run)
        if eval_log.exists():
            row["metrics"] = parse_beir_summary(eval_log.read_text())
        if code != 0:
            variant_rows.append(row)
            write_report(results_path, args.datasets, variant_rows)
            print(f"Eval failed for {variant['name']} (exit {code})", file=sys.stderr)
            return code

        variant_rows.append(row)
        write_report(results_path, args.datasets, variant_rows)

    print(f"All variants done. Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
