#!/usr/bin/env python
"""Train and evaluate the static (0-layer) SPLADE query encoder.

One learnable weight per vocabulary token — no transformer layers.
Query encoding = tokenize → per-token weight lookup.
Trained with MSE distillation against the frozen full SPLADE on queries.

Writes results to results/splade_static.md after eval completes.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

DEFAULT_DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]


def run_command(cmd: list[str], log_path: Path) -> int:
    print(" ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


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
    step_pat = re.compile(r"^\[static\] Eval at step (\d+) ")
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


def write_report(
    path: Path,
    datasets: list[str],
    best_nanoms: dict | None,
    metrics: dict[str, dict[str, float]],
    train_log: Path,
    eval_log: Path,
    checkpoint: Path,
) -> None:
    best_text = (
        f"{best_nanoms['score']:.4f} @ step {best_nanoms['step']}"
        if best_nanoms
        else "n/a"
    )
    lines = [
        "# SPLADE Static (0-Layer) Query Encoder",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Architecture: one learnable weight per vocabulary token (no transformer layers)",
        "- Training: MSE distillation against frozen full SPLADE-v3 on queries",
        f"- BEIR datasets: `{', '.join(datasets)}`",
        f"- Best NanoMSMARCO NDCG@10: `{best_text}`",
        f"- Train log: `{train_log}`",
        f"- BEIR log: `{eval_log}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "## BEIR Results",
        "",
    ]
    if metrics:
        lines.extend(["| Dataset | NDCG@10 | MRR@10 |", "|---|---:|---:|"])
        for dataset, m in metrics.items():
            lines.append(f"| {dataset} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f} |")
    else:
        lines.append("_No BEIR metrics recorded yet._")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--work-dir", default="runs/splade_static")
    parser.add_argument("--results", default="results/splade_static.md")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((root / args.config).read_text())
    sc = cfg["splade_static"]
    checkpoint = root / "checkpoints_splade-v3" / sc["output_dir"] / "best_NanoMSMARCO.pt"

    train_log = work_dir / "train.log"
    eval_log  = work_dir / "beir.log"

    # ── Train ─────────────────────────────────────────────────────────
    if checkpoint.exists() and eval_log.exists() and "Average" in eval_log.read_text():
        print("[skip] checkpoint and eval log already exist")
    else:
        train_cmd = [sys.executable, "train.py", "splade_static", "--config", args.config]
        code = run_command(train_cmd, train_log)
        if code != 0:
            print(f"Training failed (exit {code})", file=sys.stderr)
            return code
        if not checkpoint.exists():
            print(f"Missing checkpoint after training: {checkpoint}", file=sys.stderr)
            return 1

        # ── Eval ──────────────────────────────────────────────────────
        eval_cmd = [
            sys.executable, "scripts/eval_beir.py",
            "--stage", "splade_static",
            "--config", args.config,
            "--checkpoint", str(checkpoint),
            "--datasets", *args.datasets,
        ]
        code = run_command(eval_cmd, eval_log)
        if code != 0:
            print(f"Eval failed (exit {code})", file=sys.stderr)
            return code

    # ── Parse and report ──────────────────────────────────────────────
    best_nanoms = parse_nanoms_best(train_log.read_text()) if train_log.exists() else None
    metrics = parse_beir_summary(eval_log.read_text()) if eval_log.exists() else {}
    write_report(results_path, args.datasets, best_nanoms, metrics, train_log, eval_log, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
