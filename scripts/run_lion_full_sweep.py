#!/usr/bin/env python
"""Lion shallow layer sweep: 1–5 layers × first/spaced/last positions.

Layers 3–5 already have results in results/lion_shallow_layer_sweep.md —
those are loaded directly without re-training. Only 1-layer and 2-layer
variants are trained from scratch.

Writes updated results to results/lion_shallow_layer_sweep.md after each variant.
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

LION_N_LAYERS = 16
DEFAULT_DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]

# Results for 3–5 layer variants are known from previous runs (checkpoints lost).
# Loaded from the existing results MD file rather than re-trained.
EXISTING_RESULTS: dict[str, dict] = {
    "first3":  {"best_nanoms": {"score": 0.6587, "step": 40000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2929, "mrr@10": 0.4888}, "scifact": {"ndcg@10": 0.5710, "mrr@10": 0.5333}, "arguana": {"ndcg@10": 0.2948, "mrr@10": 0.1996}, "scidocs": {"ndcg@10": 0.1310, "mrr@10": 0.2364}, "fiqa": {"ndcg@10": 0.2690, "mrr@10": 0.3325}, "Average": {"ndcg@10": 0.3117, "mrr@10": 0.3581}}},
    "spaced3": {"best_nanoms": {"score": 0.6888, "step": 30000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2881, "mrr@10": 0.4807}, "scifact": {"ndcg@10": 0.5646, "mrr@10": 0.5319}, "arguana": {"ndcg@10": 0.3004, "mrr@10": 0.2026}, "scidocs": {"ndcg@10": 0.1266, "mrr@10": 0.2282}, "fiqa": {"ndcg@10": 0.2623, "mrr@10": 0.3252}, "Average": {"ndcg@10": 0.3084, "mrr@10": 0.3537}}},
    "last3":   {"best_nanoms": {"score": 0.6859, "step": 50000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2863, "mrr@10": 0.4788}, "scifact": {"ndcg@10": 0.5564, "mrr@10": 0.5323}, "arguana": {"ndcg@10": 0.2938, "mrr@10": 0.1977}, "scidocs": {"ndcg@10": 0.1295, "mrr@10": 0.2381}, "fiqa": {"ndcg@10": 0.2681, "mrr@10": 0.3322}, "Average": {"ndcg@10": 0.3068, "mrr@10": 0.3558}}},
    "first4":  {"best_nanoms": {"score": 0.6796, "step": 40000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2948, "mrr@10": 0.4915}, "scifact": {"ndcg@10": 0.5715, "mrr@10": 0.5386}, "arguana": {"ndcg@10": 0.2956, "mrr@10": 0.1996}, "scidocs": {"ndcg@10": 0.1311, "mrr@10": 0.2355}, "fiqa": {"ndcg@10": 0.2700, "mrr@10": 0.3316}, "Average": {"ndcg@10": 0.3126, "mrr@10": 0.3593}}},
    "spaced4": {"best_nanoms": {"score": 0.6895, "step": 50000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2918, "mrr@10": 0.4822}, "scifact": {"ndcg@10": 0.5794, "mrr@10": 0.5428}, "arguana": {"ndcg@10": 0.3053, "mrr@10": 0.2069}, "scidocs": {"ndcg@10": 0.1311, "mrr@10": 0.2354}, "fiqa": {"ndcg@10": 0.2737, "mrr@10": 0.3385}, "Average": {"ndcg@10": 0.3163, "mrr@10": 0.3611}}},
    "last4":   {"best_nanoms": {"score": 0.6780, "step": 50000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2899, "mrr@10": 0.4896}, "scifact": {"ndcg@10": 0.5515, "mrr@10": 0.5213}, "arguana": {"ndcg@10": 0.2984, "mrr@10": 0.1998}, "scidocs": {"ndcg@10": 0.1268, "mrr@10": 0.2284}, "fiqa": {"ndcg@10": 0.2682, "mrr@10": 0.3298}, "Average": {"ndcg@10": 0.3070, "mrr@10": 0.3538}}},
    "first5":  {"best_nanoms": {"score": 0.6751, "step": 40000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2954, "mrr@10": 0.4887}, "scifact": {"ndcg@10": 0.5700, "mrr@10": 0.5367}, "arguana": {"ndcg@10": 0.2983, "mrr@10": 0.2010}, "scidocs": {"ndcg@10": 0.1283, "mrr@10": 0.2287}, "fiqa": {"ndcg@10": 0.2703, "mrr@10": 0.3331}, "Average": {"ndcg@10": 0.3125, "mrr@10": 0.3576}}},
    "spaced5": {"best_nanoms": {"score": 0.6920, "step": 50000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2921, "mrr@10": 0.4798}, "scifact": {"ndcg@10": 0.5743, "mrr@10": 0.5435}, "arguana": {"ndcg@10": 0.3001, "mrr@10": 0.2038}, "scidocs": {"ndcg@10": 0.1318, "mrr@10": 0.2355}, "fiqa": {"ndcg@10": 0.2720, "mrr@10": 0.3347}, "Average": {"ndcg@10": 0.3141, "mrr@10": 0.3595}}},
    "last5":   {"best_nanoms": {"score": 0.6797, "step": 40000}, "metrics": {"nfcorpus": {"ndcg@10": 0.2939, "mrr@10": 0.4892}, "scifact": {"ndcg@10": 0.5496, "mrr@10": 0.5177}, "arguana": {"ndcg@10": 0.2878, "mrr@10": 0.1937}, "scidocs": {"ndcg@10": 0.1267, "mrr@10": 0.2294}, "fiqa": {"ndcg@10": 0.2632, "mrr@10": 0.3239}, "Average": {"ndcg@10": 0.3042, "mrr@10": 0.3508}}},
}


def spaced_layers(count: int, total: int = LION_N_LAYERS) -> list[int]:
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
                list(range(LION_N_LAYERS - count, LION_N_LAYERS)),
            ),
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
    step_pat = re.compile(r"^\[lion-shallow/(?:warm|full)\] step\s+(\d+)\s+\|")
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
    root = Path.cwd()
    for row in variant_rows:
        best = row.get("best_nanoms")
        best_text = f"{best['score']:.4f} @ step {best['step']}" if best else "n/a"
        suffix = " _(loaded from prior run)_" if row.get("from_existing") else ""
        lines.append(
            f"| {row['name']} | `{row['layers']}` | {best_text}{suffix} | "
            f"`{row['output_dir']}` | `{row['checkpoint']}` |"
        )

    lines.extend(["", "## BEIR Results", ""])
    for row in variant_rows:
        best = row.get("best_nanoms")
        best_text = f"{best['score']:.4f} @ step {best['step']}" if best else "n/a"
        suffix = " _(loaded from prior run)_" if row.get("from_existing") else ""
        lines.extend([
            f"### {row['name']}",
            "",
            f"- Description: {row['description']}",
            f"- Layers: `{row['layers']}`",
            f"- Best NanoMSMARCO NDCG@10: `{best_text}`{suffix}",
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
    parser.add_argument("--work-dir", default="runs/lion_shallow_layer_sweep")
    parser.add_argument("--results", default="results/lion_shallow_layer_sweep.md")
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
            root / "checkpoints_lion_shallow" / variant["output_dir"] / "best_NanoMSMARCO.pt"
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
            "from_existing": False,
        }

        # Load from prior run results if checkpoint is gone but data is known
        if variant["name"] in EXISTING_RESULTS:
            print(f"[existing] {variant['name']} — loading from prior run results")
            existing = EXISTING_RESULTS[variant["name"]]
            row["best_nanoms"] = existing["best_nanoms"]
            row["metrics"] = existing["metrics"]
            row["from_existing"] = True
            variant_rows.append(row)
            write_report(results_path, args.datasets, variant_rows)
            continue

        # Skip if fresh checkpoint + eval already done
        if already_done(checkpoint, eval_log):
            print(f"[skip] {variant['name']} — checkpoint and eval log already exist")
            row["best_nanoms"] = parse_nanoms_best(train_log.read_text()) if train_log.exists() else None
            row["metrics"] = parse_beir_summary(eval_log.read_text())
            variant_rows.append(row)
            write_report(results_path, args.datasets, variant_rows)
            continue

        train_cmd = [
            sys.executable, "train.py", "lion_shallow", "--config", str(cfg_path),
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
            "--stage", "lion_shallow",
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
