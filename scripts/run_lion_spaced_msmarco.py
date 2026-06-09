#!/usr/bin/env python
"""Train Lion spaced1-5 shallow models (MSE loss) and eval on full MS-MARCO dev.

Pipeline for each variant:
  1. Train (skip if best_NanoMSMARCO.pt already exists)
  2. BEIR eval on 5 datasets (skip if log complete)
  3. MS-MARCO dev eval against pre-built Lion index (skip if log complete)

After all variants, also runs full Lion doc-only ceiling eval.
Results written to results/lion_spaced_msmarco_eval.md after each step.

Builds the Lion MS-MARCO index automatically if not found at
data/msmarco_index_lion/ (one-time, ~2-3h).

Usage:
    uv run python3 scripts/run_lion_spaced_msmarco.py
    uv run python3 scripts/run_lion_spaced_msmarco.py --skip_train
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

LION_N_LAYERS = 16
LION_INDEX_DIR = "data/msmarco_index_lion"
BEIR_DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]


def spaced_layers(count: int, total: int = LION_N_LAYERS) -> list[int]:
    if count <= 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


VARIANTS = [
    {
        "name": f"spaced{n}",
        "layer_indices": spaced_layers(n),
        "output_dir": f"lion_shallow_spaced{n}",
    }
    for n in range(1, 6)
]


def run_command(cmd: list[str], log_path: Path, env_extra: dict | None = None) -> int:
    print(f"\n{'='*60}")
    print(" ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)
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


def log_is_complete(log_path: Path, marker: str) -> bool:
    return log_path.exists() and marker in log_path.read_text()


def parse_msmarco(log_path: Path) -> tuple[float, float] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m_n = re.search(r"NDCG@10\s*:\s*([0-9.]+)", text)
    m_m = re.search(r"MRR@10\s*:\s*([0-9.]+)", text)
    return (float(m_n.group(1)), float(m_m.group(1))) if m_n and m_m else None


def parse_beir_avg(log_path: Path) -> tuple[float, float] | None:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text().splitlines()):
        m = re.match(r"^\s*[Aa]verage\s+([0-9.]+)\s+([0-9.]+)", line.strip())
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def parse_nanoms_best(log_path: Path) -> tuple[float, int] | None:
    if not log_path.exists():
        return None
    best = None
    step = None
    for line in log_path.read_text().splitlines():
        sm = re.match(r"^\[lion-shallow/(?:warm|full)\] step\s+(\d+)", line)
        if sm:
            step = int(sm.group(1))
        sc = re.match(r"^\s*\[NanoMSMARCO\] NDCG@10\s+query_doc \(dense\)\s+:\s+([0-9.]+)", line)
        if sc and step is not None:
            score = float(sc.group(1))
            if best is None or score > best[0]:
                best = (score, step)
    return best


def write_report(
    path: Path,
    rows: list[dict],
    full_lion: tuple[float, float] | None,
) -> None:
    full_ndcg = full_lion[0] if full_lion else None

    lines = [
        "# Lion Shallow Spaced Models — MS-MARCO Dev Evaluation",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Architecture: `lion_shallow`, MSE alignment loss, spaced layer selection",
        "- Model: `hzeng/Lion-SP-1B-llama3-marco-mntp` (16 layers, LLaMA 3 1B)",
        "- Corpus: 200k-passage MS-MARCO dev subset (all relevant passages included; full 8.8M build impractical at Lion's ~70 doc/s)",
        "",
        "## MS-MARCO Dev Results",
        "",
        "| Variant | Layers | NDCG@10 | % of full | MRR@10 | % of full | Best NanoMSMARCO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    if full_lion:
        lines.append(
            f"| **Lion (full)** | all 16 | **{full_lion[0]:.4f}** | 100.0% "
            f"| **{full_lion[1]:.4f}** | 100.0% | — |"
        )

    for row in rows:
        ms = row.get("msmarco")
        best = row.get("best_nanoms")
        ndcg_pct = f"{ms[0]/full_ndcg*100:.1f}%" if ms and full_ndcg else "—"
        mrr_pct  = f"{ms[1]/full_lion[1]*100:.1f}%" if ms and full_lion else "—"
        ndcg_str = f"{ms[0]:.4f}" if ms else "—"
        mrr_str  = f"{ms[1]:.4f}" if ms else "—"
        best_str = f"{best[0]:.4f} @ step {best[1]}" if best else "—"
        lines.append(
            f"| {row['name']} | `{row['layer_indices']}` | {ndcg_str} | {ndcg_pct} "
            f"| {mrr_str} | {mrr_pct} | {best_str} |"
        )

    lines += ["", "## BEIR Results", "",
              "| Variant | NDCG@10 avg | MRR@10 avg |",
              "|---|---:|---:|"]
    for row in rows:
        beir = row.get("beir")
        lines.append(
            f"| {row['name']} | "
            f"{beir[0]:.4f} | {beir[1]:.4f} |" if beir else f"| {row['name']} | — | — |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--work_dir", default="runs/lion_spaced_msmarco")
    parser.add_argument("--results", default="results/lion_spaced_msmarco_eval.md")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_beir", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    results_path = root / args.results
    base_cfg = yaml.safe_load(Path(args.config).read_text())

    variants = VARIANTS
    if args.only:
        names = set(args.only)
        variants = [v for v in VARIANTS if v["name"] in names]

    # Lion (1B LLaMA) encodes at ~70 docs/sec — building an 8.8M-passage index
    # would take ~35h. We use the 200k-passage dev subset instead (already
    # cached at data/msmarco_dev_subset_200000.pkl). All relevant passages are
    # always included, so NDCG/MRR are valid for cross-model comparison.
    print("Using 200k-passage MS-MARCO dev subset (full index build ~35h on Lion, not practical).")

    rows: list[dict] = []
    full_lion: tuple[float, float] | None = None

    for variant in variants:
        sc = copy.deepcopy(base_cfg["lion_shallow"])
        sc["layer_indices"] = variant["layer_indices"]
        sc.pop("n_layers", None)
        sc["factorize_embeddings"] = False
        sc["freeze_head_after_warmup"] = True
        sc["output_dir"] = variant["output_dir"]
        cfg = copy.deepcopy(base_cfg)
        cfg["lion_shallow"] = sc

        cfg_path = work_dir / f"config_{variant['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        checkpoint = root / "checkpoints_lion_shallow" / variant["output_dir"] / "best_NanoMSMARCO.pt"
        train_log = work_dir / f"{variant['name']}_train.log"
        beir_log  = work_dir / f"{variant['name']}_beir.log"
        ms_log    = work_dir / f"{variant['name']}_msmarco.log"

        row: dict = {
            "name": variant["name"],
            "layer_indices": variant["layer_indices"],
            "best_nanoms": None,
            "beir": None,
            "msmarco": None,
        }

        # ── Train ────────────────────────────────────────────────────────────
        if not args.skip_train:
            if checkpoint.exists():
                print(f"[skip train] {variant['name']}: checkpoint exists")
            else:
                t0 = time.time()
                code = run_command(
                    [sys.executable, "train.py", "lion_shallow", "--config", str(cfg_path)],
                    train_log,
                )
                print(f"  [{variant['name']}] train done in {(time.time()-t0)/3600:.1f}h")
                if code != 0:
                    rows.append(row)
                    write_report(results_path, rows, full_lion)
                    return code

        # Use existing lion_shallow_layer_sweep train log if our log doesn't exist
        existing_train_log = root / "runs/lion_shallow_layer_sweep" / f"{variant['name']}_train.log"
        for tlog in [train_log, existing_train_log]:
            best = parse_nanoms_best(tlog)
            if best:
                row["best_nanoms"] = best
                break

        # ── BEIR eval ────────────────────────────────────────────────────────
        if not args.skip_beir:
            if log_is_complete(beir_log, "Average"):
                print(f"[skip beir] {variant['name']}: already complete")
            else:
                # Check existing beir log from the original layer sweep
                existing_beir = root / "runs/lion_shallow_layer_sweep" / f"{variant['name']}_beir.log"
                if log_is_complete(existing_beir, "Average"):
                    print(f"[skip beir] {variant['name']}: using existing log")
                    beir_log = existing_beir
                else:
                    code = run_command(
                        [
                            sys.executable, "scripts/eval_beir.py",
                            "--stage", "lion_shallow",
                            "--config", str(cfg_path),
                            "--checkpoint", str(checkpoint),
                            "--doc_batch_size", "4",
                            "--datasets", *BEIR_DATASETS,
                        ],
                        beir_log,
                    )
                    if code != 0:
                        rows.append(row)
                        write_report(results_path, rows, full_lion)
                        return code

        # Re-check existing beir log
        for blog in [beir_log, root / "runs/lion_shallow_layer_sweep" / f"{variant['name']}_beir.log"]:
            beir = parse_beir_avg(blog)
            if beir:
                row["beir"] = beir
                break

        # ── MS-MARCO eval ────────────────────────────────────────────────────
        if log_is_complete(ms_log, "NDCG@10"):
            print(f"[skip msmarco] {variant['name']}: already complete")
        else:
            code = run_command(
                [
                    sys.executable, "scripts/eval_msmarco.py",
                    "--stage", "lion_shallow",
                    "--checkpoint", str(checkpoint),
                    "--config", str(cfg_path),
                    "--max_corpus_size", "200000",  # subset: full 8.8M index takes ~35h to build
                    "--doc_batch_size", "16",
                ],
                ms_log,
            )
            if code != 0:
                rows.append(row)
                write_report(results_path, rows, full_lion)
                return code

        row["msmarco"] = parse_msmarco(ms_log)
        rows.append(row)
        write_report(results_path, rows, full_lion)

    # ── Full Lion doc-only ceiling ────────────────────────────────────────────
    full_log = work_dir / "lion_full_msmarco.log"
    if log_is_complete(full_log, "NDCG@10"):
        print("[skip] Full Lion ceiling: already complete")
    else:
        dummy_cfg = copy.deepcopy(base_cfg)
        dummy_cfg["lion_shallow"]["layer_indices"] = [0]
        dummy_cfg["lion_shallow"].pop("n_layers", None)
        dummy_cfg_path = work_dir / "config_full_ceiling.yaml"
        dummy_cfg_path.write_text(yaml.safe_dump(dummy_cfg, sort_keys=False))
        code = run_command(
            [
                sys.executable, "scripts/eval_msmarco.py",
                "--stage", "lion_shallow",
                "--doc_only",
                "--config", str(dummy_cfg_path),
                "--max_corpus_size", "200000",
                "--doc_batch_size", "16",
            ],
            full_log,
        )
        if code != 0:
            write_report(results_path, rows, full_lion)
            return code

    full_lion = parse_msmarco(full_log)
    write_report(results_path, rows, full_lion)
    print(f"\nAll done. Results at {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
