#!/usr/bin/env python
"""Run the complete CoCondenser SPLADE pruning experiment on one GPU VM.

The pipeline is resumable. Re-running it skips completed indexes, checkpoints,
MS MARCO evaluations, and BEIR evaluations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig


MODEL_ID = "naver/splade-cocondenser-ensembledistil"
MODEL_SLUG = MODEL_ID.rsplit("/", 1)[-1]
STAGE = "splade_shallow"
BEIR_DATASETS = [
    "nfcorpus",
    "scifact",
    "arguana",
    "scidocs",
    "fiqa",
    "trec-covid",
    "webis-touche2020",
    "quora",
    "nq",
    "dbpedia-entity",
    "hotpotqa",
    "fever",
    "climate-fever",
]
NDCG_RE = re.compile(r"NDCG@10\s*:\s*([0-9.]+)")
MRR_RE = re.compile(r"MRR@10\s*:\s*([0-9.]+)")
BEIR_HEADER_RE = re.compile(r"^\[([^]]+)]\s*$")
BEIR_SCORE_RE = re.compile(r"NDCG@10:\s*([0-9.]+)\s+MRR@10:\s*([0-9.]+)")


def spaced_layers(count: int, total: int) -> list[int]:
    if count == 1:
        return [round((total - 1) / 2)]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def build_variants(total_layers: int) -> list[dict]:
    variants = []
    for count in range(1, 6):
        selections = {
            "first": list(range(count)),
            "spaced": spaced_layers(count, total_layers),
            "last": list(range(total_layers - count, total_layers)),
        }
        for kind, indices in selections.items():
            name = f"{kind}{count}"
            variants.append(
                {
                    "name": name,
                    "count": count,
                    "kind": kind,
                    "layer_indices": indices,
                    "output_dir": f"cocondenser_shallow_{name}",
                }
            )
    return variants


def run_command(cmd: list[str], log_path: Path) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w") as log:
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
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command exited with status {code}: {' '.join(cmd)}")


def command_complete(log_path: Path, markers: list[str]) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="replace")
    return all(marker in text for marker in markers)


def index_complete(index_dir: Path, model_id: str) -> bool:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("doc_splade_hf_id") != model_id:
        return False
    target = int(manifest.get("target_total_docs", 0))
    if target <= 0:
        return False
    docs = 0
    for shard in sorted(index_dir.glob("shard_*.npz")):
        import numpy as np

        with np.load(shard, allow_pickle=True) as data:
            docs += len(data["docids"])
    return docs >= target


def parse_msmarco(log_path: Path) -> tuple[float, float] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    ndcg = NDCG_RE.search(text)
    mrr = MRR_RE.search(text)
    if not ndcg or not mrr:
        return None
    return float(ndcg.group(1)), float(mrr.group(1))


def parse_beir(log_path: Path) -> dict[str, tuple[float, float]]:
    if not log_path.exists():
        return {}
    results = {}
    current = None
    for line in log_path.read_text(errors="replace").splitlines():
        header = BEIR_HEADER_RE.match(line.strip())
        if header:
            current = header.group(1)
            continue
        score = BEIR_SCORE_RE.search(line)
        if current and score:
            results[current] = (float(score.group(1)), float(score.group(2)))
            current = None
    return results


def latest_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = []
    for path in checkpoint_dir.glob("align_step_*.pt"):
        match = re.search(r"align_step_(\d+)\.pt$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def prune_training_checkpoints(checkpoint_dir: Path, selected: Path) -> None:
    for path in checkpoint_dir.glob("align_step_*.pt"):
        path.unlink()
    final = checkpoint_dir / "align_final.pt"
    if final.exists() and final != selected:
        final.unlink()


def write_config(base: dict, variant: dict, path: Path) -> None:
    cfg = copy.deepcopy(base)
    sc = cfg[STAGE]
    sc["doc_splade_hf_id"] = MODEL_ID
    sc["layer_indices"] = variant["layer_indices"]
    sc.pop("n_layers", None)
    sc["factorize_embeddings"] = False
    sc["batch_size"] = 16
    sc["gradient_accumulation_steps"] = 2
    sc["alignment_steps"] = 30_000
    sc["align_loss_kind"] = "mse"
    sc["use_contrastive"] = False
    sc["contrastive_coeff"] = 0.0
    sc["freeze_head_after_warmup"] = True
    sc["save_every"] = 10_000
    sc["output_dir"] = variant["output_dir"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def write_report(
    path: Path,
    variants: list[dict],
    msmarco: dict[str, tuple[float, float]],
    winners: dict[int, dict],
    beir: dict[str, dict[str, tuple[float, float]]],
) -> None:
    lines = [
        "# CoCondenser SPLADE Pruning Experiment",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Model/document encoder: `{MODEL_ID}`",
        "- Training: 30,000 steps, batch size 16, gradient accumulation 2, MSE loss",
        "- Selection metric: MS MARCO dev NDCG@10",
        "",
        "## MS MARCO Dev",
        "",
        "| Variant | Layers | NDCG@10 | MRR@10 | Selected |",
        "|---|---|---:|---:|---:|",
    ]
    winner_names = {item["name"] for item in winners.values()}
    for variant in variants:
        score = msmarco.get(variant["name"])
        ndcg = f"{score[0]:.4f}" if score else "-"
        mrr = f"{score[1]:.4f}" if score else "-"
        selected = "yes" if variant["name"] in winner_names else ""
        lines.append(
            f"| {variant['name']} | `{variant['layer_indices']}` | {ndcg} | {mrr} | {selected} |"
        )

    if beir:
        baseline = beir.get("full", {})
        baseline_avg = (
            sum(score[0] for score in baseline.values()) / len(baseline)
            if baseline
            else None
        )
        lines.extend(
            [
                "",
                "## Full BEIR",
                "",
                "| Model | Layers | Average NDCG@10 | % of full | Average MRR@10 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        ordered = [winners[count]["name"] for count in sorted(winners)] + ["full"]
        for name in ordered:
            scores = beir.get(name, {})
            if not scores:
                lines.append(f"| {name} | - | - | - | - |")
                continue
            avg_ndcg = sum(score[0] for score in scores.values()) / len(scores)
            avg_mrr = sum(score[1] for score in scores.values()) / len(scores)
            pct = 100.0 * avg_ndcg / baseline_avg if baseline_avg else 0.0
            layers = "all" if name == "full" else str(next(v["layer_indices"] for v in variants if v["name"] == name))
            lines.append(
                f"| {name} | `{layers}` | {avg_ndcg:.4f} | {pct:.1f}% | {avg_mrr:.4f} |"
            )

        lines.extend(["", "### NDCG@10 by Dataset", ""])
        header = "| Dataset | " + " | ".join(ordered) + " |"
        separator = "|---|" + "---:|" * len(ordered)
        lines.extend([header, separator])
        for dataset in BEIR_DATASETS:
            values = []
            for name in ordered:
                score = beir.get(name, {}).get(dataset)
                values.append(f"{score[0]:.4f}" if score else "-")
            lines.append(f"| {dataset} | " + " | ".join(values) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def preflight(min_free_gb: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. This experiment requires a CUDA VM.")
    free = shutil.disk_usage(Path.cwd()).free
    free_gb = free / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free disk: {free_gb:.1f} GiB")
    if free_gb < min_free_gb:
        raise SystemExit(
            f"At least {min_free_gb} GiB free is required before starting; "
            f"only {free_gb:.1f} GiB is available. Increase the VM disk."
        )


def index_precision_args() -> list[str]:
    if torch.cuda.is_bf16_supported():
        return []
    print("GPU does not support BF16; index building will use FP32.")
    return ["--no_bf16"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--work-dir", default="runs/cocondenser_vm")
    parser.add_argument("--msmarco-index", default="data/cocondenser_msmarco_index")
    parser.add_argument("--beir-index", default="data/cocondenser_beir_index")
    parser.add_argument("--results", default="results/cocondenser_vm.md")
    parser.add_argument("--index-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--densify-chunk", type=int, default=4096)
    parser.add_argument("--min-free-gb", type=int, default=200)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    work_dir = root / args.work_dir
    msmarco_index = root / args.msmarco_index
    beir_index = root / args.beir_index
    results_path = root / args.results
    checkpoint_root = root / f"checkpoints_{MODEL_SLUG}"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        preflight(args.min_free_gb)

    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    total_layers = int(getattr(hf_config, "num_hidden_layers"))
    if total_layers < 5:
        raise SystemExit(f"Model has only {total_layers} layers; expected at least 5.")
    print(f"Model: {MODEL_ID} ({total_layers} layers)")

    base_cfg = yaml.safe_load(Path(args.base_config).read_text())
    variants = build_variants(total_layers)
    common_variant = variants[0]
    common_config = work_dir / "config_common.yaml"
    write_config(base_cfg, common_variant, common_config)

    # The frozen document encoder is identical for every query ablation.
    if not index_complete(msmarco_index, MODEL_ID):
        run_command(
            [
                sys.executable,
                "scripts/build_msmarco_index.py",
                "--stage",
                STAGE,
                "--config",
                str(common_config),
                "--index_dir",
                str(msmarco_index),
                "--batch_size",
                str(args.index_batch_size),
                *index_precision_args(),
            ],
            work_dir / "build_msmarco_index.log",
        )
    else:
        print(f"[skip] Complete MS MARCO index: {msmarco_index}")

    msmarco_results: dict[str, tuple[float, float]] = {}
    for variant in variants:
        config_path = work_dir / f"config_{variant['name']}.yaml"
        write_config(base_cfg, variant, config_path)
        checkpoint_dir = checkpoint_root / variant["output_dir"]
        best_checkpoint = checkpoint_dir / "best_NanoMSMARCO.pt"
        final_checkpoint = checkpoint_dir / "align_final.pt"
        training_marker = checkpoint_dir / ".training_complete"

        if not training_marker.exists() and final_checkpoint.exists():
            completed = torch.load(final_checkpoint, map_location="cpu").get("step", 0)
            if int(completed) >= 30_000:
                training_marker.write_text("30000\n")

        if not training_marker.exists():
            cmd = [sys.executable, "train.py", STAGE, "--config", str(config_path)]
            resume = latest_resume_checkpoint(checkpoint_dir)
            if resume is not None:
                cmd.extend(["--resume", str(resume)])
            run_command(cmd, work_dir / f"{variant['name']}_train.log")
            if not final_checkpoint.exists():
                raise RuntimeError(f"Training did not finish for {variant['name']}")
            training_marker.write_text("30000\n")

        selected_checkpoint = best_checkpoint if best_checkpoint.exists() else final_checkpoint
        if not selected_checkpoint.exists():
            raise RuntimeError(f"Training produced no checkpoint for {variant['name']}")

        eval_log = work_dir / f"{variant['name']}_msmarco.log"
        score = parse_msmarco(eval_log)
        if score is None:
            run_command(
                [
                    sys.executable,
                    "scripts/eval_msmarco.py",
                    "--stage",
                    STAGE,
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(selected_checkpoint),
                    "--index_dir",
                    str(msmarco_index),
                    "--encode_batch_size",
                    str(args.eval_batch_size),
                    "--densify_chunk",
                    str(args.densify_chunk),
                ],
                eval_log,
            )
            score = parse_msmarco(eval_log)
        if score is None:
            raise RuntimeError(f"Could not parse MS MARCO result for {variant['name']}")
        msmarco_results[variant["name"]] = score
        prune_training_checkpoints(checkpoint_dir, selected_checkpoint)
        print(f"[{variant['name']}] MS MARCO NDCG@10={score[0]:.4f}, MRR@10={score[1]:.4f}")

    winners: dict[int, dict] = {}
    for count in range(1, 6):
        candidates = [variant for variant in variants if variant["count"] == count]
        winners[count] = max(candidates, key=lambda item: msmarco_results[item["name"]][0])
        winner = winners[count]
        print(
            f"Winner for {count} layer(s): {winner['name']} "
            f"(NDCG@10={msmarco_results[winner['name']][0]:.4f})"
        )
    write_report(results_path, variants, msmarco_results, winners, {})

    missing_beir = [
        dataset
        for dataset in BEIR_DATASETS
        if not index_complete(beir_index / STAGE / dataset, MODEL_ID)
    ]
    if missing_beir:
        run_command(
            [
                sys.executable,
                "scripts/build_beir_index.py",
                "--stage",
                STAGE,
                "--config",
                str(common_config),
                "--index_dir",
                str(beir_index),
                "--batch_size",
                str(args.index_batch_size),
                "--datasets",
                *missing_beir,
                *index_precision_args(),
            ],
            work_dir / "build_beir_index.log",
        )
    else:
        print(f"[skip] All BEIR indexes complete: {beir_index}")

    beir_results: dict[str, dict[str, tuple[float, float]]] = {}
    for count in range(1, 6):
        variant = winners[count]
        config_path = work_dir / f"config_{variant['name']}.yaml"
        checkpoint_dir = checkpoint_root / variant["output_dir"]
        checkpoint = checkpoint_dir / "best_NanoMSMARCO.pt"
        if not checkpoint.exists():
            checkpoint = checkpoint_dir / "align_final.pt"
        log_path = work_dir / f"{variant['name']}_beir.log"
        parsed = parse_beir(log_path)
        if not all(dataset in parsed for dataset in BEIR_DATASETS):
            run_command(
                [
                    sys.executable,
                    "scripts/eval_beir.py",
                    "--stage",
                    STAGE,
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--index_dir",
                    str(beir_index),
                    "--datasets",
                    *BEIR_DATASETS,
                    "--encode_batch_size",
                    str(args.eval_batch_size),
                    "--densify_chunk",
                    str(args.densify_chunk),
                ],
                log_path,
            )
            parsed = parse_beir(log_path)
        beir_results[variant["name"]] = parsed
        write_report(results_path, variants, msmarco_results, winners, beir_results)

    full_log = work_dir / "full_beir.log"
    full_results = parse_beir(full_log)
    if not all(dataset in full_results for dataset in BEIR_DATASETS):
        run_command(
            [
                sys.executable,
                "scripts/eval_beir.py",
                "--stage",
                STAGE,
                "--config",
                str(common_config),
                "--doc_only",
                "--index_dir",
                str(beir_index),
                "--datasets",
                *BEIR_DATASETS,
                "--encode_batch_size",
                str(args.eval_batch_size),
                "--densify_chunk",
                str(args.densify_chunk),
            ],
            full_log,
        )
        full_results = parse_beir(full_log)
    beir_results["full"] = full_results
    write_report(results_path, variants, msmarco_results, winners, beir_results)

    print(f"\nExperiment complete. Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
