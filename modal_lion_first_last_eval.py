#!/usr/bin/env python
"""Train Lion first1-5 and last1-5 shallow models + eval on full MS-MARCO dev (Modal).

first1/2 and last1/2 checkpoints are pre-uploaded to the volume — training is
skipped for those. first3-5 and last3-5 are trained from scratch.

  first1: layers [0]
  first2: layers [0, 1]
  first3: layers [0, 1, 2]
  first4: layers [0, 1, 2, 3]
  first5: layers [0, 1, 2, 3, 4]
  last1:  layers [15]
  last2:  layers [14, 15]
  last3:  layers [13, 14, 15]
  last4:  layers [12, 13, 14, 15]
  last5:  layers [11, 12, 13, 14, 15]

Usage:
  modal run --detach modal_lion_first_last_eval.py::run_sweep
  modal logs <app-id>   # follow progress
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time

import modal

# ── Modal constants ───────────────────────────────────────────────────────────

APP_NAME    = "lion-first-last-sweep"
VOLUME_NAME = "sae-smo-splade-vol"
VOLUME_MOUNT = "/vol"
WORK_DIR    = "/vol/work/sae-smo-splade"
HF_CACHE    = "/vol/hf_cache"
REPO_SRC    = "/repo"
INDEX_DIR   = "/vol/indexes/msmarco_lion_index"
RESULTS_FILE = "results/lion_first_last_msmarco_eval.md"
LOGS_DIR    = "runs/lion_first_last_modal"

RUNTIME_ENV = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_CACHE": f"{HF_CACHE}/hub",
    "HF_DATASETS_CACHE": f"{HF_CACHE}/datasets",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}

# ── Variant definitions ───────────────────────────────────────────────────────

LION_N_LAYERS = 16

VARIANTS = [
    {"name": "first1", "layer_indices": [0]},
    {"name": "first2", "layer_indices": [0, 1]},
    {"name": "first3", "layer_indices": [0, 1, 2]},
    {"name": "first4", "layer_indices": [0, 1, 2, 3]},
    {"name": "first5", "layer_indices": [0, 1, 2, 3, 4]},
    {"name": "last1",  "layer_indices": [15]},
    {"name": "last2",  "layer_indices": [14, 15]},
    {"name": "last3",  "layer_indices": [13, 14, 15]},
    {"name": "last4",  "layer_indices": [12, 13, 14, 15]},
    {"name": "last5",  "layer_indices": [11, 12, 13, 14, 15]},
]

for v in VARIANTS:
    v["output_dir"] = f"lion_shallow_{v['name']}"

# ── Modal app + image ─────────────────────────────────────────────────────────

app    = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rsync", "git")
    .pip_install("uv", "pyyaml")
    .add_local_dir(
        ".",
        remote_path=REPO_SRC,
        ignore=[
            "**/__pycache__", "**/.venv", ".git",
            "data/**", "runs/**", "results/**",
            "checkpoints_*/**", "*.pt",
        ],
    )
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _env() -> dict:
    return {**os.environ, **RUNTIME_ENV}


def _venv_py() -> str:
    return f"{WORK_DIR}/.venv/bin/python"


def _run(cmd: list[str], log_path: str | None = None) -> int:
    full_cmd = [_venv_py(), *cmd]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                full_cmd, cwd=WORK_DIR, env=_env(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
        rc = proc.wait()
    else:
        rc = subprocess.run(full_cmd, cwd=WORK_DIR, env=_env(), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {' '.join(full_cmd)}")
    return rc


def _sync_code() -> None:
    subprocess.run(
        [
            "rsync", "-a", "--delete",
            "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__",
            "--exclude=data", "--exclude=runs", "--exclude=results",
            "--exclude=checkpoints_*",
            f"{REPO_SRC}/", f"{WORK_DIR}/",
        ],
        check=True,
    )
    print(f"Code synced {REPO_SRC} → {WORK_DIR}")


def _ensure_venv() -> None:
    venv_py = _venv_py()
    if os.path.exists(venv_py):
        rc = subprocess.run(
            [venv_py, "-c", "import torch; print(torch.__version__)"],
            cwd=WORK_DIR, env=_env(), capture_output=True,
        ).returncode
        if rc == 0:
            print("venv OK — skipping uv sync")
            return
    print("Installing dependencies via uv sync ...")
    subprocess.run(["uv", "sync", "--no-dev"], cwd=WORK_DIR, env=_env(), check=True)


def _write_variant_config(variant: dict) -> str:
    import yaml
    base = yaml.safe_load(open(f"{WORK_DIR}/config.yaml"))
    sc = copy.deepcopy(base["lion_shallow"])
    sc["layer_indices"] = variant["layer_indices"]
    sc.pop("n_layers", None)
    sc["factorize_embeddings"] = False
    sc["freeze_head_after_warmup"] = True
    sc["output_dir"] = variant["output_dir"]
    base["lion_shallow"] = sc
    cfg_path = f"{WORK_DIR}/{LOGS_DIR}/config_{variant['name']}.yaml"
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return cfg_path


def _checkpoint_path(variant: dict) -> str:
    return f"{WORK_DIR}/checkpoints_lion_shallow/{variant['output_dir']}/best_NanoMSMARCO.pt"


def _parse_msmarco(log_path: str) -> tuple[float, float] | None:
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    m_n = re.search(r"NDCG@10\s*:\s*([0-9.]+)", text)
    m_m = re.search(r"MRR@10\s*:\s*([0-9.]+)", text)
    return (float(m_n.group(1)), float(m_m.group(1))) if m_n and m_m else None


def _parse_best_nano(log_path: str) -> tuple[float, int] | None:
    if not os.path.exists(log_path):
        return None
    best, step = None, None
    for line in open(log_path):
        sm = re.match(r"^\[lion-shallow/(?:warm|full)\] step\s+(\d+)", line)
        if sm:
            step = int(sm.group(1))
        sc = re.match(
            r"^\s*\[NanoMSMARCO\] NDCG@10\s+query_doc \(dense\)\s+:\s+([0-9.]+)", line
        )
        if sc and step is not None:
            score = float(sc.group(1))
            if best is None or score > best[0]:
                best = (score, step)
    return best


def _write_report(rows: list[dict], full_lion: tuple[float, float] | None) -> None:
    full_ndcg = full_lion[0] if full_lion else None
    full_mrr  = full_lion[1] if full_lion else None

    lines = [
        "# Lion Shallow First/Last Models — MS-MARCO Dev Evaluation",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')} (Modal L40S)",
        "",
        "- Architecture: `lion_shallow`, MSE alignment loss",
        "- Model: `hzeng/Lion-SP-1B-llama3-marco-mntp` (16 layers, LLaMA 3 1B)",
        "- Corpus: 8,841,823-passage MS-MARCO (full, pre-built Lion index)",
        f"- Index: `{INDEX_DIR}`",
        "",
        "## MS-MARCO Dev Results",
        "",
        "| Variant | Layers | NDCG@10 | % of full | MRR@10 | % of full | Best NanoMSMARCO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    if full_lion:
        lines.append(
            f"| **Lion (full)** | all 16 | **{full_lion[0]:.4f}** | 100.0%"
            f" | **{full_lion[1]:.4f}** | 100.0% | — |"
        )

    for row in rows:
        ms   = row.get("msmarco")
        best = row.get("best_nano")
        ndcg_pct = f"{ms[0]/full_ndcg*100:.1f}%" if ms and full_ndcg else "—"
        mrr_pct  = f"{ms[1]/full_mrr*100:.1f}%"  if ms and full_mrr  else "—"
        ndcg_s   = f"{ms[0]:.4f}" if ms else "—"
        mrr_s    = f"{ms[1]:.4f}" if ms else "—"
        best_s   = f"{best[0]:.4f} @ step {best[1]}" if best else "—"
        lines.append(
            f"| {row['name']} | `{row['layer_indices']}` | {ndcg_s} | {ndcg_pct}"
            f" | {mrr_s} | {mrr_pct} | {best_s} |"
        )

    out_path = f"{WORK_DIR}/{RESULTS_FILE}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    print(f"\nResults written → {out_path}")


# ── Main Modal function ───────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    volumes={VOLUME_MOUNT: volume},
    timeout=12 * 3600,  # 12h: 6 new trains (~1h each) + 10 evals (~20min each)
    memory=32768,
)
def run_sweep() -> None:
    """Train Lion first/last 1-5 and eval all against the full MS-MARCO Lion index."""

    print("=" * 70)
    print("Lion first/last layer sweep — Modal L40S")
    print(f"Index: {INDEX_DIR}")
    print(f"Variants: {[v['name'] for v in VARIANTS]}")
    print("=" * 70)

    _sync_code()
    _ensure_venv()

    # Verify index exists before expensive training
    manifest_path = f"{INDEX_DIR}/manifest.json"
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"Lion MS-MARCO index not found at {INDEX_DIR}.")
    manifest = json.load(open(manifest_path))
    n_shards = len(list(__import__("pathlib").Path(INDEX_DIR).glob("shard_*.npz")))
    print(f"Index: {manifest['target_total_docs']:,} docs, vocab={manifest['vocab_size']}, {n_shards} shards")

    rows: list[dict] = []
    full_lion: tuple[float, float] | None = None

    for variant in VARIANTS:
        print(f"\n{'─'*60}")
        print(f"[{variant['name']}] layers={variant['layer_indices']}")
        print(f"{'─'*60}")

        cfg_path   = _write_variant_config(variant)
        checkpoint = _checkpoint_path(variant)
        train_log  = f"{WORK_DIR}/{LOGS_DIR}/{variant['name']}_train.log"
        eval_log   = f"{WORK_DIR}/{LOGS_DIR}/{variant['name']}_msmarco.log"

        row: dict = {
            "name": variant["name"],
            "layer_indices": variant["layer_indices"],
            "best_nano": None,
            "msmarco": None,
        }

        # Train (skip if checkpoint already exists — first1/2, last1/2 were pre-uploaded)
        if os.path.exists(checkpoint):
            print(f"  [skip train] checkpoint found: {checkpoint}")
        else:
            print(f"  [train] starting {variant['name']} ...")
            t0 = time.time()
            _run(["train.py", "lion_shallow", "--config", cfg_path], log_path=train_log)
            print(f"  [train] done in {(time.time()-t0)/60:.1f} min")
            volume.commit()

        row["best_nano"] = _parse_best_nano(train_log)

        # MS-MARCO eval
        if _parse_msmarco(eval_log):
            print(f"  [skip eval] results already in {eval_log}")
        else:
            print(f"  [eval] running MS-MARCO dev eval ...")
            t0 = time.time()
            _run(
                [
                    "scripts/eval_msmarco.py",
                    "--stage", "lion_shallow",
                    "--checkpoint", checkpoint,
                    "--config", cfg_path,
                    "--index_dir", INDEX_DIR,
                    "--encode_batch_size", "32",
                    "--query_batch_size", "500",
                    "--densify_chunk", "4096",
                ],
                log_path=eval_log,
            )
            print(f"  [eval] done in {(time.time()-t0)/60:.1f} min")
            volume.commit()

        row["msmarco"] = _parse_msmarco(eval_log)
        rows.append(row)
        _write_report(rows, full_lion)
        volume.commit()

    # ── Full Lion ceiling ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[ceiling] Full Lion doc encoder — upper bound")
    print(f"{'─'*60}")

    import yaml
    ceiling_cfg_path = f"{WORK_DIR}/{LOGS_DIR}/config_ceiling.yaml"
    base = yaml.safe_load(open(f"{WORK_DIR}/config.yaml"))
    sc = copy.deepcopy(base["lion_shallow"])
    sc["layer_indices"] = [0]
    sc.pop("n_layers", None)
    base["lion_shallow"] = sc
    os.makedirs(os.path.dirname(ceiling_cfg_path), exist_ok=True)
    with open(ceiling_cfg_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)

    ceiling_log = f"{WORK_DIR}/{LOGS_DIR}/ceiling_msmarco.log"
    if _parse_msmarco(ceiling_log):
        print("  [skip ceiling] already have results")
    else:
        print("  [ceiling] running ...")
        t0 = time.time()
        _run(
            [
                "scripts/eval_msmarco.py",
                "--stage", "lion_shallow",
                "--doc_only",
                "--config", ceiling_cfg_path,
                "--index_dir", INDEX_DIR,
                "--encode_batch_size", "8",
                "--query_batch_size", "500",
                "--densify_chunk", "4096",
            ],
            log_path=ceiling_log,
        )
        print(f"  [ceiling] done in {(time.time()-t0)/60:.1f} min")
        volume.commit()

    full_lion = _parse_msmarco(ceiling_log)
    _write_report(rows, full_lion)
    volume.commit()

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    if full_lion:
        print(f"Full Lion ceiling: NDCG@10={full_lion[0]:.4f}  MRR@10={full_lion[1]:.4f}")
    for row in rows:
        ms = row.get("msmarco")
        if ms and full_lion:
            print(f"  {row['name']:8s}: NDCG@10={ms[0]:.4f} ({ms[0]/full_lion[0]*100:.1f}%)  MRR@10={ms[1]:.4f}")
        elif ms:
            print(f"  {row['name']:8s}: NDCG@10={ms[0]:.4f}  MRR@10={ms[1]:.4f}")
    print(f"Results → {WORK_DIR}/{RESULTS_FILE}")
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    run_sweep.remote()
