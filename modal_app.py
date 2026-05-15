"""
Modal app for sae-smo-splade.

Wraps the repo's CLI scripts as Modal functions so jobs run unattended on
serverless GPUs. Launch any function with `--detach` and your laptop is free
to close; the job continues on Modal's servers.

ONE-TIME SETUP (from the repo root, after `pip install modal` and `modal setup`):

    modal volume create --version=2 sae-smo-splade-vol
    modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
    modal run modal_app.py::setup_workspace

After that, the typical sequence is:

    modal run --detach modal_app.py::gpu_smoke           # ~10 min sanity check
    modal run --detach modal_app.py::index_smoke         # ~5 min pipeline smoke
    modal run --detach modal_app.py::build_lion_index    # the big one (~12-24h)
    modal run --detach modal_app.py::train_lion          # 50k training steps
    modal run --detach modal_app.py::eval_lion           # full MS MARCO dev

Tail logs at any time (works from any machine):

    modal app logs sae-smo-splade

The working tree (including checkpoints) lives at /vol/work/sae-smo-splade on
the persistent Modal Volume. Re-run setup_workspace after `git pull` or local
edits — it's idempotent.
"""

import modal

# ── Constants ────────────────────────────────────────────────────────────────

APP_NAME = "sae-smo-splade"
VOLUME_NAME = "sae-smo-splade-vol"
HF_SECRET_NAME = "huggingface-token"

# Paths inside the running container:
REPO_SRC = "/repo"                                  # baked-in repo source (read-only)
VOLUME_MOUNT = "/vol"                               # persistent volume root
WORK_DIR = f"{VOLUME_MOUNT}/work/sae-smo-splade"    # writable repo copy on volume
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"               # HF models + datasets cache

# GPU choice. L40S is the project instructions' recommendation for Lion;
# A100 if you can spare the cost. A10G is tighter on memory.
GPU_TYPE = "L40S"

HOUR = 60 * 60
TIMEOUT_LONG = 24 * HOUR    # Modal's hard ceiling per function call
TIMEOUT_SHORT = 30 * 60

# ── Modal app, image, volume, secret ─────────────────────────────────────────

app = modal.App(APP_NAME)

# V2 volume — handles the larger directories the Lion index produces (~50-100 GB).
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

# HF token, created once via:
#   modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxx
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

# Image: Debian slim + uv + the repo source. The heavy CUDA-torch deps install
# into the volume's venv on first `setup_workspace` call, not at image build,
# so image rebuilds stay cheap.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "rsync", "curl", "ca-certificates")
    .pip_install("uv")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        ".",
        remote_path=REPO_SRC,
        ignore=[
            ".git", ".venv", "__pycache__",
            "*.pyc", "*.pyo", ".DS_Store",
            "checkpoints_*", "data",
            "logs_*.txt",
        ],
    )
)

# Env every script invocation needs.
RUNTIME_ENV = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_CACHE": f"{HF_CACHE}/hub",
    "HF_DATASETS_CACHE": f"{HF_CACHE}/datasets",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def _run(cmd: list[str]) -> None:
    """Invoke `uv run <cmd>` inside the workspace directory."""
    import os
    import subprocess
    env = {**os.environ, **RUNTIME_ENV}
    print(f"+ uv run {' '.join(cmd)}", flush=True)
    subprocess.run(["uv", "run", *cmd], cwd=WORK_DIR, env=env, check=True)


# ── Workspace setup ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=TIMEOUT_SHORT,
)
def setup_workspace():
    """Sync the image's baked-in repo source into the persistent volume, then
    `uv sync` to install/refresh project dependencies inside the volume's venv.
    Idempotent — safe to re-run after `git pull` or local edits."""
    import os
    import subprocess

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)

    # Copy repo source into the volume. --delete keeps it in sync but excludes
    # paths that the volume itself owns (caches, checkpoints, data).
    subprocess.run([
        "rsync", "-a", "--delete",
        "--exclude=.git", "--exclude=.venv",
        "--exclude=__pycache__", "--exclude=*.pyc",
        "--exclude=checkpoints_*", "--exclude=data",
        f"{REPO_SRC}/", f"{WORK_DIR}/",
    ], check=True)

    # Build/refresh the venv inside the volume so subsequent runs reuse it.
    subprocess.run(["uv", "sync"], cwd=WORK_DIR, check=True)

    volume.commit()
    print(f"✓ Workspace ready at {WORK_DIR}", flush=True)


# ── Smoke tests ──────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=TIMEOUT_SHORT,
)
def gpu_smoke():
    """End-to-end environment check: GPU visible, repo loads, models pull from HF.
    First run is slower (model downloads); subsequent runs are 1-2 minutes once
    the HF cache is warm."""
    import subprocess
    subprocess.run(["nvidia-smi"], check=True)
    _run(["smoke_test.py"])
    _run(["scripts/build_msmarco_index.py", "--help"])
    _run(["scripts/eval_msmarco.py", "--help"])
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=TIMEOUT_SHORT,
)
def index_smoke():
    """Tiny end-to-end index build (200 docs) to validate the full pipeline
    before launching the multi-hour real build."""
    _run([
        "scripts/build_msmarco_index.py",
        "--stage", "lion_shallow_factorized_align",
        "--config", "config.yaml",
        "--index_dir", f"{VOLUME_MOUNT}/indexes/msmarco_lion_index_smoke",
        "--batch_size", "8",
        "--shard_size", "100",
        "--limit", "200",
    ])
    volume.commit()


# ── Full Lion MS MARCO index build ───────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
def build_lion_index():
    """Embed the full 8.84M-passage MS MARCO corpus with the frozen Lion-SP-1B
    document encoder. Resumable: build_msmarco_index.py checkpoints by shard,
    so retries (and re-invocations) continue from the last completed shard.
    Plan ~12-24h on an L40S."""
    _run([
        "scripts/build_msmarco_index.py",
        "--stage", "lion_shallow_factorized_align",
        "--config", "config.yaml",
        "--index_dir", f"{VOLUME_MOUNT}/indexes/msmarco_lion_index",
        "--batch_size", "64",
        "--shard_size", "50000",
    ])
    volume.commit()


# ── Training ─────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
def train_lion():
    """Train lion_shallow_factorized_align for the configured number of steps.
    The repo's training loop writes checkpoints every save_every steps under
    checkpoints_lion_shallow/lion_shallow_factorized_align/, which is on the
    persistent volume."""
    _run([
        "train.py", "lion_shallow_factorized_align",
        "--config", "config.yaml",
    ])
    volume.commit()


# ── Evaluation ───────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=6 * HOUR,
)
def eval_lion(
    checkpoint: str = "checkpoints_lion_shallow/lion_shallow_factorized_align/align_final.pt",
):
    """Full MS MARCO dev evaluation against the prebuilt Lion index. Pass a
    different relative checkpoint path to score another checkpoint."""
    _run([
        "scripts/eval_msmarco.py",
        "--stage", "lion_shallow_factorized_align",
        "--checkpoint", checkpoint,
        "--config", "config.yaml",
        "--index_dir", f"{VOLUME_MOUNT}/indexes/msmarco_lion_index",
        "--encode_batch_size", "8",
        "--query_batch_size", "128",
        "--densify_chunk", "1024",
    ])
    volume.commit()


# ── Default entrypoint ───────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Default entrypoint. Runs `setup_workspace`. For everything else,
    invoke functions explicitly, e.g.:

        modal run --detach modal_app.py::build_lion_index
    """
    setup_workspace.remote()
