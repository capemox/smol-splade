"""
Modal app for sae-smo-splade.

Wraps the repo's CLI scripts as Modal functions so jobs run unattended on
serverless GPUs. Launch any function with `--detach` and your laptop is free
to close; the job continues on Modal's servers, and Telegram notifies you on
start, success, and failure.

ONE-TIME SETUP (from the repo root, after `pip install modal` and `modal setup`):

    modal volume create --version=2 sae-smo-splade-vol
    modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
    modal secret create telegram-bot \
        TELEGRAM_BOT_TOKEN=123456:ABC-DEF... \
        TELEGRAM_CHAT_ID=987654321
    modal run modal_app.py::setup_workspace

After that, the typical sequence is:

    modal run --detach modal_app.py::gpu_smoke           # ~10 min sanity check
    modal run --detach modal_app.py::index_smoke         # ~5 min pipeline smoke
    modal run --detach modal_app.py::build_lion_index    # the big one (~8-12h)
    modal run --detach modal_app.py::train_lion          # the research run
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
TELEGRAM_SECRET_NAME = "alerts"

# Paths inside the running container:
REPO_SRC = "/repo"                                  # baked-in repo source (read-only)
VOLUME_MOUNT = "/vol"                               # persistent volume root
WORK_DIR = f"{VOLUME_MOUNT}/work/sae-smo-splade"    # writable repo copy on volume
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"               # HF models + datasets cache

# GPU choice. L40S is the project instructions' recommendation for Lion.
GPU_TYPE = "L40S"

HOUR = 60 * 60
TIMEOUT_LONG = 24 * HOUR    # Modal's hard ceiling per function call
TIMEOUT_SHORT = 30 * 60

# ── Modal app, image, volume, secrets ────────────────────────────────────────

app = modal.App(APP_NAME)

# V2 volume — handles the larger directories the Lion index produces (~50-100 GB).
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
telegram_secret = modal.Secret.from_name(TELEGRAM_SECRET_NAME)

# Image: Debian slim + uv + the repo source. The heavy CUDA-torch deps install
# into the volume's venv on first `setup_workspace` call, not at image build,
# so image rebuilds stay cheap.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "rsync", "curl", "ca-certificates")
    .pip_install("uv")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # uv hardlinks between cache and venv when possible; on Modal those
        # live on different filesystems so we explicitly copy to silence the
        # warning we saw on the first setup_workspace run.
        "UV_LINK_MODE": "copy",
    })
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


# ── Notifications ────────────────────────────────────────────────────────────

def notify(message: str) -> None:
    """Send a Telegram message. Silent no-op if creds aren't present so the
    code stays usable even without the telegram-bot secret attached."""
    import os
    import urllib.parse
    import urllib.request

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[notify skipped — no creds] {message[:200]}", flush=True)
        return

    # Telegram caps each message at 4096 chars; leave headroom.
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message[:4000],
        "disable_web_page_preview": "true",
    }).encode()

    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=payload
        )
        urllib.request.urlopen(req, timeout=10).read()
        print("[notify ok]", flush=True)
    except Exception as exc:
        # Never let a notification failure crash the actual job.
        print(f"[notify failed: {exc}]", flush=True)


def with_notifications(job_name: str):
    """Decorator: ping Telegram on start, success, and failure (with traceback)."""
    import functools
    import time
    import traceback

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            notify(f"🚀 {job_name} started on Modal ({APP_NAME})")
            try:
                result = fn(*args, **kwargs)
                mins = (time.time() - t0) / 60
                notify(f"✅ {job_name} completed in {mins:.1f} min")
                return result
            except BaseException:
                mins = (time.time() - t0) / 60
                tb = traceback.format_exc()
                # Keep the tail of the traceback — where the actual error is.
                notify(
                    f"❌ {job_name} failed after {mins:.1f} min\n\n"
                    f"{tb[-2500:]}"
                )
                raise
        return wrapper
    return decorator


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


# ── Notification self-test ───────────────────────────────────────────────────

@app.function(
    image=image,
    secrets=[telegram_secret],
    timeout=60,
)
def notify_test():
    """Verifies the telegram-bot secret is wired correctly. Run this once after
    `modal secret create telegram-bot ...`. You should receive a Telegram
    message within a few seconds of launching."""
    notify("🔔 Test ping from Modal — telegram-bot secret is wired up correctly.")


# ── Smoke tests ──────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_SHORT,
)
@with_notifications("gpu_smoke")
def gpu_smoke():
    """End-to-end environment check: GPU visible, repo loads, models pull from HF."""
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
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_SHORT,
)
@with_notifications("index_smoke")
def index_smoke():
    """Tiny end-to-end index build (200 docs) to validate the pipeline."""
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
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
@with_notifications("build_lion_index")
def build_lion_index():
    """Embed the full 8.84M-passage MS MARCO corpus with the frozen Lion-SP-1B
    document encoder. Resumable: build_msmarco_index.py checkpoints by shard,
    so retries (and re-invocations) continue from the last completed shard."""
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
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
@with_notifications("train_lion")
def train_lion():
    """Train lion_shallow_factorized_align. Checkpoints land on the volume at
    checkpoints_lion_shallow/lion_shallow_factorized_align/."""
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
    secrets=[hf_secret, telegram_secret],
    timeout=6 * HOUR,
)
@with_notifications("eval_lion")
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
    """Default entrypoint. Runs setup_workspace. For everything else,
    invoke functions explicitly, e.g.:

        modal run --detach modal_app.py::train_lion
    """
    setup_workspace.remote()
