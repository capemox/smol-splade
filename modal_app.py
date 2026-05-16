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

USAGE — baseline (lion_shallow_factorized_align):

    modal run --detach modal_app.py::train_lion
    modal run --detach modal_app.py::eval_lion
    modal run --detach modal_app.py::eval_lion_ceiling

USAGE — any other stage (after adding it to config.yaml + train.py + eval_msmarco.py):

    modal run --detach modal_app.py::train_stage --stage lion_shallow_factorized_spaced5_align
    modal run --detach modal_app.py::eval_stage \\
        --stage lion_shallow_factorized_spaced5_align \\
        --checkpoint checkpoints_lion_shallow/lion_shallow_factorized_spaced5_align/align_final.pt
    modal run --detach modal_app.py::eval_stage_ceiling --stage lion_shallow_factorized_spaced5_align

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
TELEGRAM_SECRET_NAME = "telegram-bot"

# Paths inside the running container:
REPO_SRC = "/repo"
VOLUME_MOUNT = "/vol"
WORK_DIR = f"{VOLUME_MOUNT}/work/sae-smo-splade"
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"

GPU_TYPE = "L40S"

HOUR = 60 * 60
TIMEOUT_LONG = 24 * HOUR
TIMEOUT_SHORT = 30 * 60

# ── Modal app, image, volume, secrets ────────────────────────────────────────

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
telegram_secret = modal.Secret.from_name(TELEGRAM_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "rsync", "curl", "ca-certificates")
    .pip_install("uv")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
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

RUNTIME_ENV = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_CACHE": f"{HF_CACHE}/hub",
    "HF_DATASETS_CACHE": f"{HF_CACHE}/datasets",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


# ── Notifications ────────────────────────────────────────────────────────────

def notify(message: str) -> None:
    import os
    import urllib.parse
    import urllib.request

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[notify skipped — no creds] {message[:200]}", flush=True)
        return

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
        print(f"[notify failed: {exc}]", flush=True)


def with_notifications(job_label):
    """Decorator: ping Telegram on start, success, and failure.
    job_label can be a string or a callable(fn_args, fn_kwargs) -> string for
    dynamic labels (e.g. including the stage name being trained)."""
    import functools
    import time
    import traceback

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            label = job_label(args, kwargs) if callable(job_label) else job_label
            t0 = time.time()
            notify(f"🚀 {label} started on Modal ({APP_NAME})")
            try:
                result = fn(*args, **kwargs)
                mins = (time.time() - t0) / 60
                notify(f"✅ {label} completed in {mins:.1f} min")
                return result
            except BaseException:
                mins = (time.time() - t0) / 60
                tb = traceback.format_exc()
                notify(
                    f"❌ {label} failed after {mins:.1f} min\n\n"
                    f"{tb[-2500:]}"
                )
                raise
        return wrapper
    return decorator


def _run(cmd: list) -> None:
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
    `uv sync` to refresh project dependencies. Idempotent."""
    import os
    import subprocess

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)

    subprocess.run([
        "rsync", "-a", "--delete",
        "--exclude=.git", "--exclude=.venv",
        "--exclude=__pycache__", "--exclude=*.pyc",
        "--exclude=checkpoints_*", "--exclude=data",
        f"{REPO_SRC}/", f"{WORK_DIR}/",
    ], check=True)

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


# ── Index build (one-time) ───────────────────────────────────────────────────

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
    """Embed full 8.84M-passage MS MARCO corpus with frozen Lion-SP-1B."""
    _run([
        "scripts/build_msmarco_index.py",
        "--stage", "lion_shallow_factorized_align",
        "--config", "config.yaml",
        "--index_dir", f"{VOLUME_MOUNT}/indexes/msmarco_lion_index",
        "--batch_size", "64",
        "--shard_size", "50000",
    ])
    volume.commit()


# ── Baseline (kept for backward compatibility) ───────────────────────────────

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
    _run(["train.py", "lion_shallow_factorized_align", "--config", "config.yaml"])
    volume.commit()


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


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=6 * HOUR,
)
@with_notifications("eval_lion_ceiling")
def eval_lion_ceiling():
    """Teacher-as-query ceiling for the baseline stage."""
    _run([
        "scripts/eval_msmarco.py",
        "--stage", "lion_shallow_factorized_align",
        "--doc_only",
        "--config", "config.yaml",
        "--index_dir", f"{VOLUME_MOUNT}/indexes/msmarco_lion_index",
        "--encode_batch_size", "8",
        "--query_batch_size", "128",
        "--densify_chunk", "1024",
    ])
    volume.commit()


# ── Parameterized stage runners (for experiments) ────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
@with_notifications(lambda a, kw: f"train_stage[{kw.get('stage', a[0] if a else '?')}]")
def train_stage(stage: str = "lion_shallow_factorized_align"):
    """Train any stage by name. Stage must be defined in config.yaml AND in
    train.py's choices list + elif routing."""
    _run(["train.py", stage, "--config", "config.yaml"])
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=6 * HOUR,
)
@with_notifications(lambda a, kw: f"eval_stage[{kw.get('stage', a[0] if a else '?')}]")
def eval_stage(stage: str, checkpoint: str):
    """Eval any stage against the prebuilt Lion index. Lion-derived stages all
    share the same index since they share the same teacher doc encoder."""
    _run([
        "scripts/eval_msmarco.py",
        "--stage", stage,
        "--checkpoint", checkpoint,
        "--config", "config.yaml",
        "--index_dir", f"{VOLUME_MOUNT}/indexes/msmarco_lion_index",
        "--encode_batch_size", "8",
        "--query_batch_size", "128",
        "--densify_chunk", "1024",
    ])
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=6 * HOUR,
)
@with_notifications(lambda a, kw: f"eval_stage_ceiling[{kw.get('stage', a[0] if a else '?')}]")
def eval_stage_ceiling(stage: str):
    """Teacher-as-query ceiling for the given stage (--doc_only)."""
    _run([
        "scripts/eval_msmarco.py",
        "--stage", stage,
        "--doc_only",
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
    """Default entrypoint runs setup_workspace. For everything else, invoke
    functions explicitly, e.g.:

        modal run --detach modal_app.py::train_stage --stage lion_shallow_factorized_spaced5_align
    """
    setup_workspace.remote()
