"""
Modal app for sae-smo-splade.

Wraps the repo's CLI scripts as Modal functions so jobs run unattended on
serverless GPUs. Launch any function with `--detach` and your laptop is free
to close; the job continues on Modal's servers, and Telegram notifies you on
start, success, and failure.

ONE-TIME SETUP (from the repo root, after `pip install modal` and `modal setup`):

    modal volume create --version=2 sae-smo-splade-vol
    modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
    modal secret create alerts \
        TELEGRAM_BOT_TOKEN=123456:ABC-DEF... \
        TELEGRAM_CHAT_ID=987654321
    modal run modal_app.py::setup_workspace

USAGE — Lion factorized stages (baseline + variants from earlier experiments):

    modal run --detach modal_app.py::train_stage --stage lion_shallow_factorized_align
    modal run --detach modal_app.py::eval_stage --stage lion_shallow_factorized_align \\
        --checkpoint checkpoints_lion_shallow/lion_shallow_factorized_align/align_final.pt
    modal run --detach modal_app.py::eval_stage_ceiling --stage lion_shallow_factorized_align

USAGE — Ettin SPLADE training (encoder doc-encoder, for downstream factorization):

    # Smoke test the pipeline (~10 min, ~$0.30):
    modal run --detach modal_app.py::ettin_splade_smoke

    # 150M hyperparameter sweep (~4h, ~$8 each):
    modal run --detach modal_app.py::train_ettin_splade --tag A_baseline
    modal run --detach modal_app.py::train_ettin_splade --tag B_higher_reg \\
        --query-reg 5e-4 --doc-reg 3e-4
    modal run --detach modal_app.py::train_ettin_splade --tag C_longer \\
        --max-steps 60000
    modal run --detach modal_app.py::train_ettin_splade --tag D_higher_lr \\
        --learning-rate 5e-5
    modal run --detach modal_app.py::train_ettin_splade --tag E_best \\
        --query-reg 5e-4 --doc-reg 3e-4 --max-steps 60000  # adapt to actual winners

    # Once 150M recipe is validated, scale up:
    modal run --detach modal_app.py::train_ettin_splade --model-size 400m \\
        --query-reg <winner> --doc-reg <winner> --max-steps <winner>
    modal run --detach modal_app.py::train_ettin_splade --model-size 1b \\
        --query-reg <winner> --doc-reg <winner> --max-steps <winner>

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

REPO_SRC = "/repo"
VOLUME_MOUNT = "/vol"
WORK_DIR = f"{VOLUME_MOUNT}/work/sae-smo-splade"
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"

GPU_TYPE = "L40S"

HOUR = 60 * 60
TIMEOUT_LONG = 24 * HOUR
TIMEOUT_SHORT = 30 * 60

# Map size → (HF model id, GPU type, default batch, default grad_accum).
# Defaults chosen so each model fits comfortably with effective batch ≈ 32.
ETTIN_PRESETS = {
    "150m": ("jhu-clsp/ettin-encoder-150m", "L40S",     32, 1),
    "400m": ("jhu-clsp/ettin-encoder-400m", "L40S",     16, 2),
    "1b":   ("jhu-clsp/ettin-encoder-1b",   "A100-80GB", 16, 2),
}

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
    job_label can be a string or a callable(fn_args, fn_kwargs) -> string."""
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

@app.function(image=image, secrets=[telegram_secret], timeout=60)
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


# ── Lion index build (already done; kept for completeness) ───────────────────

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


# ── Lion-derived stage runners (parameterized) ───────────────────────────────

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
    """Train any Lion-derived stage by name. Stage must be defined in
    config.yaml AND in train.py's choices list + elif routing."""
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
    """Eval any Lion-derived stage against the prebuilt Lion MS MARCO index."""
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
    """Teacher-as-query ceiling for any Lion-derived stage."""
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


# ── Ettin SPLADE — pipeline smoke + parameterized full trainer ───────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,   # 150M smoke runs fine on L40S
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_SHORT,
)
@with_notifications("ettin_splade_smoke")
def ettin_splade_smoke():
    """500-step smoke on Ettin-150M with MS MARCO triplets. ~10 min, ~$0.30.
    Validates that sentence-transformers + Ettin + Modal all work end-to-end
    before launching any longer run."""
    _run([
        "scripts/train_ettin_splade.py",
        "--model_id", "jhu-clsp/ettin-encoder-150m",
        "--output_dir", f"{VOLUME_MOUNT}/ettin_splade/smoke",
        "--max_steps", "500",
        "--batch_size", "16",
        "--gradient_accumulation_steps", "1",
        "--save_steps", "500",
        "--eval_steps", "500",
        "--logging_steps", "50",
        "--dataset_size", "20000",
    ])
    volume.commit()

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_SHORT,
)
@with_notifications(lambda a, kw: f"eval_ettin_checkpoints[{kw.get('tag','?')}]")
def eval_ettin_checkpoints(model_size: str = "150m", tag: str = "B_higher_reg_fast"):
    """Eval all saved checkpoints from an Ettin SPLADE training run."""
    base_dir = f"{VOLUME_MOUNT}/ettin_splade/ettin-encoder-{model_size}__{tag}"
    _run(["scripts/eval_ettin_checkpoints.py", "--base_dir", base_dir])
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,   # L40S — handles 150m and 400m. Use train_ettin_splade_1b for 1B.
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
@with_notifications(
    lambda a, kw: f"train_ettin_splade[{kw.get('model_size','150m')}/{kw.get('tag','default')}]"
)
def train_ettin_splade(
    model_size: str = "150m",     # "150m" or "400m" — for 1B use train_ettin_splade_1b
    tag: str = "default",         # appended to output_dir so concurrent runs don't collide
    max_steps: int = 30_000,
    batch_size: int = 0,          # 0 = use preset default for this model size
    grad_accum: int = 0,          # 0 = use preset default for this model size
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    query_reg: float = 5e-5,
    doc_reg: float = 3e-5,
    save_steps: int = 5_000,
    eval_steps: int = 5_000,
    logging_steps: int = 200,
    dataset_size: int = 500_000,
):
    """Train Ettin 150M or 400M as SPLADE with MS MARCO triplets, on L40S.
    Defaults match the Ettin model-card recipe; override flags to sweep.
    For 1B (which needs A100-80GB) call train_ettin_splade_1b instead."""
    if model_size not in ("150m", "400m"):
        raise ValueError(
            f"train_ettin_splade supports 150m or 400m (on L40S). "
            f"For 1b use train_ettin_splade_1b. Got: {model_size}"
        )
    model_id, _gpu, default_batch, default_accum = ETTIN_PRESETS[model_size]

    if batch_size == 0:
        batch_size = default_batch
    if grad_accum == 0:
        grad_accum = default_accum

    output_dir = f"{VOLUME_MOUNT}/ettin_splade/ettin-encoder-{model_size}__{tag}"
    _run([
        "scripts/train_ettin_splade.py",
        "--model_id", model_id,
        "--output_dir", output_dir,
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(learning_rate),
        "--warmup_ratio", str(warmup_ratio),
        "--query_reg_weight", str(query_reg),
        "--doc_reg_weight", str(doc_reg),
        "--save_steps", str(save_steps),
        "--eval_steps", str(eval_steps),
        "--logging_steps", str(logging_steps),
        "--dataset_size", str(dataset_size),
    ])
    volume.commit()


# Variant with A100-80GB GPU specifically for 1B. Modal pins the GPU at
# function definition time, so we need a separate function for the 1B GPU.
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret, telegram_secret],
    timeout=TIMEOUT_LONG,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
@with_notifications(lambda a, kw: f"train_ettin_splade_1b[{kw.get('tag','default')}]")
def train_ettin_splade_1b(
    tag: str = "default",
    max_steps: int = 30_000,
    batch_size: int = 16,
    grad_accum: int = 2,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    query_reg: float = 5e-5,
    doc_reg: float = 3e-5,
    save_steps: int = 5_000,
    eval_steps: int = 5_000,
    logging_steps: int = 200,
    dataset_size: int = 500_000,
):
    """Same recipe as train_ettin_splade but pinned to A100-80GB for the 1B
    model. Use after you've settled hyperparameters on 150M."""
    output_dir = f"{VOLUME_MOUNT}/ettin_splade/ettin-encoder-1b__{tag}"
    _run([
        "scripts/train_ettin_splade.py",
        "--model_id", "jhu-clsp/ettin-encoder-1b",
        "--output_dir", output_dir,
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(learning_rate),
        "--warmup_ratio", str(warmup_ratio),
        "--query_reg_weight", str(query_reg),
        "--doc_reg_weight", str(doc_reg),
        "--save_steps", str(save_steps),
        "--eval_steps", str(eval_steps),
        "--logging_steps", str(logging_steps),
        "--dataset_size", str(dataset_size),
    ])
    volume.commit()


# ── Default entrypoint ───────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Default entrypoint runs setup_workspace. For everything else, invoke
    functions explicitly, e.g.:

        modal run --detach modal_app.py::train_ettin_splade --model-size 150m --tag A_baseline
    """
    setup_workspace.remote()
