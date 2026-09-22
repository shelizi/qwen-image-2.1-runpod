"""RunPod Serverless worker for Qwen-Image 2.1 Q4_K_M."""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import runpod
from huggingface_hub import hf_hub_download

REPO = os.environ.get(
    "MODEL_REPO",
    "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF",
)
DIFFUSION_FILE = os.environ.get("DIFFUSION_FILE", "qwen-image-2.1-Q4_K_M.gguf")
LLM_FILE = os.environ.get(
    "LLM_FILE", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
)
VAE_FILE = os.environ.get("VAE_FILE", "vae/qwen_image_2.1_vae_bf16.safetensors")
SD_PORT = int(os.environ.get("SD_PORT", "1234"))
SD_URL = f"http://127.0.0.1:{SD_PORT}"

_ready = threading.Event()
_init_error = None
_server = None
_generate_lock = threading.Lock()


def log(message: str) -> None:
    print(message, flush=True)


def model_store() -> Path:
    override = os.environ.get("MODEL_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        volume = Path("/runpod-volume")
        if volume.is_dir() and os.access(volume, os.W_OK):
            path = volume / "qwen-image-2.1"
        else:
            path = Path("/models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_snapshot() -> Path | None:
    root = Path(
        "/runpod-volume/huggingface-cache/hub/"
        "models--KasugaiSakura--Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
    )
    ref = root / "refs" / "main"
    if ref.is_file():
        snapshot = root / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snapshot.is_dir():
            return snapshot
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        found = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if found:
            return found[-1]
    return None


def resolve_file(filename: str) -> str:
    snapshot = cached_snapshot()
    if snapshot is not None:
        candidate = snapshot / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            log(f"using cached {candidate}")
            return str(candidate)
    destination = model_store()
    log(f"downloading {REPO}/{filename}")
    return hf_hub_download(
        repo_id=REPO,
        filename=filename,
        local_dir=str(destination),
    )


def wait_for_server(process, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    url = f"{SD_URL}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sd-server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("sd-server did not become ready")


def start_engine() -> None:
    global _server
    diffusion = resolve_file(DIFFUSION_FILE)
    llm = resolve_file(LLM_FILE)
    vae = resolve_file(VAE_FILE)
    binary = os.environ.get("SD_SERVER_BIN", "/sd-server")
    if not Path(binary).is_file():
        raise FileNotFoundError(f"sd-server not found at {binary}")

    command = [
        binary,
        "--diffusion-model", diffusion,
        "--vae", vae,
        "--llm", llm,
        "--cfg-scale", os.environ.get("SD_CFG", "6"),
        "--sampling-method", "euler",
        "--steps", os.environ.get("SD_STEPS", "25"),
        "-W", os.environ.get("SD_WIDTH", "1024"),
        "-H", os.environ.get("SD_HEIGHT", "1024"),
        "--seed", "-1",
        "--vae-tiling",
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(SD_PORT),
    ]
    if os.environ.get("SD_OFFLOAD", "1") != "0":
        command.append("--offload-to-cpu")

    log("starting " + " ".join(command))
    _server = subprocess.Popen(command)
    wait_for_server(_server)
    log(f"sd-server ready at {SD_URL}")


def initialize() -> None:
    global _init_error
    try:
        start_engine()
    except Exception as exc:
        _init_error = exc
        log(f"initialization failed: {exc}")
    finally:
        _ready.set()


def require_ready() -> None:
    if not _ready.wait(timeout=3600):
        raise TimeoutError("model initialization timed out")
    if _init_error is not None:
        raise RuntimeError(f"worker failed to start: {_init_error}")


def checked_size(value, default: int) -> int:
    size = int(value if value is not None else default)
    if size < 256 or size > 2048 or size % 32 != 0:
        raise ValueError("width and height must be multiples of 32 between 256 and 2048")
    return size


def strip_data_url(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def generate(job_input: dict) -> dict:
    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")

    width = checked_size(job_input.get("width"), int(os.environ.get("SD_WIDTH", "1024")))
    height = checked_size(job_input.get("height"), int(os.environ.get("SD_HEIGHT", "1024")))
    steps = int(job_input.get("steps", os.environ.get("SD_STEPS", "25")))
    if steps < 1 or steps > 60:
        raise ValueError("steps must be between 1 and 60")
    cfg_scale = float(job_input.get("cfg_scale", os.environ.get("SD_CFG", "6")))
    seed = int(job_input.get("seed", -1))

    body = {
        "prompt": prompt,
        "negative_prompt": str(job_input.get("negative_prompt", "")),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sampler_name": str(job_input.get("sampler_name", "euler")),
        "batch_size": 1,
    }
    image = job_input.get("image") or job_input.get("init_image")
    if image:
        body["extra_images"] = [strip_data_url(str(image))]

    started = time.time()
    request = urllib.request.Request(
        f"{SD_URL}/sdapi/v1/txt2img",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"sd-server returned {exc.code}: {detail}") from exc

    images = payload.get("images") or []
    if not images:
        raise RuntimeError(f"sd-server returned no image: {payload}")
    info = payload.get("info", "{}")
    if isinstance(info, str):
        info = json.loads(info)
    return {
        "image": images[0],
        "width": info.get("width", width),
        "height": info.get("height", height),
        "seed": info.get("seed", seed),
        "steps": info.get("steps", steps),
        "cfg_scale": info.get("cfg_scale", cfg_scale),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def handler(job):
    require_ready()
    job_input = job.get("input") or {}
    with _generate_lock:
        return generate(job_input)


def concurrency_modifier(_current):
    return 1


threading.Thread(target=initialize, daemon=True).start()
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
    }
)
