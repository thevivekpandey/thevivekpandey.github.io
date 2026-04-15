import os
import sys
import time
import signal
import resource
import subprocess
from concurrent.futures import ThreadPoolExecutor


# Raise soft file descriptor limit to avoid "Too many open files"
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_TOKEN"] = "<something>"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CC"] = "gcc"
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/home/ubuntu/tiktoken"
os.environ["QLORA_PATCH_VLLM_DEFAULT_WEIGHT_LOADER"] = "1"
os.environ["PYTHONPATH"] = "/home/ubuntu/qlora" + (
    ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
)

import requests


MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"
HOST = "0.0.0.0"
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"

SEED = 42
BATCH_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.92
DTYPE = "auto"
KV_CACHE_DTYPE = "fp8"
CONTEXT_TOKENS = 16384
STREAM_INTERVAL = 200
SERVER_TIMEOUT = 240
PRELOAD_WORKERS = 24
VLLM_MODEL_IMPL = "vllm"


def preload_model_weights(model_path: str) -> None:
    print(f"Loading model weights from {model_path} into OS Page Cache...")
    start_time = time.time()

    files_to_load = []
    total_size = 0
    for root, _, files in os.walk(model_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                files_to_load.append(file_path)
                total_size += os.path.getsize(file_path)

    def _read_file(path: str) -> None:
        with open(path, "rb") as file_object:
            while file_object.read(1024 * 1024 * 1024):
                pass

    with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as executor:
        list(executor.map(_read_file, files_to_load))

    elapsed = time.time() - start_time
    print(f"Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n")


def start_server() -> tuple[subprocess.Popen, object]:
    cmd = [
        sys.executable,
        "-m",
        "trl.scripts.vllm_serve",
        "--model",
        MODEL_PATH,
        "--revision",
        "main",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        DTYPE,
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--vllm-model-impl",
        VLLM_MODEL_IMPL,
        "--max-model-len",
        str(CONTEXT_TOKENS),
        "--enable-prefix-caching",
        "True",
        "--enforce-eager",
        "True",
    ]

    log_file = open("vllm_server.log", "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_file


def wait_for_server(process: subprocess.Popen, log_file) -> None:
    print("Waiting for vLLM server...")
    start_time = time.time()

    for _ in range(SERVER_TIMEOUT):
        return_code = process.poll()
        if return_code is not None:
            log_file.flush()
            with open("vllm_server.log", "r", encoding="utf-8", errors="replace") as f:
                logs = f.read()
            raise RuntimeError(f"Server died with code {return_code}. Full logs:\n{logs}\n")

        try:
            response = requests.get(f"{BASE_URL}/health/", timeout=5)
            response.raise_for_status()
            elapsed = time.time() - start_time
            print(f"Server is ready (took {elapsed:.2f} seconds).")
            print("Health endpoint responded successfully.\n")
            return client
        except Exception:
            time.sleep(1)

    raise RuntimeError("Server failed to start (timeout).")


def shutdown(process: subprocess.Popen, log_file) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
    finally:
        log_file.close()


def main() -> None:
    preload_model_weights(MODEL_PATH)
    process, log_file = start_server()

    def _handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down server...")
        shutdown(process, log_file)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    wait_for_server(process, log_file)
    print("vLLM server is running.")
    print(f"Base URL: {BASE_URL}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError("vLLM server exited unexpectedly. Check vllm_server.log.")
            time.sleep(2)
    finally:
        shutdown(process, log_file)


if __name__ == "__main__":
    main()
