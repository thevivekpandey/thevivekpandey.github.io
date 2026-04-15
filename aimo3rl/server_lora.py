import os
import signal
import subprocess
import sys
import time

import requests


MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"
ADAPTER_PATH = "/home/ubuntu/qlora/lora_adapters_final"
SERVED_MODEL_NAME = "gpt-oss"
LORA_NAME = "trained"
HOST = "0.0.0.0"
PORT = 8001
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
SERVER_TIMEOUT = 240
GPU_MEMORY_UTILIZATION = "0.97"
MAX_MODEL_LEN = "4096"


os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/home/ubuntu/tiktoken"


def start_server():
    log_file = open("/home/ubuntu/qlora/vllm_lora_server.log", "w")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        MAX_MODEL_LEN,
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--max-loras",
        "1",
        "--lora-modules",
        f"{LORA_NAME}={ADAPTER_PATH}",
    ]
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return process, log_file


def wait_for_server(process, log_file):
    start = time.time()
    for _ in range(SERVER_TIMEOUT):
        rc = process.poll()
        if rc is not None:
            log_file.flush()
            with open("/home/ubuntu/qlora/vllm_lora_server.log", "r") as f:
                raise RuntimeError(f"Server died with code {rc}. Full logs:\n{f.read()}")
        try:
            r = requests.get(f"{BASE_URL}/models", timeout=5)
            r.raise_for_status()
            print(f"Server is ready (took {time.time() - start:.2f} seconds).")
            print(r.text)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError("Timed out waiting for LoRA vLLM server.")


def main():
    process, log_file = start_server()

    def shutdown(*_args):
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    wait_for_server(process, log_file)
    print("LoRA vLLM server is running.")
    print(f"Base URL: {BASE_URL}")
    print("Press Ctrl+C to stop.")
    process.wait()


if __name__ == "__main__":
    main()
