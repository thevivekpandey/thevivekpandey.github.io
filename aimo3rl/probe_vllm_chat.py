import json
import urllib.request

from transformers import AutoTokenizer


MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"
SERVER_URL = "http://127.0.0.1:8000/chat/"


def main() -> None:
    payload = {
        "messages": [[
            {"role": "developer", "content": "You are a helpful math assistant. Answer briefly."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]],
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 32,
        "logprobs": 0,
    }

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.load(resp)

    prompt_ids = result["prompt_ids"][0]
    completion_ids = result["completion_ids"][0]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Prompt IDs:")
    print(prompt_ids)
    print()
    print("Completion IDs:")
    print(completion_ids)
    print()
    print("Decoded completion:")
    print(tok.decode(completion_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
