import json
import urllib.request


SERVER_URL = "http://127.0.0.1:8001/v1/chat/completions"


def main() -> None:
    payload = {
        "model": "gpt-oss",
        "messages": [
            {"role": "developer", "content": "You are a helpful math assistant. Answer briefly."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "extra_body": {
            "lora_request": {
                "lora_name": "trained",
                "lora_int_id": 1,
                "lora_path": "/home/ubuntu/qlora/lora_adapters_final",
            }
        },
    }

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.load(resp)

    print(json.dumps(result, indent=2))
    print()
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
