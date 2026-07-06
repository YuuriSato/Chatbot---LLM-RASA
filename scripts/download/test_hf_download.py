import time

import requests
from huggingface_hub import get_token, hf_hub_url


def main() -> None:
    url = hf_hub_url(
        "google/medgemma-1.5-4b-it",
        "model-00001-of-00002.safetensors",
    )
    started = time.time()
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Range": "bytes=0-1048575",
        },
        stream=True,
        timeout=60,
    )
    total = 0
    print(response.status_code, response.headers.get("content-length"), flush=True)
    for chunk in response.iter_content(262_144):
        total += len(chunk)
        if total >= 1_048_576:
            break
    print(f"downloaded {total} bytes in {time.time() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
