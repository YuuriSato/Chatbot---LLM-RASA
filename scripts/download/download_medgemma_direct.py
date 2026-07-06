from __future__ import annotations

from pathlib import Path
import sys
import time

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from huggingface_hub import get_token, hf_hub_url, snapshot_download


MODEL_ID = "google/medgemma-1.5-4b-it"
SHARDS = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]
CHUNK_SIZE = 8 * 1024 * 1024
MAX_ATTEMPTS = 50


def human_size(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GB"


def download_file(snapshot_dir: Path, filename: str) -> None:
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face token nao encontrado. Rode `hf auth login` primeiro.")

    target = snapshot_dir / filename
    partial = snapshot_dir / f"{filename}.part"
    url = hf_hub_url(MODEL_ID, filename)
    auth_header = {"Authorization": f"Bearer {token}"}

    head = requests.head(url, headers=auth_header, allow_redirects=True, timeout=60)
    head.raise_for_status()
    expected = int(head.headers.get("content-length", "0"))

    if target.exists() and target.stat().st_size == expected:
        print(f"[ok] {filename} ja existe ({human_size(expected)})", flush=True)
        return

    for attempt in range(1, MAX_ATTEMPTS + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = dict(auth_header)
        mode = "ab"
        if existing:
            headers["Range"] = f"bytes={existing}-"
            print(
                f"[resume] {filename}: {human_size(existing)} de "
                f"{human_size(expected)} tentativa {attempt}/{MAX_ATTEMPTS}",
                flush=True,
            )
        else:
            print(f"[download] {filename}: {human_size(expected)}", flush=True)

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=120)
            response.raise_for_status()
            if existing and response.status_code == 200:
                existing = 0
                mode = "wb"

            downloaded = existing
            started = time.time()
            last_report = started
            with partial.open(mode + "") as file:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report >= 10:
                        pct = (downloaded / expected * 100) if expected else 0
                        speed = (downloaded - existing) / max(now - started, 1)
                        print(
                            f"[progress] {filename}: {pct:.1f}% "
                            f"({human_size(downloaded)} / {human_size(expected)}) "
                            f"{speed / (1024 ** 2):.2f} MB/s",
                            flush=True,
                        )
                        last_report = now
        except (ChunkedEncodingError, ConnectionError, ReadTimeout) as exc:
            saved = partial.stat().st_size if partial.exists() else 0
            print(f"[retry] {filename}: conexao caiu em {human_size(saved)}: {exc}", flush=True)
            time.sleep(min(5 * attempt, 60))
            continue

        final_size = partial.stat().st_size
        if expected and final_size != expected:
            print(
                f"[retry] {filename}: parcial {final_size} != esperado {expected}",
                flush=True,
            )
            time.sleep(min(5 * attempt, 60))
            continue
        partial.replace(target)
        print(f"[done] {filename}", flush=True)
        return

    raise RuntimeError(f"Falha ao baixar {filename} apos {MAX_ATTEMPTS} tentativas")


def main() -> int:
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.jinja",
            ],
        )
    )
    print(f"[snapshot] {snapshot}", flush=True)
    for shard in SHARDS:
        download_file(snapshot, shard)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise
