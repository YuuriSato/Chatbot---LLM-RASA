from huggingface_hub import snapshot_download


def main() -> None:
    path = snapshot_download(
        "google/medgemma-1.5-4b-it",
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.model",
            "*.jinja",
        ],
    )
    print(path)


if __name__ == "__main__":
    main()
