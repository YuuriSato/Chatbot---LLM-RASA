from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mescla adaptador LoRA no modelo base Hugging Face antes da conversao para GGUF.",
    )
    parser.add_argument("--base", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="runtime/outputs/medgemma-dental-merged")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.adapter)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(f"Modelo mesclado salvo em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
