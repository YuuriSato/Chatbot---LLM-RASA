from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import MethodType

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"
VALID_LABELS = {
    "REAL",
    "ALTERADA_MANUALMENTE",
    "ALTERADA_DIGITALMENTE",
    "IA",
    "INDETERMINADO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA/QLoRA do MedGemma 1.5 4B para classificar imagens reais, alteradas ou IA.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-jsonl", default="data/codex_training.jsonl")
    parser.add_argument("--image-root", default="data/dataset_dental")
    parser.add_argument("--output-dir", default="runtime/outputs/medgemma-dental-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Tamanho quadrado usado pelo processor de imagem. Reduza para poupar RAM em CPU.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Numero de threads CPU do PyTorch. 0 mantem o padrao do PyTorch.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-last-n-layers",
        type=int,
        default=0,
        help="Aplica LoRA somente nas ultimas N camadas do language model. Use em CPU para acelerar.",
    )
    parser.add_argument(
        "--train-vision-lora",
        action="store_true",
        help="Tambem aplica LoRA na torre visual. Usa muito mais RAM; deixe desligado em CPU.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forca treino em CPU/RAM. Muito mais lento; nao usa CUDA nem bitsandbytes.",
    )
    parser.add_argument(
        "--cpu-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Tipo numerico ao treinar em CPU. bfloat16 reduz bastante a RAM se suportado.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduz uso de RAM durante treino em troca de mais tempo de CPU.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"image", "label", "evidence"} - row.keys()
            if missing:
                raise ValueError(f"Linha {line_number}: campos ausentes: {sorted(missing)}")
            if row["label"] not in VALID_LABELS:
                raise ValueError(f"Linha {line_number}: label invalido: {row['label']}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Nenhum exemplo encontrado em {path}")
    return rows


def make_user_prompt() -> str:
    return (
        "Classifique a imagem odontologica ou intraoral como REAL, ALTERADA_MANUALMENTE, "
        "ALTERADA_DIGITALMENTE, IA ou INDETERMINADO. Use apenas evidencias visuais. "
        "Responda no formato: VEREDITO: <classe>\\nEVIDENCIAS: <evidencias curtas>."
    )


def make_answer(row: dict[str, str]) -> str:
    return f"VEREDITO: {row['label']}\nEVIDENCIAS: {row['evidence']}"


class DentalImageJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_root: Path):
        self.rows = load_jsonl(jsonl_path)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.image_root / row["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

        image = Image.open(image_path).convert("RGB")
        user_prompt = make_user_prompt()
        answer = make_answer(row)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        prompt_messages = messages[:1]
        return {
            "image": image,
            "messages": messages,
            "prompt_messages": prompt_messages,
        }


@dataclass
class VisionLanguageCollator:
    processor: Any
    max_length: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [example["image"] for example in examples]
        full_texts = [
            self.processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                example["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in examples
        ]

        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch["interpolate_pos_encoding"] = True
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for index, prompt_text in enumerate(prompt_texts):
            prompt_inputs = self.processor(
                text=prompt_text,
                images=images[index],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_length = int(prompt_inputs["input_ids"].shape[-1])
            labels[index, :prompt_length] = -100

        batch["labels"] = labels
        return batch


def build_model_and_processor(args: argparse.Namespace):
    local_files_only = Path(args.model_id).exists()
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=local_files_only,
    )
    if args.image_size != 896:
        print(
            "Aviso: Gemma3/MedGemma exige image-size 896 no projetor multimodal; "
            f"ignorando image-size {args.image_size}.",
            flush=True,
        )
        args.image_size = 896
    if args.image_size > 0 and hasattr(processor, "image_processor"):
        processor.image_processor.size = {
            "height": args.image_size,
            "width": args.image_size,
        }

    device_map = {"": "cpu"} if args.cpu else "auto"
    torch_dtype = (
        torch.bfloat16
        if (not args.cpu or args.cpu_dtype == "bfloat16")
        else torch.float32
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    if args.image_size > 0 and args.image_size != 896:
        vision_tower = model.model.vision_tower
        original_vision_forward = vision_tower.forward

        def forward_with_interpolation(self, *forward_args: Any, **forward_kwargs: Any):
            forward_kwargs["interpolate_pos_encoding"] = True
            return original_vision_forward(*forward_args, **forward_kwargs)

        vision_tower.forward = MethodType(forward_with_interpolation, vision_tower)
    if not args.train_vision_lora:
        original_get_image_features = model.model.get_image_features

        def get_frozen_image_features(self, *feature_args: Any, **feature_kwargs: Any):
            with torch.no_grad():
                image_features = original_get_image_features(*feature_args, **feature_kwargs)
            image_features.last_hidden_state = image_features.last_hidden_state.detach()
            image_features.pooler_output = image_features.pooler_output.detach()
            return image_features

        model.model.get_image_features = MethodType(get_frozen_image_features, model.model)

    target_modules: list[str] | str
    if args.train_vision_lora:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif args.lora_last_n_layers > 0:
        layer_numbers = [
            int(name.split(".")[3])
            for name, _ in model.named_modules()
            if name.startswith("model.language_model.layers.") and name.endswith(".self_attn.q_proj")
        ]
        if not layer_numbers:
            raise RuntimeError("Nao foi possivel encontrar camadas language_model para LoRA.")
        first_layer = max(layer_numbers) - args.lora_last_n_layers + 1
        selected_layers = "|".join(str(layer) for layer in layer_numbers if layer >= first_layer)
        target_modules = (
            rf".*language_model\.layers\.({selected_layers})\..*("
            r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
            r")"
        )
        print(f"LoRA limitado as camadas finais: {selected_layers}", flush=True)
    else:
        target_modules = (
            r".*language_model.*("
            r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
            r")"
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, processor


def main() -> int:
    args = parse_args()
    if args.cpu and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    dataset = DentalImageJsonlDataset(Path(args.dataset_jsonl), Path(args.image_root))
    model, processor = build_model_and_processor(args)
    collator = VisionLanguageCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=False if args.cpu else True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        use_cpu=args.cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"LoRA salvo em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
