from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CLASSES = {
    "REAL": "Imagem original de clinica parceira ou repositorio academico, sem edicao intencional conhecida.",
    "ALTERADA_MANUALMENTE": "Imagem com rabiscos, marcacoes, setas, circulos ou esbocos manuais simulados.",
    "ALTERADA_DIGITALMENTE": "Imagem com edicoes digitais de software, como borroes, recortes, colagens ou clonagem.",
    "IA": "Imagem gerada ou modificada por modelos generativos.",
    "INDETERMINADO": "Imagem de baixa qualidade, origem incerta ou evidencia visual insuficiente para rotulo seguro.",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera JSONL de treino a partir de pastas rotuladas por classe.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/dataset_dental",
        help="Pasta base com subpastas real, alterada_manualmente, alterada_digitalmente, ia e indeterminado.",
    )
    parser.add_argument(
        "--output",
        default="data/codex_training.jsonl",
        help="Arquivo JSONL de saida.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permite gerar JSONL mesmo se nenhuma imagem for encontrada.",
    )
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_entries(dataset_dir: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    entries: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for label, evidence in CLASSES.items():
        folder_name = label.lower()
        folder = dataset_dir / folder_name
        images = iter_images(folder)
        counts[label] = len(images)

        for image_path in images:
            relative_path = image_path.relative_to(dataset_dir).as_posix()
            entries.append(
                {
                    "image": relative_path,
                    "label": label,
                    "evidence": evidence,
                }
            )

    return entries, counts


def write_jsonl(output_path: Path, entries: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    configure_stdio()
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)

    entries, counts = build_entries(dataset_dir)
    total = len(entries)

    if total == 0 and not args.allow_empty:
        print(
            f"Erro: nenhuma imagem encontrada em {dataset_dir}. "
            "Crie as subpastas rotuladas ou use --allow-empty.",
            file=sys.stderr,
        )
        return 2

    write_jsonl(output_path, entries)

    print(f"JSONL gerado: {output_path}")
    print(f"Total de imagens: {total}")
    for label in CLASSES:
        print(f"- {label}: {counts[label]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
