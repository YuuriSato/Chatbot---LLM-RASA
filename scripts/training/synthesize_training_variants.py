from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria exemplos sinteticos de imagens alteradas para bootstrapping do treino.",
    )
    parser.add_argument("--dataset-dir", default="data/dataset_dental")
    parser.add_argument("--source-class", default="real")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def draw_manual_marks(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    color = random.choice([(255, 0, 0), (255, 80, 0), (0, 140, 255)])
    stroke = max(3, min(width, height) // 90)

    for _ in range(random.randint(2, 5)):
        points = []
        start_x = random.randint(width // 8, width * 7 // 8)
        start_y = random.randint(height // 8, height * 7 // 8)
        for step in range(random.randint(4, 8)):
            points.append(
                (
                    max(0, min(width - 1, start_x + random.randint(-width // 8, width // 8) * step // 3)),
                    max(0, min(height - 1, start_y + random.randint(-height // 8, height // 8) * step // 3)),
                )
            )
        draw.line(points, fill=color, width=stroke, joint="curve")

    if random.random() < 0.6:
        x0 = random.randint(width // 5, width * 3 // 5)
        y0 = random.randint(height // 5, height * 3 // 5)
        x1 = min(width - 1, x0 + random.randint(width // 8, width // 3))
        y1 = min(height - 1, y0 + random.randint(height // 8, height // 3))
        draw.ellipse((x0, y0, x1, y1), outline=color, width=stroke)
    return image


def apply_digital_edit(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    edited = image.copy()

    crop_w = random.randint(max(8, width // 10), max(16, width // 4))
    crop_h = random.randint(max(8, height // 10), max(16, height // 4))
    src_x = random.randint(0, max(0, width - crop_w))
    src_y = random.randint(0, max(0, height - crop_h))
    dst_x = random.randint(0, max(0, width - crop_w))
    dst_y = random.randint(0, max(0, height - crop_h))

    patch = image.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))
    if random.random() < 0.7:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 4.0)))
    edited.paste(patch, (dst_x, dst_y))

    if random.random() < 0.5:
        overlay = Image.new("RGB", (crop_w, crop_h), random.choice([(245, 245, 245), (30, 30, 30)]))
        edited.paste(overlay, (src_x, src_y))
    return edited


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    source_dir = dataset_dir / args.source_class
    manual_dir = dataset_dir / "alterada_manualmente"
    digital_dir = dataset_dir / "alterada_digitalmente"
    manual_dir.mkdir(parents=True, exist_ok=True)
    digital_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(source_dir)
    if not images:
        raise SystemExit(f"Nenhuma imagem encontrada em {source_dir}")

    selected = random.sample(images, min(args.limit, len(images)))
    for index, image_path in enumerate(selected, start=1):
        with Image.open(image_path) as image:
            manual = draw_manual_marks(image.copy())
            digital = apply_digital_edit(image.copy())

        stem = f"{index:04d}_{image_path.stem}"
        manual.save(manual_dir / f"{stem}_manual.jpg", quality=92)
        digital.save(digital_dir / f"{stem}_digital.jpg", quality=92)

    indeterminate_dir = dataset_dir / "indeterminado"
    indeterminate_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(selected[: max(1, len(selected) // 5)], start=1):
        destination = indeterminate_dir / f"{index:04d}_{image_path.stem}_low_quality.jpg"
        with Image.open(image_path) as image:
            low_quality = image.convert("RGB").resize(
                (max(64, image.width // 4), max(64, image.height // 4))
            ).filter(ImageFilter.GaussianBlur(radius=2.0))
            low_quality.save(destination, quality=35)

    print(f"Alteradas manualmente: {len(selected)}")
    print(f"Alteradas digitalmente: {len(selected)}")
    print(f"Indeterminadas sinteticas: {max(1, len(selected) // 5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
