from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_COLUMNS = [
    "image_path",
    "patient_name",
    "dentist",
    "appointment_date",
    "appointment_id",
]
MANIFEST_ALIASES = {
    "caminho_arquivo": "image_path",
    "arquivo": "image_path",
    "imagem": "image_path",
    "nome_paciente": "patient_name",
    "paciente": "patient_name",
    "dentista": "dentist",
    "responsavel": "dentist",
    "data_atendimento": "appointment_date",
    "data": "appointment_date",
    "numero_atendimento": "appointment_id",
    "n_atendimento": "appointment_id",
    "atendimento": "appointment_id",
}
REPORT_COLUMNS = [
    "comparison_type",
    "base_image",
    "compared_image",
    "base_patient_name",
    "compared_patient_name",
    "base_dentist",
    "compared_dentist",
    "base_appointment_date",
    "compared_appointment_date",
    "base_appointment_id",
    "compared_appointment_id",
    "similarity_percent",
    "ssim_score",
    "keypoint_match_score",
    "edge_similarity",
    "alignment_quality",
    "radiopaque_similarity",
    "observations",
    "final_result",
]


@dataclass(frozen=True)
class XrayRecord:
    image_path: Path
    patient_name: str
    dentist: str
    appointment_date: str
    appointment_id: str
    order_index: int


@dataclass(frozen=True)
class ImageFeatures:
    raw_gray: np.ndarray
    normalized: np.ndarray
    edges: np.ndarray
    radiopaque_mask: np.ndarray
    keypoints: tuple[Any, ...]
    descriptors: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara raios-X odontologicos de atendimentos diferentes para "
            "estimar se parecem pertencer ao mesmo paciente. Nao emite diagnostico."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--manifest",
        type=Path,
        help="CSV com image_path, patient_name, dentist, appointment_date e appointment_id.",
    )
    input_group.add_argument(
        "--image-dir",
        type=Path,
        help="Pasta de imagens. Usa metadados vazios e ordem por nome de arquivo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/xray_patient_comparison.csv"),
        help="Caminho do relatorio CSV.",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Caminho opcional para salvar tambem em XLSX.",
    )
    parser.add_argument(
        "--group-by-patient",
        action="store_true",
        help="Compara separadamente registros com o mesmo patient_name.",
    )
    parser.add_argument(
        "--same-threshold",
        type=float,
        default=75.0,
        help="Percentual minimo para classificar como Mesmo paciente.",
    )
    parser.add_argument(
        "--different-threshold",
        type=float,
        default=45.0,
        help="Abaixo deste percentual classifica como Possivel paciente diferente.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def resolve_image_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_records_from_manifest(manifest_path: Path) -> list[XrayRecord]:
    dataframe = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, sep=None, engine="python")
    dataframe = dataframe.rename(
        columns={
            column: MANIFEST_ALIASES.get(column.strip().lstrip("\ufeff").lower(), column.strip().lstrip("\ufeff"))
            for column in dataframe.columns
        }
    )
    if "image_path" not in dataframe.columns:
        raise ValueError("O manifest precisa conter a coluna obrigatoria image_path.")

    for column in MANIFEST_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""

    records: list[XrayRecord] = []
    for index, row in dataframe.iterrows():
        image_path = resolve_image_path(clean_text(row["image_path"]), manifest_path.parent)
        records.append(
            XrayRecord(
                image_path=image_path,
                patient_name=clean_text(row["patient_name"]),
                dentist=clean_text(row["dentist"]),
                appointment_date=clean_text(row["appointment_date"]),
                appointment_id=clean_text(row["appointment_id"]),
                order_index=index,
            )
        )
    return records


def load_records_from_directory(image_dir: Path) -> list[XrayRecord]:
    paths = sorted(
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [
        XrayRecord(
            image_path=path,
            patient_name="",
            dentist="",
            appointment_date="",
            appointment_id=str(index + 1),
            order_index=index,
        )
        for index, path in enumerate(paths)
    ]


def sortable_date(value: str) -> tuple[int, str]:
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return (1, value)
    return (0, parsed.isoformat())


def sortable_appointment_id(value: str) -> tuple[int, int | str]:
    cleaned = value.strip()
    if cleaned.isdigit():
        return (0, int(cleaned))
    return (1, cleaned)


def sort_records(records: list[XrayRecord]) -> list[XrayRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.patient_name.lower(),
            sortable_date(record.appointment_date),
            sortable_appointment_id(record.appointment_id),
            record.order_index,
        ),
    )


def comparison_groups(records: list[XrayRecord], group_by_patient: bool) -> list[list[XrayRecord]]:
    if not group_by_patient:
        return [sort_records(records)]

    grouped: dict[str, list[XrayRecord]] = {}
    blank_records: list[XrayRecord] = []
    for record in records:
        if record.patient_name:
            grouped.setdefault(record.patient_name.lower(), []).append(record)
        else:
            blank_records.append(record)

    groups = [sort_records(group) for group in grouped.values() if len(group) >= 2]
    if blank_records:
        groups.append(sort_records(blank_records))
    return groups or [sort_records(records)]


def build_comparison_pairs(records: list[XrayRecord]) -> list[tuple[str, XrayRecord, XrayRecord]]:
    pairs: list[tuple[str, XrayRecord, XrayRecord]] = []
    seen: set[tuple[int, int]] = set()

    for index in range(len(records) - 1):
        left = records[index]
        right = records[index + 1]
        seen.add((left.order_index, right.order_index))
        label = "sequencial"
        if index == 0:
            label = "sequencial_e_base"
        pairs.append((label, left, right))

    if records:
        base = records[0]
        for right in records[1:]:
            key = (base.order_index, right.order_index)
            if key in seen:
                continue
            pairs.append(("base_vs_demais", base, right))

    return pairs


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Nao foi possivel abrir a imagem: {path}")
    return image


def crop_dark_border(image: np.ndarray) -> np.ndarray:
    threshold = max(8, int(np.percentile(image, 8)))
    mask = image > threshold
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return image
    x, y, width, height = cv2.boundingRect(coords)
    if width < image.shape[1] * 0.35 or height < image.shape[0] * 0.35:
        return image
    return image[y : y + height, x : x + width]


def preprocess(image: np.ndarray, size: int = 768) -> np.ndarray:
    image = crop_dark_border(image)
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    offset_y = (size - resized.shape[0]) // 2
    offset_x = (size - resized.shape[1]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(canvas)


def extract_features(path: Path) -> ImageFeatures:
    raw = load_grayscale(path)
    normalized = preprocess(raw)
    edges = cv2.Canny(normalized, 45, 135)
    bright_threshold = float(np.percentile(normalized, 94))
    radiopaque_mask = (normalized >= bright_threshold).astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=2200, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(normalized, None)
    return ImageFeatures(
        raw_gray=raw,
        normalized=normalized,
        edges=edges,
        radiopaque_mask=radiopaque_mask,
        keypoints=tuple(keypoints or ()),
        descriptors=descriptors,
    )


def match_keypoints(
    base: ImageFeatures,
    compared: ImageFeatures,
) -> tuple[list[Any], np.ndarray | None, float]:
    if base.descriptors is None or compared.descriptors is None:
        return [], None, 0.0
    if len(base.keypoints) < 8 or len(compared.keypoints) < 8:
        return [], None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(base.descriptors, compared.descriptors, k=2)
    good_matches = [
        first
        for pair in raw_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.76 * second.distance
    ]
    denominator = max(1, min(len(base.keypoints), len(compared.keypoints)))
    keypoint_score = min(100.0, (len(good_matches) / denominator) * 220.0)

    homography = None
    if len(good_matches) >= 10:
        source = np.float32([compared.keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        target = np.float32([base.keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if inlier_mask is not None:
            inlier_ratio = float(np.mean(inlier_mask))
            keypoint_score *= 0.55 + (0.45 * inlier_ratio)

    return good_matches, homography, float(np.clip(keypoint_score, 0.0, 100.0))


def aligned_compared_image(base: ImageFeatures, compared: ImageFeatures, homography: np.ndarray | None) -> np.ndarray:
    if homography is None:
        return compared.normalized
    aligned = cv2.warpPerspective(
        compared.normalized,
        homography,
        (base.normalized.shape[1], base.normalized.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if np.count_nonzero(aligned) < aligned.size * 0.25:
        return compared.normalized
    return aligned


def percentage_ssim(left: np.ndarray, right: np.ndarray) -> float:
    score = structural_similarity(left, right, data_range=255)
    return float(np.clip(score, 0.0, 1.0) * 100.0)


def edge_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_edges = cv2.Canny(left, 45, 135) > 0
    right_edges = cv2.Canny(right, 45, 135) > 0
    union = np.logical_or(left_edges, right_edges)
    if not np.any(union):
        return 0.0
    intersection = np.logical_and(left_edges, right_edges)
    return float(np.mean(intersection[union]) * 100.0)


def correlation_score(left: np.ndarray, right: np.ndarray) -> float:
    left_values = left.astype(np.float32).ravel()
    right_values = right.astype(np.float32).ravel()
    if float(np.std(left_values)) < 1e-6 or float(np.std(right_values)) < 1e-6:
        return 0.0
    corr = float(np.corrcoef(left_values, right_values)[0, 1])
    return float(np.clip((corr + 1.0) * 50.0, 0.0, 100.0))


def radiopaque_similarity(base: ImageFeatures, compared_aligned: np.ndarray) -> float:
    bright_threshold = float(np.percentile(compared_aligned, 94))
    compared_mask = compared_aligned >= bright_threshold
    base_mask = base.radiopaque_mask > 0
    union = np.logical_or(base_mask, compared_mask)
    if not np.any(union):
        return 0.0
    intersection = np.logical_and(base_mask, compared_mask)
    return float(np.mean(intersection[union]) * 100.0)


def final_classification(score: float, same_threshold: float, different_threshold: float) -> str:
    if score >= same_threshold:
        return "Mesmo paciente"
    if score < different_threshold:
        return "Possivel paciente diferente"
    return "Inconclusivo"


def build_observations(
    *,
    ssim_score: float,
    keypoint_score: float,
    edge_score: float,
    radiopaque_score: float,
    alignment_quality: float,
) -> str:
    observations: list[str] = []
    if alignment_quality >= 70:
        observations.append("bom alinhamento visual entre exames")
    elif alignment_quality >= 35:
        observations.append("alinhamento parcial; angulo/posicao podem diferir")
    else:
        observations.append("alinhamento fraco; comparacao menos confiavel")

    if keypoint_score >= 65:
        observations.append("pontos anatomicos com correspondencia relevante")
    elif keypoint_score < 25:
        observations.append("poucos pontos visuais correspondentes")

    if radiopaque_score >= 55:
        observations.append("estruturas radiopacas/restauracoes aparentes semelhantes")
    elif radiopaque_score < 20:
        observations.append("baixa correspondencia de estruturas radiopacas aparentes")

    if ssim_score >= 70 and edge_score >= 25:
        observations.append("padrao geral da arcada e contornos aparentam semelhante")
    elif ssim_score < 45:
        observations.append("padrao geral da imagem apresenta diferenca visual relevante")

    return "; ".join(observations)


def compare_pair(
    base_record: XrayRecord,
    compared_record: XrayRecord,
    *,
    comparison_type: str,
    same_threshold: float,
    different_threshold: float,
) -> dict[str, Any]:
    row = {
        "comparison_type": comparison_type,
        "base_image": str(base_record.image_path),
        "compared_image": str(compared_record.image_path),
        "base_patient_name": base_record.patient_name,
        "compared_patient_name": compared_record.patient_name,
        "base_dentist": base_record.dentist,
        "compared_dentist": compared_record.dentist,
        "base_appointment_date": base_record.appointment_date,
        "compared_appointment_date": compared_record.appointment_date,
        "base_appointment_id": base_record.appointment_id,
        "compared_appointment_id": compared_record.appointment_id,
    }

    if not base_record.image_path.exists() or not compared_record.image_path.exists():
        missing = [
            str(path)
            for path in (base_record.image_path, compared_record.image_path)
            if not path.exists()
        ]
        return {
            **row,
            "similarity_percent": 0.0,
            "ssim_score": 0.0,
            "keypoint_match_score": 0.0,
            "edge_similarity": 0.0,
            "alignment_quality": 0.0,
            "radiopaque_similarity": 0.0,
            "observations": f"arquivo nao encontrado: {'; '.join(missing)}",
            "final_result": "Inconclusivo",
        }

    try:
        base = extract_features(base_record.image_path)
        compared = extract_features(compared_record.image_path)
        _matches, homography, keypoint_score = match_keypoints(base, compared)
        aligned = aligned_compared_image(base, compared, homography)

        ssim_score = percentage_ssim(base.normalized, aligned)
        edge_score = edge_similarity(base.normalized, aligned)
        corr_score = correlation_score(base.normalized, aligned)
        radiopaque_score = radiopaque_similarity(base, aligned)
        alignment_quality = max(keypoint_score, (ssim_score * 0.55) + (corr_score * 0.45))
        similarity = (
            (ssim_score * 0.34)
            + (keypoint_score * 0.26)
            + (edge_score * 0.16)
            + (corr_score * 0.14)
            + (radiopaque_score * 0.10)
        )
        similarity = float(np.clip(similarity, 0.0, 100.0))
        observations = build_observations(
            ssim_score=ssim_score,
            keypoint_score=keypoint_score,
            edge_score=edge_score,
            radiopaque_score=radiopaque_score,
            alignment_quality=alignment_quality,
        )
        result = final_classification(similarity, same_threshold, different_threshold)
        return {
            **row,
            "similarity_percent": round(similarity, 2),
            "ssim_score": round(ssim_score, 2),
            "keypoint_match_score": round(keypoint_score, 2),
            "edge_similarity": round(edge_score, 2),
            "alignment_quality": round(float(np.clip(alignment_quality, 0.0, 100.0)), 2),
            "radiopaque_similarity": round(radiopaque_score, 2),
            "observations": observations,
            "final_result": result,
        }
    except Exception as exc:
        return {
            **row,
            "similarity_percent": 0.0,
            "ssim_score": 0.0,
            "keypoint_match_score": 0.0,
            "edge_similarity": 0.0,
            "alignment_quality": 0.0,
            "radiopaque_similarity": 0.0,
            "observations": f"erro ao comparar imagens: {exc}",
            "final_result": "Inconclusivo",
        }


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.manifest:
        records = load_records_from_manifest(args.manifest.resolve())
    else:
        records = load_records_from_directory(args.image_dir.resolve())

    if len(records) < 2:
        raise ValueError("Informe pelo menos duas imagens para comparar.")

    rows: list[dict[str, Any]] = []
    for group in comparison_groups(records, args.group_by_patient):
        for comparison_type, base_record, compared_record in build_comparison_pairs(group):
            rows.append(
                compare_pair(
                    base_record,
                    compared_record,
                    comparison_type=comparison_type,
                    same_threshold=args.same_threshold,
                    different_threshold=args.different_threshold,
                )
            )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def main() -> int:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Relatorio CSV salvo em: {args.output}")

    if args.excel:
        args.excel.parent.mkdir(parents=True, exist_ok=True)
        report.to_excel(args.excel, index=False)
        print(f"Relatorio Excel salvo em: {args.excel}")

    print(report[["base_image", "compared_image", "similarity_percent", "final_result"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
