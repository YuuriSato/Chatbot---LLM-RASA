from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


VALID_LABELS = {
    "REAL",
    "MODIFICADO",
    "ALTERADA_MANUALMENTE",
    "ALTERADA_DIGITALMENTE",
    "IA_GERADA_EDITADA",
    "INDETERMINADO",
}


@dataclass
class CalibrationSample:
    path: Path
    sha256: str
    label: str
    score: int
    confidence: str
    evidence: str
    justification: str
    dhash: int


@dataclass
class ForensicResult:
    score: int
    verdict_hint: str
    confidence: str
    evidence: list[str] = field(default_factory=list)
    source: str = "forense_local"
    skip_llm: bool = False
    exact_match: bool = False
    perceptual_match: bool = False
    original_used: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration_entry: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict_hint": self.verdict_hint,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
            "skip_llm": self.skip_llm,
            "exact_match": self.exact_match,
            "perceptual_match": self.perceptual_match,
            "original_used": self.original_used,
            "metrics": self.metrics,
        }


def clamp(value: float, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(int(round(value)), maximum))


def canonical_label(value: str) -> str:
    normalized = str(value or "").upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "NAO": "REAL",
        "NAO_ALTERADA": "REAL",
        "NÃO": "REAL",
        "SIM": "ALTERADA_DIGITALMENTE",
        "IA": "IA_GERADA_EDITADA",
        "IA_GERADA": "IA_GERADA_EDITADA",
        "IA_EDITADA": "IA_GERADA_EDITADA",
        "ALTERADA": "MODIFICADO",
        "ALTERADA_MANUALMENTE": "MODIFICADO",
        "ALTERADA_DIGITALMENTE": "MODIFICADO",
        "ALTERADA_MANUAL": "ALTERADA_MANUALMENTE",
        "ALTERADA_DIGITAL": "ALTERADA_DIGITALMENTE",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_LABELS else "INDETERMINADO"


def verdict_from_score(score: int, preferred: str | None = None) -> str:
    preferred = canonical_label(preferred or "")
    if score <= 29:
        return "REAL"
    if score <= 59:
        return "INDETERMINADO"
    if score <= 79:
        if preferred in {"MODIFICADO", "ALTERADA_MANUALMENTE", "ALTERADA_DIGITALMENTE"}:
            return preferred
        return "MODIFICADO"
    if preferred in {"MODIFICADO", "ALTERADA_MANUALMENTE", "ALTERADA_DIGITALMENTE", "IA_GERADA_EDITADA"}:
        return preferred
    return "IA_GERADA_EDITADA"


def confidence_from_score(score: int) -> str:
    if score <= 14 or score >= 86:
        return "alta"
    if score <= 24 or score >= 72:
        return "media"
    return "baixa" if 35 <= score <= 55 else "media"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def open_rgb(path: Path, max_side: int = 1024) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def image_to_gray_array(path: Path, max_side: int = 768) -> np.ndarray:
    image = open_rgb(path, max_side=max_side).convert("L")
    return np.asarray(image, dtype=np.float32)


def dhash_image(path: Path) -> int:
    image = open_rgb(path, max_side=512).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.int16)
    diff = values[:, 1:] > values[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(value)
    return bits


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def block_values(array: np.ndarray, block_count: int = 6) -> list[np.ndarray]:
    height, width = array.shape[:2]
    values: list[np.ndarray] = []
    step_y = max(1, height // block_count)
    step_x = max(1, width // block_count)
    for y in range(0, height - step_y + 1, step_y):
        for x in range(0, width - step_x + 1, step_x):
            values.append(array[y : y + step_y, x : x + step_x])
    return values


def robust_cv(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(arr))
    if mean <= 1e-6:
        return 0.0
    return float(np.std(arr) / mean)


def ela_metrics(path: Path) -> dict[str, float]:
    image = open_rgb(path, max_side=768)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(image, recompressed).convert("L")
    arr = np.asarray(diff, dtype=np.float32)
    block_means = [float(np.mean(block)) for block in block_values(arr)]
    return {
        "ela_mean": round(float(np.mean(arr)), 3),
        "ela_p95": round(float(np.percentile(arr, 95)), 3),
        "ela_block_cv": round(robust_cv(block_means), 3),
    }


def sharpness_metrics(path: Path) -> dict[str, float]:
    gray = image_to_gray_array(path)
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    lap_blocks = [float(np.var(block)) for block in block_values(laplacian)]
    return {
        "laplacian_var": round(float(np.var(laplacian)), 3),
        "sharpness_block_cv": round(robust_cv(lap_blocks), 3),
    }


def noise_metrics(path: Path) -> dict[str, float]:
    image = open_rgb(path, max_side=768).convert("L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    residual = np.asarray(ImageChops.difference(image, blurred), dtype=np.float32)
    block_noise = [float(np.std(block)) for block in block_values(residual)]
    return {
        "noise_mean": round(float(np.mean(residual)), 3),
        "noise_block_cv": round(robust_cv(block_noise), 3),
    }


def overlay_metrics(path: Path) -> dict[str, float]:
    image = open_rgb(path, max_side=768)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    max_channel = np.max(arr, axis=2)
    min_channel = np.min(arr, axis=2)
    saturation = max_channel - min_channel
    red = (arr[:, :, 0] > 0.72) & (arr[:, :, 1] < 0.35) & (arr[:, :, 2] < 0.35)
    green = (arr[:, :, 1] > 0.58) & (arr[:, :, 0] < 0.48) & (arr[:, :, 2] < 0.55) & (saturation > 0.28)
    yellow = (arr[:, :, 0] > 0.70) & (arr[:, :, 1] > 0.65) & (arr[:, :, 2] < 0.40) & (saturation > 0.28)
    cyan_blue = (arr[:, :, 2] > 0.60) & (arr[:, :, 0] < 0.45) & (saturation > 0.32)
    marker_overlay = red | green | yellow | cyan_blue
    vivid_overlay = (max_channel > 0.70) & (saturation > 0.42)
    strong_overlay = marker_overlay | vivid_overlay
    return {
        "saturated_overlay_ratio": round(float(np.mean(strong_overlay)), 5),
        "marker_overlay_ratio": round(float(np.mean(marker_overlay)), 5),
        "green_marker_ratio": round(float(np.mean(green)), 5),
    }


def compare_with_original(suspect_path: Path, original_path: Path) -> dict[str, float | int | bool]:
    suspect_hash = dhash_image(suspect_path)
    original_hash = dhash_image(original_path)
    distance = hamming_distance(suspect_hash, original_hash)

    suspect = open_rgb(suspect_path, max_side=768)
    original = open_rgb(original_path, max_side=768)
    width = min(suspect.width, original.width)
    height = min(suspect.height, original.height)
    suspect = ImageOps.fit(suspect, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    original = ImageOps.fit(original, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    diff = np.abs(
        np.asarray(suspect, dtype=np.float32) - np.asarray(original, dtype=np.float32)
    )
    gray_diff = np.mean(diff, axis=2)
    center = gray_diff[height // 5 : height * 4 // 5, width // 5 : width * 4 // 5]
    high_diff_ratio = float(np.mean(gray_diff > 28.0))
    central_high_diff_ratio = float(np.mean(center > 28.0)) if center.size else high_diff_ratio
    mean_diff = float(np.mean(gray_diff))
    central_mean_diff = float(np.mean(center)) if center.size else mean_diff
    same_scene = distance <= 24 or (mean_diff < 55 and central_high_diff_ratio < 0.65)
    return {
        "dhash_distance": distance,
        "mean_diff": round(mean_diff, 3),
        "central_mean_diff": round(central_mean_diff, 3),
        "high_diff_ratio": round(high_diff_ratio, 5),
        "central_high_diff_ratio": round(central_high_diff_ratio, 5),
        "same_scene": bool(same_scene),
    }


def build_calibration_samples(
    calibration: dict[str, Any],
    search_dirs: list[Path],
) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    seen: set[str] = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            try:
                sha = file_sha256(path)
            except OSError:
                continue
            if sha in seen or sha not in calibration:
                continue
            entry = calibration.get(sha)
            if not isinstance(entry, dict):
                continue
            try:
                samples.append(
                    CalibrationSample(
                        path=path,
                        sha256=sha,
                        label=canonical_label(str(entry.get("label", "INDETERMINADO"))),
                        score=clamp(float(entry.get("score", 50))),
                        confidence=str(entry.get("confidence", "alta")),
                        evidence=str(entry.get("evidence", "")),
                        justification=str(entry.get("justification", "")),
                        dhash=dhash_image(path),
                    )
                )
                seen.add(sha)
            except Exception:
                continue
    return samples


def result_from_calibration(entry: dict[str, Any], source: str, exact: bool, perceptual: bool) -> ForensicResult:
    score = clamp(float(entry.get("score", 50)))
    label = verdict_from_score(score, str(entry.get("label", "INDETERMINADO")))
    evidence = [str(entry.get("evidence", "")).strip() or "imagem corresponde a exemplo calibrado local"]
    return ForensicResult(
        score=score,
        verdict_hint=label,
        confidence=str(entry.get("confidence", confidence_from_score(score))),
        evidence=evidence,
        source=source,
        skip_llm=True,
        exact_match=exact,
        perceptual_match=perceptual,
        calibration_entry=entry,
    )


def score_single_image(metrics: dict[str, float]) -> tuple[int, list[str], str | None]:
    score = 12
    evidence: list[str] = []
    preferred: str | None = None

    if metrics["ela_p95"] >= 18 and metrics["ela_block_cv"] >= 0.65:
        score += 18
        evidence.append("ELA mostra recompressao irregular por regioes")
    elif metrics["ela_p95"] >= 24:
        score += 10
        evidence.append("ELA alto em parte da imagem")

    if metrics["sharpness_block_cv"] >= 1.15:
        score += 14
        evidence.append("nitidez inconsistente entre regioes")

    if metrics["noise_block_cv"] >= 0.95:
        score += 14
        evidence.append("ruido local inconsistente")

    marker_ratio = metrics.get("marker_overlay_ratio", 0.0)
    green_ratio = metrics.get("green_marker_ratio", 0.0)
    saturated_ratio = metrics["saturated_overlay_ratio"]
    if green_ratio >= 0.00035:
        score += 68
        preferred = "MODIFICADO"
        evidence.append("linha ou seta verde sobreposta detectada")
    elif marker_ratio >= 0.0008:
        score += 58
        preferred = "MODIFICADO"
        evidence.append("marcacao colorida sobreposta detectada")
    elif saturated_ratio >= 0.002:
        score += 38
        preferred = "MODIFICADO"
        evidence.append("pixels saturados finos sugerem desenho ou anotacao")
    elif saturated_ratio >= 0.0007:
        score += 14
        evidence.append("ha pequenos tracos coloridos incomuns")

    if metrics["laplacian_var"] < 18 and metrics["noise_mean"] < 3.5:
        score += 8
        evidence.append("textura muito lisa para foto clinica detalhada")

    return clamp(score), evidence, preferred


def apply_original_comparison(score: int, evidence: list[str], metrics: dict[str, Any]) -> tuple[int, str | None]:
    preferred: str | None = None
    comparison = metrics.get("comparison", {})
    if not comparison:
        return score, preferred

    if not comparison.get("same_scene"):
        score += 12
        evidence.append("imagem original enviada parece nao corresponder a mesma cena")
        return clamp(score), preferred

    central_diff = float(comparison.get("central_mean_diff", 0))
    high_ratio = float(comparison.get("central_high_diff_ratio", 0))
    distance = int(comparison.get("dhash_distance", 64))

    if distance <= 4 and central_diff >= 9 and high_ratio >= 0.01:
        score += 72
        preferred = "IA_GERADA_EDITADA"
        evidence.append("mesma cena da original com alteracao localizada compativel com reconstrucao por IA")
    elif central_diff >= 24 and high_ratio >= 0.18:
        score += 58
        preferred = "IA_GERADA_EDITADA"
        evidence.append("comparacao com original mostra mudanca forte em regioes centrais")
    elif central_diff >= 14 and high_ratio >= 0.08:
        score += 38
        preferred = "ALTERADA_DIGITALMENTE"
        evidence.append("comparacao com original mostra edicao localizada")
    elif central_diff >= 8 or distance > 10:
        score += 18
        evidence.append("comparacao com original mostra diferencas moderadas")
    else:
        score -= 8
        evidence.append("comparacao com original nao mostra mudanca relevante")

    return clamp(score), preferred


def analyze_forensics(
    image_path: Path,
    *,
    original_path: Path | None = None,
    calibration: dict[str, Any] | None = None,
    calibration_samples: list[CalibrationSample] | None = None,
) -> ForensicResult:
    calibration = calibration or {}
    calibration_samples = calibration_samples or []
    image_hash = file_sha256(image_path)
    exact_entry = calibration.get(image_hash)
    if isinstance(exact_entry, dict):
        return result_from_calibration(exact_entry, "calibracao_sha256", exact=True, perceptual=False)

    image_dhash = dhash_image(image_path)
    best_sample: CalibrationSample | None = None
    best_distance = math.inf
    for sample in calibration_samples:
        distance = hamming_distance(image_dhash, sample.dhash)
        if distance < best_distance:
            best_distance = distance
            best_sample = sample

    if best_sample is not None and best_distance <= 6:
        entry = {
            "label": best_sample.label,
            "score": best_sample.score,
            "confidence": best_sample.confidence,
            "evidence": f"similaridade perceptual com {best_sample.path.name}; distancia dHash {best_distance}",
            "justification": best_sample.justification,
        }
        result = result_from_calibration(entry, "calibracao_perceptual", exact=False, perceptual=True)
        result.metrics["perceptual_distance"] = int(best_distance)
        return result

    metrics: dict[str, Any] = {}
    metrics.update(ela_metrics(image_path))
    metrics.update(sharpness_metrics(image_path))
    metrics.update(noise_metrics(image_path))
    metrics.update(overlay_metrics(image_path))
    metrics["dhash"] = f"{image_dhash:016x}"
    if best_sample is not None:
        metrics["nearest_calibration_distance"] = int(best_distance)

    score, evidence, preferred = score_single_image(metrics)
    source = "forense_local"

    if original_path is not None:
        metrics["comparison"] = compare_with_original(image_path, original_path)
        score, comparison_preferred = apply_original_comparison(score, evidence, metrics)
        preferred = comparison_preferred or preferred
        source = "forense_local_comparacao"

    if best_sample is not None and best_distance <= 14:
        if best_sample.label == "IA_GERADA_EDITADA":
            score = max(score, 72)
            preferred = "IA_GERADA_EDITADA"
        elif best_sample.label == "REAL":
            score = min(score, 28)
            preferred = "REAL"
        evidence.append(f"proximo de exemplo calibrado: {best_sample.path.name}")

    if not evidence:
        evidence.append("sem sinais locais fortes de edicao")

    verdict = verdict_from_score(score, preferred)
    skip_llm = (source == "forense_local_comparacao" and (score <= 18 or score >= 82)) or (
        preferred == "MODIFICADO" and score >= 72
    )
    return ForensicResult(
        score=score,
        verdict_hint=verdict,
        confidence=confidence_from_score(score),
        evidence=evidence[:5],
        source=source,
        skip_llm=skip_llm,
        original_used=original_path is not None,
        metrics=metrics,
    )
