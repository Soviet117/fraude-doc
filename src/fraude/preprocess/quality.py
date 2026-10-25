"""Puerta de calidad de imagen (fase F2, historia E2-1).

Evalúa si una imagen es analizable antes de gastar tiempo en forense u OCR.
Si falla, la salida correcta es pedir una recaptura, nunca declarar fraude.

Solo usa Pillow y NumPy (dependencias core) para que el CI no necesite la
extra ``ml`` ni OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fraude.core.config import QualityConfig, load_config
from fraude.core.schemas import CaptureType, QualityReport

# Resoluciones típicas de pantalla completa (para detectar capturas de pantalla).
_SCREEN_RESOLUTIONS = frozenset(
    {
        (1280, 720),
        (1366, 768),
        (1440, 900),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (2880, 1800),
        (3840, 2160),
    }
)

_MAX_ANALYSIS_SIDE = 800
_COARSE_SKEW = np.arange(-5.0, 5.01, 1.0)
_FINE_SKEW = np.arange(-1.0, 1.01, 0.25)


def assess_quality(
    image: Image.Image,
    config: QualityConfig | None = None,
) -> QualityReport:
    """Evalúa resolución, desenfoque, inclinación, contraste y exposición.

    Devuelve un ``QualityReport`` con ``ok=False`` y una lista de códigos en
    ``issues`` cuando la imagen no es apta para análisis.
    """
    cfg = config or load_config().quality
    gray = _to_grayscale(image)
    issues: list[str] = []

    if min(gray.shape) < cfg.min_side_px:
        issues.append("low_resolution")

    blur = _laplacian_variance(gray)
    if blur < cfg.blur_threshold:
        issues.append("blur")

    skew = _estimate_skew(gray)
    if abs(skew) > cfg.max_skew_deg:
        issues.append("skew")

    contrast = float(gray.std())
    if contrast < cfg.min_contrast:
        issues.append("low_contrast")

    mean = float(gray.mean())
    if mean >= cfg.overexposed_mean:
        issues.append("overexposed")
    elif mean <= cfg.underexposed_mean:
        issues.append("underexposed")

    return QualityReport(
        ok=not issues,
        blur=round(blur, 2),
        skew_deg=round(skew, 2),
        issues=issues,
    )


def detect_capture_type(
    image: Image.Image,
    source_format: str | None = None,
) -> CaptureType:
    """Estima cómo llegó la imagen al sistema a partir de metadatos y tamaño."""
    fmt = (source_format or image.format or "").lower()
    if fmt == "pdf":
        return CaptureType.PDF_RENDER

    exif = image.getexif()
    if exif.get(271) or exif.get(272):  # Make / Model de la cámara
        return CaptureType.PHOTO

    dpi = estimate_dpi(image)
    if dpi is not None and dpi >= 150:
        return CaptureType.SCAN

    size = image.size
    if size in _SCREEN_RESOLUTIONS or (size[1], size[0]) in _SCREEN_RESOLUTIONS:
        return CaptureType.SCREENSHOT

    software = str(image.info.get("Software", "")).lower()
    if "screenshot" in software or "captura" in software:
        return CaptureType.SCREENSHOT

    return CaptureType.UNKNOWN


def estimate_dpi(image: Image.Image) -> int | None:
    """Devuelve el DPI horizontal declarado en la imagen, si existe."""
    dpi_info = image.info.get("dpi")
    if not dpi_info:
        return None
    try:
        return round(float(dpi_info[0]))
    except (TypeError, ValueError, IndexError):
        return None


def assess_image_path(
    path: Path,
    config: QualityConfig | None = None,
) -> tuple[QualityReport, CaptureType, int | None]:
    """Atajo para evaluar una imagen en disco."""
    with Image.open(path) as image:
        return (
            assess_quality(image, config),
            detect_capture_type(image, path.suffix.lstrip(".").lower()),
            estimate_dpi(image),
        )


def _to_grayscale(image: Image.Image) -> np.ndarray:
    """Convierte la imagen a una matriz de luminancia ``float32``."""
    return np.asarray(image.convert("L"), dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Varianza del laplaciano de 4 vecinos; cae con el desenfoque."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def _estimate_skew(gray: np.ndarray) -> float:
    """Estima el ángulo de corrección (grados) por perfil de proyección.

    Se prueba un rango de rotaciones y se elige la que maximiza la varianza de
    la suma por filas: cuando las líneas de texto quedan horizontales, las
    filas alternan texto y espacio y la varianza sube.
    """
    small = _downscale(gray)
    binary = _binarize(small)
    dark_ratio = float(binary.mean())
    if dark_ratio < 0.002 or dark_ratio > 0.9:
        return 0.0

    coarse_angle, _ = _best_rotation(binary, _COARSE_SKEW)
    fine_angle, _ = _best_rotation(binary, _FINE_SKEW + coarse_angle)
    return float(fine_angle)


def _best_rotation(binary: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
    """Devuelve el ángulo con mayor varianza de proyección y su puntaje."""
    base = Image.fromarray((binary * 255).astype(np.uint8))
    best_angle = 0.0
    best_score = -1.0
    for angle in angles:
        rotated = base.rotate(
            float(angle),
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=0,
        )
        rows = np.asarray(rotated, dtype=np.float32).sum(axis=1)
        score = float(rows.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle, best_score


def _downscale(gray: np.ndarray) -> np.ndarray:
    """Reduce la imagen para que el análisis de inclinación sea rápido."""
    longest = max(gray.shape)
    if longest <= _MAX_ANALYSIS_SIDE:
        return gray
    scale = _MAX_ANALYSIS_SIDE / longest
    height, width = gray.shape
    resized = Image.fromarray(gray.astype(np.uint8)).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Separa tinta (True) del fondo (False) con el umbral de Otsu."""
    threshold = _otsu_threshold(gray)
    return gray < threshold


def _otsu_threshold(gray: np.ndarray) -> float:
    """Umbral de Otsu calculado con el histograma de 256 niveles."""
    histogram, _ = np.histogram(gray, bins=256, range=(0.0, 256.0))
    total = float(gray.size)
    sum_all = float(np.dot(np.arange(256), histogram))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 0
    best_variance = -1.0

    for level in range(256):
        weight_background += float(histogram[level])
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += level * float(histogram[level])
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = level

    return float(best_threshold)
