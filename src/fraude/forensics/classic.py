"""Señales forenses clásicas (F3, E3-1).

Todas las señales se calculan con Pillow y NumPy para que la ruta rápida
funcione sin GPU y el CI no necesite la extra ``ml``. Las señales son
heurísticas: no deciden por sí solas, solo aportan evidencia explicable.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from fraude.core.schemas import ForensicsReport, SignalScore
from fraude.ingest.hashing import sha256_file

_MAX_SIDE = 512
_COPY_MAX_SIDE = 320
_COPY_BLOCK = 16
_COPY_STEP = 8
_COPY_MIN_STD = 12.0
_COPY_MIN_SATURATION = 8.0
_COPY_CORRELATION = 0.99
_COPY_MIN_DISTANCE = 32
_COPY_MIN_AXIS_OFFSET = 8
_EDITING_SOFTWARE = (
    "photoshop",
    "gimp",
    "canva",
    "paint",
    "illustrator",
    "affinity",
    "pixlr",
    "inkscape",
    "krita",
)


def analyze_forensics(
    image_path: Path,
    *,
    out_dir: Path | None = None,
) -> ForensicsReport:
    """Calcula el reporte forense clásico de una imagen."""
    image_path = Path(image_path)
    analysis_dir = None
    if out_dir is not None:
        digest = sha256_file(image_path).split(":", 1)[-1][:16]
        analysis_dir = Path(out_dir) / digest

    with Image.open(image_path) as source:
        source_format = (source.format or "").lower()
        image = source.convert("RGB")
        metadata = metadata_signal(source)
        jpeg = jpeg_signal(source, source_format=source_format)

    ela = ela_signal(image, out_dir=analysis_dir)
    noise = noise_signal(image, out_dir=analysis_dir)
    chromatic = chromatic_signal(image, out_dir=analysis_dir)
    copy_move = copy_move_signal(image, out_dir=analysis_dir)
    localization = SignalScore(
        applicable=copy_move.score is not None,
        score=copy_move.score,
        map_path=copy_move.map_path,
        details={"model": "copy_move", **copy_move.details},
    )

    return ForensicsReport(
        ela=ela,
        noise=noise,
        chromatic=chromatic,
        jpeg=jpeg,
        metadata=metadata,
        localization=localization,
    )


def ela_signal(image: Image.Image, *, out_dir: Path | None = None) -> SignalScore:
    """Error de recompresión JPEG (ELA) como mapa de calor."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        diff = ImageChops.difference(image.convert("RGB"), recompressed.convert("RGB"))
        array = np.asarray(diff, dtype=np.float32)
    score_map = array.max(axis=2)
    high = float(np.percentile(score_map, 99.5))
    median = float(np.median(score_map))
    fraction = float((score_map > 25.0).mean())
    score = _clip(max(fraction / 0.05, (high - median) / 80.0), 0.0, 1.0)
    return SignalScore(
        applicable=True,
        score=round(score, 4),
        map_path=_save_map(score_map, out_dir, "ela", image.size),
        details={"p99": round(high, 3), "median": round(median, 3)},
    )


def noise_signal(image: Image.Image, *, out_dir: Path | None = None) -> SignalScore:
    """Inconsistencia del residuo de alta frecuencia."""
    gray = _downscale_gray(_gray(image), _MAX_SIDE)
    residual = gray - _box_blur(gray, radius=2)
    energy = np.sqrt(np.maximum(_box_blur(residual * residual, radius=4), 0.0))
    mean = float(energy.mean())
    score = _clip(float(energy.std()) / (mean + 1e-6) / 2.0, 0.0, 1.0)
    return SignalScore(
        applicable=True,
        score=round(score, 4),
        map_path=_save_map(energy, out_dir, "noise", image.size),
        details={"mean": round(mean, 3), "std": round(float(energy.std()), 3)},
    )


def chromatic_signal(image: Image.Image, *, out_dir: Path | None = None) -> SignalScore:
    """Inconsistencia entre los residuos de los canales de color."""
    rgb = _downscale_rgb(image, _MAX_SIDE)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = red - green
    gb = green - blue
    residual = np.abs(rg - _box_blur(rg, radius=2)) + np.abs(gb - _box_blur(gb, radius=2))
    energy = _box_blur(residual, radius=3)
    high = float(np.percentile(energy, 99.0))
    score = _clip(high / 40.0, 0.0, 1.0)
    return SignalScore(
        applicable=True,
        score=round(score, 4),
        map_path=_save_map(energy, out_dir, "chromatic", image.size),
        details={"p99": round(high, 3)},
    )


def jpeg_signal(
    image: Image.Image,
    *,
    source_format: str | None = None,
) -> SignalScore:
    """Señal de recompresión JPEG: blockiness y tablas de cuantización."""
    fmt = (source_format or image.format or "").lower()
    if fmt not in {"jpeg", "jpg"}:
        return SignalScore(applicable=False, details={"reason": "not_jpeg"})

    gray = _gray(image)
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return SignalScore(applicable=False, details={"reason": "too_small"})

    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    boundary = np.concatenate([horizontal[:, 7::8].ravel(), vertical[7::8, :].ravel()])
    within_h = np.delete(horizontal, np.s_[7::8], axis=1).ravel()
    within_v = np.delete(vertical, np.s_[7::8], axis=0).ravel()
    within = np.concatenate([within_h, within_v])
    ratio = float(boundary.mean()) / (float(within.mean()) + 1e-6)
    score = _clip((ratio - 1.0) / 1.5, 0.0, 1.0)

    quantization = getattr(image, "quantization", None) or {}
    table_means = {
        str(key): round(float(np.mean(values)), 3) for key, values in sorted(quantization.items())
    }
    return SignalScore(
        applicable=True,
        score=round(score, 4),
        details={
            "blockiness": round(ratio, 4),
            "quantization_tables": table_means,
        },
    )


def metadata_signal(image: Image.Image) -> SignalScore:
    """Busca software de edición y metadatos inconsistentes."""
    info = image.info or {}
    software = str(info.get("Software", "") or "").strip()
    comment = str(info.get("Comment", "") or "").strip()
    exif = image.getexif()
    reasons: list[str] = []
    haystack = f"{software} {comment}".lower()
    for editor in _EDITING_SOFTWARE:
        if editor in haystack:
            reasons.append(f"software={software or comment}")
            break
    if exif.get(305):
        exif_software = str(exif.get(305))
        if exif_software.lower() != software.lower():
            reasons.append("exif_software_distinto")
    score = _clip(len(reasons) * 0.5, 0.0, 1.0)
    return SignalScore(
        applicable=True,
        score=round(score, 4),
        details={
            "reasons": reasons,
            "software": software or None,
            "has_exif": bool(exif),
        },
    )


def copy_move_signal(image: Image.Image, *, out_dir: Path | None = None) -> SignalScore:
    """Detección simple de copy-move por bloques normalizados.

    Solo considera bloques con color (sellos, firmas, fotografías). Así se
    evita marcar texto negro repetido como copy-move; es una limitación
    conocida para documentos estrictamente en escala de grises.

    También descarta duplicados alineados en un solo eje. Los encabezados y
    barras de color de las plantillas producen bordes rectos que correlacionan
    perfectamente entre sí; exigir desplazamiento horizontal y vertical reduce
    esos falsos positivos. Es una limitación: un copy-move puramente horizontal
    o vertical no se reportará.
    """
    rgb = _downscale_rgb(image, _COPY_MAX_SIDE)
    gray = np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert("L"), dtype=np.float32)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    block = _COPY_BLOCK
    step = _COPY_STEP
    height, width = gray.shape
    if height < block * 2 or width < block * 2:
        return SignalScore(applicable=False, details={"reason": "too_small"})

    features: list[np.ndarray] = []
    centers: list[tuple[float, float]] = []
    for y in range(0, height - block + 1, step):
        for x in range(0, width - block + 1, step):
            window = gray[y : y + block, x : x + block].astype(np.float32)
            std = float(window.std())
            color = float(saturation[y : y + block, x : x + block].mean())
            if std < _COPY_MIN_STD or color < _COPY_MIN_SATURATION:
                continue
            normalized = (window - window.mean()) / (std + 1e-6)
            features.append(normalized.ravel())
            centers.append((x + block / 2.0, y + block / 2.0))

    if len(features) < 4:
        return SignalScore(applicable=False, details={"reason": "few_colored_blocks"})

    matrix = np.stack(features)
    matrix -= matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms + 1e-6
    correlations = matrix @ matrix.T
    points = np.asarray(centers)
    mask_image = Image.fromarray(np.zeros_like(gray, dtype=np.uint8))
    draw = ImageDraw.Draw(mask_image)
    matched: set[tuple[int, int]] = set()

    for first in range(len(matrix) - 1):
        candidates = np.where(correlations[first, first + 1 :] >= _COPY_CORRELATION)[0]
        for offset in candidates:
            second = int(first + 1 + offset)
            dx = float(points[first, 0] - points[second, 0])
            dy = float(points[first, 1] - points[second, 1])
            if dx * dx + dy * dy < _COPY_MIN_DISTANCE * _COPY_MIN_DISTANCE:
                continue
            if abs(dx) < _COPY_MIN_AXIS_OFFSET or abs(dy) < _COPY_MIN_AXIS_OFFSET:
                continue
            matched.add((first, second))
            for index in (first, second):
                cx, cy = points[index]
                draw.rectangle(
                    (
                        max(0, int(cx - block)),
                        max(0, int(cy - block)),
                        min(width - 1, int(cx + block)),
                        min(height - 1, int(cy + block)),
                    ),
                    fill=255,
                )

    mask = np.asarray(mask_image)
    denominator = max(8.0, len(matrix) * 0.01)
    score = _clip(len(matched) / denominator, 0.0, 1.0)
    map_path = None
    if matched:
        map_path = _save_map(mask.astype(np.float32), out_dir, "copy_move", image.size)
    return SignalScore(
        applicable=True,
        score=round(score, 4),
        map_path=map_path,
        details={
            "matched_blocks": len(matched),
            "candidate_blocks": len(matrix),
        },
    )


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def _downscale_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    height, width = gray.shape
    longest = max(height, width)
    if longest <= max_side:
        return gray
    scale = max_side / longest
    resized = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def _downscale_rgb(image: Image.Image, max_side: int) -> np.ndarray:
    gray = _gray(image)
    height, width = gray.shape
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _box_blur(array: np.ndarray, *, radius: int) -> np.ndarray:
    clipped = np.clip(array, 0.0, 255.0).astype(np.uint8)
    blurred = Image.fromarray(clipped).filter(ImageFilter.BoxBlur(radius))
    return np.asarray(blurred, dtype=np.float32)


def _save_map(
    score_map: np.ndarray,
    out_dir: Path | None,
    name: str,
    original_size: tuple[int, int],
) -> str | None:
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    high = float(np.percentile(score_map, 99.5))
    if high <= 0:
        normalized = np.zeros_like(score_map, dtype=np.uint8)
    else:
        normalized = np.clip(score_map / high * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(normalized).resize(original_size, Image.Resampling.BILINEAR)
    path = out_dir / f"{name}.png"
    image.save(path)
    return str(path)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
