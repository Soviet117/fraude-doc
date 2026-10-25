"""Punto de entrada del pipeline.

Estado actual (fases F0-F2): ingesta, identificación por hash, contrato
`Observation` y puerta de calidad para imágenes. Las etapas de forense, OCR,
reglas y fusión se incorporan en las fases F3-F7 (ver plan §7.2 y §8).

Este módulo no debe importar torch ni modelos: la demo y los tests deben correr
sin la extra `ml`.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from fraude.core.schemas import (
    CaptureInfo,
    DoctypeInfo,
    ForensicsReport,
    FusionReport,
    Observation,
    OcrReport,
    QualityReport,
    RegistryReport,
    RiskLevel,
    SourceInfo,
)
from fraude.ingest.hashing import sha256_bytes, sha256_file
from fraude.preprocess.quality import (
    assess_quality,
    detect_capture_type,
    estimate_dpi,
)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


def analyze_bytes(data: bytes, filename: str) -> Observation:
    """Analiza un documento en memoria (usado por la demo)."""
    observation = _build_observation(sha256_bytes(data), filename, len(data))
    _apply_quality(observation, BytesIO(data), Path(filename).suffix)
    return observation


def analyze_document(path: Path) -> Observation:
    """Analiza un documento en disco."""
    observation = _build_observation(sha256_file(path), path.name, path.stat().st_size)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with path.open("rb") as handle:
            _apply_quality(observation, handle, path.suffix)
    return observation


def _apply_quality(
    observation: Observation,
    stream: BinaryIO,
    suffix: str,
) -> None:
    """Completa calidad y tipo de captura si el archivo es una imagen."""
    if suffix.lower() not in IMAGE_SUFFIXES:
        return

    try:
        with Image.open(stream) as image:
            observation.quality = assess_quality(image)
            observation.capture = CaptureInfo(
                type=detect_capture_type(image, suffix.lstrip(".").lower()),
                dpi_est=estimate_dpi(image),
            )
    except (UnidentifiedImageError, OSError):
        observation.quality = QualityReport(ok=False, issues=["unreadable_image"])


def _build_observation(document_id: str, filename: str, size: int) -> Observation:
    suffix = Path(filename).suffix.lstrip(".").lower()
    return Observation(
        document_id=document_id,
        source=SourceInfo(filename=filename, format=suffix or None, bytes=size),
        capture=CaptureInfo(),
        quality=QualityReport(),
        doctype=DoctypeInfo(),
        forensics=ForensicsReport(),
        ocr=OcrReport(),
        fields={},
        rules=[],
        registry=RegistryReport(),
        fusion=FusionReport(risk=RiskLevel.UNDETERMINED),
        timings={},
    )
