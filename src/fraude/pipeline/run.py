"""Punto de entrada del pipeline.

Estado actual (fase F0): ingesta, identificación por hash y construcción del
contrato `Observation`. Las etapas de calidad, forense, OCR, reglas y fusión se
incorporan en las fases F2-F7 (ver plan §7.2 y §8).

Este módulo no debe importar torch ni modelos: la demo y los tests deben correr
sin la extra `ml`.
"""

from __future__ import annotations

from pathlib import Path

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


def analyze_bytes(data: bytes, filename: str) -> Observation:
    """Analiza un documento en memoria (usado por la demo)."""
    return _build_observation(sha256_bytes(data), filename, len(data))


def analyze_document(path: Path) -> Observation:
    """Analiza un documento en disco."""
    return _build_observation(sha256_file(path), path.name, path.stat().st_size)


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
