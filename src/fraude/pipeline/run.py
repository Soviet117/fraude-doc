"""Punto de entrada del pipeline.

Estado actual (fases F0-F3): ingesta, identificación por hash, contrato
`Observation`, puerta de calidad, forense clásico, OCR opcional, extracción,
reglas y cruce con registro. La fusión calibrada y la explicación se
incorporan en F6-F7 (ver plan §7.2 y §8).

Este módulo no debe importar torch ni modelos: la demo y los tests deben correr
sin la extra `ml`.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

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
from fraude.extract.fields import extract_page
from fraude.forensics.classic import analyze_forensics
from fraude.ingest.hashing import sha256_bytes, sha256_file
from fraude.ocr.engine import OcrPage, OcrUnavailableError, run_ocr
from fraude.preprocess.quality import (
    assess_quality,
    detect_capture_type,
    estimate_dpi,
)
from fraude.registry.cross import RegistryEntry, cross_registry
from fraude.rules.engine import evaluate_rules

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


def analyze_bytes(
    data: bytes,
    filename: str,
    *,
    artifacts_dir: Path | None = None,
    registry: list[RegistryEntry] | None = None,
    run_text: bool = True,
    ocr_predictor: Any | None = None,
    today: date | None = None,
) -> Observation:
    """Analiza un documento en memoria (usado por la demo)."""
    observation = _build_observation(sha256_bytes(data), filename, len(data))
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return observation

    with tempfile.TemporaryDirectory(prefix="fraude-") as temp_dir:
        path = Path(temp_dir) / f"document{suffix}"
        path.write_bytes(data)
        _apply_image_analysis(
            observation,
            path,
            suffix,
            artifacts_dir=artifacts_dir,
            registry=registry,
            run_text=run_text,
            ocr_predictor=ocr_predictor,
            today=today,
        )
    return observation


def analyze_document(
    path: Path,
    *,
    artifacts_dir: Path | None = None,
    registry: list[RegistryEntry] | None = None,
    run_text: bool = True,
    ocr_predictor: Any | None = None,
    today: date | None = None,
) -> Observation:
    """Analiza un documento en disco."""
    path = Path(path)
    observation = _build_observation(sha256_file(path), path.name, path.stat().st_size)
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        _apply_image_analysis(
            observation,
            path,
            suffix,
            artifacts_dir=artifacts_dir,
            registry=registry,
            run_text=run_text,
            ocr_predictor=ocr_predictor,
            today=today,
        )
    return observation


def _apply_image_analysis(
    observation: Observation,
    path: Path,
    suffix: str,
    *,
    artifacts_dir: Path | None,
    registry: list[RegistryEntry] | None,
    run_text: bool,
    ocr_predictor: Any | None,
    today: date | None,
) -> None:
    """Aplica calidad y etapas F3 a una imagen legible."""
    if not _apply_quality(observation, path, suffix):
        return

    started = perf_counter()
    try:
        observation.forensics = analyze_forensics(path, out_dir=artifacts_dir)
    except (UnidentifiedImageError, OSError):
        observation.forensics = ForensicsReport()
    observation.timings["forensics_ms"] = _elapsed_ms(started)

    if not run_text:
        observation.ocr = OcrReport(engine="disabled")
        observation.rules = evaluate_rules(
            {},
            None,
            today=today,
            quality=observation.quality,
            forensics=observation.forensics,
        )
        return

    started = perf_counter()
    try:
        page = run_ocr(path, predictor=ocr_predictor)
    except (OcrUnavailableError, UnidentifiedImageError, OSError):
        observation.ocr = OcrReport(engine="unavailable")
        observation.rules = evaluate_rules(
            {},
            None,
            today=today,
            quality=observation.quality,
            forensics=observation.forensics,
        )
        observation.timings["ocr_ms"] = _elapsed_ms(started)
        return

    observation.timings["ocr_ms"] = _elapsed_ms(started)
    _apply_text(
        observation,
        page,
        registry=registry,
        today=today,
    )


def _apply_text(
    observation: Observation,
    page: OcrPage,
    *,
    registry: list[RegistryEntry] | None,
    today: date | None,
) -> None:
    """Extrae campos, cruza registro y evalúa reglas con una página OCR."""
    observation.ocr = OcrReport(
        engine=page.engine,
        mean_conf=page.mean_conf,
        n_words=page.n_words,
    )
    extraction = extract_page(page)
    observation.doctype = DoctypeInfo(
        label=extraction.document_type or "desconocido",
        confidence=1.0 if extraction.document_type else 0.0,
    )
    observation.fields = extraction.fields

    registry_report: RegistryReport | None = None
    if registry and extraction.fields:
        registry_report = cross_registry(extraction.fields, registry)
        observation.registry = registry_report

    started = perf_counter()
    observation.rules = evaluate_rules(
        extraction.fields,
        extraction.document_type,
        today=today,
        quality=observation.quality,
        forensics=observation.forensics,
        registry=registry_report,
        ocr_mean_conf=observation.ocr.mean_conf,
        ocr_words=observation.ocr.n_words,
    )
    observation.timings["rules_ms"] = _elapsed_ms(started)


def _apply_quality(
    observation: Observation,
    path: Path,
    suffix: str,
) -> bool:
    """Completa calidad y tipo de captura; devuelve si la imagen es legible."""
    try:
        with Image.open(path) as image:
            observation.quality = assess_quality(image)
            observation.capture = CaptureInfo(
                type=detect_capture_type(image, suffix.lstrip(".").lower()),
                dpi_est=estimate_dpi(image),
            )
        return True
    except (UnidentifiedImageError, OSError):
        observation.quality = QualityReport(ok=False, issues=["unreadable_image"])
        return False


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


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
