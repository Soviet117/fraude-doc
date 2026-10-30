"""Pruebas del punto de entrada del pipeline."""

from __future__ import annotations

from datetime import date

from PIL import Image, ImageDraw

from fraude.core.schemas import Severity
from fraude.ocr.engine import OcrPage, OcrUnavailableError, OcrWord
from fraude.pipeline.run import analyze_bytes, analyze_document
from fraude.registry.cross import RegistryEntry


def _document_image() -> Image.Image:
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    for index in range(30):
        draw.text((80, 80 + index * 35), f"Documento de prueba linea {index}", fill="black")
    return image


def _ficha_page() -> OcrPage:
    lines = (
        "FICHA DE MATRICULA",
        "Estudiante: Quispe Mamani, José Luis",
        "DNI: 12345678",
        "Periodo: 2025-I",
        "Programa: Computación e Informática",
        "Módulo: Ofimática y herramientas digitales",
        "Turno: Mañana",
        "Fecha de matrícula: 15/03/2025",
    )
    words = tuple(
        OcrWord(text=text, bbox=(80, 40 + index * 40, 700, 20), confidence=0.95)
        for index, text in enumerate(lines)
    )
    return OcrPage(words=words, width=1200, height=1600, engine="fake")


def test_analyze_bytes_builds_observation() -> None:
    observation = analyze_bytes(b"abc", "acta.pdf")

    assert observation.source.format == "pdf"
    assert observation.source.bytes == 3
    assert observation.quality.ok is True
    assert observation.ocr.n_words == 0


def test_analyze_document_matches_analyze_bytes(tmp_path) -> None:
    path = tmp_path / "ficha.png"
    path.write_bytes(b"abc")

    from_file = analyze_document(path)
    from_bytes = analyze_bytes(b"abc", path.name)

    assert from_file.document_id == from_bytes.document_id
    assert from_file.source == from_bytes.source


def test_pipeline_runs_f3_with_injected_ocr(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ficha.png"
    _document_image().save(path)
    page = _ficha_page()
    monkeypatch.setattr(
        "fraude.pipeline.run.run_ocr",
        lambda *_args, **_kwargs: page,
    )
    registry = [
        RegistryEntry(
            nombre="Quispe Mamani, José Luis",
            dni="12345678",
            programa="Computación e Informática",
            modulo="Ofimática y herramientas digitales",
            periodo="2025-I",
            fecha_nacimiento="15/03/1990",
        )
    ]

    observation = analyze_document(
        path,
        registry=registry,
        ocr_predictor=object(),
        today=date(2026, 10, 25),
    )

    assert observation.doctype.label == "ficha_matricula"
    assert observation.fields["dni"] == "12345678"
    assert observation.registry.matched is True
    assert observation.ocr.engine == "fake"
    assert observation.forensics.ela.applicable is True
    assert "ocr_ms" in observation.timings
    codes = {finding.code for finding in observation.rules}
    assert "TIPO_NO_IDENTIFICADO" not in codes
    assert "CAMPOS_FALTANTES" not in codes


def test_pipeline_reports_registry_difference(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ficha.png"
    _document_image().save(path)
    page = _ficha_page()
    monkeypatch.setattr(
        "fraude.pipeline.run.run_ocr",
        lambda *_args, **_kwargs: page,
    )
    registry = [
        RegistryEntry(
            nombre="Otra Persona",
            dni="12345678",
            programa="Computación e Informática",
            modulo="Ofimática y herramientas digitales",
            periodo="2025-I",
        )
    ]

    observation = analyze_document(
        path,
        registry=registry,
        today=date(2026, 10, 25),
    )

    finding = next(
        finding for finding in observation.rules if finding.code == "REGISTRO_DIFERENCIA"
    )
    assert finding.severity is Severity.HIGH


def test_pipeline_survives_ocr_unavailable(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ficha.png"
    _document_image().save(path)

    def _raise(*_args, **_kwargs):
        raise OcrUnavailableError("docTR no instalado")

    monkeypatch.setattr("fraude.pipeline.run.run_ocr", _raise)

    observation = analyze_document(path, today=date(2026, 10, 25))

    assert observation.ocr.engine == "unavailable"
    assert "TIPO_NO_IDENTIFICADO" in {finding.code for finding in observation.rules}
    assert observation.forensics.ela.applicable is True
