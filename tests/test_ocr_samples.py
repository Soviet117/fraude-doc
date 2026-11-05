"""Pruebas del arnés de OCR sobre el corpus de muestras."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraude.eval import ocr_samples
from fraude.eval.ocr_samples import (
    _long_date,
    _normalize,
    _normalize_periods,
    _value_present,
    _values_match,
    run_ocr_samples,
)
from fraude.ocr.engine import OcrPage, OcrWord

PERSONA_CSV = """persona_id,apellidos,nombres,dni,sexo,fecha_nacimiento,direccion,programa,modulo,periodo
P01,SÁNCHEZ SALAZAR,ÉDGAR MARTÍN,61531833,M,03/06/1999,"Av. Circunvalación 45, Soritor",Computación e Informática,Ofimática y herramientas digitales,2025-II
"""

INDEX_CSV = """id,tipo,variante,ataque,persona,documento_padre,campo_original,campo_alterado,png
M01,dni,authentic,,P01,,,,png/M01_dni.png
"""


def _word(text: str, x: int, y: int) -> OcrWord:
    return OcrWord(text, (x, y, max(8, len(text) * 8), 12), 0.95)


def _fake_dni_page() -> OcrPage:
    words = (
        _word("DOCUMENTO", 10, 10),
        _word("NACIONAL", 120, 10),
        _word("DE", 210, 10),
        _word("IDENTIDAD", 240, 10),
        _word("N.o", 10, 40),
        _word("61531833", 60, 40),
        _word("PRIMERA", 10, 70),
        _word("APELLIDO/", 90, 70),
        _word("SEGUNDOAPELLIDO", 180, 70),
        _word("SÁNCHEZ", 10, 100),
        _word("SALAZAR", 100, 100),
        _word("PRENOMBRES", 10, 130),
        _word("ÉDGAR", 10, 160),
        _word("MARTÍN", 90, 160),
        _word("FECHADEI", 10, 190),
        _word("NACIMIENTO", 110, 190),
        _word("03/06/1999", 10, 220),
    )
    return OcrPage(words=words, width=1000, height=300)


def test_normalize_periods() -> None:
    assert _normalize_periods("2024-1") == "2024-i"
    assert _normalize_periods("2024-I") == "2024-i"
    assert _normalize_periods("2024-11") == "2024-ii"
    assert _normalize_periods("2025-II") == "2025-ii"
    assert _normalize("Periodo académico: 2025-2") == "periodo academico 2025-ii"


def test_value_present_ignores_lost_spaces() -> None:
    text = _normalize("PRIMERA APELLIDO BAUTISTAHUAMAN PATRICIAS SOFIA")

    assert _value_present("BAUTISTA HUAMÁN", text, "apellidos")
    assert not _value_present("PATRICIA SOFÍA", text, "nombres")


def test_values_match_keeps_real_ocr_errors() -> None:
    assert _values_match("2024-I", "2024-1", "periodo")
    assert _values_match("BAUTISTA HUAMÁN", "BAUTISTAHUAMAN", "apellidos")
    assert not _values_match("PATRICIA SOFÍA", "PATRICIAS SOFIA", "nombres")
    assert not _values_match(
        "Computación e Informática",
        "Computacion el Informâtica",
        "programa",
    )


def test_long_date() -> None:
    assert _long_date("05/05/2002") == "5 de mayo de 2002"
    assert _long_date("fecha inválida") == "fecha inválida"


def test_run_ocr_samples_with_fake_predictor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "muestras"
    (root / "png").mkdir(parents=True)
    (root / "png" / "M01_dni.png").write_bytes(b"fake image")
    (root / "_personas.csv").write_text(PERSONA_CSV, encoding="utf-8")
    (root / "_indice.csv").write_text(INDEX_CSV, encoding="utf-8")

    page = _fake_dni_page()
    monkeypatch.setattr(ocr_samples, "run_ocr", lambda _path: page)

    payload = run_ocr_samples(
        root,
        out_path=tmp_path / "ocr.json",
        text_dir=tmp_path / "text",
        warmup=False,
    )

    summary = payload["summary"]
    assert summary["n_samples"] == 1
    assert summary["n_ok"] == 1
    assert summary["n_errors"] == 0
    assert summary["ocr_field_recall"] == 1.0
    assert summary["doctype_accuracy"] == 1.0
    assert summary["extraction_field_accuracy"] == 1.0
    assert summary["by_document_type"]["dni"]["n"] == 1

    text_file = tmp_path / "text" / "M01.txt"
    assert text_file.is_file()
    assert "DOCUMENTO NACIONAL DE IDENTIDAD" in text_file.read_text(encoding="utf-8")

    written = json.loads((tmp_path / "ocr.json").read_text(encoding="utf-8"))
    assert written["samples"][0]["sample_id"] == "M01"
    assert written["samples"][0]["extraction"]["doctype_ok"] is True
    assert written["samples"][0]["text_path"] == str(text_file)


def test_run_ocr_samples_reports_missing_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "muestras"
    root.mkdir()
    (root / "_personas.csv").write_text(PERSONA_CSV, encoding="utf-8")
    (root / "_indice.csv").write_text(INDEX_CSV, encoding="utf-8")
    monkeypatch.setattr(ocr_samples, "run_ocr", lambda _path: _fake_dni_page())

    payload = run_ocr_samples(
        root,
        out_path=None,
        text_dir=None,
        warmup=False,
    )

    assert payload["summary"]["n_ok"] == 0
    assert payload["summary"]["n_errors"] == 1
    assert "No existe la imagen" in payload["samples"][0]["error"]
