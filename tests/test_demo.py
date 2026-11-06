"""Pruebas de la lógica de presentación de la demo Streamlit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from fraude.core.schemas import (
    DoctypeInfo,
    Observation,
    OcrReport,
    QualityReport,
    SourceInfo,
)

APP_PATH = Path(__file__).resolve().parents[1] / "apps" / "demo" / "app.py"
_SPEC = importlib.util.spec_from_file_location("demo_app", APP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = demo
_SPEC.loader.exec_module(demo)


def _observation(**overrides: Any) -> Observation:
    values: dict[str, Any] = {
        "document_id": "sha256:" + "a" * 64,
        "source": SourceInfo(filename="sample.png", format="png", bytes=16),
        "doctype": DoctypeInfo(label="dni", confidence=1.0),
        "quality": QualityReport(ok=True),
        "ocr": OcrReport(engine="docTR", mean_conf=0.9, n_words=20),
    }
    values.update(overrides)
    return Observation(**values)


def test_risk_band_prioritizes_quality_gate() -> None:
    observation = _observation(quality=QualityReport(ok=False, issues=["blur"]))

    band = demo._risk_band(observation, score=1.0)

    assert band.key == "indeterminado"
    assert band.label == "Indeterminado"
    assert "recapturar" in band.note


def test_risk_band_marks_high_at_threshold() -> None:
    band = demo._risk_band(_observation(), score=0.5)

    assert band.key == "alto"
    assert band.label == "Alto"


def test_field_label_handles_indexed_acta_fields() -> None:
    assert demo._field_label("estudiante_0") == "Estudiante 1"
    assert demo._field_label("dni_2") == "DNI 3"
    assert demo._field_label("nota_1") == "Nota 2"
    assert demo._field_label("fecha_emision") == "Fecha de emisión"


def test_acta_rows_are_sorted_and_complete() -> None:
    rows = demo._acta_rows(
        {
            "estudiante_1": "RUIZ VÁSQUEZ, GABRIELA RUTH",
            "dni_1": "79352308",
            "nota_1": "16",
            "estudiante_0": "SÁNCHEZ SALAZAR, ÉDGAR MARTÍN",
            "dni_0": "61531833",
            "nota_0": "18",
        }
    )

    assert rows == [
        (1, "SÁNCHEZ SALAZAR, ÉDGAR MARTÍN", "61531833", "18"),
        (2, "RUIZ VÁSQUEZ, GABRIELA RUTH", "79352308", "16"),
    ]
