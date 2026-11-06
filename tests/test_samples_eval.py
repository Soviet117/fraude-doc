"""Pruebas del evaluador del pipeline sobre el corpus de muestras."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from fraude.core.schemas import (
    DoctypeInfo,
    Observation,
    OcrReport,
    QualityReport,
    RegistryReport,
    RuleFinding,
    Severity,
    SourceInfo,
)
from fraude.eval import samples_eval
from fraude.eval.samples_eval import (
    _load_registry,
    _strip_metadata,
    run_samples_eval,
)

PERSONAS_CSV = """persona_id,apellidos,nombres,dni,sexo,fecha_nacimiento,direccion,programa,modulo,periodo
P01,SÁNCHEZ SALAZAR,ÉDGAR MARTÍN,61531833,M,03/06/1999,"Av. Circunvalación 45, Soritor",Computación e Informática,Ofimática y herramientas digitales,2025-II
P02,RUIZ VÁSQUEZ,GABRIELA RUTH,79352308,F,28/05/2002,"Av. San Martín 1180, Nueva Cajamarca",Administración de Empresas,Contabilidad básica,2025-II
"""

INDEX_CSV = """id,tipo,variante,ataque,persona,documento_padre,campo_original,campo_alterado,png
A1,dni,authentic,,P01,,,,png/A1.png
A2,dni,authentic,,P02,,,,png/A2.png
T1,dni,tampered,T1,P01,,,,png/T1.png
T2,dni,tampered,T2,P02,,,,png/T2.png
"""


def _write_png(path: Path, *, software: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), "white")
    if software is None:
        image.save(path)
        return
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", software)
    image.save(path, pnginfo=info)


def _write_corpus(root: Path, *, software: str | None = "Adobe Photoshop 25.0") -> None:
    (root / "png").mkdir(parents=True)
    (root / "_personas.csv").write_text(PERSONAS_CSV, encoding="utf-8")
    (root / "_indice.csv").write_text(INDEX_CSV, encoding="utf-8")
    for sample_id in ("A1", "A2", "T1", "T2"):
        _write_png(root / "png" / f"{sample_id}.png", software=software)


def _observation(rules: tuple[RuleFinding, ...] = ()) -> Observation:
    return Observation(
        document_id="sha256:" + "a" * 64,
        source=SourceInfo(filename="sample.png", format="png", bytes=16),
        doctype=DoctypeInfo(label="dni", confidence=1.0),
        quality=QualityReport(ok=True),
        ocr=OcrReport(engine="fake", mean_conf=0.95, n_words=20),
        fields={"dni": "12345678"},
        rules=list(rules),
        registry=RegistryReport(matched=True, score=1.0),
        timings={"ocr_ms": 1},
    )


def test_strip_metadata_removes_text_chunks(tmp_path: Path) -> None:
    src = tmp_path / "src.png"
    _write_png(src, software="Adobe Photoshop 25.0")
    with Image.open(src) as image:
        assert "Software" in image.info

    dst = _strip_metadata(src, tmp_path / "dst.png")

    with Image.open(dst) as image:
        assert "Software" not in image.info
        assert image.size == (16, 16)


def test_load_registry_from_personas(tmp_path: Path) -> None:
    path = tmp_path / "personas.csv"
    path.write_text(PERSONAS_CSV, encoding="utf-8")

    registry = _load_registry(path)

    assert len(registry) == 2
    assert registry[0].nombre == "SÁNCHEZ SALAZAR, ÉDGAR MARTÍN"
    assert registry[0].dni == "61531833"
    assert registry[0].periodo == "2025-II"
    assert registry[0].source_id == "P01"


def test_run_samples_eval_computes_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "muestras"
    _write_corpus(root)

    def fake_analyze(path: Path, *, registry, today):
        del registry, today
        sample_id = Path(path).stem
        if sample_id == "T1":
            rules = (
                RuleFinding(
                    code="REGISTRO_DIFERENCIA",
                    severity=Severity.HIGH,
                    message="diferencia de identidad",
                ),
            )
        elif sample_id == "A2":
            rules = (
                RuleFinding(
                    code="RUIDO_LEVE",
                    severity=Severity.LOW,
                    message="señal leve",
                ),
            )
        else:
            rules = ()
        return _observation(rules)

    monkeypatch.setattr(samples_eval, "analyze_document", fake_analyze)

    payload = run_samples_eval(
        root,
        modes=("as_is",),
        out_path=tmp_path / "eval.json",
        n_boot=50,
        seed=7,
    )

    summary = payload["runs"]["as_is"]["summary"]
    metrics = summary["metrics"]
    assert metrics["tp"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert summary["doctype_accuracy"] == 1.0
    assert summary["metadata_findings"] == {"authentic": 0, "tampered": 0}
    assert summary["by_attack"]["authentic"]["n_detected"] == 0
    assert summary["by_attack"]["T1"]["n_detected"] == 1
    assert summary["by_attack"]["T2"]["n_detected"] == 0

    written = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert written["corpus"]["n_samples"] == 4
    assert written["runs"]["as_is"]["samples"][2]["sample_id"] == "T1"


def test_stripped_mode_removes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "muestras"
    _write_corpus(root, software="Adobe Photoshop 25.0")
    seen: dict[str, bool] = {}

    def fake_analyze(path: Path, *, registry, today):
        del registry, today
        with Image.open(path) as image:
            seen["has_software"] = "Software" in image.info
        return _observation()

    monkeypatch.setattr(samples_eval, "analyze_document", fake_analyze)

    run_samples_eval(
        root,
        modes=("stripped",),
        out_path=None,
        n_boot=10,
    )

    assert seen["has_software"] is False


def test_run_samples_eval_reports_sample_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "muestras"
    _write_corpus(root, software=None)

    def fake_analyze(path: Path, *, registry, today):
        del registry, today
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(samples_eval, "analyze_document", fake_analyze)

    payload = run_samples_eval(
        root,
        modes=("as_is",),
        out_path=None,
        n_boot=10,
    )

    summary = payload["runs"]["as_is"]["summary"]
    assert summary["n_ok"] == 0
    assert summary["n_errors"] == 4
    assert summary["errors"][0]["error"].startswith("fallo simulado")
    assert summary["metrics"]["precision_defined"] is False


def test_default_today_is_fixed() -> None:
    assert date(2026, 11, 5) == samples_eval.DEFAULT_TODAY
