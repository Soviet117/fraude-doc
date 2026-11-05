"""Pruebas de la auditoría de integridad de corpus."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from fraude.core.schemas import ForensicsReport, QualityReport, RuleFinding, Severity, SignalScore
from fraude.eval.corpus_audit import audit_corpus


def _make_image(
    path: Path,
    marker: str,
    size: tuple[int, int] = (1200, 1600),
) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 40), marker, fill="black")
    for index in range(30):
        draw.text((80, 80 + index * 35), f"Documento de prueba linea {index}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _make_mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int] | None) -> None:
    mask = Image.new("L", size, 0)
    if box is not None:
        ImageDraw.Draw(mask).rectangle(box, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path, format="PNG")


def _make_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n% documento de prueba\n")


def _base_row(
    sample_id: str, *, tipo: str, variante: str, ataque: str = "", **extra: str
) -> dict[str, str]:
    row = {
        "id": sample_id,
        "tipo": tipo,
        "variante": variante,
        "ataque": ataque,
        "persona": "P01",
        "documento_padre": "",
        "campo_original": "",
        "campo_alterado": "",
        "pdf": f"pdf/{sample_id}.pdf",
        "png": f"png/{sample_id}.png",
        "mascara": "",
        "bbox_px": "",
        "sha256_png": "",
    }
    row.update(extra)
    return row


def _write_index(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "_indice.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "_personas.csv").write_text("persona_id\nP01\n", encoding="utf-8")
    image_size = (1200, 1600)
    for sample_id in ("A01", "M01", "S01"):
        _make_image(root / "png" / f"{sample_id}.png", sample_id, image_size)
        _make_pdf(root / "pdf" / f"{sample_id}.pdf")

    _make_mask(root / "mask" / "M01_mask.png", image_size, (10, 10, 40, 40))
    _make_mask(root / "mask" / "S01_mask.png", image_size, None)

    rows = [
        _base_row(
            "A01",
            tipo="dni",
            variante="authentic",
            sha256_png=_sha(root / "png" / "A01.png"),
        ),
        _base_row(
            "M01",
            tipo="dni",
            variante="tampered",
            ataque="T1",
            documento_padre="A01",
            campo_original="14",
            campo_alterado="18",
            mascara="mask/M01_mask.png",
            bbox_px="10,10,40,40",
            sha256_png=_sha(root / "png" / "M01.png"),
        ),
        _base_row(
            "S01",
            tipo="ficha_matricula",
            variante="tampered",
            ataque="T8",
            campo_original="periodo coherente",
            campo_alterado="matrícula fuera de periodo",
            mascara="mask/S01_mask.png",
            sha256_png=_sha(root / "png" / "S01.png"),
        ),
    ]
    _write_index(root, rows)
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_corpus_accepts_visual_and_semantic_masks(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "muestras")

    report = audit_corpus(root)

    assert report.ok is True
    assert report.n_samples == 3
    assert report.n_errors == 0
    assert report.n_warnings == 0
    assert report.by_attack == {"T1": 1, "T8": 1}
    by_id = {sample.sample_id: sample for sample in report.samples}
    assert by_id["M01"].mask_pixels and by_id["M01"].mask_pixels > 0
    assert by_id["M01"].bbox == (10, 10, 40, 40)
    assert by_id["S01"].mask_pixels == 0
    assert by_id["S01"].bbox is None


def test_audit_corpus_reports_hash_mismatch(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "muestras")
    rows = list(csv.DictReader((root / "_indice.csv").open(encoding="utf-8")))
    rows[0]["sha256_png"] = "0" * 64
    _write_index(root, rows)

    report = audit_corpus(root)

    assert report.ok is False
    assert "sha256_mismatch" in {issue.code for issue in report.issues}


def test_audit_corpus_requires_mask_for_visual_attack(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "muestras")
    (root / "mask" / "M01_mask.png").unlink()

    report = audit_corpus(root)

    assert report.ok is False
    assert "missing_mask" in {issue.code for issue in report.issues}


def test_audit_corpus_detects_bbox_mask_mismatch(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "muestras")
    rows = list(csv.DictReader((root / "_indice.csv").open(encoding="utf-8")))
    rows[1]["bbox_px"] = "20,20,50,50"
    _write_index(root, rows)

    report = audit_corpus(root)

    assert report.ok is False
    assert "bbox_mask_mismatch" in {issue.code for issue in report.issues}


def test_audit_corpus_warns_nonempty_mask_on_semantic_attack(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "muestras")
    _make_mask(root / "mask" / "S01_mask.png", (1200, 1600), (10, 10, 40, 40))

    report = audit_corpus(root)

    assert report.ok is True
    assert "mask_on_semantic_attack" in {issue.code for issue in report.issues}


def test_audit_corpus_flags_metadata_shortcut(tmp_path: Path, monkeypatch) -> None:
    root = _corpus(tmp_path / "muestras")

    def fake_analyze(path, **_kwargs):
        flagged = not Path(path).stem.startswith("A")
        metadata = SignalScore(
            score=0.5 if flagged else 0.0,
            details={"reasons": ["software=Adobe Photoshop"] if flagged else []},
        )
        rules = (
            [
                RuleFinding(
                    code="METADATA_EDICION",
                    severity=Severity.MEDIUM,
                    message="metadatos de edición",
                )
            ]
            if flagged
            else []
        )
        return SimpleNamespace(
            quality=QualityReport(ok=True),
            forensics=ForensicsReport(metadata=metadata, localization=SignalScore(score=0.0)),
            rules=rules,
        )

    monkeypatch.setattr("fraude.eval.corpus_audit.analyze_document", fake_analyze)

    report = audit_corpus(root, analyze_images=True)

    summary = report.pipeline_summary
    assert summary is not None
    assert summary["n_analyzed"] == 3
    assert summary["by_label"]["authentic"]["metadata_flagged"] == 0
    assert summary["by_label"]["tampered"]["metadata_flagged"] == 2
    codes = {warning["code"] for warning in summary["warnings"]}
    assert "metadata_shortcut" in codes
    assert "copy_move_not_detected" in codes


def test_audit_corpus_rejects_index_without_required_columns(tmp_path: Path) -> None:
    root = tmp_path / "muestras"
    root.mkdir()
    (root / "_indice.csv").write_text("id,tipo\nM01,dni\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Faltan columnas"):
        audit_corpus(root)
