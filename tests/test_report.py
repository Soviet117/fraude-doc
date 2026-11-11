"""Pruebas del informe PDF."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from fraude.report import build_report


def test_build_report_creates_pdf(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.png"
    Image.new("RGB", (80, 110), (220, 220, 220)).save(overlay)
    analysis = {
        "created_at": "2026-11-10T12:00:00+00:00",
        "observation": {
            "document_id": "sha256:" + "d" * 64,
            "source": {"filename": "dni.png"},
            "doctype": {"label": "dni"},
            "fields": {"dni": "12345678", "nombres": "José"},
        },
        "fusor": {
            "p_fake": 0.91,
            "risk": "alto",
            "calibration_method": "identity",
            "evidences": [
                {"severity": "alta", "message": "El nombre difiere del registro."},
                {"severity": "media", "message": "La señal visual F4 es alta."},
            ],
        },
        "explanation": {"text": "Indicios para revisión humana.", "source": "template"},
        "f5": {"overlay_path": str(overlay)},
        "model_versions": {"f4": "fold_2", "f7": "fusion"},
        "timings_ms": {"total_ms": 1200},
    }

    output = build_report(analysis, tmp_path / "informe.pdf")

    assert output.is_file()
    assert output.stat().st_size > 2000
    assert output.read_bytes().startswith(b"%PDF")
