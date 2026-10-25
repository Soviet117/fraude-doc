"""Pruebas del inventario documental (F1, E1-1)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfgen import canvas

from fraude.ingest.inventory import (
    build_inventory,
    iter_supported_files,
    write_inventory,
)


def _make_pdf(path: Path, pages: int = 1) -> None:
    pdf = canvas.Canvas(str(path), pagesize=(200, 300))
    for _ in range(pages):
        pdf.drawString(20, 250, "documento de prueba")
        pdf.showPage()
    pdf.save()


def test_build_inventory_describes_images_and_pdf(tmp_path: Path) -> None:
    Image.new("RGB", (120, 80), "white").save(tmp_path / "acta.png", dpi=(300, 300))
    _make_pdf(tmp_path / "certificado.pdf", pages=2)

    entries = build_inventory(tmp_path)
    by_name = {entry.relative_path: entry for entry in entries}

    assert set(by_name) == {"acta.png", "certificado.pdf"}

    png = by_name["acta.png"]
    assert png.format == "png"
    assert png.width == 120
    assert png.height == 80
    assert png.dpi == 300
    assert png.pages is None
    assert png.document_id.startswith("sha256:")
    assert png.bytes > 0

    pdf = by_name["certificado.pdf"]
    assert pdf.format == "pdf"
    assert pdf.pages == 2
    assert pdf.width is None


def test_write_inventory_creates_csv(tmp_path: Path) -> None:
    Image.new("RGB", (10, 10), "white").save(tmp_path / "doc.png")
    entries = build_inventory(tmp_path)
    out = tmp_path / "registry" / "inventory.csv"

    write_inventory(entries, out)

    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["relative_path"] == "doc.png"
    assert rows[0]["format"] == "png"


def test_iter_supported_files_skips_hidden_and_unsupported(tmp_path: Path) -> None:
    Image.new("RGB", (10, 10), "white").save(tmp_path / "visible.png")
    Image.new("RGB", (10, 10), "white").save(tmp_path / ".oculto.png")
    (tmp_path / "notas.txt").write_text("no es un documento", encoding="utf-8")
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(hidden_dir / "interno.png")

    paths = [path.name for path in iter_supported_files(tmp_path)]

    assert paths == ["visible.png"]


def test_build_inventory_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        build_inventory(tmp_path / "no_existe")
