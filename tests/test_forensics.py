"""Pruebas de las señales forenses clásicas (F3, E3-1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from fraude.forensics.classic import analyze_forensics


def _document_image() -> Image.Image:
    image = Image.new("RGB", (300, 400), "white")
    draw = ImageDraw.Draw(image)
    for index in range(20):
        draw.text(
            (20, 20 + index * 15), f"Documento linea {index} valor {index * 17}", fill="black"
        )
    rng = np.random.default_rng(0)
    array = np.asarray(image, dtype=np.int16)
    array = np.clip(array + rng.normal(0, 3, array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _copy_move_image() -> Image.Image:
    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
    patch = array[20:80, 20:80].copy()
    array[140:200, 140:200] = patch
    return Image.fromarray(array)


def test_analyze_forensics_returns_bounded_signals(tmp_path: Path) -> None:
    path = tmp_path / "documento.png"
    _document_image().save(path)

    report = analyze_forensics(path, out_dir=tmp_path / "maps")

    assert report.ela.applicable is True
    assert report.noise.applicable is True
    assert report.chromatic.applicable is True
    assert report.jpeg.applicable is False
    assert report.metadata.score == 0.0
    assert report.localization.applicable is False
    for signal in (report.ela, report.noise, report.chromatic, report.metadata):
        assert signal.score is not None
        assert 0.0 <= signal.score <= 1.0
        if signal.map_path is not None:
            assert Path(signal.map_path).exists()


def test_saves_ela_noise_and_chromatic_maps(tmp_path: Path) -> None:
    path = tmp_path / "documento.png"
    _document_image().save(path)

    report = analyze_forensics(path, out_dir=tmp_path / "maps")

    for signal, name in (
        (report.ela, "ela.png"),
        (report.noise, "noise.png"),
        (report.chromatic, "chromatic.png"),
    ):
        assert signal.map_path is not None
        assert Path(signal.map_path).name == name
        assert Path(signal.map_path).exists()


def test_without_output_dir_does_not_write_maps(tmp_path: Path) -> None:
    path = tmp_path / "documento.png"
    _document_image().save(path)

    report = analyze_forensics(path)

    assert report.ela.map_path is None
    assert report.noise.map_path is None
    assert report.chromatic.map_path is None


def test_jpeg_signal_is_applicable_only_for_jpeg(tmp_path: Path) -> None:
    image = _document_image()
    png = tmp_path / "documento.png"
    jpeg = tmp_path / "documento.jpg"
    image.save(png)
    image.save(jpeg, quality=85)

    png_report = analyze_forensics(png)
    jpeg_report = analyze_forensics(jpeg)

    assert png_report.jpeg.applicable is False
    assert jpeg_report.jpeg.applicable is True
    assert jpeg_report.jpeg.score is not None
    assert 0.0 <= jpeg_report.jpeg.score <= 1.0
    assert jpeg_report.jpeg.details["quantization_tables"]


def test_metadata_flags_editing_software(tmp_path: Path) -> None:
    path = tmp_path / "editado.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "GIMP 2.10")
    _document_image().save(path, pnginfo=info)

    report = analyze_forensics(path)

    assert report.metadata.score is not None
    assert report.metadata.score > 0.0
    assert "software=GIMP 2.10" in report.metadata.details["reasons"]


def test_copy_move_detects_colored_duplicated_patch(tmp_path: Path) -> None:
    path = tmp_path / "copiado.png"
    _copy_move_image().save(path)

    report = analyze_forensics(path, out_dir=tmp_path / "maps")

    assert report.localization.applicable is True
    assert report.localization.score is not None
    assert report.localization.score > 0.5
    assert report.localization.map_path is not None
    assert Path(report.localization.map_path).exists()
    assert report.localization.details["matched_blocks"] > 0


def test_copy_move_ignores_repeated_black_text(tmp_path: Path) -> None:
    path = tmp_path / "texto.png"
    _document_image().save(path)

    report = analyze_forensics(path)

    assert report.localization.applicable is False
    assert report.localization.details["reason"] == "few_colored_blocks"
