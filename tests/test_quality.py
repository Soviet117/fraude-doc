"""Pruebas de la puerta de calidad (F2, E2-1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from fraude.core.config import QualityConfig, load_config
from fraude.core.schemas import CaptureType
from fraude.pipeline.run import analyze_bytes, analyze_document
from fraude.preprocess.quality import (
    assess_quality,
    detect_capture_type,
    estimate_dpi,
)


def _text_like_image(
    size: tuple[int, int] = (1200, 1600),
    angle: float = 0.0,
) -> Image.Image:
    """Genera una imagen con líneas horizontales que simulan renglones de texto."""
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    for y in range(120, size[1] - 120, 55):
        draw.rectangle([100, y, size[0] - 100, y + 14], fill=0)
    if angle:
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=255,
        )
    return image.convert("RGB")


def test_sharp_image_passes() -> None:
    report = assess_quality(_text_like_image())

    assert report.ok is True
    assert report.issues == []
    assert report.blur is not None and report.blur > 100
    assert abs(report.skew_deg or 0.0) <= 2.0


def test_blurred_image_is_rejected() -> None:
    blurred = _text_like_image().filter(ImageFilter.GaussianBlur(radius=10))

    report = assess_quality(blurred)

    assert report.ok is False
    assert "blur" in report.issues


def test_low_resolution_is_rejected() -> None:
    report = assess_quality(_text_like_image(size=(200, 260)))

    assert report.ok is False
    assert "low_resolution" in report.issues


def test_skewed_image_is_detected() -> None:
    skewed = _text_like_image(angle=4.0)

    report = assess_quality(skewed)

    assert "skew" in report.issues
    assert abs(report.skew_deg or 0.0) > 2.0


def test_overexposed_and_underexposed_images_are_rejected() -> None:
    white = Image.new("RGB", (1200, 1600), "white")
    black = Image.new("RGB", (1200, 1600), "black")

    white_report = assess_quality(white)
    black_report = assess_quality(black)

    assert "overexposed" in white_report.issues
    assert "underexposed" in black_report.issues


def test_quality_config_is_loaded_from_yaml() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert isinstance(config.quality, QualityConfig)
    assert config.quality.min_side_px == 1000
    assert config.quality.max_skew_deg == 2.0


def test_capture_type_from_dpi_pdf_and_screen(tmp_path: Path) -> None:
    scanned = _text_like_image()
    assert detect_capture_type(scanned, "pdf") is CaptureType.PDF_RENDER

    screen = Image.new("RGB", (1920, 1080), "white")
    assert detect_capture_type(screen, "png") is CaptureType.SCREENSHOT

    photo_path = tmp_path / "foto.jpg"
    exif = Image.Exif()
    exif[271] = "TestMake"
    exif[272] = "TestModel"
    Image.new("RGB", (1200, 1600), "white").save(photo_path, exif=exif)
    with Image.open(photo_path) as photo:
        assert detect_capture_type(photo, "jpg") is CaptureType.PHOTO


def test_estimate_dpi(tmp_path: Path) -> None:
    path = tmp_path / "scan.png"
    _text_like_image().save(path, dpi=(300, 300))

    with Image.open(path) as image:
        assert estimate_dpi(image) == 300


def test_pipeline_assesses_quality_for_images(tmp_path: Path) -> None:
    path = tmp_path / "acta.png"
    _text_like_image().save(path, dpi=(300, 300))

    observation = analyze_document(path)

    assert observation.quality.ok is True
    assert observation.capture.type is CaptureType.SCAN
    assert observation.capture.dpi_est == 300


def test_pipeline_flags_unreadable_image() -> None:
    observation = analyze_bytes(b"no es una imagen", "ficha.png")

    assert observation.quality.ok is False
    assert observation.quality.issues == ["unreadable_image"]


@pytest.mark.parametrize("angle", [-3.0, 3.0])
def test_skew_sign_is_consistent(angle: float) -> None:
    report = assess_quality(_text_like_image(angle=angle))

    assert report.skew_deg is not None
    assert report.skew_deg * angle < 0  # el ángulo reportado es la corrección
