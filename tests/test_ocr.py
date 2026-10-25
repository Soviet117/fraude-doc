"""Pruebas del adaptador OCR y layout (F3, E3-2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from fraude.ocr.engine import OcrPage, OcrWord, group_lines, parse_doctr_export, run_ocr
from fraude.ocr.layout import layout_page, line_text, text_by_region


def _export() -> dict[str, Any]:
    return {
        "pages": [
            {
                "blocks": [
                    {
                        "lines": [
                            {
                                "words": [
                                    {
                                        "value": "Hola",
                                        "confidence": 0.9,
                                        "geometry": ((0.1, 0.1), (0.3, 0.2)),
                                    },
                                    {
                                        "value": "mundo",
                                        "confidence": 0.8,
                                        "geometry": ((0.35, 0.1), (0.6, 0.2)),
                                    },
                                    {
                                        "value": "segunda",
                                        "confidence": 0.7,
                                        "geometry": ((0.1, 0.5), (0.4, 0.6)),
                                    },
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }


def test_parse_doctr_export_normalizes_geometry() -> None:
    page = parse_doctr_export(_export(), page_size=(100, 200))

    assert page.n_words == 3
    assert page.width == 100
    assert page.height == 200
    assert page.words[0].bbox == (10, 20, 20, 20)
    assert page.words[1].bbox == (35, 20, 25, 20)
    assert page.words[2].bbox == (10, 100, 30, 20)
    assert page.mean_conf == 0.8
    assert page.text == "Hola mundo\nsegunda"


def test_run_ocr_accepts_injected_predictor(tmp_path: Path) -> None:
    path = tmp_path / "documento.png"
    Image.new("RGB", (100, 200), "white").save(path)

    def predictor(document: Any) -> dict[str, Any]:
        assert document == path
        return _export()

    page = run_ocr(path, predictor=predictor)

    assert page.n_words == 3
    assert page.engine == "doctr"


def test_group_lines_builds_reading_order() -> None:
    words = (
        OcrWord("segunda", (10, 100, 30, 20), 0.7),
        OcrWord("mundo", (40, 20, 25, 20), 0.8),
        OcrWord("Hola", (10, 20, 20, 20), 0.9),
    )

    lines = group_lines(words)

    assert [line.text for line in lines] == ["Hola mundo", "segunda"]
    assert lines[0].bbox == (10, 20, 55, 20)
    assert lines[0].confidence == 0.85


def test_layout_classifies_header_body_and_footer() -> None:
    page = OcrPage(
        words=(
            OcrWord("CETPRO", (10, 0, 30, 10), 0.99),
            OcrWord("contenido", (10, 500, 40, 10), 0.95),
            OcrWord("firma", (10, 950, 20, 10), 0.90),
        ),
        width=100,
        height=1000,
    )

    regions = [item.region for item in layout_page(page)]

    assert regions == ["header", "body", "footer"]
    assert line_text(page) == "CETPRO\ncontenido\nfirma"
    assert text_by_region(page) == {
        "header": "CETPRO",
        "body": "contenido",
        "footer": "firma",
    }


def test_empty_page_has_no_confidence() -> None:
    page = OcrPage(words=(), width=100, height=200)

    assert page.n_words == 0
    assert page.mean_conf is None
    assert page.text == ""
    assert page.to_dict()["words"] == []
