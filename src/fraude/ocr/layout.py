"""Layout simple de líneas OCR por bandas verticales (F3, E3-2)."""

from __future__ import annotations

from dataclasses import dataclass

from fraude.ocr.engine import OcrLine, OcrPage, group_lines

HEADER_LIMIT = 0.15
FOOTER_LIMIT = 0.85


@dataclass(frozen=True)
class LaidOutLine:
    """Línea OCR con su región aproximada."""

    line: OcrLine
    region: str


def layout_page(page: OcrPage) -> tuple[LaidOutLine, ...]:
    """Clasifica las líneas en encabezado, cuerpo o pie."""
    lines: list[LaidOutLine] = []
    for line in group_lines(page.words):
        _x, y, _width, height = line.bbox
        center = (y + height / 2) / max(1, page.height)
        if center < HEADER_LIMIT:
            region = "header"
        elif center > FOOTER_LIMIT:
            region = "footer"
        else:
            region = "body"
        lines.append(LaidOutLine(line=line, region=region))
    return tuple(lines)


def text_by_region(page: OcrPage) -> dict[str, str]:
    """Devuelve el texto concatenado por región."""
    grouped: dict[str, list[str]] = {"header": [], "body": [], "footer": []}
    for item in layout_page(page):
        grouped[item.region].append(item.line.text)
    return {region: "\n".join(lines) for region, lines in grouped.items()}


def line_text(page: OcrPage) -> str:
    """Texto completo en orden de lectura."""
    return "\n".join(item.line.text for item in layout_page(page))
