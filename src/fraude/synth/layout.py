"""Lienzo de dibujo con registro de cajas por campo.

Las plantillas dibujan con reportlab y guardan, para cada campo relevante, su
rectángulo en puntos PDF. Después se convierten a píxeles para que el generador
de manipulaciones (F2.3) sepa exactamente dónde actuar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

Alignment = Literal["left", "center", "right"]
Rect = tuple[float, float, float, float]


class DocCanvas:
    """Canvas de una página con registro de rectángulos por clave."""

    def __init__(self, pdf_path: Path, page_size: tuple[float, float] = A4) -> None:
        self.page_size = page_size
        self.pdf = canvas.Canvas(str(pdf_path), pagesize=page_size)
        self.rects: dict[str, Rect] = {}

    def text(
        self,
        key: str | None,
        x: float,
        y: float,
        text: str,
        *,
        font: str = "Helvetica",
        size: float = 10,
        color: colors.Color = colors.black,
        align: Alignment = "left",
    ) -> Rect:
        """Dibuja texto y registra su caja si se indica una clave."""
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(color)
        width = pdfmetrics.stringWidth(text, font, size)

        if align == "center":
            self.pdf.drawCentredString(x, y, text)
            left = x - width / 2
        elif align == "right":
            self.pdf.drawRightString(x, y, text)
            left = x - width
        else:
            self.pdf.drawString(x, y, text)
            left = x

        rect = (left, y - 2, max(width, 1.0), size + 4)
        if key:
            self.rects[key] = rect
        return rect

    def box(
        self,
        key: str | None,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        stroke: bool = True,
        fill: bool = False,
        color: colors.Color = colors.black,
        line_width: float = 0.6,
    ) -> Rect:
        """Dibuja un rectángulo y registra su caja."""
        self.pdf.setStrokeColor(color)
        self.pdf.setFillColor(color)
        self.pdf.setLineWidth(line_width)
        self.pdf.rect(x, y, width, height, stroke=int(stroke), fill=int(fill))
        rect = (x, y, width, height)
        if key:
            self.rects[key] = rect
        return rect

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: colors.Color = colors.black,
        width: float = 0.8,
    ) -> None:
        """Dibuja una línea simple."""
        self.pdf.setStrokeColor(color)
        self.pdf.setLineWidth(width)
        self.pdf.line(x1, y1, x2, y2)

    def save(self) -> None:
        """Cierra la página y guarda el PDF."""
        self.pdf.showPage()
        self.pdf.save()

    def rects_px(self, dpi: int) -> dict[str, tuple[int, int, int, int]]:
        """Convierte las cajas de puntos PDF a píxeles con origen arriba-izquierda."""
        scale = dpi / 72.0
        page_width, page_height = self.page_size
        result: dict[str, tuple[int, int, int, int]] = {}

        for key, (x, y, width, height) in self.rects.items():
            left = min(max(x, 0.0), page_width)
            right = min(max(x + width, 0.0), page_width)
            top = min(max(page_height - (y + height), 0.0), page_height)
            bottom = min(max(page_height - y, 0.0), page_height)
            result[key] = (
                round(left * scale),
                round(top * scale),
                max(1, round((right - left) * scale)),
                max(1, round((bottom - top) * scale)),
            )

        return result
