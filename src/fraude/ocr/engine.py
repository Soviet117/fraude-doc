"""Motor OCR docTR y estructura de palabras (F3, E3-2).

docTR vive en la extra ``ml``. El import es diferido para que el CI y la ruta
rápida puedan importar este módulo sin instalar PyTorch. La salida se normaliza
a ``OcrPage`` para que extracción, reglas y tests no dependan de docTR.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

Box = tuple[int, int, int, int]


class OcrUnavailableError(RuntimeError):
    """docTR no está instalado o no pudo cargarse."""


@dataclass(frozen=True)
class OcrWord:
    """Palabra reconocida con caja en píxeles y confianza."""

    text: str
    bbox: Box
    confidence: float


@dataclass(frozen=True)
class OcrPage:
    """Resultado OCR de una página."""

    words: tuple[OcrWord, ...]
    width: int
    height: int
    engine: str = "doctr"

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def mean_conf(self) -> float | None:
        if not self.words:
            return None
        return round(sum(word.confidence for word in self.words) / len(self.words), 4)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in group_lines(self.words))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "width": self.width,
            "height": self.height,
            "words": [
                {
                    "text": word.text,
                    "bbox": list(word.bbox),
                    "confidence": word.confidence,
                }
                for word in self.words
            ],
        }


@dataclass(frozen=True)
class OcrLine:
    """Palabras agrupadas en una línea de texto."""

    words: tuple[OcrWord, ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def bbox(self) -> Box:
        left = min(word.bbox[0] for word in self.words)
        top = min(word.bbox[1] for word in self.words)
        right = max(word.bbox[0] + word.bbox[2] for word in self.words)
        bottom = max(word.bbox[1] + word.bbox[3] for word in self.words)
        return (left, top, right - left, bottom - top)

    @property
    def confidence(self) -> float:
        return round(sum(word.confidence for word in self.words) / len(self.words), 4)


def run_ocr(image_path: Path, *, predictor: Any | None = None) -> OcrPage:
    """Ejecuta docTR sobre una imagen y devuelve palabras normalizadas.

    ``predictor`` permite inyectar un motor falso en tests. En producción se
    carga una sola vez con ``_get_predictor``.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size

    if predictor is None:
        try:
            from doctr.io import DocumentFile
        except ImportError as exc:  # pragma: no cover - depende de la extra ml
            raise OcrUnavailableError(
                "docTR no está instalado; ejecuta `uv sync --extra ml` o usa Docker"
            ) from exc
        predictor = _get_predictor()
        document = DocumentFile.from_images(str(image_path))
    else:
        document = image_path

    result = predictor(document)
    export = result.export() if hasattr(result, "export") else result
    return parse_doctr_export(export, page_size=(width, height))


@lru_cache(maxsize=1)
def _get_predictor() -> Any:  # pragma: no cover - requiere docTR y modelos
    """Carga docTR una sola vez por proceso."""
    try:
        from doctr.models import ocr_predictor
    except ImportError as exc:
        raise OcrUnavailableError("docTR no está instalado") from exc
    return ocr_predictor(pretrained=True)


def parse_doctr_export(
    export: dict[str, Any],
    *,
    page_size: tuple[int, int],
) -> OcrPage:
    """Convierte el JSON de docTR a ``OcrPage`` usando la geometría normalizada."""
    width, height = page_size
    words: list[OcrWord] = []
    for page in export.get("pages", [])[:1]:
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    text = str(word.get("value", "")).strip()
                    if not text:
                        continue
                    bbox = _normalized_bbox(word.get("geometry"), width, height)
                    if bbox is None:
                        continue
                    words.append(
                        OcrWord(
                            text=text,
                            bbox=bbox,
                            confidence=float(word.get("confidence", 0.0)),
                        )
                    )
    return OcrPage(words=tuple(words), width=width, height=height)


def group_lines(words: tuple[OcrWord, ...] | list[OcrWord]) -> list[OcrLine]:
    """Agrupa palabras por línea usando el centro vertical y el solapamiento."""
    if not words:
        return []
    ordered = sorted(words, key=lambda word: (word.bbox[1] + word.bbox[3] / 2, word.bbox[0]))
    lines: list[list[OcrWord]] = []
    for word in ordered:
        center = word.bbox[1] + word.bbox[3] / 2
        placed = False
        for line in lines:
            line_center = sum(item.bbox[1] + item.bbox[3] / 2 for item in line) / len(line)
            tolerance = max(5.0, 0.6 * min(item.bbox[3] for item in (*line, word)))
            if abs(center - line_center) <= tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    result = [OcrLine(words=tuple(sorted(line, key=lambda word: word.bbox[0]))) for line in lines]
    return sorted(result, key=lambda line: line.bbox[1])


def _normalized_bbox(geometry: Any, width: int, height: int) -> Box | None:
    try:
        (x0, y0), (x1, y1) = geometry
    except (TypeError, ValueError):
        return None
    left = max(0, min(width - 1, round(float(x0) * width)))
    top = max(0, min(height - 1, round(float(y0) * height)))
    right = max(left + 1, min(width, round(float(x1) * width)))
    bottom = max(top + 1, min(height, round(float(y1) * height)))
    return (left, top, right - left, bottom - top)
