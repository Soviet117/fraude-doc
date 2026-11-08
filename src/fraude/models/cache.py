"""Caché de imágenes redimensionadas para acelerar F4 en CPU.

Las páginas sintéticas miden 1241x1754 px. Leer y redimensionar el PNG
original en cada época domina el tiempo de entrenamiento en CPU, así que se
genera una copia cuadrada de 256 px con relleno blanco (letterbox). El
letterbox conserva la página completa y su relación de aspecto; un resize
directo a 224x224 deformaría el texto y las firmas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from fraude.synth.dataset import DocumentRecord

DEFAULT_CACHE_SIZE = 256
WHITE = (255, 255, 255)


def letterbox_square(
    image: Image.Image, size: int, fill: tuple[int, int, int] = WHITE
) -> Image.Image:
    """Escala la imagen completa y la centra en un lienzo cuadrado."""
    if size < 32:
        raise ValueError("size debe ser al menos 32")
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def build_cache(
    records: list[DocumentRecord],
    data_dir: Path,
    cache_dir: Path,
    *,
    size: int = DEFAULT_CACHE_SIZE,
    overwrite: bool = False,
) -> dict[str, str]:
    """Genera la caché y devuelve ``sample_id`` -> ruta relativa."""
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    written = 0

    for record in records:
        target = cache_dir / f"{record.sample_id}.png"
        mapping[record.sample_id] = str(target)
        if target.exists() and not overwrite:
            continue
        source = data_dir / record.image_path
        if not source.exists():
            raise FileNotFoundError(f"Falta la imagen {source}")
        with Image.open(source) as image:
            letterbox_square(image, size).save(target, format="PNG")
        written += 1

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.models.cache",
        "size": size,
        "count": len(mapping),
        "written": written,
    }
    (cache_dir / "cache.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return mapping


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return Path(cache_dir) / f"{sample_id}.png"
