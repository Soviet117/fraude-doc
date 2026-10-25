"""Inventario técnico de documentos (fase F1, historia E1-1).

Recorre una carpeta, calcula el hash SHA-256 de cada documento y extrae
metadatos útiles para la caracterización: formato, tamaño, dimensiones y DPI
(imágenes) o número de páginas (PDF). No interpreta el contenido ni decide
nada sobre autenticidad.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from fraude.ingest.hashing import sha256_file

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
SUPPORTED_PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | SUPPORTED_PDF_SUFFIXES


@dataclass(frozen=True)
class InventoryEntry:
    """Una fila del inventario documental."""

    relative_path: str
    document_id: str
    bytes: int
    format: str
    width: int | None = None
    height: int | None = None
    dpi: int | None = None
    pages: int | None = None


def iter_supported_files(root: Path) -> Iterable[Path]:
    """Devuelve los archivos soportados bajo ``root``, en orden estable.

    Se ignoran archivos y carpetas ocultas (por ejemplo ``.git``).
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        yield path


def _image_metadata(path: Path) -> tuple[int | None, int | None, int | None]:
    """Devuelve ``(width, height, dpi)`` de una imagen, o ``None`` si falla."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            dpi_info = image.info.get("dpi")
            dpi = round(float(dpi_info[0])) if dpi_info else None
            return width, height, dpi
    except (UnidentifiedImageError, OSError):
        return None, None, None


def _pdf_pages(path: Path) -> int | None:
    """Devuelve el número de páginas de un PDF, o ``None`` si no se puede leer."""
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover - pypdfium2 es dependencia core
        return None

    try:
        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return None


def describe_file(path: Path, root: Path) -> InventoryEntry:
    """Construye la entrada de inventario para un archivo soportado."""
    suffix = path.suffix.lower()
    width = height = dpi = pages = None

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        width, height, dpi = _image_metadata(path)
    elif suffix in SUPPORTED_PDF_SUFFIXES:
        pages = _pdf_pages(path)

    return InventoryEntry(
        relative_path=path.relative_to(root).as_posix(),
        document_id=sha256_file(path),
        bytes=path.stat().st_size,
        format=suffix.lstrip("."),
        width=width,
        height=height,
        dpi=dpi,
        pages=pages,
    )


def build_inventory(root: Path) -> list[InventoryEntry]:
    """Recorre ``root`` y devuelve una entrada por documento soportado."""
    if not root.is_dir():
        raise NotADirectoryError(f"No es un directorio: {root}")
    return [describe_file(path, root) for path in iter_supported_files(root)]


def write_inventory(entries: Sequence[InventoryEntry], out_path: Path) -> None:
    """Escribe el inventario en CSV (UTF-8), creando el directorio si falta."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(InventoryEntry.__dataclass_fields__)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m fraude.ingest.inventory <carpeta> [--out CSV]``."""
    parser = argparse.ArgumentParser(description="Inventario técnico de documentos.")
    parser.add_argument("root", type=Path, help="Carpeta con los documentos.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/registry/inventory.csv"),
        help="Ruta del CSV de salida.",
    )
    args = parser.parse_args(argv)

    entries = build_inventory(args.root)
    write_inventory(entries, args.out)
    print(f"{len(entries)} documentos inventariados -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
