"""Renderizado de documentos sintéticos: PDF con reportlab y PNG con pypdfium2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

from fraude.ingest.hashing import sha256_file
from fraude.synth.data import Identity
from fraude.synth.dataset import DocumentRecord
from fraude.synth.layout import DocCanvas
from fraude.synth.templates import TEMPLATES

TEMPLATE_VERSION = "v1"


def render_document(
    *,
    sample_id: str,
    document_type: str,
    identity: Identity,
    fields: dict[str, Any],
    out_dir: Path,
    seed: int = 42,
    dpi: int = 150,
) -> DocumentRecord:
    """Dibuja una plantilla, la rasteriza y devuelve su registro de dataset."""
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    image_dir = out_dir / "images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{sample_id}.pdf"
    image_path = image_dir / f"{sample_id}.png"

    template = TEMPLATES.get(document_type)
    if template is None:
        raise ValueError(f"Tipo de documento no soportado: {document_type}")

    dc = DocCanvas(pdf_path)
    template(dc, identity, fields)
    dc.save()

    image = rasterize_pdf(pdf_path, dpi=dpi)
    image.save(image_path, format="PNG", dpi=(dpi, dpi))

    return DocumentRecord(
        sample_id=sample_id,
        document_id=sha256_file(image_path),
        source_id=identity.source_id,
        document_type=document_type,
        template_version=TEMPLATE_VERSION,
        variant="authentic",
        parent_id=None,
        is_authentic=True,
        pdf_path=_relative(pdf_path, out_dir),
        image_path=_relative(image_path, out_dir),
        fields=dict(fields),
        field_boxes=dc.rects_px(dpi),
        page_size=image.size,
        capture_type="pdf_render",
        seed=seed,
    )


def rasterize_pdf(pdf_path: Path, dpi: int = 150) -> Image.Image:
    """Convierte la primera página de un PDF a una imagen RGB."""
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[0]
        bitmap = page.render(scale=dpi / 72.0)
        return bitmap.to_pil().convert("RGB")
    finally:
        document.close()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
