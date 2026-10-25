"""Extracción de campos estructurados desde OCR."""

from fraude.extract.fields import (
    DOCUMENT_KEYWORDS,
    FIELD_SPECS,
    ExtractionResult,
    detect_document_type,
    extract_fields,
    extract_from_lines,
    extract_from_text,
    extract_page,
)

__all__ = [
    "DOCUMENT_KEYWORDS",
    "FIELD_SPECS",
    "ExtractionResult",
    "detect_document_type",
    "extract_fields",
    "extract_from_lines",
    "extract_from_text",
    "extract_page",
]
