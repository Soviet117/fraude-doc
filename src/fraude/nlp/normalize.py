"""Normalización y similitud de entidades para la rama semántica (F6)."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_DNI_RE = re.compile(r"\b\d{8}\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?51\s?)?9\d{2}[\s-]?\d{3}[\s-]?\d{3}\b")


def fold_text(value: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.lower().split())


def normalize_name(value: str) -> str:
    """Ordena ``Apellidos, Nombres`` como ``nombres apellidos`` y colapsa."""
    folded = fold_text(value).replace(",", " ")
    return " ".join(folded.split())


def content_fingerprint(fields: dict[str, object]) -> str:
    """Huella textual estable de los campos verificables de un documento.

    El dHash de una página sintética está dominado por la plantilla: dos
    documentos distintos del mismo tipo pueden tener el mismo hash. La huella
    de contenido evita confundirlos y solo se usa como compuerta para los casi
    duplicados; la copia exacta se detecta por píxeles.
    """
    values = [
        fold_text(str(fields[key])) for key in sorted(fields) if fields[key] not in (None, "")
    ]
    return " | ".join(values)


def normalize_dni(value: str) -> str | None:
    match = _DNI_RE.search(value)
    return match.group(0) if match else None


def normalize_date(value: str) -> str | None:
    match = _DATE_RE.search(value)
    return match.group(0) if match else None


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extrae entidades verificables por patrón: DNI, fechas, correo y teléfono."""
    return {
        "dni": sorted(set(_DNI_RE.findall(text))),
        "fechas": sorted(set(_DATE_RE.findall(text))),
        "correos": sorted(set(_EMAIL_RE.findall(text))),
        "telefonos": sorted(set(_PHONE_RE.findall(text))),
    }


def name_similarity(first: str, second: str) -> float:
    """Similitud 0-1 tolerante a tildes, orden y errores leves de OCR."""
    left = normalize_name(first)
    right = normalize_name(second)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    try:
        from rapidfuzz import fuzz

        return round(float(fuzz.token_set_ratio(left, right)) / 100.0, 4)
    except ImportError:
        return round(SequenceMatcher(None, left, right).ratio(), 4)


def canonical_name(fields: dict[str, object]) -> str | None:
    """Devuelve el nombre del titular en el orden canónico si está disponible."""
    for key in ("estudiante", "nombre"):
        value = fields.get(key)
        if value:
            return str(value)
    nombres = fields.get("nombres")
    apellidos = fields.get("apellidos")
    if nombres and apellidos:
        return f"{apellidos}, {nombres}"
    return None


def field_matches(gold: object, extracted: object, *, kind: str = "text") -> bool:
    """Compara un campo extraído con el valor de referencia del manifiesto."""
    if gold in (None, "") or extracted in (None, ""):
        return False
    gold_text = str(gold)
    extracted_text = str(extracted)
    if kind == "text":
        return normalize_name(gold_text) == normalize_name(extracted_text)
    if kind == "dni":
        return normalize_dni(gold_text) == normalize_dni(extracted_text)
    if kind == "date":
        return normalize_date(gold_text) == normalize_date(extracted_text)
    return fold_text(gold_text) == fold_text(extracted_text)
