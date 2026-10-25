"""Extracción de campos por tipo documental (F3, E3-3).

La extracción es deliberadamente conservadora: si un campo no aparece con una
etiqueta reconocible, se omite para que las reglas lo reporten como faltante.
No usa LLM ni modelos; solo texto OCR, etiquetas y validaciones de formato.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from fraude.ocr.engine import OcrPage, group_lines

# (alias, tipo de valor). Los alias más largos se evalúan primero.
FieldSpec = dict[str, tuple[tuple[str, ...], str]]

FIELD_SPECS: dict[str, FieldSpec] = {
    "dni": {
        "numero": (("N°", "Número", "Numero", "DNI"), "dni"),
        "apellidos": (("Apellidos",), "text"),
        "nombres": (("Nombres",), "text"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "sexo": (("Sexo",), "text"),
        "estado_civil": (("Estado civil",), "text"),
        "direccion": (("Domicilio", "Dirección", "Direccion"), "text"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
        "fecha_vencimiento": (("Vencimiento",), "date"),
    },
    "certificado_estudios": {
        "estudiante": (("Estudiante",), "text"),
        "dni": (("DNI",), "dni"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "institucion": (("Institución educativa", "Institucion educativa"), "text"),
        "anio_egreso": (("Año de egreso", "Anio de egreso"), "digits"),
        "promedio": (("Promedio general", "Promedio"), "decimal"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
    },
    "partida_nacimiento": {
        "nombre": (("Nombre",), "text"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "lugar_nacimiento": (("Lugar de nacimiento",), "text"),
        "padre": (("Padre",), "text"),
        "madre": (("Madre",), "text"),
        "numero_acta": (("Número de acta", "Numero de acta", "Acta"), "digits"),
        "fecha_registro": (("Fecha de registro", "Registro"), "date"),
    },
    "ficha_matricula": {
        "estudiante": (("Estudiante",), "text"),
        "dni": (("DNI",), "dni"),
        "periodo": (("Periodo",), "text"),
        "programa": (("Programa",), "text"),
        "modulo": (("Módulo", "Modulo"), "text"),
        "turno": (("Turno",), "text"),
        "fecha_matricula": (("Fecha de matrícula", "Fecha de matricula"), "date"),
        "sede": (("Sede",), "text"),
    },
    "acta_evaluacion": {
        "programa": (("Programa",), "text"),
        "modulo": (("Módulo", "Modulo"), "text"),
        "docente": (("Docente",), "text"),
        "periodo": (("Periodo",), "text"),
        "fecha_evaluacion": (("Fecha de evaluación", "Fecha de evaluacion"), "date"),
    },
    "certificado_modulo": {
        "estudiante": (("Estudiante",), "text"),
        "dni": (("DNI",), "dni"),
        "programa": (("Programa",), "text"),
        "modulo": (("Módulo", "Modulo"), "text"),
        "horas": (("Horas",), "digits"),
        "nota": (("Nota final", "Nota"), "digits"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
    },
    "titulo_auxiliar_tecnico": {
        "estudiante": (("Estudiante",), "text"),
        "dni": (("DNI",), "dni"),
        "programa": (("Programa",), "text"),
        "resolucion": (("Resolución", "Resolucion"), "text"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
    },
}

DOCUMENT_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("documento nacional de identidad", "dni"),
    ("certificado de estudios", "certificado_estudios"),
    ("partida de nacimiento", "partida_nacimiento"),
    ("ficha de matricula", "ficha_matricula"),
    ("acta de evaluacion", "acta_evaluacion"),
    ("certificado de modulo ocupacional", "certificado_modulo"),
    ("titulo de auxiliar tecnico", "titulo_auxiliar_tecnico"),
)

_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_DNI_RE = re.compile(r"\b\d{8}\b")
_ROW_RE = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+(\d{8})\s+(\d{1,2})\s*$")


@dataclass(frozen=True)
class ExtractionResult:
    """Campos extraídos y tipo detectado (si no se proporcionó)."""

    document_type: str | None
    fields: dict[str, Any]


def detect_document_type(text: str) -> str | None:
    """Detecta el tipo documental por palabras clave del encabezado."""
    normalized = _normalize(text)
    for keyword, document_type in DOCUMENT_KEYWORDS:
        if keyword in normalized:
            return document_type
    return None


def extract_fields(page: OcrPage, document_type: str) -> dict[str, Any]:
    """Extrae campos de las líneas OCR según el tipo documental."""
    lines = [line.text for line in group_lines(page.words)]
    return extract_from_lines(lines, document_type)


def extract_from_text(text: str, document_type: str) -> dict[str, Any]:
    """Extrae campos desde texto plano (útil en tests y OCR alternativo)."""
    return extract_from_lines(text.splitlines(), document_type)


def extract_from_lines(lines: list[str], document_type: str) -> dict[str, Any]:
    """Extrae campos desde líneas ya ordenadas en orden de lectura."""
    spec = FIELD_SPECS.get(document_type)
    if spec is None:
        raise ValueError(f"Tipo documental no soportado: {document_type}")

    clean_lines = [line.strip() for line in lines if line.strip()]
    fields: dict[str, Any] = {}
    for field, (aliases, kind) in spec.items():
        value = _find_labeled_value(clean_lines, aliases, kind)
        if value is not None:
            fields[field] = value

    if document_type == "acta_evaluacion":
        fields.update(_extract_acta_rows(clean_lines))
    return fields


def extract_page(page: OcrPage) -> ExtractionResult:
    """Detecta el tipo y extrae campos en una sola llamada."""
    document_type = detect_document_type(page.text)
    if document_type is None:
        return ExtractionResult(document_type=None, fields={})
    return ExtractionResult(
        document_type=document_type,
        fields=extract_fields(page, document_type),
    )


def _find_labeled_value(
    lines: list[str],
    aliases: tuple[str, ...],
    kind: str,
) -> Any | None:
    ordered = sorted(aliases, key=len, reverse=True)
    for index, line in enumerate(lines):
        normalized = _normalize(line)
        for alias in ordered:
            alias_norm = _normalize(alias)
            if not normalized.startswith(alias_norm):
                continue
            remainder = line[len(alias) :]
            if remainder and not remainder[0].isspace() and remainder[0] not in ":-":
                continue
            value = remainder.lstrip(" :-").strip()
            if not value and index + 1 < len(lines):
                value = lines[index + 1].strip()
            parsed = _parse_value(value, kind)
            if parsed is not None:
                return parsed
    return None


def _parse_value(value: str, kind: str) -> Any | None:
    value = value.strip(" \t:;.,")
    if not value:
        return None
    if kind == "dni":
        match = _DNI_RE.search(value)
        return match.group(0) if match else None
    if kind == "date":
        match = _DATE_RE.search(value)
        return match.group(0) if match else None
    if kind == "digits":
        match = re.search(r"\d+", value)
        return match.group(0) if match else None
    if kind == "decimal":
        match = re.search(r"\d{1,2}[.,]\d", value)
        return match.group(0).replace(",", ".") if match else None
    return value or None


def _extract_acta_rows(lines: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    next_index = 0
    for line in lines:
        match = _ROW_RE.match(line)
        if match:
            row = int(match.group(1)) - 1
            name, dni, grade = match.group(2), match.group(3), match.group(4)
        else:
            fallback = re.search(r"(\d{8})\s+(\d{1,2})\s*$", line)
            if not fallback:
                continue
            row = next_index
            name = line[: fallback.start(1)].strip(" -|")
            dni, grade = fallback.group(1), fallback.group(2)
        if not 0 <= row < 5 or not name:
            continue
        rows[f"estudiante_{row}"] = name.strip(" -|")
        rows[f"dni_{row}"] = dni
        rows[f"nota_{row}"] = grade
        next_index = max(next_index, row + 1)
    return rows


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.lower().split())
