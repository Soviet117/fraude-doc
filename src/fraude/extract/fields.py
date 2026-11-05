"""Extracción de campos por tipo documental (F3, E3-3).

La extracción es deliberadamente conservadora: si un campo no aparece con una
etiqueta reconocible, se omite para que las reglas lo reporten como faltante.
No usa LLM ni modelos; solo texto OCR, etiquetas y validaciones de formato.

El OCR de docTR pega palabras con frecuencia (``FECHADEI NACIMIENTO``,
``MODULOOCUPACIONAL``). Por eso las etiquetas se comparan sobre una forma
compacta (sin espacios, tildes ni puntuación), pero el valor se recorta del
texto original para no perder mayúsculas ni separadores.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from fraude.ocr.engine import OcrLine, OcrPage, OcrWord, group_lines

# (alias, tipo de valor). Los alias más largos se evalúan primero.
FieldSpec = dict[str, tuple[tuple[str, ...], str]]

FIELD_SPECS: dict[str, FieldSpec] = {
    "dni": {
        "numero": (
            (
                "N°",
                "N.°",
                "N.o",
                "Nº",
                "Número",
                "Numero",
                "DNI",
                "Documento nacional de identidad",
                "Identidad",
            ),
            "dni",
        ),
        "apellidos": (
            ("Apellidos", "Primera apellido", "Primera apelldo", "Apellido paterno"),
            "text",
        ),
        "nombres": (("Nombres", "Prenombres"), "text"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "sexo": (("Sexo",), "text"),
        "estado_civil": (("Estado civil", "Estadocivil"), "text"),
        "direccion": (("Domicilio", "Dirección", "Direccion"), "text"),
        "fecha_emision": (
            ("Fecha de emisión", "Emisión", "Emision", "Fechadeemision"),
            "date",
        ),
        "fecha_vencimiento": (("Vencimiento", "Fechadevencimiento"), "date"),
    },
    "certificado_estudios": {
        "estudiante": (("Estudiante", "Apellidos y nombres", "Apellidos"), "text"),
        "dni": (("DNI", "Documento nacional de identidad", "Identidad"), "dni"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "institucion": (("Institución educativa", "Institucion educativa"), "text"),
        "anio_egreso": (
            ("Año de egreso", "Anio de egreso", "Ano dei egreso"),
            "digits",
        ),
        "promedio": (("Promedio general", "Promedio"), "decimal"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
    },
    "partida_nacimiento": {
        "nombre": (("Nombre", "Nombres y apellidos", "Nombres"), "text"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "lugar_nacimiento": (
            ("Lugar de nacimiento", "Lugari dei nacimiento", "Lugar dei nacimiento"),
            "text",
        ),
        "padre": (("Padre",), "text"),
        "madre": (("Madre",), "text"),
        "numero_acta": (("Número de acta", "Numero de acta", "Acta"), "digits"),
        "fecha_registro": (("Fecha de registro", "Registro"), "date"),
    },
    "ficha_matricula": {
        "estudiante": (("Estudiante", "Apellidos y nombres", "Apellidos"), "text"),
        "dni": (("DNI", "Documento nacional de identidad", "Identidad"), "dni"),
        "fecha_nacimiento": (("Fecha de nacimiento", "Nacimiento"), "date"),
        "periodo": (("Periodo",), "period"),
        "programa": (("Programa", "Programa de estudios"), "text"),
        "modulo": (("Módulo", "Modulo", "Modulo ocupacional"), "text"),
        "turno": (("Turno",), "text"),
        "fecha_matricula": (
            ("Fecha de matrícula", "Fecha de matricula", "Matricula"),
            "date",
        ),
        "sede": (("Sede",), "text"),
    },
    "acta_evaluacion": {
        "programa": (("Programa", "Programa de estudios"), "text"),
        "modulo": (("Módulo", "Modulo", "Modulo ocupacional"), "text"),
        "docente": (("Docente",), "text"),
        "periodo": (("Periodo",), "period"),
        "fecha_evaluacion": (
            ("Fecha de evaluación", "Fecha de evaluacion", "Evaluación", "Evaluacion"),
            "date",
        ),
    },
    "certificado_modulo": {
        "estudiante": (("Estudiante", "Apellidos y nombres", "Apellidos"), "text"),
        "dni": (("DNI", "Documento nacional de identidad", "Identidad"), "dni"),
        "programa": (("Programa", "Programa de estudios"), "text"),
        "modulo": (("Módulo", "Modulo", "Modulo ocupacional"), "text"),
        "periodo": (("Periodo",), "period"),
        "horas": (("Horas", "Carga horaria"), "digits"),
        "nota": (("Nota final", "Nota"), "digits"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
    },
    "titulo_auxiliar_tecnico": {
        "estudiante": (("Estudiante", "Apellidos y nombres", "Apellidos"), "text"),
        "dni": (("DNI", "Documento nacional de identidad", "Identidad"), "dni"),
        "programa": (
            ("Programa", "Auxiliar técnico en", "Auxiliar tecnico en"),
            "text",
        ),
        "resolucion": (("Resolución", "Resolucion"), "text"),
        "fecha_emision": (("Fecha de emisión", "Emisión", "Emision"), "date"),
    },
}

# El título del documento manda. Se busca la primera aparición de cada patrón
# en el texto completo: las menciones del cuerpo ("acta de evaluacion",
# "certificado de estudios", "documento nacional de identidad" como etiqueta)
# aparecen después del encabezado y no deben cambiar el tipo.
DOCUMENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dni",
        (
            r"documento nacional de identidad",
            r"\bdni\b(?:\s+electronico)?",
        ),
    ),
    ("certificado_estudios", (r"certi(?:fi)?ficado de estudios",)),
    ("partida_nacimiento", (r"partida de nacimiento",)),
    ("ficha_matricula", (r"ficha de matricula",)),
    ("acta_evaluacion", (r"acta de evaluacion",)),
    ("certificado_modulo", (r"certi(?:fi)?ficado de modulo(?:\s+ocupacional)?",)),
    ("titulo_auxiliar_tecnico", (r"titulo de auxiliar tecnico",)),
)

_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_DNI_RE = re.compile(r"\b\d{8}\b")
_PERIOD_RE = re.compile(
    r"\b(20\d{2})\s*[-\u2013]\s*(I{1,2}|1{1,2}|2)\b",
    re.IGNORECASE,
)
_ROW_RE = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+(\d{8})\s+(\d{1,2})(?:\s+.*)?$")
_ROW_FALLBACK_RE = re.compile(r"(\d{8})\s+(\d{1,2})(?:\s+.*)?$")
_TRAILING_COLUMN_RE = re.compile(r"\s+\d{1,3}(?::\d{2})?\s*$")
_TRAILING_AULA_RE = re.compile(r"\s+aula\s+\d+.*$", re.IGNORECASE)
_TRAILING_REGISTRO_RE = re.compile(r"\s+registro\s+.*$", re.IGNORECASE)
_TURNO_PERIOD_RE = re.compile(
    r"^\d{4}\s*[-\u2013]\s*(?:I{1,2}|1{1,2}|2)\s+",
    re.IGNORECASE,
)

_CONNECTOR_PATTERNS = (
    re.compile(r"^(?:y|yi|syi)\s+nombres?\s+"),
    re.compile(r"^(?:yn|syin|yin)\s+nombres?\s+"),
    re.compile(r"^(?:ynombres|syinombres|yinombres)\s+"),
    re.compile(r"^(?:de|dei|dee|del)\s+estudios?\s+"),
    re.compile(r"^estudios?\s+"),
    re.compile(r"^ocupacional\s+"),
    re.compile(r"^academico\s+"),
    re.compile(r"^directoral[il]?\s+"),
    re.compile(r"^y\s+apellidos?\s+"),
)

# Etiquetas cortas que no deben confundirse con apellidos o valores. Las
# cadenas largas se buscan como subcadena para tolerar palabras pegadas por OCR
# (por ejemplo, ``MODULOMATRICULADO``); las cortas se comparan como token para
# no rechazar nombres legítimos como "Horacio".
_LABEL_TOKENS = frozenset(
    {
        "acta",
        "actas",
        "aula",
        "carga",
        "centro",
        "codigo",
        "dni",
        "escala",
        "folio",
        "firma",
        "general",
        "hora",
        "horas",
        "huella",
        "inicio",
        "madre",
        "nota",
        "notas",
        "padre",
        "presencial",
        "republica",
        "sede",
        "titular",
        "turno",
    }
)
_LABEL_STEMS = (
    "academic",
    "apellid",
    "auxiliar",
    "calificaciones",
    "certificada",
    "condicion",
    "correo",
    "credito",
    "curricular",
    "didactic",
    "direccion",
    "director",
    "directoral",
    "docente",
    "documento",
    "educacion",
    "educativa",
    "egreso",
    "emision",
    "estudios",
    "evaluacion",
    "expedido",
    "fecha",
    "horario",
    "identidad",
    "inscripcion",
    "institucion",
    "matricula",
    "matricul",
    "modalidad",
    "modulo",
    "nacimiento",
    "nombres",
    "observaciones",
    "ocupacional",
    "oficina",
    "periodo",
    "prenombres",
    "productiva",
    "programa",
    "promedio",
    "registro",
    "regional",
    "resolucion",
    "seccion",
    "secretaria",
    "telefono",
    "titulo",
    "unidad",
    "vencimiento",
    "verifique",
)

_MAX_ACTA_ROWS = 50
_LOOKAHEAD_LINES = 2


@dataclass(frozen=True)
class ExtractionResult:
    """Campos extraídos y tipo detectado (si no se proporcionó)."""

    document_type: str | None
    fields: dict[str, Any]


def detect_document_type(text: str) -> str | None:
    """Detecta el tipo documental por el título más temprano del texto."""
    normalized = _normalize(text)
    best: tuple[int, int, str] | None = None
    for document_type, patterns in DOCUMENT_PATTERNS:
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match is None:
                continue
            candidate = (match.start(), len(match.group(0)), document_type)
            if (
                best is None
                or candidate[0] < best[0]
                or (candidate[0] == best[0] and candidate[1] > best[1])
            ):
                best = candidate
    return best[2] if best is not None else None


def extract_fields(page: OcrPage, document_type: str) -> dict[str, Any]:
    """Extrae campos de las líneas OCR según el tipo documental."""
    lines = [line.text for line in group_lines(page.words)]
    fields = extract_from_lines(lines, document_type)
    if document_type == "ficha_matricula":
        # Programa y módulo pueden compartir una fila de dos columnas. Si el
        # extractor por etiquetas devolvió la misma fila para ambos, no se
        # pueden separar sin geometría: es preferible omitirlos a duplicarlos.
        if fields.get("programa") and fields.get("programa") == fields.get("modulo"):
            fields.pop("programa", None)
            fields.pop("modulo", None)
        fields.update(_extract_ficha_columns(page))
    return fields


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
    if document_type == "dni":
        fields.update(_extract_dni_dates(clean_lines))
    if document_type == "ficha_matricula" and "turno" in fields:
        fields["turno"] = _TURNO_PERIOD_RE.sub("", str(fields["turno"])).strip()
    if document_type == "titulo_auxiliar_tecnico":
        student = _extract_titulo_student(clean_lines)
        if student is not None:
            fields.setdefault("estudiante", student)
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
    ordered = sorted(aliases, key=lambda alias: len(_compact(alias)[0]), reverse=True)
    # Primero se buscan valores en la misma línea; solo después se mira la
    # línea siguiente. Así una etiqueta específica ("Nota final") gana a una
    # etiqueta genérica ("Nota") que apunta a una fila de tabla.
    for allow_lookahead in (False, True):
        for alias in ordered:
            alias_compact = _compact(alias)[0]
            if len(alias_compact) < 2:
                continue
            for index, line in enumerate(lines):
                line_compact, index_map = _compact(line)
                if not line_compact:
                    continue
                position = line_compact.find(alias_compact)
                if position < 0:
                    continue
                # En campos de texto la etiqueta debe abrir la línea; si no,
                # una mención en prosa ("...la institucion educativa que se
                # indica") se confundiría con el campo real.
                if kind == "text" and position > 0:
                    continue
                end = index_map[position + len(alias_compact) - 1] + 1
                raw = line[end:]
                if kind == "text" and raw.lstrip(" \t").startswith(("/", "-")):
                    value = None
                else:
                    value = _value_from_remainder(_clean_value(raw), kind)
                if value is not None:
                    return value
                if not allow_lookahead:
                    continue
                lookahead_limit = 1 if kind == "text" else _LOOKAHEAD_LINES
                for lookahead in range(1, lookahead_limit + 1):
                    if index + lookahead >= len(lines):
                        break
                    candidate = _clean_value(lines[index + lookahead])
                    value = _value_from_remainder(candidate, kind)
                    if value is not None:
                        return value
    return None


def _value_from_remainder(value: str, kind: str) -> Any | None:
    if not value:
        return None
    if kind == "text":
        if _is_label_like(value):
            return None
        return value
    return _parse_value(value, kind)


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
    if kind == "period":
        match = _PERIOD_RE.search(value)
        if match is None:
            return None
        year, suffix = match.group(1), match.group(2).upper()
        if suffix in {"1", "I"}:
            return f"{year}-I"
        if suffix in {"2", "11", "II"}:
            return f"{year}-II"
        return None
    if kind == "digits":
        match = re.search(r"\d+", value)
        return match.group(0) if match else None
    if kind == "decimal":
        match = re.search(r"\d{1,2}[.,]\d", value)
        return match.group(0).replace(",", ".") if match else None
    return value or None


def _clean_value(value: str) -> str:
    value = value.strip(" \t:;.,-|/")
    for _ in range(3):
        folded = _fold(value)
        changed = False
        for pattern in _CONNECTOR_PATTERNS:
            match = pattern.match(folded)
            if match is None:
                continue
            value = value[match.end() :].lstrip(" \t:;.,-|/")
            changed = True
            break
        if not changed:
            break
    value = _TRAILING_AULA_RE.sub("", value)
    value = _TRAILING_REGISTRO_RE.sub("", value)
    value = _TRAILING_COLUMN_RE.sub("", value)
    return value.strip(" \t:;.,-|/")


def _is_label_like(value: str) -> bool:
    folded = _fold(value).strip()
    if not folded or folded[0] in "/-|":
        return True
    tokens = re.findall(r"[a-z0-9]+", folded)
    if any(token in _LABEL_TOKENS for token in tokens):
        return True
    return any(stem in token for token in tokens for stem in _LABEL_STEMS)


def _extract_dni_dates(lines: list[str]) -> dict[str, Any]:
    """Asigna emisión y vencimiento cuando comparten una línea con dos fechas."""
    for index, line in enumerate(lines):
        folded = _fold(line)
        if "emision" not in folded or "vencimiento" not in folded:
            continue
        for candidate in lines[index + 1 : index + 1 + _LOOKAHEAD_LINES]:
            dates = _DATE_RE.findall(candidate)
            if len(dates) >= 2:
                return {"fecha_emision": dates[0], "fecha_vencimiento": dates[1]}
    return {}


def _extract_titulo_student(lines: list[str]) -> str | None:
    """En el título el nombre va suelto justo antes de la línea del DNI."""
    for index, line in enumerate(lines):
        if re.match(r"(?i)^dni\s+\d{8}$", line.strip()) is None:
            continue
        for candidate in reversed(lines[:index]):
            value = candidate.strip()
            if not value or _is_label_like(value):
                continue
            if len(value.split()) >= 2 and re.search(r"[^\W\d_]", value):
                return value
        return None
    return None


def _extract_ficha_columns(page: OcrPage) -> dict[str, Any]:
    """Separa las columnas programa/módulo, periodo/turno y sección/fecha.

    En la ficha, docTR entrega la fila de valores como una sola línea
    (``Cocina Seguridad e higiene alimentaria``). La posición x del encabezado
    derecho permite cortar la fila sin depender de un catálogo de programas.
    """
    lines = group_lines(page.words)
    fields: dict[str, Any] = {}
    pairs = (
        ("programa", "modulo", "programa", "modulo", "text", "text"),
        ("periodo", "turno", "periodo", "turno", "period", "text"),
        ("seccion", "fecha", "seccion", "fecha_matricula", "text", "date"),
    )
    for left_alias, right_alias, left_field, right_field, left_kind, right_kind in pairs:
        for index, line in enumerate(lines[:-1]):
            folded = _fold(line.text)
            if left_alias not in folded or right_alias not in folded:
                continue
            left_word = _find_word(line.words, left_alias)
            right_word = _find_word(line.words, right_alias)
            if left_word is None or right_word is None:
                continue
            if right_word.bbox[0] <= left_word.bbox[0]:
                continue
            value_line = _next_nonempty_line(lines, index + 1)
            if value_line is None:
                continue
            split_x = right_word.bbox[0] - 0.02 * page.width
            left_words = [
                word for word in value_line.words if word.bbox[0] + word.bbox[2] / 2 < split_x
            ]
            right_words = [
                word for word in value_line.words if word.bbox[0] + word.bbox[2] / 2 >= split_x
            ]
            left_value = _value_from_remainder(
                _clean_value(" ".join(word.text for word in left_words)),
                left_kind,
            )
            right_value = _value_from_remainder(
                _clean_value(" ".join(word.text for word in right_words)),
                right_kind,
            )
            if left_value is not None:
                fields[left_field] = left_value
            if right_value is not None:
                fields[right_field] = right_value
            break
    return fields


def _find_word(words: tuple[OcrWord, ...], alias: str) -> OcrWord | None:
    alias_compact = _compact(alias)[0]
    for word in words:
        if alias_compact in _compact(word.text)[0]:
            return word
    return None


def _next_nonempty_line(lines: list[OcrLine], start: int) -> OcrLine | None:
    for line in lines[start:]:
        if line.words:
            return line
    return None


def _extract_acta_rows(lines: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    next_index = 0
    for line in lines:
        match = _ROW_RE.match(line)
        if match:
            row = int(match.group(1)) - 1
            name, dni, grade = match.group(2), match.group(3), match.group(4)
        else:
            fallback = _ROW_FALLBACK_RE.search(line)
            if not fallback:
                continue
            row = next_index
            name = line[: fallback.start(1)].strip(" -|")
            dni, grade = fallback.group(1), fallback.group(2)
        if not 0 <= row < _MAX_ACTA_ROWS or not name:
            continue
        rows[f"estudiante_{row}"] = name.strip(" -|")
        rows[f"dni_{row}"] = dni
        rows[f"nota_{row}"] = grade
        next_index = max(next_index, row + 1)
    return rows


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def _compact(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    for index, original in enumerate(value):
        decomposed = unicodedata.normalize("NFKD", original)
        base = "".join(char for char in decomposed if not unicodedata.combining(char)).lower()
        for char in base:
            if char.isalnum():
                chars.append(char)
                index_map.append(index)
    return "".join(chars), index_map


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.lower().split())
