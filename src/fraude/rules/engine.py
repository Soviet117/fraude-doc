"""Motor de reglas semánticas y de señales (F3, E3-4).

Las reglas no deciden: generan hallazgos explicables que la fusión y la revisión
humana ponderan. Cada código tiene severidad y mensaje en español.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from fraude.core.schemas import (
    ForensicsReport,
    QualityReport,
    RegistryReport,
    RuleFinding,
    Severity,
)
from fraude.synth.data import PROGRAMS

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "dni": ("numero", "apellidos", "nombres", "fecha_nacimiento"),
    "certificado_estudios": (
        "estudiante",
        "dni",
        "fecha_nacimiento",
        "institucion",
        "anio_egreso",
        "promedio",
        "fecha_emision",
    ),
    "partida_nacimiento": (
        "nombre",
        "fecha_nacimiento",
        "lugar_nacimiento",
        "padre",
        "madre",
        "numero_acta",
        "fecha_registro",
    ),
    "ficha_matricula": (
        "estudiante",
        "dni",
        "periodo",
        "programa",
        "modulo",
        "turno",
        "fecha_matricula",
    ),
    "acta_evaluacion": (
        "programa",
        "modulo",
        "docente",
        "periodo",
        "fecha_evaluacion",
        "estudiante_0",
        "dni_0",
        "nota_0",
    ),
    "certificado_modulo": (
        "estudiante",
        "dni",
        "programa",
        "modulo",
        "horas",
        "nota",
        "fecha_emision",
    ),
    "titulo_auxiliar_tecnico": (
        "estudiante",
        "dni",
        "programa",
        "resolucion",
        "fecha_emision",
    ),
}

KNOWN_PROGRAMS = frozenset(PROGRAMS)
KNOWN_MODULES = frozenset(module for modules in PROGRAMS.values() for module in modules)
_PERIOD_RE = re.compile(r"^\d{4}-(I|II)$")
_RESOLUTION_RE = re.compile(r"^RD-\d{3,5}-\d{4}-[A-Z0-9-]+$")
_DNI_RE = re.compile(r"^\d{8}$")
_DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")


def evaluate_rules(
    fields: dict[str, Any],
    document_type: str | None,
    *,
    today: date | None = None,
    quality: QualityReport | None = None,
    forensics: ForensicsReport | None = None,
    registry: RegistryReport | None = None,
    ocr_mean_conf: float | None = None,
    ocr_words: int = 0,
) -> list[RuleFinding]:
    """Evalúa reglas y devuelve hallazgos ordenados por severidad."""
    findings: list[RuleFinding] = []
    current_day = today or date.today()
    values = {key: str(value) for key, value in fields.items() if value is not None}

    if document_type is None:
        _add(
            findings,
            "TIPO_NO_IDENTIFICADO",
            Severity.MEDIUM,
            "No se pudo identificar el tipo de documento.",
        )
    else:
        _check_required(values, document_type, findings)
        _check_dates(values, current_day, findings)
        _check_names(values, findings)
        _check_dni(values, findings)
        _check_catalog(values, findings)

    _check_quality(quality, findings)
    _check_ocr(ocr_mean_conf, ocr_words, findings)
    _check_forensics(forensics, findings)
    _check_registry(registry, findings)
    return _sort_findings(findings)


def _check_required(
    fields: dict[str, str],
    document_type: str,
    findings: list[RuleFinding],
) -> None:
    required = REQUIRED_FIELDS.get(document_type, ())
    missing = [field for field in required if not fields.get(field)]
    if missing:
        _add(
            findings,
            "CAMPOS_FALTANTES",
            Severity.HIGH
            if document_type in {"dni", "titulo_auxiliar_tecnico"}
            else Severity.MEDIUM,
            "Faltan campos obligatorios: " + ", ".join(missing),
        )


def _check_dates(fields: dict[str, str], today: date, findings: list[RuleFinding]) -> None:
    parsed: dict[str, date] = {}
    invalid: list[str] = []
    for key, value in fields.items():
        if not key.startswith("fecha_"):
            continue
        parsed_value = _parse_date(value)
        if parsed_value is None:
            invalid.append(key)
        else:
            parsed[key] = parsed_value
    if invalid:
        _add(
            findings,
            "FECHA_INVALIDA",
            Severity.HIGH,
            "Fechas con formato inválido: " + ", ".join(sorted(invalid)),
        )

    for key, value in parsed.items():
        if key == "fecha_vencimiento":
            continue
        if value > today:
            _add(
                findings,
                "FECHA_FUTURA",
                Severity.HIGH,
                f"La fecha de {key} está en el futuro ({value:%d/%m/%Y}).",
            )
    birth = parsed.get("fecha_nacimiento")
    if birth is not None:
        if birth > today:
            _add(
                findings,
                "FECHA_FUTURA",
                Severity.HIGH,
                "La fecha de nacimiento está en el futuro.",
            )
        age = (today - birth).days / 365.25
        if age < 0 or age > 100:
            _add(
                findings,
                "EDAD_IMPLAUSIBLE",
                Severity.MEDIUM,
                f"La edad calculada no es plausible ({age:.1f} años).",
            )
        for key, value in parsed.items():
            if key in {"fecha_nacimiento", "fecha_vencimiento"}:
                continue
            if value < birth:
                _add(
                    findings,
                    "ORDEN_FECHAS",
                    Severity.HIGH,
                    f"{key} es anterior a la fecha de nacimiento.",
                )

    emission = parsed.get("fecha_emision")
    expiration = parsed.get("fecha_vencimiento")
    if emission is not None and expiration is not None and expiration <= emission:
        _add(
            findings,
            "ORDEN_FECHAS",
            Severity.HIGH,
            "La fecha de vencimiento no es posterior a la de emisión.",
        )


def _check_names(fields: dict[str, str], findings: list[RuleFinding]) -> None:
    for key, value in fields.items():
        if "nombre" not in key and key not in {"estudiante", "apellidos"}:
            continue
        if not key.startswith(("nombre", "estudiante")) and key != "apellidos":
            continue
        cleaned = value.strip()
        if len(cleaned) < 3 or any(char.isdigit() for char in cleaned):
            _add(
                findings,
                "NOMBRE_INVALIDO",
                Severity.MEDIUM,
                f"El campo {key} no parece un nombre válido.",
            )

    name = fields.get("nombre")
    student = fields.get("estudiante")
    if name and student and _normalize_name(name) != _normalize_name(student):
        _add(
            findings,
            "NOMBRE_INCONSISTENTE",
            Severity.MEDIUM,
            "El nombre del titular no coincide entre campos.",
        )


def _check_dni(fields: dict[str, str], findings: list[RuleFinding]) -> None:
    dni_values: list[tuple[str, str]] = []
    for key, value in fields.items():
        if key == "dni" or key.startswith("dni_"):
            dni_values.append((key, value))
            if not _DNI_RE.match(value):
                _add(
                    findings,
                    "DNI_INVALIDO",
                    Severity.HIGH,
                    f"El {key} no tiene 8 dígitos.",
                )
    duplicates = _duplicates(value for _, value in dni_values)
    if duplicates:
        _add(
            findings,
            "DNI_DUPLICADO",
            Severity.HIGH,
            "Hay DNI repetidos en el documento: " + ", ".join(sorted(duplicates)),
        )


def _check_catalog(fields: dict[str, str], findings: list[RuleFinding]) -> None:
    program = fields.get("programa")
    if program is not None and program not in KNOWN_PROGRAMS:
        _add(
            findings,
            "PROGRAMA_DESCONOCIDO",
            Severity.MEDIUM,
            f"El programa '{program}' no está en el catálogo institucional.",
        )
    module = fields.get("modulo")
    if module is not None and module not in KNOWN_MODULES:
        _add(
            findings,
            "MODULO_DESCONOCIDO",
            Severity.MEDIUM,
            f"El módulo '{module}' no está en el catálogo institucional.",
        )
    period = fields.get("periodo")
    if period is not None and not _PERIOD_RE.match(period):
        _add(
            findings,
            "PERIODO_INVALIDO",
            Severity.MEDIUM,
            f"El periodo '{period}' no tiene el formato AAAA-I/II.",
        )
    resolution = fields.get("resolucion")
    if resolution is not None and not _RESOLUTION_RE.match(resolution):
        _add(
            findings,
            "RESOLUCION_INVALIDA",
            Severity.MEDIUM,
            f"La resolución '{resolution}' no tiene el formato esperado.",
        )

    _check_number(fields, "nota", 0.0, 20.0, "NOTA_FUERA_DE_RANGO", "nota", findings)
    for key, value in fields.items():
        if key.startswith("nota_"):
            _check_number(
                {key: value},
                key,
                0.0,
                20.0,
                "NOTA_FUERA_DE_RANGO",
                key,
                findings,
            )
    _check_number(
        fields,
        "promedio",
        0.0,
        20.0,
        "PROMEDIO_FUERA_DE_RANGO",
        "promedio",
        findings,
    )
    _check_number(
        fields,
        "horas",
        40.0,
        1000.0,
        "HORAS_IMPLAUSIBLES",
        "horas",
        findings,
    )


def _check_number(
    fields: dict[str, str],
    key: str,
    lower: float,
    upper: float,
    code: str,
    label: str,
    findings: list[RuleFinding],
) -> None:
    if key not in fields:
        return
    try:
        number = float(fields[key].replace(",", "."))
    except ValueError:
        _add(findings, code, Severity.HIGH, f"El campo {label} no es numérico.")
        return
    if not lower <= number <= upper:
        _add(
            findings,
            code,
            Severity.HIGH,
            f"El campo {label} está fuera del rango {lower:g}-{upper:g}: {number:g}.",
        )


def _check_quality(quality: QualityReport | None, findings: list[RuleFinding]) -> None:
    if quality is None or quality.ok:
        return
    issues = ", ".join(quality.issues) or "sin detalle"
    _add(
        findings,
        "CALIDAD_BAJA",
        Severity.LOW,
        f"La imagen no es apta para análisis ({issues}); se recomienda recapturar.",
    )


def _check_ocr(
    mean_conf: float | None,
    words: int,
    findings: list[RuleFinding],
) -> None:
    if mean_conf is not None and mean_conf < 0.60:
        _add(
            findings,
            "OCR_BAJO",
            Severity.MEDIUM,
            f"La confianza media del OCR es baja ({mean_conf:.2f}).",
        )
    elif words and words < 10:
        _add(
            findings,
            "OCR_BAJO",
            Severity.MEDIUM,
            f"El OCR reconoció muy pocas palabras ({words}).",
        )


def _check_forensics(
    forensics: ForensicsReport | None,
    findings: list[RuleFinding],
) -> None:
    if forensics is None:
        return
    if forensics.metadata.score is not None and forensics.metadata.score >= 0.5:
        reasons = forensics.metadata.details.get("reasons", [])
        detail = ", ".join(str(reason) for reason in reasons) or "metadatos sospechosos"
        _add(
            findings,
            "METADATA_EDICION",
            Severity.MEDIUM,
            f"Los metadatos sugieren edición ({detail}).",
        )
    if forensics.localization.score is not None and forensics.localization.score >= 0.5:
        _add(
            findings,
            "COPY_MOVE",
            Severity.HIGH,
            "Se detectaron regiones duplicadas compatibles con copy-move.",
        )


def _check_registry(
    registry: RegistryReport | None,
    findings: list[RuleFinding],
) -> None:
    if registry is None or registry.score is None:
        return
    if not registry.matched:
        _add(
            findings,
            "REGISTRO_NO_COINCIDENTE",
            Severity.HIGH,
            "La persona no coincide con el registro interno del CETPRO "
            f"(mejor score={registry.score:.2f}).",
        )
        return
    if not registry.diffs:
        return
    detail = "; ".join(registry.diffs[:4])
    identity_diff = any(diff.startswith(("nombre:", "dni:")) for diff in registry.diffs)
    _add(
        findings,
        "REGISTRO_DIFERENCIA",
        Severity.HIGH if identity_diff else Severity.MEDIUM,
        f"El documento difiere del registro interno: {detail}.",
    )


def _parse_date(value: str) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().replace(",", " ").split())


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def _add(
    findings: list[RuleFinding],
    code: str,
    severity: Severity,
    message: str,
) -> None:
    findings.append(RuleFinding(code=code, severity=severity, message=message))


def _sort_findings(findings: list[RuleFinding]) -> list[RuleFinding]:
    order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
    return sorted(findings, key=lambda finding: (order[finding.severity], finding.code))
