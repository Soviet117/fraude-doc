"""Pruebas del motor de reglas semánticas (F3, E3-4)."""

from __future__ import annotations

from datetime import date

import pytest

from fraude.core.schemas import (
    ForensicsReport,
    QualityReport,
    RegistryReport,
    Severity,
    SignalScore,
)
from fraude.rules.engine import evaluate_rules

TODAY = date(2026, 10, 25)

CLEAN_DNI = {
    "numero": "12345678",
    "apellidos": "Quispe Mamani",
    "nombres": "José Luis",
    "fecha_nacimiento": "15/03/1990",
    "sexo": "M",
    "estado_civil": "Soltero(a)",
    "direccion": "Jr. Los Olivos 245, Nueva Cajamarca",
    "fecha_emision": "10/01/2023",
    "fecha_vencimiento": "10/01/2028",
}

CLEAN_CERTIFICADO = {
    "estudiante": "Quispe Mamani, José Luis",
    "dni": "12345678",
    "fecha_nacimiento": "15/03/1990",
    "institucion": "I.E. San Agustín de Nueva Cajamarca",
    "anio_egreso": "2021",
    "promedio": "16.5",
    "fecha_emision": "10/01/2023",
}

CLEAN_PARTIDA = {
    "nombre": "José Luis Quispe Mamani",
    "fecha_nacimiento": "15/03/1990",
    "lugar_nacimiento": "Nueva Cajamarca, San Martín",
    "padre": "Luis Quispe",
    "madre": "Rosa Mamani",
    "numero_acta": "123456",
    "fecha_registro": "20/03/1990",
}

CLEAN_FICHA = {
    "estudiante": "Quispe Mamani, José Luis",
    "dni": "12345678",
    "periodo": "2025-I",
    "programa": "Computación e Informática",
    "modulo": "Ofimática y herramientas digitales",
    "turno": "Mañana",
    "fecha_matricula": "15/03/2025",
}

CLEAN_ACTA = {
    "programa": "Computación e Informática",
    "modulo": "Ofimática y herramientas digitales",
    "docente": "Ing. Rosa Elena Paredes",
    "periodo": "2025-I",
    "fecha_evaluacion": "10/07/2025",
    "estudiante_0": "Quispe Mamani, José Luis",
    "dni_0": "12345678",
    "nota_0": "16",
}

CLEAN_MODULO = {
    "estudiante": "Quispe Mamani, José Luis",
    "dni": "12345678",
    "programa": "Computación e Informática",
    "modulo": "Ofimática y herramientas digitales",
    "horas": "120",
    "nota": "16",
    "fecha_emision": "10/07/2025",
}

CLEAN_TITULO = {
    "estudiante": "Quispe Mamani, José Luis",
    "dni": "12345678",
    "programa": "Computación e Informática",
    "resolucion": "RD-1234-2025-CETPRO-SA",
    "fecha_emision": "10/07/2025",
}


def _codes(fields: dict[str, str], document_type: str, **kwargs: object) -> set[str]:
    return {
        finding.code for finding in evaluate_rules(fields, document_type, today=TODAY, **kwargs)
    }


@pytest.mark.parametrize(
    ("document_type", "fields"),
    (
        ("dni", CLEAN_DNI),
        ("certificado_estudios", CLEAN_CERTIFICADO),
        ("partida_nacimiento", CLEAN_PARTIDA),
        ("ficha_matricula", CLEAN_FICHA),
        ("acta_evaluacion", CLEAN_ACTA),
        ("certificado_modulo", CLEAN_MODULO),
        ("titulo_auxiliar_tecnico", CLEAN_TITULO),
    ),
)
def test_clean_documents_have_no_findings(
    document_type: str,
    fields: dict[str, str],
) -> None:
    assert evaluate_rules(fields, document_type, today=TODAY) == []


def test_missing_fields_are_reported() -> None:
    codes = _codes({"numero": "12345678"}, "dni")

    assert "CAMPOS_FALTANTES" in codes


def test_invalid_date_is_reported() -> None:
    fields = {**CLEAN_DNI, "fecha_nacimiento": "31/02/1990"}

    assert "FECHA_INVALIDA" in _codes(fields, "dni")


def test_future_date_is_reported() -> None:
    fields = {**CLEAN_FICHA, "fecha_matricula": "10/01/2030"}

    assert "FECHA_FUTURA" in _codes(fields, "ficha_matricula")


def test_date_order_is_reported() -> None:
    fields = {**CLEAN_CERTIFICADO, "fecha_emision": "10/01/1980"}

    assert "ORDEN_FECHAS" in _codes(fields, "certificado_estudios")


def test_implausible_age_is_reported() -> None:
    fields = {**CLEAN_DNI, "fecha_nacimiento": "01/01/1850"}

    assert "EDAD_IMPLAUSIBLE" in _codes(fields, "dni")


def test_invalid_dni_is_reported() -> None:
    fields = {**CLEAN_FICHA, "dni": "1234"}

    assert "DNI_INVALIDO" in _codes(fields, "ficha_matricula")


def test_duplicate_dni_in_acta_is_reported() -> None:
    fields = {
        **CLEAN_ACTA,
        "estudiante_1": "Flores Rojas, Ana María",
        "dni_1": "12345678",
        "nota_1": "15",
    }

    assert "DNI_DUPLICADO" in _codes(fields, "acta_evaluacion")


def test_invalid_name_is_reported() -> None:
    fields = {**CLEAN_FICHA, "estudiante": "12345"}

    assert "NOMBRE_INVALIDO" in _codes(fields, "ficha_matricula")


def test_name_inconsistency_is_reported() -> None:
    fields = {
        **CLEAN_PARTIDA,
        "estudiante": "Otra Persona",
        "nombre": "José Luis Quispe Mamani",
    }

    assert "NOMBRE_INCONSISTENTE" in _codes(fields, "partida_nacimiento")


def test_unknown_program_is_reported() -> None:
    fields = {**CLEAN_FICHA, "programa": "Programa Falso"}

    assert "PROGRAMA_DESCONOCIDO" in _codes(fields, "ficha_matricula")


def test_unknown_module_is_reported() -> None:
    fields = {**CLEAN_FICHA, "modulo": "Módulo Falso"}

    assert "MODULO_DESCONOCIDO" in _codes(fields, "ficha_matricula")


def test_invalid_period_is_reported() -> None:
    fields = {**CLEAN_FICHA, "periodo": "2025-1"}

    assert "PERIODO_INVALIDO" in _codes(fields, "ficha_matricula")


def test_invalid_resolution_is_reported() -> None:
    fields = {**CLEAN_TITULO, "resolucion": "RES-123"}

    assert "RESOLUCION_INVALIDA" in _codes(fields, "titulo_auxiliar_tecnico")


def test_out_of_range_grade_is_reported() -> None:
    fields = {**CLEAN_ACTA, "nota_0": "25"}

    assert "NOTA_FUERA_DE_RANGO" in _codes(fields, "acta_evaluacion")


def test_out_of_range_average_is_reported() -> None:
    fields = {**CLEAN_CERTIFICADO, "promedio": "25"}

    assert "PROMEDIO_FUERA_DE_RANGO" in _codes(fields, "certificado_estudios")


def test_implausible_hours_are_reported() -> None:
    fields = {**CLEAN_MODULO, "horas": "10"}

    assert "HORAS_IMPLAUSIBLES" in _codes(fields, "certificado_modulo")


def test_low_quality_is_reported_as_low_severity() -> None:
    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        quality=QualityReport(ok=False, issues=["blur"]),
    )

    assert [finding.code for finding in findings] == ["CALIDAD_BAJA"]
    assert findings[0].severity is Severity.LOW


def test_low_ocr_confidence_is_reported() -> None:
    assert "OCR_BAJO" in _codes(CLEAN_FICHA, "ficha_matricula", ocr_mean_conf=0.4)
    assert "OCR_BAJO" in _codes(CLEAN_FICHA, "ficha_matricula", ocr_words=5)


def test_metadata_signal_is_reported() -> None:
    forensics = ForensicsReport(
        metadata=SignalScore(score=0.8, details={"reasons": ["software=GIMP 2.10"]})
    )

    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        forensics=forensics,
    )

    assert [finding.code for finding in findings] == ["METADATA_EDICION"]
    assert "GIMP" in findings[0].message


def test_copy_move_signal_is_reported_as_high() -> None:
    forensics = ForensicsReport(localization=SignalScore(score=0.9))

    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        forensics=forensics,
    )

    assert [finding.code for finding in findings] == ["COPY_MOVE"]
    assert findings[0].severity is Severity.HIGH


def test_registry_match_without_diffs_has_no_finding() -> None:
    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        registry=RegistryReport(matched=True, score=1.0, diffs=[]),
    )

    assert findings == []


def test_registry_mismatch_is_reported_as_high() -> None:
    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        registry=RegistryReport(matched=False, score=0.42, diffs=["sin coincidencia"]),
    )

    assert [finding.code for finding in findings] == ["REGISTRO_NO_COINCIDENTE"]
    assert findings[0].severity is Severity.HIGH


def test_registry_identity_diff_is_high() -> None:
    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        registry=RegistryReport(
            matched=True,
            score=0.8,
            diffs=["nombre: extraido='Otra Persona' registro='Quispe Mamani, José Luis'"],
        ),
    )

    assert [finding.code for finding in findings] == ["REGISTRO_DIFERENCIA"]
    assert findings[0].severity is Severity.HIGH
    assert "Otra Persona" in findings[0].message


def test_registry_academic_diff_is_medium() -> None:
    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        registry=RegistryReport(
            matched=True,
            score=0.9,
            diffs=["programa: extraido='Programa Falso' registro='Computación e Informática'"],
        ),
    )

    assert [finding.code for finding in findings] == ["REGISTRO_DIFERENCIA"]
    assert findings[0].severity is Severity.MEDIUM


def test_registry_unavailable_does_not_add_finding() -> None:
    findings = evaluate_rules(
        CLEAN_FICHA,
        "ficha_matricula",
        today=TODAY,
        registry=RegistryReport(matched=False, score=None, diffs=[]),
    )

    assert findings == []


def test_unknown_document_type_is_reported() -> None:
    findings = evaluate_rules({}, None, today=TODAY)

    assert [finding.code for finding in findings] == ["TIPO_NO_IDENTIFICADO"]


def test_findings_are_sorted_by_severity_then_code() -> None:
    findings = evaluate_rules(
        {**CLEAN_FICHA, "dni": "1234", "programa": "Programa Falso"},
        "ficha_matricula",
        today=TODAY,
        quality=QualityReport(ok=False, issues=["blur"]),
    )

    assert [finding.code for finding in findings][:2] == [
        "DNI_INVALIDO",
        "PROGRAMA_DESCONOCIDO",
    ]
    assert findings[-1].severity is Severity.LOW
