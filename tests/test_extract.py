"""Extracción de campos por tipo documental."""

from fraude.extract.fields import (
    detect_document_type,
    extract_from_lines,
    extract_from_text,
    extract_page,
)
from fraude.ocr.engine import OcrPage, OcrWord

DNI_TEXT = """DOCUMENTO NACIONAL DE IDENTIDAD
N° 12345678
Apellidos: Quispe Mamani
Nombres: José Luis
Nacimiento: 15/03/1990
Sexo: M
Estado civil: Soltero
Domicilio: Jr. Los Olivos 245
Emisión: 10/01/2023
Vencimiento: 10/01/2028"""

CERTIFICADO_TEXT = """CERTIFICADO DE ESTUDIOS
Estudiante: Quispe Mamani, José Luis
DNI: 12345678
Fecha de nacimiento: 15/03/1990
Institución educativa: I.E. San Agustín
Año de egreso: 2021
Promedio general: 16.5
Fecha de emisión: 10/01/2023"""

PARTIDA_TEXT = """PARTIDA DE NACIMIENTO
Nombre: José Luis Quispe Mamani
Fecha de nacimiento: 15/03/1990
Lugar de nacimiento: Nueva Cajamarca, San Martín
Padre: Luis Quispe
Madre: Rosa Mamani
Número de acta: 123456
Fecha de registro: 20/03/1990"""

FICHA_TEXT = """FICHA DE MATRÍCULA
Estudiante: Quispe Mamani, José Luis
DNI: 12345678
Periodo: 2025-I
Programa: Computación e Informática
Módulo: Ofimática y herramientas digitales
Turno: Mañana
Fecha de matrícula: 15/03/2025
Sede: Nueva Cajamarca"""

ACTA_TEXT = """ACTA DE EVALUACIÓN
Programa: Computación e Informática
Módulo: Ofimática y herramientas digitales
Docente: Ing. Rosa Elena Paredes
Periodo: 2025-I
Fecha de evaluación: 10/07/2025
1 Quispe Mamani, José Luis 12345678 16
2 Flores Rojas, Ana María 87654321 15"""

CERTIFICADO_MODULO_TEXT = """CERTIFICADO DE MÓDULO OCUPACIONAL
Estudiante: Quispe Mamani, José Luis
DNI: 12345678
Programa: Computación e Informática
Módulo: Ofimática y herramientas digitales
Horas: 120
Nota final: 16
Fecha de emisión: 10/07/2025"""

TITULO_TEXT = """TÍTULO DE AUXILIAR TÉCNICO
Estudiante: Quispe Mamani, José Luis
DNI: 12345678
Programa: Computación e Informática
Resolución: RD-1234-2025-CETPRO-SA
Fecha de emisión: 10/07/2025"""


def test_detect_document_type() -> None:
    assert detect_document_type(DNI_TEXT) == "dni"
    assert detect_document_type(CERTIFICADO_TEXT) == "certificado_estudios"
    assert detect_document_type(PARTIDA_TEXT) == "partida_nacimiento"
    assert detect_document_type(FICHA_TEXT) == "ficha_matricula"
    assert detect_document_type(ACTA_TEXT) == "acta_evaluacion"
    assert detect_document_type(CERTIFICADO_MODULO_TEXT) == "certificado_modulo"
    assert detect_document_type(TITULO_TEXT) == "titulo_auxiliar_tecnico"
    assert detect_document_type("texto sin título") is None


def test_extract_dni_fields() -> None:
    fields = extract_from_text(DNI_TEXT, "dni")

    assert fields["numero"] == "12345678"
    assert fields["apellidos"] == "Quispe Mamani"
    assert fields["nombres"] == "José Luis"
    assert fields["fecha_nacimiento"] == "15/03/1990"
    assert fields["sexo"] == "M"
    assert fields["fecha_vencimiento"] == "10/01/2028"


def test_extract_certificado_fields() -> None:
    fields = extract_from_text(CERTIFICADO_TEXT, "certificado_estudios")

    assert fields["estudiante"] == "Quispe Mamani, José Luis"
    assert fields["dni"] == "12345678"
    assert fields["anio_egreso"] == "2021"
    assert fields["promedio"] == "16.5"
    assert fields["fecha_emision"] == "10/01/2023"


def test_extract_partida_fields() -> None:
    fields = extract_from_text(PARTIDA_TEXT, "partida_nacimiento")

    assert fields["nombre"] == "José Luis Quispe Mamani"
    assert fields["lugar_nacimiento"] == "Nueva Cajamarca, San Martín"
    assert fields["numero_acta"] == "123456"
    assert fields["fecha_registro"] == "20/03/1990"


def test_extract_ficha_fields() -> None:
    fields = extract_from_text(FICHA_TEXT, "ficha_matricula")

    assert fields["periodo"] == "2025-I"
    assert fields["programa"] == "Computación e Informática"
    assert fields["modulo"] == "Ofimática y herramientas digitales"
    assert fields["turno"] == "Mañana"
    assert fields["sede"] == "Nueva Cajamarca"


def test_extract_acta_fields_and_rows() -> None:
    fields = extract_from_text(ACTA_TEXT, "acta_evaluacion")

    assert fields["docente"] == "Ing. Rosa Elena Paredes"
    assert fields["fecha_evaluacion"] == "10/07/2025"
    assert fields["estudiante_0"] == "Quispe Mamani, José Luis"
    assert fields["dni_0"] == "12345678"
    assert fields["nota_0"] == "16"
    assert fields["estudiante_1"] == "Flores Rojas, Ana María"
    assert fields["nota_1"] == "15"


def test_extract_certificado_modulo_and_titulo() -> None:
    modulo = extract_from_text(CERTIFICADO_MODULO_TEXT, "certificado_modulo")
    titulo = extract_from_text(TITULO_TEXT, "titulo_auxiliar_tecnico")

    assert modulo["horas"] == "120"
    assert modulo["nota"] == "16"
    assert titulo["resolucion"] == "RD-1234-2025-CETPRO-SA"


def test_missing_fields_are_omitted() -> None:
    fields = extract_from_text("Estudiante: Ana Pérez", "ficha_matricula")

    assert fields == {"estudiante": "Ana Pérez"}


def test_extract_page_detects_type_and_words() -> None:
    words = (
        OcrWord("CERTIFICADO", (10, 10, 60, 12), 0.99),
        OcrWord("DE", (75, 10, 15, 12), 0.99),
        OcrWord("ESTUDIOS", (95, 10, 45, 12), 0.99),
        OcrWord("Estudiante:", (10, 100, 50, 12), 0.98),
        OcrWord("Ana", (70, 100, 20, 12), 0.97),
        OcrWord("Pérez", (95, 100, 30, 12), 0.97),
        OcrWord("DNI:", (10, 130, 25, 12), 0.96),
        OcrWord("12345678", (45, 130, 45, 12), 0.99),
    )
    page = OcrPage(words=words, width=300, height=400)

    result = extract_page(page)

    assert result.document_type == "certificado_estudios"
    assert result.fields["estudiante"] == "Ana Pérez"
    assert result.fields["dni"] == "12345678"


def test_unknown_document_type_is_omitted() -> None:
    page = OcrPage(
        words=(OcrWord("texto", (0, 0, 30, 10), 0.9),),
        width=100,
        height=100,
    )

    result = extract_page(page)

    assert result.document_type is None
    assert result.fields == {}


def test_unknown_document_type_raises_on_direct_extraction() -> None:
    import pytest

    with pytest.raises(ValueError):
        extract_from_lines(["texto"], "no_existe")
