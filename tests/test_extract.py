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


def test_detect_document_type_prefers_header_title() -> None:
    certificado = """CERTIFICADO DE ESTUDIOS
    APELLIDOS SYI NOMBRES PAREDES MAMANI, FERNANDO JOSÉ
    DOCUMENTO NACIONAL DE IDENTIDAD 49346326
    DNI 16785432"""
    ficha = """FICHA DE MATRICULA
    APELLIDOS Y NOMBRES HUAMÂN CASTILLO, LUCIA FERNANDA
    En la Secretaría Académica se convalida con el certificado de estudios."""
    modulo = """CERTIFICADO DE MODULO OCUPACIONAL
    DOCUMENTO NACIONAL DE IDENTIDAD 77964873
    según consta en el acta de evaluacion correspondiente."""
    misspelled = """CERTIFIFICADO DE ESTUDIOS
    DOCUMENTO NACIONAL DE IDENTIDAD 45141610"""
    titulo = """REPUBLICA DEL PERU
    TITULO
    DE AUXILIAR TÉCNICO
    DNI 74609924"""

    assert detect_document_type(certificado) == "certificado_estudios"
    assert detect_document_type(ficha) == "ficha_matricula"
    assert detect_document_type(modulo) == "certificado_modulo"
    assert detect_document_type(misspelled) == "certificado_estudios"
    assert detect_document_type(titulo) == "titulo_auxiliar_tecnico"


def test_extract_dni_ocr_labels() -> None:
    text = """DOCUMENTO NACIONAL DE IDENTIDAD
    N.o 61531833
    PRIMERA APELLIDO/ SEGUNDOAPELLIDO
    SÁNCHEZ SALAZAR
    PRENOMBRES
    ÉDGAR MARTIN
    SEXO
    MASCULINO
    ESTADOCIVIL
    SOLTERO(A)
    FECHADEI NACIMIENTO
    03/06/1999
    DOMICILIO
    Av. Circunvalacion 45, Soritor
    FECHADEEMISION FECHADEV VENCIMIENTO HUELLA
    15/01/2024 15/01/2032"""

    fields = extract_from_text(text, "dni")

    assert fields["numero"] == "61531833"
    assert fields["apellidos"] == "SÁNCHEZ SALAZAR"
    assert fields["nombres"] == "ÉDGAR MARTIN"
    assert fields["fecha_nacimiento"] == "03/06/1999"
    assert fields["estado_civil"] == "SOLTERO(A)"
    assert fields["direccion"] == "Av. Circunvalacion 45, Soritor"
    assert fields["fecha_emision"] == "15/01/2024"
    assert fields["fecha_vencimiento"] == "15/01/2032"


def test_extract_certificate_ocr_labels() -> None:
    text = """CERTIFICADO DE ESTUDIOS
    APELLIDOS SYI NOMBRES PAREDES MAMANI, FERNANDO JOSÉ
    DOCUMENTO NACIONAL DE IDENTIDAD 49346326
    FECHA DE NACIMIENTO 17/01/2000
    INSTITUCION EDUCATIVA I.E. Andrés Avelino Cáceres
    ANO DEI EGRESO 2022
    PROMEDIO GENERAL: 14.6 - ORDEN DE MÉRITO: 20 de 34"""

    fields = extract_from_text(text, "certificado_estudios")

    assert fields["estudiante"] == "PAREDES MAMANI, FERNANDO JOSÉ"
    assert fields["dni"] == "49346326"
    assert fields["fecha_nacimiento"] == "17/01/2000"
    assert fields["institucion"] == "I.E. Andrés Avelino Cáceres"
    assert fields["anio_egreso"] == "2022"
    assert fields["promedio"] == "14.6"


def test_extract_acta_ocr_labels() -> None:
    text = """ACTA DE EVALUACION
    PROGRAMA DEI ESTUDIOS N.ODEI MATRICULADOS
    Administracion de Empresas 20
    MODULOOCUPACIONAL HORAI DEI INICIO
    Contabilidad basica 20:00
    DOCENTE AULA/TURNO
    Téc. Pedro Saavedra Moreno Aula 160 I Tarde
    PERIODO FECHAI DEE EVALUACION
    2025-1 18/07/2026
    1 ROJAS RUIZ, Rosa Milagros 71569494 13
    7 GUEVARA VEGA, Miguel Angel 66709866 09 DESAPR."""

    fields = extract_from_text(text, "acta_evaluacion")

    assert fields["programa"] == "Administracion de Empresas"
    assert fields["modulo"] == "Contabilidad basica"
    assert fields["docente"] == "Téc. Pedro Saavedra Moreno"
    assert fields["periodo"] == "2025-I"
    assert fields["fecha_evaluacion"] == "18/07/2026"
    assert fields["nota_0"] == "13"
    assert fields["nota_6"] == "09"


def test_extract_certificado_modulo_ocr_labels() -> None:
    text = """CERTIFICADO DE MODULO OCUPACIONAL
    APELLIDOS YI NOMBRES SANCHEZ ZAVALETA, ANA LUCIA
    DOCUMENTO NACIONALI DE IDENTIDAD 77964873
    PROGRAMA DE ESTUDIOS Mecánica Automotriz
    MODULO OCUPACIONAL Mantenimiento de motores
    PERIODO ACADÉMICO 2024-1
    CARGA HORARIA 120 horas académicas
    NOTA FINAL DEL MODULO: 16 CONDICION: APROBADO"""

    fields = extract_from_text(text, "certificado_modulo")

    assert fields["estudiante"] == "SANCHEZ ZAVALETA, ANA LUCIA"
    assert fields["dni"] == "77964873"
    assert fields["programa"] == "Mecánica Automotriz"
    assert fields["modulo"] == "Mantenimiento de motores"
    assert fields["periodo"] == "2024-I"
    assert fields["horas"] == "120"
    assert fields["nota"] == "16"


def test_extract_titulo_student_and_program() -> None:
    text = """TITULO
    DE AUXILIAR TÉCNICO
    ROSA MILAGROS BAUTISTA SANCHEZ
    DNI 74609924
    como
    AUXILIAR TÉCNICO EN
    COCINA
    RESOLUCION DIRECTORALI N.° 252-2026-DG-CETPRO-SA REGISTRO N.O T-56498-2026"""

    fields = extract_from_text(text, "titulo_auxiliar_tecnico")

    assert fields["estudiante"] == "ROSA MILAGROS BAUTISTA SANCHEZ"
    assert fields["dni"] == "74609924"
    assert fields["programa"] == "COCINA"
    assert fields["resolucion"] == "N.° 252-2026-DG-CETPRO-SA"


def test_extract_ficha_columns_from_geometry() -> None:
    def word(text: str, x: int, y: int) -> OcrWord:
        return OcrWord(text, (x, y, max(8, len(text) * 6), 12), 0.99)

    words = (
        word("FICHA", 10, 10),
        word("DE", 60, 10),
        word("MATRICULA", 90, 10),
        word("APELLIDOS", 10, 40),
        word("Y", 90, 40),
        word("NOMBRES", 110, 40),
        word("HUAMÂN", 180, 40),
        word("CASTILLO,", 250, 40),
        word("LUCIA", 330, 40),
        word("FERNANDA", 390, 40),
        word("DNI", 10, 70),
        word("48437194", 60, 70),
        word("FECHA", 10, 100),
        word("DEI", 70, 100),
        word("NACIMIENTO", 110, 100),
        word("05/09/2000", 220, 100),
        word("MODALIDADI", 10, 130),
        word("PRESENCIAL", 110, 130),
        word("Periodo", 250, 130),
        word("académico:", 320, 130),
        word("2025-1", 430, 130),
        word("PROGRAMAI", 10, 160),
        word("DEE", 100, 160),
        word("ESTUDIOS", 140, 160),
        word("MODULOMATRICULADO", 400, 160),
        word("Cocina", 10, 190),
        word("Seguridad", 400, 190),
        word("e", 500, 190),
        word("higiene", 530, 190),
        word("alimentaria", 600, 190),
    )
    page = OcrPage(words=words, width=1000, height=400)

    result = extract_page(page)

    assert result.document_type == "ficha_matricula"
    assert result.fields["estudiante"] == "HUAMÂN CASTILLO, LUCIA FERNANDA"
    assert result.fields["dni"] == "48437194"
    assert result.fields["fecha_nacimiento"] == "05/09/2000"
    assert result.fields["periodo"] == "2025-I"
    assert result.fields["programa"] == "Cocina"
    assert result.fields["modulo"] == "Seguridad e higiene alimentaria"


def test_extract_ficha_keeps_labeled_columns_when_geometry_fails() -> None:
    lines = (
        "FICHA DE MATRICULA",
        "Estudiante: Quispe Mamani, José Luis",
        "DNI: 12345678",
        "Periodo: 2025-I",
        "Programa: Computación e Informática",
        "Módulo: Ofimática y herramientas digitales",
        "Turno: Mañana",
        "Fecha de matrícula: 15/03/2025",
    )
    words = tuple(
        OcrWord(text=text, bbox=(80, 40 + index * 40, 700, 20), confidence=0.95)
        for index, text in enumerate(lines)
    )
    page = OcrPage(words=words, width=1200, height=1600)

    result = extract_page(page)

    assert result.document_type == "ficha_matricula"
    assert result.fields["programa"] == "Computación e Informática"
    assert result.fields["modulo"] == "Ofimática y herramientas digitales"


def test_extract_period_normalization() -> None:
    assert extract_from_text("Periodo: 2024-1", "ficha_matricula")["periodo"] == "2024-I"
    assert extract_from_text("Periodo: 2025-11", "ficha_matricula")["periodo"] == "2025-II"
    assert extract_from_text("Periodo: 2025-II", "ficha_matricula")["periodo"] == "2025-II"


def test_extract_long_emission_date_on_next_line() -> None:
    text = """CERTIFICADO DE ESTUDIOS
Estudiante: Quispe Mamani, José Luis
Se expide el presente certificado a solicitud del interesado para los fines que estime conveniente.
Nueva Cajamarca, 15 de febrero de 2026."""

    fields = extract_from_text(text, "certificado_estudios")

    assert fields["fecha_emision"] == "15/02/2026"


def test_extract_long_emission_date_inline() -> None:
    text = """CERTIFICADO DE MÓDULO OCUPACIONAL
Estudiante: Quispe Mamani, José Luis
Se expide el presente certificado en la ciudad de Nueva Cajamarca, 30 de agosto de 2026, para los fines que el interesado estime conveniente."""

    fields = extract_from_text(text, "certificado_modulo")

    assert fields["fecha_emision"] == "30/08/2026"


def test_extract_long_emission_date_for_titulo() -> None:
    text = """TÍTULO DE AUXILIAR TÉCNICO
Estudiante: Quispe Mamani, José Luis
EXPEDIDO EL 29 DE SETIEMBRE DE 2026"""

    fields = extract_from_text(text, "titulo_auxiliar_tecnico")

    assert fields["fecha_emision"] == "29/09/2026"


def test_invalid_long_date_is_omitted() -> None:
    text = """CERTIFICADO DE ESTUDIOS
Se expide el presente certificado a solicitud del interesado.
Nueva Cajamarca, 31 de febrero de 2026."""

    fields = extract_from_text(text, "certificado_estudios")

    assert "fecha_emision" not in fields


def test_three_digit_day_is_not_silently_corrected() -> None:
    text = """TÍTULO DE AUXILIAR TÉCNICO
EXPEDIDO EL 291 DE SETIEMBRE DE 2026"""

    fields = extract_from_text(text, "titulo_auxiliar_tecnico")

    assert "fecha_emision" not in fields
