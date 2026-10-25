"""Plantillas sintéticas para los siete tipos de documento del alcance.

No reproducen documentos oficiales: son maquetas genéricas con datos
inventados, pensadas para entrenar y evaluar el pipeline. Cada campo que luego
puede manipularse se dibuja con una clave estable (``estudiante``, ``nota_0``,
``sello``...), de modo que el generador de ataques conozca su caja exacta.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics

from fraude.synth.data import Identity
from fraude.synth.layout import DocCanvas

PAGE_WIDTH, PAGE_HEIGHT = A4
NAVY = colors.HexColor("#1F3864")
LIGHT_BLUE = colors.HexColor("#D9E2F3")
GRAY = colors.HexColor("#555555")
LIGHT_GRAY = colors.HexColor("#F2F2F2")

Template = Callable[[DocCanvas, Identity, dict[str, Any]], None]


def _header(dc: DocCanvas, title: str, subtitle: str | None = None) -> None:
    dc.pdf.setFillColor(NAVY)
    dc.pdf.rect(0, PAGE_HEIGHT - 82, PAGE_WIDTH, 82, fill=1, stroke=0)
    dc.text(
        None,
        40,
        PAGE_HEIGHT - 36,
        "CETPRO SAN AGUSTÍN",
        font="Helvetica-Bold",
        size=17,
        color=colors.white,
    )
    dc.text(
        None,
        40,
        PAGE_HEIGHT - 51,
        "Centro de Educación Técnico-Productiva",
        size=8.5,
        color=colors.white,
    )
    dc.text(
        None,
        40,
        PAGE_HEIGHT - 64,
        "Nueva Cajamarca — San Martín",
        size=8.5,
        color=colors.white,
    )
    dc.text(
        None,
        PAGE_WIDTH - 40,
        PAGE_HEIGHT - 36,
        "RUC 20481234567",
        size=8,
        color=colors.white,
        align="right",
    )
    dc.text(
        None,
        PAGE_WIDTH - 40,
        PAGE_HEIGHT - 49,
        "Código modular 1234567",
        size=8,
        color=colors.white,
        align="right",
    )

    dc.text(
        None,
        PAGE_WIDTH / 2,
        PAGE_HEIGHT - 116,
        title,
        font="Helvetica-Bold",
        size=15,
        align="center",
    )
    if subtitle:
        dc.text(
            None,
            PAGE_WIDTH / 2,
            PAGE_HEIGHT - 133,
            subtitle,
            size=9,
            color=GRAY,
            align="center",
        )


def _footer(dc: DocCanvas) -> None:
    dc.line(40, 55, PAGE_WIDTH - 40, 55, color=GRAY, width=0.5)
    dc.text(
        None,
        40,
        40,
        "Documento sintético generado para investigación académica. No válido como documento oficial.",
        size=6.5,
        color=GRAY,
    )
    dc.text(
        None,
        PAGE_WIDTH - 40,
        40,
        "CETPRO San Agustín",
        size=6.5,
        color=GRAY,
        align="right",
    )


def _field(
    dc: DocCanvas,
    key: str,
    label: str,
    value: Any,
    x: float,
    y: float,
    *,
    label_width: float = 125,
    size: float = 10,
) -> None:
    dc.text(None, x, y, f"{label}:", font="Helvetica-Bold", size=size)
    dc.text(key, x + label_width, y, str(value), size=size)


def _paragraph(
    dc: DocCanvas,
    x: float,
    y: float,
    width: float,
    text: str,
    *,
    size: float = 10,
    leading: float = 15,
) -> float:
    """Escribe un párrafo con ajuste de línea y devuelve la siguiente coordenada y."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, "Helvetica", size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        dc.text(None, x, y, line, size=size)
        y -= leading
    return y


def _signature(
    dc: DocCanvas,
    key: str,
    x: float,
    y: float,
    name: str,
    role: str,
) -> None:
    dc.line(x - 90, y, x + 90, y, color=colors.black, width=0.8)
    dc.text(key, x, y - 14, name, size=9, align="center")
    dc.text(None, x, y - 26, role, size=8, color=GRAY, align="center")


def _stamp(dc: DocCanvas, key: str, x: float, y: float, radius: float = 42) -> None:
    dc.pdf.setStrokeColor(NAVY)
    dc.pdf.setLineWidth(1.2)
    dc.pdf.circle(x, y, radius, stroke=1, fill=0)
    dc.pdf.setLineWidth(0.6)
    dc.pdf.circle(x, y, radius - 4, stroke=1, fill=0)
    dc.text(None, x, y + 11, "CETPRO", font="Helvetica-Bold", size=8, color=NAVY, align="center")
    dc.text(
        None, x, y - 1, "SAN AGUSTÍN", font="Helvetica-Bold", size=7, color=NAVY, align="center"
    )
    dc.text(None, x, y - 13, "NUEVA CAJAMARCA", size=5.5, color=NAVY, align="center")
    dc.rects[key] = (x - radius, y - radius, 2 * radius, 2 * radius)


def _photo(dc: DocCanvas, key: str, x: float, y: float, width: float, height: float) -> None:
    dc.box(None, x, y, width, height, stroke=True, fill=True, color=LIGHT_GRAY)
    dc.pdf.setStrokeColor(GRAY)
    dc.pdf.setLineWidth(0.7)
    head_radius = width * 0.18
    center_x = x + width / 2
    dc.pdf.circle(x + width / 2, y + height * 0.68, head_radius, stroke=1, fill=0)
    dc.pdf.ellipse(
        center_x - width * 0.3,
        y + height * 0.12,
        center_x + width * 0.3,
        y + height * 0.5,
        stroke=1,
        fill=0,
    )
    dc.rects[key] = (x, y, width, height)


def _table_header(dc: DocCanvas, y: float) -> float:
    columns = _acta_columns()
    dc.pdf.setFillColor(LIGHT_BLUE)
    dc.pdf.rect(45, y - 6, PAGE_WIDTH - 90, 24, fill=1, stroke=0)
    dc.pdf.setFillColor(colors.black)
    for label, (x, width) in columns.items():
        dc.text(None, x + width / 2, y, label, font="Helvetica-Bold", size=8.5, align="center")
    return y - 24


def _acta_columns() -> dict[str, tuple[float, float]]:
    return {
        "N°": (50, 30),
        "Apellidos y nombres": (80, 290),
        "DNI": (370, 80),
        "Nota": (450, 95),
    }


def _acta_table(dc: DocCanvas, y: float, fields: dict[str, Any]) -> float:
    columns = _acta_columns()
    row_height = 26
    top = _table_header(dc, y)

    for row_index in range(5):
        row_y = top - row_height * row_index
        dc.line(
            45,
            row_y - row_height + 8,
            PAGE_WIDTH - 55,
            row_y - row_height + 8,
            color=GRAY,
            width=0.4,
        )
        dc.text(None, columns["N°"][0] + 15, row_y - 10, str(row_index + 1), size=9, align="center")

        name_x, _ = columns["Apellidos y nombres"]
        dc.text(
            f"estudiante_{row_index}",
            name_x + 4,
            row_y - 10,
            str(fields[f"estudiante_{row_index}"]),
            size=8.5,
        )

        dni_x, dni_width = columns["DNI"]
        dc.text(
            f"dni_{row_index}",
            dni_x + dni_width / 2,
            row_y - 10,
            str(fields[f"dni_{row_index}"]),
            size=8.5,
            align="center",
        )

        grade_x, grade_width = columns["Nota"]
        dc.text(
            f"nota_{row_index}",
            grade_x + grade_width / 2,
            row_y - 10,
            str(fields[f"nota_{row_index}"]),
            font="Helvetica-Bold",
            size=9,
            align="center",
        )

    bottom = top - row_height * 5 + 8
    dc.box(None, 45, bottom, PAGE_WIDTH - 100, top - bottom, stroke=True, fill=False)
    for x in (80, 370, 450, 545):
        dc.line(x, bottom, x, top, color=GRAY, width=0.4)
    return bottom


def render_dni(dc: DocCanvas, identity: Identity, fields: dict[str, Any]) -> None:
    _header(
        dc, "DOCUMENTO NACIONAL DE IDENTIDAD", "Representación sintética — no válida para trámites"
    )
    dc.box(
        None, 65, 410, PAGE_WIDTH - 130, 300, stroke=True, fill=False, color=NAVY, line_width=1.2
    )

    _photo(dc, "foto", 85, 555, 110, 130)
    dc.text(None, 215, 665, "N°", font="Helvetica-Bold", size=11)
    dc.text("numero", 245, 665, str(fields["numero"]), font="Helvetica-Bold", size=13)

    _field(dc, "apellidos", "Apellidos", fields["apellidos"], 215, 630, label_width=85)
    _field(dc, "nombres", "Nombres", fields["nombres"], 215, 600, label_width=85)
    _field(
        dc, "fecha_nacimiento", "Nacimiento", fields["fecha_nacimiento"], 215, 570, label_width=85
    )
    _field(dc, "sexo", "Sexo", fields["sexo"], 215, 540, label_width=85)
    _field(dc, "estado_civil", "Estado civil", fields["estado_civil"], 215, 510, label_width=85)
    _field(dc, "direccion", "Domicilio", fields["direccion"], 85, 465, label_width=85, size=9)
    _field(dc, "fecha_emision", "Emisión", fields["fecha_emision"], 85, 435, label_width=85, size=9)
    _field(
        dc,
        "fecha_vencimiento",
        "Vencimiento",
        fields["fecha_vencimiento"],
        315,
        435,
        label_width=95,
        size=9,
    )

    dc.text(
        None,
        85,
        385,
        "La autenticidad de este documento no puede ser validada oficialmente por el sistema.",
        size=8,
        color=GRAY,
    )
    _footer(dc)


def render_certificado_estudios(dc: DocCanvas, identity: Identity, fields: dict[str, Any]) -> None:
    _header(dc, "CERTIFICADO DE ESTUDIOS")
    y = _paragraph(
        dc,
        65,
        690,
        PAGE_WIDTH - 130,
        "La Dirección del Centro de Educación Técnico-Productiva San Agustín hace constar que "
        "la persona identificada en este certificado culminó sus estudios secundarios en la "
        "institución educativa que se indica.",
    )
    _field(dc, "estudiante", "Estudiante", fields["estudiante"], 65, y - 10)
    _field(dc, "dni", "DNI", fields["dni"], 65, y - 38)
    _field(dc, "fecha_nacimiento", "Fecha de nacimiento", fields["fecha_nacimiento"], 65, y - 66)
    _field(dc, "institucion", "Institución educativa", fields["institucion"], 65, y - 94)
    _field(dc, "anio_egreso", "Año de egreso", fields["anio_egreso"], 65, y - 122)
    _field(dc, "promedio", "Promedio general", fields["promedio"], 65, y - 150)

    dc.text(
        None,
        65,
        y - 190,
        "Se expide el presente certificado para los fines que estime conveniente.",
        size=9.5,
    )
    _field(dc, "fecha_emision", "Fecha de emisión", fields["fecha_emision"], 65, y - 220)

    _stamp(dc, "sello", 150, 180, 38)
    _signature(dc, "firma", 420, 190, fields["director"], "Director General")
    _footer(dc)


def render_partida_nacimiento(dc: DocCanvas, identity: Identity, fields: dict[str, Any]) -> None:
    _header(dc, "PARTIDA DE NACIMIENTO", "Copia certificada sintética")
    _field(dc, "nombre", "Nombre", fields["nombre"], 65, 680)
    _field(dc, "fecha_nacimiento", "Fecha de nacimiento", fields["fecha_nacimiento"], 65, 645)
    _field(dc, "lugar_nacimiento", "Lugar de nacimiento", fields["lugar_nacimiento"], 65, 610)
    _field(dc, "padre", "Padre", fields["padre"], 65, 575)
    _field(dc, "madre", "Madre", fields["madre"], 65, 540)
    _field(dc, "numero_acta", "Número de acta", fields["numero_acta"], 65, 505)
    _field(dc, "fecha_registro", "Fecha de registro", fields["fecha_registro"], 65, 470)

    dc.text(None, 65, 420, "Oficina Registral de Nueva Cajamarca — San Martín", size=9, color=GRAY)
    _stamp(dc, "sello", 150, 200, 40)
    _signature(dc, "firma", 420, 210, "Lic. Carmen Rosa Paredes", "Registrador Civil")
    _footer(dc)


def render_ficha_matricula(dc: DocCanvas, identity: Identity, fields: dict[str, Any]) -> None:
    _header(dc, "FICHA DE MATRÍCULA", "Periodo académico")
    _field(dc, "estudiante", "Estudiante", fields["estudiante"], 65, 680)
    _field(dc, "dni", "DNI", fields["dni"], 65, 645)
    _field(dc, "periodo", "Periodo", fields["periodo"], 65, 610)
    _field(dc, "programa", "Programa", fields["programa"], 65, 575)
    _field(dc, "modulo", "Módulo", fields["modulo"], 65, 540)
    _field(dc, "turno", "Turno", fields["turno"], 65, 505)
    _field(dc, "fecha_matricula", "Fecha de matrícula", fields["fecha_matricula"], 65, 470)
    _field(dc, "sede", "Sede", fields["sede"], 65, 435)

    dc.box(None, 65, 280, PAGE_WIDTH - 130, 100, stroke=True, fill=False, color=NAVY)
    dc.text(None, 80, 355, "Módulo matriculado", font="Helvetica-Bold", size=10)
    dc.text(None, 80, 330, str(fields["modulo"]), size=10)
    dc.text(None, 80, 305, f"Periodo: {fields['periodo']}  |  Turno: {fields['turno']}", size=9)

    _stamp(dc, "sello", 150, 180, 38)
    _signature(dc, "firma", 420, 190, "Lic. Ana María Chávez", "Secretaría Académica")
    _footer(dc)


def render_acta_evaluacion(dc: DocCanvas, identity: Identity, fields: dict[str, Any]) -> None:
    _header(dc, "ACTA DE EVALUACIÓN", "Evaluación de módulo ocupacional")
    _field(dc, "programa", "Programa", fields["programa"], 55, 690, size=9)
    _field(dc, "modulo", "Módulo", fields["modulo"], 55, 665, size=9)
    _field(dc, "docente", "Docente", fields["docente"], 55, 640, size=9)
    _field(dc, "periodo", "Periodo", fields["periodo"], 55, 615, size=9)
    _field(
        dc,
        "fecha_evaluacion",
        "Fecha de evaluación",
        fields["fecha_evaluacion"],
        330,
        615,
        size=9,
        label_width=115,
    )

    bottom = _acta_table(dc, 570, fields)
    dc.text(
        None,
        55,
        bottom - 24,
        "Escala vigesimal (0-20). La nota mínima aprobatoria es 13.",
        size=8.5,
        color=GRAY,
    )

    _signature(dc, "firma_docente", 160, 240, fields["docente"], "Docente del módulo")
    _signature(dc, "firma_director", 420, 240, fields["director"], "Director General")
    _stamp(dc, "sello", 290, 220, 34)
    _footer(dc)


def render_certificado_modulo(dc: DocCanvas, identity: Identity, fields: dict[str, Any]) -> None:
    _header(dc, "CERTIFICADO DE MÓDULO OCUPACIONAL")
    y = _paragraph(
        dc,
        65,
        690,
        PAGE_WIDTH - 130,
        "El Centro de Educación Técnico-Productiva San Agustín certifica que la persona "
        "indicada aprobó satisfactoriamente el módulo ocupacional descrito, según el registro "
        "académico institucional.",
    )
    _field(dc, "estudiante", "Estudiante", fields["estudiante"], 65, y - 10)
    _field(dc, "dni", "DNI", fields["dni"], 65, y - 40)
    _field(dc, "programa", "Programa", fields["programa"], 65, y - 70)
    _field(dc, "modulo", "Módulo", fields["modulo"], 65, y - 100)
    _field(dc, "horas", "Horas", fields["horas"], 65, y - 130)
    _field(dc, "nota", "Nota final", fields["nota"], 65, y - 160)
    _field(dc, "fecha_emision", "Fecha de emisión", fields["fecha_emision"], 65, y - 190)

    _stamp(dc, "sello", 150, 180, 38)
    _signature(dc, "firma", 420, 190, fields["director"], "Director General")
    _footer(dc)


def render_titulo_auxiliar_tecnico(
    dc: DocCanvas, identity: Identity, fields: dict[str, Any]
) -> None:
    _header(dc, "TÍTULO DE AUXILIAR TÉCNICO")
    y = _paragraph(
        dc,
        65,
        690,
        PAGE_WIDTH - 130,
        "La Dirección del Centro de Educación Técnico-Productiva San Agustín, en uso de sus "
        "atribuciones, otorga el presente título a la persona que acredita haber culminado el "
        "programa de estudios correspondiente.",
    )
    _field(dc, "estudiante", "Estudiante", fields["estudiante"], 65, y - 15)
    _field(dc, "dni", "DNI", fields["dni"], 65, y - 50)
    _field(dc, "programa", "Programa", fields["programa"], 65, y - 85)
    _field(dc, "resolucion", "Resolución", fields["resolucion"], 65, y - 120)
    _field(dc, "fecha_emision", "Fecha de emisión", fields["fecha_emision"], 65, y - 155)

    _stamp(dc, "sello", 150, 180, 38)
    _signature(dc, "firma", 420, 190, fields["director"], "Director General")
    _footer(dc)


TEMPLATES: dict[str, Template] = {
    "dni": render_dni,
    "certificado_estudios": render_certificado_estudios,
    "partida_nacimiento": render_partida_nacimiento,
    "ficha_matricula": render_ficha_matricula,
    "acta_evaluacion": render_acta_evaluacion,
    "certificado_modulo": render_certificado_modulo,
    "titulo_auxiliar_tecnico": render_titulo_auxiliar_tecnico,
}
