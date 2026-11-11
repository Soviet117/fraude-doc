"""Informe PDF de un análisis integral (F9).

Genera un documento breve con la banda de riesgo, las evidencias, los campos
extraídos, la explicación y el mapa de localización. Está pensado para que el
evaluador pueda adjuntar el resultado a la revisión manual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RISK_LABELS = {
    "bajo": ("Riesgo bajo", colors.HexColor("#1B7F3B")),
    "medio": ("Riesgo medio", colors.HexColor("#B26A00")),
    "alto": ("Riesgo alto", colors.HexColor("#B3261E")),
    "indeterminado": ("Sin determinar", colors.HexColor("#5F6B7A")),
}
SEVERITY_LABELS = {"alta": "Alta", "media": "Media", "baja": "Baja"}


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def _rich(text: str, style: ParagraphStyle) -> Paragraph:
    """Para textos estáticos con marcas mínimas (negritas)."""
    return Paragraph(str(text), style)


def build_report(analysis: dict[str, Any], out_path: Path | str) -> Path:
    """Escribe el informe PDF y devuelve la ruta."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    observation = analysis.get("observation") or {}
    source = observation.get("source") or {}
    doctype = (observation.get("doctype") or {}).get("label") or "desconocido"
    fusor = analysis.get("fusor") or {}
    risk = str(fusor.get("risk") or "indeterminado")
    risk_label, risk_color = RISK_LABELS.get(risk, RISK_LABELS["indeterminado"])
    p_fake = fusor.get("p_fake")
    explanation = analysis.get("explanation") or {}
    fields = observation.get("fields") or {}
    raw_evidences = fusor.get("evidences") or []
    evidences = []
    seen_messages: set[str] = set()
    for evidence in raw_evidences:
        message = str(evidence.get("message", ""))
        if message in seen_messages:
            continue
        seen_messages.add(message)
        evidences.append(evidence)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleFraude",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F2740"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleFraude",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#5F6B7A"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "HeadingFraude",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F2740"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyFraude",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#243447"),
    )
    small_style = ParagraphStyle(
        "SmallFraude",
        parent=body_style,
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#5F6B7A"),
    )

    story: list[Any] = [
        Paragraph("Informe de verificación documental", title_style),
        Paragraph(
            "CETPRO San Agustín · prototipo de apoyo a la revisión humana · "
            "no emite una decisión automática",
            subtitle_style,
        ),
        Table(
            [
                [
                    _rich(
                        f"<b>Documento:</b> {escape(str(source.get('filename', '—')))}", body_style
                    ),
                    _rich(f"<b>Tipo detectado:</b> {escape(str(doctype))}", body_style),
                ],
                [
                    _rich(
                        f"<b>Huella:</b> {escape(str(observation.get('document_id', '—'))[:31])}…",
                        body_style,
                    ),
                    _rich(
                        f"<b>Fecha:</b> {escape(str(analysis.get('created_at', '—')))}",
                        body_style,
                    ),
                ],
            ],
            colWidths=[9.2 * cm, 7.2 * cm],
        ),
        Spacer(1, 10),
        Table(
            [
                [
                    Paragraph(
                        f"<font color='{risk_color.hexval()}'><b>{risk_label}</b></font>",
                        ParagraphStyle("Risk", parent=body_style, fontSize=16, leading=20),
                    ),
                    _rich(
                        "<b>Score F7:</b> "
                        + ("—" if p_fake is None else f"{float(p_fake):.3f}")
                        + " <font size='8'>("
                        + escape(str(fusor.get("calibration_method", "sin calibrar")))
                        + ")</font>",
                        body_style,
                    ),
                ]
            ],
            colWidths=[9.2 * cm, 7.2 * cm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D6DEE8")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FC")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        ),
        Paragraph("Evidencias para la revisión", heading_style),
    ]

    if evidences:
        rows = [
            [
                Paragraph("<b>Severidad</b>", small_style),
                Paragraph("<b>Hallazgo</b>", small_style),
            ]
        ]
        for evidence in evidences[:10]:
            rows.append(
                [
                    _paragraph(
                        SEVERITY_LABELS.get(
                            str(evidence.get("severity")), str(evidence.get("severity"))
                        ),
                        small_style,
                    ),
                    _paragraph(evidence.get("message", ""), body_style),
                ]
            )
        story.append(
            Table(
                rows,
                colWidths=[2.6 * cm, 13.8 * cm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F6")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6DEE8")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            )
        )
    else:
        story.append(Paragraph("No se registraron evidencias automáticas.", body_style))

    story.append(Paragraph("Explicación", heading_style))
    story.append(_paragraph(explanation.get("text") or "Sin explicación disponible.", body_style))
    story.append(
        _paragraph(
            f"Fuente de la explicación: {explanation.get('source', '—')}. "
            "El sistema prioriza revisión humana; no certifica autenticidad.",
            small_style,
        )
    )

    overlay_path = (analysis.get("f5") or {}).get("overlay_path")
    if overlay_path and Path(overlay_path).is_file():
        story.append(Paragraph("Mapa de localización F5 (experimental)", heading_style))
        story.append(
            Image(str(overlay_path), width=8.0 * cm, height=11.3 * cm, kind="proportional")
        )
        story.append(
            _paragraph(
                "El mapa F5 es una ayuda cualitativa sobre datos sintéticos; "
                "no localiza de forma fiable ediciones pequeñas de texto.",
                small_style,
            )
        )

    if fields:
        story.append(PageBreak())
        story.append(Paragraph("Campos extraídos por OCR", heading_style))
        rows = [[Paragraph("<b>Campo</b>", small_style), Paragraph("<b>Valor</b>", small_style)]]
        for key, value in list(fields.items())[:24]:
            rows.append(
                [
                    _paragraph(str(key), small_style),
                    _paragraph(str(value), body_style),
                ]
            )
        story.append(
            Table(
                rows,
                colWidths=[4.2 * cm, 12.2 * cm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F6")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6DEE8")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                ),
            )
        )

    story.append(Spacer(1, 12))
    story.append(
        _paragraph(
            "Limitaciones: el prototipo se evalúa con documentos sintéticos generados "
            "por el equipo; no hay validación con documentos reales del CETPRO ni "
            "integración con RENIEC/MINEDU. Los scores no son una probabilidad "
            "institucional de fraude. Ley N.° 29733: los datos no salen del entorno local.",
            small_style,
        )
    )
    story.append(
        _paragraph(
            "Modelos: "
            + json.dumps(analysis.get("model_versions") or {}, ensure_ascii=False)
            + " · "
            + "Tiempos (ms): "
            + json.dumps(analysis.get("timings_ms") or {}, ensure_ascii=False),
            small_style,
        )
    )

    document = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Informe de verificación documental",
        author="Prototipo CETPRO San Agustín",
    )
    document.build(story)
    return out_path
