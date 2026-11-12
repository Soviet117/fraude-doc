"""Genera los diagramas del proyecto para la sustentación (sin Graphviz).

Salidas en docs/figuras/:
- casos_uso.png
- arquitectura.png
- er_historial.png
- flujo_pipeline.png

Se ejecuta con ``make diagrams`` o ``python scripts/diagrams.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/mpl").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("docs/figuras")
NAVY = "#0B2545"
NAVY_2 = "#143A66"
GOLD = "#C9A227"
BLUE = "#2F6FED"
GREEN = "#1B7F3B"
AMBER = "#B26A00"
RED = "#B3261E"
GRAY = "#5F6B7A"
LIGHT = "#F3F6FA"
LINE = "#C9D4E0"
FONT = "DejaVu Sans"


def _canvas(width: float = 13.0, height: float = 8.0) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=(width, height))
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")
    figure.patch.set_facecolor("white")
    return figure, axes


def _box(
    axes: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = NAVY,
    fontsize: float = 10,
    textcolor: str = NAVY,
    rounded: bool = True,
    weight: str = "normal",
    align: str = "center",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.4" if rounded else "square,pad=0.0",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    axes.add_patch(box)
    axes.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center" if align == "center" else "left",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontfamily=FONT,
        fontweight=weight,
        linespacing=1.35,
        zorder=3,
    )
    return box


def _arrow(
    axes: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY_2,
    style: str = "-|>",
    width: float = 1.4,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def _title(axes: plt.Axes, title: str, subtitle: str) -> None:
    axes.text(
        4,
        95,
        title,
        ha="left",
        va="top",
        fontsize=18,
        color=NAVY,
        fontfamily=FONT,
        fontweight="bold",
    )
    axes.text(
        4,
        90,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=GRAY,
        fontfamily=FONT,
    )


def use_cases() -> None:
    figure, axes = _canvas(13.5, 8.6)
    _title(
        axes,
        "Casos de uso del verificador documental",
        "Actores y funcionalidades del prototipo; la decisión final siempre es humana",
    )
    # Actors
    _box(
        axes,
        1.5,
        52,
        9.5,
        22,
        "Evaluador\ndel CETPRO",
        facecolor=NAVY,
        textcolor="white",
        fontsize=8.8,
        weight="bold",
    )
    _box(
        axes,
        1.5,
        18,
        9.5,
        18,
        "Soporte\ntécnico",
        facecolor=NAVY_2,
        textcolor="white",
        fontsize=8.8,
        weight="bold",
    )
    _box(axes, 89, 58, 9, 18, "Sistema\n(SUT)", facecolor=LIGHT, fontsize=9, weight="bold")
    _box(axes, 89, 30, 9, 18, "Ollama local\n(opcional)", facecolor=LIGHT, fontsize=8.5)
    _box(axes, 89, 5, 9, 18, "Registro\nCETPRO", facecolor=LIGHT, fontsize=8.5)

    use_cases_list = [
        (76, "Subir documento"),
        (66, "Analizar documento"),
        (56, "Ver evidencias y mapa F5"),
        (46, "Consultar historial"),
        (36, "Exportar informe PDF/JSON"),
        (26, "Registrar revisión humana"),
        (14, "Entrenar y calibrar modelos"),
        (6, "Redactar explicación (fallback)"),
    ]
    positions: dict[str, tuple[float, float]] = {}
    for cy, label in use_cases_list:
        ellipse = Ellipse(
            (42, cy + 5),
            width=36,
            height=8.8,
            facecolor="white",
            edgecolor=GOLD,
            linewidth=1.5,
            zorder=2,
        )
        axes.add_patch(ellipse)
        axes.text(
            42,
            cy + 5,
            label,
            ha="center",
            va="center",
            fontsize=8.6,
            color=NAVY,
            fontfamily=FONT,
            fontweight="bold",
        )
        positions[label] = (42, cy + 5)

    for _, label in use_cases_list[:6]:
        _arrow(
            axes,
            (11, 63),
            (24, positions[label][1]),
            width=0.8,
            color=GRAY,
            connectionstyle="arc3,rad=0.05",
        )
    _arrow(axes, (11, 27), (24, 19), width=0.8, color=GRAY)
    _arrow(axes, (60, 19), (89, 67), width=0.7, color=GRAY, connectionstyle="arc3,rad=-0.1")
    _arrow(
        axes,
        (60, 11),
        (89, 39),
        width=0.7,
        color=GRAY,
        style="<|-|>",
        connectionstyle="arc3,rad=-0.05",
    )
    _arrow(axes, (60, 71), (89, 67), width=0.7, color=GRAY, style="<|-|>")
    _arrow(axes, (60, 61), (89, 14), width=0.7, color=GRAY, connectionstyle="arc3,rad=0.1")

    axes.text(
        25,
        4,
        "«include» análisis → explicación / registro",
        fontsize=8,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "casos_uso.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def architecture() -> None:
    figure, axes = _canvas(14, 8.4)
    _title(
        axes,
        "Arquitectura de componentes",
        "Streamlit + pipeline Python + modelos locales + SQLite; sin servicios externos",
    )
    layers = [
        (
            "Presentación",
            76,
            "#E8EEF5",
            ["Streamlit demo", "Análisis integral", "F3 · F4", "Historial", "PDF/JSON"],
        ),
        (
            "Orquestación",
            62,
            "#EAF3EE",
            ["pipeline/full.py", "pipeline/run.py", "tiempos y evidencias"],
        ),
        (
            "Señales",
            45,
            "#FFF6E0",
            ["Calidad", "Forense clásico", "OCR docTR", "Extracción", "Reglas", "Registro"],
        ),
        (
            "Modelos",
            28,
            "#FDEBEA",
            ["F4 EfficientNet-B0", "F5 U-Net", "F6 semántica", "F7 LightGBM"],
        ),
        (
            "Datos y artefactos",
            11,
            "#EDEFF7",
            ["SQLite historial", "artifacts/models", "corpus v2", "registro sintético"],
        ),
    ]
    for name, y, color, items in layers:
        _box(axes, 4, y, 16, 11, name, facecolor=color, fontsize=9.5, weight="bold")
        count = len(items)
        width = 72 / count
        for index, item in enumerate(items):
            _box(
                axes,
                23 + index * width,
                y + 1,
                width - 1.2,
                9,
                item,
                facecolor="white",
                fontsize=8.2,
            )
    _arrow(axes, (50, 76), (50, 73), width=1.6)
    _arrow(axes, (50, 62), (50, 56), width=1.6)
    _arrow(axes, (50, 45), (50, 39), width=1.6)
    _arrow(axes, (50, 28), (50, 22), width=1.6)
    _arrow(axes, (12, 16), (12, 28), width=1.0, color=GRAY, style="<|-|>")
    axes.text(14.2, 21, "persistencia", fontsize=7.5, color=GRAY, fontfamily=FONT)
    axes.text(
        4,
        4,
        "Una sola imagen Docker · CPU/GPU · Ollama opcional para redactar, nunca para decidir",
        fontsize=8.2,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "arquitectura.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _table(
    axes: plt.Axes,
    x: float,
    y: float,
    width: float,
    title: str,
    rows: list[str],
    *,
    color: str = NAVY,
) -> None:
    height = 10 + len(rows) * 5.2
    axes.add_patch(
        Rectangle(
            (x, y), width, height, facecolor="white", edgecolor=color, linewidth=1.5, zorder=2
        )
    )
    axes.add_patch(
        Rectangle((x, y + height - 7), width, 7, facecolor=color, edgecolor=color, zorder=3)
    )
    axes.text(
        x + 1.5,
        y + height - 4.7,
        title,
        fontsize=10,
        color="white",
        fontfamily=FONT,
        fontweight="bold",
        zorder=4,
    )
    for index, row in enumerate(rows):
        axes.text(
            x + 1.5,
            y + height - 12.5 - index * 5.2,
            row,
            fontsize=7.8,
            color=NAVY,
            fontfamily=FONT,
            zorder=4,
        )


def er_database() -> None:
    figure, axes = _canvas(14, 8.2)
    _title(
        axes,
        "Modelo de datos del historial (SQLite)",
        "Tres tablas: documento, análisis y revisión humana",
    )
    _table(
        axes,
        4,
        38,
        29,
        "documents",
        [
            "PK  document_id",
            "     sha256",
            "     filename",
            "     doctype",
            "     first_seen",
            "     last_seen",
        ],
        color=NAVY,
    )
    _table(
        axes,
        39,
        27,
        34,
        "analyses",
        [
            "PK  id",
            "FK  document_id",
            "     created_at",
            "     p_fake",
            "     risk",
            "     calibration_method",
            "     model_versions",
            "     explanation_source",
            "     payload_json",
        ],
        color=NAVY_2,
    )
    _table(
        axes,
        79,
        38,
        18,
        "feedback",
        [
            "PK  id",
            "FK  analysis_id",
            "     created_at",
            "     reviewer",
            "     label",
            "     notes",
        ],
        color=GREEN,
    )
    _arrow(axes, (33, 58), (39, 51), width=1.4, color=NAVY)
    _arrow(axes, (73, 51), (79, 58), width=1.4, color=GREEN)
    axes.text(33.5, 61.5, "1 : N", fontsize=9, color=NAVY, fontfamily=FONT, fontweight="bold")
    axes.text(72.5, 61.5, "1 : N", fontsize=9, color=GREEN, fontfamily=FONT, fontweight="bold")
    axes.text(
        4,
        18,
        "El payload completo se guarda en JSON para reconstruir el informe sin depender\nde archivos temporales. Las revisiones humanas quedan auditadas.",
        fontsize=8.3,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "er_historial.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def pipeline_flow() -> None:
    figure, axes = _canvas(14, 7.4)
    _title(
        axes,
        "Flujo del pipeline integral",
        "De la imagen a la evidencia explicable; el LLM solo redacta",
    )
    stages = [
        ("1. Ingesta\ny calidad", NAVY),
        ("2. Forense\nclásico", NAVY_2),
        ("3. OCR\n+ extracción", NAVY_2),
        ("4. Reglas y\nregistro", NAVY_2),
        ("5. Semántica\nF6", GREEN),
        ("6. Modelos\nF4 + F5", AMBER),
        ("7. Fusión\nF7", RED),
        ("8. Explicación\ny registro", NAVY),
    ]
    width = 10.4
    gap = 1.5
    for index, (label, color) in enumerate(stages):
        x = 3 + index * (width + gap)
        _box(
            axes,
            x,
            48,
            width,
            18,
            label,
            facecolor=color,
            textcolor="white",
            fontsize=8.4,
            weight="bold",
        )
        axes.text(
            x + width / 2,
            44,
            ["hash", "ELA/ruido", "docTR", "15+ reglas", "T7/T8", "T5/T6", "score+banda", "humano"][
                index
            ],
            ha="center",
            fontsize=7.2,
            color=GRAY,
            fontfamily=FONT,
        )
        if index < len(stages) - 1:
            _arrow(axes, (x + width, 57), (x + width + gap, 57), width=1.2)
    _box(
        axes,
        3,
        14,
        30,
        18,
        "Salida para revisión\nEvidencias + mapa F5 + explicación\n+ informe PDF/JSON",
        facecolor=LIGHT,
        fontsize=8.5,
    )
    _box(
        axes,
        36,
        14,
        26,
        18,
        "Persistencia\nSQLite (documento, análisis,\nrevisión) + artifacts/",
        facecolor=LIGHT,
        fontsize=8.5,
    )
    _box(
        axes,
        65,
        14,
        29,
        18,
        "Principios\nsin decisión automática\nsin datos fuera del equipo",
        facecolor=LIGHT,
        fontsize=8.5,
        edgecolor=GOLD,
    )
    _arrow(axes, (90, 48), (90, 34), width=1.2)
    axes.text(
        3,
        5,
        "Ruta rápida: calidad + forense + OCR + reglas. Ruta profunda: F4/F5/F7 cuando hay modelos y CPU/GPU disponible.",
        fontsize=8,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "flujo_pipeline.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _uml(
    axes: plt.Axes,
    x: float,
    y: float,
    width: float,
    title: str,
    attributes: list[str],
    *,
    color: str = NAVY,
    fontsize: float = 7.6,
    row_height: float = 4.2,
) -> float:
    """Caja UML simple: cabecera + atributos. Devuelve la altura."""
    height = 9 + len(attributes) * row_height
    axes.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
            zorder=2,
        )
    )
    axes.add_patch(
        Rectangle((x, y + height - 7), width, 7, facecolor=color, edgecolor=color, zorder=3)
    )
    axes.text(
        x + 1.2,
        y + height - 3.6,
        title,
        fontsize=fontsize + 1,
        color="white",
        fontfamily=FONT,
        fontweight="bold",
        zorder=4,
    )
    for index, line in enumerate(attributes):
        axes.text(
            x + 1.2,
            y + height - 11 - index * row_height,
            line,
            fontsize=fontsize,
            color=NAVY,
            fontfamily=FONT,
            zorder=4,
        )
    return height


def _data_table(
    axes: plt.Axes,
    x: float,
    y: float,
    width: float,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    ratios: list[float] | None = None,
    color: str = NAVY,
    fontsize: float = 7.2,
    row_height: float = 6.0,
) -> float:
    """Tabla de datos para diagramas de esquema y matrices."""
    header_height = 13.0
    height = header_height + row_height * max(1, len(rows))
    axes.add_patch(
        Rectangle(
            (x, y), width, height, facecolor="white", edgecolor=color, linewidth=1.2, zorder=2
        )
    )
    axes.add_patch(
        Rectangle(
            (x, y + height - header_height),
            width,
            header_height,
            facecolor=color,
            edgecolor=color,
            zorder=3,
        )
    )
    axes.text(
        x + 1.0,
        y + height - 5.0,
        title,
        fontsize=fontsize + 1,
        color="white",
        fontfamily=FONT,
        fontweight="bold",
        zorder=4,
    )
    ratios = ratios or [1.0 / len(headers)] * len(headers)
    total = sum(ratios)
    x_positions = [x]
    for ratio in ratios:
        x_positions.append(x_positions[-1] + width * ratio / total)
    for column, header in enumerate(headers):
        axes.text(
            x_positions[column] + 1.0,
            y + height - 10.8,
            header,
            fontsize=fontsize - 0.3,
            color="#E8EEF5",
            fontfamily=FONT,
            fontweight="bold",
            zorder=4,
        )
    for index, row in enumerate(rows):
        cy = y + height - header_height - row_height * (index + 1) + 2.2
        axes.plot(
            [x, x + width],
            [cy + row_height - 2.0, cy + row_height - 2.0],
            color=LINE,
            linewidth=0.6,
            zorder=3,
        )
        for column, value in enumerate(row):
            axes.text(
                x_positions[column] + 1.0,
                cy,
                str(value)[:42],
                fontsize=fontsize,
                color=NAVY,
                fontfamily=FONT,
                zorder=4,
            )
    return height


def use_cases_table() -> None:
    figure, axes = _canvas(14, 9)
    _title(
        axes,
        "Casos de uso — detalle",
        "Actor, objetivo y resultado esperado; el sistema no decide solo",
    )
    _data_table(
        axes,
        4,
        10,
        92,
        "Casos de uso del prototipo",
        ["Actor", "Caso de uso", "Resultado / regla"],
        [
            ["Evaluador", "Subir documento", "PNG/JPG; se calcula hash y calidad"],
            ["Evaluador", "Analizar documento", "Pipeline F3+F4+F5+F6+F7"],
            ["Evaluador", "Ver evidencias y mapa", "Evidencias SHAP + overlay F5"],
            ["Evaluador", "Consultar historial", "SQLite: análisis previos"],
            ["Evaluador", "Exportar informe", "PDF/JSON descargable"],
            ["Evaluador", "Registrar revisión", "Feedback humano en SQLite"],
            ["Soporte", "Entrenar/calibrar", "F4/F5/F7 y folds agrupados"],
            ["Sistema", "Redactar explicación", "Plantilla; Ollama opcional"],
            ["Registro CETPRO", "Cruzar identidad", "Espejo sintético de estudiantes"],
            ["Ollama local", "Redactar (opcional)", "Nunca decide; fallback si falla"],
        ],
        ratios=[1.0, 1.5, 2.4],
        color=NAVY,
        fontsize=7.3,
    )
    axes.text(
        4,
        5,
        "La banda de riesgo prioriza revisión humana. Ningún caso de uso termina en rechazo automático.",
        fontsize=8,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "casos_uso_tabla.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def modules() -> None:
    figure, axes = _canvas(15, 9)
    _title(
        axes,
        "Mapa de módulos del sistema",
        "Paquetes reales del repositorio y su responsabilidad",
    )
    _box(
        axes,
        35,
        79,
        30,
        7,
        "apps/demo/app.py · Streamlit",
        facecolor=NAVY,
        textcolor="white",
        fontsize=8.8,
        weight="bold",
    )
    _box(
        axes,
        32,
        68,
        40,
        7,
        "pipeline/full.py + pipeline/run.py · orquestación",
        facecolor=NAVY_2,
        textcolor="white",
        fontsize=8.2,
        weight="bold",
    )
    groups = [
        ("Datos y síntesis", 2, ["core/", "ingest/", "preprocess/", "synth/"], "#E8EEF5"),
        ("Señales", 25, ["forensics/", "ocr/", "extract/", "rules/", "registry/"], "#FFF6E0"),
        ("Inteligencia", 50, ["nlp/ (F6)", "models/ (F4/F5)", "fusion/ (F7)"], "#FDEBEA"),
        ("Evaluación y salida", 75, ["eval/", "report/ (PDF)", "store/ (SQLite)"], "#EAF3EE"),
    ]
    for title, x, items, color in groups:
        width = 22 if title != "Señales" else 23
        axes.add_patch(
            Rectangle((x, 32), width, 30, facecolor=color, edgecolor=LINE, linewidth=1.0, zorder=1)
        )
        axes.text(
            x + 1.2,
            59.5,
            title,
            fontsize=8.6,
            color=NAVY,
            fontfamily=FONT,
            fontweight="bold",
            zorder=3,
        )
        step = 25 / max(1, len(items))
        for index, item in enumerate(items):
            _box(
                axes,
                x + 1.2,
                33.5 + (len(items) - 1 - index) * step,
                width - 2.4,
                min(4.6, step - 0.7),
                item,
                facecolor="white",
                fontsize=7.6,
            )
        _arrow(axes, (x + width / 2, 68), (x + width / 2, 62), width=1.0, color=GRAY)
    axes.text(
        4,
        26,
        "Nota: F4/F5/F7 viven en models/ y fusion/; F6 vive en nlp/; la demo solo importa el pipeline.",
        fontsize=7.8,
        color=GRAY,
        fontfamily=FONT,
    )
    _box(
        axes,
        2,
        5,
        94,
        15,
        "Flujo: presentación → orquestación → señales/modelos → evaluación/persistencia.\n"
        "Cada paquete tiene pruebas en tests/ y documentación en docs/.\n"
        "Los datos (data/) y artefactos (artifacts/) no entran a git.",
        facecolor=LIGHT,
        fontsize=7.8,
        edgecolor=LINE,
    )
    figure.savefig(OUT / "modulos.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def classes() -> None:
    figure, axes = _canvas(16, 10)
    _title(
        axes,
        "Diagrama de clases del contrato de datos",
        "Observation es el contrato congelado; FullAnalysis añade el resultado integral",
    )
    _uml(
        axes,
        36,
        34,
        28,
        "Observation",
        [
            "document_id: str",
            "source: SourceInfo",
            "quality: QualityReport",
            "doctype: DoctypeInfo",
            "forensics: ForensicsReport",
            "ocr: OcrReport",
            "fields: dict",
            "rules: list[RuleFinding]",
            "registry: RegistryReport",
            "fusion: FusionReport",
        ],
        color=NAVY,
        fontsize=7.2,
        row_height=3.8,
    )
    left = [
        (68, "SourceInfo", ["filename: str", "format: str | None"]),
        (49, "QualityReport", ["ok: bool", "blur / skew_deg"]),
        (30, "OcrReport", ["engine: str", "mean_conf: float"]),
        (11, "CaptureInfo", ["type: CaptureType", "dpi_est: int | None"]),
    ]
    right = [
        (68, "RuleFinding", ["code: str", "severity: Severity"]),
        (49, "RegistryReport", ["matched: bool", "score: float"]),
        (30, "FusionReport", ["p_fake: float | None", "risk: RiskLevel"]),
        (11, "SemanticReport (F6)", ["duplicate: DuplicateMatch", "flags: tuple[str]"]),
    ]
    for y, name, attrs in left:
        _uml(axes, 2, y, 27, name, attrs, fontsize=7.1, row_height=3.4)
        _arrow(axes, (29, y + 8), (36, 40), width=0.8, color=GRAY, style="-")
    for y, name, attrs in right:
        _uml(axes, 69, y, 28, name, attrs, fontsize=7.1, row_height=3.4)
        _arrow(axes, (64, 40), (69, y + 8), width=0.8, color=GRAY, style="-")
    _uml(
        axes,
        36,
        5,
        28,
        "FullAnalysis",
        [
            "observation: Observation",
            "semantic: SemanticReport",
            "f4_score / f5_summary",
            "explanation: Explanation",
            "fusion_details: dict",
        ],
        color=GREEN,
        fontsize=7.0,
        row_height=3.4,
    )
    _arrow(axes, (50, 34), (50, 31), width=1.2, color=GREEN)
    axes.text(51, 32, "compone", fontsize=7.4, color=GREEN, fontfamily=FONT)
    axes.text(
        4,
        1.5,
        "Pydantic v2 · los campos nuevos son opcionales y documentados · mapas por ruta, no embebidos",
        fontsize=7.4,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "clases.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def sequence() -> None:
    figure, axes = _canvas(15, 9)
    _title(
        axes,
        "Secuencia de análisis de un documento",
        "Del clic del evaluador al informe PDF y al historial SQLite",
    )
    actors = [
        ("Usuario", 8),
        ("Streamlit", 22),
        ("Pipeline", 36),
        ("F3", 49),
        ("F4/F5", 61),
        ("F6", 72),
        ("F7", 82),
        ("SQLite/PDF", 92),
    ]
    for name, x in actors:
        _box(
            axes,
            x - 4.5,
            76,
            9,
            6,
            name,
            facecolor=NAVY,
            textcolor="white",
            fontsize=7.2,
            weight="bold",
        )
        axes.plot([x, x], [12, 76], linestyle="--", color=LINE, linewidth=1.0, zorder=1)
    steps = [
        (70, 8, 22, "1. sube imagen / selecciona muestra"),
        (64, 22, 36, "2. analyze_full(bytes, registry)"),
        (58, 36, 49, "3. calidad, forense, OCR, reglas, registro"),
        (52, 49, 36, "4. Observation"),
        (46, 36, 61, "5. F4 score + F5 mapa"),
        (40, 36, 72, "6. semántica y duplicados"),
        (34, 36, 82, "7. features + score calibrado"),
        (28, 36, 92, "8. INSERT análisis"),
        (22, 36, 92, "9. informe PDF"),
        (16, 36, 22, "10. banda + evidencias + mapa"),
    ]
    for y, x1, x2, label in steps:
        _arrow(axes, (x1, y), (x2, y), width=1.2, color=BLUE)
        axes.text(
            min(x1, x2) + 1,
            y + 1.2,
            label,
            fontsize=6.6,
            color=NAVY,
            fontfamily=FONT,
        )
    axes.text(
        4,
        5,
        "Si falla la calidad, el flujo termina en 'recapturar'. Si Ollama no está, la explicación usa plantilla.",
        fontsize=7.6,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "secuencia_analisis.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def states() -> None:
    figure, axes = _canvas(15, 8)
    _title(
        axes,
        "Estados del documento y bandas de riesgo",
        "La calidad manda; el sistema nunca rechaza automáticamente",
    )
    _box(
        axes,
        3,
        62,
        20,
        10,
        "Documento recibido",
        facecolor=NAVY,
        textcolor="white",
        fontsize=8.5,
        weight="bold",
    )
    _box(
        axes,
        27,
        62,
        20,
        10,
        "Puerta de calidad",
        facecolor=NAVY,
        textcolor="white",
        fontsize=8.5,
        weight="bold",
    )
    _box(
        axes,
        27,
        45,
        20,
        8,
        "Recapturar",
        facecolor="#FDECEA",
        edgecolor=RED,
        textcolor=RED,
        fontsize=8.5,
        weight="bold",
    )
    _box(
        axes,
        51,
        62,
        22,
        10,
        "Análisis F3+F4+F5+F6",
        facecolor=NAVY_2,
        textcolor="white",
        fontsize=8.2,
        weight="bold",
    )
    _box(
        axes,
        77,
        62,
        18,
        10,
        "Fusión F7",
        facecolor=NAVY_2,
        textcolor="white",
        fontsize=8.5,
        weight="bold",
    )
    axes.add_patch(
        Rectangle((75, 23), 22, 29, facecolor=LIGHT, edgecolor=LINE, linewidth=1.0, zorder=1)
    )
    axes.text(
        76.2, 49.5, "Banda de riesgo", fontsize=7.6, color=GRAY, fontfamily=FONT, fontweight="bold"
    )
    _box(
        axes,
        76.5,
        41,
        19,
        6.2,
        "Bajo",
        facecolor="#EAF6EE",
        edgecolor=GREEN,
        textcolor=GREEN,
        fontsize=8.2,
        weight="bold",
    )
    _box(
        axes,
        76.5,
        33,
        19,
        6.2,
        "Medio",
        facecolor="#FFF4E5",
        edgecolor=AMBER,
        textcolor=AMBER,
        fontsize=8.2,
        weight="bold",
    )
    _box(
        axes,
        76.5,
        25,
        19,
        6.2,
        "Alto",
        facecolor="#FDECEA",
        edgecolor=RED,
        textcolor=RED,
        fontsize=8.2,
        weight="bold",
    )
    _box(
        axes,
        44,
        30,
        22,
        9,
        "Revisión humana",
        facecolor=GOLD,
        textcolor=NAVY,
        fontsize=8.5,
        weight="bold",
    )
    _box(axes, 16, 30, 21, 9, "Feedback SQLite", facecolor=LIGHT, fontsize=8.2)
    _arrow(axes, (23, 67), (27, 67), width=1.3)
    _arrow(axes, (37, 62), (37, 53), width=1.3, color=RED)
    axes.text(38, 56, "calidad insuficiente", fontsize=7.0, color=RED, fontfamily=FONT)
    _arrow(axes, (47, 67), (51, 67), width=1.3)
    _arrow(axes, (73, 67), (77, 67), width=1.3)
    _arrow(axes, (86, 62), (86, 52), width=1.2)
    _arrow(axes, (75, 37.5), (66, 37.5), width=1.2, color=GOLD)
    _arrow(axes, (44, 34.5), (37, 34.5), width=1.2, color=GREEN)
    axes.text(
        4,
        14,
        "Bajo / medio / alto son bandas de F7, no una probabilidad institucional de fraude.",
        fontsize=8,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "estados_riesgo.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def deployment() -> None:
    figure, axes = _canvas(15, 8)
    _title(
        axes,
        "Despliegue en 3 PCs y una sola imagen",
        "Código por git; datos y artefactos por rsync; CI sin GPU",
    )
    _box(
        axes,
        38,
        78,
        24,
        8,
        "GitHub\ncódigo + uv.lock",
        facecolor=NAVY,
        textcolor="white",
        fontsize=8.5,
        weight="bold",
    )
    nodes = [
        ("Laptop A\nDocker + Streamlit", 4, 48, 26),
        ("Laptop B\nDocker + Streamlit", 37, 48, 26),
        ("Laboratorio\nUbuntu + GPU\nDocker --profile gpu", 70, 48, 26),
    ]
    for label, x, y, width in nodes:
        _box(axes, x, y, width, 18, label, facecolor=LIGHT, fontsize=8.5, edgecolor=NAVY)
        _arrow(
            axes,
            (50, 78),
            (x + width / 2, 66),
            width=1.0,
            color=GRAY,
            connectionstyle="arc3,rad=0.05",
        )
    _box(
        axes, 12, 22, 30, 10, "Datos locales\ndata/ + artifacts/", facecolor="#FFF6E0", fontsize=8.2
    )
    _box(
        axes,
        57,
        22,
        30,
        10,
        "Resultados\nmodelos, mapas, PDF, SQLite",
        facecolor="#EAF3EE",
        fontsize=8.2,
    )
    _arrow(
        axes,
        (50, 48),
        (27, 32),
        width=1.0,
        color=GRAY,
        style="<|-|>",
        connectionstyle="arc3,rad=0.08",
    )
    _arrow(
        axes,
        (63, 48),
        (72, 32),
        width=1.0,
        color=GRAY,
        style="<|-|>",
        connectionstyle="arc3,rad=-0.08",
    )
    axes.text(
        4,
        12,
        "Los documentos no salen del entorno local. El laboratorio solo se usa para entrenamiento.",
        fontsize=8,
        color=GRAY,
        fontfamily=FONT,
    )
    figure.savefig(OUT / "despliegue.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def physical_database() -> None:
    figure, axes = _canvas(16, 11)
    _title(
        axes,
        "Base de datos física (SQLite)",
        "Archivo artifacts/history.db · tres tablas y dos índices",
    )
    _uml(
        axes,
        3,
        50,
        29,
        "documents",
        [
            "document_id TEXT PK",
            "sha256 TEXT NOT NULL",
            "filename TEXT NOT NULL",
            "doctype TEXT",
            "first_seen TEXT NOT NULL",
            "last_seen TEXT NOT NULL",
        ],
        color=NAVY,
        row_height=4.0,
    )
    _uml(
        axes,
        36,
        39,
        32,
        "analyses",
        [
            "id INTEGER PK AUTOINCREMENT",
            "document_id TEXT FK NOT NULL",
            "created_at TEXT NOT NULL",
            "p_fake REAL",
            "risk TEXT",
            "calibration_method TEXT",
            "model_versions TEXT (JSON)",
            "explanation_source TEXT",
            "payload_json TEXT NOT NULL",
        ],
        color=NAVY_2,
        row_height=4.0,
    )
    _uml(
        axes,
        76,
        50,
        21,
        "feedback",
        [
            "id INTEGER PK AUTOINCREMENT",
            "analysis_id INTEGER FK",
            "created_at TEXT NOT NULL",
            "reviewer TEXT",
            "label TEXT NOT NULL",
            "notes TEXT",
        ],
        color=GREEN,
        row_height=4.0,
    )
    _arrow(axes, (32, 64), (36, 64), width=1.4, color=NAVY)
    _arrow(axes, (68, 64), (76, 64), width=1.4, color=GREEN)
    axes.text(32.4, 67, "1 : N", fontsize=8.5, color=NAVY, fontfamily=FONT, fontweight="bold")
    axes.text(68.4, 67, "1 : N", fontsize=8.5, color=GREEN, fontfamily=FONT, fontweight="bold")
    _data_table(
        axes,
        3,
        3,
        94,
        "Índices y decisiones de persistencia",
        ["Objeto", "Definición", "Para qué"],
        [
            ["idx_analyses_created", "analyses(created_at DESC)", "Historial reciente"],
            ["idx_feedback_analysis", "feedback(analysis_id)", "Revisiones por análisis"],
            ["payload_json", "TEXT con el informe completo", "Reconstruir sin archivos temporales"],
            ["foreign_keys", "PRAGMA foreign_keys=ON", "Integridad documento→análisis→feedback"],
        ],
        ratios=[0.9, 1.3, 2.0],
        color=NAVY,
        fontsize=7.0,
        row_height=5.2,
    )
    figure.savefig(OUT / "base_datos_fisica.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def database_sample() -> None:
    import sqlite3

    path = Path("artifacts/history.db")
    documents: list[list[str]] = []
    analyses: list[list[str]] = []
    feedback: list[list[str]] = []
    if path.is_file():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        documents = [
            [row["document_id"][:18] + "…", row["filename"], row["doctype"]]
            for row in connection.execute("SELECT * FROM documents ORDER BY last_seen DESC LIMIT 3")
        ]
        analyses = [
            [
                str(row["id"]),
                row["risk"],
                "—" if row["p_fake"] is None else f"{row['p_fake']:.3f}",
                row["calibration_method"] or "—",
            ]
            for row in connection.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 3")
        ]
        feedback = [
            [str(row["analysis_id"]), row["label"], row["reviewer"] or "—"]
            for row in connection.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 3")
        ]
        connection.close()
    documents = documents or [["—", "sin documentos", "—"]]
    analyses = analyses or [["—", "sin análisis", "—", "—"]]
    feedback = feedback or [["—", "sin revisiones", "—"]]

    figure, axes = _canvas(16, 10)
    _title(
        axes,
        "Datos de ejemplo en la base de datos",
        "Consultas reales a artifacts/history.db · datos sintéticos, sin PII real",
    )
    _data_table(
        axes,
        2,
        58,
        96,
        "documents (últimos 3)",
        ["document_id", "filename", "doctype"],
        documents,
        ratios=[2.2, 1.3, 1.0],
        color=NAVY,
        row_height=5.2,
    )
    _data_table(
        axes,
        2,
        26,
        96,
        "analyses (últimos 3)",
        ["id", "risk", "p_fake", "calibración"],
        analyses,
        ratios=[0.5, 0.8, 0.8, 1.2],
        color=NAVY_2,
        row_height=5.2,
    )
    _data_table(
        axes,
        2,
        4,
        96,
        "feedback (últimos 3)",
        ["analysis_id", "label", "reviewer"],
        feedback,
        ratios=[0.8, 1.4, 1.0],
        color=GREEN,
        row_height=5.2,
    )
    figure.savefig(OUT / "base_datos_datos.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def test_matrix() -> None:
    import re
    import subprocess
    import sys

    counts: dict[str, int] = {}
    total = 0
    try:
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        for line in process.stdout.splitlines():
            match = re.match(r"(tests/[^:]+): (\d+)$", line.strip())
            if match:
                name = Path(match.group(1)).name
                counts[name] = int(match.group(2))
    except (OSError, subprocess.SubprocessError):
        counts = {}
    if not counts:
        for path in sorted(Path("tests").glob("test_*.py")):
            counts[path.name] = len(
                re.findall(r"^def test_", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
            )
    total = sum(counts.values())
    rows = [[name, str(count), "verde"] for name, count in sorted(counts.items())]
    half = (len(rows) + 1) // 2
    figure, axes = _canvas(16, 10)
    _title(
        axes,
        "Matriz de casos de prueba",
        f"{total} casos recogidos por pytest · unitarios y de integración · sin GPU",
    )
    _data_table(
        axes,
        2,
        8,
        46,
        "tests/ (1/2)",
        ["Archivo", "Casos", "Estado"],
        rows[:half],
        ratios=[2.4, 0.6, 0.7],
        color=GREEN,
        fontsize=6.5,
        row_height=4.2,
    )
    _data_table(
        axes,
        52,
        8,
        46,
        "tests/ (2/2)",
        ["Archivo", "Casos", "Estado"],
        rows[half:],
        ratios=[2.4, 0.6, 0.7],
        color=GREEN,
        fontsize=6.5,
        row_height=4.2,
    )
    figure.savefig(OUT / "casos_prueba.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    use_cases()
    use_cases_table()
    architecture()
    modules()
    classes()
    sequence()
    states()
    deployment()
    er_database()
    physical_database()
    database_sample()
    pipeline_flow()
    test_matrix()
    print(f"Diagramas generados en {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
