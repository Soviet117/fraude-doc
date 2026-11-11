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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    use_cases()
    architecture()
    er_database()
    pipeline_flow()
    print(f"Diagramas generados en {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
