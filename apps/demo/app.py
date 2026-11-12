"""Interfaz de producto del verificador documental — CETPRO San Agustín.

Vista de análisis integral (F3+F4+F5+F6+F7), historial SQLite y ayuda.
Incluye tema claro/oscuro, tooltips contextuales y exportación PDF/JSON.
Ninguna salida es una decisión automática: el sistema prioriza la revisión
humana y declara sus límites.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from fraude.core.schemas import Observation, Severity, SignalScore
from fraude.pipeline.full import analyze_full
from fraude.registry.build import build_registry_from_manifest
from fraude.registry.cross import RegistryEntry
from fraude.report import build_report
from fraude.rules.engine import REQUIRED_FIELDS
from fraude.store import HistoryStore

SAMPLE_ROOT = Path("data/muestras")
SAMPLE_INDEX = SAMPLE_ROOT / "_indice.csv"
SAMPLE_PEOPLE = SAMPLE_ROOT / "_personas.csv"
V2_ROOT = Path("data/synthetic/v2")
HISTORY_DB = Path("artifacts/history.db")
REPORTS_DIR = Path("artifacts/reports")
RUNS_DIR = Path("artifacts/runs")
F4_CHECKPOINT = Path("artifacts/models/f4/fold_2.pt")
F5_CHECKPOINT = Path("artifacts/models/f5/fold_0.pt")
F7_MODEL = Path("artifacts/models/f7/fusion.joblib")
DEFAULT_THRESHOLD = 0.5
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})

DOCUMENT_LABELS = {
    "dni": "DNI",
    "certificado_estudios": "Certificado de estudios",
    "partida_nacimiento": "Partida de nacimiento",
    "ficha_matricula": "Ficha de matrícula",
    "acta_evaluacion": "Acta de evaluación",
    "certificado_modulo": "Certificado de módulo",
    "titulo_auxiliar_tecnico": "Título de Auxiliar Técnico",
    "desconocido": "Tipo no identificado",
}
CAPTURE_LABELS = {
    "scan": "Escaneo",
    "photo": "Fotografía",
    "screenshot": "Captura de pantalla",
    "pdf_render": "PDF renderizado",
    "unknown": "Desconocido",
}
QUALITY_LABELS = {
    "low_resolution": "Resolución baja",
    "blur": "Desenfoque",
    "skew": "Inclinación",
    "low_contrast": "Contraste bajo",
    "overexposed": "Sobreexpuesta",
    "underexposed": "Subexpuesta",
    "unreadable_image": "Imagen ilegible",
}
SIGNAL_LABELS = {
    "metadata": "Metadatos",
    "localization": "Copy-move",
    "ela": "ELA",
    "noise": "Ruido",
    "chromatic": "Cromática",
    "jpeg": "JPEG",
}
SIGNAL_HELP = {
    "metadata": "Datos del archivo; un editor gráfico o una fecha de modificación pueden ser indicio.",
    "localization": "Busca regiones duplicadas dentro de la misma imagen (copy-move).",
    "ela": "Error Level Analysis: útil solo en JPEG; en escaneos sin pérdida no revela nada.",
    "noise": "Analiza si el ruido de la página es homogéneo o hay zonas inconsistentes.",
    "chromatic": "Busca inconsistencias de color compatibles con edición.",
    "jpeg": "Detecta doble compresión o rastros de recodificación JPEG.",
}
FIELD_LABELS = {
    "numero": "DNI",
    "apellidos": "Apellidos",
    "nombres": "Nombres",
    "fecha_nacimiento": "Fecha de nacimiento",
    "sexo": "Sexo",
    "estado_civil": "Estado civil",
    "direccion": "Dirección",
    "fecha_emision": "Fecha de emisión",
    "fecha_vencimiento": "Fecha de vencimiento",
    "estudiante": "Estudiante",
    "dni": "DNI",
    "institucion": "Institución educativa",
    "anio_egreso": "Año de egreso",
    "promedio": "Promedio",
    "nombre": "Nombre",
    "lugar_nacimiento": "Lugar de nacimiento",
    "padre": "Padre",
    "madre": "Madre",
    "numero_acta": "Número de acta",
    "fecha_registro": "Fecha de registro",
    "periodo": "Periodo",
    "programa": "Programa",
    "modulo": "Módulo",
    "turno": "Turno",
    "fecha_matricula": "Fecha de matrícula",
    "sede": "Sede",
    "docente": "Docente",
    "director": "Director",
    "nota": "Nota",
    "horas": "Horas",
    "resolucion": "Resolución",
    "foto": "Fotografía",
    "sello_copiado": "Sello copiado",
    "sello_insertado": "Sello insertado",
    "firma": "Firma",
}
CODE_LABELS = {
    "F6_REUTILIZACION": "Reutilización de documento",
    "F6_SIMILITUD": "Similitud con documento registrado",
    "F5_LOCALIZACION": "Localización de la alteración",
    "REGISTRO": "Cruce con el registro",
    "REGISTRO_DIFERENCIA": "Diferencia con el registro",
    "REGLAS": "Reglas de consistencia",
    "F4_SCORE": "Señal visual de página",
    "F4_INDICIO_VISUAL": "Señal visual de página",
    "OCR_LOG_WORDS": "Volumen de texto reconocido",
    "OCR_MEAN_CONF": "Confianza del OCR",
    "FORENSICS_ELA": "Análisis ELA",
    "FORENSICS_NOISE": "Residuo de ruido",
    "FORENSICS_CHROMATIC": "Inconsistencia cromática",
    "FORENSICS_JPEG": "Doble compresión JPEG",
    "FORENSICS_METADATA": "Metadatos de edición",
    "FORENSICS_LOCALIZATION": "Copy-move",
    "SEMANTICA": "Consistencia semántica",
    "RECAPTURAR": "Calidad insuficiente",
    "F7_SIN_INDICIOS": "Sin indicios suficientes",
}
PHASES = [
    ("F0", "Base operativa y contrato de datos", "Cerrada en local"),
    ("F1", "Datos, taxonomía y corpus sintético", "Cerrada con sintéticos"),
    ("F2", "Preproceso, ataques y folds", "Cerrada"),
    ("F3", "Calidad, forense, OCR, reglas y registro", "Cerrada provisional"),
    ("F4", "Clasificador visual de página", "Cerrada experimental"),
    ("F5", "Localización y explicabilidad", "Cerrada experimental"),
    ("F6", "Rama semántica y explicación", "Cerrada experimental"),
    ("F7", "Fusión y calibración", "Cerrada experimental"),
    ("F8", "Evaluación integral", "Parcial: semillas; faltan robustez y kappa"),
    ("F9", "Demo final y Cap. IV", "En curso"),
]

RISK_UI = {
    "bajo": ("Riesgo bajo", "", "bajo"),
    "medio": ("Riesgo medio", "", "medio"),
    "alto": ("Riesgo alto", "", "alto"),
    "indeterminado": ("Sin determinar", "", "indeterminado"),
}
RISK_NOTES = {
    "bajo": "No se acumularon señales suficientes para priorizar revisión. Esto no certifica autenticidad.",
    "medio": "Hay señales parciales. Conviene revisar el detalle antes de decidir.",
    "alto": "Priorizar revisión humana. El sistema no rechaza documentos ni emite una condena.",
    "indeterminado": "La imagen no permite un análisis fiable. La salida correcta es recapturar.",
}
CALIBRATION_LABELS = {
    "identity": "Identidad (score del modelo)",
    "isotonic": "Isotónica",
    "platt": "Platt",
    "sin calibrar": "Sin calibrar",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_samples(path_str: str) -> list[dict[str, str]]:
    path = Path(path_str)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@st.cache_data(show_spinner=False)
def _load_people(path_str: str) -> tuple[tuple[str, ...], ...]:
    path = Path(path_str)
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(
        (
            _text(row.get("apellidos")),
            _text(row.get("nombres")),
            _text(row.get("dni")),
            _text(row.get("programa")),
            _text(row.get("modulo")),
            _text(row.get("periodo")),
            _text(row.get("fecha_nacimiento")),
            _text(row.get("persona_id")),
        )
        for row in rows
        if _text(row.get("apellidos")) or _text(row.get("nombres"))
    )


@st.cache_data(show_spinner=False)
def _load_v2_samples(root_str: str) -> list[dict[str, str]]:
    root = Path(root_str)
    rows: list[dict[str, str]] = []
    for name in ("manifest.jsonl", "attacks_manifest.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            regions = record.get("regions") or []
            first = regions[0] if regions else {}
            rows.append(
                {
                    "id": str(record.get("sample_id", "")),
                    "tipo": str(record.get("document_type", "")),
                    "variante": str(record.get("variant", "")),
                    "ataque": str(record.get("primary_attack") or ""),
                    "persona": str(record.get("source_id", "")),
                    "documento_padre": str(record.get("parent_id") or ""),
                    "campo_original": str(first.get("original_value") or ""),
                    "campo_alterado": str(first.get("altered_value") or ""),
                    "png": str(record.get("image_path", "")),
                    "mascara": (record.get("mask_paths") or [""])[0],
                    "sha256_png": str(record.get("document_id", "")).removeprefix("sha256:"),
                }
            )
    return sorted(rows, key=lambda row: row["id"])


@st.cache_data(show_spinner=False)
def _v2_registry_rows(root_str: str) -> tuple[tuple[str, ...], ...]:
    path = Path(root_str) / "manifest.jsonl"
    if not path.is_file():
        return ()
    entries = build_registry_from_manifest(path)
    rows: list[tuple[str, ...]] = []
    for entry in entries:
        apellidos, _, nombres = entry.nombre.partition(", ")
        rows.append(
            (
                apellidos,
                nombres,
                entry.dni or "",
                entry.programa or "",
                entry.modulo or "",
                entry.periodo or "",
                entry.fecha_nacimiento or "",
                entry.source_id or "",
            )
        )
    return tuple(rows)


def _entry_from_row(row: tuple[str, ...]) -> RegistryEntry:
    apellidos, nombres, dni, programa, modulo, periodo, birth, source_id = row
    return RegistryEntry(
        nombre=f"{apellidos}, {nombres}".strip(", "),
        dni=dni or None,
        programa=programa or None,
        modulo=modulo or None,
        periodo=periodo or None,
        fecha_nacimiento=birth or None,
        source_id=source_id or None,
    )


# ---------------------------------------------------------------------------
# Texto y formato
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_conf(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_ms(value: int | None) -> str:
    return "—" if value is None else f"{value} ms"


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(max(0, num_bytes))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{num_bytes} B"


def _field_label(key: str) -> str:
    match = re.fullmatch(r"(.+)_(\d+)", key)
    if match:
        base, index = match.groups()
        return f"{FIELD_LABELS.get(base, base)} {int(index) + 1}"
    return FIELD_LABELS.get(key, key.replace("_", " ").capitalize())


def _acta_rows(fields: dict[str, Any]) -> list[tuple[int, str, str, str]]:
    indices = {
        int(match.group(1)) for key in fields if (match := re.fullmatch(r"estudiante_(\d+)", key))
    }
    rows: list[tuple[int, str, str, str]] = []
    for index in sorted(indices):
        rows.append(
            (
                index + 1,
                _text(fields.get(f"estudiante_{index}")),
                _text(fields.get(f"dni_{index}")),
                _text(fields.get(f"nota_{index}")),
            )
        )
    return rows


def _code_label(code: str) -> str:
    if code in CODE_LABELS:
        return CODE_LABELS[code]
    if code.startswith("RULE_"):
        return code[5:].replace("_", " ").capitalize()
    return code.replace("_", " ").capitalize()


def _missing_fields(observation: Observation) -> list[str]:
    required = REQUIRED_FIELDS.get(observation.doctype.label, ())
    return [field for field in required if not observation.fields.get(field)]


@dataclass(frozen=True)
class RiskBand:
    key: str
    label: str
    icon: str
    note: str
    tone: str


def _risk_band(observation: Observation, score: float) -> RiskBand:
    """Banda heurística de F3; se conserva para pruebas y lectura por señal."""
    if not observation.quality.ok:
        return RiskBand(
            "indeterminado",
            "Indeterminado",
            "⚪",
            "La imagen no es apta; se recomienda recapturar.",
            "indeterminado",
        )
    if observation.ocr.engine in {"unavailable", "not_run", "disabled"}:
        if score >= DEFAULT_THRESHOLD:
            return RiskBand(
                "alto", "Alto", "🔴", "Solo forense: no se pudo leer el contenido.", "alto"
            )
        return RiskBand(
            "indeterminado", "Indeterminado", "⚪", "No se pudo leer el contenido.", "indeterminado"
        )
    if score >= DEFAULT_THRESHOLD:
        return RiskBand(
            "alto", "Alto", "🔴", "Priorizar revisión humana. No es una probabilidad.", "alto"
        )
    if score >= 0.25:
        return RiskBand(
            "medio", "Medio", "🟠", "Hay señales parciales; revisar el detalle.", "medio"
        )
    return RiskBand("bajo", "Bajo", "🟢", "Las señales implementadas no se activaron.", "bajo")


def _tip(text: str) -> str:
    safe = html.escape(str(text), quote=True)
    return (
        f'<span class="tip" data-tip="{safe}" title="{safe}" tabindex="0" '
        f'aria-label="{safe}">?</span>'
    )


# ---------------------------------------------------------------------------
# Tema visual
# ---------------------------------------------------------------------------
LIGHT = {
    "bg": "#F3F7FA",
    "surface": "#FFFFFF",
    "surface-2": "#EDF3F8",
    "surface-3": "#E1EAF2",
    "border": "#D4E0EA",
    "text": "#07111C",
    "text-2": "#2E4357",
    "muted": "#566D80",
    "primary": "#007A69",
    "primary-2": "#00A38B",
    "on-primary": "#FFFFFF",
    "accent": "#A85A00",
    "success": "#007A5A",
    "success-bg": "#E5F7F0",
    "warning": "#9A5900",
    "warning-bg": "#FFF3DF",
    "danger": "#B3132D",
    "danger-bg": "#FDE8EC",
    "shadow": "0 1px 0 rgba(7, 17, 28, 0.04)",
    "input": "#FFFFFF",
    "tooltip": "#07111C",
    "tooltip-text": "#F4FAFF",
    "grid": "rgba(7, 17, 28, 0.045)",
    "glow": "rgba(0, 122, 105, 0.10)",
    "font-display": 'ui-sans-serif, "SF Pro Display", "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif',
    "font-mono": 'ui-monospace, "SFMono-Regular", "Cascadia Code", "JetBrains Mono", "Liberation Mono", monospace',
}
DARK = {
    "bg": "#050B12",
    "surface": "#0A1522",
    "surface-2": "#0F1E2E",
    "surface-3": "#162A3D",
    "border": "#1C3248",
    "text": "#F2F8FF",
    "text-2": "#C8D8E6",
    "muted": "#8CA3B8",
    "primary": "#4FF0D0",
    "primary-2": "#8BFFE9",
    "on-primary": "#041014",
    "accent": "#FFB86B",
    "success": "#4FF0D0",
    "success-bg": "#072A24",
    "warning": "#FFB020",
    "warning-bg": "#2A1C05",
    "danger": "#FF5C7A",
    "danger-bg": "#2A0B12",
    "shadow": "0 0 0 1px rgba(79, 240, 208, 0.02)",
    "input": "#07111C",
    "tooltip": "#F2F8FF",
    "tooltip-text": "#050B12",
    "grid": "rgba(79, 240, 208, 0.055)",
    "glow": "rgba(79, 240, 208, 0.11)",
    "font-display": 'ui-sans-serif, "SF Pro Display", "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif',
    "font-mono": 'ui-monospace, "SFMono-Regular", "Cascadia Code", "JetBrains Mono", "Liberation Mono", monospace',
}

BASE_CSS = r"""
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
html, body, [class*="css"], .stApp, .stMarkdown, .stText, button, input, textarea {
    font-family: var(--font-display);
}
.stApp {
    background-image:
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px),
        radial-gradient(1100px 540px at 82% -8%, var(--glow), transparent 62%);
    background-size: 48px 48px, 48px 48px, 100% 100%;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"], .stDeployButton {
    display: none !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1480px; }
h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    font-family: var(--font-display);
    letter-spacing: -0.035em;
}
p, li { color: var(--text-2); }
.stMarkdown p, .stMarkdown li { color: var(--text-2); }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--text-2) !important; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
[data-testid="stMarkdownContainer"] { color: var(--text-2); }
a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="tab"]:focus-visible, [role="radio"]:focus-visible, [role="checkbox"]:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
}

/* Controles */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 2px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700 !important;
    padding: 0.58rem 0.85rem !important;
    transition: border-color .15s ease, color .15s ease, background .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: var(--on-primary) !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
    background: var(--primary-2) !important;
    border-color: var(--primary-2) !important;
    color: var(--on-primary) !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stTextInput input, .stTextArea textarea {
    background: var(--input) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    border-radius: 2px !important;
    font-family: var(--font-display);
}
div[data-baseweb="popover"], ul[data-testid="stSelectboxVirtualDropdown"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 7px 9px;
    background: var(--surface-2);
    color: var(--text-2) !important;
    transition: border-color .15s ease, background .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { border-color: var(--primary); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    border-color: var(--primary);
    background: var(--success-bg);
}
[data-testid="stSidebar"] [role="radiogroup"] label p { font-family: var(--font-mono); font-size: .68rem; letter-spacing: .02em; }
[data-testid="stSidebar"] .stToggle label { color: var(--text-2) !important; }
[data-testid="stAlert"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 16px;
    box-shadow: none;
}
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 12px 14px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: var(--font-mono); font-size: .65rem; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
pre, code, .stCode { background: var(--surface-2) !important; color: var(--text) !important; }
pre { border: 1px solid var(--border); border-radius: 2px; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 0;
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: .66rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary);
}
.stExpander {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}

/* Marca y barra lateral */
.side-brand { display: flex; gap: 11px; align-items: center; padding: 2px 2px 16px; }
.side-mark, .brand-mark {
    width: 40px; height: 40px;
    background: var(--primary); color: var(--on-primary);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-weight: 800; font-size: 17px;
    clip-path: polygon(0 0, 100% 0, 100% 68%, 68% 100%, 0 100%);
}
.side-brand-title { font-family: var(--font-mono); font-weight: 800; font-size: .92rem; color: var(--text); letter-spacing: .12em; line-height: 1.1; }
.side-brand-sub { font-family: var(--font-mono); font-size: .58rem; color: var(--muted); margin-top: 4px; letter-spacing: .1em; text-transform: uppercase; }
.side-label {
    font-family: var(--font-mono); font-size: .58rem; text-transform: uppercase;
    letter-spacing: .14em; color: var(--accent); font-weight: 800; margin: 14px 0 6px;
}
.side-note { font-size: .7rem; color: var(--muted); line-height: 1.45; }
.side-hr { height: 1px; background: var(--border); margin: 14px 0; }
.side-footer { font-family: var(--font-mono); font-size: .58rem; color: var(--muted); line-height: 1.7; letter-spacing: .04em; }

/* Cabecera */
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 20px; }
.brandline { display: flex; align-items: center; gap: 10px; color: var(--text); }
.brandline .brand-mark { width: 30px; height: 30px; font-size: 13px; }
.brandline span { font-family: var(--font-mono); font-size: .68rem; letter-spacing: .13em; text-transform: uppercase; font-weight: 700; }
.brandline b { color: var(--accent); font-weight: 800; }
.topbar-meta { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--muted); border-radius: 2px; padding: 4px 8px;
    font-family: var(--font-mono); font-size: .58rem; letter-spacing: .08em;
    text-transform: uppercase; font-weight: 700; white-space: nowrap;
}
.chip-good { color: var(--success); border-color: var(--success); background: var(--success-bg); }
.pulse { width: 6px; height: 6px; border-radius: 50%; background: var(--success); box-shadow: 0 0 0 0 var(--success); animation: pulse 2.4s ease-out infinite; }
.page-head { margin-bottom: 12px; }
.kicker {
    font-family: var(--font-mono); font-size: .62rem; text-transform: uppercase;
    letter-spacing: .16em; color: var(--accent); font-weight: 800; margin-bottom: 8px;
}
.page-title {
    font-size: clamp(2rem, 4.2vw, 4rem); line-height: .96; margin: 0;
    color: var(--text); font-weight: 750; letter-spacing: -.05em;
    text-wrap: balance; max-width: 980px;
}
.page-sub { color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 10px 0 0; max-width: 72ch; text-wrap: pretty; }
.status-strip { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--border); background: var(--surface); margin: 16px 0 22px; }
.status-item { padding: 9px 12px; border-right: 1px solid var(--border); }
.status-item:last-child { border-right: 0; }
.status-key { font-family: var(--font-mono); font-size: .55rem; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; }
.status-value { font-family: var(--font-mono); font-size: .66rem; color: var(--text); margin-top: 3px; letter-spacing: .04em; }

/* Riesgo */
.risk-hero {
    position: relative; display: grid; grid-template-columns: 1fr auto; gap: 24px;
    border: 1px solid var(--border); border-left: 3px solid var(--risk-color, var(--muted));
    border-radius: 2px; padding: 24px 26px; background: var(--surface);
    overflow: hidden; margin-bottom: 10px;
}
.risk-hero::before {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg, var(--risk-soft, transparent), transparent 58%);
    pointer-events: none;
}
.risk-hero > * { position: relative; z-index: 1; }
.riesgo-bajo { --risk-color: var(--success); --risk-soft: var(--success-bg); }
.riesgo-medio { --risk-color: var(--warning); --risk-soft: var(--warning-bg); }
.riesgo-alto { --risk-color: var(--danger); --risk-soft: var(--danger-bg); }
.riesgo-indeterminado { --risk-color: var(--muted); --risk-soft: var(--surface-2); }
.risk-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 6px; background: var(--muted); vertical-align: middle; }
.risk-dot.bajo { background: var(--success); }
.risk-dot.medio { background: var(--warning); }
.risk-dot.alto { background: var(--danger); }
.risk-eyebrow { font-family: var(--font-mono); font-size: .6rem; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); font-weight: 800; }
.risk-title { font-size: clamp(2rem, 3.2vw, 3.3rem); font-weight: 800; letter-spacing: -.05em; line-height: .98; margin: 8px 0 8px; color: var(--risk-color); }
.risk-desc { color: var(--text-2); font-size: .84rem; line-height: 1.55; max-width: 62ch; }
.risk-right { display: flex; flex-direction: column; align-items: flex-end; justify-content: center; gap: 8px; }
.score-ring {
    --p: 0%;
    width: 142px; height: 142px; border-radius: 50%;
    background: conic-gradient(var(--risk-color) var(--p), var(--surface-3) 0);
    display: grid; place-items: center; position: relative;
}
.score-ring::before { content: ""; position: absolute; inset: 7px; border-radius: 50%; background: var(--surface); }
.score-ring-inner { position: relative; z-index: 1; text-align: center; }
.risk-score { font-family: var(--font-mono); font-size: 2.05rem; font-weight: 800; color: var(--text); line-height: 1; font-variant-numeric: tabular-nums; }
.score-tag { font-family: var(--font-mono); font-size: .56rem; color: var(--muted); letter-spacing: .14em; margin-top: 4px; }
.risk-score-label { font-family: var(--font-mono); font-size: .57rem; color: var(--muted); text-align: right; letter-spacing: .04em; }
.risk-method { font-family: var(--font-mono); font-size: .58rem; color: var(--muted); text-align: right; }

/* Métricas */
.metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); border: 1px solid var(--border); background: var(--surface); margin: 14px 0 6px; }
.metric-card { padding: 13px 15px; border-right: 1px solid var(--border); min-height: 82px; background: transparent; }
.metric-card:last-child { border-right: 0; }
.metric-card::before { content: ""; display: block; width: 18px; height: 2px; background: var(--accent); margin-bottom: 9px; }
.metric-label { font-family: var(--font-mono); font-size: .55rem; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
.metric-value { font-family: var(--font-mono); font-size: 1.25rem; font-weight: 800; color: var(--text); margin-top: 5px; line-height: 1.05; font-variant-numeric: tabular-nums; }
.metric-sub { font-size: .67rem; color: var(--muted); margin-top: 5px; line-height: 1.35; }
.tone-good { color: var(--success) !important; }
.tone-warn { color: var(--warning) !important; }
.tone-bad { color: var(--danger) !important; }
.tone-neutral { color: var(--text) !important; }

/* Secciones y evidencias */
.card { background: var(--surface); border: 1px solid var(--border); padding: 16px; }
.section-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin: 24px 0 10px; border-top: 1px solid var(--border); padding-top: 13px; }
.section-title { font-size: 1rem; font-weight: 750; color: var(--text); margin: 0; letter-spacing: -.02em; }
.section-index { font-family: var(--font-mono); color: var(--accent); font-size: .62rem; letter-spacing: .1em; margin-right: 8px; }
.section-sub { font-family: var(--font-mono); font-size: .56rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.evidence-card {
    display: grid; grid-template-columns: 32px 1fr; gap: 10px;
    border: 1px solid var(--border); border-left: 2px solid var(--muted);
    background: var(--surface); padding: 10px 12px; margin-bottom: 7px;
}
.evidence-card.sev-alta { border-left-color: var(--danger); }
.evidence-card.sev-media { border-left-color: var(--warning); }
.evidence-card.sev-baja { border-left-color: var(--success); }
.evidence-index { font-family: var(--font-mono); font-size: .62rem; color: var(--muted); border-right: 1px solid var(--border); padding-right: 8px; padding-top: 2px; }
.evidence-top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.evidence-code { font-family: var(--font-mono); font-weight: 800; color: var(--text); font-size: .68rem; letter-spacing: .02em; text-transform: uppercase; }
.evidence-msg { color: var(--text-2); font-size: .79rem; line-height: 1.5; margin-top: 5px; }
.evidence-weight { font-family: var(--font-mono); font-size: .58rem; color: var(--muted); margin-top: 5px; }
.badge { display: inline-flex; align-items: center; border-radius: 2px; padding: 2px 6px; font-family: var(--font-mono); font-size: .55rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.sev-alta.badge { color: var(--danger); background: var(--danger-bg); }
.sev-media.badge { color: var(--warning); background: var(--warning-bg); }
.sev-baja.badge { color: var(--success); background: var(--success-bg); }
.pill { border-radius: 2px; padding: 3px 7px; font-family: var(--font-mono); font-size: .58rem; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
.pill-good { color: var(--success); background: var(--success-bg); }
.pill-warn { color: var(--warning); background: var(--warning-bg); }
.pill-bad { color: var(--danger); background: var(--danger-bg); }
.pill-neutral { color: var(--muted); background: var(--surface-2); }

/* Tablas */
.data-table { width: 100%; border-collapse: collapse; font-size: .78rem; }
.data-table th { text-align: left; color: var(--muted); font-family: var(--font-mono); font-size: .56rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; padding: 8px 10px; border-bottom: 1px solid var(--border); }
.data-table td { padding: 9px 10px; border-bottom: 1px solid var(--border); color: var(--text-2); vertical-align: top; }
.data-table tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover td { background: var(--surface-2); }
.data-table td strong { color: var(--text); }
.data-table .num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

/* Señales */
.signal-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0; border: 1px solid var(--border); }
.signal-card { background: var(--surface); border-right: 1px solid var(--border); padding: 11px 12px; }
.signal-card:last-child { border-right: 0; }
.signal-name { font-family: var(--font-mono); font-size: .58rem; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }
.signal-score { font-family: var(--font-mono); font-size: 1.15rem; font-weight: 800; color: var(--text); margin: 4px 0; font-variant-numeric: tabular-nums; }
.signal-bar { height: 3px; background: var(--surface-3); overflow: hidden; }
.signal-bar span { display: block; height: 100%; background: var(--primary); }
.signal-detail { font-family: var(--font-mono); font-size: .55rem; color: var(--muted); margin-top: 5px; line-height: 1.4; }

/* Vacío, ayuda y footer */
.empty { border: 1px dashed var(--border); background: var(--surface); padding: 22px; color: var(--text-2); }
.step-grid, .help-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0; border: 1px solid var(--border); background: var(--surface); margin: 14px 0; }
.step-card, .help-card { padding: 18px; border-right: 1px solid var(--border); }
.step-card:last-child, .help-card:last-child { border-right: 0; }
.step-number { width: 26px; height: 26px; border: 1px solid var(--accent); color: var(--accent); display: flex; align-items: center; justify-content: center; font-family: var(--font-mono); font-weight: 800; font-size: .65rem; }
.step-title, .help-title { font-weight: 750; color: var(--text); margin-top: 10px; font-size: .9rem; }
.step-body, .help-body { color: var(--muted); font-size: .78rem; line-height: 1.5; margin-top: 5px; }
.glossary-row { display: grid; grid-template-columns: 170px 1fr; gap: 12px; padding: 11px 0; border-bottom: 1px solid var(--border); }
.glossary-term { font-family: var(--font-mono); font-weight: 800; color: var(--accent); font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; }
.glossary-def { color: var(--text-2); font-size: .79rem; line-height: 1.5; }
.footer { margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--muted); font-family: var(--font-mono); font-size: .56rem; letter-spacing: .06em; text-transform: uppercase; display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; }

/* Tooltip */
.tip {
    position: relative; display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; margin-left: 5px; border-radius: 50%;
    border: 1px solid var(--border); background: var(--surface-2);
    color: var(--muted); font-family: var(--font-mono); font-size: 9px; font-weight: 800;
    cursor: help; vertical-align: middle;
}
.tip::after {
    content: attr(data-tip); position: absolute; left: 50%; bottom: calc(100% + 8px);
    transform: translateX(-50%); min-width: 210px; max-width: 290px;
    background: var(--tooltip); color: var(--tooltip-text);
    padding: 8px 10px; border-left: 2px solid var(--accent);
    font-family: var(--font-display); font-size: 11px; line-height: 1.45;
    font-weight: 500; text-align: left; white-space: normal;
    opacity: 0; visibility: hidden; transition: opacity .15s ease;
    box-shadow: 0 12px 28px rgba(0,0,0,.25); z-index: 9999; pointer-events: none;
}
.tip:hover::after, .tip:focus::after { opacity: 1; visibility: visible; }

/* Marco de escaneo sobre imágenes */
[data-testid="stImage"] {
    position: relative; padding: 5px; background: var(--surface-2);
    border: 1px solid var(--border); overflow: hidden;
}
[data-testid="stImage"]::before {
    content: "SCAN / VISUAL INPUT"; position: absolute; top: -1px; left: 8px;
    background: var(--surface-2); color: var(--accent);
    font-family: var(--font-mono); font-size: .5rem; letter-spacing: .12em;
    padding: 0 5px; z-index: 2;
}
[data-testid="stImage"]::after {
    content: ""; position: absolute; left: 5px; right: 5px; height: 1px;
    background: var(--primary); box-shadow: 0 0 10px var(--primary);
    animation: scan 5s ease-in-out infinite; opacity: .65; z-index: 2;
}
[data-testid="stImage"] img { display: block; width: 100%; }
@keyframes scan { 0%, 100% { top: 5px; } 50% { top: calc(100% - 6px); } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 var(--success); } 70% { box-shadow: 0 0 0 6px transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }

@media (max-width: 1100px) {
    .metric-grid { grid-template-columns: repeat(3, 1fr); }
    .metric-card:nth-child(3n) { border-right: 0; }
    .metric-card:nth-child(n+4) { border-top: 1px solid var(--border); }
}
@media (max-width: 900px) {
    .risk-hero { grid-template-columns: 1fr; }
    .risk-right { flex-direction: row; align-items: center; justify-content: flex-start; }
    .risk-score-label, .risk-method { text-align: left; }
    .status-strip { grid-template-columns: repeat(2, 1fr); }
    .status-item:nth-child(2n) { border-right: 0; }
    .status-item:nth-child(n+3) { border-top: 1px solid var(--border); }
    .topbar { flex-direction: column; align-items: flex-start; }
    .glossary-row { grid-template-columns: 1fr; gap: 3px; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .metric-card:nth-child(2n) { border-right: 0; }
    .metric-card:nth-child(n+3) { border-top: 1px solid var(--border); }
    .step-grid, .help-grid { grid-template-columns: 1fr; }
    .step-card, .help-card { border-right: 0; border-bottom: 1px solid var(--border); }
    .step-card:last-child, .help-card:last-child { border-bottom: 0; }
}
@media (max-width: 520px) {
    .block-container { padding-left: 12px; padding-right: 12px; }
    .page-sub { font-size: .84rem; }
    .score-ring { width: 120px; height: 120px; }
    .risk-score { font-size: 1.75rem; }
    .metric-grid { grid-template-columns: 1fr; }
    .metric-card { border-right: 0; border-top: 1px solid var(--border); }
    .metric-card:first-child { border-top: 0; }
    .signal-card { border-right: 0; border-bottom: 1px solid var(--border); }
    .signal-card:last-child { border-bottom: 0; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
}
"""


def _theme_css(dark: bool) -> str:
    tokens = DARK if dark else LIGHT
    root = ":root{" + "".join(f"--{key}:{value};" for key, value in tokens.items())
    root += "color-scheme:" + ("dark" if dark else "light") + ";}"
    return root + BASE_CSS


# ---------------------------------------------------------------------------
# Componentes HTML
# ---------------------------------------------------------------------------
def _metric_card(
    label: str, value: str, sub: str = "", tone: str = "neutral", help_text: str = ""
) -> str:
    tip = _tip(help_text) if help_text else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}{tip}</div>'
        f'<div class="metric-value tone-{tone}">{html.escape(value)}</div>'
        f'<div class="metric-sub">{html.escape(sub)}</div>'
        "</div>"
    )


def _metric_grid(items: list[tuple[str, str, str, str, str]]) -> str:
    return (
        '<div class="metric-grid">'
        + "".join(
            _metric_card(label, value, sub, tone, help_text)
            for label, value, sub, tone, help_text in items
        )
        + "</div>"
    )


def _section(title: str, sub: str = "", help_text: str = "", index: str = "") -> str:
    tip = _tip(help_text) if help_text else ""
    prefix = f'<span class="section-index">{html.escape(index)}</span>' if index else ""
    return (
        '<div class="section-head">'
        f'<div class="section-title">{prefix}{html.escape(title)}{tip}</div>'
        f'<div class="section-sub">{html.escape(sub)}</div>'
        "</div>"
    )


def _evidence_card(evidence: Any, index: int = 1) -> str:
    severity = getattr(evidence, "severity", None)
    sev_key = severity.value if isinstance(severity, Severity) else str(severity or "baja").lower()
    sev_label = {"alta": "Alta", "media": "Media", "baja": "Baja"}.get(
        sev_key, str(sev_key).capitalize()
    )
    code = str(getattr(evidence, "code", ""))
    message = str(getattr(evidence, "message", ""))
    weight = getattr(evidence, "weight", None)
    weight_html = ""
    if weight is not None and abs(float(weight)) >= 0.01:
        weight_html = (
            f'<div class="evidence-weight">Contribución al riesgo: {float(weight):+.2f} '
            f"{_tip('Valor SHAP de la señal. Indica cuánto empuja el score; no es una probabilidad.')}</div>"
        )
    return (
        f'<div class="evidence-card sev-{sev_key}">'
        f'<div class="evidence-index">{index:02d}</div>'
        "<div>"
        '<div class="evidence-top">'
        f'<span class="badge sev-{sev_key}">{html.escape(sev_label)}</span>'
        f'<span class="evidence-code">{html.escape(_code_label(code))}</span>'
        "</div>"
        f'<div class="evidence-msg">{html.escape(message)}</div>'
        f"{weight_html}"
        "</div>"
        "</div>"
    )


def _fields_table(fields: dict[str, Any]) -> str:
    if not fields:
        return '<div class="empty">No se extrajeron campos por OCR.</div>'
    rows = []
    for key, value in fields.items():
        if value in (None, ""):
            continue
        rows.append(
            f"<tr><td><strong>{html.escape(_field_label(key))}</strong></td>"
            f"<td>{html.escape(_text(value))}</td></tr>"
        )
    if not rows:
        return '<div class="empty">No se extrajeron campos con valor.</div>'
    return (
        '<table class="data-table"><thead><tr><th>Campo</th><th>Valor extraído</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _acta_table(rows: list[tuple[int, str, str, str]]) -> str:
    if not rows:
        return ""
    body = "".join(
        f"<tr><td class='num'>{index}</td><td>{html.escape(name)}</td>"
        f"<td>{html.escape(dni)}</td><td class='num'>{html.escape(nota)}</td></tr>"
        for index, name, dni, nota in rows
    )
    return (
        '<table class="data-table"><thead><tr><th>#</th><th>Estudiante</th><th>DNI</th>'
        f"<th>Nota</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _signal_card(name: str, signal: SignalScore) -> str:
    label = SIGNAL_LABELS.get(name, name.capitalize())
    label_html = html.escape(label) + _tip(SIGNAL_HELP.get(name, "Señal forense."))
    score = signal.score
    if not signal.applicable:
        value = "No aplica"
        width = 0.0
    elif score is None:
        value = "—"
        width = 0.0
    else:
        value = f"{score:.2f}"
        width = max(0.0, min(1.0, float(score))) * 100
    detail = ""
    if signal.details:
        items = [f"{key}: {value}" for key, value in list(signal.details.items())[:2]]
        detail = " · ".join(html.escape(str(item)) for item in items)
    return (
        '<div class="signal-card">'
        f'<div class="signal-name">{label_html}</div>'
        f'<div class="signal-score">{html.escape(value)}</div>'
        f'<div class="signal-bar"><span style="width:{width:.0f}%"></span></div>'
        f'<div class="signal-detail">{detail}</div>'
        "</div>"
    )


def _registry_card(observation: Observation) -> str:
    registry = observation.registry
    if registry.matched:
        tone = "good"
        label = "Coincide con el registro"
    elif registry.score is None:
        tone = "neutral"
        label = "Sin datos de registro"
    else:
        tone = "bad"
        label = "No coincide con el registro"
    score = "—" if registry.score is None else f"{registry.score:.2f}"
    diffs = "".join(f"<li>{html.escape(str(diff))}</li>" for diff in registry.diffs[:6])
    diffs_html = (
        f"<ul>{diffs}</ul>"
        if diffs
        else "<p style='color:var(--muted)'>Sin diferencias registradas.</p>"
    )
    return (
        '<div class="card">'
        f'<span class="pill pill-{tone}">{html.escape(label)}</span>'
        f'<div class="metric-value" style="margin-top:8px;">{score}</div>'
        f'<div class="metric-sub">Score de coincidencia con el espejo del CETPRO {_tip("Compara nombre, DNI, programa y fechas contra el registro sintético. No sale del equipo.")}</div>'
        f"<div style='margin-top:10px;color:var(--text-2);font-size:.8rem;'>{diffs_html}</div>"
        "</div>"
    )


def _quality_rows(observation: Observation) -> list[tuple[str, str]]:
    quality = observation.quality
    rows = [
        ("Analizable", "Sí" if quality.ok else "No"),
        ("Desenfoque", _fmt_score(quality.blur)),
        ("Inclinación", "—" if quality.skew_deg is None else f"{quality.skew_deg:.1f}°"),
        (
            "Tipo de captura",
            CAPTURE_LABELS.get(observation.capture.type.value, observation.capture.type.value),
        ),
        (
            "DPI estimado",
            "—" if observation.capture.dpi_est is None else str(observation.capture.dpi_est),
        ),
        (
            "Incidencias",
            ", ".join(QUALITY_LABELS.get(item, item) for item in quality.issues) or "Ninguna",
        ),
    ]
    return rows


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td><strong>{html.escape(str(key))}</strong></td><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return f'<table class="data-table"><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------
def _analysis_key(
    data: bytes,
    filename: str,
    selected_row: dict[str, str] | None,
    registry_rows: tuple[tuple[str, ...], ...],
    use_llm: bool,
) -> str:
    digest = hashlib.sha256(data).hexdigest()[:16]
    sample = _text((selected_row or {}).get("id"))
    return f"{digest}:{filename}:{sample}:{len(registry_rows)}:{int(use_llm)}"


def _run_integral(
    data: bytes,
    filename: str,
    selected_row: dict[str, str] | None,
    registry_rows: tuple[tuple[str, ...], ...],
    *,
    use_llm: bool,
    force: bool = False,
) -> dict[str, Any]:
    key = _analysis_key(data, filename, selected_row, registry_rows, use_llm)
    cache: dict[str, Any] = st.session_state.setdefault("integral_cache", {})
    if force:
        cache.pop(key, None)
    if key in cache:
        return cache[key]

    registry = [_entry_from_row(row) for row in registry_rows] or None
    with st.spinner("Analizando calidad, forense, OCR, semántica y fusión…"):
        result = analyze_full(
            data,
            filename=filename,
            registry=registry,
            data_dir=V2_ROOT,
            artifacts_dir=RUNS_DIR,
            exclude_sample_id=_text((selected_row or {}).get("id")) or None,
            use_llm=use_llm,
            today=date.today(),
            f4_checkpoint=F4_CHECKPOINT,
            f5_checkpoint=F5_CHECKPOINT,
            f7_model=F7_MODEL,
        )
    payload = result.to_dict()
    payload["created_at"] = datetime.now(UTC).isoformat()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    pdf_path = (
        REPORTS_DIR / f"{payload['document_id'].removeprefix('sha256:')[:16]}_{fingerprint}.pdf"
    )
    try:
        build_report(payload, pdf_path)
    except Exception as exc:  # pragma: no cover
        st.warning(f"No se pudo generar el PDF: {exc}")
        pdf_path = None

    analysis_id = None
    try:
        with HistoryStore(HISTORY_DB) as store:
            analysis_id = store.save_analysis(payload)
    except Exception as exc:  # pragma: no cover
        st.warning(f"No se pudo guardar en el historial: {exc}")

    entry = {
        "payload": payload,
        "pdf": str(pdf_path) if pdf_path is not None else None,
        "analysis_id": analysis_id,
    }
    cache[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar(
    sample_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
) -> tuple[bytes | None, str, dict[str, str] | None, tuple[tuple[str, ...], ...], str]:
    with st.sidebar:
        st.markdown(
            '<div class="side-brand">'
            '<div class="side-mark">V</div>'
            '<div><div class="side-brand-title">Verifica</div>'
            '<div class="side-brand-sub">CETPRO San Agustín</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">Navegación</div>', unsafe_allow_html=True)
        view = st.radio(
            "Navegación",
            ("Análisis", "Historial", "Ayuda"),
            label_visibility="collapsed",
            help="Cambia entre el análisis de documentos, el historial y la ayuda.",
        )
        st.markdown('<div class="side-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-label">Documento</div>', unsafe_allow_html=True)
        origin = st.radio(
            "Origen del documento",
            (
                "Corpus v2 · 50 documentos",
                "Corpus v1 · 20 muestras",
                "Subir archivo",
            ),
            help=(
                "Corpus v2: documentos base con los que se entrenaron F4/F5/F7. "
                "Corpus v1: una muestra por ataque T1-T8. Subir archivo: imagen propia."
            ),
        )

        data: bytes | None = None
        filename = ""
        selected_row: dict[str, str] | None = None
        if origin.startswith("Corpus v2"):
            if v2_rows:
                labels = [
                    f"{row['id']} · {DOCUMENT_LABELS.get(row['tipo'], row['tipo'])}"
                    f" · {row['ataque'] or 'auténtico'}"
                    for row in v2_rows
                ]
                selected = st.selectbox(
                    "Documento",
                    labels,
                    help="Documentos base del corpus v2; las aumentaciones no se listan.",
                )
                selected_row = v2_rows[labels.index(selected)]
                path = V2_ROOT / _text(selected_row.get("png"))
                if path.is_file():
                    data = path.read_bytes()
                    filename = path.name
                else:
                    st.error(f"No se encontró la imagen: {path}")
            else:
                st.warning("No se encontró el corpus v2. Ejecuta `make synth-v2`.")
        elif origin.startswith("Corpus v1"):
            if sample_rows:
                labels = [
                    f"{row['id']} · {DOCUMENT_LABELS.get(row['tipo'], row['tipo'])}"
                    for row in sample_rows
                ]
                selected = st.selectbox(
                    "Muestra",
                    labels,
                    help="20 muestras del corpus v1, una por ataque T1-T8.",
                )
                selected_row = sample_rows[labels.index(selected)]
                path = SAMPLE_ROOT / _text(selected_row.get("png"))
                if path.is_file():
                    data = path.read_bytes()
                    filename = path.name
                else:
                    st.error(f"No se encontró la imagen: {path}")
            else:
                st.warning("No se encontró data/muestras.")
        else:
            uploaded = st.file_uploader(
                "Imagen del documento",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
                help="Formatos: PNG, JPG, TIFF, BMP y WEBP. Para PDF, exporta la página a PNG.",
            )
            if uploaded is not None:
                data = uploaded.getvalue()
                filename = uploaded.name

        use_registry = st.toggle(
            "Cruzar con registro",
            value=True,
            help="Compara nombre/DNI con el espejo sintético del CETPRO. No sale del equipo.",
        )
        if not use_registry:
            registry_rows: tuple[tuple[str, ...], ...] = ()
        elif origin.startswith("Corpus v1"):
            registry_rows = _load_people(str(SAMPLE_PEOPLE))
        else:
            registry_rows = _v2_registry_rows(str(V2_ROOT))

        st.markdown('<div class="side-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-label">Apariencia</div>', unsafe_allow_html=True)
        st.toggle(
            "Modo oscuro",
            value=bool(st.session_state.get("dark_mode", True)),
            key="dark_mode",
            help="Cambia entre tema claro y oscuro. La preferencia se mantiene en la sesión.",
        )
        st.markdown('<div class="side-hr"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-footer">'
            "Versión 1.0 · prototipo de curso<br>"
            "Datos locales · sin PII real<br>"
            "Sin decisión automática"
            "</div>",
            unsafe_allow_html=True,
        )
    return data, filename, selected_row, registry_rows, view


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------
def _page_header(title: str, subtitle: str, kicker: str = "Verificación documental") -> None:
    st.markdown(
        '<div class="topbar">'
        '<div class="brandline"><span class="brand-mark">V</span>'
        "<span>VERIFICA <b>/ CETPRO</b></span></div>"
        '<div class="topbar-meta">'
        '<span class="chip">NODE 01 / LOCAL</span>'
        '<span class="chip chip-good"><span class="pulse"></span>SISTEMA LISTO</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-head">'
        f'<div class="kicker">// {html.escape(kicker).upper()}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-sub">{html.escape(subtitle)}</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-strip">'
        '<div class="status-item"><div class="status-key">Entorno</div><div class="status-value">LOCAL / OFFLINE</div></div>'
        '<div class="status-item"><div class="status-key">Motor</div><div class="status-value">F3-F7 / CPU</div></div>'
        '<div class="status-item"><div class="status-key">Datos</div><div class="status-value">SINTÉTICOS</div></div>'
        '<div class="status-item"><div class="status-key">Política</div><div class="status-value">REVISIÓN HUMANA</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _footer() -> None:
    st.markdown(
        '<div class="footer">'
        "<span>Verifica CETPRO · v1.0 · datos sintéticos</span>"
        "<span>El sistema prioriza revisión humana; no rechaza documentos.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _empty_analysis() -> None:
    _page_header(
        "Mesa de verificación documental",
        "Sube una imagen o elige un documento del corpus. El sistema devuelve una banda de riesgo, "
        "evidencias explicables, mapa de localización, explicación y un informe descargable.",
    )
    st.markdown(
        '<div class="step-grid">'
        '<div class="step-card"><div class="step-number">1</div>'
        '<div class="step-title">Elige el origen</div>'
        '<div class="step-body">Corpus v2, corpus v1 o sube una imagen PNG/JPG. Está en la barra lateral.</div></div>'
        '<div class="step-card"><div class="step-number">2</div>'
        '<div class="step-title">Analiza</div>'
        '<div class="step-body">Calidad, forense, OCR, reglas, registro, semántica y fusión F7.</div></div>'
        '<div class="step-card"><div class="step-number">3</div>'
        '<div class="step-title">Revisa y exporta</div>'
        '<div class="step-body">Lee evidencias y mapa, descarga PDF/JSON y registra la revisión humana.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.info("Selecciona un documento en la barra lateral para comenzar.")


def _analysis_page(
    data: bytes | None,
    filename: str,
    selected_row: dict[str, str] | None,
    registry_rows: tuple[tuple[str, ...], ...],
) -> None:
    if data is None:
        _empty_analysis()
        return

    _page_header(
        "Resultado de la verificación",
        "Evidencias, mapa de localización y explicación para apoyar la revisión humana. "
        "Ninguna salida es una decisión automática.",
    )
    controls = st.columns([1.2, 1.2, 2.6])
    with controls[0]:
        use_llm = st.toggle(
            "Explicación con Ollama",
            value=False,
            help="Si Ollama está disponible, redacta la explicación. Si falla, se usa la plantilla local.",
        )
    with controls[1]:
        force = st.button("Volver a analizar", width="stretch")
    with controls[2]:
        st.caption("El análisis se guarda automáticamente en el historial local.")

    try:
        entry = _run_integral(
            data,
            filename,
            selected_row,
            registry_rows,
            use_llm=use_llm,
            force=force,
        )
    except Exception as exc:  # pragma: no cover
        st.error(f"No se pudo analizar el documento: {exc}")
        return

    _render_analysis(entry, data, filename, selected_row)


def _render_analysis(
    entry: dict[str, Any],
    data: bytes,
    filename: str,
    selected_row: dict[str, str] | None,
) -> None:
    payload = entry["payload"]
    observation = Observation.model_validate(payload["observation"])
    fusor = payload.get("fusor") or {}
    risk = str(fusor.get("risk") or "indeterminado")
    label, _icon, tone = RISK_UI.get(risk, RISK_UI["indeterminado"])
    note = RISK_NOTES.get(risk, RISK_NOTES["indeterminado"])
    p_fake = fusor.get("p_fake")
    raw_score = fusor.get("raw_score")
    method = str(fusor.get("calibration_method") or "sin calibrar")
    pct = 0.0 if p_fake is None else max(0.0, min(100.0, float(p_fake) * 100.0))

    st.markdown(
        f'<div class="risk-hero riesgo-{tone}">'
        "<div>"
        f'<div class="risk-eyebrow"><span class="risk-dot {tone}"></span>RESULTADO DE LA FUSIÓN F7</div>'
        f'<div class="risk-title">{html.escape(label)}</div>'
        f'<div class="risk-desc">{html.escape(note)}</div>'
        "</div>"
        '<div class="risk-right">'
        f'<div class="score-ring" style="--p:{pct:.1f}%;">'
        '<div class="score-ring-inner">'
        f'<div class="risk-score">{_fmt_score(p_fake)}</div>'
        '<div class="score-tag">F7 / SCORE</div>'
        "</div></div>"
        f'<div class="risk-score-label">Score F7 {_tip("Score del modelo de fusión. No es una probabilidad institucional de fraude.")}</div>'
        f'<div class="risk-method">{html.escape(CALIBRATION_LABELS.get(method, method))}</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    download_cols = st.columns([1, 1, 3])
    with download_cols[0]:
        st.download_button(
            "Descargar PDF",
            data=Path(entry["pdf"]).read_bytes()
            if entry.get("pdf") and Path(entry["pdf"]).is_file()
            else b"",
            file_name=Path(entry["pdf"]).name if entry.get("pdf") else "informe.pdf",
            mime="application/pdf",
            width="stretch",
            type="primary",
            disabled=not (entry.get("pdf") and Path(entry["pdf"]).is_file()),
        )
    with download_cols[1]:
        st.download_button(
            "Descargar JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name=f"{payload['document_id'][7:19]}_informe.json",
            mime="application/json",
            width="stretch",
        )
    with download_cols[2]:
        if selected_row is not None:
            st.caption(
                f"Documento del corpus: `{_text(selected_row.get('id'))}` · {DOCUMENT_LABELS.get(_text(selected_row.get('tipo')), '')}"
            )
        else:
            st.caption(f"Archivo: `{filename}`")

    metrics = [
        (
            "Tipo documental",
            DOCUMENT_LABELS.get(observation.doctype.label, observation.doctype.label),
            "Detectado por OCR + reglas",
            "neutral",
            "Tipo inferido del texto y la plantilla; puede fallar en documentos fabricados.",
        ),
        (
            "Captura",
            CAPTURE_LABELS.get(observation.capture.type.value, observation.capture.type.value),
            "Tipo de imagen",
            "neutral",
            "Determina qué señales forenses aplican (por ejemplo, ELA solo en JPEG).",
        ),
        (
            "OCR",
            _fmt_conf(observation.ocr.mean_conf),
            f"{observation.ocr.n_words} palabras",
            "neutral",
            "Confianza media del reconocimiento óptico de caracteres.",
        ),
        (
            "F4 página",
            _fmt_score(payload.get("f4_score")),
            "EfficientNet-B0",
            "neutral",
            "Clasificador visual de página; detecta cambios grandes, no ediciones pequeñas.",
        ),
        (
            "Latencia",
            _fmt_ms(payload.get("timings_ms", {}).get("total_ms")),
            "Pipeline completo",
            "neutral",
            "Tiempo total en CPU; el OCR suele ser el mayor costo.",
        ),
    ]
    st.markdown(_metric_grid(metrics), unsafe_allow_html=True)

    left, right = st.columns([1.0, 1.55], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Documento original**")
            st.image(data, caption=filename, width="stretch")
        with st.container(border=True):
            st.markdown("**Mapa de localización F5**")
            overlay = (payload.get("f5") or {}).get("overlay_path")
            if overlay and Path(overlay).is_file():
                st.image(overlay, caption="Probabilidad de manipulación", width="stretch")
                st.caption(
                    "Ayuda cualitativa. F5 funciona mejor en cambios grandes (foto, página completa) "
                    "y falla en ediciones pequeñas de texto (T1-T4)."
                )
            else:
                st.info("F5 no disponible para este documento.")
    with right:
        st.markdown(
            _section(
                "Evidencias",
                f"{len(observation.fusion.evidences)} hallazgos",
                "Señales ordenadas por contribución al riesgo. No son una condena.",
                index="01",
            ),
            unsafe_allow_html=True,
        )
        if observation.fusion.evidences:
            for index, evidence in enumerate(observation.fusion.evidences, start=1):
                st.markdown(_evidence_card(evidence, index), unsafe_allow_html=True)
        else:
            st.info("No se registraron evidencias automáticas.")
        explanation = payload.get("explanation") or {}
        st.markdown(
            _section(
                "Explicación",
                f"Fuente: {explanation.get('source', '—')}",
                "Resumen en lenguaje natural a partir de los hallazgos estructurados.",
                index="02",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card"><div style="color:var(--text-2);font-size:.84rem;line-height:1.55;">'
            f"{html.escape(str(explanation.get('text', 'Sin explicación.'))).replace(chr(10), '<br>')}"
            "</div></div>",
            unsafe_allow_html=True,
        )
        if entry.get("analysis_id"):
            st.caption(f"Guardado en historial SQLite · análisis #{entry['analysis_id']}")

    detail_tabs = st.tabs(["Campos extraídos", "Señales técnicas", "Registro", "JSON"])
    with detail_tabs[0]:
        if observation.fields:
            st.markdown(_fields_table(observation.fields), unsafe_allow_html=True)
            acta = _acta_rows(observation.fields)
            if acta:
                st.markdown("**Tabla del acta**")
                st.markdown(_acta_table(acta), unsafe_allow_html=True)
        else:
            st.info("No se extrajeron campos por OCR.")
        missing = _missing_fields(observation)
        if missing:
            st.warning("Campos requeridos no encontrados: " + ", ".join(missing))
    with detail_tabs[1]:
        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "F4 · página",
                _fmt_score(payload.get("f4_score")),
                help="Clasificador visual de página; detecta cambios grandes, no ediciones pequeñas.",
            )
            st.caption("Clasificador de página; no localiza.")
        with cols[1]:
            f5 = (payload.get("f5") or {}).get("summary") or {}
            st.metric(
                "F5 · max",
                "—" if not f5 else f"{float(f5.get('max_prob', 0)):.3f}",
                help="Probabilidad máxima del mapa de localización F5. Es una ayuda cualitativa.",
            )
            st.caption(f"Área marcada: {0 if not f5 else float(f5.get('area_fraction', 0)):.1%}")
        with cols[2]:
            semantic = payload.get("semantic") or {}
            st.metric(
                "F6 · banderas",
                str(len(semantic.get("flags") or [])),
                help="Banderas de la rama semántica: reutilización, nombre inconsistente, fecha improbable.",
            )
            st.caption(", ".join(semantic.get("flags") or []) or "sin banderas")
        st.markdown("**Señales forenses**")
        cards = "".join(
            _signal_card(name, getattr(observation.forensics, name))
            for name in ("metadata", "localization", "ela", "noise", "chromatic", "jpeg")
        )
        st.markdown(f'<div class="signal-grid">{cards}</div>', unsafe_allow_html=True)
        st.markdown("**Calidad y captura**")
        st.markdown(_kv_table(_quality_rows(observation)), unsafe_allow_html=True)
        if raw_score is not None:
            st.caption(
                f"Score crudo: {float(raw_score):.3f} · calibración: {CALIBRATION_LABELS.get(method, method)}"
            )
    with detail_tabs[2]:
        st.markdown(_registry_card(observation), unsafe_allow_html=True)
        if observation.registry.diffs:
            st.caption("Diferencias detectadas frente al espejo sintético del CETPRO.")
    with detail_tabs[3]:
        st.json(payload)


def _history_page() -> None:
    _page_header(
        "Registro de auditoría",
        "Consulta los análisis guardados y registra la revisión humana. La base es SQLite, local.",
        kicker="Trazabilidad",
    )
    try:
        with HistoryStore(HISTORY_DB) as store:
            stats = store.stats()
            rows = store.recent(50)
            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    _metric_card("Análisis", str(stats["total"]), "guardados en SQLite", "neutral"),
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    _metric_card(
                        "Riesgo alto", str(stats["by_risk"].get("alto", 0)), "priorizados", "bad"
                    ),
                    unsafe_allow_html=True,
                )
            with cols[2]:
                st.markdown(
                    _metric_card(
                        "Revisados", str(stats["reviewed"]), "con feedback humano", "good"
                    ),
                    unsafe_allow_html=True,
                )
            if not rows:
                st.info(
                    "Todavía no hay análisis guardados. Analiza un documento en la pestaña Análisis."
                )
                return
            st.dataframe(
                [
                    {
                        "id": row["id"],
                        "fecha": row["created_at"][:19].replace("T", " "),
                        "documento": row["filename"],
                        "tipo": row["doctype"],
                        "score": "—" if row["p_fake"] is None else round(float(row["p_fake"]), 3),
                        "riesgo": row["risk"],
                        "explicación": row["explanation_source"],
                    }
                    for row in rows
                ],
                width="stretch",
                hide_index=True,
            )
            options = {
                f"#{row['id']} · {row['filename']} · {row['risk']}": row["id"] for row in rows
            }
            selected = st.selectbox("Ver detalle", list(options))
            analysis_id = options[selected]
            payload = store.get(analysis_id)
            if payload:
                observation = Observation.model_validate(payload["observation"])
                risk = str((payload.get("fusor") or {}).get("risk") or "indeterminado")
                label, _icon, tone = RISK_UI.get(risk, RISK_UI["indeterminado"])
                st.markdown(
                    f'<div class="card"><div class="risk-eyebrow"><span class="risk-dot {tone}"></span>{html.escape(label)}</div>'
                    f'<div class="metric-sub">{html.escape(payload.get("created_at", "")[:19])} · '
                    f"{html.escape(observation.doctype.label)}</div></div>",
                    unsafe_allow_html=True,
                )
                explanation = payload.get("explanation") or {}
                st.markdown(f"> {html.escape(str(explanation.get('text', 'Sin explicación.')))}")
                for index, evidence in enumerate(observation.fusion.evidences, start=1):
                    st.markdown(_evidence_card(evidence, index), unsafe_allow_html=True)
                st.markdown("**Revisión humana**")
                with st.form(f"feedback_{analysis_id}"):
                    label = st.radio(
                        "Resultado de la revisión",
                        ("requiere_revision", "confirmado_manualmente", "descartado"),
                        horizontal=True,
                    )
                    notes = st.text_area("Notas de la revisión", height=80)
                    submitted = st.form_submit_button("Guardar revisión")
                if submitted:
                    store.add_feedback(analysis_id, label=label, notes=notes, reviewer="demo")
                    st.success("Revisión guardada en SQLite.")
                previous = store.feedback_for(analysis_id)
                if previous:
                    st.caption(
                        "Revisiones previas: "
                        + "; ".join(
                            f"{item['label']} ({item['created_at'][:10]})" for item in previous
                        )
                    )
    except Exception as exc:
        st.error(f"No se pudo abrir el historial SQLite: {exc}")


def _help_page() -> None:
    _page_header(
        "Centro de ayuda",
        "Cómo interpretar el resultado, qué significa cada término y qué límites tiene el prototipo.",
        kicker="Soporte",
    )
    tabs = st.tabs(["Cómo leer", "Glosario", "Límites", "Estado"])
    with tabs[0]:
        st.markdown(
            '<div class="help-grid">'
            '<div class="help-card"><div class="pill pill-good">Bajo</div>'
            '<div class="help-title">Sin señales suficientes</div>'
            '<div class="help-body">No se acumularon indicios para priorizar revisión. No certifica autenticidad.</div></div>'
            '<div class="help-card"><div class="pill pill-warn">Medio</div>'
            '<div class="help-title">Revisar el detalle</div>'
            '<div class="help-body">Hay señales parciales. Conviene leer las evidencias y el mapa antes de decidir.</div></div>'
            '<div class="help-card"><div class="pill pill-bad">Alto</div>'
            '<div class="help-title">Priorizar revisión</div>'
            '<div class="help-body">El sistema sugiere revisar primero. No rechaza ni condena; la persona decide.</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _section(
                "Cómo funciona",
                "cuatro capas + fusión",
                "El pipeline combina señales independientes; ninguna decide sola.",
                index="01",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="step-grid">'
            '<div class="step-card"><div class="step-number">1</div><div class="step-title">Calidad</div>'
            '<div class="step-body">¿La imagen es analizable? Si no, la salida correcta es recapturar.</div></div>'
            '<div class="step-card"><div class="step-number">2</div><div class="step-title">Forense y OCR</div>'
            '<div class="step-body">Huellas de edición + texto, campos y coordenadas.</div></div>'
            '<div class="step-card"><div class="step-number">3</div><div class="step-title">Reglas y semántica</div>'
            '<div class="step-body">Fechas, DNI, promedios, registro y consistencia de entidades.</div></div>'
            '<div class="step-card"><div class="step-number">4</div><div class="step-title">Fusión F7</div>'
            '<div class="step-body">Modelo LightGBM + calibración + evidencias SHAP.</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with tabs[1]:
        terms = [
            (
                "Score F7",
                "Score del modelo de fusión. En este corpus la calibración elegida es identidad; no es una probabilidad institucional.",
            ),
            (
                "Banda de riesgo",
                "Clasificación cualitativa (bajo/medio/alto) para priorizar revisión, no un veredicto.",
            ),
            ("F3", "Línea base: calidad, forense clásico, OCR, extracción, reglas y registro."),
            (
                "F4",
                "Clasificador visual de página; detecta cambios grandes como foto o documento completo.",
            ),
            ("F5", "Modelo de localización; marca zonas sospechosas. Falla en ediciones pequeñas."),
            ("F6", "Rama semántica: entidades, duplicados y explicación con plantilla."),
            ("F7", "Fusión calibrada que combina todas las señales."),
            ("OCR", "Reconocimiento óptico de caracteres: convierte la imagen en texto."),
            ("SHAP", "Técnica para medir cuánto contribuye cada señal al score."),
            (
                "Registro",
                "Espejo sintético del CETPRO para comparar nombre, DNI, programa y fechas.",
            ),
            (
                "T1-T8",
                "Taxonomía de fraude del proyecto: sustitución, sello, copy-move, borrado, foto, fabricado, reutilización e inconsistencia.",
            ),
            ("Recapturar", "La imagen no es analizable. No es una acusación de fraude."),
            (
                "Calibración",
                "Método que ajusta el score: identidad, isotónica o Platt. Se elige por validación.",
            ),
        ]
        rows = "".join(
            f'<div class="glossary-row"><div class="glossary-term">{html.escape(term)}</div>'
            f'<div class="glossary-def">{html.escape(description)}</div></div>'
            for term, description in terms
        )
        st.markdown(rows, unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(
            '<div class="card"><ul style="margin:0;padding-left:18px;line-height:1.8;color:var(--text-2);font-size:.86rem;">'
            "<li>Todo el corpus es <strong>sintético</strong>; no hay documentos reales del CETPRO.</li>"
            "<li>El corpus base tiene 50 documentos y 10 personas; los intervalos son anchos.</li>"
            "<li>F5 no localiza bien T1-T4 (ediciones pequeñas de texto).</li>"
            "<li>T7 del corpus es una copia exacta; no mide reutilización real.</li>"
            "<li>La calibración elegida es identidad porque con n=50 las alternativas empeoran el ECE.</li>"
            "<li>No hay validación con RENIEC/MINEDU ni despliegue institucional.</li>"
            "<li>El sistema no rechaza documentos: solo prioriza revisión humana.</li>"
            "<li>F8 (robustez, kappa) sigue parcial; F0 no se verificó en las 3 PCs.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with tabs[3]:
        rows = "".join(
            f'<div class="glossary-row"><div class="glossary-term">{html.escape(phase)}</div>'
            f'<div class="glossary-def"><strong>{html.escape(title)}</strong><br>{html.escape(status)}</div></div>'
            for phase, title, status in PHASES
        )
        st.markdown(rows, unsafe_allow_html=True)
        st.caption("Versión 1.0 · prototipo de investigación aplicada · 2026")


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Verifica CETPRO · Verificación documental",
        page_icon="V",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )
    dark = bool(st.session_state.get("dark_mode", True))
    st.markdown(f"<style>{_theme_css(dark)}</style>", unsafe_allow_html=True)

    sample_rows = _load_samples(str(SAMPLE_INDEX))
    v2_rows = _load_v2_samples(str(V2_ROOT))
    data, filename, selected_row, registry_rows, view = _sidebar(sample_rows, v2_rows)

    if view == "Historial":
        _history_page()
    elif view == "Ayuda":
        _help_page()
    else:
        _analysis_page(data, filename, selected_row, registry_rows)
    _footer()


if __name__ == "__main__":
    main()
