"""Demo Streamlit — capa de presentación del prototipo.

Fase F3: selección de una muestra del corpus local o subida de un documento,
calidad, forense clásico, OCR, extracción, reglas y cruce con registro. La
fusión calibrada y la explicación con LLM llegan en F6-F7.

La banda que se muestra es heurística y no calibrada: sirve para priorizar
revisión humana, no para rechazar documentos ni para afirmar probabilidades.
"""

from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from fraude.core.schemas import Observation, RuleFinding, Severity, SignalScore
from fraude.eval.baseline import SEVERITY_WEIGHTS, score_findings
from fraude.pipeline.run import analyze_bytes
from fraude.registry.cross import RegistryEntry
from fraude.rules.engine import REQUIRED_FIELDS

SAMPLE_ROOT = Path("data/muestras")
SAMPLE_INDEX = SAMPLE_ROOT / "_indice.csv"
SAMPLE_PEOPLE = SAMPLE_ROOT / "_personas.csv"
F4_CHECKPOINT = Path("artifacts/models/f4/fold_2.pt")
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

SEVERITY_STYLE = {
    Severity.HIGH: ("Alta", "#B3261E", "#FDECEA"),
    Severity.MEDIUM: ("Media", "#B26A00", "#FFF4E5"),
    Severity.LOW: ("Baja", "#1B7F3B", "#EAF6EE"),
}

SIGNAL_LABELS = {
    "metadata": "Metadatos",
    "localization": "Copy-move",
    "ela": "ELA",
    "noise": "Ruido",
    "chromatic": "Cromática",
    "jpeg": "JPEG",
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
    "fecha_evaluacion": "Fecha de evaluación",
    "horas": "Horas",
    "nota": "Nota",
    "resolucion": "Resolución",
    "seccion": "Sección",
}

FINDING_HELP = {
    "CAMPOS_FALTANTES": "El OCR no encontró uno o más campos obligatorios para este tipo documental.",
    "FECHA_INVALIDA": "Una fecha extraída no se pudo interpretar; puede ser un error de OCR o una alteración.",
    "FECHA_FUTURA": "Una fecha del documento es posterior a la fecha de análisis.",
    "EDAD_IMPLAUSIBLE": "La fecha de nacimiento produce una edad fuera del rango plausible.",
    "ORDEN_FECHAS": "Las fechas no siguen el orden esperado entre nacimiento, emisión y vencimiento.",
    "NOMBRE_INVALIDO": "Un campo de nombre contiene números o es demasiado corto.",
    "NOMBRE_INCONSISTENTE": "El nombre del titular no coincide entre campos del mismo documento.",
    "DNI_INVALIDO": "El DNI no tiene exactamente 8 dígitos.",
    "DNI_DUPLICADO": "El mismo DNI aparece más de una vez en el documento.",
    "PROGRAMA_DESCONOCIDO": "El programa no coincide con el catálogo, aun tolerando variantes de OCR.",
    "MODULO_DESCONOCIDO": "El módulo no coincide con el catálogo, aun tolerando variantes de OCR.",
    "PERIODO_INVALIDO": "El periodo no tiene el formato AAAA-I/II.",
    "RESOLUCION_INVALIDA": "La resolución no tiene el formato esperado.",
    "NOTA_FUERA_DE_RANGO": "Una nota está fuera de la escala vigesimal 0-20.",
    "PROMEDIO_FUERA_DE_RANGO": "El promedio está fuera de la escala vigesimal 0-20.",
    "HORAS_IMPLAUSIBLES": "La carga horaria está fuera del rango plausible.",
    "CALIDAD_BAJA": "La imagen no cumple los mínimos de calidad; la salida correcta es recapturar.",
    "OCR_BAJO": "El OCR reconoció poco texto o con baja confianza.",
    "METADATA_EDICION": "Los metadatos del archivo conservan el nombre de un editor gráfico.",
    "COPY_MOVE": "Se detectaron regiones duplicadas compatibles con copy-move.",
    "REGISTRO_NO_COINCIDENTE": "La persona no coincide con el registro interno del CETPRO.",
    "REGISTRO_DIFERENCIA": "Algún campo del documento difiere del registro interno.",
    "TIPO_NO_IDENTIFICADO": "No se pudo identificar el tipo documental por el título.",
}

PHASES = [
    {"Fase": "F0", "Entregable": "Base operativa y contrato de datos", "Estado": "Cerrado"},
    {
        "Fase": "F1",
        "Entregable": "Datos, taxonomía y anotación",
        "Estado": "Cerrado con corpus sintético; sin documentos reales",
    },
    {"Fase": "F2", "Entregable": "Preproceso y dataset sintético", "Estado": "Cerrado (sintético)"},
    {
        "Fase": "F3",
        "Entregable": "Línea base sin deep learning",
        "Estado": "Cerrado provisional; 20 muestras evaluadas",
    },
    {
        "Fase": "F4",
        "Entregable": "Modelo visual de documento",
        "Estado": "Cerrado experimental: detecta T5/T6; T1-T4 quedan para F5",
    },
    {"Fase": "F5", "Entregable": "Localización y explicabilidad", "Estado": "Pendiente"},
    {"Fase": "F6", "Entregable": "Rama semántica y NLP", "Estado": "Pendiente"},
    {"Fase": "F7", "Entregable": "Fusión y calibración", "Estado": "Pendiente"},
    {"Fase": "F8", "Entregable": "Evaluación integral", "Estado": "Pendiente"},
    {
        "Fase": "F9",
        "Entregable": "Demo final y Cap. IV",
        "Estado": "En curso: demo, model card y guía de presentación",
    },
]

CSS = """
:root {
  --navy: #0B2545;
  --navy-2: #143A66;
  --gold: #C9A227;
  --ink: #16283D;
  --muted: #5C6C7C;
  --line: #DDE5EE;
  --bg: #F3F6FA;
  --card: #FFFFFF;
  --green: #1B7F3B;
  --amber: #B26A00;
  --red: #B3261E;
}
.stApp {
  background: radial-gradient(circle at 12% 0%, #FFFFFF 0%, #F4F7FB 42%, #EAF0F6 100%);
  color: var(--ink);
}
.block-container {
  max-width: 1280px;
  padding-top: 1.4rem;
  padding-bottom: 3rem;
}
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.02em; }
.hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(120deg, #0B2545 0%, #143A66 58%, #1D4E89 100%);
  border-radius: 22px;
  padding: 2.1rem 2.3rem;
  box-shadow: 0 20px 45px rgba(11, 37, 69, 0.20);
  margin-bottom: 1.4rem;
}
.hero::after {
  content: "";
  position: absolute;
  right: -70px;
  top: -80px;
  width: 280px;
  height: 280px;
  border-radius: 50%;
  background: rgba(201, 162, 39, 0.18);
}
.hero::before {
  content: "";
  position: absolute;
  right: 90px;
  bottom: -110px;
  width: 190px;
  height: 190px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.06);
}
.kicker {
  position: relative;
  z-index: 1;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 0.72rem;
  font-weight: 700;
  color: #F0D57A;
}
.hero h1 {
  position: relative;
  z-index: 1;
  color: #FFFFFF;
  font-size: 2.15rem;
  line-height: 1.12;
  margin: 0.35rem 0 0.55rem;
}
.hero h1 span { color: #F0D57A; }
.hero p {
  position: relative;
  z-index: 1;
  color: #D7E2EF;
  max-width: 820px;
  margin: 0;
  font-size: 0.98rem;
}
.badges { position: relative; z-index: 1; display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.1rem; }
.badge {
  background: rgba(255, 255, 255, 0.10);
  border: 1px solid rgba(255, 255, 255, 0.18);
  color: #FFFFFF;
  padding: 0.28rem 0.72rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  box-shadow: 0 8px 24px rgba(22, 40, 61, 0.06);
  min-height: 112px;
}
.metric-label { color: var(--muted); font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 1.65rem; font-weight: 750; line-height: 1.15; margin-top: 0.25rem; }
.metric-sub { color: var(--muted); font-size: 0.8rem; margin-top: 0.25rem; }
.score-track { position: relative; height: 11px; background: #E8EEF5; border-radius: 999px; overflow: hidden; margin: 0.7rem 0 0.25rem; }
.score-fill { height: 100%; border-radius: 999px; }
.score-threshold { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--navy); opacity: 0.55; }
.score-scale { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.7rem; }
.meta-row { display: flex; gap: 0.45rem; flex-wrap: wrap; margin: 0.8rem 0 1.2rem; }
.meta-pill { background: #EAF0F6; color: #33475B; border-radius: 999px; padding: 0.24rem 0.65rem; font-size: 0.76rem; }
.section-title { display: flex; align-items: baseline; gap: 0.6rem; margin: 1.1rem 0 0.6rem; }
.section-title h3 { margin: 0; font-size: 1.18rem; }
.section-title span { color: var(--muted); font-size: 0.82rem; }
.finding {
  border: 1px solid var(--line);
  border-left: 6px solid;
  border-radius: 13px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.65rem;
  background: #FFFFFF;
  box-shadow: 0 4px 14px rgba(22, 40, 61, 0.04);
}
.finding-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap; }
.finding-code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; font-weight: 750; }
.finding-sev { font-size: 0.72rem; font-weight: 700; padding: 0.16rem 0.52rem; border-radius: 999px; }
.finding-msg { color: var(--ink); margin-top: 0.35rem; font-size: 0.92rem; }
.finding-help { color: var(--muted); font-size: 0.8rem; margin-top: 0.3rem; }
.empty-state {
  background: #FFFFFF;
  border: 1px dashed #C7D2DF;
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  color: var(--muted);
  font-size: 0.9rem;
}
.signal-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; }
.signal { background: #FFFFFF; border: 1px solid var(--line); border-radius: 13px; padding: 0.8rem 0.9rem; }
.signal-name { color: var(--muted); font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
.signal-score { font-size: 1.25rem; font-weight: 750; color: var(--navy); margin-top: 0.1rem; }
.signal-note { color: var(--muted); font-size: 0.72rem; margin-top: 0.25rem; }
.signal-bar { height: 6px; background: #E8EEF5; border-radius: 999px; overflow: hidden; margin-top: 0.45rem; }
.signal-fill { height: 100%; border-radius: 999px; }
.field-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.field-table th { text-align: left; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid var(--line); padding: 0.45rem 0.5rem; }
.field-table td { padding: 0.45rem 0.5rem; border-bottom: 1px solid #EEF2F7; vertical-align: top; }
.field-table td:first-child { color: var(--muted); width: 34%; }
.pill { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 999px; font-size: 0.74rem; font-weight: 700; }
.pill-green { background: #EAF6EE; color: var(--green); }
.pill-amber { background: #FFF4E5; color: var(--amber); }
.pill-red { background: #FDECEA; color: var(--red); }
.pill-gray { background: #EAF0F6; color: #33475B; }
.small-note { color: var(--muted); font-size: 0.8rem; }
.sidebar-brand {
  background: linear-gradient(135deg, #0B2545 0%, #143A66 100%);
  color: #FFFFFF;
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
  box-shadow: 0 10px 24px rgba(11, 37, 69, 0.18);
}
.sidebar-brand .brand-kicker { color: #F0D57A; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; }
.sidebar-brand .brand-title { font-size: 1.05rem; font-weight: 750; margin-top: 0.2rem; }
.sidebar-brand .brand-sub { color: #D7E2EF; font-size: 0.78rem; margin-top: 0.25rem; }
.phase-row { display: grid; grid-template-columns: 52px 1fr 1.1fr; gap: 0.7rem; align-items: center; background: #FFFFFF; border: 1px solid var(--line); border-radius: 11px; padding: 0.55rem 0.75rem; margin-bottom: 0.4rem; }
.phase-id { font-weight: 750; color: var(--navy); }
.phase-name { font-size: 0.88rem; }
.phase-status { font-size: 0.78rem; color: var(--muted); }
@media (max-width: 900px) {
  .signal-grid { grid-template-columns: 1fr 1fr; }
  .hero h1 { font-size: 1.7rem; }
}
"""


@dataclass(frozen=True)
class RiskBand:
    """Banda heurística de presentación; no es una probabilidad calibrada."""

    key: str
    label: str
    color: str
    icon: str
    note: str


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
def _analyze_cached(
    data: bytes,
    filename: str,
    registry_rows: tuple[tuple[str, ...], ...],
    today_iso: str,
) -> dict[str, Any]:
    registry = [_entry_from_row(row) for row in registry_rows] or None
    observation = analyze_bytes(
        data,
        filename,
        registry=registry,
        today=date.fromisoformat(today_iso),
    )
    return observation.model_dump(mode="json")


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


def _risk_band(observation: Observation, score: float) -> RiskBand:
    if not observation.quality.ok:
        return RiskBand(
            key="indeterminado",
            label="Indeterminado",
            color="#5F6B7A",
            icon="⚪",
            note="La imagen no es apta; se recomienda recapturar.",
        )
    if observation.ocr.engine in {"unavailable", "not_run", "disabled"}:
        if score >= DEFAULT_THRESHOLD:
            return RiskBand(
                key="alto_forense",
                label="Alto",
                color="#B3261E",
                icon="🔴",
                note="Solo forense: no se pudo leer el contenido.",
            )
        return RiskBand(
            key="indeterminado",
            label="Indeterminado",
            color="#5F6B7A",
            icon="⚪",
            note="No se pudo leer el contenido del documento.",
        )
    if score >= DEFAULT_THRESHOLD:
        return RiskBand(
            key="alto",
            label="Alto",
            color="#B3261E",
            icon="🔴",
            note="Priorizar revisión humana. No es una probabilidad.",
        )
    if score >= 0.25:
        return RiskBand(
            key="medio",
            label="Medio",
            color="#B26A00",
            icon="🟠",
            note="Hay señales parciales; revisar el detalle.",
        )
    return RiskBand(
        key="bajo",
        label="Bajo",
        color="#1B7F3B",
        icon="🟢",
        note="Las señales implementadas no se activaron.",
    )


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_conf(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _fmt_ms(value: int | None) -> str:
    return "—" if value is None else f"{value} ms"


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _field_label(key: str) -> str:
    indexed = re.fullmatch(r"(estudiante|dni|nota)_(\d+)", key)
    if indexed is not None:
        base, index = indexed.groups()
        return f"{FIELD_LABELS.get(base, base)} {int(index) + 1}"
    return FIELD_LABELS.get(key, key.replace("_", " ").capitalize())


def _acta_rows(fields: dict[str, Any]) -> list[tuple[int, str, str, str]]:
    indices: list[int] = []
    for key in fields:
        match = re.fullmatch(r"estudiante_(\d+)", key)
        if match is not None:
            indices.append(int(match.group(1)))
    return [
        (
            index + 1,
            _text(fields.get(f"estudiante_{index}")) or "—",
            _text(fields.get(f"dni_{index}")) or "—",
            _text(fields.get(f"nota_{index}")) or "—",
        )
        for index in sorted(set(indices))
    ]


def _signal_color(score: float | None) -> str:
    if score is None:
        return "#5F6B7A"
    if score >= 0.5:
        return "#B3261E"
    if score >= 0.25:
        return "#B26A00"
    return "#1B7F3B"


def _signal_card(name: str, signal: SignalScore) -> str:
    if not signal.applicable:
        reason = _text(signal.details.get("reason")) or "no aplica a esta captura"
        return (
            '<div class="signal">'
            f'<div class="signal-name">{html.escape(name)}</div>'
            '<div class="signal-score">No aplica</div>'
            f'<div class="signal-note">{html.escape(reason)}</div>'
            "</div>"
        )
    score = signal.score
    color = _signal_color(score)
    width = 0.0 if score is None else max(0.0, min(1.0, score)) * 100
    value = "—" if score is None else f"{score:.2f}"
    detail = _signal_detail(name, signal.details)
    return (
        '<div class="signal">'
        f'<div class="signal-name">{html.escape(name)}</div>'
        f'<div class="signal-score">{value}</div>'
        f'<div class="signal-bar"><div class="signal-fill" style="width:{width:.1f}%;'
        f'background:{color};"></div></div>'
        f'<div class="signal-note">{html.escape(detail)}</div>'
        "</div>"
    )


def _signal_detail(name: str, details: dict[str, Any]) -> str:
    if name == "Metadatos":
        software = _text(details.get("software")) or "sin software declarado"
        reasons = details.get("reasons") or []
        reason_text = ", ".join(str(reason) for reason in reasons) or "sin razones"
        return f"Software: {software} · {reason_text}"
    if name == "Copy-move":
        model = _text(details.get("model")) or "copy_move"
        rest = ", ".join(f"{key}={value}" for key, value in details.items() if key != "model")
        return f"Modelo: {model}" + (f" · {rest}" if rest else "")
    if not details:
        return "Sin detalle adicional."
    return ", ".join(f"{key}={value}" for key, value in details.items())


def _finding_card(finding: RuleFinding) -> str:
    label, color, background = SEVERITY_STYLE[finding.severity]
    help_text = FINDING_HELP.get(
        finding.code,
        "Señal generada por una regla del pipeline; revisar el contexto del documento.",
    )
    weight = SEVERITY_WEIGHTS[finding.severity]
    return (
        f'<div class="finding" style="border-left-color:{color};'
        f'background:linear-gradient(90deg,{background} 0%,#FFFFFF 48%);">'
        '<div class="finding-head">'
        f'<span class="finding-code" style="color:{color}">{html.escape(finding.code)}</span>'
        f'<span class="finding-sev" style="background:{background};color:{color};">'
        f"{label} · peso {weight:.2f}</span>"
        "</div>"
        f'<div class="finding-msg">{html.escape(finding.message)}</div>'
        f'<div class="finding-help">{html.escape(help_text)}</div>'
        "</div>"
    )


def _metric_card(label: str, value: str, sub: str, color: str, extra: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f"{extra}"
        "</div>"
    )


def _score_track(score: float, color: str) -> str:
    width = max(0.0, min(1.0, score)) * 100
    threshold = DEFAULT_THRESHOLD * 100
    return (
        '<div class="score-track">'
        f'<div class="score-fill" style="width:{width:.1f}%;background:{color};"></div>'
        f'<div class="score-threshold" style="left:{threshold:.1f}%;"></div>'
        "</div>"
        '<div class="score-scale"><span>0.00</span><span>umbral 0.50</span><span>1.00</span></div>'
    )


def _fields_table(fields: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(_field_label(key))}</td>"
        f"<td>{html.escape(_text(value)) or '—'}</td>"
        "</tr>"
        for key, value in fields.items()
        if not re.fullmatch(r"(estudiante|dni|nota)_\d+", key)
    )
    if not rows:
        return '<div class="empty-state">El OCR no devolvió campos para este documento.</div>'
    return (
        '<table class="field-table"><thead><tr><th>Campo</th><th>Valor extraído</th></tr>'
        f"</thead><tbody>{rows}</tbody></table>"
    )


def _acta_table(rows: list[tuple[int, str, str, str]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(dni)}</td>"
        f"<td>{html.escape(grade)}</td>"
        "</tr>"
        for index, name, dni, grade in rows
    )
    return (
        '<table class="field-table"><thead><tr><th>#</th><th>Estudiante</th>'
        f"<th>DNI</th><th>Nota</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _missing_fields(observation: Observation) -> list[str]:
    required = REQUIRED_FIELDS.get(observation.doctype.label, ())
    return [field for field in required if not observation.fields.get(field)]


def _registry_html(observation: Observation) -> str:
    report = observation.registry
    if report.score is None:
        reason = report.diffs[0] if report.diffs else "registro no disponible"
        return (
            '<div class="empty-state">'
            "<strong>El cruce de registro no aplica.</strong><br>"
            f"{html.escape(reason)}"
            "</div>"
        )
    if report.matched:
        color = "#B26A00" if report.diffs else "#1B7F3B"
        pill = "pill-amber" if report.diffs else "pill-green"
        title = "Coincide con el registro" if not report.diffs else "Coincide con diferencias"
        detail = (
            "".join(f"<li>{html.escape(diff)}</li>" for diff in report.diffs)
            or "<li>Sin diferencias en los campos comparados.</li>"
        )
        return (
            '<div class="metric-card">'
            f'<span class="pill {pill}">{title}</span>'
            f'<div class="metric-value" style="color:{color};font-size:1.35rem;">'
            f"score {report.score:.2f}</div>"
            f'<ul class="small-note">{detail}</ul>'
            "</div>"
        )
    return (
        '<div class="metric-card">'
        '<span class="pill pill-red">Sin coincidencia suficiente</span>'
        f'<div class="metric-value" style="color:#B3261E;font-size:1.35rem;">'
        f"mejor score {report.score:.2f}</div>"
        f'<div class="metric-sub">{html.escape("; ".join(report.diffs[:3]))}</div>'
        "</div>"
    )


def _quality_rows(observation: Observation) -> list[tuple[str, str]]:
    quality = observation.quality
    capture = CAPTURE_LABELS.get(observation.capture.type.value, observation.capture.type.value)
    return [
        ("Imagen analizable", "Sí" if quality.ok else "No"),
        ("Desenfoque (varianza)", "—" if quality.blur is None else f"{quality.blur:.1f}"),
        ("Inclinación", "—" if quality.skew_deg is None else f"{quality.skew_deg:.2f}°"),
        ("Tipo de captura", capture),
        (
            "DPI estimado",
            "—" if observation.capture.dpi_est is None else str(observation.capture.dpi_est),
        ),
        ("Motor OCR", observation.ocr.engine),
        ("Palabras reconocidas", str(observation.ocr.n_words)),
        ("Confianza media OCR", _fmt_conf(observation.ocr.mean_conf)),
    ]


def _timing_rows(observation: Observation) -> list[tuple[str, str]]:
    return [
        ("Forense", _fmt_ms(observation.timings.get("forensics_ms"))),
        ("OCR", _fmt_ms(observation.timings.get("ocr_ms"))),
        ("Reglas", _fmt_ms(observation.timings.get("rules_ms"))),
        ("Total", _fmt_ms(sum(observation.timings.values()) if observation.timings else None)),
    ]


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return (
        '<table class="field-table"><thead><tr><th>Métrica</th><th>Valor</th></tr>'
        f"</thead><tbody>{body}</tbody></table>"
    )


def _hero() -> None:
    st.markdown(
        '<div class="hero">'
        '<div class="kicker">Prototipo de investigación · Fase F3</div>'
        "<h1>Verificador documental <span>CETPRO San Agustín</span></h1>"
        "<p>Calidad de imagen, forense clásico, OCR, extracción de campos, reglas "
        "semánticas y cruce con registro. Cada resultado se acompaña de evidencias "
        "explicables y prioriza la revisión humana.</p>"
        '<div class="badges">'
        '<span class="badge">Corpus sintético de 20 documentos</span>'
        '<span class="badge">Score heurístico, no calibrado</span>'
        '<span class="badge">Sin decisión automática</span>'
        '<span class="badge">Datos locales</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _landing(sample_rows: list[dict[str, str]], people_rows: tuple[tuple[str, ...], ...]) -> None:
    st.markdown(
        '<div class="section-title"><h3>Qué analiza el sistema</h3>'
        "<span>cuatro capas, una sola salida explicable</span></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    layers = [
        (
            "01 · Calidad",
            "¿La imagen es analizable?",
            "Resolución, desenfoque, inclinación y contraste.",
        ),
        (
            "02 · Forense",
            "¿Hay señales de edición?",
            "Metadatos, ELA, ruido, cromática y copy-move.",
        ),
        ("03 · Contenido", "¿El texto es coherente?", "OCR, tipo documental, campos y reglas."),
        (
            "04 · Registro",
            "¿Coincide con el CETPRO?",
            "Cruce con el espejo sintético de estudiantes.",
        ),
    ]
    for column, (title, question, body) in zip(columns, layers, strict=True):
        with column:
            st.markdown(
                '<div class="metric-card">'
                f'<div class="metric-label">{html.escape(title)}</div>'
                f'<div class="metric-value" style="font-size:1.05rem;">{html.escape(question)}</div>'
                f'<div class="metric-sub">{html.escape(body)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title"><h3>Corpus disponible</h3>'
        "<span>documentos generados por IA en data/muestras</span></div>",
        unsafe_allow_html=True,
    )
    authentic = sum(_text(row.get("variante")) == "authentic" for row in sample_rows)
    tampered = len(sample_rows) - authentic
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            _metric_card("Documentos", str(len(sample_rows)), "PNG de 300 dpi", "#0B2545"),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            _metric_card("Auténticos", str(authentic), "etiqueta del lote", "#1B7F3B"),
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            _metric_card("Manipulados", str(tampered), "ataques T1-T8", "#B3261E"),
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            _metric_card("Registro", str(len(people_rows)), "personas ficticias", "#B26A00"),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title"><h3>Cómo leer el resultado</h3>'
        "<span>el sistema no rechaza documentos</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="empty-state">'
        "<strong>La banda es heurística.</strong> Se calcula sumando pesos por severidad "
        "de las reglas activadas; el umbral de referencia es 0.50. No es una probabilidad "
        "calibrada y no sustituye la revisión de una persona. "
        "<br><br><strong>Una señal no es una condena.</strong> Los metadatos de edición, "
        "los campos faltantes o una diferencia de registro deben interpretarse junto con "
        "el resto de evidencias. La calibración y la fusión llegan en F6-F7."
        "</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Selecciona una muestra en la barra lateral o sube un PNG/JPG para comenzar.",
        icon="👈",
    )


def _phases_tab() -> None:
    st.markdown(
        '<div class="section-title"><h3>Fases del proyecto</h3>'
        "<span>sin fechas: el cierre es antes de terminar el ciclo</span></div>",
        unsafe_allow_html=True,
    )
    for phase in PHASES:
        status = str(phase["Estado"])
        lower = status.lower()
        if "cerrado" in lower:
            pill = "pill-green"
        elif "parcial" in lower or "provisional" in lower:
            pill = "pill-amber"
        elif "no iniciado" in lower or "bloqueado" in lower:
            pill = "pill-red"
        else:
            pill = "pill-gray"
        st.markdown(
            '<div class="phase-row">'
            f'<div class="phase-id">{html.escape(str(phase["Fase"]))}</div>'
            f'<div class="phase-name">{html.escape(str(phase["Entregable"]))}</div>'
            f'<div class="phase-status"><span class="pill {pill}">{html.escape(status)}</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "F4 se entrenó con 50 documentos sintéticos (25/25) y validación cruzada agrupada "
        "de 5 vías. Detecta T5/T6 y falla T1-T4, como se esperaba de un clasificador de "
        "página. Sigue siendo una prueba de viabilidad: no hay documentos reales y las "
        "cifras no estiman generalización institucional."
    )


@st.cache_resource(show_spinner=False)
def _load_f4_model(checkpoint: str) -> Any:
    from fraude.models.infer import load_checkpoint

    return load_checkpoint(Path(checkpoint))


def _f4_tab(data: bytes | None, filename: str) -> None:
    """Panel experimental del clasificador visual de página (F4)."""
    st.markdown(
        '<div class="section-title"><h3>F4 · modelo visual de página</h3>'
        "<span>experimental, entrenado con 50 documentos sintéticos</span></div>",
        unsafe_allow_html=True,
    )
    checkpoint = F4_CHECKPOINT
    if not checkpoint.is_file():
        st.info(
            "Todavía no hay un checkpoint F4 en `artifacts/models/f4/`. "
            "Entrena el modelo y vuelve a cargar la demo."
        )
        st.code('make train-f4 ARGS="--epochs 12 --device cpu"', language="bash")
        return
    if data is None:
        st.info("Selecciona una muestra o sube un documento en la barra lateral.")
        return

    import io

    from PIL import Image

    from fraude.models.infer import explain_image, predict_image

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        model, config = _load_f4_model(str(checkpoint))
        image_size = int(config.get("image_size", 224))
        score = predict_image(model, image, image_size=image_size)
        heatmap, heat_score, _ = explain_image(model, image, image_size=image_size)
    except Exception as exc:  # pragma: no cover - depende del checkpoint local
        st.error(f"No se pudo ejecutar F4: {exc}")
        return

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(image, caption=filename or "documento", use_container_width=True)
    with right:
        band = (
            "Indicio visual (score ≥ 0.50)" if score >= DEFAULT_THRESHOLD else "Sin indicio visual"
        )
        st.metric("Score F4", f"{score:.3f}", band)
        st.caption(
            "Salida de un EfficientNet-B0 de página entrenado con el corpus v2 "
            "(checkpoint del fold 2, el de mejor PR-AUC de validación). "
            "No es una probabilidad calibrada y no reemplaza la revisión humana. "
            "T7/T8 quedan fuera de esta rama visual. Las muestras del menú son "
            "del corpus v1: para la demo F4 conviene subir una imagen de "
            "`data/synthetic/v2/images/`."
        )
        st.image(
            heatmap,
            caption=f"Grad-CAM · score {heat_score:.3f}",
            use_container_width=True,
        )
    st.caption(
        "El mapa de calor es una ayuda cualitativa: F4 clasifica la página completa, "
        "no localiza la manipulación. La localización formal es F5."
    )


def _render_result(
    observation: Observation,
    data: bytes,
    filename: str,
    selected_row: dict[str, str] | None,
) -> None:
    score = score_findings(observation.rules)
    band = _risk_band(observation, score)
    doctype = DOCUMENT_LABELS.get(observation.doctype.label, observation.doctype.label)
    confidence = observation.doctype.confidence
    total_ms = sum(observation.timings.values()) if observation.timings else 0

    col_band, col_score, col_type, col_ocr = st.columns([1.15, 1.2, 1.25, 1.15])
    with col_band:
        st.markdown(
            _metric_card(
                "Banda heurística",
                f"{band.icon} {band.label}",
                html.escape(band.note),
                band.color,
            ),
            unsafe_allow_html=True,
        )
    with col_score:
        st.markdown(
            _metric_card(
                "Score heurístico F3",
                f"{score:.2f}",
                "suma de pesos · no es probabilidad",
                band.color,
                _score_track(score, band.color),
            ),
            unsafe_allow_html=True,
        )
    with col_type:
        st.markdown(
            _metric_card(
                "Tipo documental",
                html.escape(doctype),
                f"confianza {_fmt_conf(confidence)}" if confidence else "sin identificar",
                "#0B2545",
            ),
            unsafe_allow_html=True,
        )
    with col_ocr:
        st.markdown(
            _metric_card(
                "OCR",
                _fmt_conf(observation.ocr.mean_conf),
                f"{observation.ocr.n_words} palabras · {observation.ocr.engine}",
                "#0B2545",
            ),
            unsafe_allow_html=True,
        )

    capture = CAPTURE_LABELS.get(observation.capture.type.value, observation.capture.type.value)
    meta = [
        f"ID {observation.document_id[7:19]}…",
        _human_size(observation.source.bytes),
        f"Captura: {capture}",
        f"DPI: {observation.capture.dpi_est}" if observation.capture.dpi_est else "DPI: —",
        f"Análisis: {date.today():%d/%m/%Y}",
        f"Tiempo: {total_ms} ms" if total_ms else "Tiempo: —",
    ]
    st.markdown(
        '<div class="meta-row">'
        + "".join(f'<span class="meta-pill">{html.escape(item)}</span>' for item in meta)
        + "</div>",
        unsafe_allow_html=True,
    )

    preview_col, summary_col = st.columns([1.05, 1.55], gap="large")
    with preview_col, st.container(border=True):
        suffix = Path(filename).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            st.image(data, width="stretch")
        else:
            st.info("Vista previa no disponible para este formato.")
        st.caption(filename)

    with summary_col:
        st.markdown(
            '<div class="section-title"><h3>Evidencias</h3>'
            "<span>señales explicables, no una decisión automática</span></div>",
            unsafe_allow_html=True,
        )
        if observation.rules:
            for finding in observation.rules:
                st.markdown(_finding_card(finding), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-state">'
                "<strong>Sin hallazgos en las reglas actuales.</strong><br>"
                "Esto no certifica autenticidad: solo indica que las señales implementadas "
                "no se activaron en esta muestra."
                "</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "El score suma pesos fijos por severidad (alta 0.60, media 0.25, baja 0.05) "
            "y se trunca a 1.00. No está calibrado."
        )

    tabs = st.tabs(["Campos extraídos", "Señales forenses", "Registro", "Calidad y OCR", "JSON"])

    with tabs[0]:
        missing = _missing_fields(observation)
        if missing:
            chips = "".join(
                f'<span class="pill pill-amber">{html.escape(_field_label(field))}</span>'
                for field in missing
            )
            st.markdown(
                '<div class="meta-row">'
                '<span class="pill pill-amber">Campos obligatorios faltantes</span>'
                f"{chips}</div>",
                unsafe_allow_html=True,
            )
        acta = _acta_rows(observation.fields)
        if acta:
            st.markdown("**Tabla del acta**")
            st.markdown(_acta_table(acta), unsafe_allow_html=True)
        st.markdown("**Campos del titular y del documento**")
        st.markdown(_fields_table(observation.fields), unsafe_allow_html=True)
        st.caption(
            "Los campos provienen del OCR real. Los errores de lectura se muestran tal cual; "
            "no se corrigen con coincidencias difusas."
        )

    with tabs[1]:
        forensics = observation.forensics
        signals = [
            ("Metadatos", forensics.metadata),
            ("Copy-move", forensics.localization),
            ("ELA", forensics.ela),
            ("Ruido", forensics.noise),
            ("Cromática", forensics.chromatic),
            ("JPEG", forensics.jpeg),
        ]
        cards = "".join(_signal_card(name, signal) for name, signal in signals)
        st.markdown(f'<div class="signal-grid">{cards}</div>', unsafe_allow_html=True)
        st.caption(
            "Metadatos y copy-move alimentan las reglas. ELA, ruido, cromática y JPEG son "
            "señales exploratorias: no se usan en el score F3."
        )

    with tabs[2]:
        st.markdown(_registry_html(observation), unsafe_allow_html=True)
        st.caption(
            "El registro es un espejo sintético de las personas ficticias del corpus. "
            "No es el registro institucional del CETPRO. Las actas con varios estudiantes "
            "no se cruzan contra un único titular."
        )

    with tabs[3]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Calidad y captura**")
            st.markdown(_kv_table(_quality_rows(observation)), unsafe_allow_html=True)
        with right:
            st.markdown("**Tiempos**")
            st.markdown(_kv_table(_timing_rows(observation)), unsafe_allow_html=True)
        if observation.quality.issues:
            issues = ", ".join(
                QUALITY_LABELS.get(issue, issue) for issue in observation.quality.issues
            )
            st.warning(f"Incidencias de calidad: {issues}. La salida correcta es recapturar.")
        else:
            st.success("La imagen pasó la puerta de calidad.")
        st.caption("La puerta de calidad no acusa fraude: solo indica si la imagen es analizable.")

    with tabs[4]:
        st.code(observation.model_dump_json(indent=2), language="json")
        st.download_button(
            "Descargar JSON",
            data=observation.model_dump_json(indent=2),
            file_name=f"{observation.document_id[7:19]}.json",
            mime="application/json",
        )

    if selected_row is not None:
        _ground_truth(observation, selected_row, score, band)


def _ground_truth(
    observation: Observation,
    row: dict[str, str],
    score: float,
    band: RiskBand,
) -> None:
    with st.expander("Etiqueta del corpus (solo evaluación)", expanded=False):
        variant = _text(row.get("variante"))
        attack = _text(row.get("ataque"))
        expected_type = _text(row.get("tipo"))
        extracted_type = observation.doctype.label
        if variant == "authentic":
            st.markdown("**Etiqueta:** auténtico")
        else:
            st.markdown(f"**Etiqueta:** manipulado · ataque `{attack}`")
        st.markdown(
            f"**Persona del corpus:** `{html.escape(_text(row.get('persona')))}` · "
            f"**Tipo esperado:** {html.escape(DOCUMENT_LABELS.get(expected_type, expected_type))}"
        )
        original = _text(row.get("campo_original"))
        altered = _text(row.get("campo_alterado"))
        if original or altered:
            st.markdown(f"**Campo alterado:** `{html.escape(original)}` → `{html.escape(altered)}`")
        if extracted_type == expected_type:
            st.success("El tipo documental extraído coincide con la etiqueta.")
        else:
            st.warning(
                f"El tipo extraído (`{extracted_type}`) no coincide con el esperado "
                f"(`{expected_type}`)."
            )
        if observation.quality.ok and observation.ocr.engine not in {
            "unavailable",
            "not_run",
            "disabled",
        }:
            predicted_positive = band.key in {"alto", "alto_forense"}
            actual_positive = variant != "authentic"
            if predicted_positive == actual_positive:
                st.success(
                    f"La banda heurística ({band.label}, score {score:.2f}) coincide con la etiqueta."
                )
            else:
                st.warning(
                    f"La banda heurística ({band.label}, score {score:.2f}) no coincide con la etiqueta."
                )
        else:
            st.info("Sin OCR o calidad suficiente: no se compara automáticamente con la etiqueta.")
        st.caption(
            "Esta comparación solo tiene sentido para el lote sintético de 20 muestras; "
            "no es una estimación de desempeño institucional."
        )


def _sidebar(
    sample_rows: list[dict[str, str]],
    people_rows: tuple[tuple[str, ...], ...],
) -> tuple[bytes | None, str, dict[str, str] | None, tuple[tuple[str, ...], ...]]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="brand-kicker">CETPRO San Agustín</div>'
            '<div class="brand-title">Verificador documental</div>'
            '<div class="brand-sub">Prototipo F3 · corpus sintético</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        source = st.radio(
            "Origen del documento",
            ("Muestra del corpus", "Subir archivo"),
            help="Las muestras son los 20 documentos generados por IA en data/muestras.",
        )

        data: bytes | None = None
        filename = ""
        selected_row: dict[str, str] | None = None

        if source == "Muestra del corpus":
            if sample_rows:
                labels = [
                    f"{row['id']} · {DOCUMENT_LABELS.get(row['tipo'], row['tipo'])}"
                    for row in sample_rows
                ]
                selected = st.selectbox(
                    "Muestra",
                    labels,
                    help="Selecciona un documento del lote para analizarlo con el pipeline completo.",
                )
                selected_row = sample_rows[labels.index(selected)]
                path = SAMPLE_ROOT / _text(selected_row.get("png"))
                if path.is_file():
                    data = path.read_bytes()
                    filename = path.name
                else:
                    st.error(f"No se encontró la imagen: {path}")
            else:
                st.warning(
                    "No se encontró data/muestras. Sube un archivo o genera el corpus "
                    "con `uv run python data/muestras/_generador.py`."
                )
        else:
            uploaded = st.file_uploader(
                "Imagen del documento",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
                help="Formatos aceptados: PNG, JPG, TIFF, BMP y WEBP.",
            )
            if uploaded is not None:
                data = uploaded.getvalue()
                filename = uploaded.name

        use_registry = st.toggle(
            "Cruzar con registro sintético",
            value=True,
            help="Compara los campos extraídos con las 12 personas ficticias de _personas.csv.",
        )
        registry_rows = people_rows if use_registry else ()

        st.divider()
        st.markdown("**Lectura del resultado**")
        st.markdown(
            '<span class="pill pill-green">Bajo</span> '
            '<span class="pill pill-amber">Medio</span> '
            '<span class="pill pill-red">Alto</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "La banda suma pesos por severidad y usa 0.50 como referencia. "
            "No es una probabilidad calibrada."
        )
        st.markdown("**Principios**")
        st.markdown(
            "- Prioriza revisión humana, nunca rechaza.\n"
            "- Toda señal debe ser explicable.\n"
            "- Los datos no salen del entorno local."
        )
        st.divider()
        st.caption("Fuente de verdad: `plan-sistema-fraude-documental.md`")

    return data, filename, selected_row, registry_rows


def main() -> None:
    st.set_page_config(
        page_title="Verificador documental — CETPRO San Agustín",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    _hero()

    sample_rows = _load_samples(str(SAMPLE_INDEX))
    people_rows = _load_people(str(SAMPLE_PEOPLE))
    data, filename, selected_row, registry_rows = _sidebar(sample_rows, people_rows)

    analysis_tab, f4_tab, phases_tab = st.tabs(
        ["Análisis", "F4 experimental", "Fases del proyecto"]
    )

    with analysis_tab:
        if data is None:
            _landing(sample_rows, people_rows)
        else:
            with st.spinner("Analizando: calidad, forense, OCR, reglas y registro…"):
                payload = _analyze_cached(
                    data,
                    filename,
                    registry_rows,
                    date.today().isoformat(),
                )
            observation = Observation.model_validate(payload)
            _render_result(observation, data, filename, selected_row)

    with f4_tab:
        _f4_tab(data, filename)

    with phases_tab:
        _phases_tab()


if __name__ == "__main__":
    main()
