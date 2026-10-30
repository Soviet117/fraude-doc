"""Demo Streamlit — capa de presentación del prototipo.

Fase F3: subida de documento, identificación por hash, calidad, forense clásico,
OCR opcional, extracción, reglas y cruce con registro. La fusión calibrada y la
explicación llegan en F6-F7.
"""

from __future__ import annotations

import streamlit as st

from fraude.pipeline.run import analyze_bytes

RISK_STYLE: dict[str, tuple[str, str]] = {
    "bajo": ("#1B7F3B", "🟢"),
    "medio": ("#B26A00", "🟠"),
    "alto": ("#B3261E", "🔴"),
    "indeterminado": ("#5F6B7A", "⚪"),
}

PHASES = [
    {"Fase": "F0", "Entregable": "Base operativa y contrato de datos", "Estado": "Cerrado"},
    {
        "Fase": "F1",
        "Entregable": "Datos, taxonomía y anotación",
        "Estado": "Parcial: faltan documentos reales",
    },
    {"Fase": "F2", "Entregable": "Preproceso y dataset sintético", "Estado": "Cerrado (sintético)"},
    {
        "Fase": "F3",
        "Entregable": "Línea base sin deep learning",
        "Estado": "Cerrado provisional",
    },
    {
        "Fase": "F4",
        "Entregable": "Modelo visual de documento",
        "Estado": "Bloqueado: requiere documentos reales",
    },
    {"Fase": "F5", "Entregable": "Localización y explicabilidad", "Estado": "Pendiente"},
    {"Fase": "F6", "Entregable": "Rama semántica y NLP", "Estado": "Pendiente"},
    {"Fase": "F7", "Entregable": "Fusión y calibración", "Estado": "Pendiente"},
    {"Fase": "F8", "Entregable": "Evaluación integral", "Estado": "Pendiente"},
    {"Fase": "F9", "Entregable": "Demo final y Cap. IV", "Estado": "Pendiente"},
]


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


st.set_page_config(
    page_title="Verificador documental — CETPRO San Agustín",
    page_icon="🔎",
    layout="wide",
)

st.title("Verificador documental")
st.caption("CETPRO San Agustín · Nueva Cajamarca · Prototipo de investigación (8.º ciclo)")

with st.sidebar:
    st.subheader("Estado del prototipo")
    st.info(
        "**Fase F3 (provisional)**\n\n"
        "Calidad, forense clásico, OCR opcional, extracción, reglas y registro. "
        "La fusión calibrada y la explicación llegan en F6-F7."
    )
    st.markdown("**Principios del sistema**")
    st.markdown(
        "- Prioriza revisión humana, nunca rechaza.\n"
        "- Toda señal debe ser explicable.\n"
        "- Los datos no salen del entorno local."
    )
    st.divider()
    st.caption("Fuente de verdad: `plan-sistema-fraude-documental.md`")

analysis_tab, phases_tab = st.tabs(["Análisis", "Fases del proyecto"])

with analysis_tab:
    st.subheader("1. Carga del documento")
    uploaded = st.file_uploader(
        "Selecciona la imagen o PDF del documento",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp", "pdf"],
        help="Formatos aceptados: PNG, JPG, TIFF, WEBP y PDF.",
    )

    if uploaded is None:
        st.markdown("#### Cómo funciona")
        cols = st.columns(4)
        steps = [
            ("1. Calidad", "¿La imagen es analizable?"),
            ("2. Forense", "¿La imagen fue manipulada?"),
            ("3. Contenido", "¿El texto es coherente?"),
            ("4. Registro", "¿Coincide con los datos?"),
        ]
        for col, (title, body) in zip(cols, steps, strict=True):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
        st.info("Sube un documento para generar su identificador y el contrato de datos.")
    else:
        observation = analyze_bytes(uploaded.getvalue(), uploaded.name)
        color, icon = RISK_STYLE[observation.fusion.risk.value]

        st.subheader("2. Resultado")
        col_risk, col_hash, col_size, col_time = st.columns([1.2, 2.2, 1, 1])
        with col_risk:
            st.markdown(
                f"<div style='border-left:6px solid {color}; padding:0.6rem 1rem; "
                f"background:#F4F6F8; border-radius:6px;'>"
                f"<div style='font-size:0.85rem; color:#5F6B7A;'>Nivel de riesgo</div>"
                f"<div style='font-size:1.6rem; font-weight:600; color:{color};'>"
                f"{icon} {observation.fusion.risk.value.title()}</div></div>",
                unsafe_allow_html=True,
            )
        col_hash.metric("Identificador", observation.document_id[:23] + "…")
        col_size.metric("Tamaño", human_size(observation.source.bytes))
        total_ms = sum(observation.timings.values())
        col_time.metric("Tiempo", f"{total_ms} ms" if total_ms else "—")

        if observation.source.format == "pdf":
            st.info(
                "PDF: por ahora solo se calcula el hash. El render de páginas se "
                "incorpora más adelante.",
                icon="📄",
            )
        elif observation.ocr.engine == "unavailable":
            st.info(
                "docTR no está instalado en este entorno. Se ejecutaron calidad y "
                "forense; los campos y las reglas semánticas quedan pendientes.",
                icon="📌",
            )
        elif observation.ocr.engine == "not_run":
            st.warning(
                "La imagen no es legible o no es analizable; se recomienda recapturar.",
                icon="⚠️",
            )
        else:
            st.caption(
                f"OCR: `{observation.ocr.engine}` · "
                f"{observation.ocr.n_words} palabras · "
                f"confianza media: {observation.ocr.mean_conf}"
            )

        st.subheader("3. Evidencias")
        if observation.rules:
            st.dataframe(
                [
                    {
                        "Código": finding.code,
                        "Severidad": finding.severity.value,
                        "Mensaje": finding.message,
                    }
                    for finding in observation.rules
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Sin hallazgos de reglas para esta muestra.")
        st.caption(
            "La fusión calibrada (F7) convertirá estos hallazgos en un nivel de "
            "riesgo explicable; por ahora el riesgo es indeterminado."
        )

        with st.expander("Contrato `Observation` (JSON)", expanded=False):
            st.code(observation.model_dump_json(indent=2), language="json")

        st.download_button(
            "Descargar JSON",
            data=observation.model_dump_json(indent=2),
            file_name=f"{observation.document_id[7:19]}.json",
            mime="application/json",
        )

with phases_tab:
    st.subheader("Fases del proyecto")
    st.caption("Sin fechas: el cierre es antes de terminar el ciclo. Ver plan §8.")
    st.dataframe(PHASES, hide_index=True, use_container_width=True)
