"""Explicaciones en español con LLM local y fallback por plantilla (F6).

El LLM nunca decide: solo redacta a partir de hallazgos estructurados. Si no
está disponible, tarda o devuelve algo vacío, la plantilla sigue funcionando.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from fraude.core.schemas import Observation
from fraude.nlp.semantic import SemanticReport

LOGGER = logging.getLogger("fraude.nlp.explain")

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:4b-instruct-2507"
DEFAULT_TIMEOUT = 8.0
DISCLAIMER = "El sistema no rechaza documentos ni emite una decisión automática."


@dataclass(frozen=True)
class Explanation:
    text: str
    source: str
    fallback_reason: str | None = None
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
            "latency_ms": self.latency_ms,
        }


def build_context(
    observation: Observation,
    *,
    semantic: SemanticReport | None = None,
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Resume la observación en datos que el LLM o la plantilla pueden citar."""
    findings = [
        {
            "code": finding.code,
            "severity": finding.severity.value,
            "message": finding.message,
        }
        for finding in observation.rules
    ]
    return {
        "document_type": observation.doctype.label,
        "quality_ok": observation.quality.ok,
        "ocr_mean_conf": observation.ocr.mean_conf,
        "findings": findings,
        "registry": {
            "matched": observation.registry.matched,
            "score": observation.registry.score,
            "diffs": list(observation.registry.diffs),
        },
        "forensics": {
            "ela": observation.forensics.ela.score,
            "noise": observation.forensics.noise.score,
            "chromatic": observation.forensics.chromatic.score,
            "jpeg": observation.forensics.jpeg.score,
            "metadata": observation.forensics.metadata.score,
            "localization": observation.forensics.localization.score,
        },
        "semantic": semantic.to_dict() if semantic is not None else None,
        "f4_score": f4_score,
        "f5_summary": f5_summary,
        "disclaimer": DISCLAIMER,
    }


def template_explanation(context: dict[str, Any]) -> str:
    """Fallback determinista, sin LLM, con los hallazgos estructurados."""
    lines = ["Indicios para revisión humana:"]
    findings = list(context.get("findings") or [])
    for finding in findings[:5]:
        lines.append(f"- [{finding['severity']}] {finding['message']}")

    registry = context.get("registry") or {}
    if registry.get("diffs"):
        lines.append(
            "- El documento difiere del registro interno: " + "; ".join(registry["diffs"][:3])
        )
    elif registry.get("matched") is False and registry.get("score") is not None:
        lines.append("- La persona no coincide con el registro interno del CETPRO.")

    semantic = context.get("semantic") or {}
    for note in list(semantic.get("notes") or [])[:3]:
        lines.append(f"- {note}")

    f4_score = context.get("f4_score")
    if f4_score is not None:
        if float(f4_score) >= 0.5:
            lines.append(
                f"- La señal visual F4 es alta ({float(f4_score):.3f}); "
                "es un indicio sin calibrar, no una probabilidad de fraude."
            )
        else:
            lines.append(f"- La señal visual F4 es baja ({float(f4_score):.3f}).")

    f5_summary = context.get("f5_summary") or {}
    if f5_summary.get("max_prob") is not None and float(f5_summary["max_prob"]) >= 0.5:
        lines.append(
            "- F5 localiza una región con probabilidad "
            f"{float(f5_summary['max_prob']):.3f}; el mapa es una ayuda cualitativa."
        )

    if len(lines) == 1:
        lines.append("- No se encontraron indicios automáticos; esto no certifica autenticidad.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def llm_explanation(
    context: dict[str, Any],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Explanation:
    """Intenta redactar con Ollama; ante cualquier fallo, usa la plantilla."""
    prompt = (
        "/no_think\n"
        "Eres un asistente que redacta explicaciones para revisión humana de "
        "documentos académicos. No determines fraude, autenticidad ni rechazo. "
        "Usa solo los hallazgos estructurados. Responde en español, en 3 a 5 "
        "líneas, sin inventar datos.\n\n"
        f"Hallazgos: {json.dumps(context, ensure_ascii=False)}"
    )
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 220},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = str(data.get("response", "")).strip()
        if not text:
            raise ValueError("respuesta vacía")
        if len(text) > 2000:
            text = text[:2000].rstrip()
        return Explanation(
            text=text,
            source="ollama",
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("LLM no disponible, se usa plantilla: %s", exc)
        return Explanation(
            text=template_explanation(context),
            source="template",
            fallback_reason=f"{type(exc).__name__}: {exc}",
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


def explain_observation(
    observation: Observation,
    *,
    semantic: SemanticReport | None = None,
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
    use_llm: bool = False,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Explanation:
    context = build_context(
        observation,
        semantic=semantic,
        f4_score=f4_score,
        f5_summary=f5_summary,
    )
    if use_llm:
        return llm_explanation(context, endpoint=endpoint, model=model, timeout=timeout)
    started = time.perf_counter()
    return Explanation(
        text=template_explanation(context),
        source="template",
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
