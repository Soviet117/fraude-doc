"""Vector de características para la fusión F7.

Cada señal se convierte en un número; los valores ausentes se marcan con un
indicador ``*_missing`` para que el modelo no los confunda con un cero real.
"""

from __future__ import annotations

from typing import Any

from fraude.core.schemas import Observation, Severity
from fraude.eval.baseline import SEVERITY_WEIGHTS
from fraude.nlp.semantic import SemanticReport

RULE_CODES = (
    "CALIDAD_BAJA",
    "CAMPOS_FALTANTES",
    "COPY_MOVE",
    "DNI_DUPLICADO",
    "DNI_INVALIDO",
    "EDAD_IMPLAUSIBLE",
    "FECHA_FUTURA",
    "FECHA_INVALIDA",
    "HORAS_IMPLAUSIBLES",
    "METADATA_EDICION",
    "MODULO_DESCONOCIDO",
    "NOMBRE_INCONSISTENTE",
    "NOTA_FUERA_DE_RANGO",
    "OCR_BAJO",
    "ORDEN_FECHAS",
    "PERIODO_INVALIDO",
    "PROMEDIO_FUERA_DE_RANGO",
    "PROGRAMA_DESCONOCIDO",
    "REGISTRO_DIFERENCIA",
    "REGISTRO_NO_COINCIDENTE",
    "RESOLUCION_INVALIDA",
    "TIPO_NO_IDENTIFICADO",
)
CAPTURE_TYPES = ("scan", "photo", "screenshot", "pdf_render", "unknown")
FEATURE_GROUPS = {
    "rules": (
        "quality_ok",
        "blur",
        "skew_abs",
        *(f"capture_{name}" for name in CAPTURE_TYPES),
        "ocr_mean_conf",
        "ocr_log_words",
        "ocr_missing",
        "n_rules",
        "n_high",
        "n_medium",
        "n_low",
        "rule_weight",
        *(f"rule_{code}" for code in RULE_CODES),
    ),
    "forensics": (
        "forensics_ela",
        "forensics_noise",
        "forensics_chromatic",
        "forensics_jpeg",
        "forensics_metadata",
        "forensics_localization",
    ),
    "visual": ("f4_score", "f4_missing"),
    "localization": (
        "f5_max_prob",
        "f5_mean_prob",
        "f5_area_fraction",
        "f5_missing",
    ),
    "semantic": (
        "doctype_confidence",
        "registry_matched",
        "registry_score",
        "registry_n_diffs",
        "registry_identity_diff",
        "n_fields",
        "f6_duplicate",
        "f6_name_consistency",
        "f6_n_flags",
        "f6_document_reuse",
        "f6_name_mismatch",
        "f6_implausible_emission_date",
    ),
}
FEATURE_NAMES = tuple(name for group in FEATURE_GROUPS.values() for name in group)


def observation_features(
    observation: Observation,
    *,
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
    semantic: SemanticReport | None = None,
) -> dict[str, float]:
    """Convierte una observación y sus señales externas en un vector numérico."""
    findings = observation.rules
    severity_counts = {
        severity: sum(1 for finding in findings if finding.severity == severity)
        for severity in (Severity.HIGH, Severity.MEDIUM, Severity.LOW)
    }
    codes = {finding.code for finding in findings}
    capture = observation.capture.type.value
    registry_identity_diff = any(
        diff.startswith(("nombre:", "dni:")) for diff in observation.registry.diffs
    )
    semantic_flags = set(semantic.flags) if semantic is not None else set()

    features: dict[str, float] = {
        "quality_ok": 1.0 if observation.quality.ok else 0.0,
        "blur": _number(observation.quality.blur),
        "skew_abs": abs(_number(observation.quality.skew_deg)),
        **{f"capture_{name}": 1.0 if capture == name else 0.0 for name in CAPTURE_TYPES},
        "ocr_mean_conf": _number(observation.ocr.mean_conf),
        "ocr_log_words": _log1p(observation.ocr.n_words),
        "ocr_missing": 1.0 if observation.ocr.mean_conf is None else 0.0,
        "n_rules": float(len(findings)),
        "n_high": float(severity_counts[Severity.HIGH]),
        "n_medium": float(severity_counts[Severity.MEDIUM]),
        "n_low": float(severity_counts[Severity.LOW]),
        "rule_weight": round(
            sum(SEVERITY_WEIGHTS[finding.severity] for finding in findings),
            4,
        ),
        **{f"rule_{code}": 1.0 if code in codes else 0.0 for code in RULE_CODES},
        "forensics_ela": _number(observation.forensics.ela.score),
        "forensics_noise": _number(observation.forensics.noise.score),
        "forensics_chromatic": _number(observation.forensics.chromatic.score),
        "forensics_jpeg": _number(observation.forensics.jpeg.score),
        "forensics_metadata": _number(observation.forensics.metadata.score),
        "forensics_localization": _number(observation.forensics.localization.score),
        "doctype_confidence": _number(observation.doctype.confidence),
        "registry_matched": 1.0 if observation.registry.matched else 0.0,
        "registry_score": _number(observation.registry.score),
        "registry_n_diffs": float(len(observation.registry.diffs)),
        "registry_identity_diff": 1.0 if registry_identity_diff else 0.0,
        "n_fields": float(len(observation.fields)),
        "f4_score": _number(f4_score),
        "f4_missing": 1.0 if f4_score is None else 0.0,
        "f5_max_prob": _number((f5_summary or {}).get("max_prob")),
        "f5_mean_prob": _number((f5_summary or {}).get("mean_prob")),
        "f5_area_fraction": _number((f5_summary or {}).get("area_fraction")),
        "f5_missing": 1.0 if not f5_summary else 0.0,
        "f6_duplicate": 1.0 if semantic is not None and semantic.duplicate is not None else 0.0,
        "f6_name_consistency": _number(semantic.name_consistency if semantic is not None else None),
        "f6_n_flags": float(len(semantic_flags)),
        "f6_document_reuse": 1.0 if "document_reuse" in semantic_flags else 0.0,
        "f6_name_mismatch": 1.0 if "name_mismatch" in semantic_flags else 0.0,
        "f6_implausible_emission_date": (
            1.0 if "implausible_emission_date" in semantic_flags else 0.0
        ),
    }
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError("Faltan características: " + ", ".join(missing))
    return {name: float(features[name]) for name in FEATURE_NAMES}


def feature_vector(
    observation: Observation,
    *,
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
    semantic: SemanticReport | None = None,
) -> list[float]:
    features = observation_features(
        observation,
        f4_score=f4_score,
        f5_summary=f5_summary,
        semantic=semantic,
    )
    return [features[name] for name in FEATURE_NAMES]


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _log1p(value: int | float) -> float:
    import math

    return round(math.log1p(max(0.0, float(value))), 4)
