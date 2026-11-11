"""Inferencia de la fusión F7 para la demo.

Además del score calibrado, devuelve las contribuciones SHAP de las señales
que más empujan el riesgo. El LLM nunca decide: estas evidencias son las que
alimentan la explicación y la revisión humana.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from fraude.core.schemas import Evidence, FusionReport, Observation, RiskLevel, Severity
from fraude.fusion.features import FEATURE_NAMES, observation_features
from fraude.nlp.semantic import SemanticReport

MIN_DISPLAY_CONTRIBUTION = 0.05

FEATURE_MESSAGES = {
    "f6_duplicate": "La imagen coincide con un documento auténtico ya registrado.",
    "f6_document_reuse": "El documento reutiliza un escaneo previamente conocido.",
    "f6_name_mismatch": "El nombre del titular no coincide entre campos del documento.",
    "f6_implausible_emission_date": "La fecha de emisión no es plausible para el periodo.",
    "f6_name_consistency": "La consistencia del nombre entre campos es baja.",
    "f4_score": "La señal visual de página F4 es alta.",
    "f5_max_prob": "F5 localiza una región con alta probabilidad de manipulación.",
    "f5_mean_prob": "F5 detecta probabilidad elevada en una parte de la página.",
    "f5_area_fraction": "La región localizada por F5 cubre una fracción apreciable.",
    "registry_score": "La similitud con el registro interno es baja.",
    "registry_identity_diff": "El nombre o DNI difiere del registro interno.",
    "registry_n_diffs": "Hay varias diferencias frente al registro interno.",
    "n_rules": "El documento acumula varias reglas de riesgo.",
    "n_high": "Hay reglas de severidad alta activadas.",
    "n_medium": "Hay reglas de severidad media activadas.",
    "rule_weight": "La suma ponderada de reglas es alta.",
    "quality_ok": "La calidad de la imagen es insuficiente para un análisis fiable.",
    "blur": "El desenfoque de la imagen es alto.",
    "skew_abs": "La inclinación de la página es alta.",
    "ocr_mean_conf": "La confianza del OCR es baja.",
    "ocr_log_words": "El volumen de texto reconocido es anómalo.",
    "ocr_missing": "No se pudo ejecutar OCR sobre el documento.",
    "forensics_ela": "ELA detecta indicios de recompresión o edición.",
    "forensics_noise": "El residuo de ruido es inconsistente en la página.",
    "forensics_chromatic": "Hay inconsistencias cromáticas compatibles con edición.",
    "forensics_jpeg": "El análisis JPEG muestra indicios de doble compresión.",
    "forensics_metadata": "Los metadatos indican edición con software.",
    "forensics_localization": "Copy-move marca regiones duplicadas.",
}


def load_fusion_model(path: Path) -> dict[str, Any]:
    return joblib.load(Path(path))


def _risk_from_probability(probability: float) -> RiskLevel:
    if probability < 0.33:
        return RiskLevel.LOW
    if probability < 0.66:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def _feature_message(
    name: str,
    observation: Observation,
    features: dict[str, float],
) -> str | None:
    value = float(features.get(name, 0.0))
    if name == "f4_score":
        return f"La señal visual de página F4 es {value:.3f} y aporta al riesgo."
    if name == "f5_max_prob":
        return f"F5 alcanza una probabilidad máxima de {value:.3f} en la página."
    if name == "f5_mean_prob":
        return f"F5 marca una probabilidad media de {value:.3f}."
    if name == "f5_area_fraction":
        return f"F5 marca el {value:.1%} del área de la página."
    if name == "registry_score":
        return f"El cruce con el registro interno tiene score {value:.3f}."
    if name == "ocr_mean_conf":
        return f"La confianza media del OCR es {value:.3f}."
    if name == "blur":
        return f"El estadístico de nitidez (varianza) es {value:.1f}."
    if name in FEATURE_MESSAGES:
        return FEATURE_MESSAGES[name]
    if name.startswith("rule_"):
        code = name[len("rule_") :]
        for finding in observation.rules:
            if finding.code == code:
                return finding.message
        return f"Se activó la regla {code}."
    if name.startswith("capture_"):
        return "El tipo de captura condiciona qué señales forenses aplican."
    return None


def _contributions(
    bundle: dict[str, Any],
    features: dict[str, float],
    observation: Observation,
) -> list[dict[str, Any]]:
    model = bundle.get("model")
    booster = getattr(model, "booster_", None)
    if booster is None:
        return []
    names = tuple(bundle.get("feature_names", FEATURE_NAMES))
    vector = np.asarray([[features.get(name, 0.0) for name in names]], dtype=np.float32)
    try:
        values = np.asarray(booster.predict(vector, pred_contrib=True))[0]
    except (ValueError, AttributeError):
        return []
    contributions: list[dict[str, Any]] = []
    for name, value in sorted(
        zip(names, values[:-1], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    ):
        message = _feature_message(name, observation, features)
        if message is None:
            continue
        contributions.append(
            {
                "feature": name,
                "contribution": round(float(value), 4),
                "message": message,
            }
        )
    return contributions


def _evidence_code(feature: str) -> str:
    """Agrupa features de la misma familia en una sola evidencia."""
    if feature.startswith("rule_"):
        return feature[len("rule_") :]
    if feature.startswith("f5_"):
        return "F5_LOCALIZACION"
    if feature.startswith("registry_"):
        return "REGISTRO"
    if feature in {"n_rules", "n_high", "n_medium", "rule_weight"}:
        return "REGLAS"
    if feature in {"f6_duplicate", "f6_document_reuse"}:
        return "F6_REUTILIZACION"
    return feature.upper()


def predict_fusion_details(
    observation: Observation,
    *,
    bundle: dict[str, Any],
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
    semantic: SemanticReport | None = None,
    max_evidences: int = 8,
) -> dict[str, Any]:
    """Devuelve el informe, el score crudo y las contribuciones SHAP."""
    features = observation_features(
        observation,
        f4_score=f4_score,
        f5_summary=f5_summary,
        semantic=semantic,
    )
    names = tuple(bundle.get("feature_names", FEATURE_NAMES))
    vector = np.asarray([[features.get(name, 0.0) for name in names]], dtype=np.float32)
    raw = float(bundle["model"].predict_proba(vector)[0, 1])
    probability = float(np.clip(bundle["calibrator"].predict([raw])[0], 0.0, 1.0))
    risk = _risk_from_probability(probability)

    contributions = _contributions(bundle, features, observation)
    evidences: list[Evidence] = []
    seen: set[str] = set()
    seen_messages: set[str] = set()

    allow_positive = probability >= 0.33
    for item in contributions:
        if len(evidences) >= min(max_evidences, 5):
            break
        contribution = float(item["contribution"])
        if contribution < MIN_DISPLAY_CONTRIBUTION or not allow_positive:
            continue
        severity = (
            Severity.HIGH
            if contribution >= 0.5
            else Severity.MEDIUM
            if contribution >= 0.15
            else Severity.LOW
        )
        code = _evidence_code(str(item["feature"]))
        if code == "REGISTRO" and observation.registry.diffs:
            continue
        message = item["message"]
        if item["feature"] in {"f6_duplicate", "f6_document_reuse"} and semantic is not None:
            duplicate = semantic.duplicate
            if duplicate is not None:
                if duplicate.exact:
                    code = "F6_REUTILIZACION"
                    message = (
                        "El documento es una copia exacta de un documento auténtico "
                        f"ya registrado ({duplicate.sample_id})."
                    )
                else:
                    code = "F6_SIMILITUD"
                    message = (
                        "La imagen es muy similar a un documento auténtico ya registrado "
                        f"({duplicate.sample_id}); puede ser una copia modificada o reutilizada."
                    )
        if code in seen:
            continue
        seen.add(code)
        evidences.append(
            Evidence(
                code=code,
                severity=severity,
                message=message,
                weight=contribution,
            )
        )
        seen_messages.add(message)

    for finding in observation.rules:
        if len(evidences) >= max_evidences:
            break
        if finding.code in seen or finding.message in seen_messages:
            continue
        seen.add(finding.code)
        seen_messages.add(finding.message)
        evidences.append(
            Evidence(
                code=finding.code,
                severity=finding.severity,
                message=finding.message,
            )
        )
    for diff in observation.registry.diffs:
        if len(evidences) >= max_evidences:
            break
        code = "REGISTRO_DIFERENCIA"
        message = f"El documento difiere del registro interno: {diff}."
        if code in seen or message in seen_messages:
            continue
        seen.add(code)
        seen_messages.add(message)
        evidences.append(
            Evidence(
                code=code,
                severity=Severity.HIGH if diff.startswith(("nombre:", "dni:")) else Severity.MEDIUM,
                message=message,
            )
        )
    if semantic is not None:
        for note in semantic.notes:
            if len(evidences) >= max_evidences:
                break
            if (
                "F6_REUTILIZACION" in seen or "F6_SIMILITUD" in seen
            ) and "documento auténtico" in note.lower():
                continue
            code = "SEMANTICA"
            if code in seen or note in seen_messages:
                break
            seen.add(code)
            seen_messages.add(note)
            evidences.append(Evidence(code=code, severity=Severity.MEDIUM, message=note))
    if f4_score is not None and f4_score >= 0.5 and "F4_SCORE" not in seen:
        evidences.append(
            Evidence(
                code="F4_INDICIO_VISUAL",
                severity=Severity.MEDIUM,
                message=(
                    f"La señal visual F4 es alta ({f4_score:.3f}); "
                    "es un indicio sin calibrar, no una probabilidad de fraude."
                ),
            )
        )
    if (
        f5_summary
        and float(f5_summary.get("max_prob", 0.0)) >= 0.9
        and float(f5_summary.get("area_fraction", 0.0)) >= 0.02
        and "F5_LOCALIZACION" not in seen
    ):
        evidences.append(
            Evidence(
                code="F5_LOCALIZACION",
                severity=Severity.MEDIUM,
                message=(
                    "F5 localiza una región con probabilidad "
                    f"{float(f5_summary['max_prob']):.3f}; el mapa es cualitativo."
                ),
            )
        )

    if probability < 0.33 and not evidences:
        evidences.append(
            Evidence(
                code="F7_SIN_INDICIOS",
                severity=Severity.LOW,
                message=(
                    "No se acumularon indicios suficientes para priorizar revisión; "
                    "esto no certifica que el documento sea auténtico."
                ),
            )
        )

    report = FusionReport(
        p_fake=round(probability, 4),
        risk=risk,
        evidences=evidences[:max_evidences],
    )
    return {
        "report": report,
        "raw_score": round(raw, 6),
        "calibrated_score": round(probability, 6),
        "calibration_method": bundle.get("calibration_method", "isotonic"),
        "contributions": contributions[:8],
        "features": features,
    }


def predict_fusion(
    observation: Observation,
    *,
    bundle: dict[str, Any],
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
    semantic: SemanticReport | None = None,
) -> FusionReport:
    """Devuelve una probabilidad calibrada y evidencias explicables."""
    return predict_fusion_details(
        observation,
        bundle=bundle,
        f4_score=f4_score,
        f5_summary=f5_summary,
        semantic=semantic,
    )["report"]
