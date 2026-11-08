"""Inferencia de la fusión F7 para la demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from fraude.core.schemas import Evidence, FusionReport, Observation, RiskLevel, Severity
from fraude.fusion.features import FEATURE_NAMES, observation_features
from fraude.nlp.semantic import SemanticReport


def load_fusion_model(path: Path) -> dict[str, Any]:
    return joblib.load(Path(path))


def predict_fusion(
    observation: Observation,
    *,
    bundle: dict[str, Any],
    f4_score: float | None = None,
    f5_summary: dict[str, float] | None = None,
    semantic: SemanticReport | None = None,
) -> FusionReport:
    """Devuelve una probabilidad calibrada y evidencias explicables."""
    features = observation_features(
        observation,
        f4_score=f4_score,
        f5_summary=f5_summary,
        semantic=semantic,
    )
    names = tuple(bundle.get("feature_names", FEATURE_NAMES))
    vector = np.asarray([[features.get(name, 0.0) for name in names]], dtype=np.float32)
    raw = float(bundle["model"].predict_proba(vector)[0, 1])
    probability = float(bundle["calibrator"].predict([raw])[0])
    probability = min(1.0, max(0.0, probability))
    if probability < 0.33:
        risk = RiskLevel.LOW
    elif probability < 0.66:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.HIGH

    evidences: list[Evidence] = []
    for finding in observation.rules[:6]:
        evidences.append(
            Evidence(
                code=finding.code,
                severity=finding.severity,
                message=finding.message,
            )
        )
    for diff in observation.registry.diffs[:3]:
        evidences.append(
            Evidence(
                code="REGISTRO_DIFERENCIA",
                severity=Severity.HIGH if diff.startswith(("nombre:", "dni:")) else Severity.MEDIUM,
                message=f"El documento difiere del registro interno: {diff}.",
            )
        )
    if semantic is not None:
        for note in semantic.notes[:3]:
            evidences.append(
                Evidence(
                    code="SEMANTICA",
                    severity=Severity.MEDIUM,
                    message=note,
                )
            )
    if f4_score is not None and f4_score >= 0.5:
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
    if f5_summary and float(f5_summary.get("max_prob", 0.0)) >= 0.5:
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
    return FusionReport(p_fake=round(probability, 4), risk=risk, evidences=evidences)
