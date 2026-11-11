"""Pruebas del pipeline integral y de su serialización."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from fraude.core.schemas import (
    DoctypeInfo,
    Evidence,
    FusionReport,
    Observation,
    OcrReport,
    QualityReport,
    RiskLevel,
    Severity,
    SourceInfo,
)
from fraude.nlp.explain import Explanation
from fraude.nlp.semantic import SemanticReport
from fraude.pipeline.full import FullAnalysis, analyze_full


def _observation() -> Observation:
    return Observation(
        document_id="sha256:" + "e" * 64,
        source=SourceInfo(filename="dni.png", format="png", bytes=16),
        doctype=DoctypeInfo(label="dni", confidence=1.0),
        quality=QualityReport(ok=True),
        ocr=OcrReport(engine="docTR", mean_conf=0.9, n_words=20),
        fusion=FusionReport(
            p_fake=0.91,
            risk=RiskLevel.HIGH,
            evidences=[
                Evidence(
                    code="F6_REUTILIZACION",
                    severity=Severity.HIGH,
                    message="Copia exacta.",
                    weight=1.2,
                )
            ],
        ),
    )


def test_full_analysis_to_dict_is_json_serializable() -> None:
    analysis = FullAnalysis(
        observation=_observation(),
        semantic=SemanticReport(flags=("document_reuse",), name_consistency=0.9),
        explanation=Explanation(text="Indicios.", source="template"),
        f4_score=0.9,
        f5_summary={"max_prob": 0.8, "mean_prob": 0.1, "area_fraction": 0.02},
        f5_overlay_path="overlay.png",
        fusion_details={"raw_score": 0.9, "calibration_method": "identity"},
        timings_ms={"total_ms": 1200},
        model_versions={"f4": "fold_2", "f7": "fusion"},
    )

    payload = analysis.to_dict()

    assert payload["fusor"]["risk"] == "alto"
    assert payload["fusor"]["p_fake"] == 0.91
    assert payload["semantic"]["flags"] == ["document_reuse"]
    assert payload["f4_score"] == 0.9
    json.dumps(payload)


def test_analyze_full_uses_available_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observation = _observation()
    observation.fusion = FusionReport()
    report = FusionReport(
        p_fake=0.88,
        risk=RiskLevel.HIGH,
        evidences=[Evidence(code="F7", severity=Severity.HIGH, message="Señal alta.")],
    )

    monkeypatch.setattr("fraude.pipeline.full.analyze_bytes", lambda *a, **k: observation)
    monkeypatch.setattr("fraude.pipeline.full._run_f4", lambda *a, **k: 0.7)
    monkeypatch.setattr(
        "fraude.pipeline.full._run_f5",
        lambda *a, **k: ({"max_prob": 0.95, "mean_prob": 0.2, "area_fraction": 0.1}, None, None),
    )
    monkeypatch.setattr(
        "fraude.pipeline.full._run_f7",
        lambda *a, **k: {
            "report": report,
            "raw_score": 0.9,
            "calibration_method": "identity",
            "contributions": [],
        },
    )

    buffer = io.BytesIO()
    Image.new("RGB", (60, 80), (255, 255, 255)).save(buffer, format="PNG")
    analysis = analyze_full(
        buffer.getvalue(),
        filename="dni.png",
        artifacts_dir=tmp_path,
        f4_checkpoint=Path("no-existe.pt"),
        f5_checkpoint=Path("no-existe.pt"),
        f7_model=Path("no-existe.joblib"),
    )

    assert analysis.p_fake == 0.88
    assert analysis.risk is RiskLevel.HIGH
    assert analysis.f4_score == 0.7
    assert analysis.f5_summary is not None
    assert analysis.model_versions["f7"] == "no-existe"
    assert list(tmp_path.glob("**/*.png"))


def test_analyze_full_marks_quality_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observation = _observation()
    observation.quality = QualityReport(ok=False, issues=["blur"])
    monkeypatch.setattr("fraude.pipeline.full.analyze_bytes", lambda *a, **k: observation)

    buffer = io.BytesIO()
    Image.new("RGB", (60, 80), (255, 255, 255)).save(buffer, format="PNG")
    analysis = analyze_full(buffer.getvalue(), filename="dni.png", artifacts_dir=tmp_path)

    assert analysis.risk is RiskLevel.UNDETERMINED
    assert any(evidence.code == "RECAPTURAR" for evidence in analysis.observation.fusion.evidences)
