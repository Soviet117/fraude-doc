"""Pruebas de F7: vector de características, fusión y calibración."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fraude.core.schemas import (
    DoctypeInfo,
    Observation,
    RegistryReport,
    RuleFinding,
    Severity,
    SourceInfo,
)
from fraude.fusion.features import FEATURE_NAMES, observation_features
from fraude.fusion.infer import predict_fusion
from fraude.fusion.train import FusionConfig, cross_validate, train_fusion
from fraude.nlp.semantic import SemanticReport


def _observation(**overrides: Any) -> Observation:
    payload: dict[str, Any] = {
        "document_id": "sha256:" + "0" * 64,
        "source": SourceInfo(filename="sample.png", bytes=1),
        "doctype": DoctypeInfo(label="dni", confidence=0.9),
        "fields": {"dni": "12345678", "nombres": "José Alberto"},
        "rules": [
            RuleFinding(
                code="FECHA_INVALIDA",
                severity=Severity.HIGH,
                message="La fecha de emisión no es válida.",
            ),
            RuleFinding(
                code="OCR_BAJO",
                severity=Severity.LOW,
                message="La confianza media del OCR es baja.",
            ),
        ],
        "registry": RegistryReport(matched=False, score=0.4, diffs=["nombre: esperado X"]),
    }
    payload.update(overrides)
    return Observation(**payload)


def _row(sample_id: str, fold: int, label: int, score: float) -> dict[str, Any]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["f4_score"] = score
    features["rule_weight"] = 0.6 if label else 0.0
    return {
        "sample_id": sample_id,
        "source_id": f"persona_{sample_id}",
        "fold": fold,
        "label": label,
        "attack": "T1" if label else None,
        "document_type": "dni",
        "features": features,
    }


def test_observation_features_shape_and_missing_flags() -> None:
    features = observation_features(_observation())

    assert tuple(features) == FEATURE_NAMES
    assert features["f4_missing"] == 1.0
    assert features["f5_missing"] == 1.0
    assert features["ocr_missing"] == 1.0
    assert features["rule_FECHA_INVALIDA"] == 1.0
    assert features["rule_OCR_BAJO"] == 1.0
    assert features["n_high"] == 1.0
    assert features["registry_identity_diff"] == 1.0
    assert all(isinstance(value, float) for value in features.values())


def test_observation_features_uses_semantic_report() -> None:
    semantic = SemanticReport(
        flags=("document_reuse", "name_mismatch"),
        name_consistency=0.42,
    )
    features = observation_features(
        _observation(),
        f4_score=0.8,
        f5_summary={"max_prob": 0.7, "mean_prob": 0.01, "area_fraction": 0.002},
        semantic=semantic,
    )

    assert features["f4_score"] == 0.8
    assert features["f4_missing"] == 0.0
    assert features["f5_max_prob"] == 0.7
    assert features["f5_missing"] == 0.0
    assert features["f6_document_reuse"] == 1.0
    assert features["f6_name_mismatch"] == 1.0
    assert features["f6_name_consistency"] == 0.42
    assert features["f6_n_flags"] == 2.0


def test_cross_validate_is_grouped_and_bounded() -> None:
    rows = []
    for index in range(24):
        fold = index % 4
        label = 1 if index % 2 else 0
        rows.append(_row(f"doc_{index:03d}", fold, label, 0.9 if label else 0.1))
    cfg = FusionConfig(n_estimators=20, n_boot=10)

    oof = cross_validate(rows, feature_names=FEATURE_NAMES, cfg=cfg)

    assert len(oof) == len(rows)
    assert sorted(row["sample_id"] for row in oof) == sorted(row["sample_id"] for row in rows)
    for row in oof:
        assert 0.0 <= row["raw_score"] <= 1.0
        assert row["fold"] == int(row["sample_id"].split("_")[1]) % 4


def test_train_fusion_smoke(tmp_path: Path) -> None:
    rows = []
    for index in range(30):
        fold = index % 5
        label = 1 if index % 2 else 0
        rows.append(_row(f"doc_{index:03d}", fold, label, 0.95 if label else 0.05))
    features_path = tmp_path / "features.jsonl"
    features_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "f7"
    cfg = FusionConfig(
        features_path=features_path,
        out_dir=out_dir,
        n_estimators=30,
        n_boot=20,
    )

    result = train_fusion(cfg)

    assert result["n"] == 30
    assert "pooled" in result
    assert "calibrated_pooled" in result
    assert "ablations" in result
    assert (out_dir / "fusion.joblib").is_file()
    assert (out_dir / "oof.jsonl").is_file()
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "eval.md").is_file()
    oof_rows = [
        json.loads(line)
        for line in (out_dir / "oof.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all("calibrated_score" in row for row in oof_rows)


class _DummyModel:
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray([[0.2, 0.8]])


class _DummyCalibrator:
    def predict(self, values: list[float]) -> np.ndarray:
        return np.asarray(values)


def test_predict_fusion_returns_calibrated_risk_and_evidence() -> None:
    semantic = SemanticReport(
        flags=("document_reuse",),
        notes=("La imagen coincide con un documento auténtico ya conocido.",),
    )
    report = predict_fusion(
        _observation(),
        bundle={
            "model": _DummyModel(),
            "calibrator": _DummyCalibrator(),
            "feature_names": list(FEATURE_NAMES),
        },
        f4_score=0.8,
        f5_summary={"max_prob": 0.7, "mean_prob": 0.01, "area_fraction": 0.002},
        semantic=semantic,
    )

    assert report.p_fake == 0.8
    assert report.risk.value == "alto"
    codes = {evidence.code for evidence in report.evidences}
    assert "FECHA_INVALIDA" in codes
    assert "REGISTRO_DIFERENCIA" in codes
    assert "SEMANTICA" in codes
    assert "F4_INDICIO_VISUAL" in codes
    assert "F5_LOCALIZACION" in codes
