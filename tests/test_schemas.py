"""Pruebas del contrato de datos."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fraude.core.schemas import Observation, RiskLevel
from fraude.pipeline.run import analyze_bytes


def test_observation_roundtrip() -> None:
    observation = analyze_bytes(b"contenido de prueba", "documento.png")

    restored = Observation.model_validate_json(observation.model_dump_json())

    assert restored.document_id == observation.document_id
    assert restored.source.filename == "documento.png"
    assert restored.fusion.risk is RiskLevel.UNDETERMINED


def test_document_id_has_sha256_prefix() -> None:
    observation = analyze_bytes(b"abc", "documento.pdf")

    assert observation.document_id.startswith("sha256:")
    assert len(observation.document_id) == len("sha256:") + 64


def test_unknown_fields_are_rejected() -> None:
    payload = analyze_bytes(b"abc", "documento.pdf").model_dump()
    payload["campo_no_previsto"] = True

    with pytest.raises(ValidationError):
        Observation.model_validate(payload)


def test_invalid_document_id_is_rejected() -> None:
    payload = analyze_bytes(b"abc", "documento.pdf").model_dump()
    payload["document_id"] = "no-es-un-hash"

    with pytest.raises(ValidationError):
        Observation.model_validate(payload)
