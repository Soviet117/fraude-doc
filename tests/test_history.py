"""Pruebas del historial SQLite (documentos, análisis y feedback)."""

from __future__ import annotations

from pathlib import Path

from fraude.core.schemas import (
    DoctypeInfo,
    FusionReport,
    Observation,
    RiskLevel,
    SourceInfo,
)
from fraude.store.history import HistoryStore


def _analysis(document_id: str, filename: str, risk: str = "alto") -> dict:
    observation = Observation(
        document_id=document_id,
        source=SourceInfo(filename=filename, format="png", bytes=10),
        doctype=DoctypeInfo(label="dni", confidence=1.0),
        fusion=FusionReport(p_fake=0.91, risk=RiskLevel(risk)),
    )
    return {
        "document_id": document_id,
        "observation": observation.model_dump(mode="json"),
        "fusor": {
            "p_fake": 0.91,
            "risk": risk,
            "calibration_method": "identity",
        },
        "explanation": {"text": "Indicios para revisión.", "source": "template"},
        "model_versions": {"f4": "fold_2", "f7": "fusion"},
        "timings_ms": {"total_ms": 1200},
    }


def test_history_roundtrip_and_feedback(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    first = store.save_analysis(_analysis("sha256:" + "a" * 64, "uno.png"))
    second = store.save_analysis(_analysis("sha256:" + "b" * 64, "dos.png", risk="bajo"))

    assert first == 1
    assert second == 2
    recent = store.recent()
    assert len(recent) == 2
    assert recent[0]["filename"] == "dos.png"

    payload = store.get(first)
    assert payload is not None
    assert payload["analysis_id"] == 1
    assert payload["observation"]["doctype"]["label"] == "dni"

    feedback_id = store.add_feedback(first, label="requiere_revision", notes="revisar")
    assert feedback_id == 1
    feedback = store.feedback_for(first)
    assert feedback[0]["label"] == "requiere_revision"
    assert store.stats() == {"total": 2, "by_risk": {"alto": 1, "bajo": 1}, "reviewed": 1}

    store.close()


def test_history_upserts_document(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    document_id = "sha256:" + "c" * 64
    store.save_analysis(_analysis(document_id, "uno.png"))
    store.save_analysis(_analysis(document_id, "uno_renombrado.png"))

    rows = store.recent()
    assert len(rows) == 2
    assert {row["filename"] for row in rows} == {"uno_renombrado.png"}
    store.close()
