"""Pruebas del validador de corpus y del lote de humo (F4)."""

from __future__ import annotations

import json
from pathlib import Path

from fraude.synth.build import build_dataset
from fraude.synth.smoke import build_smoke
from fraude.synth.validate import validate_dataset


def test_smoke_batch_has_three_documents_and_valid_manifests(tmp_path: Path) -> None:
    records = build_smoke(tmp_path, dpi=72)

    assert len(records) == 3
    assert sum(1 for record in records if record.is_authentic) == 2
    assert sum(1 for record in records if not record.is_authentic) == 1
    assert (tmp_path / "smoke.json").exists()
    payload = json.loads((tmp_path / "smoke.json").read_text(encoding="utf-8"))
    assert payload["validation"]["ok"] is True
    assert len(payload["documents"]) == 3


def test_validator_detects_missing_image(tmp_path: Path) -> None:
    build_dataset(
        tmp_path,
        count=2,
        identities=2,
        variants=(),
        attacks=("T1",),
        per_attack={"T1": 1},
        split=False,
        dpi=72,
    )
    (tmp_path / "images" / "dni_0000.png").unlink()

    report = validate_dataset(tmp_path, expect_authentic=2, expect_tampered=1)

    assert report["ok"] is False
    assert any("falta la imagen" in error for error in report["errors"])


def test_validator_detects_wrong_balance(tmp_path: Path) -> None:
    build_dataset(
        tmp_path,
        count=2,
        identities=2,
        variants=(),
        attacks=("T1",),
        per_attack={"T1": 1},
        split=False,
        dpi=72,
    )

    report = validate_dataset(tmp_path, expect_authentic=3, expect_tampered=1)

    assert report["ok"] is False
    assert any("Auténticos base" in error for error in report["errors"])
