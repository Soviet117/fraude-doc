"""Prueba de integración de la construcción del dataset sintético v1 (F2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraude.synth.build import build_dataset
from fraude.synth.splits import load_dataset


def test_builds_all_artifacts_and_links_splits(tmp_path: Path) -> None:
    records = build_dataset(
        tmp_path,
        count=7,
        identities=7,
        copies=1,
        seed=42,
        dpi=72,
        variants=("scan",),
        holdout_attack="T5",
    )

    for filename in (
        "manifest.jsonl",
        "attacks_manifest.jsonl",
        "augmented_manifest.jsonl",
        "split_manifest.jsonl",
        "split_summary.json",
        "build.json",
    ):
        assert (tmp_path / filename).exists(), filename

    loaded = load_dataset(tmp_path)
    assert len(loaded) == len(records)
    assert all(record.split is not None for record in records)
    assert {record.split for record in records} == {
        "train",
        "val",
        "calibration",
        "test",
        "unseen_attack",
    }

    build = json.loads((tmp_path / "build.json").read_text(encoding="utf-8"))
    assert build["counts"]["total"] == len(records)
    assert build["counts"]["total"] == sum(
        build["counts"][key]
        for key in (
            "authentic",
            "attacks",
            "augmented",
        )
    )
    assert build["counts"]["augmented"] == (
        build["counts"]["authentic"] + build["counts"]["attacks"]
    )

    summary = json.loads((tmp_path / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == len(records)
    assert summary["holdout_attack"] == "T5"
    assert summary["by_attack"]["unseen_attack"]["T5"] > 0


def test_rejects_unknown_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_dataset(tmp_path, count=7, identities=7, variants=("unknown",))
    with pytest.raises(ValueError):
        build_dataset(tmp_path, count=7, identities=7, variants=())
