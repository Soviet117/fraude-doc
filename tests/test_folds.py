"""Pruebas de la validación cruzada agrupada por persona (F4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraude.synth.attacks import generate_attacks
from fraude.synth.dataset import DocumentRecord
from fraude.synth.folds import (
    assign_grouped_folds,
    summarize_folds,
    verify_fold_grouping,
    write_folds,
)
from fraude.synth.generate import generate_dataset


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[DocumentRecord]]:
    root = tmp_path_factory.mktemp("folds")
    authentic = generate_dataset(root, count=6, identities=6, seed=42, dpi=72)
    attacks = generate_attacks(authentic, root, seed=42, copies=1, dpi=72)
    return root, [*authentic, *attacks]


def test_each_person_falls_in_exactly_one_fold(
    corpus: tuple[Path, list[DocumentRecord]],
) -> None:
    _, records = corpus
    assignment = assign_grouped_folds(records, k=3, seed=42)

    assert set(assignment) == {record.source_id for record in records}
    verify_fold_grouping(records, assignment, k=3)
    for fold in range(3):
        assert any(value == fold for value in assignment.values())


def test_rejects_more_folds_than_people(corpus: tuple[Path, list[DocumentRecord]]) -> None:
    _, records = corpus
    with pytest.raises(ValueError):
        assign_grouped_folds(records, k=99, seed=42)


def test_write_folds_produces_manifest_and_summary(
    corpus: tuple[Path, list[DocumentRecord]],
    tmp_path: Path,
) -> None:
    _, records = corpus
    summary = write_folds(records, tmp_path, k=3, seed=42)

    assert (tmp_path / "folds_manifest.jsonl").exists()
    assert (tmp_path / "folds_summary.json").exists()
    assert summary["k"] == 3
    assert summary["groups"] == 6
    assert sum(item["samples"] for item in summary["folds"].values()) == len(records)


def test_summary_marks_both_classes_per_fold(corpus: tuple[Path, list[DocumentRecord]]) -> None:
    _, records = corpus
    assignment = assign_grouped_folds(records, k=3, seed=42)
    summary = summarize_folds(records, assignment, k=3, seed=42)

    assert all(item["authentic"] > 0 for item in summary["folds"].values())
