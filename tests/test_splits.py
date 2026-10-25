"""Pruebas de particiones agrupadas y sin fuga (F2.5)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fraude.synth.attacks import generate_attacks
from fraude.synth.augment import augment_dataset
from fraude.synth.dataset import DocumentRecord
from fraude.synth.generate import generate_dataset
from fraude.synth.splits import (
    RATIO_SPLITS,
    REAL_SPLIT,
    assign_splits,
    load_dataset,
    summarize_splits,
    verify_holdout,
    verify_no_group_leak,
    write_split_manifest,
)


@pytest.fixture(scope="module")
def dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[DocumentRecord], list[DocumentRecord]]:
    root = tmp_path_factory.mktemp("splits")
    authentic = generate_dataset(root, count=14, identities=7, seed=42, dpi=72)
    attacks = generate_attacks(authentic, root, seed=42, copies=1, dpi=72)
    augmented = augment_dataset(
        [*authentic, *attacks],
        root,
        seed=42,
        variants=("scan", "jpeg"),
    )
    originals = [*authentic, *attacks]
    return root, originals, [*originals, *augmented]


def test_assigns_every_record_and_uses_all_splits(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack="T5")

    assert len(labelled) == len(records)
    assert all(record.split is not None for record in labelled)
    assert {record.split for record in labelled} == {
        "train",
        "val",
        "calibration",
        "test",
        "unseen_attack",
    }


def test_no_source_or_document_crosses_splits(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack="T5")

    source_split: dict[str, str] = {}
    document_split: dict[str, str] = {}
    for record in labelled:
        assert source_split.setdefault(record.source_id, record.split) == record.split
        key = record.origin_id or record.document_id
        assert document_split.setdefault(key, record.split) == record.split

    verify_no_group_leak(labelled)


def test_holdout_attack_stays_out_of_training(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack="T5")

    holdout = [record for record in labelled if record.primary_attack == "T5"]
    assert holdout
    assert all(record.split == "unseen_attack" for record in holdout)
    assert not any(
        record.primary_attack == "T5" and record.split in {"train", "val", "calibration", "test"}
        for record in labelled
    )

    holdout_sources = {record.source_id for record in holdout}
    for record in labelled:
        if record.source_id in holdout_sources:
            assert record.split == "unseen_attack"

    verify_holdout(labelled, "T5")


def test_holdout_can_be_disabled(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack=None)

    assert {record.split for record in labelled} == set(RATIO_SPLITS)
    verify_holdout(labelled, None)


def test_assignment_is_deterministic(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset

    first = assign_splits(records, seed=7, holdout_attack="T5")
    second = assign_splits(records, seed=7, holdout_attack="T5")

    assert [(record.sample_id, record.split) for record in first] == [
        (record.sample_id, record.split) for record in second
    ]


def test_summary_counts_are_consistent(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack="T5")
    summary = summarize_splits(labelled, seed=42, holdout_attack="T5")

    assert summary["count"] == len(records)
    assert sum(summary["splits"].values()) == len(records)
    assert sum(summary["groups_by_split"].values()) == len({record.source_id for record in records})
    assert summary["holdout_attack"] == "T5"
    assert summary["by_attack"]["unseen_attack"]["T5"] == len(
        [record for record in labelled if record.primary_attack == "T5"]
    )


def test_split_manifest_is_written(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
    tmp_path: Path,
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack="T5")
    write_split_manifest(labelled, tmp_path, seed=42, holdout_attack="T5")

    lines = (tmp_path / "split_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    parsed = [json.loads(line) for line in lines]
    assert all(item["split"] is not None for item in parsed)

    summary = json.loads((tmp_path / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == len(records)
    assert summary["holdout_attack"] == "T5"


def test_load_dataset_reads_all_manifests(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, originals, combined = dataset

    loaded = load_dataset(root)

    assert len(loaded) == len(combined)
    assert len(loaded) > len(originals)


def test_real_records_are_reserved(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    real_record = replace(
        records[0],
        sample_id="real_0001_dni",
        document_id="sha256:real0001",
        source_id="real_0001",
        origin_id=None,
    )

    labelled = assign_splits([*records, real_record], seed=42, holdout_attack="T5")
    real = next(record for record in labelled if record.sample_id == "real_0001_dni")

    assert real.split == REAL_SPLIT


def test_verify_detects_injected_leaks(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = dataset
    labelled = assign_splits(records, seed=42, holdout_attack="T5")

    train_record = next(record for record in labelled if record.split == "train")
    leaked = [
        replace(record, split="test") if record.sample_id == train_record.sample_id else record
        for record in labelled
    ]
    with pytest.raises(ValueError):
        verify_no_group_leak(leaked)

    holdout_record = next(record for record in labelled if record.primary_attack == "T5")
    leaked_holdout = [
        replace(record, split="train") if record.sample_id == holdout_record.sample_id else record
        for record in labelled
    ]
    with pytest.raises(ValueError):
        verify_holdout(leaked_holdout, "T5")


def test_invalid_inputs_are_rejected(
    dataset: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
    tmp_path: Path,
) -> None:
    _, _, records = dataset

    assert assign_splits([]) == []
    with pytest.raises(ValueError):
        assign_splits(records, holdout_attack="T9")
    with pytest.raises(ValueError):
        assign_splits(records[:1], holdout_attack=None)
    with pytest.raises(ValueError):
        assign_splits(records, seed=1, ratios={"train": 1.0})
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)
