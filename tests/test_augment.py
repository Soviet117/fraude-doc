"""Pruebas de la aumentación de captura F2.4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

from fraude.synth.attacks import generate_attacks
from fraude.synth.augment import (
    DEFAULT_VARIANTS,
    apply_augmentation,
    augment_dataset,
)
from fraude.synth.dataset import DocumentRecord
from fraude.synth.generate import generate_dataset

LOCAL_ATTACKS = {"T1", "T2", "T3", "T4", "T5", "T6"}


@pytest.fixture(scope="module")
def augmented(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[DocumentRecord], list[DocumentRecord]]:
    root = tmp_path_factory.mktemp("augment")
    authentic = generate_dataset(root, count=7, identities=7, seed=42, dpi=72)
    attacks = generate_attacks(authentic, root, seed=42, copies=1, dpi=72)
    originals = [*authentic, *attacks]
    records = augment_dataset(originals, root, seed=42)
    return root, originals, records


def test_covers_every_variant_and_keeps_all_samples(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, originals, records = augmented

    assert len(records) == len(originals) * len(DEFAULT_VARIANTS)
    assert {record.augmentations[-1] for record in records} == set(DEFAULT_VARIANTS)
    assert len({record.sample_id for record in records}) == len(records)
    assert len({record.document_id for record in records}) == len(records)


def test_preserves_labels_and_links_to_origin(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, originals, records = augmented
    by_sample_id = {record.sample_id: record for record in originals}

    for record in records:
        origin = by_sample_id[record.sample_id.rsplit("_", 1)[0]]
        assert record.origin_id == origin.document_id
        assert record.variant == origin.variant
        assert record.is_authentic == origin.is_authentic
        assert record.attack_types == origin.attack_types
        assert record.primary_attack == origin.primary_attack
        assert record.parent_id == origin.parent_id
        assert record.source_id == origin.source_id
        assert record.document_type == origin.document_type
        assert record.template_version == origin.template_version
        assert record.fields == origin.fields
        assert record.page_size == origin.page_size
        assert record.augmentations == (record.sample_id.rsplit("_", 1)[-1],)


def test_masks_and_regions_follow_the_same_transform(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, _, records = augmented

    for record in records:
        if record.primary_attack is None:
            assert record.regions == ()
            assert record.mask_paths == ()
            continue
        assert record.regions, record.primary_attack
        if record.primary_attack in LOCAL_ATTACKS:
            assert len(record.mask_paths) == 1
            with Image.open(root / record.mask_paths[0]) as mask:
                assert mask.size == record.page_size
                array = np.asarray(mask.convert("L"))
                coverage = float((array > 127).mean())
                assert coverage > 0
                if record.primary_attack == "T6":
                    assert coverage > 0.9
                else:
                    assert coverage < 0.5
            assert record.regions[0].mask_path == record.mask_paths[0]
        else:
            assert record.mask_paths == ()


def test_boxes_stay_inside_the_augmented_page(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = augmented

    for record in records:
        page_width, page_height = record.page_size
        boxes = [*record.field_boxes.values()]
        boxes.extend(region.bbox for region in record.regions)
        for x, y, width, height in boxes:
            assert width > 0 and height > 0
            assert 0 <= x < page_width
            assert 0 <= y < page_height
            assert x + width <= page_width
            assert y + height <= page_height


def test_capture_type_reflects_the_variant(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, originals, records = augmented
    by_sample_id = {record.sample_id: record for record in originals}

    for record in records:
        origin = by_sample_id[record.sample_id.rsplit("_", 1)[0]]
        variant = record.augmentations[-1]
        if variant == "scan":
            assert record.capture_type == "scan"
        elif variant == "photo":
            assert record.capture_type == "photo"
        else:
            assert record.capture_type == origin.capture_type


def test_every_variant_changes_pixels(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, originals, records = augmented
    by_sample_id = {record.sample_id: record for record in originals}

    for record in records:
        origin = by_sample_id[record.sample_id.rsplit("_", 1)[0]]
        with (
            Image.open(root / origin.image_path) as before,
            Image.open(root / record.image_path) as after,
        ):
            diff = ImageChops.difference(before.convert("RGB"), after.convert("RGB"))
            assert diff.getbbox() is not None, record.sample_id


def test_manifest_and_summary(
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, _, records = augmented

    lines = (root / "augmented_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert [json.loads(line)["sample_id"] for line in lines] == [
        record.sample_id for record in records
    ]

    summary = json.loads((root / "augmented_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == len(records)
    assert summary["variants"] == list(DEFAULT_VARIANTS)
    assert summary["by_augmentation"] == {variant: 15 for variant in DEFAULT_VARIANTS}
    assert summary["by_variant"] == {"authentic": 35, "tampered": 40}
    assert summary["by_attack"]["None"] == 35


def test_augmentation_is_deterministic(
    tmp_path: Path,
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, originals, _ = augmented
    origin = originals[0]

    first = apply_augmentation(origin, "photo", tmp_path / "a", source_dir=root, seed=7)
    second = apply_augmentation(origin, "photo", tmp_path / "b", source_dir=root, seed=7)

    assert first.document_id == second.document_id
    assert first.field_boxes == second.field_boxes
    assert first.regions == second.regions
    assert first.page_size == second.page_size


def test_chained_augmentations_keep_the_original_origin(
    tmp_path: Path,
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, originals, _ = augmented
    origin = originals[0]

    first = apply_augmentation(origin, "scan", tmp_path, source_dir=root, seed=3)
    second = apply_augmentation(first, "jpeg", tmp_path, source_dir=tmp_path, seed=3)

    assert second.augmentations == ("scan", "jpeg")
    assert second.origin_id == origin.document_id
    assert second.parent_id == origin.parent_id


def test_invalid_variants_are_rejected(
    tmp_path: Path,
    augmented: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, originals, _ = augmented

    with pytest.raises(ValueError):
        apply_augmentation(originals[0], "unknown", tmp_path)
    with pytest.raises(ValueError):
        augment_dataset(originals, tmp_path, variants=("unknown",))
