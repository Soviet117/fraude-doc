"""Pruebas del generador de manipulaciones T1-T8 (F2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

from fraude.synth.attacks import (
    apply_attack,
    generate_attacks,
    load_manifest,
    write_attacks_manifest,
)
from fraude.synth.dataset import ATTACK_TYPES, DocumentRecord
from fraude.synth.generate import generate_dataset


@pytest.fixture(scope="module")
def attacked(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[DocumentRecord], list[DocumentRecord]]:
    root = tmp_path_factory.mktemp("attacks")
    authentic = generate_dataset(root, count=14, identities=7, seed=42, dpi=72)
    records = generate_attacks(authentic, root, seed=42, copies=1, dpi=72)
    return root, authentic, records


def test_generates_every_attack_type(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = attacked

    assert {record.primary_attack for record in records} == set(ATTACK_TYPES)
    assert len({record.sample_id for record in records}) == len(records)


def test_each_record_has_provenance_and_label(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = attacked

    for record in records:
        assert record.variant == "tampered"
        assert record.is_authentic is False
        assert record.attack_types == (record.primary_attack,)
        assert record.parent_id is not None
        assert record.image_path.startswith("images/")
        assert record.regions, record.primary_attack


def test_local_attacks_have_masks_and_semantic_attacks_do_not(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, _, records = attacked
    local = {"T1", "T2", "T3", "T4", "T5", "T6"}

    for record in records:
        if record.primary_attack in local:
            assert len(record.mask_paths) == 1, record.primary_attack
            mask_path = root / record.mask_paths[0]
            assert mask_path.is_file()
            with Image.open(mask_path) as mask:
                assert mask.size == record.page_size
                mask_array = np.asarray(mask.convert("L"))
                coverage = int((mask_array > 127).sum())
                assert coverage > 0
                if record.primary_attack != "T6":
                    assert coverage < mask.width * mask.height
        else:
            assert record.mask_paths == ()


def test_regions_are_inside_the_page(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = attacked

    for record in records:
        page_width, page_height = record.page_size
        for region in record.regions:
            x, y, width, height = region.bbox
            assert width > 0 and height > 0
            assert 0 <= x < page_width
            assert 0 <= y < page_height
            assert x + width <= page_width
            assert y + height <= page_height


def test_attack_specific_labels(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, _, records = attacked
    by_attack = {record.primary_attack: record for record in records}

    t1 = by_attack["T1"].regions[0]
    assert t1.field not in {None, "foto", "sello"}
    assert t1.original_value is not None
    assert t1.altered_value is not None
    assert t1.original_value != t1.altered_value

    assert by_attack["T2"].regions[0].field == "sello_insertado"
    assert by_attack["T3"].regions[0].altered_value == "duplicado"
    assert by_attack["T4"].regions[0].original_value is not None
    assert by_attack["T4"].regions[0].altered_value is None
    assert by_attack["T5"].regions[0].field == "foto"
    assert by_attack["T6"].regions[0].altered_value == "documento_fabricado"
    assert "full_fabrication" in by_attack["T6"].semantic_flags
    assert "document_reuse" in by_attack["T7"].semantic_flags
    assert by_attack["T7"].regions[0].original_value != by_attack["T7"].regions[0].altered_value
    assert "semantic_inconsistency" in by_attack["T8"].semantic_flags


def test_t6_mask_covers_the_whole_page(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, _, records = attacked
    t6 = next(record for record in records if record.primary_attack == "T6")

    with Image.open(root / t6.mask_paths[0]) as mask:
        assert mask.size == t6.page_size
        assert bool((np.asarray(mask.convert("L")) == 255).all())
    assert t6.field_boxes == {}
    assert t6.template_version == "forged"


def test_t7_reuses_the_donor_image_and_keeps_its_labels(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, authentic, records = attacked
    t7 = next(record for record in records if record.primary_attack == "T7")
    donor = next(record for record in authentic if record.document_id == t7.document_id)

    assert (root / t7.image_path).read_bytes() == (root / donor.image_path).read_bytes()
    assert t7.fields == donor.fields
    assert t7.source_id == donor.source_id


def test_t8_changes_a_field_without_touching_pixels(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, authentic, records = attacked
    t8 = next(record for record in records if record.primary_attack == "T8")
    parent = next(record for record in authentic if record.document_id == t8.parent_id)
    region = t8.regions[0]

    assert region.altered_value is not None
    assert t8.fields[region.field] == region.altered_value
    assert t8.mask_paths == ()
    assert t8.page_size == parent.page_size


def test_local_attacks_really_change_pixels(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, authentic, records = attacked
    by_document_id = {record.document_id: record for record in authentic}

    for record in records:
        if record.primary_attack in {"T7", "T8"}:
            continue
        parent = by_document_id[record.parent_id]
        with (
            Image.open(root / parent.image_path) as before,
            Image.open(root / record.image_path) as after,
        ):
            diff = ImageChops.difference(before.convert("RGB"), after.convert("RGB"))
            assert diff.getbbox() is not None, record.primary_attack


def test_attack_manifest_and_loader(
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    root, authentic, records = attacked

    lines = (root / "attacks_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert [json.loads(line)["sample_id"] for line in lines] == [
        record.sample_id for record in records
    ]

    summary = json.loads((root / "attacks_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 8
    assert summary["by_attack"] == {attack: 1 for attack in ATTACK_TYPES}

    loaded = load_manifest(root / "manifest.jsonl")
    assert [record.sample_id for record in loaded] == [record.sample_id for record in authentic]
    assert loaded[0].field_boxes == authentic[0].field_boxes


def test_manifest_writer_accepts_an_empty_dataset(tmp_path: Path) -> None:
    write_attacks_manifest([], tmp_path, seed=1, copies=1)

    summary = json.loads((tmp_path / "attacks_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 0
    assert summary["by_attack"] == {}


def test_invalid_inputs_are_rejected(
    tmp_path: Path,
    attacked: tuple[Path, list[DocumentRecord], list[DocumentRecord]],
) -> None:
    _, authentic, _ = attacked

    with pytest.raises(ValueError):
        generate_attacks([], tmp_path)
    with pytest.raises(ValueError):
        generate_attacks(authentic, tmp_path, copies=0)
    with pytest.raises(ValueError):
        apply_attack(authentic[0], "T9", tmp_path)
    with pytest.raises(ValueError):
        apply_attack(authentic[0], "T7", tmp_path)
