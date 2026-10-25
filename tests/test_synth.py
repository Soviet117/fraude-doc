"""Pruebas del generador de documentos auténticos sintéticos (F2.2)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from fraude.core.schemas import CaptureType
from fraude.preprocess.quality import assess_image_path
from fraude.synth.data import build_fields, generate_identity
from fraude.synth.dataset import DOCUMENT_TYPES, DocumentRecord
from fraude.synth.generate import generate_dataset
from fraude.synth.render import render_document

EXPECTED_BOXES: dict[str, set[str]] = {
    "dni": {"numero", "nombres", "apellidos", "fecha_nacimiento", "foto"},
    "certificado_estudios": {"estudiante", "dni", "institucion", "promedio", "firma"},
    "partida_nacimiento": {"nombre", "fecha_nacimiento", "padre", "madre", "numero_acta"},
    "ficha_matricula": {"estudiante", "dni", "programa", "modulo", "firma"},
    "acta_evaluacion": {"programa", "docente", "estudiante_0", "nota_0", "nota_4"},
    "certificado_modulo": {"estudiante", "dni", "modulo", "nota", "sello"},
    "titulo_auxiliar_tecnico": {"estudiante", "dni", "programa", "resolucion", "firma"},
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[DocumentRecord]]:
    out_dir = tmp_path_factory.mktemp("synth")
    records = generate_dataset(out_dir, count=7, seed=42, dpi=72, identities=7)
    return out_dir, records


def test_generates_every_document_type(dataset: tuple[Path, list[DocumentRecord]]) -> None:
    _, records = dataset

    assert len(records) == 7
    assert {record.document_type for record in records} == set(DOCUMENT_TYPES)


def test_manifest_is_jsonl_with_one_record_per_sample(
    dataset: tuple[Path, list[DocumentRecord]],
) -> None:
    out_dir, records = dataset

    lines = (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    parsed = [json.loads(line) for line in lines]
    assert [item["sample_id"] for item in parsed] == [record.sample_id for record in records]

    summary = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
    assert summary["count"] == 7
    assert summary["seed"] == 42
    assert summary["by_variant"] == {"authentic": 7}
    assert summary["by_document_type"] == {name: 1 for name in DOCUMENT_TYPES}


def test_files_exist_and_paths_are_relative(
    dataset: tuple[Path, list[DocumentRecord]],
) -> None:
    out_dir, records = dataset

    for record in records:
        assert record.is_authentic is True
        assert record.variant == "authentic"
        assert record.attack_types == ()
        assert record.regions == ()
        assert not Path(record.image_path).is_absolute()
        assert not Path(record.pdf_path or "").is_absolute()
        assert (out_dir / record.image_path).is_file()
        assert (out_dir / (record.pdf_path or "")).is_file()
        assert record.document_id.startswith("sha256:")


def test_field_boxes_are_inside_the_page(
    dataset: tuple[Path, list[DocumentRecord]],
) -> None:
    _, records = dataset

    for record in records:
        width, height = record.page_size
        assert width > 0 and height > 0
        for key, (x, y, box_width, box_height) in record.field_boxes.items():
            assert box_width > 0 and box_height > 0, key
            assert 0 <= x <= width, key
            assert 0 <= y <= height, key
            assert x + box_width <= width, key
            assert y + box_height <= height, key


def test_each_template_registers_expected_fields(
    dataset: tuple[Path, list[DocumentRecord]],
) -> None:
    _, records = dataset

    for record in records:
        expected = EXPECTED_BOXES[record.document_type]
        assert expected <= set(record.field_boxes), record.document_type


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = generate_dataset(tmp_path / "a", count=7, seed=7, dpi=72, identities=7)
    second = generate_dataset(tmp_path / "b", count=7, seed=7, dpi=72, identities=7)

    assert [(item.sample_id, item.document_id) for item in first] == [
        (item.sample_id, item.document_id) for item in second
    ]
    assert [(item.fields, item.field_boxes) for item in first] == [
        (item.fields, item.field_boxes) for item in second
    ]


def test_generated_document_passes_quality_gate(tmp_path: Path) -> None:
    rng = random.Random(11)
    identity = generate_identity(rng, 0, set())
    fields = build_fields(rng, "dni", identity)
    record = render_document(
        sample_id="dni_quality",
        document_type="dni",
        identity=identity,
        fields=fields,
        out_dir=tmp_path,
        dpi=150,
    )

    report, capture_type, dpi = assess_image_path(tmp_path / record.image_path)

    assert report.ok is True, report.issues
    assert capture_type is CaptureType.SCAN
    assert dpi == 150
    assert record.page_size[0] >= 1200


def test_invalid_parameters_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, count=0)
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, count=1, dpi=72 - 1)
