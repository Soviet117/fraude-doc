"""Pruebas del cruce con registro interno (F3, E3-5)."""

from __future__ import annotations

import json
from pathlib import Path

from fraude.registry.build import build_registry_from_manifest
from fraude.registry.cross import (
    RegistryEntry,
    cross_registry,
    load_registry,
    save_registry,
)

ENTRIES = [
    RegistryEntry(
        source_id="persona_0000",
        nombre="Quispe Mamani, José Luis",
        dni="12345678",
        programa="Computación e Informática",
        modulo="Ofimática y herramientas digitales",
        periodo="2025-I",
        fecha_nacimiento="15/03/1990",
    ),
    RegistryEntry(
        source_id="persona_0001",
        nombre="Flores Rojas, Ana María",
        dni="87654321",
        programa="Enfermería Técnica",
        modulo="Primeros auxilios",
        periodo="2025-II",
        fecha_nacimiento="20/08/1995",
    ),
]


def test_exact_dni_has_no_diffs() -> None:
    report = cross_registry(
        {
            "estudiante": "Quispe Mamani, José Luis",
            "dni": "12345678",
            "programa": "Computación e Informática",
            "modulo": "Ofimática y herramientas digitales",
            "periodo": "2025-I",
            "fecha_nacimiento": "15/03/1990",
        },
        ENTRIES,
    )

    assert report.matched is True
    assert report.score is not None
    assert report.score >= 0.95
    assert report.diffs == []


def test_name_order_and_accents_are_matched_without_dni() -> None:
    report = cross_registry(
        {
            "nombre": "Jose Luis Quispe Mamani",
            "programa": "Computacion e Informatica",
            "periodo": "2025-I",
        },
        ENTRIES,
    )

    assert report.matched is True
    assert report.score is not None
    assert report.score >= 0.86
    assert report.diffs == []


def test_exact_name_without_dni_or_program_matches() -> None:
    report = cross_registry(
        {"nombre": "José Luis Quispe Mamani"},
        ENTRIES,
    )

    assert report.matched is True
    assert report.score is not None
    assert report.score >= 0.86
    assert report.diffs == []


def test_exact_dni_with_wrong_program_reports_diff() -> None:
    report = cross_registry(
        {
            "estudiante": "Quispe Mamani, José Luis",
            "dni": "12345678",
            "programa": "Programa Falso",
        },
        ENTRIES,
    )

    assert report.matched is True
    assert any(diff.startswith("programa:") for diff in report.diffs)


def test_ocr_variant_program_and_module_do_not_report_diff() -> None:
    report = cross_registry(
        {
            "estudiante": "Quispe Mamani, José Luis",
            "dni": "12345678",
            "programa": "Computacion el Informâtica",
            "modulo": "Ofimática yt herramientas digitales",
            "periodo": "2025-I",
        },
        ENTRIES,
    )

    assert report.matched is True
    assert report.diffs == []


def test_real_program_difference_still_reports_diff() -> None:
    report = cross_registry(
        {
            "estudiante": "Quispe Mamani, José Luis",
            "dni": "12345678",
            "programa": "Enfermería Técnica",
        },
        ENTRIES,
    )

    assert any(diff.startswith("programa:") for diff in report.diffs)


def test_multi_student_acta_is_not_crossed_against_one_student() -> None:
    report = cross_registry(
        {
            "programa": "Computación e Informática",
            "modulo": "Ofimática y herramientas digitales",
            "periodo": "2025-I",
            "estudiante_0": "Otra Persona",
            "dni_0": "00000000",
            "nota_0": "16",
            "estudiante_1": "Flores Rojas, Ana María",
            "dni_1": "87654321",
            "nota_1": "15",
        },
        ENTRIES,
    )

    assert report.matched is False
    assert report.score is None
    assert "varios estudiantes" in report.diffs[0]


def test_unknown_person_does_not_match() -> None:
    report = cross_registry(
        {
            "estudiante": "Persona Inexistente",
            "dni": "00000000",
            "programa": "Programa Falso",
        },
        ENTRIES,
    )

    assert report.matched is False
    assert report.score is not None
    assert report.score < 0.86
    assert "sin coincidencia suficiente" in report.diffs[0]


def test_empty_registry_is_reported() -> None:
    report = cross_registry({"dni": "12345678"}, [])

    assert report.matched is False
    assert report.score is None
    assert report.diffs == ["registro vacío o no disponible"]


def test_save_and_load_jsonl_roundtrip(tmp_path: Path) -> None:
    path = save_registry(ENTRIES, tmp_path / "registry.jsonl")

    loaded = load_registry(path)

    assert loaded == ENTRIES


def test_load_csv_registry(tmp_path: Path) -> None:
    path = tmp_path / "students.csv"
    path.write_text(
        "nombre,dni,programa,modulo,periodo\n"
        "Quispe Mamani José Luis,12345678,Computación e Informática,"
        "Ofimática y herramientas digitales,2025-I\n",
        encoding="utf-8",
    )

    loaded = load_registry(path)

    assert len(loaded) == 1
    assert loaded[0].dni == "12345678"
    assert loaded[0].nombre == "Quispe Mamani José Luis"


def test_build_registry_from_manifest_aggregates_identities(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {
            "source_id": "persona_0000",
            "sample_id": "dni_0000",
            "document_type": "dni",
            "is_authentic": True,
            "fields": {
                "numero": "12345678",
                "apellidos": "Quispe",
                "nombres": "José Luis",
                "fecha_nacimiento": "15/03/1990",
            },
        },
        {
            "source_id": "persona_0000",
            "sample_id": "ficha_0000",
            "document_type": "ficha_matricula",
            "is_authentic": True,
            "fields": {
                "estudiante": "Quispe Mamani, José Luis",
                "dni": "12345678",
                "programa": "Computación e Informática",
                "modulo": "Ofimática y herramientas digitales",
                "periodo": "2025-I",
            },
        },
        {
            "source_id": "persona_0000",
            "sample_id": "ficha_0000_t1",
            "document_type": "ficha_matricula",
            "is_authentic": False,
            "fields": {
                "estudiante": "Nombre Alterado",
                "dni": "99999999",
                "programa": "Programa Falso",
            },
        },
        {
            "source_id": "persona_0001",
            "sample_id": "dni_0001",
            "document_type": "dni",
            "is_authentic": True,
            "fields": {
                "numero": "87654321",
                "apellidos": "Flores",
                "nombres": "Ana María",
                "fecha_nacimiento": "20/08/1995",
            },
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    entries = build_registry_from_manifest(manifest)

    assert len(entries) == 2
    first = entries[0]
    assert first.source_id == "persona_0000"
    assert first.nombre == "Quispe Mamani, José Luis"
    assert first.dni == "12345678"
    assert first.programa == "Computación e Informática"
    assert first.fecha_nacimiento == "15/03/1990"
    assert entries[1].dni == "87654321"
