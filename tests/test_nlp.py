"""Pruebas de la rama semántica F6."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from fraude.core.schemas import (
    DoctypeInfo,
    Observation,
    RuleFinding,
    Severity,
    SourceInfo,
)
from fraude.nlp.duplicates import DuplicateIndex, dhash, hamming_distance
from fraude.nlp.explain import explain_observation, llm_explanation, template_explanation
from fraude.nlp.normalize import (
    content_fingerprint,
    field_matches,
    name_similarity,
    normalize_name,
)
from fraude.nlp.semantic import analyze_semantics
from fraude.synth.dataset import DocumentRecord


def _observation(fields: dict[str, str]) -> Observation:
    return Observation(
        document_id="sha256:" + "0" * 64,
        source=SourceInfo(filename="sample.png", bytes=1),
        doctype=DoctypeInfo(label="dni", confidence=1.0),
        fields=fields,
        rules=[
            RuleFinding(
                code="FECHA_INVALIDA",
                severity=Severity.HIGH,
                message="La fecha de emisión no es válida.",
            )
        ],
    )


def _record(
    tmp_path: Path,
    sample_id: str,
    image: Image.Image,
    *,
    fields: dict[str, str] | None = None,
) -> DocumentRecord:
    path = tmp_path / "images" / f"{sample_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return DocumentRecord(
        sample_id=sample_id,
        document_id="sha256:" + "0" * 64,
        source_id=f"persona_{sample_id}",
        document_type="dni",
        template_version="v1",
        variant="authentic",
        parent_id=None,
        is_authentic=True,
        image_path=f"images/{sample_id}.png",
        fields=fields or {},
    )


def test_normalize_name_and_similarity() -> None:
    assert normalize_name("  Huamán,   José Alberto ") == "huaman jose alberto"
    assert name_similarity("Huamán, José Alberto", "José Alberto Huamán") >= 0.9
    assert name_similarity("Huamán, José Alberto", "Rojas, María") < 0.5


def test_field_matches_kinds() -> None:
    assert field_matches("12345678", "12345678", kind="dni")
    assert not field_matches("12345678", "87654321", kind="dni")
    assert field_matches("15/04/2026", "15/04/2026", kind="date")
    assert field_matches("José Alberto", "JOSE ALBERTO", kind="text")


def test_dhash_and_duplicate_index(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), "white")
    for x in range(10, 30):
        for y in range(10, 30):
            image.putpixel((x, y), (0, 0, 0))
    other = Image.new("RGB", (64, 64), "white")
    for x in range(40, 60):
        for y in range(40, 60):
            other.putpixel((x, y), (0, 0, 0))

    assert hamming_distance(dhash(image), dhash(image)) == 0
    assert hamming_distance(dhash(image), dhash(other)) > 0

    record = _record(tmp_path, "doc_0000", image, fields={"dni": "12345678"})
    index = DuplicateIndex.from_records([record], tmp_path)
    match = index.query(image, exclude_sample_id="other")
    assert match is not None
    assert match.sample_id == "doc_0000"
    assert match.distance == 0
    assert match.exact
    assert index.query(image, exclude_sample_id="doc_0000") is None


def test_near_duplicate_requires_text_gate(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), "white")
    for x in range(10, 30):
        for y in range(10, 30):
            image.putpixel((x, y), (0, 0, 0))
    record = _record(
        tmp_path,
        "doc_0000",
        image,
        fields={"dni": "12345678", "nombres": "José Alberto"},
    )
    index = DuplicateIndex.from_records([record], tmp_path)

    # Un píxel cambia el hash exacto, pero no el dHash: sin texto no se acepta.
    altered = image.copy()
    altered.putpixel((0, 0), (1, 1, 1))
    assert index.query(altered, exclude_sample_id="other") is None
    assert index.query(altered, text="otra persona", exclude_sample_id="other") is None

    match = index.query(
        altered,
        text=content_fingerprint(record.fields),
        exclude_sample_id="other",
    )
    assert match is not None
    assert match.sample_id == "doc_0000"
    assert not match.exact


def test_semantic_flags_name_and_duplicate(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), "white")
    record = _record(tmp_path, "doc_0000", image)
    index = DuplicateIndex.from_records([record], tmp_path)
    observation = _observation(
        {
            "nombre": "José Alberto Huamán",
            "estudiante": "María Rojas",
            "dni": "12345678",
            "fecha_emision": "01/01/1900",
        }
    )
    report = analyze_semantics(
        observation,
        image=image,
        duplicate_index=index,
        exclude_sample_id="query",
    )
    assert "name_mismatch" in report.flags
    assert "implausible_emission_date" in report.flags
    assert "document_reuse" in report.flags
    assert report.duplicate is not None


def test_template_explanation_mentions_findings_and_disclaimer() -> None:
    observation = _observation({"dni": "12345678"})
    explanation = explain_observation(observation)
    assert explanation.source == "template"
    assert "FECHA_INVALIDA" not in explanation.text
    assert "La fecha de emisión no es válida." in explanation.text
    assert "decisión automática" in explanation.text


def test_llm_fallback_uses_template() -> None:
    context = {
        "findings": [{"code": "X", "severity": "alta", "message": "Hallazgo de prueba."}],
        "registry": {},
        "semantic": {},
        "f4_score": None,
        "f5_summary": None,
    }
    explanation = llm_explanation(
        context,
        endpoint="http://127.0.0.1:9/api/generate",
        timeout=0.2,
    )
    assert explanation.source == "template"
    assert explanation.fallback_reason is not None
    assert explanation.text == template_explanation(context)
