"""Pruebas del punto de entrada del pipeline."""

from __future__ import annotations

from fraude.pipeline.run import analyze_bytes, analyze_document


def test_analyze_bytes_builds_observation() -> None:
    observation = analyze_bytes(b"abc", "acta.pdf")

    assert observation.source.format == "pdf"
    assert observation.source.bytes == 3
    assert observation.quality.ok is True
    assert observation.ocr.n_words == 0


def test_analyze_document_matches_analyze_bytes(tmp_path) -> None:
    path = tmp_path / "ficha.png"
    path.write_bytes(b"abc")

    from_file = analyze_document(path)
    from_bytes = analyze_bytes(b"abc", path.name)

    assert from_file.document_id == from_bytes.document_id
    assert from_file.source == from_bytes.source
