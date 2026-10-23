"""Pruebas de hashing."""

from __future__ import annotations

from fraude.ingest.hashing import sha256_bytes, sha256_file


def test_sha256_bytes_known_value() -> None:
    # SHA-256 de b"abc"
    expected = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    assert sha256_bytes(b"abc") == expected


def test_sha256_file_matches_bytes(tmp_path) -> None:
    path = tmp_path / "documento.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == sha256_bytes(b"abc")


def test_sha256_file_reads_large_files_in_chunks(tmp_path) -> None:
    path = tmp_path / "grande.bin"
    data = b"x" * (3 * 1024 * 1024 + 17)
    path.write_bytes(data)

    assert sha256_file(path) == sha256_bytes(data)
