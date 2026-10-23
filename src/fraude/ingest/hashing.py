"""Hashing criptográfico para identificación y deduplicación de documentos."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Devuelve el identificador ``sha256:<hex>`` de un contenido en memoria."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Devuelve el identificador ``sha256:<hex>`` de un archivo, sin cargarlo entero."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
