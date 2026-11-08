"""Detección de documentos reutilizados con hash exacto y perceptual (F6).

El corpus sintético T7 reutiliza una imagen auténtica de otra persona. Una
copia exacta se detecta comparando los píxeles decodificados. Un reenvío
recomprimido o reescaneado cambia los píxeles, así que además se usa un dHash
de 64 bits. El dHash de una página está dominado por la plantilla: dos
documentos distintos del mismo tipo pueden compartir hash. Por eso los casi
duplicados exigen también una huella de contenido (campos) similar.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

from fraude.nlp.normalize import content_fingerprint
from fraude.synth.dataset import DocumentRecord

DEFAULT_MAX_DISTANCE = 6
DEFAULT_MIN_TEXT_SIMILARITY = 0.95


@dataclass(frozen=True)
class DuplicateMatch:
    sample_id: str
    source_id: str
    document_type: str
    distance: int
    exact: bool


@dataclass(frozen=True)
class _IndexEntry:
    sample_id: str
    source_id: str
    document_type: str
    pixel_hash: str
    dhash: int
    text: str


class DuplicateIndex:
    """Índice de documentos auténticos conocidos."""

    def __init__(self, entries: list[_IndexEntry] | None = None) -> None:
        self._entries = list(entries or [])

    @classmethod
    def from_records(
        cls,
        records: list[DocumentRecord],
        data_dir: Path,
        *,
        only_authentic: bool = True,
    ) -> DuplicateIndex:
        entries: list[_IndexEntry] = []
        for record in records:
            if only_authentic and not record.is_authentic:
                continue
            path = Path(data_dir) / record.image_path
            if not path.is_file():
                continue
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                entries.append(
                    _IndexEntry(
                        sample_id=record.sample_id,
                        source_id=record.source_id,
                        document_type=record.document_type,
                        pixel_hash=pixel_hash(rgb),
                        dhash=dhash(rgb),
                        text=content_fingerprint(record.fields),
                    )
                )
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def query(
        self,
        image: Image.Image,
        *,
        text: str | None = None,
        document_type: str | None = None,
        exclude_sample_id: str | None = None,
        max_distance: int = DEFAULT_MAX_DISTANCE,
        min_text_similarity: float = DEFAULT_MIN_TEXT_SIMILARITY,
    ) -> DuplicateMatch | None:
        """Devuelve la coincidencia más cercana o ``None``.

        Una copia exacta se reconoce por píxeles. Un casi duplicado necesita
        ``text`` y una similitud de contenido alta; sin texto no se arriesga un
        falso positivo por plantilla compartida.
        """
        rgb = image.convert("RGB")
        probe_hash = pixel_hash(rgb)
        probe_dhash = dhash(rgb)
        best: DuplicateMatch | None = None
        for entry in self._entries:
            if exclude_sample_id is not None and entry.sample_id == exclude_sample_id:
                continue
            if document_type is not None and entry.document_type != document_type:
                continue
            if entry.pixel_hash == probe_hash:
                return DuplicateMatch(
                    sample_id=entry.sample_id,
                    source_id=entry.source_id,
                    document_type=entry.document_type,
                    distance=0,
                    exact=True,
                )
            if text is None:
                continue
            distance = hamming_distance(probe_dhash, entry.dhash)
            if distance > max_distance:
                continue
            if _text_similarity(text, entry.text) < min_text_similarity:
                continue
            if best is None or distance < best.distance:
                best = DuplicateMatch(
                    sample_id=entry.sample_id,
                    source_id=entry.source_id,
                    document_type=entry.document_type,
                    distance=distance,
                    exact=False,
                )
        return best


def pixel_hash(image: Image.Image) -> str:
    """SHA-256 de los píxeles RGB decodificados (ignora el contenedor)."""
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def dhash(image: Image.Image, *, hash_size: int = 8) -> int:
    """dHash de ``hash_size`` bits por lado; devuelve un entero."""
    if hash_size < 1:
        raise ValueError("hash_size debe ser positivo")
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def _text_similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return round(float(fuzz.token_set_ratio(first, second)) / 100.0, 4)
    except ImportError:
        return round(SequenceMatcher(None, first, second).ratio(), 4)
