"""Rama semántica de F6: consistencia de entidades y reutilización de documentos."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from fraude.core.schemas import Observation
from fraude.nlp.duplicates import DuplicateIndex, DuplicateMatch
from fraude.nlp.normalize import (
    canonical_name,
    content_fingerprint,
    name_similarity,
    normalize_date,
)

MIN_NAME_SIMILARITY = 0.85


@dataclass(frozen=True)
class SemanticReport:
    duplicate: DuplicateMatch | None = None
    name_consistency: float | None = None
    flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate": (
                None
                if self.duplicate is None
                else {
                    "sample_id": self.duplicate.sample_id,
                    "source_id": self.duplicate.source_id,
                    "document_type": self.duplicate.document_type,
                    "distance": self.duplicate.distance,
                    "exact": self.duplicate.exact,
                }
            ),
            "name_consistency": self.name_consistency,
            "flags": list(self.flags),
            "notes": list(self.notes),
        }


def analyze_semantics(
    observation: Observation,
    *,
    image: Image.Image | None = None,
    duplicate_index: DuplicateIndex | None = None,
    exclude_sample_id: str | None = None,
    today: date | None = None,
) -> SemanticReport:
    """Genera banderas semánticas explicables a partir de una observación."""
    fields = {
        key: str(value) for key, value in observation.fields.items() if value not in (None, "")
    }
    flags: list[str] = []
    notes: list[str] = []
    current_day = today or date.today()

    consistency: float | None = None
    name = canonical_name(fields)
    if name is not None:
        candidates = [
            value
            for key, value in fields.items()
            if key in {"nombre", "estudiante", "apellidos"} or key.startswith("estudiante_")
        ]
        scores = [name_similarity(name, candidate) for candidate in candidates]
        if scores:
            consistency = round(min(scores), 4)
            if consistency < MIN_NAME_SIMILARITY:
                flags.append("name_mismatch")
                notes.append(
                    "El nombre del titular no coincide entre campos "
                    f"(similitud mínima {consistency:.2f})."
                )

    dni_values = {
        value
        for key, value in fields.items()
        if key == "dni" or key.startswith("dni_")
        if len(value) == 8 and value.isdigit()
    }
    if len(dni_values) > 1:
        flags.append("dni_mismatch")
        notes.append("Hay más de un DNI distinto en el documento.")

    emission = normalize_date(fields.get("fecha_emision", ""))
    if emission is not None:
        year = int(emission[-4:])
        if year < 1990 or year > current_day.year + 1:
            flags.append("implausible_emission_date")
            notes.append(f"La fecha de emisión ({emission}) no es plausible.")

    duplicate: DuplicateMatch | None = None
    if image is not None and duplicate_index is not None:
        duplicate = duplicate_index.query(
            image,
            text=content_fingerprint(fields),
            document_type=observation.doctype.label,
            exclude_sample_id=exclude_sample_id,
        )
        if duplicate is not None:
            flags.append("document_reuse")
            notes.append(
                "La imagen coincide con un documento auténtico ya conocido "
                f"({duplicate.sample_id}, distancia {duplicate.distance})."
            )

    return SemanticReport(
        duplicate=duplicate,
        name_consistency=consistency,
        flags=tuple(sorted(set(flags))),
        notes=tuple(notes),
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
