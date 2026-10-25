"""Registros del dataset sintético (etiquetas independientes del contrato Observation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DOCUMENT_TYPES = (
    "dni",
    "certificado_estudios",
    "partida_nacimiento",
    "ficha_matricula",
    "acta_evaluacion",
    "certificado_modulo",
    "titulo_auxiliar_tecnico",
)

ATTACK_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


@dataclass(frozen=True)
class Region:
    """Región alterada de una muestra sintética."""

    attack_type: str
    bbox: tuple[int, int, int, int]
    field: str | None = None
    original_value: str | None = None
    altered_value: str | None = None
    mask_path: str | None = None


@dataclass(frozen=True)
class DocumentRecord:
    """Una muestra del dataset, con sus rutas, etiquetas y cajas de campos."""

    sample_id: str
    document_id: str
    source_id: str
    document_type: str
    template_version: str
    variant: str
    parent_id: str | None
    is_authentic: bool
    attack_types: tuple[str, ...] = ()
    primary_attack: str | None = None
    pdf_path: str | None = None
    image_path: str = ""
    mask_paths: tuple[str, ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)
    field_boxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    page_size: tuple[int, int] = (0, 0)
    regions: tuple[Region, ...] = ()
    semantic_flags: tuple[str, ...] = ()
    capture_type: str = "pdf_render"
    seed: int = 42
    augmentations: tuple[str, ...] = ()
    origin_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte el registro a un diccionario serializable en JSON."""
        return asdict(self)
