"""Contrato de datos del sistema (ver plan §7.1).

`Observation` es la interfaz entre todos los módulos y el registro de auditoría.
Cualquier cambio de esquema debe ser compatible hacia atrás y acordarse con el
equipo: lo consumen el pipeline, la demo y los informes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CaptureType(StrEnum):
    """Cómo llegó la imagen al sistema (determina qué señales forenses aplican)."""

    SCAN = "scan"
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    PDF_RENDER = "pdf_render"
    UNKNOWN = "unknown"


class RiskLevel(StrEnum):
    """Banda de riesgo mostrada al usuario. Nunca implica rechazo automático."""

    LOW = "bajo"
    MEDIUM = "medio"
    HIGH = "alto"
    UNDETERMINED = "indeterminado"


class Severity(StrEnum):
    """Severidad de una evidencia o regla violada."""

    LOW = "baja"
    MEDIUM = "media"
    HIGH = "alta"


class StrictModel(BaseModel):
    """Base con campos desconocidos prohibidos: detecta errores de integración."""

    model_config = ConfigDict(extra="forbid")


class SourceInfo(StrictModel):
    filename: str
    format: str | None = None
    bytes: int = Field(ge=0)


class CaptureInfo(StrictModel):
    type: CaptureType = CaptureType.UNKNOWN
    dpi_est: int | None = Field(default=None, ge=0)


class QualityReport(StrictModel):
    """Puerta de calidad: si falla, la salida correcta es 'recapturar', no 'fraude'."""

    ok: bool = True
    blur: float | None = None
    skew_deg: float | None = None
    issues: list[str] = Field(default_factory=list)


class DoctypeInfo(StrictModel):
    label: str = "desconocido"
    template_version: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SignalScore(StrictModel):
    """Resultado de una señal forense individual."""

    applicable: bool = True
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    map_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ForensicsReport(StrictModel):
    ela: SignalScore = Field(default_factory=SignalScore)
    noise: SignalScore = Field(default_factory=SignalScore)
    chromatic: SignalScore = Field(default_factory=SignalScore)
    jpeg: SignalScore = Field(default_factory=SignalScore)
    metadata: SignalScore = Field(default_factory=SignalScore)
    localization: SignalScore = Field(default_factory=SignalScore)


class OcrReport(StrictModel):
    engine: str = "doctr"
    mean_conf: float | None = Field(default=None, ge=0.0, le=1.0)
    n_words: int = Field(default=0, ge=0)
    text_path: str | None = None


class RuleFinding(StrictModel):
    code: str
    severity: Severity
    message: str


class RegistryReport(StrictModel):
    matched: bool = False
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    diffs: list[str] = Field(default_factory=list)


class Evidence(StrictModel):
    code: str
    severity: Severity
    message: str
    weight: float | None = None


class FusionReport(StrictModel):
    p_fake: float | None = Field(default=None, ge=0.0, le=1.0)
    risk: RiskLevel = RiskLevel.UNDETERMINED
    evidences: list[Evidence] = Field(default_factory=list)


class Observation(StrictModel):
    """Salida única del pipeline para un documento."""

    document_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source: SourceInfo
    capture: CaptureInfo = Field(default_factory=CaptureInfo)
    quality: QualityReport = Field(default_factory=QualityReport)
    doctype: DoctypeInfo = Field(default_factory=DoctypeInfo)
    forensics: ForensicsReport = Field(default_factory=ForensicsReport)
    ocr: OcrReport = Field(default_factory=OcrReport)
    fields: dict[str, Any] = Field(default_factory=dict)
    rules: list[RuleFinding] = Field(default_factory=list)
    registry: RegistryReport = Field(default_factory=RegistryReport)
    fusion: FusionReport = Field(default_factory=FusionReport)
    timings: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
