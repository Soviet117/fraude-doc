"""Pipeline integral F3+F4+F5+F6+F7 para la demo y el informe.

Encadena las cuatro capas del plan (calidad, forense, OCR/reglas y consistencia
semántica) con las tres ramas experimentales: clasificador de página F4,
localización F5 y fusión calibrada F7. El resultado es una ``Observation``
completa más las evidencias, el mapa de calor y la explicación.

Los imports de torch/OCR son perezosos: este módulo debe poder importarse en un
entorno sin la extra ``ml`` (CI) y la demo decide qué modelos están disponibles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from fraude.core.schemas import FusionReport, Observation, RiskLevel, Severity
from fraude.ingest.hashing import sha256_bytes, sha256_file
from fraude.nlp.explain import Explanation, explain_observation
from fraude.nlp.semantic import SemanticReport, analyze_semantics
from fraude.pipeline.run import analyze_bytes, analyze_document
from fraude.registry.cross import RegistryEntry

LOGGER = logging.getLogger("fraude.pipeline.full")

DEFAULT_DATA = Path("data/synthetic/v2")
DEFAULT_ARTIFACTS = Path("artifacts/runs")
DEFAULT_F4_CHECKPOINT = Path("artifacts/models/f4/fold_2.pt")
DEFAULT_F5_CHECKPOINT = Path("artifacts/models/f5/fold_0.pt")
DEFAULT_F7_MODEL = Path("artifacts/models/f7/fusion.joblib")

_MODEL_CACHE: dict[tuple[str, str, float], Any] = {}
_DUPLICATE_CACHE: dict[str, Any] = {}


@dataclass
class FullAnalysis:
    """Resultado integral serializable para la demo, SQLite y el PDF."""

    observation: Observation
    semantic: SemanticReport | None
    explanation: Explanation
    f4_score: float | None = None
    f5_summary: dict[str, float] | None = None
    f5_overlay_path: str | None = None
    f5_map_path: str | None = None
    fusion_details: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    artifact_dir: str | None = None

    @property
    def p_fake(self) -> float | None:
        return self.observation.fusion.p_fake

    @property
    def risk(self) -> RiskLevel:
        return self.observation.fusion.risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.observation.document_id,
            "observation": self.observation.model_dump(mode="json"),
            "semantic": self.semantic.to_dict() if self.semantic is not None else None,
            "fusor": {
                "p_fake": self.observation.fusion.p_fake,
                "risk": self.observation.fusion.risk.value,
                "raw_score": self.fusion_details.get("raw_score"),
                "calibration_method": self.fusion_details.get("calibration_method"),
                "contributions": self.fusion_details.get("contributions", []),
                "evidences": [
                    evidence.model_dump(mode="json")
                    for evidence in self.observation.fusion.evidences
                ],
            },
            "f4_score": self.f4_score,
            "f5": {
                "summary": self.f5_summary,
                "overlay_path": self.f5_overlay_path,
                "map_path": self.f5_map_path,
            },
            "explanation": self.explanation.to_dict(),
            "timings_ms": self.timings_ms,
            "model_versions": self.model_versions,
            "artifact_dir": self.artifact_dir,
        }


def _cached(path: Path, loader) -> Any:
    key = (str(path), loader.__name__, path.stat().st_mtime)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = loader(path)
    return _MODEL_CACHE[key]


def _get_duplicate_index(data_dir: Path):
    key = str(Path(data_dir).resolve())
    if key in _DUPLICATE_CACHE:
        return _DUPLICATE_CACHE[key]
    try:
        from fraude.nlp.duplicates import DuplicateIndex
        from fraude.synth.attacks import load_manifest

        manifest = Path(data_dir) / "manifest.jsonl"
        if not manifest.is_file():
            _DUPLICATE_CACHE[key] = None
            return None
        records = [record for record in load_manifest(manifest) if record.is_authentic]
        index = DuplicateIndex.from_records(records, Path(data_dir))
        _DUPLICATE_CACHE[key] = index
        return index
    except (ImportError, OSError, ValueError) as exc:
        LOGGER.warning("No se pudo construir el índice de duplicados: %s", exc)
        _DUPLICATE_CACHE[key] = None
        return None


def _load_image(source: bytes | Path) -> Image.Image:
    if isinstance(source, bytes):
        with Image.open(BytesIO(source)) as image:
            return image.convert("RGB").copy()
    with Image.open(Path(source)) as image:
        return image.convert("RGB").copy()


def _run_f4(image: Image.Image, checkpoint: Path, device: str = "cpu") -> float | None:
    if not Path(checkpoint).is_file():
        return None
    try:
        from fraude.models.infer import load_checkpoint, predict_image

        model, config = _cached(
            Path(checkpoint),
            lambda path: load_checkpoint(path, device=device),
        )
        return float(
            predict_image(
                model,
                image,
                image_size=int(config.get("image_size", 224)),
            )
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("F4 no disponible: %s", exc)
        return None


def _run_f5(
    image: Image.Image,
    checkpoint: Path,
    *,
    artifact_dir: Path | None,
    prefix: str,
) -> tuple[dict[str, float] | None, str | None, str | None]:
    if not Path(checkpoint).is_file():
        return None, None, None
    try:
        import numpy as np

        from fraude.models.seg_infer import (
            heatmap_overlay,
            load_checkpoint,
            localization_summary,
            predict_mask,
        )

        model, config = _cached(Path(checkpoint), load_checkpoint)
        probability = predict_mask(
            model,
            image,
            image_size=int(config.get("image_size", 256)),
        )
        summary = localization_summary(probability)
        if artifact_dir is None:
            return summary, None, None
        artifact_dir.mkdir(parents=True, exist_ok=True)
        overlay = heatmap_overlay(image, probability)
        overlay_path = artifact_dir / f"{prefix}_f5_overlay.png"
        overlay.save(overlay_path)
        map_path = artifact_dir / f"{prefix}_f5_map.npy"
        np.save(map_path, probability.astype(np.float32))
        return summary, str(overlay_path), str(map_path)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("F5 no disponible: %s", exc)
        return None, None, None


def _run_f7(
    observation: Observation,
    *,
    model_path: Path,
    f4_score: float | None,
    f5_summary: dict[str, float] | None,
    semantic: SemanticReport | None,
) -> dict[str, Any]:
    if not Path(model_path).is_file():
        return {}
    try:
        from fraude.fusion.infer import load_fusion_model, predict_fusion_details

        bundle = _cached(Path(model_path), load_fusion_model)
        return predict_fusion_details(
            observation,
            bundle=bundle,
            f4_score=f4_score,
            f5_summary=f5_summary,
            semantic=semantic,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("F7 no disponible: %s", exc)
        return {}


def _save_upload(data: bytes, artifact_dir: Path, document_id: str, filename: str) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".png"
    path = artifact_dir / f"{document_id.removeprefix('sha256:')[:16]}{suffix}"
    path.write_bytes(data)
    return path


def analyze_full(
    source: bytes | Path,
    *,
    filename: str | None = None,
    registry: list[RegistryEntry] | None = None,
    data_dir: Path = DEFAULT_DATA,
    artifacts_dir: Path | None = DEFAULT_ARTIFACTS,
    duplicate_index: Any | None = None,
    exclude_sample_id: str | None = None,
    use_llm: bool = False,
    today: date | None = None,
    f4_checkpoint: Path = DEFAULT_F4_CHECKPOINT,
    f5_checkpoint: Path = DEFAULT_F5_CHECKPOINT,
    f7_model: Path = DEFAULT_F7_MODEL,
) -> FullAnalysis:
    """Analiza una imagen con todas las capas y devuelve un resultado integral."""
    started = perf_counter()
    if isinstance(source, bytes):
        name = filename or "documento.png"
        document_id = sha256_bytes(source)
    else:
        path = Path(source)
        name = filename or path.name
        document_id = sha256_file(path)

    if isinstance(source, bytes):
        observation = analyze_bytes(
            source,
            name,
            artifacts_dir=None,
            registry=registry,
            today=today,
        )
    else:
        observation = analyze_document(
            Path(source),
            artifacts_dir=None,
            registry=registry,
            today=today,
        )

    artifact_dir = (
        Path(artifacts_dir) / document_id.removeprefix("sha256:")[:16] if artifacts_dir else None
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    image: Image.Image | None = None
    if observation.quality.ok:
        try:
            image = _load_image(source)
        except (OSError, ValueError) as exc:
            LOGGER.warning("No se pudo abrir la imagen para F4/F5: %s", exc)

    f4_score: float | None = None
    f5_summary: dict[str, float] | None = None
    f5_overlay: str | None = None
    f5_map: str | None = None
    if image is not None:
        model_started = perf_counter()
        f4_score = _run_f4(image, Path(f4_checkpoint))
        f5_summary, f5_overlay, f5_map = _run_f5(
            image,
            Path(f5_checkpoint),
            artifact_dir=artifact_dir,
            prefix=document_id.removeprefix("sha256:")[:16],
        )
        observation.timings["models_ms"] = round((perf_counter() - model_started) * 1000)

    semantic: SemanticReport | None = None
    if image is not None:
        index = duplicate_index
        if index is None:
            index = _get_duplicate_index(Path(data_dir))
        semantic = analyze_semantics(
            observation,
            image=image,
            duplicate_index=index,
            exclude_sample_id=exclude_sample_id,
            today=today,
        )
    else:
        semantic = analyze_semantics(observation, today=today)

    fusion_started = perf_counter()
    fusion_details: dict[str, Any] = {}
    if observation.quality.ok:
        fusion_details = _run_f7(
            observation,
            model_path=Path(f7_model),
            f4_score=f4_score,
            f5_summary=f5_summary,
            semantic=semantic,
        )
        if fusion_details:
            observation.fusion = fusion_details["report"]
        else:
            observation.fusion = FusionReport(
                risk=RiskLevel.UNDETERMINED,
                evidences=[
                    {
                        "code": "F7_NO_DISPONIBLE",
                        "severity": Severity.LOW,
                        "message": "La fusión F7 no está disponible; se muestran las señales base.",
                    }
                ],
            )
    else:
        observation.fusion = FusionReport(
            risk=RiskLevel.UNDETERMINED,
            evidences=[
                {
                    "code": "RECAPTURAR",
                    "severity": Severity.HIGH,
                    "message": (
                        "La calidad de la imagen no permite un análisis fiable. "
                        "La salida correcta es recapturar el documento."
                    ),
                }
            ],
        )
    observation.timings["fusion_ms"] = round((perf_counter() - fusion_started) * 1000)

    explanation_started = perf_counter()
    explanation = explain_observation(
        observation,
        semantic=semantic,
        f4_score=f4_score,
        f5_summary=f5_summary,
        use_llm=use_llm,
    )
    observation.timings["explanation_ms"] = round((perf_counter() - explanation_started) * 1000)
    observation.timings["total_ms"] = round((perf_counter() - started) * 1000)

    model_versions: dict[str, str] = {}
    if f4_score is not None:
        model_versions["f4"] = Path(f4_checkpoint).stem
    if f5_summary is not None:
        model_versions["f5"] = Path(f5_checkpoint).stem
    if fusion_details:
        model_versions["f7"] = Path(f7_model).stem
    model_versions["explanation"] = explanation.source

    if isinstance(source, bytes) and artifact_dir is not None:
        _save_upload(source, artifact_dir, document_id, name)

    return FullAnalysis(
        observation=observation,
        semantic=semantic,
        explanation=explanation,
        f4_score=f4_score,
        f5_summary=f5_summary,
        f5_overlay_path=f5_overlay,
        f5_map_path=f5_map,
        fusion_details=fusion_details,
        timings_ms=dict(observation.timings),
        model_versions=model_versions,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
    )
