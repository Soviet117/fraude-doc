"""Auditoría de integridad de un corpus etiquetado.

El módulo no decide autenticidad ni calcula métricas de desempeño. Verifica que
el índice, las imágenes, los PDF y las máscaras sean coherentes antes de usar
el corpus en una evaluación o en entrenamiento. Está pensado para el corpus
local de muestras y para futuros lotes anonimizados del CETPRO.

Uso:
    uv run python -m fraude.eval.corpus_audit data/muestras \
        --out artifacts/eval/muestras_audit.json --pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from fraude.ingest.hashing import sha256_file
from fraude.pipeline.run import analyze_document
from fraude.synth.dataset import ATTACK_TYPES, DOCUMENT_TYPES

REQUIRED_COLUMNS = (
    "id",
    "tipo",
    "variante",
    "ataque",
    "persona",
    "documento_padre",
    "campo_original",
    "campo_alterado",
    "pdf",
    "png",
    "mascara",
    "bbox_px",
    "sha256_png",
)
VISUAL_ATTACKS = frozenset({"T1", "T2", "T3", "T4", "T5", "T7"})
SEMANTIC_ATTACKS = frozenset({"T6", "T8"})
VALID_VARIANTS = frozenset({"authentic", "tampered"})
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_BBOX_TOLERANCE = 1


@dataclass(frozen=True)
class AuditIssue:
    """Un problema detectado durante la auditoría."""

    sample_id: str
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class PipelineAudit:
    """Señales del pipeline para una muestra, sin OCR."""

    quality_ok: bool
    quality_issues: tuple[str, ...]
    metadata_score: float | None
    metadata_reasons: tuple[str, ...]
    copy_move_score: float | None
    finding_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_ok": self.quality_ok,
            "quality_issues": list(self.quality_issues),
            "metadata_score": self.metadata_score,
            "metadata_reasons": list(self.metadata_reasons),
            "copy_move_score": self.copy_move_score,
            "finding_codes": list(self.finding_codes),
        }


@dataclass(frozen=True)
class SampleAudit:
    """Resultado de auditoría de una muestra."""

    sample_id: str
    document_type: str
    variant: str
    attack: str | None
    person_id: str
    parent_id: str | None
    image_path: str
    pdf_path: str
    mask_path: str | None
    sha256: str | None
    width: int | None
    height: int | None
    mask_pixels: int | None
    bbox: tuple[int, int, int, int] | None
    issues: tuple[AuditIssue, ...]
    pipeline: PipelineAudit | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "document_type": self.document_type,
            "variant": self.variant,
            "attack": self.attack,
            "person_id": self.person_id,
            "parent_id": self.parent_id,
            "image_path": self.image_path,
            "pdf_path": self.pdf_path,
            "mask_path": self.mask_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "mask_pixels": self.mask_pixels,
            "bbox": list(self.bbox) if self.bbox else None,
            "issues": [issue.to_dict() for issue in self.issues],
            "pipeline": self.pipeline.to_dict() if self.pipeline else None,
        }


@dataclass(frozen=True)
class CorpusAuditReport:
    """Reporte serializable de una auditoría de corpus."""

    root: Path
    index_path: Path
    samples: tuple[SampleAudit, ...]
    issues: tuple[AuditIssue, ...]
    by_document_type: dict[str, int]
    by_attack: dict[str, int]
    pipeline_summary: dict[str, Any] | None = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_errors(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def n_warnings(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generator": "fraude.eval.corpus_audit",
            "root": str(self.root),
            "index_path": str(self.index_path),
            "counts": {
                "samples": self.n_samples,
                "authentic": sum(sample.variant == "authentic" for sample in self.samples),
                "tampered": sum(sample.variant == "tampered" for sample in self.samples),
                "errors": self.n_errors,
                "warnings": self.n_warnings,
                "by_document_type": dict(sorted(self.by_document_type.items())),
                "by_attack": dict(sorted(self.by_attack.items())),
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "samples": [sample.to_dict() for sample in self.samples],
            "pipeline_summary": self.pipeline_summary,
        }


def audit_corpus(
    root: Path,
    *,
    index_name: str = "_indice.csv",
    people_name: str = "_personas.csv",
    analyze_images: bool = False,
    artifacts_dir: Path | None = None,
    today: date | None = None,
) -> CorpusAuditReport:
    """Valida un corpus etiquetado y, opcionalmente, ejecuta el pipeline.

    Los errores de integridad se acumulan en el reporte; no se lanza una
    excepción por una muestra rota. Si el índice no existe o no tiene las
    columnas mínimas, sí se lanza ``FileNotFoundError`` o ``ValueError``
    porque sin índice no hay nada que auditar.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"No es un directorio: {root}")
    index_path = root / index_name
    rows = _load_index(index_path)
    people_ids = _load_people_ids(root / people_name)

    issues: list[AuditIssue] = []
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = _text(row.get("id"))
        if not sample_id:
            issues.append(_issue("", "missing_id", "Hay una fila sin id."))
            continue
        if sample_id in by_id:
            issues.append(
                _issue(sample_id, "duplicate_id", f"El id {sample_id} aparece más de una vez.")
            )
            continue
        by_id[sample_id] = row

    samples: list[SampleAudit] = []
    seen_hashes: dict[str, str] = {}
    for row in rows:
        sample_id = _text(row.get("id"))
        if not sample_id or sample_id not in by_id:
            continue
        sample_issues: list[AuditIssue] = []
        sample = _audit_sample(
            root,
            row,
            by_id=by_id,
            people_ids=people_ids,
            seen_hashes=seen_hashes,
            issues=sample_issues,
        )
        issues.extend(sample_issues)
        samples.append(sample)

    by_document_type = Counter(sample.document_type for sample in samples)
    by_attack = Counter(sample.attack for sample in samples if sample.attack)
    pipeline_summary = None
    if analyze_images:
        samples, pipeline_issues = _run_pipeline(
            root,
            samples,
            artifacts_dir=artifacts_dir,
            today=today,
        )
        issues.extend(pipeline_issues)
        pipeline_summary = _summarize_pipeline(samples, today=today)

    return CorpusAuditReport(
        root=root,
        index_path=index_path,
        samples=tuple(samples),
        issues=tuple(issues),
        by_document_type=dict(by_document_type),
        by_attack=dict(by_attack),
        pipeline_summary=pipeline_summary,
    )


def _load_index(index_path: Path) -> list[dict[str, str]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"No existe el índice: {index_path}")
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError("Faltan columnas en el índice: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError(f"El índice está vacío: {index_path}")
    return rows


def _load_people_ids(people_path: Path) -> frozenset[str] | None:
    if not people_path.is_file():
        return None
    with people_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "persona_id" not in (reader.fieldnames or []):
            return None
        return frozenset(_text(row.get("persona_id")) for row in reader if row.get("persona_id"))


def _audit_sample(
    root: Path,
    row: dict[str, str],
    *,
    by_id: dict[str, dict[str, str]],
    people_ids: frozenset[str] | None,
    seen_hashes: dict[str, str],
    issues: list[AuditIssue],
) -> SampleAudit:
    sample_id = _text(row.get("id"))
    document_type = _text(row.get("tipo"))
    variant = _text(row.get("variante"))
    attack = _text(row.get("ataque")) or None
    person_id = _text(row.get("persona"))
    parent_id = _text(row.get("documento_padre")) or None

    if document_type not in DOCUMENT_TYPES:
        issues.append(
            _issue(
                sample_id,
                "unknown_document_type",
                f"Tipo documental no reconocido: {document_type!r}.",
            )
        )
    if variant not in VALID_VARIANTS:
        issues.append(_issue(sample_id, "unknown_variant", f"Variante no reconocida: {variant!r}."))
    if variant == "authentic" and attack:
        issues.append(
            _issue(sample_id, "attack_on_authentic", "Una muestra auténtica no debe tener ataque.")
        )
    if variant == "tampered" and attack not in ATTACK_TYPES:
        issues.append(_issue(sample_id, "unknown_attack", f"Ataque no reconocido: {attack!r}."))
    if not person_id:
        issues.append(_issue(sample_id, "missing_person", "La muestra no tiene persona."))
    elif people_ids is not None and person_id not in people_ids:
        issues.append(
            _issue(
                sample_id,
                "unknown_person",
                f"La persona {person_id} no está en el archivo de personas.",
            )
        )

    if parent_id:
        parent = by_id.get(parent_id)
        if parent is None:
            issues.append(
                _issue(sample_id, "missing_parent", f"El documento padre {parent_id} no existe.")
            )
        else:
            if _text(parent.get("variante")) != "authentic":
                issues.append(
                    _issue(
                        sample_id,
                        "parent_not_authentic",
                        f"El documento padre {parent_id} no es auténtico.",
                    )
                )
            if _text(parent.get("tipo")) != document_type:
                issues.append(
                    _issue(
                        sample_id,
                        "parent_type_mismatch",
                        f"El padre {parent_id} es de otro tipo documental.",
                    )
                )

    image_path = _text(row.get("png"))
    pdf_path = _text(row.get("pdf"))
    mask_path = _text(row.get("mascara")) or None
    image_file = root / image_path if image_path else None
    pdf_file = root / pdf_path if pdf_path else None
    mask_file = root / mask_path if mask_path else None

    width = height = None
    digest = None
    if image_file is None:
        issues.append(_issue(sample_id, "missing_image_path", "La fila no tiene ruta de imagen."))
    elif not image_file.is_file():
        issues.append(_issue(sample_id, "missing_image", f"No existe la imagen {image_path}."))
    else:
        try:
            with Image.open(image_file) as image:
                width, height = image.size
                if image.format != "PNG":
                    issues.append(
                        _issue(
                            sample_id,
                            "image_not_png",
                            f"La imagen no está en formato PNG ({image.format}).",
                            severity="warning",
                        )
                    )
        except (UnidentifiedImageError, OSError) as exc:
            issues.append(
                _issue(sample_id, "unreadable_image", f"No se puede leer {image_path}: {exc}.")
            )
        digest = sha256_file(image_file).split(":", 1)[-1]
        expected_hash = _text(row.get("sha256_png")).lower()
        if not _HEX_RE.match(expected_hash):
            issues.append(
                _issue(sample_id, "invalid_sha256", "sha256_png no es un SHA-256 hexadecimal.")
            )
        elif digest != expected_hash:
            issues.append(
                _issue(
                    sample_id,
                    "sha256_mismatch",
                    f"El SHA-256 de {image_path} no coincide con el índice.",
                )
            )
        previous = seen_hashes.get(digest)
        if previous is not None:
            issues.append(
                _issue(
                    sample_id,
                    "duplicate_image",
                    f"La imagen es idéntica a {previous} (mismo SHA-256).",
                    severity="warning",
                )
            )
        else:
            seen_hashes[digest] = sample_id

    if pdf_file is None:
        issues.append(_issue(sample_id, "missing_pdf_path", "La fila no tiene ruta de PDF."))
    elif not pdf_file.is_file():
        issues.append(_issue(sample_id, "missing_pdf", f"No existe el PDF {pdf_path}."))
    else:
        with pdf_file.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                issues.append(
                    _issue(sample_id, "invalid_pdf_header", f"{pdf_path} no empieza con %PDF-.")
                )

    bbox = _parse_bbox(_text(row.get("bbox_px")), sample_id, issues)
    mask_pixels = None
    if variant == "authentic":
        if mask_path:
            issues.append(
                _issue(
                    sample_id,
                    "mask_on_authentic",
                    "Una muestra auténtica no debe tener máscara.",
                    severity="warning",
                )
            )
        if bbox is not None:
            issues.append(
                _issue(
                    sample_id,
                    "bbox_on_authentic",
                    "Una muestra auténtica no debe tener bbox.",
                    severity="warning",
                )
            )
    elif attack in VISUAL_ATTACKS:
        if mask_file is None:
            issues.append(
                _issue(sample_id, "missing_mask_path", "Un ataque visual debe tener máscara.")
            )
        elif not mask_file.is_file():
            issues.append(_issue(sample_id, "missing_mask", f"No existe la máscara {mask_path}."))
        else:
            mask_pixels, mask_bbox = _mask_stats(
                mask_file,
                expected_size=(width, height),
                sample_id=sample_id,
                issues=issues,
            )
            if mask_pixels == 0:
                issues.append(
                    _issue(sample_id, "empty_mask", "La máscara de un ataque visual está vacía.")
                )
            if bbox is None:
                issues.append(
                    _issue(
                        sample_id, "missing_bbox", "Un ataque visual debe tener bbox en el índice."
                    )
                )
            elif mask_bbox is not None and not _bbox_close(bbox, mask_bbox):
                issues.append(
                    _issue(
                        sample_id,
                        "bbox_mask_mismatch",
                        f"El bbox {bbox} no coincide con la máscara {mask_bbox}.",
                    )
                )
        if bbox is not None and width is not None and height is not None:
            _validate_bbox_bounds(bbox, width, height, sample_id, issues)
    elif attack in SEMANTIC_ATTACKS:
        if mask_file is not None and mask_file.is_file():
            mask_pixels, _ = _mask_stats(
                mask_file,
                expected_size=(width, height),
                sample_id=sample_id,
                issues=issues,
            )
            if mask_pixels:
                issues.append(
                    _issue(
                        sample_id,
                        "mask_on_semantic_attack",
                        "Un ataque semántico no debería tener píxeles en la máscara.",
                        severity="warning",
                    )
                )
        if bbox is not None:
            issues.append(
                _issue(
                    sample_id,
                    "bbox_on_semantic_attack",
                    "Un ataque semántico no debería tener bbox.",
                    severity="warning",
                )
            )

    if variant == "tampered":
        if not _text(row.get("campo_original")) or not _text(row.get("campo_alterado")):
            issues.append(
                _issue(
                    sample_id,
                    "missing_field_change",
                    "La muestra manipulada no declara campo original y alterado.",
                    severity="warning",
                )
            )
    elif _text(row.get("campo_original")) or _text(row.get("campo_alterado")):
        issues.append(
            _issue(
                sample_id,
                "field_change_on_authentic",
                "Una muestra auténtica no debe declarar campo alterado.",
                severity="warning",
            )
        )

    return SampleAudit(
        sample_id=sample_id,
        document_type=document_type,
        variant=variant,
        attack=attack,
        person_id=person_id,
        parent_id=parent_id,
        image_path=image_path,
        pdf_path=pdf_path,
        mask_path=mask_path,
        sha256=digest,
        width=width,
        height=height,
        mask_pixels=mask_pixels,
        bbox=bbox,
        issues=tuple(issues),
    )


def _run_pipeline(
    root: Path,
    samples: list[SampleAudit],
    *,
    artifacts_dir: Path | None,
    today: date | None,
) -> tuple[list[SampleAudit], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    updated: list[SampleAudit] = []
    for sample in samples:
        if sample.width is None:
            updated.append(sample)
            continue
        try:
            observation = analyze_document(
                root / sample.image_path,
                artifacts_dir=artifacts_dir,
                run_text=False,
                today=today,
            )
        except Exception as exc:  # una muestra rota no debe abortar el lote
            issues.append(
                _issue(
                    sample.sample_id,
                    "pipeline_error",
                    f"El pipeline falló sobre {sample.image_path}: {exc}.",
                )
            )
            updated.append(sample)
            continue
        metadata = observation.forensics.metadata
        localization = observation.forensics.localization
        pipeline = PipelineAudit(
            quality_ok=observation.quality.ok,
            quality_issues=tuple(observation.quality.issues),
            metadata_score=metadata.score,
            metadata_reasons=tuple(str(reason) for reason in metadata.details.get("reasons", [])),
            copy_move_score=localization.score,
            finding_codes=tuple(finding.code for finding in observation.rules),
        )
        updated.append(
            SampleAudit(
                sample_id=sample.sample_id,
                document_type=sample.document_type,
                variant=sample.variant,
                attack=sample.attack,
                person_id=sample.person_id,
                parent_id=sample.parent_id,
                image_path=sample.image_path,
                pdf_path=sample.pdf_path,
                mask_path=sample.mask_path,
                sha256=sample.sha256,
                width=sample.width,
                height=sample.height,
                mask_pixels=sample.mask_pixels,
                bbox=sample.bbox,
                issues=sample.issues,
                pipeline=pipeline,
            )
        )
    return updated, issues


def _summarize_pipeline(
    samples: list[SampleAudit],
    *,
    today: date | None,
) -> dict[str, Any]:
    analyzed = [sample for sample in samples if sample.pipeline is not None]
    by_label = {label: _label_summary(analyzed, label) for label in ("authentic", "tampered")}
    by_attack = {
        attack: _attack_summary(analyzed, attack)
        for attack in ATTACK_TYPES
        if any(sample.attack == attack for sample in analyzed)
    }
    warnings: list[dict[str, str]] = []
    authentic = by_label["authentic"]
    tampered = by_label["tampered"]
    if (
        authentic["n"] > 0
        and tampered["n"] > 0
        and authentic["metadata_flagged"] == 0
        and tampered["metadata_flagged"] == tampered["n"]
    ):
        warnings.append(
            {
                "code": "metadata_shortcut",
                "message": (
                    "Los metadatos separan perfectamente auténticos y manipulados. "
                    "Es un atajo del generador, no una señal visual: no usar esta "
                    "separación como métrica de detección."
                ),
            }
        )
    has_visual_attack = any(
        sample.attack in VISUAL_ATTACKS for sample in analyzed if sample.pipeline
    )
    if has_visual_attack and tampered["copy_move_flagged"] == 0:
        warnings.append(
            {
                "code": "copy_move_not_detected",
                "message": (
                    "El detector conservador de copy-move no marcó ninguna muestra "
                    "manipulada de este lote."
                ),
            }
        )
    if any(not sample.pipeline.quality_ok for sample in analyzed if sample.pipeline):
        warnings.append(
            {
                "code": "quality_rejections",
                "message": "El pipeline rechazó al menos una muestra por calidad.",
            }
        )
    return {
        "n_analyzed": len(analyzed),
        "ocr": "disabled",
        "today": today.isoformat() if today else None,
        "by_label": by_label,
        "by_attack": by_attack,
        "warnings": warnings,
    }


def _label_summary(samples: list[SampleAudit], label: str) -> dict[str, int]:
    subset = [sample for sample in samples if sample.variant == label and sample.pipeline]
    return {
        "n": len(subset),
        "quality_ok": sum(
            bool(sample.pipeline and sample.pipeline.quality_ok) for sample in subset
        ),
        "metadata_flagged": sum(
            bool(sample.pipeline and (sample.pipeline.metadata_score or 0.0) >= 0.5)
            for sample in subset
        ),
        "copy_move_flagged": sum(
            bool(sample.pipeline and (sample.pipeline.copy_move_score or 0.0) >= 0.5)
            for sample in subset
        ),
    }


def _attack_summary(samples: list[SampleAudit], attack: str) -> dict[str, int]:
    subset = [sample for sample in samples if sample.attack == attack and sample.pipeline]
    return {
        "n": len(subset),
        "quality_ok": sum(
            bool(sample.pipeline and sample.pipeline.quality_ok) for sample in subset
        ),
        "metadata_flagged": sum(
            bool(sample.pipeline and (sample.pipeline.metadata_score or 0.0) >= 0.5)
            for sample in subset
        ),
        "copy_move_flagged": sum(
            bool(sample.pipeline and (sample.pipeline.copy_move_score or 0.0) >= 0.5)
            for sample in subset
        ),
    }


def _parse_bbox(
    value: str,
    sample_id: str,
    issues: list[AuditIssue],
) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        issues.append(_issue(sample_id, "invalid_bbox", f"bbox_px no tiene 4 valores: {value!r}."))
        return None
    try:
        bbox = tuple(int(part) for part in parts)
    except ValueError:
        issues.append(_issue(sample_id, "invalid_bbox", f"bbox_px no es entero: {value!r}."))
        return None
    x0, y0, x1, y1 = bbox
    if x0 >= x1 or y0 >= y1:
        issues.append(
            _issue(sample_id, "invalid_bbox", f"bbox_px no tiene área positiva: {value!r}.")
        )
        return None
    return bbox


def _validate_bbox_bounds(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    sample_id: str,
    issues: list[AuditIssue],
) -> None:
    x0, y0, x1, y1 = bbox
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        issues.append(
            _issue(
                sample_id,
                "bbox_out_of_bounds",
                f"El bbox {bbox} sale de la imagen {width}x{height}.",
            )
        )


def _mask_stats(
    mask_path: Path,
    *,
    expected_size: tuple[int | None, int | None],
    sample_id: str,
    issues: list[AuditIssue],
) -> tuple[int | None, tuple[int, int, int, int] | None]:
    try:
        with Image.open(mask_path) as mask_image:
            if (
                expected_size[0] is not None
                and expected_size[1] is not None
                and mask_image.size != expected_size
            ):
                issues.append(
                    _issue(
                        sample_id,
                        "mask_size_mismatch",
                        f"La máscara mide {mask_image.size} y la imagen {expected_size}.",
                    )
                )
            array = np.asarray(mask_image.convert("L"))
    except (UnidentifiedImageError, OSError) as exc:
        issues.append(
            _issue(sample_id, "unreadable_mask", f"No se puede leer {mask_path.name}: {exc}.")
        )
        return None, None
    ys, xs = np.nonzero(array)
    pixels = int(xs.size)
    if pixels == 0:
        return 0, None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return pixels, bbox


def _bbox_close(
    expected: tuple[int, int, int, int],
    actual: tuple[int, int, int, int],
) -> bool:
    return all(
        abs(left - right) <= _BBOX_TOLERANCE for left, right in zip(expected, actual, strict=True)
    )


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _issue(
    sample_id: str,
    code: str,
    message: str,
    *,
    severity: str = "error",
) -> AuditIssue:
    return AuditIssue(sample_id=sample_id, code=code, message=message, severity=severity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoría de integridad de un corpus etiquetado.")
    parser.add_argument("root", type=Path, help="Carpeta que contiene el índice y los documentos.")
    parser.add_argument("--index", default="_indice.csv", help="Nombre del índice CSV.")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Ejecuta calidad y forense sobre cada imagen (sin OCR).",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--today", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Ruta del JSON de salida.")
    args = parser.parse_args(argv)

    report = audit_corpus(
        args.root,
        index_name=args.index,
        analyze_images=args.pipeline,
        artifacts_dir=args.artifacts_dir,
        today=args.today,
    )
    payload = report.to_dict()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"{report.n_samples} muestras · {report.n_errors} errores · "
        f"{report.n_warnings} advertencias"
    )
    if report.pipeline_summary is not None:
        summary = report.pipeline_summary
        print(
            "pipeline: "
            f"{summary['n_analyzed']} analizadas · "
            f"metadatos auténticos={summary['by_label']['authentic']['metadata_flagged']} · "
            f"metadatos manipulados={summary['by_label']['tampered']['metadata_flagged']}"
        )
        for warning in summary["warnings"]:
            print(f"ADVERTENCIA [{warning['code']}] {warning['message']}")
    if args.out is not None:
        print(f"reporte -> {args.out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
