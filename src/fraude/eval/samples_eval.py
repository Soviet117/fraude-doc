"""Evaluación del pipeline completo sobre el corpus local de muestras.

A diferencia de ``fraude.eval.baseline``, este arnés **no usa campos oráculo**:
ejecuta calidad, forense, OCR real, extracción, reglas y cruce de registro sobre
cada PNG de ``data/muestras``.

Se evalúan dos modos:

- ``as_is``: PNG original, con los metadatos que dejó el generador.
- ``stripped``: copia temporal sin chunks de texto/EXIF. Sirve para separar el
  atajo de metadatos (Photoshop) de las señales que sobreviven al contenido.

El score es el mismo heurístico de F3: suma de pesos por severidad, truncada a
1.0. No es una probabilidad calibrada y el umbral 0.50 es fijo; los números
describen este lote de 20 muestras, no el desempeño institucional.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from fraude.core.schemas import Observation, RuleFinding
from fraude.eval.baseline import SEVERITY_WEIGHTS, score_findings
from fraude.eval.metrics import binary_metrics, bootstrap_ci, ranking_metrics
from fraude.pipeline.run import analyze_document
from fraude.registry.cross import RegistryEntry

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/muestras")
DEFAULT_OUT = Path("artifacts/eval/samples_pipeline.json")
DEFAULT_INDEX = "_indice.csv"
DEFAULT_PEOPLE = "_personas.csv"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TODAY = date(2026, 11, 5)
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 42
MODES = ("as_is", "stripped")

REQUIRED_COLUMNS = ("id", "tipo", "variante", "ataque", "persona", "png")

LIMITATIONS = (
    "Corpus sintético de 20 documentos y 12 personas; los números describen "
    "este lote, no el desempeño institucional del CETPRO.",
    "Hay una sola muestra por ataque T1-T8; no existe recall por ataque "
    "defendible ni posibilidad de entrenar un modelo con este volumen.",
    "El score es heurístico y no está calibrado; 0.50 es un umbral fijo, no "
    "un punto operativo validado.",
    "El modo as_is conserva los metadatos del generador; el modo stripped los "
    "elimina. La diferencia entre ambos mide cuánto depende el resultado del "
    "atajo de metadatos, no una señal de contenido.",
    "Los intervalos bootstrap con n=20 son anchos y tratan las muestras como "
    "independientes; se reportan solo como referencia.",
    "No hay evaluación humana, kappa ni documentos reales en este lote.",
)


@dataclass(frozen=True)
class SampleEvalResult:
    """Resultado del pipeline para una muestra y un modo."""

    sample_id: str
    mode: str
    document_type: str
    attack: str
    variant: str
    label: int
    score: float | None
    predicted: int | None
    extracted_doctype: str
    doctype_ok: bool
    fields: dict[str, Any]
    findings: tuple[dict[str, Any], ...]
    registry: dict[str, Any]
    quality: dict[str, Any]
    forensics: dict[str, Any]
    ocr: dict[str, Any]
    timings: dict[str, int]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "mode": self.mode,
            "document_type": self.document_type,
            "attack": self.attack,
            "variant": self.variant,
            "label": self.label,
            "score": self.score,
            "predicted": self.predicted,
            "extraction": {
                "document_type": self.extracted_doctype,
                "doctype_ok": self.doctype_ok,
                "fields": self.fields,
            },
            "findings": list(self.findings),
            "registry": self.registry,
            "quality": self.quality,
            "forensics": self.forensics,
            "ocr": self.ocr,
            "timings": self.timings,
            "error": self.error,
        }


def run_samples_eval(
    root: Path = DEFAULT_ROOT,
    *,
    index_name: str = DEFAULT_INDEX,
    people_name: str = DEFAULT_PEOPLE,
    modes: tuple[str, ...] = MODES,
    threshold: float = DEFAULT_THRESHOLD,
    today: date = DEFAULT_TODAY,
    out_path: Path | None = DEFAULT_OUT,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline sobre el corpus y devuelve el reporte serializable."""
    root = Path(root)
    rows = _load_rows(root / index_name)
    registry = _load_registry(root / people_name)
    if limit is not None:
        rows = rows[:limit]
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"Modo no soportado: {mode}")

    runs: dict[str, Any] = {}
    for mode in modes:
        results: list[SampleEvalResult] = []
        with tempfile.TemporaryDirectory(prefix=f"fraude-samples-{mode}-") as temp_dir:
            for index, row in enumerate(rows, start=1):
                sample_id = _text(row.get("id"))
                logger.info("%s %s (%d/%d)", mode, sample_id, index, len(rows))
                image_path = root / _text(row.get("png"))
                if mode == "stripped":
                    image_path = _strip_metadata(
                        image_path,
                        Path(temp_dir) / f"{sample_id}.png",
                    )
                results.append(
                    _run_sample(
                        row,
                        image_path=image_path,
                        registry=registry,
                        mode=mode,
                        threshold=threshold,
                        today=today,
                    )
                )
        runs[mode] = {
            "summary": _summarize(
                results,
                threshold=threshold,
                n_boot=n_boot,
                seed=seed,
            ),
            "samples": [result.to_dict() for result in results],
        }

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.eval.samples_eval",
        "corpus": {
            "root": str(root),
            "index": index_name,
            "people": people_name,
            "n_samples": len(rows),
            "n_authentic": sum(_text(row.get("variante")) == "authentic" for row in rows),
            "n_tampered": sum(_text(row.get("variante")) != "authentic" for row in rows),
        },
        "config": {
            "modes": list(modes),
            "threshold": threshold,
            "today": today.isoformat(),
            "n_boot": n_boot,
            "seed": seed,
            "limit": limit,
        },
        "runs": runs,
        "limitations": list(LIMITATIONS),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def _run_sample(
    row: dict[str, str],
    *,
    image_path: Path,
    registry: list[RegistryEntry],
    mode: str,
    threshold: float,
    today: date,
) -> SampleEvalResult:
    document_type = _text(row.get("tipo"))
    variant = _text(row.get("variante"))
    attack = _text(row.get("ataque")) or "authentic"
    label = 0 if variant == "authentic" else 1

    if not image_path.is_file():
        return _error_result(
            row,
            mode=mode,
            image_path=str(image_path),
            error=f"No existe la imagen {image_path}",
        )

    try:
        observation = analyze_document(
            image_path,
            registry=registry,
            today=today,
        )
    except Exception as exc:  # una muestra rota no debe abortar el lote
        return _error_result(
            row,
            mode=mode,
            image_path=str(image_path),
            error=str(exc),
        )

    score = score_findings(observation.rules)
    predicted = int(score >= threshold)
    return _observation_result(
        row,
        observation,
        mode=mode,
        score=score,
        predicted=predicted,
        label=label,
        document_type=document_type,
        variant=variant,
        attack=attack,
    )


def _observation_result(
    row: dict[str, str],
    observation: Observation,
    *,
    mode: str,
    score: float,
    predicted: int,
    label: int,
    document_type: str,
    variant: str,
    attack: str,
) -> SampleEvalResult:
    return SampleEvalResult(
        sample_id=_text(row.get("id")),
        mode=mode,
        document_type=document_type,
        attack=attack,
        variant=variant,
        label=label,
        score=score,
        predicted=predicted,
        extracted_doctype=observation.doctype.label,
        doctype_ok=observation.doctype.label == document_type,
        fields=observation.fields,
        findings=tuple(_finding_dict(finding) for finding in observation.rules),
        registry={
            "matched": observation.registry.matched,
            "score": observation.registry.score,
            "diffs": observation.registry.diffs,
        },
        quality={
            "ok": observation.quality.ok,
            "issues": observation.quality.issues,
            "blur": observation.quality.blur,
            "skew_deg": observation.quality.skew_deg,
        },
        forensics=_forensics_dict(observation),
        ocr={
            "engine": observation.ocr.engine,
            "mean_conf": observation.ocr.mean_conf,
            "n_words": observation.ocr.n_words,
        },
        timings=dict(observation.timings),
    )


def _error_result(
    row: dict[str, str],
    *,
    mode: str,
    image_path: str,
    error: str,
) -> SampleEvalResult:
    variant = _text(row.get("variante"))
    return SampleEvalResult(
        sample_id=_text(row.get("id")),
        mode=mode,
        document_type=_text(row.get("tipo")),
        attack=_text(row.get("ataque")) or "authentic",
        variant=variant,
        label=0 if variant == "authentic" else 1,
        score=None,
        predicted=None,
        extracted_doctype="",
        doctype_ok=False,
        fields={},
        findings=(),
        registry={"matched": False, "score": None, "diffs": []},
        quality={"ok": False, "issues": [], "blur": None, "skew_deg": None},
        forensics={},
        ocr={},
        timings={},
        error=f"{error} ({image_path})",
    )


def _finding_dict(finding: RuleFinding) -> dict[str, Any]:
    return {
        "code": finding.code,
        "severity": finding.severity.value,
        "weight": SEVERITY_WEIGHTS[finding.severity],
        "message": finding.message,
    }


def _forensics_dict(observation: Observation) -> dict[str, Any]:
    forensics = observation.forensics
    return {
        "metadata": {
            "score": forensics.metadata.score,
            "details": forensics.metadata.details,
        },
        "localization": {
            "score": forensics.localization.score,
            "details": forensics.localization.details,
        },
        "ela": {"score": forensics.ela.score},
        "noise": {"score": forensics.noise.score},
        "chromatic": {"score": forensics.chromatic.score},
        "jpeg": {"score": forensics.jpeg.score},
    }


def _summarize(
    results: list[SampleEvalResult],
    *,
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    ok = [
        result
        for result in results
        if result.error is None and result.score is not None and result.predicted is not None
    ]
    labels = [result.label for result in ok]
    scores = [float(result.score) for result in ok if result.score is not None]
    predictions = [int(result.predicted) for result in ok if result.predicted is not None]
    metrics = {**binary_metrics(labels, predictions), **ranking_metrics(labels, scores)}
    ci = {
        metric: bootstrap_ci(
            labels,
            predictions if metric in {"precision", "recall", "f1", "fpr"} else scores,
            metric,  # type: ignore[arg-type]
            n_boot=n_boot,
            seed=seed,
        )
        for metric in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")
    }
    return {
        "n_samples": len(results),
        "n_ok": len(ok),
        "n_errors": len(results) - len(ok),
        "threshold": threshold,
        "metrics": metrics,
        "ci95": ci,
        "doctype_accuracy": _ratio(
            sum(result.doctype_ok for result in ok),
            len(ok),
        ),
        "metadata_findings": {
            "authentic": sum(
                _has_code(result, "METADATA_EDICION") for result in ok if result.label == 0
            ),
            "tampered": sum(
                _has_code(result, "METADATA_EDICION") for result in ok if result.label == 1
            ),
        },
        "by_attack": _group_summary(ok, "attack"),
        "by_document_type": _group_summary(ok, "document_type"),
        "errors": [result.to_dict() for result in results if result.error is not None],
    }


def _group_summary(
    results: list[SampleEvalResult],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[SampleEvalResult]] = {}
    for result in results:
        key = str(getattr(result, attribute) or "unknown")
        grouped.setdefault(key, []).append(result)
    return {
        key: {
            "n": len(group),
            "n_positive": sum(result.label for result in group),
            "n_detected": sum(int(result.predicted or 0) for result in group),
            "mean_score": round(
                sum(float(result.score or 0.0) for result in group) / len(group),
                4,
            ),
            "codes": sorted({finding["code"] for result in group for finding in result.findings}),
        }
        for key, group in sorted(grouped.items())
    }


def _has_code(result: SampleEvalResult, code: str) -> bool:
    return any(finding["code"] == code for finding in result.findings)


def _strip_metadata(src: Path, dst: Path) -> Path:
    """Copia una imagen sin chunks de texto/EXIF para aislar señales de contenido."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        clean = Image.new("RGB", image.size)
        clean.paste(image.convert("RGB"))
        clean.save(dst, format="PNG")
    return dst


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el índice: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Faltan columnas en el índice: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError(f"El índice está vacío: {path}")
    return rows


def _load_registry(path: Path) -> list[RegistryEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de personas: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        RegistryEntry(
            nombre=f"{_text(row.get('apellidos'))}, {_text(row.get('nombres'))}",
            dni=_text(row.get("dni")) or None,
            programa=_text(row.get("programa")) or None,
            modulo=_text(row.get("modulo")) or None,
            periodo=_text(row.get("periodo")) or None,
            fecha_nacimiento=_text(row.get("fecha_nacimiento")) or None,
            source_id=_text(row.get("persona_id")) or None,
        )
        for row in rows
        if _text(row.get("apellidos")) or _text(row.get("nombres"))
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evalúa el pipeline completo sobre el corpus local de muestras."
    )
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--people", default=DEFAULT_PEOPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=(*MODES, "both"), default="both")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--today", type=date.fromisoformat, default=DEFAULT_TODAY)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    modes = MODES if args.mode == "both" else (args.mode,)
    payload = run_samples_eval(
        args.root,
        index_name=args.index,
        people_name=args.people,
        modes=modes,
        threshold=args.threshold,
        today=args.today,
        out_path=args.out,
        n_boot=args.n_boot,
        seed=args.seed,
        limit=args.limit,
    )
    for mode, run in payload["runs"].items():
        summary = run["summary"]
        metrics = summary["metrics"]
        print(
            f"[{mode}] n={summary['n_ok']} "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} "
            f"precision={_fmt(metrics['precision'])} "
            f"recall={_fmt(metrics['recall'])} "
            f"f1={_fmt(metrics['f1'])} "
            f"fpr={_fmt(metrics['fpr'])} "
            f"pr_auc={_fmt(metrics['pr_auc'])} "
            f"roc_auc={_fmt(metrics['roc_auc'])}"
        )
    print(f"reporte -> {args.out}")
    return 0 if all(run["summary"]["n_errors"] == 0 for run in payload["runs"].values()) else 1


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
