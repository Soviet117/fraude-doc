"""Línea base F3: forense clásico + reglas + registro (plan §8 y §11).

La línea base usa los campos y el tipo documental del manifiesto como OCR
oráculo mientras docTR no esté disponible. El score es una suma explicable de
pesos por severidad, no una probabilidad calibrada. El umbral se elige solo en
validación; test se evalúa una vez con el umbral fijo, el elegido y el oráculo.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from fraude.core.schemas import RuleFinding, Severity
from fraude.eval.metrics import (
    binary_metrics,
    bootstrap_ci,
    ranking_metrics,
    select_threshold,
)
from fraude.forensics.classic import analyze_forensics
from fraude.preprocess.quality import assess_quality
from fraude.registry.cross import RegistryEntry, cross_registry, load_registry
from fraude.rules.engine import evaluate_rules

DEFAULT_MANIFEST = Path("data/synthetic/v1/split_manifest.jsonl")
DEFAULT_REGISTRY = Path("data/registry/synthetic.jsonl")
DEFAULT_OUT = Path("artifacts/eval/f3_baseline_v1.json")
DEFAULT_FIXED_THRESHOLD = 0.5
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 42

# Pesos fijados antes de mirar test: una señal de severidad alta basta para
# superar 0.5; una media sola no. El score se trunca a 1.0.
SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.HIGH: 0.60,
    Severity.MEDIUM: 0.25,
    Severity.LOW: 0.05,
}

LIMITATIONS = (
    "Los campos y el tipo documental provienen del manifiesto (OCR oráculo); "
    "docTR no está instalado en este host. La tabla no mide OCR ni extracción real.",
    "El registro es un espejo sintético de 14 personas, no el registro institucional del CETPRO.",
    "El dataset es sintético v1; no sustituye documentos reales ni permite "
    "afirmar desempeño en producción.",
    "El test contiene un solo grupo (persona_0001): sus 30 muestras son "
    "variantes del mismo documento base y de sus ataques. Los IC bootstrap "
    "tratan las muestras como independientes y son optimistas; no sustituyen "
    "una validación con más personas.",
    "El test v1 solo contiene T1 y T2; no hay recall para T3-T8 en esta tabla. "
    "Validación solo contiene T4, así que el umbral elegido está optimizado "
    "para ese ataque.",
    "T2 (inserción de sello) no modifica los campos ni deja señal en las reglas "
    "actuales; la línea base no lo detecta.",
    "T3 (copy-move de sello) queda subdetectado tras hacer conservador el "
    "detector; un sello pequeño puede no reportarse.",
    "Las señales noise y chromatic saturan en auténticos sintéticos y no se usan "
    "en el score; ELA y JPEG quedan como exploratorias.",
    "El score es heurístico y no está calibrado; no es una probabilidad. La "
    "calibración corresponde a F6.",
    "El umbral 0.5 se reporta junto con el elegido en validación y el oráculo en "
    "test; este último no es un punto operativo válido.",
)


@dataclass(frozen=True)
class SampleResult:
    """Resultado F3 de una muestra, antes de aplicar un umbral."""

    sample_id: str
    split: str
    document_type: str
    attack: str
    label: int
    score: float
    findings: tuple[dict[str, Any], ...]
    registry_matched: bool
    registry_score: float | None
    quality_ok: bool
    quality_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "document_type": self.document_type,
            "attack": self.attack,
            "label": self.label,
            "score": self.score,
            "findings": list(self.findings),
            "registry": {
                "matched": self.registry_matched,
                "score": self.registry_score,
            },
            "quality": {
                "ok": self.quality_ok,
                "issues": list(self.quality_issues),
            },
        }


def score_findings(findings: Iterable[RuleFinding]) -> float:
    """Convierte hallazgos en un score heurístico de 0 a 1."""
    total = sum(SEVERITY_WEIGHTS[finding.severity] for finding in findings)
    return round(min(1.0, total), 4)


def evaluate_sample(
    record: dict[str, Any],
    *,
    root: Path,
    registry: list[RegistryEntry],
    today: date,
) -> SampleResult:
    """Evalúa una muestra del manifiesto con campos y tipo oráculo."""
    fields = record.get("fields") or {}
    document_type = record.get("document_type") or None
    image_path = root / str(record["image_path"])
    forensics = analyze_forensics(image_path)
    with Image.open(image_path) as image:
        quality = assess_quality(image)

    registry_report = cross_registry(fields, registry) if registry else None
    findings = evaluate_rules(
        fields,
        document_type,
        today=today,
        quality=quality,
        forensics=forensics,
        registry=registry_report,
    )
    label = 0 if record.get("is_authentic", True) else 1
    attack = "authentic" if label == 0 else str(record.get("primary_attack") or "unknown")
    return SampleResult(
        sample_id=str(record.get("sample_id") or image_path.stem),
        split=str(record.get("split") or "unknown"),
        document_type=str(document_type or "unknown"),
        attack=attack,
        label=label,
        score=score_findings(findings),
        findings=tuple(
            {
                "code": finding.code,
                "severity": finding.severity.value,
                "weight": SEVERITY_WEIGHTS[finding.severity],
            }
            for finding in findings
        ),
        registry_matched=bool(registry_report.matched) if registry_report else False,
        registry_score=registry_report.score if registry_report else None,
        quality_ok=quality.ok,
        quality_issues=tuple(quality.issues),
    )


def run_baseline(
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    today: date | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Ejecuta la línea base F3 y devuelve el reporte serializable."""
    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    rows = _load_manifest(manifest_path)
    registry = load_registry(registry_path)
    current_day = today or date.today()
    root = manifest_path.parent

    results = [
        evaluate_sample(row, root=root, registry=registry, today=current_day)
        for row in rows
        if row.get("split") in {"val", "test"}
    ]
    grouped: dict[str, list[SampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.split].append(result)
    val_results = grouped["val"]
    test_results = grouped["test"]
    if not val_results or not test_results:
        raise ValueError("El manifiesto debe tener muestras en val y test")

    val_labels = [result.label for result in val_results]
    val_scores = [result.score for result in val_results]
    test_labels = [result.label for result in test_results]
    test_scores = [result.score for result in test_results]

    selection = select_threshold(val_labels, val_scores)
    val_threshold = float(selection["threshold"])
    oracle = select_threshold(test_labels, test_scores)
    oracle_threshold = float(oracle["threshold"])

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.eval.baseline",
        "dataset": {
            "manifest": str(manifest_path),
            "registry": str(registry_path),
            "data_kind": "synthetic_v1",
            "fields_source": "manifest_oracle",
            "doctype_source": "manifest_oracle",
            "registry_source": "synthetic_mirror",
            "n_registry_entries": len(registry),
            "split_sizes": {
                "val": len(val_results),
                "test": len(test_results),
            },
            "groups": {
                split: len(
                    {
                        str(row.get("source_id") or row.get("sample_id"))
                        for row in rows
                        if row.get("split") == split
                    }
                )
                for split in ("val", "test")
            },
        },
        "config": {
            "score": "suma de pesos por severidad, truncada a 1.0",
            "severity_weights": {
                severity.value: weight for severity, weight in SEVERITY_WEIGHTS.items()
            },
            "fixed_threshold": fixed_threshold,
            "bootstrap": {
                "method": "percentile",
                "confidence": 0.95,
                "n_boot": n_boot,
                "seed": seed,
            },
            "today": current_day.isoformat(),
            "signals_used": [
                "rules",
                "registry",
                "quality",
                "forensics.copy_move",
                "forensics.metadata",
            ],
            "signals_excluded": ["ela", "noise", "chromatic", "jpeg"],
        },
        "thresholds": {
            "fixed": fixed_threshold,
            "selected_on_val": val_threshold,
            "oracle_on_test": oracle_threshold,
        },
        "evaluation": {
            "val": {
                "fixed_0_5": _evaluation_block(
                    val_results,
                    fixed_threshold,
                    n_boot=n_boot,
                    seed=seed,
                ),
                "val_selected": _evaluation_block(
                    val_results,
                    val_threshold,
                    n_boot=n_boot,
                    seed=seed,
                ),
            },
            "test": {
                "fixed_0_5": _evaluation_block(
                    test_results,
                    fixed_threshold,
                    n_boot=n_boot,
                    seed=seed,
                ),
                "val_selected": _evaluation_block(
                    test_results,
                    val_threshold,
                    n_boot=n_boot,
                    seed=seed,
                ),
                "oracle": _evaluation_block(
                    test_results,
                    oracle_threshold,
                    n_boot=n_boot,
                    seed=seed,
                ),
            },
        },
        "samples": [result.to_dict() for result in results],
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


def _evaluation_block(
    results: list[SampleResult],
    threshold: float,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    labels = [result.label for result in results]
    scores = [result.score for result in results]
    predictions = [1 if score >= threshold else 0 for score in scores]
    metrics = {**binary_metrics(labels, predictions), **ranking_metrics(labels, scores)}
    bootstrap = {
        metric: bootstrap_ci(
            labels,
            scores if metric in {"pr_auc", "roc_auc"} else predictions,
            metric,
            n_boot=n_boot,
            seed=seed,
        )
        for metric in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")
    }
    return {
        "threshold": round(threshold, 4),
        "metrics": metrics,
        "bootstrap_95": bootstrap,
        "by_attack": _group_metrics(results, threshold, "attack"),
        "by_document_type": _group_metrics(results, threshold, "document_type"),
    }


def _group_metrics(
    results: list[SampleResult],
    threshold: float,
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[SampleResult]] = defaultdict(list)
    for result in results:
        groups[str(getattr(result, key))].append(result)
    output: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        labels = [result.label for result in group]
        predictions = [1 if result.score >= threshold else 0 for result in group]
        metrics = binary_metrics(labels, predictions)
        metrics["mean_score"] = round(
            sum(result.score for result in group) / len(group),
            4,
        )
        output[name] = metrics
    return output


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"Manifiesto vacío: {path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta la línea base F3.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--today",
        help="Fecha de referencia ISO (YYYY-MM-DD); por defecto, hoy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    today = date.fromisoformat(args.today) if args.today else date.today()
    payload = run_baseline(
        args.manifest,
        args.registry,
        fixed_threshold=args.fixed_threshold,
        n_boot=args.n_boot,
        seed=args.seed,
        today=today,
        out_path=args.out,
    )
    fixed = payload["evaluation"]["test"]["fixed_0_5"]
    selected = payload["evaluation"]["test"]["val_selected"]
    metrics = fixed["metrics"]
    print(f"Línea base F3: test n={metrics['n']} (umbral {fixed['threshold']:.2f})")
    print(
        "  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
        "fpr={fpr:.3f} pr_auc={pr_auc}".format(**metrics)
    )
    print(
        f"  umbral val={payload['thresholds']['selected_on_val']:.2f} | "
        f"oráculo test={payload['thresholds']['oracle_on_test']:.2f}"
    )
    print("  recall con umbral val={recall:.3f} f1={f1:.3f}".format(**selected["metrics"]))
    print(f"  artefacto: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
