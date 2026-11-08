"""Evaluación y calibración de las predicciones out-of-fold de F4.

Sigue el protocolo del plan §7.4: umbral fijo 0.5, umbral calibrado y umbral
oráculo; Brier/ECE; IC bootstrap; desglose por ataque y tipo documental. La
calibración isotónica se ajusta de forma cruzada (cada fold se calibra con los
otros folds) para no evaluar con un calibrador entrenado en la misma muestra.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fraude.eval.metrics import (
    binary_metrics,
    bootstrap_ci,
    ranking_metrics,
    select_threshold,
)


def brier_score(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.0
    return float(np.mean((np.asarray(labels) - np.asarray(probabilities)) ** 2))


def expected_calibration_error(
    labels: list[int],
    probabilities: list[float],
    *,
    bins: int = 10,
) -> float:
    """ECE con bins de igual ancho; devuelve 0.0 si no hay muestras."""
    if not labels:
        return 0.0
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities_array >= lower) & (
            probabilities_array < upper if index < bins - 1 else probabilities_array <= upper
        )
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities_array[mask]))
        accuracy = float(np.mean(labels_array[mask]))
        error += float(np.sum(mask)) / labels_array.size * abs(confidence - accuracy)
    return error


def reliability_curve(
    labels: list[int],
    probabilities: list[float],
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[dict[str, float | int]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities_array >= lower) & (
            probabilities_array < upper if index < bins - 1 else probabilities_array <= upper
        )
        count = int(np.sum(mask))
        if not count:
            continue
        curve.append(
            {
                "bin": index,
                "lower": round(float(lower), 3),
                "upper": round(float(upper), 3),
                "n": count,
                "confidence": round(float(np.mean(probabilities_array[mask])), 4),
                "accuracy": round(float(np.mean(labels_array[mask])), 4),
            }
        )
    return curve


def cross_fitted_isotonic(
    labels: list[int],
    scores: list[float],
    folds: list[int],
) -> list[float]:
    """Calibra cada fold con un isotónico ajustado en los folds restantes."""
    from sklearn.isotonic import IsotonicRegression

    labels_array = np.asarray(labels, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    folds_array = np.asarray(folds, dtype=np.int64)
    calibrated = scores_array.copy()

    for fold in sorted(set(folds_array.tolist())):
        train_mask = folds_array != fold
        test_mask = folds_array == fold
        train_labels = labels_array[train_mask]
        if len(set(train_labels.tolist())) < 2:
            continue
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores_array[train_mask], train_labels)
        calibrated[test_mask] = model.predict(scores_array[test_mask])

    return [float(value) for value in np.clip(calibrated, 0.0, 1.0)]


def select_threshold_with_fpr(
    labels: list[int],
    scores: list[float],
    *,
    max_fpr: float = 0.2,
) -> dict[str, Any] | None:
    """Umbral que maximiza recall sin superar un FPR objetivo."""
    best: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    for step in range(5, 96):
        threshold = step / 100
        predictions = [1 if score >= threshold else 0 for score in scores]
        metrics = binary_metrics(labels, predictions)
        if float(metrics["fpr"]) > max_fpr:
            continue
        candidate = (
            float(metrics["recall"]),
            float(metrics["f1"]),
            -abs(threshold - 0.5),
        )
        if best is None or candidate > best:
            best = candidate
            best_threshold = threshold
            best_metrics = metrics
    if best is None:
        return None
    return {"threshold": round(best_threshold, 4), "metrics": best_metrics}


def threshold_table(
    labels: list[int],
    scores: list[float],
    calibrated: list[float],
) -> dict[str, Any]:
    fixed = binary_metrics(labels, [1 if score >= 0.5 else 0 for score in scores])
    calibrated_fixed = binary_metrics(labels, [1 if score >= 0.5 else 0 for score in calibrated])
    calibrated_threshold = select_threshold(labels, calibrated)
    oracle = select_threshold(labels, scores)
    return {
        "fixed_0_5": {"threshold": 0.5, "metrics": fixed},
        "calibrated_0_5": {"threshold": 0.5, "metrics": calibrated_fixed},
        "calibrated": calibrated_threshold,
        "calibrated_fpr_0_2": select_threshold_with_fpr(labels, calibrated, max_fpr=0.2),
        "oracle": oracle,
    }


def evaluate_oof(
    oof: list[dict[str, Any]],
    *,
    seed: int = 42,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Calcula el informe completo a partir de las predicciones out-of-fold."""
    if not oof:
        raise ValueError("No hay predicciones out-of-fold")

    labels = [int(row["label"]) for row in oof]
    scores = [float(row["score"]) for row in oof]
    folds = [int(row["fold"]) for row in oof]
    calibrated = cross_fitted_isotonic(labels, scores, folds)
    predictions = [1 if score >= 0.5 else 0 for score in scores]

    pooled = binary_metrics(labels, predictions)
    pooled.update(ranking_metrics(labels, scores))
    confidence_intervals = {
        "f1": bootstrap_ci(labels, predictions, "f1", n_boot=n_boot, seed=seed),
        "fpr": bootstrap_ci(labels, predictions, "fpr", n_boot=n_boot, seed=seed),
        "pr_auc": bootstrap_ci(labels, scores, "pr_auc", n_boot=n_boot, seed=seed),
        "roc_auc": bootstrap_ci(labels, scores, "roc_auc", n_boot=n_boot, seed=seed),
    }

    calibration = {
        "raw": {
            "brier": round(brier_score(labels, scores), 4),
            "ece": round(expected_calibration_error(labels, scores), 4),
        },
        "isotonic_cross_fitted": {
            "brier": round(brier_score(labels, calibrated), 4),
            "ece": round(expected_calibration_error(labels, calibrated), 4),
        },
        "reliability_raw": reliability_curve(labels, scores),
        "reliability_calibrated": reliability_curve(labels, calibrated),
    }

    by_fold = _group_metrics(oof, lambda row: str(row["fold"]))
    by_attack = _group_metrics(
        oof,
        lambda row: str(row.get("attack") or "authentic"),
    )
    by_document_type = _group_metrics(oof, lambda row: str(row["document_type"]))
    base_rows = [row for row in oof if not row.get("augmentations")]
    by_attack_base = _group_metrics(
        base_rows,
        lambda row: str(row.get("attack") or "authentic"),
    )

    fold_scores = [
        float(value["pr_auc"]) for value in by_fold.values() if value.get("pr_auc") is not None
    ]
    fold_f1 = [float(value["f1"]) for value in by_fold.values()]

    return {
        "n": len(oof),
        "base_n": len(base_rows),
        "prevalence": round(sum(labels) / len(labels), 4),
        "no_skill_pr_auc": round(sum(labels) / len(labels), 4),
        "pooled": pooled,
        "confidence_intervals": confidence_intervals,
        "thresholds": threshold_table(labels, scores, calibrated),
        "calibration": calibration,
        "by_fold": by_fold,
        "by_attack": by_attack,
        "by_attack_base": by_attack_base,
        "by_document_type": by_document_type,
        "aggregate": {
            "folds": len(by_fold),
            "pr_auc_mean": _mean(fold_scores),
            "pr_auc_std": _std(fold_scores),
            "f1_mean": _mean(fold_f1),
            "f1_std": _std(fold_f1),
        },
    }


def _group_metrics(
    rows: list[dict[str, Any]],
    key: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    report: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        labels = [int(row["label"]) for row in group]
        scores = [float(row["score"]) for row in group]
        predictions = [1 if score >= 0.5 else 0 for score in scores]
        metrics = binary_metrics(labels, predictions)
        metrics.update(ranking_metrics(labels, scores))
        report[name] = metrics
    return report


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _std(values: list[float]) -> float | None:
    return round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# F4 — evaluación out-of-fold",
        "",
        f"- Muestras: {result['n']} ({result['base_n']} sin aumentación)",
        f"- PR-AUC: {_fmt(result['pooled'].get('pr_auc'))} "
        f"(IC 95 %: {_ci(result['confidence_intervals']['pr_auc'])})",
        f"- ROC-AUC: {_fmt(result['pooled'].get('roc_auc'))} "
        f"(IC 95 %: {_ci(result['confidence_intervals']['roc_auc'])})",
        f"- F1 @0.5: {_fmt(result['pooled'].get('f1'))} "
        f"(IC 95 %: {_ci(result['confidence_intervals']['f1'])})",
        f"- Prevalencia (PR-AUC sin habilidad): {_fmt(result.get('no_skill_pr_auc'))}",
        "",
        "## Umbrales",
        "",
        "| Escenario | Umbral | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, key in (
        ("Fijo 0.5", "fixed_0_5"),
        ("Calibrado @0.5", "calibrated_0_5"),
        ("Calibrado max-F1", "calibrated"),
        ("Calibrado FPR≤0.2", "calibrated_fpr_0_2"),
        ("Oráculo", "oracle"),
    ):
        item = result["thresholds"].get(key)
        if item is None:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        metrics = item["metrics"]
        lines.append(
            f"| {name} | {item['threshold']:.2f} | {_fmt(metrics.get('precision'))} | "
            f"{_fmt(metrics.get('recall'))} | {_fmt(metrics.get('f1'))} | "
            f"{_fmt(metrics.get('fpr'))} |"
        )

    lines.extend(
        [
            "",
            "## Calibración",
            "",
            "| Versión | Brier | ECE |",
            "|---|---:|---:|",
            f"| Cruda | {result['calibration']['raw']['brier']} | "
            f"{result['calibration']['raw']['ece']} |",
            f"| Isotónica cruzada | {result['calibration']['isotonic_cross_fitted']['brier']} | "
            f"{result['calibration']['isotonic_cross_fitted']['ece']} |",
            "",
            "## Por ataque (sin aumentación)",
            "",
            "| Grupo | n | Recall @0.5 | FPR @0.5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in result["by_attack_base"].items():
        recall = metrics.get("recall") if metrics.get("recall_defined", True) else None
        fpr = metrics.get("fpr") if metrics.get("fpr_defined", True) else None
        lines.append(f"| {name} | {metrics['n']} | {_fmt(recall)} | {_fmt(fpr)} |")

    lines.extend(
        [
            "",
            "## Por fold",
            "",
            "| Fold | n | PR-AUC | F1 @0.5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in result["by_fold"].items():
        lines.append(
            f"| {name} | {metrics['n']} | {_fmt(metrics.get('pr_auc'))} | "
            f"{_fmt(metrics.get('f1'))} |"
        )

    lines.extend(
        [
            "",
            "> Corpus sintético de 50 documentos base. Las cifras no estiman "
            "generalización a documentos reales y no sustituyen la validación con el CETPRO.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    return f"{float(value):.3f}"


def _ci(interval: dict[str, float | None]) -> str:
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None:
        return "—"
    return f"{low:.3f}-{high:.3f}"


def load_oof(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa predicciones out-of-fold de F4.")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_oof(load_oof(args.oof), seed=args.seed, n_boot=args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        write_markdown(result, args.markdown)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
