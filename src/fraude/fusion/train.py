"""Entrenamiento y evaluación de la fusión F7 con LightGBM.

Protocolo: validación cruzada agrupada por persona, calibración isotónica
cruzada, umbrales fijo/calibrado/FPR<=0.2/oráculo, IC bootstrap y ablaciones
acumulativas por familia de señal. El modelo final se reentrena con todas las
filas y se guarda con su calibrador para la demo.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from fraude.eval.metrics import (
    binary_metrics,
    bootstrap_ci,
    ranking_metrics,
)
from fraude.fusion.features import FEATURE_GROUPS, FEATURE_NAMES
from fraude.models.evaluate import (
    brier_score,
    cross_fitted_isotonic,
    expected_calibration_error,
    select_threshold_with_fpr,
    threshold_table,
)

LOGGER = logging.getLogger("fraude.fusion.train")

DEFAULT_FEATURES = Path("artifacts/models/f7/features.jsonl")
DEFAULT_OUT = Path("artifacts/models/f7")


@dataclass
class FusionConfig:
    features_path: Path = DEFAULT_FEATURES
    out_dir: Path = DEFAULT_OUT
    seed: int = 42
    n_boot: int = 1000
    max_fpr: float = 0.2
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 7
    max_depth: int = 3
    min_child_samples: int = 3


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float32,
    )


def _labels(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def _make_model(cfg: FusionConfig) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=cfg.seed,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def cross_validate(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    cfg: FusionConfig,
) -> list[dict[str, Any]]:
    """Predicciones out-of-fold agrupadas por persona."""
    folds = sorted({int(row["fold"]) for row in rows})
    output: list[dict[str, Any]] = []
    for fold in folds:
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        val_rows = [row for row in rows if int(row["fold"]) == fold]
        if not train_rows or not val_rows:
            continue
        model = _make_model(cfg)
        model.fit(
            _matrix(train_rows, feature_names),
            _labels(train_rows),
        )
        scores = model.predict_proba(_matrix(val_rows, feature_names))[:, 1]
        for row, score in zip(val_rows, scores, strict=True):
            output.append(
                {
                    **row,
                    "raw_score": round(float(score), 6),
                }
            )
    output.sort(key=lambda row: row["sample_id"])
    return output


def _group_metrics(
    rows: list[dict[str, Any]],
    key: str,
    *,
    score_key: str,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    report: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        labels = _labels(group)
        predictions = [1 if float(row[score_key]) >= threshold else 0 for row in group]
        scores = [float(row[score_key]) for row in group]
        metrics = binary_metrics(labels, predictions)
        metrics.update(ranking_metrics(labels, scores))
        report[name] = metrics
    return report


def _ablation(
    rows: list[dict[str, Any]],
    cfg: FusionConfig,
) -> dict[str, Any]:
    cumulative: list[str] = []
    results: dict[str, Any] = {}
    for name, features in FEATURE_GROUPS.items():
        cumulative.extend(features)
        oof = cross_validate(rows, feature_names=tuple(cumulative), cfg=cfg)
        labels = _labels(oof)
        scores = [float(row["raw_score"]) for row in oof]
        predictions = [1 if score >= 0.5 else 0 for score in scores]
        metrics = binary_metrics(labels, predictions)
        metrics.update(ranking_metrics(labels, scores))
        results[name] = {
            "features": len(cumulative),
            "metrics": metrics,
        }
    return results


def train_fusion(cfg: FusionConfig) -> dict[str, Any]:
    rows = _load_rows(cfg.features_path)
    if not rows:
        raise ValueError(f"No hay características en {cfg.features_path}")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = cross_validate(rows, feature_names=FEATURE_NAMES, cfg=cfg)
    labels = _labels(oof)
    raw_scores = [float(row["raw_score"]) for row in oof]
    folds = [int(row["fold"]) for row in oof]
    calibrated = cross_fitted_isotonic(labels, raw_scores, folds)
    for row, value in zip(oof, calibrated, strict=True):
        row["calibrated_score"] = round(float(value), 6)

    raw_predictions = [1 if score >= 0.5 else 0 for score in raw_scores]
    calibrated_predictions = [1 if score >= 0.5 else 0 for score in calibrated]
    pooled = binary_metrics(labels, raw_predictions)
    pooled.update(ranking_metrics(labels, raw_scores))
    calibrated_pooled = binary_metrics(labels, calibrated_predictions)
    calibrated_pooled.update(ranking_metrics(labels, calibrated))
    confidence_intervals = {
        metric: bootstrap_ci(
            labels,
            raw_predictions if metric in {"precision", "recall", "f1", "fpr"} else raw_scores,
            metric,  # type: ignore[arg-type]
            n_boot=cfg.n_boot,
            seed=cfg.seed,
        )
        for metric in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")
    }
    thresholds = threshold_table(labels, raw_scores, calibrated)
    thresholds["calibrated_fpr_0_2"] = select_threshold_with_fpr(
        labels,
        calibrated,
        max_fpr=cfg.max_fpr,
    )
    calibration = {
        "raw": {
            "brier": round(brier_score(labels, raw_scores), 4),
            "ece": round(expected_calibration_error(labels, raw_scores), 4),
        },
        "isotonic_cross_fitted": {
            "brier": round(brier_score(labels, calibrated), 4),
            "ece": round(expected_calibration_error(labels, calibrated), 4),
        },
    }

    final_model = _make_model(cfg)
    final_model.fit(_matrix(rows, FEATURE_NAMES), _labels(rows))
    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores, labels)

    contributions = final_model.booster_.predict(
        _matrix(rows, FEATURE_NAMES),
        pred_contrib=True,
    )
    mean_abs = np.mean(np.abs(contributions[:, :-1]), axis=0)
    importance = [
        {"feature": name, "mean_abs_shap": round(float(value), 6)}
        for name, value in sorted(
            zip(FEATURE_NAMES, mean_abs, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    import joblib

    model_path = out_dir / "fusion.joblib"
    joblib.dump(
        {
            "model": final_model,
            "calibrator": calibrator,
            "feature_names": list(FEATURE_NAMES),
            "config": {
                **asdict(cfg),
                "features_path": str(cfg.features_path),
                "out_dir": str(cfg.out_dir),
            },
        },
        model_path,
    )

    ablations = _ablation(rows, cfg)
    result = {
        "schema_version": "1.0",
        "generator": "fraude.fusion.train",
        "config": {
            **asdict(cfg),
            "features_path": str(cfg.features_path),
            "out_dir": str(cfg.out_dir),
        },
        "n": len(rows),
        "prevalence": round(sum(labels) / len(labels), 4),
        "pooled": pooled,
        "calibrated_pooled": calibrated_pooled,
        "confidence_intervals": confidence_intervals,
        "thresholds": thresholds,
        "calibration": calibration,
        "by_attack": _group_metrics(oof, "attack", score_key="raw_score"),
        "by_attack_calibrated": _group_metrics(oof, "attack", score_key="calibrated_score"),
        "by_document_type": _group_metrics(oof, "document_type", score_key="raw_score"),
        "by_fold": _group_metrics(oof, "fold", score_key="raw_score"),
        "ablations": ablations,
        "feature_importance": importance[:20],
        "model_path": str(model_path),
    }

    oof_path = out_dir / "oof.jsonl"
    with oof_path.open("w", encoding="utf-8") as handle:
        for row in oof:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["oof"] = str(oof_path)

    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(result, out_dir / "eval.md")
    LOGGER.info("Fusión guardada en %s", model_path)
    return result


def write_markdown(result: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    pooled = result["pooled"]
    calibrated = result["calibrated_pooled"]
    lines = [
        "# F7 — fusión y calibración",
        "",
        f"- Documentos: {result['n']} · prevalencia {result['prevalence']:.4f}",
        f"- PR-AUC crudo: {_fmt(pooled.get('pr_auc'))} "
        f"(IC 95 %: {_ci(result['confidence_intervals']['pr_auc'])})",
        f"- F1 @0.5 crudo: {_fmt(pooled.get('f1'))} "
        f"(IC 95 %: {_ci(result['confidence_intervals']['f1'])})",
        f"- F1 @0.5 calibrado: {_fmt(calibrated.get('f1'))}",
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
            "## Ablaciones acumulativas",
            "",
            "| Grupo | Features | PR-AUC | F1 @0.5 | FPR @0.5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in result["ablations"].items():
        metrics = values["metrics"]
        lines.append(
            f"| {name} | {values['features']} | {_fmt(metrics.get('pr_auc'))} | "
            f"{_fmt(metrics.get('f1'))} | {_fmt(metrics.get('fpr'))} |"
        )
    lines.extend(
        [
            "",
            "## Importancia (SHAP medio absoluto)",
            "",
            "| Feature | SHAP |",
            "|---|---:|",
        ]
    )
    for item in result["feature_importance"][:15]:
        lines.append(f"| {item['feature']} | {item['mean_abs_shap']:.6f} |")
    lines.extend(
        [
            "",
            "> Corpus sintético de 50 documentos base. Las ablaciones son acumulativas",
            "> y con n pequeño: no son una comparación definitiva entre familias.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _ci(interval: dict[str, float | None]) -> str:
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None:
        return "—"
    return f"{low:.4f}-{high:.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena la fusión F7.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--max-fpr", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = FusionConfig(
        features_path=args.features,
        out_dir=args.out,
        seed=args.seed,
        n_boot=args.n_boot,
        max_fpr=args.max_fpr,
    )
    result = train_fusion(cfg)
    print(json.dumps(result["pooled"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
