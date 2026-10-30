"""Métricas binarias, ranking y bootstrap para la evaluación F3 (plan §11).

No depende de scikit-learn: el CI corre solo con las dependencias core. Las
métricas se calculan sobre etiquetas 0/1 y, para PR-AUC/ROC-AUC, sobre scores
continuos. Las métricas sin denominador devuelven ``None`` en el bootstrap y
``0.0`` en el resumen para no romper la tabla; los flags ``*_defined`` indican
cuándo el valor es interpretable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

MetricName = Literal[
    "precision",
    "recall",
    "f1",
    "fpr",
    "pr_auc",
    "roc_auc",
]
RANKING_METRICS = frozenset({"pr_auc", "roc_auc"})


def binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, float | int | bool]:
    """Calcula matriz de confusión y métricas binarias sin sklearn."""
    tp, tn, fp, fn = _confusion(labels, predictions)
    precision = _metric_value(tp, tn, fp, fn, "precision")
    recall = _metric_value(tp, tn, fp, fn, "recall")
    f1 = _metric_value(tp, tn, fp, fn, "f1")
    fpr = _metric_value(tp, tn, fp, fn, "fpr")
    accuracy = _divide(tp + tn, len(labels))
    return {
        "n": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision or 0.0, 4),
        "recall": round(recall or 0.0, 4),
        "f1": round(f1 or 0.0, 4),
        "fpr": round(fpr or 0.0, 4),
        "accuracy": round(accuracy, 4),
        "precision_defined": precision is not None,
        "recall_defined": recall is not None,
        "f1_defined": f1 is not None,
        "fpr_defined": fpr is not None,
    }


def ranking_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, float | None]:
    """Calcula PR-AUC (average precision) y ROC-AUC sin sklearn."""
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.size != scores_array.size:
        raise ValueError("labels y scores deben tener la misma longitud")
    positives = int(labels_array.sum())
    negatives = int(labels_array.size - positives)
    if positives == 0 or negatives == 0:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": round(_average_precision(labels_array, scores_array), 4),
        "roc_auc": round(_roc_auc(labels_array, scores_array), 4),
    }


def bootstrap_ci(
    labels: Sequence[int],
    values: Sequence[float],
    metric: MetricName,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float | None]:
    """IC bootstrap percentil 95 % para una métrica.

    Para ``precision``, ``recall``, ``f1`` y ``fpr``, ``values`` son
    predicciones 0/1. Para ``pr_auc`` y ``roc_auc``, ``values`` son scores
    continuos. Las remuestras sin denominador se omiten.
    """
    labels_array = np.asarray(labels, dtype=np.int8)
    values_array = np.asarray(values, dtype=np.float64)
    if labels_array.size != values_array.size:
        raise ValueError("labels y values deben tener la misma longitud")
    if labels_array.size == 0:
        return {"low": None, "high": None}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, labels_array.size, size=labels_array.size)
        sampled_labels = labels_array[indices]
        sampled_values = values_array[indices]
        if metric in RANKING_METRICS:
            value = ranking_metrics(sampled_labels, sampled_values)[metric]
        else:
            tp, tn, fp, fn = _confusion(sampled_labels, sampled_values)
            value = _metric_value(tp, tn, fp, fn, metric)
        if value is not None:
            samples.append(float(value))
    if not samples:
        return {"low": None, "high": None}
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": round(float(low), 4), "high": round(float(high), 4)}


def select_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    thresholds: Iterable[float] | None = None,
) -> dict[str, float | int]:
    """Elige el umbral con mejor F1; desempata cerca de 0.5."""
    if len(labels) != len(scores):
        raise ValueError("labels y scores deben tener la misma longitud")
    candidates = (
        list(thresholds) if thresholds is not None else [index / 100 for index in range(5, 96, 5)]
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predictions = [1 if score >= threshold else 0 for score in scores]
        metrics = binary_metrics(labels, predictions)
        f1 = float(metrics["f1"])
        distance = abs(threshold - 0.5)
        if best is None or f1 > best[0] or (f1 == best[0] and distance < best[2]):
            best = (f1, float(threshold), distance)
    if best is None:
        raise ValueError("No hay umbrales candidatos")
    threshold = best[1]
    predictions = [1 if score >= threshold else 0 for score in scores]
    metrics = binary_metrics(labels, predictions)
    return {"threshold": round(threshold, 4), "metrics": metrics}


def _confusion(
    labels: Sequence[int],
    predictions: Sequence[float],
) -> tuple[int, int, int, int]:
    if len(labels) != len(predictions):
        raise ValueError("labels y predictions deben tener la misma longitud")
    tp = tn = fp = fn = 0
    for label, prediction in zip(labels, predictions, strict=True):
        positive_label = int(label) == 1
        positive_prediction = float(prediction) >= 0.5
        if positive_label and positive_prediction:
            tp += 1
        elif not positive_label and not positive_prediction:
            tn += 1
        elif not positive_label and positive_prediction:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def _metric_value(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    metric: MetricName,
) -> float | None:
    if metric == "precision":
        return _divide(tp, tp + fp) if tp + fp else None
    if metric == "recall":
        return _divide(tp, tp + fn) if tp + fn else None
    if metric == "fpr":
        return _divide(fp, fp + tn) if fp + tn else None
    if metric == "f1":
        if tp + fn == 0:
            return None
        precision = _divide(tp, tp + fp) if tp + fp else 0.0
        recall = _divide(tp, tp + fn)
        return _divide(2 * precision * recall, precision + recall) if precision + recall else 0.0
    raise ValueError(f"Métrica binaria no soportada: {metric}")


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    thresholds = np.unique(scores)[::-1]
    previous_recall = 0.0
    average_precision = 0.0
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        recall = tp / positives
        precision = tp / (tp + fp) if tp + fp else 0.0
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    index = 0
    while index < labels.size:
        end = index
        while end + 1 < labels.size and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        ranks[order[index : end + 1]] = average_rank
        index = end + 1
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
