"""Métricas de localización para F5 (Pixel-F1, IoU y Pixel-AUC)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def pixel_confusion(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Devuelve ``(tp, fp, fn)`` a nivel de píxel."""
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities y targets deben tener la misma forma")
    predicted = probabilities >= threshold
    positive = targets > 0.5
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    return tp, fp, fn


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall, F1 e IoU a partir de conteos agregados."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
    }


def threshold_sweep(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    """Barrido de umbrales con métricas agregadas de píxel."""
    candidates = (
        list(thresholds)
        if thresholds is not None
        else [round(index / 20, 2) for index in range(1, 20)]
    )
    results: list[dict[str, float]] = []
    for threshold in candidates:
        tp, fp, fn = pixel_confusion(probabilities, targets, threshold=threshold)
        results.append({"threshold": round(float(threshold), 4), **metrics_from_counts(tp, fp, fn)})
    return results


def select_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Elige el umbral con mejor Pixel-F1; desempata cerca de 0.5."""
    candidates = (
        list(thresholds)
        if thresholds is not None
        else [round(index / 100, 2) for index in range(5, 96, 5)]
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        tp, fp, fn = pixel_confusion(probabilities, targets, threshold=threshold)
        f1 = metrics_from_counts(tp, fp, fn)["f1"]
        distance = abs(float(threshold) - 0.5)
        if best is None or f1 > best[0] or (f1 == best[0] and distance < best[2]):
            best = (f1, float(threshold), distance)
    if best is None:
        raise ValueError("No hay umbrales candidatos")
    tp, fp, fn = pixel_confusion(probabilities, targets, threshold=best[1])
    return {
        "threshold": round(best[1], 4),
        "metrics": metrics_from_counts(tp, fp, fn),
        "counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def pixel_auc(probabilities: np.ndarray, targets: np.ndarray) -> float | None:
    """ROC-AUC a nivel de píxel; ``None`` si solo hay una clase."""
    labels = (targets.ravel() > 0.5).astype(np.int8)
    scores = probabilities.ravel().astype(np.float64)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return round(float(roc_auc_score(labels, scores)), 4)
    except ImportError:
        return _roc_auc(labels, scores)


def bootstrap_pixel_ci(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = "f1",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float | None]:
    """IC bootstrap sobre imágenes, no sobre píxeles.

    Remuestrear píxeles ignoraría la correlación espacial dentro de una página.
    Cada fila debe traer ``tp``, ``fp`` y ``fn`` para el umbral evaluado.
    """
    if not rows:
        return {"low": None, "high": None}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(rows), size=len(rows))
        tp = sum(int(rows[index]["tp"]) for index in indices)
        fp = sum(int(rows[index]["fp"]) for index in indices)
        fn = sum(int(rows[index]["fn"]) for index in indices)
        value = metrics_from_counts(tp, fp, fn).get(metric)
        if value is not None:
            values.append(float(value))
    if not values:
        return {"low": None, "high": None}
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": round(float(low), 4), "high": round(float(high), 4)}


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
