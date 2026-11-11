"""Calibradores de score para F7 (isotónico y Platt).

Con 50 documentos, la isotónica puede sobreajustar y empeorar el ECE. Este
módulo permite ajustar ambos métodos, compararlos de forma cruzada y guardar el
que mejor calibra. El calibrador expone siempre ``predict(values)`` para que la
inferencia no dependa del método elegido.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

METHODS = ("identity", "isotonic", "platt")


def _as_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


@dataclass
class ScoreCalibrator:
    """Envoltorio uniforme de un calibrador ajustado."""

    method: str
    model: Any

    def predict(self, values: list[float] | np.ndarray) -> np.ndarray:
        scores = _as_array(values)
        if self.method == "identity":
            return np.clip(scores, 0.0, 1.0)
        if self.method == "platt":
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(scores), dtype=np.float64)


def fit_calibrator(
    method: str,
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> ScoreCalibrator:
    """Ajusta un calibrador isotónico o Platt sobre scores crudos."""
    scores_array = _as_array(scores)
    labels_array = _as_array(labels)
    if method == "identity":
        return ScoreCalibrator(method="identity", model=None)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(scores_array.reshape(-1, 1), labels_array.astype(int))
        return ScoreCalibrator(method="platt", model=model)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores_array, labels_array)
        return ScoreCalibrator(method="isotonic", model=model)
    raise ValueError(f"Método de calibración desconocido: {method}")


def cross_fitted_calibration(
    labels: list[int],
    scores: list[float],
    folds: list[int],
    *,
    method: str,
) -> list[float]:
    """Calibra cada fold con un calibrador ajustado en los folds restantes."""
    labels_array = _as_array(labels)
    scores_array = _as_array(scores)
    folds_array = np.asarray(folds, dtype=np.int64)
    calibrated = scores_array.copy()
    for fold in sorted(set(folds_array.tolist())):
        train_mask = folds_array != fold
        test_mask = folds_array == fold
        if len(set(labels_array[train_mask].tolist())) < 2:
            continue
        calibrator = fit_calibrator(method, scores_array[train_mask], labels_array[train_mask])
        calibrated[test_mask] = calibrator.predict(scores_array[test_mask])
    return [float(value) for value in np.clip(calibrated, 0.0, 1.0)]
