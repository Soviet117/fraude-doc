"""Pruebas de los calibradores de F7."""

from __future__ import annotations

import numpy as np

from fraude.fusion.calibration import (
    cross_fitted_calibration,
    fit_calibrator,
)


def test_identity_calibrator_returns_input() -> None:
    calibrator = fit_calibrator("identity", [0.1, 0.8], [0, 1])

    values = calibrator.predict([0.2, 0.9])

    assert calibrator.method == "identity"
    assert np.allclose(values, [0.2, 0.9])


def test_isotonic_and_platt_stay_in_range() -> None:
    scores = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95] * 4
    labels = [0, 0, 0, 0, 1, 1, 1] * 4
    for method in ("isotonic", "platt"):
        calibrator = fit_calibrator(method, scores, labels)
        values = calibrator.predict(scores)

        assert values.shape == (len(scores),)
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_cross_fitted_calibration_keeps_length_and_bounds() -> None:
    labels = [0, 0, 1, 1, 0, 1, 0, 1]
    scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6]
    folds = [0, 0, 1, 1, 2, 2, 3, 3]

    values = cross_fitted_calibration(labels, scores, folds, method="isotonic")

    assert len(values) == len(scores)
    assert min(values) >= 0.0
    assert max(values) <= 1.0
