"""Pruebas de las figuras reproducibles de F4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")

from fraude.models.report import generate_all


def _metrics() -> dict[str, Any]:
    return {
        "folds": [
            {
                "fold": 0,
                "best_epoch": 2,
                "history": [
                    {
                        "epoch": 1,
                        "train_loss": 1.0,
                        "val_loss": 1.1,
                        "val_pr_auc": 0.5,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.8,
                        "val_loss": 0.9,
                        "val_pr_auc": 0.7,
                    },
                ],
            }
        ]
    }


def _eval() -> dict[str, Any]:
    return {
        "pooled": {"n": 4, "tn": 1, "fp": 1, "fn": 1, "tp": 1},
        "by_attack_base": {
            "T1": {"n": 1, "recall": 0.0},
            "T2": {"n": 1, "recall": 1.0},
            "authentic": {"n": 2, "recall": 0.0},
        },
        "calibration": {
            "reliability_raw": [
                {"confidence": 0.2, "accuracy": 0.0},
                {"confidence": 0.8, "accuracy": 1.0},
            ],
            "reliability_calibrated": [
                {"confidence": 0.3, "accuracy": 0.0},
                {"confidence": 0.7, "accuracy": 1.0},
            ],
        },
    }


def test_generate_all_creates_four_pngs(tmp_path: Path) -> None:
    paths = generate_all(_metrics(), _eval(), tmp_path)

    assert len(paths) == 4
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
