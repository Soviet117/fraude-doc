"""Figuras reproducibles de F4 para el informe y la sustentación.

Lee ``metrics.json`` (historial de entrenamiento) y ``eval.json`` (métricas
out-of-fold) y genera PNGs con curvas, matriz de confusión, recall por ataque y
curva de calibración. No calcula métricas nuevas: solo dibuja lo que ya está en
los artefactos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _values(history: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) if row.get(key) is not None else float("nan") for row in history]


def plot_training_curves(metrics: dict[str, Any], path: Path) -> Path:
    folds = metrics.get("folds", [])
    if not folds:
        raise ValueError("metrics.json no tiene folds para graficar")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for fold in folds:
        history = fold.get("history", [])
        if not history:
            continue
        epochs = [int(row["epoch"]) for row in history]
        label = f"fold {fold['fold']}"
        axes[0].plot(epochs, _values(history, "train_loss"), label=f"{label} train")
        axes[0].plot(
            epochs,
            _values(history, "val_loss"),
            linestyle="--",
            label=f"{label} val",
        )
        axes[1].plot(epochs, _values(history, "val_pr_auc"), marker="o", label=label)
        best_epoch = int(fold.get("best_epoch", 0))
        best_row = next((row for row in history if int(row["epoch"]) == best_epoch), None)
        if best_row is not None and best_row.get("val_pr_auc") is not None:
            axes[1].scatter(
                [best_epoch],
                [float(best_row["val_pr_auc"])],
                s=45,
                zorder=5,
                edgecolors="black",
            )

    axes[0].set_title("Pérdida por época")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("BCE con pos_weight")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("PR-AUC de validación (punto = mejor época)")
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_confusion(eval_result: dict[str, Any], path: Path) -> Path:
    pooled = eval_result["pooled"]
    matrix = np.array(
        [
            [int(pooled["tn"]), int(pooled["fp"])],
            [int(pooled["fn"]), int(pooled["tp"])],
        ]
    )
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=color, fontsize=14)
    axis.set_xticks([0, 1], ["Auténtico", "Manipulado"])
    axis.set_yticks([0, 1], ["Auténtico", "Manipulado"])
    axis.set_xlabel("predicción @0.5")
    axis.set_ylabel("etiqueta")
    axis.set_title(f"Matriz de confusión (n={int(pooled['n'])})")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_attack_recall(eval_result: dict[str, Any], path: Path) -> Path:
    groups = {
        name: values
        for name, values in eval_result.get("by_attack_base", {}).items()
        if name != "authentic"
    }
    if not groups:
        raise ValueError("eval.json no tiene desglose por ataque")

    names = list(groups)
    recalls = [float(groups[name]["recall"]) for name in names]
    counts = [int(groups[name]["n"]) for name in names]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(names, recalls, color="#3B6EA8")
    for bar, count, recall in zip(bars, counts, recalls, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(recall + 0.03, 0.96),
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("recall @0.5")
    axis.set_title("Recall por ataque (documentos base)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_calibration(eval_result: dict[str, Any], path: Path) -> Path:
    calibration = eval_result.get("calibration", {})
    raw = calibration.get("reliability_raw", [])
    fitted = calibration.get("reliability_calibrated", [])
    if not raw and not fitted:
        raise ValueError("eval.json no tiene curvas de calibración")

    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    axis.plot([0, 1], [0, 1], linestyle=":", color="gray", label="ideal")
    if raw:
        axis.plot(
            [float(row["confidence"]) for row in raw],
            [float(row["accuracy"]) for row in raw],
            marker="o",
            label="cruda",
        )
    if fitted:
        axis.plot(
            [float(row["confidence"]) for row in fitted],
            [float(row["accuracy"]) for row in fitted],
            marker="s",
            label="isotónica cruzada",
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("confianza media del bin")
    axis.set_ylabel("fracción de positivos")
    axis.set_title("Curva de fiabilidad")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def generate_all(
    metrics: dict[str, Any],
    eval_result: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_training_curves(metrics, out_dir / "training_curves.png"),
        plot_confusion(eval_result, out_dir / "confusion_matrix.png"),
        plot_attack_recall(eval_result, out_dir / "attack_recall.png"),
        plot_calibration(eval_result, out_dir / "calibration.png"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera figuras de F4.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/models/f4/metrics.json"),
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("artifacts/models/f4/eval.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f4/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    eval_result = json.loads(args.eval.read_text(encoding="utf-8"))
    paths = generate_all(metrics, eval_result, args.out)
    print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
