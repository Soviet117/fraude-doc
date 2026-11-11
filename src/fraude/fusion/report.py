"""Figuras reproducibles de F7 para el Cap. IV y la sustentación."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/mpl").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

NAVY = "#0B2545"
BLUE = "#2F6FED"
GREEN = "#1B7F3B"
AMBER = "#B26A00"
RED = "#B3261E"
GRAY = "#5F6B7A"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _style(axes) -> None:
    axes.grid(True, color="#E3E9F1", linewidth=0.7)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def roc_pr(oof: list[dict], out: Path) -> None:
    labels = np.asarray([row["label"] for row in oof], dtype=int)
    scores = np.asarray([row["calibrated_score"] for row in oof], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0].plot(fpr, tpr, color=BLUE, linewidth=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    axes[0].set_title("Curva ROC · F7 OOF")
    axes[0].legend(loc="lower right")
    _style(axes[0])

    precision, recall, _ = precision_recall_curve(labels, scores)
    axes[1].plot(recall, precision, color=GREEN, linewidth=2)
    axes[1].axhline(labels.mean(), color=GRAY, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Curva PR · prevalencia {labels.mean():.2f}")
    _style(axes[1])
    figure.tight_layout()
    figure.savefig(out / "roc_pr.png", dpi=190, facecolor="white")
    plt.close(figure)


def confusion(oof: list[dict], out: Path) -> None:
    labels = np.asarray([row["label"] for row in oof], dtype=int)
    scores = np.asarray([row["calibrated_score"] for row in oof], dtype=float)
    predictions = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axes = plt.subplots(figsize=(5.2, 4.4))
    image = axes.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if matrix[row, column] > matrix.max() / 2 else NAVY,
                fontweight="bold",
            )
    axes.set_xticks([0, 1], ["Pred. auténtico", "Pred. manipulado"])
    axes.set_yticks([0, 1], ["Real auténtico", "Real manipulado"])
    axes.set_title("Matriz de confusión · F7 @0.5")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(out / "confusion.png", dpi=190, facecolor="white")
    plt.close(figure)


def calibration_plot(oof: list[dict], metrics: dict, out: Path) -> None:
    labels = np.asarray([row["label"] for row in oof], dtype=int)
    raw = np.asarray([row["raw_score"] for row in oof], dtype=float)
    calibrated = np.asarray([row["calibrated_score"] for row in oof], dtype=float)
    figure, axes = plt.subplots(figsize=(5.6, 4.8))
    axes.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
    for values, label, color in (
        (raw, "Crudo", BLUE),
        (calibrated, "Calibrado", GREEN),
    ):
        bins = np.linspace(0, 1, 9)
        centers, observed = [], []
        for index in range(len(bins) - 1):
            mask = (values >= bins[index]) & (values < bins[index + 1])
            if mask.sum() < 2:
                continue
            centers.append(float(values[mask].mean()))
            observed.append(float(labels[mask].mean()))
        axes.plot(centers, observed, "o-", color=color, linewidth=1.8, markersize=5, label=label)
    axes.set_xlabel("Score medio del bin")
    axes.set_ylabel("Fracción de manipulados")
    axes.set_title("Curva de fiabilidad · F7")
    axes.legend(loc="upper left")
    _style(axes)
    figure.tight_layout()
    figure.savefig(out / "calibration.png", dpi=190, facecolor="white")
    plt.close(figure)


def threshold_sweep(oof: list[dict], out: Path) -> None:
    labels = np.asarray([row["label"] for row in oof], dtype=int)
    scores = np.asarray([row["calibrated_score"] for row in oof], dtype=float)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s, fprs = [], []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    axes.plot(thresholds, f1s, color=GREEN, linewidth=2, label="F1")
    axes.plot(thresholds, fprs, color=RED, linewidth=2, label="FPR")
    axes.axvline(0.5, color=GRAY, linestyle="--", linewidth=1, label="umbral 0.5")
    axes.set_xlabel("Umbral")
    axes.set_ylabel("Métrica")
    axes.set_title("Barrido de umbral · F7 OOF")
    axes.legend(loc="best")
    _style(axes)
    figure.tight_layout()
    figure.savefig(out / "threshold_sweep.png", dpi=190, facecolor="white")
    plt.close(figure)


def ablation(metrics: dict, out: Path) -> None:
    names = list(metrics.get("ablations", {}))
    values = [metrics["ablations"][name]["metrics"].get("pr_auc", 0.0) for name in names]
    figure, axes = plt.subplots(figsize=(8.2, 4.2))
    bars = axes.bar(names, values, color=[NAVY, BLUE, BLUE, AMBER, GREEN][: len(names)])
    for bar, value in zip(bars, values, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
            color=NAVY,
            fontweight="bold",
        )
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("PR-AUC")
    axes.set_title("Ablaciones acumulativas · F7")
    _style(axes)
    figure.tight_layout()
    figure.savefig(out / "ablations.png", dpi=190, facecolor="white")
    plt.close(figure)


def importance(metrics: dict, out: Path) -> None:
    items = metrics.get("feature_importance", [])[:12]
    names = [item["feature"] for item in reversed(items)]
    values = [item["mean_abs_shap"] for item in reversed(items)]
    figure, axes = plt.subplots(figsize=(8.2, 5.0))
    axes.barh(names, values, color=BLUE)
    axes.set_xlabel("SHAP medio absoluto")
    axes.set_title("Importancia de señales · F7")
    _style(axes)
    figure.tight_layout()
    figure.savefig(out / "importance.png", dpi=190, facecolor="white")
    plt.close(figure)


def build(metrics_path: Path, oof_path: Path, out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    oof = _load_jsonl(Path(oof_path))
    roc_pr(oof, out_dir)
    confusion(oof, out_dir)
    calibration_plot(oof, metrics, out_dir)
    threshold_sweep(oof, out_dir)
    ablation(metrics, out_dir)
    importance(metrics, out_dir)
    return sorted(str(path) for path in out_dir.glob("*.png"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Figuras de F7.")
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/models/f7/metrics.json"))
    parser.add_argument("--oof", type=Path, default=Path("artifacts/models/f7/oof.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f7/figures"))
    args = parser.parse_args(argv)
    files = build(args.metrics, args.oof, args.out)
    print(json.dumps(files, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
