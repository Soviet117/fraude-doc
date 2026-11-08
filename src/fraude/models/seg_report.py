"""Figuras reproducibles de F5 para el informe y la sustentación."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fraude.synth.validate import load_corpus


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
        axes[1].plot(epochs, _values(history, "val_pixel_f1"), marker="o", label=label)
        best_epoch = int(fold.get("best_epoch", 0))
        best_row = next((row for row in history if int(row["epoch"]) == best_epoch), None)
        if best_row is not None and best_row.get("val_pixel_f1") is not None:
            axes[1].scatter(
                [best_epoch],
                [float(best_row["val_pixel_f1"])],
                s=45,
                zorder=5,
                edgecolors="black",
            )
    axes[0].set_title("Pérdida por época (BCE + Dice)")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("pérdida")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("Pixel-F1 de validación (punto = mejor época)")
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("Pixel-F1 @0.5")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_threshold_sweep(eval_result: dict[str, Any], path: Path) -> Path:
    sweep = eval_result.get("sweep", [])
    if not sweep:
        raise ValueError("eval.json no tiene barrido de umbrales")
    thresholds = [row["threshold"] for row in sweep]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(thresholds, [row["f1"] for row in sweep], marker="o", label="Pixel-F1")
    axis.plot(thresholds, [row["iou"] for row in sweep], marker="s", label="IoU")
    fixed = eval_result["fixed_0_5"]
    calibrated = eval_result["calibrated"]
    oracle = eval_result["oracle"]
    axis.axvline(0.5, color="gray", linestyle=":", label="fijo 0.5")
    axis.axvline(
        float(np.mean(list(calibrated["thresholds_by_fold"].values()))),
        color="tab:green",
        linestyle="--",
        label="calibrado (media por fold)",
    )
    axis.axvline(
        float(oracle["threshold"]),
        color="tab:red",
        linestyle="--",
        label="oráculo",
    )
    axis.scatter([0.5], [fixed["f1"]], color="black", zorder=5)
    axis.set_xlabel("umbral")
    axis.set_ylabel("métrica de píxel")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Barrido de umbral (OOF)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_attack_f1(eval_result: dict[str, Any], path: Path) -> Path:
    groups = eval_result.get("by_attack", {})
    names = [name for name in groups if name != "authentic"]
    if not names:
        raise ValueError("eval.json no tiene desglose por ataque")
    fixed = [float(groups[name]["fixed_0_5"]["f1"]) for name in names]
    calibrated = [float(groups[name]["calibrated"]["f1"]) for name in names]
    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(positions - width / 2, fixed, width, label="fijo 0.5")
    axis.bar(positions + width / 2, calibrated, width, label="calibrado")
    axis.set_xticks(positions, names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Pixel-F1")
    axis.set_title("Pixel-F1 por ataque (OOF)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_qualitative(
    eval_result: dict[str, Any],
    oof_path: Path,
    data_dir: Path,
    path: Path,
    *,
    image_size: int = 256,
) -> Path:
    rows = [
        json.loads(line)
        for line in Path(oof_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("oof.jsonl vacío")
    records = {record.sample_id: record for record in load_corpus(data_dir)}
    manipulated = [row for row in rows if row["label"] == 1]
    authentic = [row for row in rows if row["label"] == 0]
    if not manipulated or not authentic:
        raise ValueError("Se necesitan filas auténticas y manipuladas")
    tp = max(manipulated, key=lambda row: float(row["pixel_f1"]))
    fn = min(manipulated, key=lambda row: float(row["pixel_f1"]))
    fp = max(authentic, key=lambda row: int(row["fp"]))
    tn = min(authentic, key=lambda row: int(row["fp"]))
    selected = [
        ("TP", tp),
        ("FN", fn),
        ("FP", fp),
        ("TN", tn),
    ]

    figure, axes = plt.subplots(4, 3, figsize=(10.5, 13.5))
    for row_index, (label, row) in enumerate(selected):
        record = records[row["sample_id"]]
        with Image.open(Path(data_dir) / record.image_path) as image:
            page = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        with Image.open(Path(oof_path).parent / row["map_path"]) as map_image:
            probability = np.asarray(map_image.convert("L"), dtype=np.float32) / 255.0
        mask = Image.new("L", (image_size, image_size), 0)
        if row.get("mask_path"):
            with Image.open(Path(data_dir) / row["mask_path"]) as mask_image:
                mask = mask_image.convert("L").resize(
                    (image_size, image_size),
                    Image.Resampling.NEAREST,
                )
        target = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        prediction = (probability >= 0.5).astype(np.float32)
        overlay = np.asarray(page, dtype=np.float32) / 255.0
        overlay = overlay.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + 0.45 * prediction, 0, 1)
        overlay[..., 1] = np.clip(overlay[..., 1] - 0.25 * prediction, 0, 1)
        overlay[..., 2] = np.clip(overlay[..., 2] - 0.25 * prediction, 0, 1)

        axes[row_index, 0].imshow(page)
        axes[row_index, 0].set_title(
            f"{label} · {row['sample_id']} · {row.get('attack') or 'auténtico'}",
            fontsize=8,
        )
        axes[row_index, 0].axis("off")
        axes[row_index, 1].imshow(target, cmap="Greens", vmin=0, vmax=1)
        axes[row_index, 1].set_title("máscara real", fontsize=8)
        axes[row_index, 1].axis("off")
        axes[row_index, 2].imshow(overlay)
        axes[row_index, 2].set_title(
            f"predicción @0.5 · F1 {row['pixel_f1']:.3f}",
            fontsize=8,
        )
        axes[row_index, 2].axis("off")
    figure.suptitle("F5 · ejemplos cualitativos (rojo = píxeles predichos)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def generate_all(
    metrics: dict[str, Any],
    eval_result: dict[str, Any],
    *,
    oof_path: Path,
    data_dir: Path,
    out_dir: Path,
    image_size: int = 256,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_training_curves(metrics, out_dir / "training_curves.png"),
        plot_threshold_sweep(eval_result, out_dir / "threshold_sweep.png"),
        plot_attack_f1(eval_result, out_dir / "attack_pixel_f1.png"),
        plot_qualitative(
            eval_result,
            oof_path,
            data_dir,
            out_dir / "qualitative.png",
            image_size=image_size,
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera figuras de F5.")
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/models/f5/metrics.json"))
    parser.add_argument("--eval", type=Path, default=Path("artifacts/models/f5/eval.json"))
    parser.add_argument("--oof", type=Path, default=Path("artifacts/models/f5/oof.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v2"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f5/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    eval_result = json.loads(args.eval.read_text(encoding="utf-8"))
    paths = generate_all(
        metrics,
        eval_result,
        oof_path=args.oof,
        data_dir=args.data,
        out_dir=args.out,
        image_size=args.image_size,
    )
    print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
