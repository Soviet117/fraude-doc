"""Evaluación de F5: Pixel-F1, IoU y Pixel-AUC con umbral fijo, calibrado y oráculo."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fraude.models.seg_metrics import (
    bootstrap_pixel_ci,
    metrics_from_counts,
    pixel_auc,
    pixel_confusion,
    select_threshold,
    threshold_sweep,
)

THRESHOLDS = [round(index / 20, 2) for index in range(1, 20)]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_mask(path: Path | None, image_size: int) -> np.ndarray:
    mask = Image.new("L", (image_size, image_size), 0)
    if path is not None and Path(path).is_file():
        with Image.open(path) as image:
            mask = image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)


def _load_map(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if array.shape != (image_size, image_size):
        raise ValueError(
            f"Mapa {path} con forma {array.shape}, se esperaba {(image_size, image_size)}"
        )
    return array


def _aggregate(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    tp = fp = fn = 0
    for row in rows:
        probabilities = row["probabilities"]
        targets = row["targets"]
        row_tp, row_fp, row_fn = pixel_confusion(probabilities, targets, threshold=threshold)
        tp += row_tp
        fp += row_fp
        fn += row_fn
    metrics = metrics_from_counts(tp, fp, fn)
    return {
        "threshold": round(float(threshold), 4),
        "n_images": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **metrics,
    }


def _with_counts(rows: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        tp, fp, fn = pixel_confusion(
            row["probabilities"],
            row["targets"],
            threshold=threshold,
        )
        output.append({**row, "tp": tp, "fp": fp, "fn": fn})
    return output


def _bootstrap(
    rows: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    return {
        metric: bootstrap_pixel_ci(rows, metric=metric, n_boot=n_boot, seed=seed)
        for metric in ("f1", "iou")
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return dict(sorted(grouped.items()))


def evaluate_oof(
    oof_path: Path,
    *,
    data_dir: Path,
    image_size: int = 256,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Carga mapas OOF y calcula la tabla de localización."""
    rows = _load_rows(Path(oof_path))
    if not rows:
        raise ValueError(f"No hay filas OOF en {oof_path}")
    data_dir = Path(data_dir)
    for row in rows:
        row["probabilities"] = _load_map(Path(oof_path).parent / row["map_path"], image_size)
        mask_path = data_dir / row["mask_path"] if row.get("mask_path") else None
        row["targets"] = _load_mask(mask_path, image_size)

    all_probabilities = np.concatenate([row["probabilities"].ravel() for row in rows])
    all_targets = np.concatenate([row["targets"].ravel() for row in rows])

    fixed_rows = _with_counts(rows, threshold=0.5)
    fixed = _aggregate(rows, threshold=0.5)
    fixed["ci95"] = _bootstrap(fixed_rows, n_boot=n_boot, seed=seed)

    sweep = threshold_sweep(all_probabilities, all_targets, thresholds=THRESHOLDS)

    calibrated_rows: list[dict[str, Any]] = []
    per_fold: dict[str, Any] = {}
    folds = sorted({int(row["fold"]) for row in rows})
    for fold in folds:
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        val_rows = [row for row in rows if int(row["fold"]) == fold]
        train_prob = np.concatenate([row["probabilities"].ravel() for row in train_rows])
        train_target = np.concatenate([row["targets"].ravel() for row in train_rows])
        selected = select_threshold(train_prob, train_target, thresholds=THRESHOLDS)
        threshold = float(selected["threshold"])
        fold_rows = _with_counts(val_rows, threshold=threshold)
        calibrated_rows.extend(fold_rows)
        per_fold[str(fold)] = {
            "threshold": threshold,
            "n_images": len(val_rows),
            **metrics_from_counts(
                sum(row["tp"] for row in fold_rows),
                sum(row["fp"] for row in fold_rows),
                sum(row["fn"] for row in fold_rows),
            ),
        }

    calibrated = _aggregate(
        [{"probabilities": row["probabilities"], "targets": row["targets"]} for row in rows],
        threshold=0.5,
    )
    # La agregación anterior solo vale para el umbral 0.5; se recalcula con los
    # conteos por fold ya seleccionados.
    calibrated = {
        "thresholds_by_fold": {str(fold): per_fold[str(fold)]["threshold"] for fold in folds},
        "n_images": len(calibrated_rows),
        "tp": sum(row["tp"] for row in calibrated_rows),
        "fp": sum(row["fp"] for row in calibrated_rows),
        "fn": sum(row["fn"] for row in calibrated_rows),
        **metrics_from_counts(
            sum(row["tp"] for row in calibrated_rows),
            sum(row["fp"] for row in calibrated_rows),
            sum(row["fn"] for row in calibrated_rows),
        ),
    }
    calibrated["ci95"] = _bootstrap(calibrated_rows, n_boot=n_boot, seed=seed)

    oracle = select_threshold(all_probabilities, all_targets, thresholds=THRESHOLDS)
    oracle["n_images"] = len(rows)
    oracle["ci95"] = _bootstrap(
        _with_counts(rows, threshold=float(oracle["threshold"])),
        n_boot=n_boot,
        seed=seed,
    )

    image_aucs = [
        value
        for row in rows
        if (value := pixel_auc(row["probabilities"], row["targets"])) is not None
    ]
    result = {
        "schema_version": "1.0",
        "generator": "fraude.models.seg_evaluate",
        "n_images": len(rows),
        "n_positive_images": sum(1 for row in rows if row["n_positive"] > 0),
        "n_positive_pixels": int(all_targets.sum()),
        "pixel_prevalence": round(float(all_targets.mean()), 6),
        "pixel_auc": pixel_auc(all_probabilities, all_targets),
        "pixel_auc_image_mean": round(float(np.mean(image_aucs)), 4) if image_aucs else None,
        "pixel_auc_image_std": round(float(np.std(image_aucs)), 4) if image_aucs else None,
        "fixed_0_5": fixed,
        "calibrated": calibrated,
        "oracle": oracle,
        "sweep": sweep,
        "by_attack": {
            key: {
                "n_images": len(group),
                "fixed_0_5": _aggregate(group, threshold=0.5),
                "calibrated": _aggregate_counts(group, calibrated_rows),
            }
            for key, group in _group(rows, "attack").items()
        },
        "by_document_type": {
            key: {
                "n_images": len(group),
                "fixed_0_5": _aggregate(group, threshold=0.5),
                "calibrated": _aggregate_counts(group, calibrated_rows),
            }
            for key, group in _group(rows, "document_type").items()
        },
        "per_fold": per_fold,
    }
    return result


def _aggregate_counts(
    group: list[dict[str, Any]],
    calibrated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = {row["sample_id"] for row in group}
    selected = [row for row in calibrated_rows if row["sample_id"] in ids]
    tp = sum(row["tp"] for row in selected)
    fp = sum(row["fp"] for row in selected)
    fn = sum(row["fn"] for row in selected)
    return {
        "n_images": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **metrics_from_counts(tp, fp, fn),
    }


def write_markdown(result: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = result["fixed_0_5"]
    calibrated = result["calibrated"]
    oracle = result["oracle"]
    lines = [
        "# F5 — evaluación de localización (U-Net)",
        "",
        f"- Imágenes OOF: {result['n_images']} "
        f"({result['n_positive_images']} con máscara positiva)",
        f"- Píxeles positivos: {result['n_positive_pixels']} "
        f"(prevalencia {result['pixel_prevalence']:.4%})",
        f"- Pixel-AUC global: {_fmt(result['pixel_auc'])}",
        f"- Pixel-AUC media por imagen: {_fmt(result['pixel_auc_image_mean'])} "
        f"± {_fmt(result['pixel_auc_image_std'])}",
        "",
        "## Umbrales",
        "",
        "| Escenario | Umbral | Precision | Recall | Pixel-F1 | IoU |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Fijo 0.5 | 0.50 | {_fmt(fixed['precision'])} | {_fmt(fixed['recall'])} | "
        f"{_fmt(fixed['f1'])} | {_fmt(fixed['iou'])} |",
        f"| Calibrado por fold | por fold | {_fmt(calibrated['precision'])} | "
        f"{_fmt(calibrated['recall'])} | {_fmt(calibrated['f1'])} | {_fmt(calibrated['iou'])} |",
        f"| Oráculo | {oracle['threshold']:.2f} | {_fmt(oracle['metrics']['precision'])} | "
        f"{_fmt(oracle['metrics']['recall'])} | {_fmt(oracle['metrics']['f1'])} | "
        f"{_fmt(oracle['metrics']['iou'])} |",
        "",
        "## Intervalos bootstrap (sobre imágenes)",
        "",
        "| Escenario | Pixel-F1 | IC 95 % | IoU | IC 95 % |",
        "|---|---:|---|---:|---|",
        f"| Fijo 0.5 | {_fmt(fixed['f1'])} | {_ci(fixed['ci95']['f1'])} | "
        f"{_fmt(fixed['iou'])} | {_ci(fixed['ci95']['iou'])} |",
        f"| Calibrado | {_fmt(calibrated['f1'])} | {_ci(calibrated['ci95']['f1'])} | "
        f"{_fmt(calibrated['iou'])} | {_ci(calibrated['ci95']['iou'])} |",
        "",
        "## Por ataque (documentos base)",
        "",
        "| Grupo | n | Pixel-F1 @0.5 | Pixel-F1 calibrado | IoU @0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, group in result["by_attack"].items():
        lines.append(
            f"| {key} | {group['n_images']} | {_fmt(group['fixed_0_5']['f1'])} | "
            f"{_fmt(group['calibrated']['f1'])} | {_fmt(group['fixed_0_5']['iou'])} |"
        )
    lines.extend(
        [
            "",
            "## Por tipo documental",
            "",
            "| Tipo | n | Pixel-F1 @0.5 | Pixel-F1 calibrado | IoU @0.5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, group in result["by_document_type"].items():
        lines.append(
            f"| {key} | {group['n_images']} | {_fmt(group['fixed_0_5']['f1'])} | "
            f"{_fmt(group['calibrated']['f1'])} | {_fmt(group['fixed_0_5']['iou'])} |"
        )
    lines.extend(
        [
            "",
            "## Por fold",
            "",
            "| Fold | Umbral calibrado | n | Pixel-F1 | IoU |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold, values in result["per_fold"].items():
        lines.append(
            f"| {fold} | {values['threshold']:.2f} | {values['n_images']} | "
            f"{_fmt(values['f1'])} | {_fmt(values['iou'])} |"
        )
    lines.extend(
        [
            "",
            "> Corpus sintético. El umbral calibrado de cada fold se elige con los",
            "> otros folds; el oráculo usa el mismo OOF y solo sirve como techo",
            "> diagnóstico. Pixel-AUC global se calcula sobre todos los píxeles OOF.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def _ci(value: dict[str, float | None]) -> str:
    if value.get("low") is None or value.get("high") is None:
        return "—"
    return f"{value['low']:.4f}-{value['high']:.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa F5 (localización).")
    parser.add_argument("--oof", type=Path, default=Path("artifacts/models/f5/oof.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v2"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f5/eval.json"))
    parser.add_argument("--markdown", type=Path, default=Path("artifacts/models/f5/eval.md"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_oof(
        args.oof,
        data_dir=args.data,
        image_size=args.image_size,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.markdown)
    print(json.dumps(result["fixed_0_5"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
