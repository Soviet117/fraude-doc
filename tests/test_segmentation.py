"""Pruebas de F5: dataset, U-Net, métricas y evaluación OOF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fraude.models.seg_dataset import SegmentationDataset
from fraude.models.seg_evaluate import evaluate_oof
from fraude.models.seg_metrics import (
    bootstrap_pixel_ci,
    metrics_from_counts,
    pixel_auc,
    pixel_confusion,
    select_threshold,
)
from fraude.models.seg_train import _dice_loss
from fraude.models.unet import UNet
from fraude.synth.dataset import DocumentRecord


def _record(tmp_path: Path, *, mask: bool = True) -> DocumentRecord:
    image = Image.new("RGB", (64, 64), "white")
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    mask_paths: tuple[str, ...] = ()
    if mask:
        mask_image = Image.new("L", (64, 64), 0)
        for x in range(20, 40):
            for y in range(20, 40):
                mask_image.putpixel((x, y), 255)
        mask_file = tmp_path / "masks" / "sample.png"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        mask_image.save(mask_file)
        mask_paths = ("masks/sample.png",)
    return DocumentRecord(
        sample_id="sample",
        document_id="sha256:" + "0" * 64,
        source_id="persona_0000",
        document_type="dni",
        template_version="v1",
        variant="tampered" if mask else "authentic",
        parent_id=None,
        is_authentic=not mask,
        attack_types=("T1",) if mask else (),
        primary_attack="T1" if mask else None,
        image_path="images/sample.png",
        mask_paths=mask_paths,
    )


def test_segmentation_dataset_returns_aligned_mask(tmp_path: Path) -> None:
    dataset = SegmentationDataset([_record(tmp_path)], tmp_path, image_size=32)
    image, mask, index = dataset[0]

    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert float(mask.sum()) > 0
    assert int(index) == 0


def test_segmentation_dataset_uses_empty_mask_for_authentic(tmp_path: Path) -> None:
    dataset = SegmentationDataset([_record(tmp_path, mask=False)], tmp_path, image_size=32)
    _, mask, _ = dataset[0]
    assert float(mask.sum()) == 0.0


def test_unet_forward_shape() -> None:
    model = UNet(base_channels=4)
    output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_dice_loss_is_low_for_perfect_prediction() -> None:
    targets = torch.zeros(1, 1, 8, 8)
    targets[..., 2:5, 2:5] = 1.0
    logits = torch.where(targets > 0.5, 20.0, -20.0)
    loss = _dice_loss(logits, targets)
    assert float(loss) < 0.01


def test_pixel_metrics_from_counts() -> None:
    metrics = metrics_from_counts(tp=8, fp=2, fn=2)
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.8
    assert metrics["f1"] == 0.8
    assert metrics["iou"] == 0.6667


def test_pixel_confusion_and_threshold() -> None:
    probabilities = np.array([[0.9, 0.4], [0.6, 0.1]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    tp, fp, fn = pixel_confusion(probabilities, targets, threshold=0.5)
    assert (tp, fp, fn) == (2, 0, 0)
    selected = select_threshold(probabilities, targets, thresholds=[0.3, 0.5, 0.8])
    assert selected["threshold"] == 0.5
    assert selected["metrics"]["f1"] == 1.0


def test_pixel_auc_is_one_for_separable_scores() -> None:
    probabilities = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    targets = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert pixel_auc(probabilities, targets) == 1.0


def test_bootstrap_pixel_ci_returns_bounds() -> None:
    rows = [
        {"tp": 10, "fp": 1, "fn": 2},
        {"tp": 8, "fp": 2, "fn": 4},
        {"tp": 12, "fp": 0, "fn": 1},
    ]
    result = bootstrap_pixel_ci(rows, metric="f1", n_boot=50, seed=42)
    assert result["low"] is not None
    assert result["high"] is not None
    assert result["low"] <= result["high"]


def test_evaluate_oof_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    rows = []
    for index, (attack, label) in enumerate(
        [("authentic", 0), ("authentic", 0), ("T1", 1), ("T1", 1)]
    ):
        sample_id = f"sample_{index}"
        mask = Image.new("L", (16, 16), 0)
        if label:
            for x in range(4, 8):
                for y in range(4, 8):
                    mask.putpixel((x, y), 255)
        mask_path = data_dir / "masks" / f"{sample_id}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path)
        probability = np.zeros((16, 16), dtype=np.uint8)
        if label:
            probability[4:8, 4:8] = 220
        else:
            probability[0:2, 0:2] = 40
        map_path = out_dir / "oof_maps" / f"{sample_id}.png"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(probability, mode="L").save(map_path)
        rows.append(
            {
                "sample_id": sample_id,
                "source_id": f"persona_{index}",
                "fold": index % 2,
                "label": label,
                "attack": attack,
                "document_type": "dni",
                "augmentations": [],
                "map_path": str(map_path.relative_to(out_dir)),
                "mask_path": str(mask_path.relative_to(data_dir)),
                "n_positive": int((np.asarray(mask) > 127).sum()),
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pixel_f1": 0.0,
                "pixel_iou": 0.0,
                "max_prob": 0.0,
                "mean_prob": 0.0,
            }
        )
    oof_path = out_dir / "oof.jsonl"
    oof_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_oof(
        oof_path,
        data_dir=data_dir,
        image_size=16,
        n_boot=10,
        seed=42,
    )

    assert result["n_images"] == 4
    assert result["fixed_0_5"]["f1"] == 1.0
    assert result["pixel_auc"] == 1.0
    assert "calibrated" in result
    assert "by_attack" in result
