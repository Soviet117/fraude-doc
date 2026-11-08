"""Pruebas de la rama F4: caché, dataset, modelo, evaluación y Grad-CAM."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytest.importorskip("sklearn")

import torch
from torch import nn

from fraude.models.cache import build_cache, letterbox_square
from fraude.models.dataset import PageDataset
from fraude.models.evaluate import (
    brier_score,
    cross_fitted_isotonic,
    evaluate_oof,
    expected_calibration_error,
)
from fraude.models.gradcam import GradCAM, colorize, overlay, select_cases
from fraude.models.model import build_model, last_conv_layer
from fraude.models.train import select_records
from fraude.synth.dataset import DocumentRecord


def _record(sample_id: str, *, authentic: bool, image_path: str) -> DocumentRecord:
    return DocumentRecord(
        sample_id=sample_id,
        document_id="sha256:" + sample_id,
        source_id="persona_0",
        document_type="dni",
        template_version="v1",
        variant="authentic" if authentic else "tampered",
        parent_id=None if authentic else "sha256:base",
        is_authentic=authentic,
        primary_attack=None if authentic else "T1",
        image_path=image_path,
    )


def test_letterbox_keeps_page_and_square_size() -> None:
    image = Image.new("RGB", (100, 200), (255, 255, 255))
    result = letterbox_square(image, 64)

    assert result.size == (64, 64)
    assert result.getpixel((32, 32)) == (255, 255, 255)


def test_dataset_returns_tensor_label_and_index(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (80, 120), (240, 240, 240)).save(images / f"{name}.png")
    records = [
        _record("a", authentic=True, image_path="images/a.png"),
        _record("b", authentic=False, image_path="images/b.png"),
    ]
    cache_dir = tmp_path / "cache"
    build_cache(records, tmp_path, cache_dir, size=64)

    dataset = PageDataset(records, cache_dir, image_size=56, train=False)
    tensor, label, index = dataset[1]

    assert tensor.shape == (3, 56, 56)
    assert float(label) == 1.0
    assert int(index) == 1


def test_model_forward_without_pretrained() -> None:
    model = build_model("efficientnet_b0", pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))

    assert output.shape == (1, 1)
    assert isinstance(last_conv_layer(model), nn.Conv2d)


def test_calibration_metrics_and_cross_fitted_isotonic() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    folds = [0, 0, 1, 1, 2, 2]

    calibrated = cross_fitted_isotonic(labels, scores, folds)

    assert len(calibrated) == len(scores)
    assert all(0.0 <= value <= 1.0 for value in calibrated)
    assert brier_score(labels, scores) < 0.1
    assert 0.0 <= expected_calibration_error(labels, scores) <= 1.0


def test_evaluate_oof_reports_thresholds_and_breakdowns() -> None:
    rows = []
    for fold in range(2):
        for index in range(10):
            label = index % 2
            rows.append(
                {
                    "sample_id": f"s{fold}_{index}",
                    "source_id": f"p{index % 5}",
                    "fold": fold,
                    "label": label,
                    "score": 0.9 if label else 0.1,
                    "attack": "T1" if label else None,
                    "document_type": "dni",
                    "augmentations": [],
                }
            )

    result = evaluate_oof(rows, n_boot=50)

    assert result["pooled"]["f1"] == 1.0
    assert result["thresholds"]["oracle"]["threshold"] == 0.5
    assert result["by_attack_base"]["T1"]["recall"] == 1.0
    assert result["by_fold"]["0"]["n"] == 10


def test_gradcam_and_overlay_on_tiny_model() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x.mean(dim=(2, 3))

    model = Tiny().eval()
    cam = GradCAM(model, model.conv2)
    heat, score = cam(torch.rand(1, 3, 32, 32))

    assert heat.shape == (32, 32)
    assert 0.0 <= score <= 1.0
    base = Image.new("RGB", (32, 32), (255, 255, 255))
    assert colorize(heat).size == (32, 32)
    assert overlay(base, heat).size == (32, 32)


def test_select_cases_picks_tp_tn_fp_fn() -> None:
    rows = [
        {"sample_id": "tp", "label": 1, "score": 0.9},
        {"sample_id": "fn", "label": 1, "score": 0.2},
        {"sample_id": "fp", "label": 0, "score": 0.7},
        {"sample_id": "tn", "label": 0, "score": 0.1},
    ]

    cases = {case["case"]: case["sample_id"] for case in select_cases(rows)}

    assert cases == {"tp": "tp", "fn": "fn", "fp": "fp", "tn": "tn"}


def test_json_round_trip_of_evaluation(tmp_path: Path) -> None:
    payload = {"n": 2, "values": [np.float64(0.5)]}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload, default=float), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["values"] == [0.5]


def test_select_records_excludes_semantic_attacks() -> None:
    authentic = _record("auth", authentic=True, image_path="images/auth.png")
    visual = _record("visual", authentic=False, image_path="images/visual.png")
    semantic = replace(
        _record("semantic", authentic=False, image_path="images/semantic.png"),
        primary_attack="T7",
    )

    selected = select_records(
        [authentic, visual, semantic],
        include_attacks=("T1", "T2"),
    )

    assert [record.sample_id for record in selected] == ["auth", "visual"]
    assert len(select_records([authentic, visual, semantic], include_attacks=None)) == 3


def test_infer_preprocess_shape() -> None:
    from fraude.models.infer import preprocess

    tensor = preprocess(Image.new("RGB", (80, 120), (255, 255, 255)), image_size=64)

    assert tensor.shape == (1, 3, 64, 64)
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 3.0


def test_snapshot_state_does_not_alias_live_parameters() -> None:
    from fraude.models.train import _snapshot_state

    model = nn.Linear(2, 1)
    snapshot = _snapshot_state(model)
    with torch.no_grad():
        model.weight.add_(1.0)

    assert not torch.equal(snapshot["weight"], model.weight)


def test_gradcam_cases_are_restricted_to_checkpoint_fold(tmp_path: Path) -> None:
    from fraude.models.gradcam import _load_cases

    rows = [
        {"sample_id": "a", "fold": 0, "label": 1, "score": 0.9},
        {"sample_id": "b", "fold": 0, "label": 1, "score": 0.2},
        {"sample_id": "c", "fold": 0, "label": 0, "score": 0.7},
        {"sample_id": "d", "fold": 0, "label": 0, "score": 0.1},
        {"sample_id": "e", "fold": 1, "label": 1, "score": 0.8},
        {"sample_id": "f", "fold": 1, "label": 1, "score": 0.3},
        {"sample_id": "g", "fold": 1, "label": 0, "score": 0.6},
        {"sample_id": "h", "fold": 1, "label": 0, "score": 0.2},
    ]
    path = tmp_path / "oof.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cases = _load_cases(path, fold=1)

    assert cases
    assert {case["sample_id"] for case in cases} <= {"e", "f", "g", "h"}
