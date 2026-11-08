"""Grad-CAM sobre la última convolución de F4.

Genera mapas de calor para aciertos y errores concretos (TP, TN, FP, FN). No
es una prueba de localización: F4 es un clasificador de página y Grad-CAM
solo indica qué regiones pesaron en la decisión. La localización formal es F5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from fraude.models.cache import cache_path
from fraude.models.dataset import PageDataset
from fraude.models.model import build_model, last_conv_layer
from fraude.synth.dataset import DocumentRecord
from fraude.synth.validate import load_corpus


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: Any, inputs: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: Any, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor) -> tuple[np.ndarray, float]:
        if self.activations is None or self.gradients is None:
            self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(image).squeeze(1)
            score = float(torch.sigmoid(logits).item())
            logits.backward()
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), score


def colorize(cam: np.ndarray) -> Image.Image:
    """Mapa de calor azul→rojo sin depender de matplotlib."""
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.6 * cam, 0.0, 1.0)
    green = np.clip(1.4 * (1.0 - np.abs(cam - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(1.6 * (1.0 - cam), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = colorize(cam).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heat, alpha=alpha)


def select_cases(oof: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Elige TP, TN, FP y FN representativos por score."""
    cases: dict[str, dict[str, Any]] = {}
    positives = [row for row in oof if int(row["label"]) == 1]
    negatives = [row for row in oof if int(row["label"]) == 0]
    if positives:
        cases["tp"] = max(positives, key=lambda row: float(row["score"]))
        cases["fn"] = min(positives, key=lambda row: float(row["score"]))
    if negatives:
        cases["fp"] = max(negatives, key=lambda row: float(row["score"]))
        cases["tn"] = min(negatives, key=lambda row: float(row["score"]))
    return [{"case": name, **row} for name, row in cases.items()]


def run(
    checkpoint: Path,
    data_dir: Path,
    out_dir: Path,
    *,
    oof_path: Path | None = None,
    cache_dir: Path | None = None,
    max_cases: int = 8,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {})
    model_name = str(config.get("model_name", "efficientnet_b0"))
    image_size = int(config.get("image_size", 224))
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    cam_model = GradCAM(model, last_conv_layer(model))
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(checkpoint).parent / "cache"

    records = {record.sample_id: record for record in load_corpus(data_dir)}
    fold = payload.get("fold")
    cases = _load_cases(oof_path, fold=fold) if oof_path else []
    if not cases:
        cases = [
            {
                "case": "sample",
                "sample_id": record.sample_id,
                "label": 0 if record.is_authentic else 1,
                "score": None,
            }
            for record in list(records.values())[:max_cases]
        ]
    cases = cases[:max_cases]

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = PageDataset(list(records.values()), cache_dir, image_size=image_size)
    index: list[dict[str, Any]] = []
    for case in cases:
        record = records.get(str(case["sample_id"]))
        if record is None:
            continue
        tensor, _, _ = _load_case_tensor(record, dataset, cache_dir, image_size)
        tensor = tensor.unsqueeze(0).to(device)
        cam, score = cam_model(tensor)
        base = _case_image(record, cache_dir, image_size)
        heat = overlay(base, cam)
        filename = f"{case['case']}_{record.sample_id}.png"
        heat.save(out_dir / filename)
        index.append(
            {
                "case": case["case"],
                "sample_id": record.sample_id,
                "attack": record.primary_attack,
                "document_type": record.document_type,
                "label": 0 if record.is_authentic else 1,
                "score": round(score, 4),
                "image": filename,
            }
        )

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"cases": index, "count": len(index)}


def _load_cases(path: Path, *, fold: int | None = None) -> list[dict[str, Any]]:
    """Carga OOF y, si se indica, se queda solo con el fold del checkpoint.

    Un caso de otro fold habría visto ese documento durante el entrenamiento y
    el Grad-CAM no sería out-of-fold.
    """
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if fold is not None and int(row.get("fold", -1)) != int(fold):
            continue
        rows.append(row)
    return select_cases(rows)


def _load_case_tensor(
    record: DocumentRecord,
    dataset: PageDataset,
    cache_dir: Path,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset.records = [record]
    return dataset[0]


def _case_image(record: DocumentRecord, cache_dir: Path, image_size: int) -> Image.Image:
    path = cache_path(cache_dir, record.sample_id)
    with Image.open(path) as image:
        page = image.convert("RGB")
    if page.width > image_size:
        left = (page.width - image_size) // 2
        top = (page.height - image_size) // 2
        page = page.crop((left, top, left + image_size, top + image_size))
    return page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera Grad-CAM para casos de F4.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v2"))
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f4/gradcam"))
    parser.add_argument("--max", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        args.checkpoint,
        args.data,
        args.out,
        oof_path=args.oof,
        cache_dir=args.cache_dir,
        max_cases=args.max,
    )
    print(json.dumps(result["cases"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
