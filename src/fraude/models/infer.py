"""Inferencia F4 para una sola imagen (demo y pruebas manuales)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as transform_functional

from fraude.models.cache import letterbox_square
from fraude.models.dataset import build_transform
from fraude.models.gradcam import GradCAM, overlay
from fraude.models.model import build_model, last_conv_layer


def load_checkpoint(
    checkpoint: Path,
    *,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(payload.get("config", {}))
    model = build_model(str(config.get("model_name", "efficientnet_b0")), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def preprocess(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Letterbox a 256, recorte central y normalización ImageNet."""
    canvas = max(256, image_size)
    square = letterbox_square(image, canvas)
    if canvas > image_size:
        square = transform_functional.center_crop(square, image_size)
    return build_transform(image_size, train=False)(square).unsqueeze(0)


def predict_tensor(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(tensor).squeeze(1)
        return float(torch.sigmoid(logits).item())


def predict_image(model: torch.nn.Module, image: Image.Image, *, image_size: int = 224) -> float:
    return predict_tensor(model, preprocess(image, image_size))


def explain_image(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    image_size: int = 224,
) -> tuple[Image.Image, float, np.ndarray]:
    """Devuelve el overlay Grad-CAM, el score y el mapa normalizado."""
    tensor = preprocess(image, image_size)
    cam = GradCAM(model, last_conv_layer(model))
    heat, score = cam(tensor)
    canvas = max(256, image_size)
    base = letterbox_square(image, canvas)
    if canvas > image_size:
        base = transform_functional.center_crop(base, image_size)
    return overlay(base, heat), score, heat
