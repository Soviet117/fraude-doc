"""Inferencia F5 para la demo: máscara de probabilidad y superposición."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fraude.models.dataset import IMAGENET_MEAN, IMAGENET_STD
from fraude.models.unet import UNet


def load_checkpoint(path: Path) -> tuple[UNet, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = UNet(base_channels=int(config.get("base_channels", 16)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def predict_mask(
    model: UNet,
    image: Image.Image,
    *,
    image_size: int = 256,
) -> np.ndarray:
    """Devuelve un mapa de probabilidad ``image_size x image_size`` en [0, 1]."""
    page = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(transforms.ToTensor()(page))
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probability


def overlay_prediction(
    image: Image.Image,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    alpha: float = 0.45,
) -> Image.Image:
    """Superpone en rojo la región con probabilidad ≥ umbral."""
    base = image.convert("RGB").resize(
        (probability.shape[1], probability.shape[0]),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(base, dtype=np.float32).copy()
    mask = probability >= threshold
    array[mask, 0] = np.clip(array[mask, 0] + 255 * alpha, 0, 255)
    array[mask, 1] = np.clip(array[mask, 1] * (1 - alpha), 0, 255)
    array[mask, 2] = np.clip(array[mask, 2] * (1 - alpha), 0, 255)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def localization_summary(probability: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    return {
        "max_prob": round(float(probability.max()), 4),
        "mean_prob": round(float(probability.mean()), 6),
        "area_fraction": round(float((probability >= threshold).mean()), 6),
    }
