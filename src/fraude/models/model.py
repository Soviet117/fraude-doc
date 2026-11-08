"""Modelo visual de página para F4 (EfficientNet-B0 vía timm)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_model(
    model_name: str = "efficientnet_b0",
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Crea el clasificador binario con una sola salida logit."""
    import timm

    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=dropout,
        )
    except Exception as exc:  # pragma: no cover - depende de la red y la caché
        raise RuntimeError(
            f"No se pudo crear {model_name} con pretrained={pretrained}: {exc}. "
            "Usa --no-pretrained si no hay acceso a los pesos."
        ) from exc
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_metadata(model: nn.Module, *, model_name: str, pretrained: bool) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "pretrained": pretrained,
        "parameters": count_parameters(model),
        "torch": torch.__version__,
    }


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Devuelve la última convolución, usada como capa objetivo de Grad-CAM."""
    last: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("El modelo no tiene capas convolucionales")
    return last
