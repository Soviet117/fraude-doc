"""U-Net pequeña para localización de manipulaciones (F5).

La red se entrena desde cero: el corpus sintético es pequeño y el objetivo es
tener una línea base reproducible de segmentación, no un modelo de producción.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net de cuatro niveles con salida de un logit por píxel."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.encoder1 = _block(3, channels[0])
        self.encoder2 = _block(channels[0], channels[1])
        self.encoder3 = _block(channels[1], channels[2])
        self.encoder4 = _block(channels[2], channels[3])
        self.bottleneck = _block(channels[3], channels[3] * 2)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(channels[3] * 2, channels[3], 2, stride=2)
        self.decoder4 = _block(channels[3] * 2, channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.decoder3 = _block(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.decoder2 = _block(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.decoder1 = _block(channels[0] * 2, channels[0])
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(images)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.decoder4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.head(dec1)
