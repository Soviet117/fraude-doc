"""Dataset de segmentación para F5.

Cada muestra devuelve ``(imagen, máscara, índice)``. Las imágenes auténticas no
tienen máscara en el manifiesto: se usa una máscara vacía. Las máscaras se
redimensionan con vecino más cercano para no inventar píxeles positivos.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fraude.models.dataset import IMAGENET_MEAN, IMAGENET_STD
from fraude.synth.dataset import DocumentRecord


class SegmentationDataset(Dataset):
    """Páginas y máscaras binarias a una resolución cuadrada fija."""

    def __init__(
        self,
        records: list[DocumentRecord],
        data_dir: Path,
        *,
        image_size: int = 256,
        train: bool = False,
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size debe ser al menos 32")
        self.records = list(records)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.train = train
        self._color = (
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04)
            if train
            else None
        )
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = self.data_dir / record.image_path
        with Image.open(image_path) as image:
            page = image.convert("RGB")
        if self._color is not None:
            page = self._color(page)
        page = page.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        mask = Image.new("L", (self.image_size, self.image_size), 0)
        if record.mask_paths:
            with Image.open(self.data_dir / record.mask_paths[0]) as mask_image:
                mask = mask_image.convert("L").resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.NEAREST,
                )

        image_tensor = self._normalize(self._to_tensor(page))
        mask_tensor = (self._to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor, torch.tensor(index)

    def mask_for(self, index: int) -> torch.Tensor:
        """Devuelve solo la máscara binaria, útil para evaluación."""
        record = self.records[index]
        mask = Image.new("L", (self.image_size, self.image_size), 0)
        if record.mask_paths:
            with Image.open(self.data_dir / record.mask_paths[0]) as mask_image:
                mask = mask_image.convert("L").resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.NEAREST,
                )
        return (self._to_tensor(mask) > 0.5).float()
