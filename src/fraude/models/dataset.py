"""Dataset de páginas para F4.

Cada muestra devuelve ``(imagen, etiqueta, índice)``. La etiqueta es 1 para
manipulado y 0 para auténtico. El índice permite recuperar los metadatos
(``sample_id``, ataque, persona) sin romper el ``collate_fn`` por defecto.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fraude.models.cache import cache_path
from fraude.synth.dataset import DocumentRecord

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int, *, train: bool) -> transforms.Compose:
    """Transformaciones leves: no invierten texto ni recortan la página."""
    steps: list = []
    if train:
        steps.extend(
            [
                transforms.RandomRotation(1.5, fill=255),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
            ]
        )
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


class PageDataset(Dataset):
    """Carga páginas desde la caché cuadrada y aplica recorte central o aleatorio."""

    def __init__(
        self,
        records: list[DocumentRecord],
        cache_dir: Path,
        *,
        image_size: int = 224,
        train: bool = False,
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size debe ser al menos 32")
        self.records = list(records)
        self.cache_dir = Path(cache_dir)
        self.image_size = image_size
        self.train = train
        self.transform = build_transform(image_size, train=train)
        self._crop = (
            transforms.RandomCrop(image_size) if train else transforms.CenterCrop(image_size)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        path = cache_path(self.cache_dir, record.sample_id)
        if not path.exists():
            raise FileNotFoundError(f"Falta la caché de {record.sample_id}: {path}")
        with Image.open(path) as image:
            page = image.convert("RGB")
        if page.width != page.height:
            raise ValueError(f"La caché de {record.sample_id} no es cuadrada: {page.size}")
        if page.width < self.image_size:
            page = page.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        elif page.width > self.image_size:
            page = self._crop(page)
        tensor = self.transform(page)
        label = 0.0 if record.is_authentic else 1.0
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(index)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Deshace la normalización ImageNet para visualizar la página."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
