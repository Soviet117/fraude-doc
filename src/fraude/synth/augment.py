"""Aumentación de captura para F2.4: escaneo, foto, JPEG, rotación y sombra.

La transformación geométrica se aplica de forma conjunta a la imagen, a la
máscara y a las cajas de campos/regiones. Si se rotara solo la imagen, las
etiquetas de localización quedarían desalineadas y el modelo aprendería ruido.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from fraude.ingest.hashing import sha256_file
from fraude.synth.attacks import load_manifest
from fraude.synth.dataset import DocumentRecord

Box = tuple[int, int, int, int]
BoxTransform = Callable[[Box], Box]

DEFAULT_VARIANTS = ("scan", "photo", "jpeg", "rotate", "shadow")
WHITE = (255, 255, 255)
PHOTO_BACKGROUND = (242, 242, 242)


def augment_dataset(
    records: list[DocumentRecord],
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    seed: int = 42,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> list[DocumentRecord]:
    """Genera una copia aumentada por variante y escribe el manifiesto."""
    if variants and any(variant not in DEFAULT_VARIANTS for variant in variants):
        raise ValueError(f"Variantes no soportadas: {variants}")
    source_dir = Path(source_dir) if source_dir is not None else Path(out_dir)
    out_dir = Path(out_dir)

    augmented: list[DocumentRecord] = []
    for record in records:
        for variant in variants:
            augmented.append(
                apply_augmentation(
                    record,
                    variant,
                    out_dir,
                    source_dir=source_dir,
                    seed=seed,
                )
            )

    write_augmented_manifest(augmented, out_dir, seed=seed, variants=variants)
    return augmented


def apply_augmentation(
    record: DocumentRecord,
    variant: str,
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    seed: int = 42,
) -> DocumentRecord:
    """Aplica una variante de captura a un registro y conserva sus etiquetas."""
    if variant not in DEFAULT_VARIANTS:
        raise ValueError(f"Variante no soportada: {variant}")
    source_dir = Path(source_dir) if source_dir is not None else Path(out_dir)
    out_dir = Path(out_dir)
    rng = random.Random(f"{seed}:{record.seed}:{record.sample_id}:{variant}")

    image = Image.open(source_dir / record.image_path).convert("RGB")
    masks = [Image.open(source_dir / mask_path).convert("L") for mask_path in record.mask_paths]
    box_transform: BoxTransform = _identity_box
    capture_type = record.capture_type

    if variant == "scan":
        image = _desaturate(image, 0.25)
        angle = rng.uniform(-0.9, 0.9)
        image, masks = _rotate(image, masks, angle)
        box_transform = _rotation_transform(angle, image.size)
        image = _add_noise(image, rng, sigma=rng.uniform(1.5, 3.5))
        capture_type = "scan"
    elif variant == "photo":
        coefficients = _photo_coefficients(image.size, rng)
        image, masks = _perspective(image, masks, coefficients)
        box_transform = _perspective_transform(coefficients, image.size)
        image = _add_shadow(image, rng)
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
        image = _add_noise(image, rng, sigma=rng.uniform(3.0, 7.0))
        image = _jpeg(image, quality=rng.randint(55, 80))
        capture_type = "photo"
    elif variant == "jpeg":
        image = _add_noise(image, rng, sigma=rng.uniform(0.5, 2.0))
        image = _jpeg(image, quality=rng.randint(35, 70))
    elif variant == "rotate":
        angle = rng.uniform(-4.0, 4.0)
        image, masks = _rotate(image, masks, angle)
        box_transform = _rotation_transform(angle, image.size)
    elif variant == "shadow":
        image = _add_shadow(image, rng)
    else:  # pragma: no cover - la validación de arriba lo hace inalcanzable
        raise ValueError(f"Variante no soportada: {variant}")

    sample_id = f"{record.sample_id}_{variant}"
    image_rel = f"images/{sample_id}.png"
    image_path = out_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="PNG", dpi=image.info.get("dpi", (150, 150)))

    new_mask_paths: list[str] = []
    mask_map: dict[str, str] = {}
    for index, mask in enumerate(masks):
        suffix = f"_{index}" if len(masks) > 1 else ""
        mask_rel = f"masks/{sample_id}{suffix}.png"
        mask_path = out_dir / mask_rel
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path, format="PNG")
        new_mask_paths.append(mask_rel)
        mask_map[record.mask_paths[index]] = mask_rel

    field_boxes = {
        key: _clip_box(box_transform(box), image.size) for key, box in record.field_boxes.items()
    }
    regions = tuple(
        replace(
            region,
            bbox=_clip_box(box_transform(region.bbox), image.size),
            mask_path=mask_map.get(region.mask_path) if region.mask_path else None,
        )
        for region in record.regions
    )

    return DocumentRecord(
        sample_id=sample_id,
        document_id=sha256_file(image_path),
        source_id=record.source_id,
        document_type=record.document_type,
        template_version=record.template_version,
        variant=record.variant,
        parent_id=record.parent_id,
        is_authentic=record.is_authentic,
        attack_types=record.attack_types,
        primary_attack=record.primary_attack,
        pdf_path=None,
        image_path=image_rel,
        mask_paths=tuple(new_mask_paths),
        fields=dict(record.fields),
        field_boxes=field_boxes,
        page_size=image.size,
        regions=regions,
        semantic_flags=record.semantic_flags,
        capture_type=capture_type,
        seed=record.seed,
        augmentations=(*record.augmentations, variant),
        origin_id=record.origin_id or record.document_id,
    )


def _identity_box(box: Box) -> Box:
    return box


def _rotation_transform(angle: float, size: tuple[int, int]) -> BoxTransform:
    width, height = size
    center_x, center_y = width / 2.0, height / 2.0
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    def transform(box: Box) -> Box:
        x, y, box_width, box_height = box
        corners = (
            (x, y),
            (x + box_width, y),
            (x + box_width, y + box_height),
            (x, y + box_height),
        )
        rotated = []
        for corner_x, corner_y in corners:
            dx = corner_x - center_x
            dy = corner_y - center_y
            rotated.append(
                (
                    center_x + dx * cos_theta + dy * sin_theta,
                    center_y - dx * sin_theta + dy * cos_theta,
                )
            )
        return _bounds(rotated, size)

    return transform


def _perspective_transform(coefficients: tuple[float, ...], size: tuple[int, int]) -> BoxTransform:
    forward = _forward_homography(coefficients)

    def transform(box: Box) -> Box:
        x, y, box_width, box_height = box
        corners = (
            (x, y),
            (x + box_width, y),
            (x + box_width, y + box_height),
            (x, y + box_height),
        )
        mapped = [_apply_homography(forward, corner_x, corner_y) for corner_x, corner_y in corners]
        return _bounds(mapped, size)

    return transform


def _bounds(points: list[tuple[float, float]], size: tuple[int, int]) -> Box:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width, height = size
    left = max(0, min(math.floor(min(xs)), width - 1))
    top = max(0, min(math.floor(min(ys)), height - 1))
    right = max(left + 1, min(math.ceil(max(xs)), width))
    bottom = max(top + 1, min(math.ceil(max(ys)), height))
    return left, top, right - left, bottom - top


def _clip_box(box: Box, size: tuple[int, int]) -> Box:
    return _bounds(
        [
            (box[0], box[1]),
            (box[0] + box[2], box[1] + box[3]),
        ],
        size,
    )


def _rotate(
    image: Image.Image,
    masks: list[Image.Image],
    angle: float,
) -> tuple[Image.Image, list[Image.Image]]:
    rotated = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=WHITE,
    )
    rotated_masks = [
        mask.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )
        for mask in masks
    ]
    return rotated, rotated_masks


def _perspective(
    image: Image.Image,
    masks: list[Image.Image],
    coefficients: tuple[float, ...],
) -> tuple[Image.Image, list[Image.Image]]:
    warped = image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=PHOTO_BACKGROUND,
    )
    warped_masks = [
        mask.transform(
            mask.size,
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        for mask in masks
    ]
    return warped, warped_masks


def _photo_coefficients(size: tuple[int, int], rng: random.Random) -> tuple[float, ...]:
    width, height = size
    shift_x = width * 0.018
    shift_y = height * 0.015
    source_quad = (
        (rng.uniform(-shift_x, shift_x), rng.uniform(-shift_y, shift_y)),
        (width + rng.uniform(-shift_x, shift_x), rng.uniform(-shift_y, shift_y)),
        (width + rng.uniform(-shift_x, shift_x), height + rng.uniform(-shift_y, shift_y)),
        (rng.uniform(-shift_x, shift_x), height + rng.uniform(-shift_y, shift_y)),
    )
    return _perspective_coefficients(source_quad, size)


def _perspective_coefficients(
    source_quad: tuple[tuple[float, float], ...],
    size: tuple[int, int],
) -> tuple[float, ...]:
    """Coeficientes de PIL que mapean cada píxel de salida a la imagen fuente."""
    width, height = size
    destination = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(destination, source_quad, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    solution = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(values))
    return tuple(float(value) for value in solution)


def _forward_homography(coefficients: tuple[float, ...]) -> np.ndarray:
    a, b, c, d, e, f, g, h = coefficients
    inverse = np.asarray([[a, b, c], [d, e, f], [g, h, 1.0]], dtype=np.float64)
    return np.linalg.inv(inverse)


def _apply_homography(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    vector = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
    if abs(vector[2]) < 1e-12:
        return x, y
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


def _desaturate(image: Image.Image, alpha: float) -> Image.Image:
    gray = image.convert("L").convert("RGB")
    return Image.blend(image, gray, alpha)


def _add_noise(image: Image.Image, rng: random.Random, *, sigma: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    generator = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noise = generator.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))


def _add_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    width, height = image.size
    dark = rng.uniform(0.72, 0.86)
    if rng.random() < 0.5:
        ramp = np.linspace(dark, 1.0, width, dtype=np.float32)[None, :, None]
    else:
        ramp = np.linspace(1.0, dark, height, dtype=np.float32)[:, None, None]
    return Image.fromarray(np.clip(array * ramp, 0, 255).astype(np.uint8))


def _jpeg(image: Image.Image, *, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def write_augmented_manifest(
    records: list[DocumentRecord],
    out_dir: Path,
    *,
    seed: int = 42,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "augmented_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.synth.augment",
        "count": len(records),
        "seed": seed,
        "variants": list(variants),
        "by_augmentation": _count(
            record.augmentations[-1] if record.augmentations else "none" for record in records
        ),
        "by_capture_type": _count(record.capture_type for record in records),
        "by_variant": _count(record.variant for record in records),
        "by_attack": _count(record.primary_attack for record in records),
    }
    (out_dir / "augmented_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera aumentación de captura F2.4.")
    parser.add_argument("--data", type=Path, required=True, help="Directorio del dataset.")
    parser.add_argument("--out", type=Path, default=None, help="Directorio de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Variantes separadas por coma.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out is not None else data_dir
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())

    records = load_manifest(data_dir / "manifest.jsonl")
    attacks_path = data_dir / "attacks_manifest.jsonl"
    if attacks_path.exists():
        records.extend(load_manifest(attacks_path))

    augmented = augment_dataset(
        records,
        out_dir,
        source_dir=data_dir,
        seed=args.seed,
        variants=variants,
    )
    print(f"Generadas {len(augmented)} muestras aumentadas en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
