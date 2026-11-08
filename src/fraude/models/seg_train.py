"""Entrenamiento F5: U-Net para localizar manipulaciones en la página.

Uso mínimo (humo en CPU):
    uv run python -m fraude.models.seg_train --data data/synthetic/v2_smoke \
        --out artifacts/models/f5_smoke --folds 2 --epochs 1 --image-size 64 \
        --batch-size 2

Uso completo:
    uv run python -m fraude.models.seg_train --data data/synthetic/v2 \
        --out artifacts/models/f5 --folds 5 --epochs 12 --image-size 256
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from fraude.models.seg_dataset import SegmentationDataset
from fraude.models.seg_metrics import metrics_from_counts, pixel_confusion
from fraude.models.unet import UNet
from fraude.synth.dataset import DocumentRecord
from fraude.synth.folds import assign_grouped_folds, load_folds
from fraude.synth.validate import load_corpus

LOGGER = logging.getLogger("fraude.models.seg_train")


@dataclass
class SegTrainConfig:
    data_dir: Path = Path("data/synthetic/v2")
    out_dir: Path = Path("artifacts/models/f5")
    folds: int = 5
    folds_list: tuple[int, ...] | None = None
    epochs: int = 12
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 256
    base_channels: int = 16
    seed: int = 42
    device: str = "auto"
    patience: int = 4
    pos_weight: float = 20.0
    positive_oversample: float = 3.0
    limit_groups: int | None = None
    include_attacks: tuple[str, ...] | None = ("T1", "T2", "T3", "T4", "T5", "T6")
    n_boot: int = 1000
    evaluate: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def select_records(
    records: list[DocumentRecord],
    *,
    include_attacks: tuple[str, ...] | None = None,
    limit_groups: int | None = None,
    seed: int = 42,
) -> list[DocumentRecord]:
    """Conserva auténticos y ataques con máscara local (T1-T6)."""
    selected = list(records)
    if include_attacks is not None:
        allowed = set(include_attacks)
        selected = [
            record for record in selected if record.is_authentic or record.primary_attack in allowed
        ]
    if not selected:
        raise ValueError("El filtro dejó el corpus vacío")
    if limit_groups is None:
        return selected
    groups: dict[str, list[DocumentRecord]] = {}
    for record in selected:
        groups.setdefault(record.source_id, []).append(record)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    keep = set(names[:limit_groups])
    return [record for record in selected if record.source_id in keep]


def resolve_folds(records: list[DocumentRecord], cfg: SegTrainConfig) -> dict[str, int]:
    manifest = Path(cfg.data_dir) / "folds_manifest.jsonl"
    if cfg.limit_groups is None and manifest.exists():
        folds = load_folds(manifest)
        if all(record.sample_id in folds for record in records):
            return folds
    assignment = assign_grouped_folds(records, k=cfg.folds, seed=cfg.seed)
    return {record.sample_id: assignment[record.source_id] for record in records}


def _make_loader(
    records: list[DocumentRecord],
    cfg: SegTrainConfig,
    *,
    train: bool,
) -> DataLoader:
    dataset = SegmentationDataset(
        records,
        Path(cfg.data_dir),
        image_size=cfg.image_size,
        train=train,
    )
    if train:
        weights = [
            cfg.positive_oversample if not record.is_authentic else 1.0 for record in records
        ]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(records),
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, *, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(2, 3))
    union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    pos_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    tp = fp = fn = 0
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type, enabled=amp_enabled):
                logits = model(images)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    masks,
                    pos_weight=torch.tensor(pos_weight, device=device),
                )
                loss = bce + _dice_loss(logits, masks)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * images.size(0)
        with torch.no_grad():
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()
        batch_tp, batch_fp, batch_fn = pixel_confusion(probabilities, targets, threshold=0.5)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    metrics = metrics_from_counts(tp, fp, fn)
    return total_loss / max(1, len(loader.dataset)), metrics


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_probability_map(probabilities: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(probabilities * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _predict_maps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> list[tuple[DocumentRecord, np.ndarray, Path]]:
    model.eval()
    records = loader.dataset.records
    results: list[tuple[DocumentRecord, np.ndarray, Path]] = []
    offset = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(images)).detach().cpu().numpy()[:, 0]
            for probability in probabilities:
                record = records[offset]
                map_path = out_dir / "oof_maps" / f"{record.sample_id}.png"
                _save_probability_map(probability, map_path)
                results.append((record, probability, map_path))
                offset += 1
    return results


def train_fold(
    fold: int,
    records: list[DocumentRecord],
    folds: dict[str, int],
    cfg: SegTrainConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_records = [record for record in records if folds[record.sample_id] != fold]
    val_records = [record for record in records if folds[record.sample_id] == fold]
    if not train_records or not val_records:
        raise ValueError(f"Fold {fold} sin datos de entrenamiento o validación")

    train_loader = _make_loader(train_records, cfg, train=True)
    val_loader = _make_loader(val_records, cfg, train=False)

    set_seed(cfg.seed + fold)
    model = UNet(base_channels=cfg.base_channels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    stale = 0

    for epoch in range(cfg.epochs):
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            pos_weight=cfg.pos_weight,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            pos_weight=cfg.pos_weight,
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_pixel_f1": train_metrics["f1"],
            "val_pixel_f1": val_metrics["f1"],
            "val_pixel_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(row)
        LOGGER.info(
            "fold %d epoch %d/%d train_loss=%.4f val_loss=%.4f val_pixel_f1=%.4f",
            fold,
            epoch + 1,
            cfg.epochs,
            train_loss,
            val_loss,
            val_metrics["f1"],
        )

        if val_metrics["f1"] > best_score + 1e-4:
            best_score = val_metrics["f1"]
            best_epoch = epoch + 1
            best_state = _snapshot_state(model)
            best_metrics = val_metrics
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                LOGGER.info("fold %d: early stopping en la época %d", fold, epoch + 1)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint_path = Path(cfg.out_dir) / f"fold_{fold}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "fold": fold,
            "epoch": best_epoch,
            "config": {
                **asdict(cfg),
                "data_dir": str(cfg.data_dir),
                "out_dir": str(cfg.out_dir),
            },
            "metrics": best_metrics,
        },
        checkpoint_path,
    )

    oof: list[dict[str, Any]] = []
    for record, probability, map_path in _predict_maps(
        model,
        val_loader,
        device,
        Path(cfg.out_dir),
    ):
        mask = _load_mask(record, cfg.data_dir, cfg.image_size)
        tp, fp, fn = pixel_confusion(probability, mask, threshold=0.5)
        metrics = metrics_from_counts(tp, fp, fn)
        oof.append(
            {
                "sample_id": record.sample_id,
                "source_id": record.source_id,
                "fold": fold,
                "label": 0 if record.is_authentic else 1,
                "attack": record.primary_attack,
                "document_type": record.document_type,
                "augmentations": list(record.augmentations),
                "map_path": str(map_path.relative_to(Path(cfg.out_dir))),
                "mask_path": record.mask_paths[0] if record.mask_paths else None,
                "n_positive": int(mask.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pixel_f1": metrics["f1"],
                "pixel_iou": metrics["iou"],
                "max_prob": round(float(probability.max()), 4),
                "mean_prob": round(float(probability.mean()), 6),
            }
        )

    result = {
        "fold": fold,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "positives_train": sum(1 for record in train_records if not record.is_authentic),
        "best_epoch": best_epoch,
        "best": best_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    return result, oof


def _load_mask(record: DocumentRecord, data_dir: Path, image_size: int) -> np.ndarray:
    mask = Image.new("L", (image_size, image_size), 0)
    if record.mask_paths:
        with Image.open(Path(data_dir) / record.mask_paths[0]) as mask_image:
            mask = mask_image.convert("L").resize(
                (image_size, image_size),
                Image.Resampling.NEAREST,
            )
    return (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)


def train_cv(cfg: SegTrainConfig) -> dict[str, Any]:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_corpus(data_dir)
    if not records:
        raise ValueError(f"No hay manifiestos en {data_dir}")
    records = select_records(
        records,
        include_attacks=cfg.include_attacks,
        limit_groups=cfg.limit_groups,
        seed=cfg.seed,
    )
    folds = resolve_folds(records, cfg)
    (out_dir / "folds.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    device = resolve_device(cfg.device)
    LOGGER.info("Dispositivo: %s; %d imágenes", device, len(records))
    fold_ids = list(cfg.folds_list) if cfg.folds_list is not None else list(range(cfg.folds))
    results: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    for fold in fold_ids:
        LOGGER.info("Entrenando fold %d", fold)
        result, fold_oof = train_fold(fold, records, folds, cfg, device)
        results.append(result)
        oof.extend(fold_oof)

    oof_path = out_dir / "oof.jsonl"
    with oof_path.open("w", encoding="utf-8") as handle:
        for row in sorted(oof, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload: dict[str, Any] = {
        "config": {**asdict(cfg), "data_dir": str(cfg.data_dir), "out_dir": str(cfg.out_dir)},
        "model": {
            "model_name": "unet",
            "base_channels": cfg.base_channels,
            "parameters": sum(
                parameter.numel() for parameter in UNet(cfg.base_channels).parameters()
            ),
        },
        "device": str(device),
        "folds": results,
        "oof": str(oof_path),
    }
    if cfg.evaluate and oof:
        from fraude.models.seg_evaluate import evaluate_oof, write_markdown

        payload["pooled"] = evaluate_oof(
            oof_path,
            data_dir=data_dir,
            image_size=cfg.image_size,
            n_boot=cfg.n_boot,
            seed=cfg.seed,
        )
        write_markdown(payload["pooled"], out_dir / "eval.md")

    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena F5 (U-Net de localización).")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v2"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f5"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--folds-list", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--positive-oversample", type=float, default=3.0)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--include-attacks", default="T1,T2,T3,T4,T5,T6")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folds_list = (
        tuple(int(item) for item in args.folds_list.split(",") if item.strip())
        if args.folds_list
        else None
    )
    include_attacks = (
        None
        if args.include_attacks.strip().lower() in {"all", "*"}
        else tuple(item.strip() for item in args.include_attacks.split(",") if item.strip())
    )
    cfg = SegTrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        folds=args.folds,
        folds_list=folds_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        base_channels=args.base_channels,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        pos_weight=args.pos_weight,
        positive_oversample=args.positive_oversample,
        limit_groups=args.limit_groups,
        include_attacks=include_attacks,
        n_boot=args.n_boot,
        evaluate=not args.no_eval,
    )
    payload = train_cv(cfg)
    print(json.dumps(payload.get("pooled", {}).get("fixed_0_5", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
