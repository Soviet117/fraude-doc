"""Entrenamiento F4 con validación cruzada agrupada por persona.

Uso mínimo (humo en CPU):
    uv run python -m fraude.models.train --data data/synthetic/v2_smoke \
        --out artifacts/models/f4_smoke --folds 2 --epochs 1 --image-size 128 \
        --no-pretrained --batch-size 2

Uso completo (GPU o CPU):
    uv run python -m fraude.models.train --data data/synthetic/v2 \
        --out artifacts/models/f4 --folds 5 --epochs 15 --batch-size 16
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
from torch import nn
from torch.utils.data import DataLoader

from fraude.eval.metrics import binary_metrics, ranking_metrics
from fraude.models.cache import DEFAULT_CACHE_SIZE, build_cache
from fraude.models.dataset import PageDataset
from fraude.models.evaluate import evaluate_oof, write_markdown
from fraude.models.model import build_model, model_metadata
from fraude.synth.dataset import DocumentRecord
from fraude.synth.folds import assign_grouped_folds, load_folds
from fraude.synth.validate import load_corpus

LOGGER = logging.getLogger("fraude.models.train")


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/synthetic/v2")
    out_dir: Path = Path("artifacts/models/f4")
    cache_dir: Path | None = None
    folds: int = 5
    folds_list: tuple[int, ...] | None = None
    epochs: int = 15
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    cache_size: int = DEFAULT_CACHE_SIZE
    seed: int = 42
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    patience: int = 4
    exclude_split: str | None = None
    include_attacks: tuple[str, ...] | None = ("T1", "T2", "T3", "T4", "T5", "T6")
    limit_groups: int | None = None
    n_boot: int = 2000
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
    exclude_split: str | None = None,
    include_attacks: tuple[str, ...] | None = None,
    limit_groups: int | None = None,
    seed: int = 42,
) -> list[DocumentRecord]:
    """Filtra particiones y familias de ataque, y limita el número de personas.

    El clasificador visual no debe entrenarse con T7/T8: T7 reutiliza un
    documento auténtico de otra persona y T8 cambia contenido semántico, así
    que no dejan una huella visual local. Esas familias se cubren con reglas y
    registro (F3) y con la rama semántica (F6).
    """
    selected = (
        list(records)
        if exclude_split is None
        else [record for record in records if record.split != exclude_split]
    )
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
    chosen: list[str] = []
    for name in names:
        chosen.append(name)
        if len(chosen) >= limit_groups:
            break
    keep = set(chosen)
    return [record for record in selected if record.source_id in keep]


def resolve_folds(
    records: list[DocumentRecord],
    cfg: TrainConfig,
) -> dict[str, int]:
    manifest = Path(cfg.data_dir) / "folds_manifest.jsonl"
    use_manifest = cfg.limit_groups is None and cfg.exclude_split is None
    if use_manifest and manifest.exists():
        folds = load_folds(manifest)
        if all(record.sample_id in folds for record in records):
            return folds
    folds = assign_grouped_folds(records, k=cfg.folds, seed=cfg.seed)
    return {record.sample_id: folds[record.source_id] for record in records}


def _make_loader(
    records: list[DocumentRecord],
    cache_dir: Path,
    cfg: TrainConfig,
    *,
    train: bool,
) -> DataLoader:
    dataset = PageDataset(
        records,
        cache_dir,
        image_size=cfg.image_size,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
) -> tuple[float, list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    scores: list[float] = []
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type, enabled=amp_enabled):
                logits = model(images).squeeze(1)
                loss = criterion(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * images.size(0)
        labels.extend(int(value) for value in targets.detach().cpu().tolist())
        scores.extend(float(value) for value in torch.sigmoid(logits.detach()).cpu().tolist())

    return total_loss / max(1, len(labels)), labels, scores


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copia profunda de los pesos; en CPU ``.cpu()`` no copia y aliasa el estado vivo."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_fold(
    fold: int,
    records: list[DocumentRecord],
    folds: dict[str, int],
    cache_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_records = [record for record in records if folds[record.sample_id] != fold]
    val_records = [record for record in records if folds[record.sample_id] == fold]
    if not train_records or not val_records:
        raise ValueError(f"Fold {fold} sin datos de entrenamiento o validación")

    train_loader = _make_loader(train_records, cache_dir, cfg, train=True)
    val_loader = _make_loader(val_records, cache_dir, cfg, train=False)

    set_seed(cfg.seed + fold)
    model = build_model(cfg.model_name, pretrained=cfg.pretrained).to(device)
    positives = sum(1 for record in train_records if not record.is_authentic)
    negatives = len(train_records) - positives
    pos_weight = torch.tensor([negatives / max(1, positives)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    amp_enabled = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    stale = 0

    for epoch in range(cfg.epochs):
        train_loss, train_labels, train_scores = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_labels, val_scores = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_ranking = ranking_metrics(val_labels, val_scores)
        val_binary = binary_metrics(val_labels, [1 if score >= 0.5 else 0 for score in val_scores])
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_pr_auc": ranking_metrics(train_labels, train_scores)["pr_auc"],
            "val_pr_auc": val_ranking["pr_auc"],
            "val_roc_auc": val_ranking["roc_auc"],
            "val_f1": val_binary["f1"],
            "val_recall": val_binary["recall"],
            "val_fpr": val_binary["fpr"],
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(epoch_metrics)
        LOGGER.info(
            "fold %d epoch %d/%d train_loss=%.4f val_loss=%.4f val_pr_auc=%s val_f1=%.3f",
            fold,
            epoch + 1,
            cfg.epochs,
            train_loss,
            val_loss,
            epoch_metrics["val_pr_auc"],
            float(epoch_metrics["val_f1"] or 0.0),
        )

        score = float(val_ranking["pr_auc"] or 0.0)
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch + 1
            best_state = _snapshot_state(model)
            best_metrics = epoch_metrics
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
            "config": {**asdict(cfg), "data_dir": str(cfg.data_dir), "out_dir": str(cfg.out_dir)},
            "metrics": best_metrics,
        },
        checkpoint_path,
    )

    _, val_labels, val_scores = _run_epoch(model, val_loader, criterion, device)
    oof = [
        {
            "sample_id": record.sample_id,
            "source_id": record.source_id,
            "fold": fold,
            "label": 0 if record.is_authentic else 1,
            "score": float(score),
            "attack": record.primary_attack,
            "document_type": record.document_type,
            "augmentations": list(record.augmentations),
        }
        for record, score in zip(val_records, val_scores, strict=True)
    ]
    result = {
        "fold": fold,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "positives_train": positives,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    return result, oof


def train_cv(cfg: TrainConfig) -> dict[str, Any]:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else out_dir / "cache"

    records = load_corpus(data_dir)
    if not records:
        raise ValueError(f"No hay manifiestos en {data_dir}")
    records = select_records(
        records,
        exclude_split=cfg.exclude_split,
        include_attacks=cfg.include_attacks,
        limit_groups=cfg.limit_groups,
        seed=cfg.seed,
    )
    folds = resolve_folds(records, cfg)
    (out_dir / "folds.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Construyendo caché de %d imágenes en %s", len(records), cache_dir)
    build_cache(records, data_dir, cache_dir, size=cfg.cache_size)

    device = resolve_device(cfg.device)
    LOGGER.info("Dispositivo: %s", device)
    fold_ids = list(cfg.folds_list) if cfg.folds_list is not None else list(range(cfg.folds))
    results: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []

    for fold in fold_ids:
        LOGGER.info("Entrenando fold %d", fold)
        result, fold_oof = train_fold(fold, records, folds, cache_dir, cfg, device)
        results.append(result)
        oof.extend(fold_oof)

    oof_path = out_dir / "oof.jsonl"
    with oof_path.open("w", encoding="utf-8") as handle:
        for row in sorted(oof, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload: dict[str, Any] = {
        "config": {**asdict(cfg), "data_dir": str(cfg.data_dir), "out_dir": str(cfg.out_dir)},
        "model": model_metadata(
            build_model(cfg.model_name, pretrained=False),
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
        ),
        "device": str(device),
        "folds": results,
        "oof": str(oof_path),
    }
    if cfg.evaluate and oof:
        payload["pooled"] = evaluate_oof(oof, seed=cfg.seed, n_boot=cfg.n_boot)
        write_markdown(payload["pooled"], out_dir / "eval.md")

    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena F4 con folds agrupados.")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/v2"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f4"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--folds-list", default=None, help="Folds a entrenar, p. ej. 0,1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--exclude-split", default=None)
    parser.add_argument(
        "--include-attacks",
        default="T1,T2,T3,T4,T5,T6",
        help="Familias visuales a entrenar; 'all' incluye también T7/T8.",
    )
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
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
    cfg = TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        cache_dir=args.cache_dir,
        folds=args.folds,
        folds_list=folds_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        cache_size=args.cache_size,
        seed=args.seed,
        model_name=args.model,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        patience=args.patience,
        exclude_split=args.exclude_split,
        include_attacks=include_attacks,
        limit_groups=args.limit_groups,
        n_boot=args.n_boot,
        evaluate=not args.no_eval,
    )
    payload = train_cv(cfg)
    print(json.dumps(payload.get("pooled", {}).get("aggregate", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
