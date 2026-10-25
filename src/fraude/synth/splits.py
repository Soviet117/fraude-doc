"""Particiones agrupadas y sin fuga para F2.5.

El grupo es la persona (``source_id``): todas sus muestras auténticas,
manipuladas y aumentadas caen en la misma partición. Además se reserva una
familia completa de ataque para ``unseen_attack`` y se verifica que ningún
``source_id`` ni ningún documento original cruce dos particiones.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from fraude.synth.attacks import load_manifest
from fraude.synth.dataset import ATTACK_TYPES, DocumentRecord

SPLITS = ("train", "val", "calibration", "test", "unseen_attack")
RATIO_SPLITS = ("train", "val", "calibration", "test")
EVAL_SPLITS = ("test", "val", "calibration")
DEFAULT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "calibration": 0.05,
    "test": 0.10,
}
REAL_SPLIT = "real"
REAL_PREFIX = "real_"


def assign_splits(
    records: list[DocumentRecord],
    *,
    seed: int = 42,
    holdout_attack: str | None = "T5",
    ratios: dict[str, float] | None = None,
) -> list[DocumentRecord]:
    """Asigna una partición por persona y devuelve los registros etiquetados."""
    if not records:
        return []
    if holdout_attack is not None and holdout_attack not in ATTACK_TYPES:
        raise ValueError(f"Ataque no soportado para holdout: {holdout_attack}")

    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != set(RATIO_SPLITS):
        raise ValueError(f"Las ratios deben cubrir exactamente {RATIO_SPLITS}")
    if any(value < 0 for value in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError("Las ratios deben ser no negativas y sumar más que cero")

    groups = _group_by_source(records)
    real_groups = {source_id for source_id in groups if source_id.startswith(REAL_PREFIX)}
    holdout_groups = {
        source_id
        for source_id, group in groups.items()
        if source_id not in real_groups
        and holdout_attack is not None
        and any(record.primary_attack == holdout_attack for record in group)
    }
    available = [
        source_id
        for source_id in groups
        if source_id not in real_groups and source_id not in holdout_groups
    ]

    if len(available) < len(RATIO_SPLITS):
        raise ValueError(
            "Se necesitan al menos "
            f"{len(RATIO_SPLITS)} personas para train/val/calibration/test; "
            f"hay {len(available)} disponibles"
        )

    assignment = _assign_groups(groups, available, ratios, seed=seed)
    for source_id in holdout_groups:
        assignment[source_id] = "unseen_attack"
    for source_id in real_groups:
        assignment[source_id] = REAL_SPLIT

    labelled = [replace(record, split=assignment[record.source_id]) for record in records]
    verify_no_group_leak(labelled)
    verify_holdout(labelled, holdout_attack)
    return labelled


def _assign_groups(
    groups: dict[str, list[DocumentRecord]],
    available: list[str],
    ratios: dict[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    attack_families = {
        source_id: {record.primary_attack for record in group if record.primary_attack is not None}
        for source_id, group in groups.items()
    }
    remaining = set(available)
    assignment: dict[str, str] = {}

    # Primero cubre en train todas las familias de ataque posibles, dejando al
    # menos un grupo con ataque para cada partición de evaluación.
    uncovered = set().union(*(attack_families[source_id] for source_id in available))
    while uncovered:
        attack_groups = [source_id for source_id in remaining if attack_families[source_id]]
        if len(attack_groups) <= len(EVAL_SPLITS):
            break
        candidates = [
            source_id for source_id in attack_groups if attack_families[source_id] & uncovered
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda source_id: (
                -len(attack_families[source_id] & uncovered),
                -len(groups[source_id]),
                source_id,
            )
        )
        chosen = candidates[0]
        assignment[chosen] = "train"
        remaining.remove(chosen)
        uncovered -= attack_families[chosen]

    # Reserva un grupo con ataque para test, val y calibración. Así las
    # particiones de evaluación no quedan formadas solo por auténticos.
    eval_covered: set[str] = set()
    for split in EVAL_SPLITS:
        if not remaining:
            break
        attack_groups = [source_id for source_id in remaining if attack_families[source_id]]
        pool = attack_groups or list(remaining)
        rng.shuffle(pool)
        pool.sort(
            key=lambda source_id: (
                -len(attack_families[source_id] - eval_covered),
                -len(attack_families[source_id]),
                -len(groups[source_id]),
                source_id,
            )
        )
        chosen = pool[0]
        assignment[chosen] = split
        remaining.remove(chosen)
        eval_covered |= attack_families[chosen]

    # El resto se asigna a la partición con mayor déficit respecto a su target,
    # completando sobre todo train.
    current: dict[str, int] = {split: 0 for split in RATIO_SPLITS}
    for source_id, split in assignment.items():
        current[split] += len(groups[source_id])
    total = sum(len(groups[source_id]) for source_id in available)
    targets = {split: max(1.0, total * ratios[split]) for split in RATIO_SPLITS}

    for source_id in sorted(remaining, key=lambda item: (-len(groups[item]), item)):
        split = max(
            RATIO_SPLITS,
            key=lambda candidate: (
                targets[candidate] - current[candidate],
                -RATIO_SPLITS.index(candidate),
            ),
        )
        assignment[source_id] = split
        current[split] += len(groups[source_id])

    return assignment


def _group_by_source(records: list[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
    groups: dict[str, list[DocumentRecord]] = {}
    for record in records:
        groups.setdefault(record.source_id, []).append(record)
    return groups


def verify_no_group_leak(records: list[DocumentRecord]) -> None:
    """Falla si una persona o un documento original aparece en dos particiones."""
    source_split: dict[str, str] = {}
    document_split: dict[str, str] = {}

    for record in records:
        if record.split is None:
            raise ValueError(f"{record.sample_id} no tiene partición asignada")
        _check_mapping(source_split, record.source_id, record.split, "source_id")
        document_key = record.origin_id or record.document_id
        _check_mapping(document_split, document_key, record.split, "document_id")

    # T7 reutiliza bytes del donante: su document_id debe caer en la misma partición.
    by_document_id: dict[str, str] = {}
    for record in records:
        _check_mapping(by_document_id, record.document_id, record.split, "document_id")


def verify_holdout(records: list[DocumentRecord], holdout_attack: str | None) -> None:
    if holdout_attack is None:
        return
    leaked = [
        record
        for record in records
        if record.primary_attack == holdout_attack and record.split != "unseen_attack"
    ]
    if leaked:
        raise ValueError(
            f"El ataque reservado {holdout_attack} aparece fuera de unseen_attack: "
            f"{leaked[0].sample_id}"
        )


def _check_mapping(mapping: dict[str, str], key: str, split: str, label: str) -> None:
    previous = mapping.get(key)
    if previous is None:
        mapping[key] = split
    elif previous != split:
        raise ValueError(f"Fuga de {label} {key}: {previous} y {split}")


def summarize_splits(
    records: list[DocumentRecord],
    *,
    seed: int,
    holdout_attack: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generator": "fraude.synth.splits",
        "seed": seed,
        "holdout_attack": holdout_attack,
        "count": len(records),
        "splits": _count(record.split for record in records),
        "groups_by_split": _groups_by_split(records),
        "by_document_type": _nested_count(records, "document_type"),
        "by_attack": _nested_count(records, "primary_attack"),
        "by_variant": _nested_count(records, "variant"),
        "by_augmentation": _nested_count(records, "augmentation"),
    }


def write_split_manifest(
    records: list[DocumentRecord],
    out_dir: Path,
    *,
    seed: int = 42,
    holdout_attack: str | None = "T5",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    summary = summarize_splits(records, seed=seed, holdout_attack=holdout_attack)
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_dataset(data_dir: Path) -> list[DocumentRecord]:
    """Carga auténticos, ataques y aumentados disponibles en el directorio."""
    data_dir = Path(data_dir)
    records: list[DocumentRecord] = []
    for filename in ("manifest.jsonl", "attacks_manifest.jsonl", "augmented_manifest.jsonl"):
        path = data_dir / filename
        if path.exists():
            records.extend(load_manifest(path))
    if not records:
        raise FileNotFoundError(f"No hay manifiestos en {data_dir}")
    return records


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _nested_count(records: list[DocumentRecord], attribute: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for record in records:
        split = str(record.split)
        if attribute == "augmentation":
            value = record.augmentations[-1] if record.augmentations else "none"
        else:
            value = str(getattr(record, attribute))
        grouped.setdefault(split, Counter())[value] += 1
    return {split: dict(sorted(counter.items())) for split, counter in sorted(grouped.items())}


def _groups_by_split(records: list[DocumentRecord]) -> dict[str, int]:
    groups: dict[str, set[str]] = {}
    for record in records:
        groups.setdefault(str(record.split), set()).add(record.source_id)
    return {split: len(values) for split, values in sorted(groups.items())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crea particiones agrupadas F2.5.")
    parser.add_argument("--data", type=Path, required=True, help="Directorio del dataset.")
    parser.add_argument("--out", type=Path, default=None, help="Directorio de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument(
        "--holdout",
        default="T5",
        help="Ataque reservado para unseen_attack; 'none' para desactivarlo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out is not None else data_dir
    holdout = None if args.holdout.lower() == "none" else args.holdout

    records = load_dataset(data_dir)
    labelled = assign_splits(records, seed=args.seed, holdout_attack=holdout)
    write_split_manifest(labelled, out_dir, seed=args.seed, holdout_attack=holdout)
    print(f"Asignadas {len(labelled)} muestras a {out_dir / 'split_manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
