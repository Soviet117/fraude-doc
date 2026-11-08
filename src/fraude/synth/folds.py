"""Validación cruzada agrupada por persona para F4.

Los splits fijos de F2.5 separan train/val/calibration/test una sola vez. Para
estimar desempeño con 50 documentos se necesita además un k-fold agrupado: la
misma persona (y por tanto todas sus muestras auténticas, manipuladas y
aumentadas) cae siempre en el mismo fold. El reparto es determinista por
semilla y equilibra tamaño y número de manipulados por fold.

Uso:
    uv run python -m fraude.synth.folds data/synthetic/v2 --folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fraude.synth.dataset import DocumentRecord
from fraude.synth.validate import load_corpus


def assign_grouped_folds(
    records: list[DocumentRecord],
    *,
    k: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """Asigna un fold 0..k-1 a cada ``source_id`` y verifica que no haya fuga."""
    if k < 2:
        raise ValueError("k debe ser al menos 2")
    groups: dict[str, list[DocumentRecord]] = {}
    for record in records:
        groups.setdefault(record.source_id, []).append(record)
    if len(groups) < k:
        raise ValueError(f"Se necesitan al menos {k} personas; hay {len(groups)}")

    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda source_id: (-len(groups[source_id]), source_id))

    fold_sizes = [0] * k
    fold_positives = [0] * k
    assignment: dict[str, int] = {}
    for source_id in order:
        group = groups[source_id]
        positives = sum(1 for record in group if not record.is_authentic)
        fold = min(
            range(k),
            key=lambda index: (fold_sizes[index], fold_positives[index], index),
        )
        assignment[source_id] = fold
        fold_sizes[fold] += len(group)
        fold_positives[fold] += positives

    verify_fold_grouping(records, assignment, k=k)
    return assignment


def verify_fold_grouping(
    records: list[DocumentRecord],
    assignment: dict[str, int],
    *,
    k: int,
) -> None:
    """Falla si una persona aparece en más de un fold o falta alguna asignación."""
    seen: dict[str, int] = {}
    for record in records:
        fold = assignment.get(record.source_id)
        if fold is None:
            raise ValueError(f"{record.source_id} no tiene fold asignado")
        if fold < 0 or fold >= k:
            raise ValueError(f"Fold inválido para {record.source_id}: {fold}")
        previous = seen.get(record.source_id)
        if previous is not None and previous != fold:
            raise ValueError(f"Fuga de persona {record.source_id}: folds {previous} y {fold}")
        seen[record.source_id] = fold


def folds_manifest(
    records: list[DocumentRecord],
    assignment: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": record.sample_id,
            "source_id": record.source_id,
            "fold": assignment[record.source_id],
            "is_authentic": record.is_authentic,
            "primary_attack": record.primary_attack,
            "document_type": record.document_type,
            "augmentations": list(record.augmentations),
            "split": record.split,
        }
        for record in records
    ]


def summarize_folds(
    records: list[DocumentRecord],
    assignment: dict[str, int],
    *,
    k: int,
    seed: int,
) -> dict[str, Any]:
    by_fold: dict[int, list[DocumentRecord]] = {fold: [] for fold in range(k)}
    for record in records:
        by_fold[assignment[record.source_id]].append(record)
    return {
        "schema_version": "1.0",
        "generator": "fraude.synth.folds",
        "k": k,
        "seed": seed,
        "count": len(records),
        "groups": len(assignment),
        "folds": {
            str(fold): {
                "samples": len(group),
                "authentic": sum(1 for record in group if record.is_authentic),
                "tampered": sum(1 for record in group if not record.is_authentic),
                "groups": len({record.source_id for record in group}),
                "by_attack": dict(
                    sorted(Counter(str(record.primary_attack) for record in group).items())
                ),
                "by_document_type": dict(
                    sorted(Counter(record.document_type for record in group).items())
                ),
            }
            for fold, group in sorted(by_fold.items())
        },
    }


def write_folds(
    records: list[DocumentRecord],
    out_dir: Path,
    *,
    k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    assignment = assign_grouped_folds(records, k=k, seed=seed)
    summary = summarize_folds(records, assignment, k=k, seed=seed)
    manifest_path = out_dir / "folds_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in folds_manifest(records, assignment):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "folds_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_folds(path: Path) -> dict[str, int]:
    """Carga ``folds_manifest.jsonl`` y devuelve sample_id -> fold."""
    folds: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        folds[row["sample_id"]] = int(row["fold"])
    return folds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asigna folds agrupados por persona.")
    parser.add_argument("data", type=Path, help="Directorio del corpus.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_corpus(args.data)
    if not records:
        raise SystemExit(f"No hay manifiestos en {args.data}")
    summary = write_folds(records, args.data, k=args.folds, seed=args.seed)
    print(json.dumps(summary["folds"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
