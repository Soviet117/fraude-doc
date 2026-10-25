"""Generador del dataset sintético auténtico (F2.2) y su manifiesto reproducible.

Uso:
    uv run python -m fraude.synth.generate --out data/synthetic/v1 --count 21 --seed 42

El dataset generado queda fuera de git (``data/`` está ignorado). El manifiesto
``manifest.jsonl`` es la etiqueta de cada muestra; ``dataset.json`` resume la
generación (semilla, DPI, conteos y tipos).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fraude.synth.data import Identity, build_fields, generate_identity
from fraude.synth.dataset import DOCUMENT_TYPES, DocumentRecord
from fraude.synth.render import TEMPLATE_VERSION, render_document

DEFAULT_OUT = Path("data/synthetic/v1")
DEFAULT_COUNT = 21
DEFAULT_SEED = 42
DEFAULT_DPI = 150


def generate_dataset(
    out_dir: Path = DEFAULT_OUT,
    *,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    dpi: int = DEFAULT_DPI,
    identities: int | None = None,
) -> list[DocumentRecord]:
    """Genera ``count`` documentos auténticos sintéticos y escribe el manifiesto."""
    if count < 1:
        raise ValueError("count debe ser mayor que cero")
    if dpi < 72:
        raise ValueError("dpi debe ser al menos 72")

    out_dir = Path(out_dir)
    rng = random.Random(seed)
    n_identities = identities if identities is not None else max(1, count // 3)
    n_identities = max(1, min(n_identities, count))

    used_dnis: set[str] = set()
    people: list[Identity] = [
        generate_identity(rng, index, used_dnis) for index in range(n_identities)
    ]

    records: list[DocumentRecord] = []
    for index in range(count):
        person_index = index % n_identities
        round_index = index // n_identities
        identity = people[person_index]
        document_type = DOCUMENT_TYPES[(round_index + person_index) % len(DOCUMENT_TYPES)]
        fields = build_fields(rng, document_type, identity)
        sample_id = f"{document_type}_{index:04d}"

        records.append(
            render_document(
                sample_id=sample_id,
                document_type=document_type,
                identity=identity,
                fields=fields,
                out_dir=out_dir,
                seed=seed,
                dpi=dpi,
            )
        )

    write_manifest(records, out_dir, seed=seed, dpi=dpi, count=count)
    return records


def write_manifest(
    records: list[DocumentRecord],
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    dpi: int = DEFAULT_DPI,
    count: int | None = None,
) -> None:
    """Escribe ``manifest.jsonl`` (una muestra por línea) y ``dataset.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.synth",
        "template_version": TEMPLATE_VERSION,
        "count": count if count is not None else len(records),
        "seed": seed,
        "dpi": dpi,
        "by_document_type": dict(
            sorted(Counter(record.document_type for record in records).items())
        ),
        "by_variant": dict(sorted(Counter(record.variant for record in records).items())),
        "by_source_id": dict(sorted(Counter(record.source_id for record in records).items())),
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera documentos auténticos sintéticos.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directorio de salida.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Número de documentos.")
    parser.add_argument(
        "--identities", type=int, default=None, help="Número de personas ficticias."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Resolución de rasterizado.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = generate_dataset(
        args.out,
        count=args.count,
        seed=args.seed,
        dpi=args.dpi,
        identities=args.identities,
    )
    print(f"Generados {len(records)} documentos en {args.out}")
    print(f"Manifiesto: {args.out / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
