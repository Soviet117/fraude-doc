"""Construcción reproducible del dataset sintético v1 (F2).

Ejecuta en orden generación de auténticos, ataques T1-T8, aumentación de
captura y particiones agrupadas. El resultado queda en ``data/synthetic/v1``
con todos los manifiestos y un resumen ``build.json``.

Uso:
    uv run python -m fraude.synth.build --out data/synthetic/v1 --count 42 --identities 14
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fraude.synth.attacks import generate_attacks
from fraude.synth.augment import DEFAULT_VARIANTS, augment_dataset
from fraude.synth.dataset import DocumentRecord
from fraude.synth.generate import DEFAULT_DPI, DEFAULT_SEED, generate_dataset
from fraude.synth.render import TEMPLATE_VERSION
from fraude.synth.splits import assign_splits, write_split_manifest

DEFAULT_OUT = Path("data/synthetic/v1")
DEFAULT_COUNT = 42
DEFAULT_IDENTITIES = 14
DEFAULT_COPIES = 2


def build_dataset(
    out_dir: Path = DEFAULT_OUT,
    *,
    count: int = DEFAULT_COUNT,
    identities: int = DEFAULT_IDENTITIES,
    seed: int = DEFAULT_SEED,
    dpi: int = DEFAULT_DPI,
    copies: int = DEFAULT_COPIES,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    holdout_attack: str | None = "T5",
) -> list[DocumentRecord]:
    """Genera el dataset v1 completo y devuelve las muestras con partición."""
    if not variants:
        raise ValueError("Se necesita al menos una variante de captura")
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Variantes no soportadas: {', '.join(unknown)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    authentic = generate_dataset(
        out_dir,
        count=count,
        identities=identities,
        seed=seed,
        dpi=dpi,
    )
    attacks = generate_attacks(
        authentic,
        out_dir,
        seed=seed,
        copies=copies,
        dpi=dpi,
    )
    augmented = augment_dataset(
        [*authentic, *attacks],
        out_dir,
        seed=seed,
        variants=variants,
    )
    labelled = assign_splits(
        [*authentic, *attacks, *augmented],
        seed=seed,
        holdout_attack=holdout_attack,
    )
    write_split_manifest(labelled, out_dir, seed=seed, holdout_attack=holdout_attack)

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.synth.build",
        "template_version": TEMPLATE_VERSION,
        "seed": seed,
        "dpi": dpi,
        "holdout_attack": holdout_attack,
        "variants": list(variants),
        "copies": copies,
        "counts": {
            "authentic": len(authentic),
            "attacks": len(attacks),
            "augmented": len(augmented),
            "total": len(labelled),
        },
        "by_split": dict(sorted(Counter(str(record.split) for record in labelled).items())),
        "by_document_type": dict(
            sorted(Counter(record.document_type for record in labelled).items())
        ),
        "by_attack": dict(
            sorted(Counter(str(record.primary_attack) for record in labelled).items())
        ),
    }
    (out_dir / "build.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return labelled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construye el dataset sintético v1 completo.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directorio de salida.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Documentos auténticos.")
    parser.add_argument(
        "--identities", type=int, default=DEFAULT_IDENTITIES, help="Personas ficticias."
    )
    parser.add_argument("--copies", type=int, default=DEFAULT_COPIES, help="Copias por ataque.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Resolución de rasterizado.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Variantes de captura separadas por coma.",
    )
    parser.add_argument(
        "--holdout",
        default="T5",
        help="Ataque reservado para unseen_attack; 'none' para desactivarlo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants:
        raise SystemExit("Debes indicar al menos una variante de captura")
    holdout = None if args.holdout.lower() == "none" else args.holdout

    records = build_dataset(
        args.out,
        count=args.count,
        identities=args.identities,
        seed=args.seed,
        dpi=args.dpi,
        copies=args.copies,
        variants=variants,
        holdout_attack=holdout,
    )
    print(f"Dataset v1 construido: {len(records)} muestras en {args.out}")
    print(f"Manifiesto: {args.out / 'split_manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
