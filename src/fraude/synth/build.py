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
    attacks: tuple[str, ...] | None = None,
    per_attack: dict[str, int] | None = None,
    split: bool = True,
) -> list[DocumentRecord]:
    """Genera el dataset sintético completo y devuelve las muestras con partición.

    ``variants=()`` desactiva la aumentación y ``split=False`` omite las
    particiones; ambos se usan para el lote de humo de tres documentos.
    """
    if variants:
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
    attack_records = generate_attacks(
        authentic,
        out_dir,
        seed=seed,
        copies=copies,
        dpi=dpi,
        attacks=attacks,
        per_attack=per_attack,
    )
    augmented = (
        augment_dataset(
            [*authentic, *attack_records],
            out_dir,
            seed=seed,
            variants=variants,
        )
        if variants
        else []
    )
    labelled = [*authentic, *attack_records, *augmented]
    if split:
        labelled = assign_splits(labelled, seed=seed, holdout_attack=holdout_attack)
        write_split_manifest(labelled, out_dir, seed=seed, holdout_attack=holdout_attack)

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.synth.build",
        "template_version": TEMPLATE_VERSION,
        "seed": seed,
        "dpi": dpi,
        "holdout_attack": holdout_attack if split else None,
        "split": split,
        "variants": list(variants),
        "copies": copies,
        "per_attack": dict(sorted(per_attack.items())) if per_attack is not None else None,
        "counts": {
            "authentic": len(authentic),
            "attacks": len(attack_records),
            "augmented": len(augmented),
            "total": len(labelled),
        },
        "by_split": (
            dict(sorted(Counter(str(record.split) for record in labelled).items())) if split else {}
        ),
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
    parser = argparse.ArgumentParser(description="Construye un dataset sintético completo.")
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
        help="Variantes de captura separadas por coma; vacío para desactivar.",
    )
    parser.add_argument(
        "--attacks",
        default=None,
        help="Subconjunto de ataques separados por coma (por defecto, T1-T8).",
    )
    parser.add_argument(
        "--per-attack",
        default=None,
        help="Copias por ataque con formato T1=4,T2=3 (sobrescribe --copies).",
    )
    parser.add_argument(
        "--holdout",
        default="T5",
        help="Ataque reservado para unseen_attack; 'none' para desactivarlo.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="No asigna particiones (lote de humo).",
    )
    return parser


def _parse_per_attack(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    parsed: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, separator, raw = item.partition("=")
        if not separator or not raw.strip().isdigit():
            raise SystemExit(f"Formato inválido en --per-attack: {item!r} (usa T1=4)")
        parsed[key.strip()] = int(raw.strip())
    return parsed or None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    attacks = (
        tuple(item.strip() for item in args.attacks.split(",") if item.strip())
        if args.attacks
        else None
    )
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
        attacks=attacks,
        per_attack=_parse_per_attack(args.per_attack),
        split=not args.no_split,
    )
    print(f"Dataset construido: {len(records)} muestras en {args.out}")
    if not args.no_split:
        print(f"Manifiesto: {args.out / 'split_manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
