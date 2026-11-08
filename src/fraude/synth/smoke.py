"""Lote de humo de tres documentos para validar el camino completo.

Genera dos auténticos y un manipulado T1 del primer auténtico, sin
aumentación y sin particiones. Sirve para probar manifiestos, validación,
carga de imágenes y un forward/backward en CPU antes de construir el corpus
completo de 50 documentos.

Uso:
    uv run python -m fraude.synth.smoke
    uv run python -m fraude.synth.smoke --out data/synthetic/v2_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraude.synth.build import build_dataset
from fraude.synth.dataset import DocumentRecord
from fraude.synth.validate import validate_dataset

DEFAULT_OUT = Path("data/synthetic/v2_smoke")


def build_smoke(
    out_dir: Path = DEFAULT_OUT, *, seed: int = 42, dpi: int = 150
) -> list[DocumentRecord]:
    """Construye el lote de humo y falla si la validación no pasa."""
    records = build_dataset(
        out_dir,
        count=2,
        identities=2,
        seed=seed,
        dpi=dpi,
        copies=1,
        variants=(),
        attacks=("T1",),
        per_attack={"T1": 1},
        split=False,
    )
    report = validate_dataset(
        out_dir,
        expect_authentic=2,
        expect_tampered=1,
        require_splits=False,
    )
    payload = {
        "schema_version": "1.0",
        "purpose": "smoke_test",
        "documents": [record.sample_id for record in records],
        "validation": report,
    }
    (Path(out_dir) / "smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["ok"]:
        raise SystemExit("El lote de humo no pasó la validación: " + "; ".join(report["errors"]))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construye el lote de humo de 3 documentos.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = build_smoke(args.out, seed=args.seed, dpi=args.dpi)
    print(f"Lote de humo: {len(records)} documentos en {args.out}")
    for record in records:
        print(f"  {record.sample_id}: {record.variant} {record.primary_attack or ''}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
