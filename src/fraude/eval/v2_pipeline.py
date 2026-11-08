"""Ejecuta el pipeline F3 sobre los documentos base del corpus v2.

Genera una tabla de observaciones reutilizable por F6 (semántica) y F7
(fusión). No usa campos oráculo: el OCR, la extracción y las reglas se ejecutan
sobre la imagen.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from fraude.pipeline.run import analyze_document
from fraude.registry.build import build_registry_from_manifest
from fraude.synth.attacks import load_manifest

LOGGER = logging.getLogger("fraude.eval.v2_pipeline")

DEFAULT_DATA = Path("data/synthetic/v2")
DEFAULT_OUT = Path("artifacts/eval/v2_pipeline.jsonl")
DEFAULT_TODAY = date(2026, 11, 5)


def base_records(data_dir: Path) -> list[Any]:
    """Carga auténticos y manipulados base; excluye aumentaciones."""
    records = []
    for name in ("manifest.jsonl", "attacks_manifest.jsonl"):
        path = Path(data_dir) / name
        if path.exists():
            records.extend(load_manifest(path))
    return records


def run_pipeline(
    data_dir: Path = DEFAULT_DATA,
    *,
    out_path: Path = DEFAULT_OUT,
    today: date = DEFAULT_TODAY,
    force: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    records = base_records(data_dir)
    if limit is not None:
        records = records[:limit]
    registry = build_registry_from_manifest(data_dir / "manifest.jsonl")
    LOGGER.info(
        "Pipeline v2: %d documentos base, %d entradas de registro", len(records), len(registry)
    )

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        LOGGER.info("%s (%d/%d)", record.sample_id, index, len(records))
        observation = analyze_document(
            data_dir / record.image_path,
            registry=registry,
            today=today,
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "source_id": record.source_id,
                "document_type": record.document_type,
                "variant": record.variant,
                "primary_attack": record.primary_attack,
                "is_authentic": record.is_authentic,
                "image_path": record.image_path,
                "mask_paths": list(record.mask_paths),
                "fields_gold": record.fields,
                "observation": observation.model_dump(mode="json"),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Observaciones escritas en %s", out_path)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta F3 sobre el corpus v2 base.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--today", type=date.fromisoformat, default=DEFAULT_TODAY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rows = run_pipeline(
        args.data,
        out_path=args.out,
        today=args.today,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps({"documents": len(rows), "out": str(args.out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
