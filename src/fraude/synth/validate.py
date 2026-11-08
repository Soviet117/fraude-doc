"""Validación de integridad y balance de un corpus sintético.

Comprueba que los manifiestos, las imágenes, las máscaras y las particiones
sean coherentes entre sí antes de gastar tiempo de entrenamiento. No mide
calidad del modelo: solo detecta corpus roto, etiquetas huérfanas y fuga de
grupos entre particiones.

Uso:
    uv run python -m fraude.synth.validate data/synthetic/v2 \
        --expect-authentic 25 --expect-tampered 25 --require-splits
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fraude.ingest.hashing import sha256_file
from fraude.synth.attacks import load_manifest
from fraude.synth.dataset import ATTACK_TYPES, DocumentRecord
from fraude.synth.splits import verify_no_group_leak

MANIFESTS = ("manifest.jsonl", "attacks_manifest.jsonl", "augmented_manifest.jsonl")


def load_corpus(data_dir: Path) -> list[DocumentRecord]:
    """Carga el manifiesto con particiones si existe; si no, los manifiestos base."""
    data_dir = Path(data_dir)
    split_path = data_dir / "split_manifest.jsonl"
    if split_path.exists():
        return load_manifest(split_path)
    records: list[DocumentRecord] = []
    for filename in MANIFESTS:
        path = data_dir / filename
        if path.exists():
            records.extend(load_manifest(path))
    return records


def validate_dataset(
    data_dir: Path,
    *,
    expect_authentic: int | None = None,
    expect_tampered: int | None = None,
    require_splits: bool = False,
    check_hashes: bool = True,
) -> dict[str, Any]:
    """Valida el corpus y devuelve un resumen con la lista de errores."""
    data_dir = Path(data_dir)
    errors: list[str] = []
    warnings: list[str] = []

    records = load_corpus(data_dir)
    if not records:
        errors.append(f"No hay manifiestos en {data_dir}")
        return {"ok": False, "errors": errors, "warnings": warnings, "counts": {}}

    by_document_id: dict[str, list[DocumentRecord]] = {}
    for record in records:
        by_document_id.setdefault(record.document_id, []).append(record)

    image_sizes: dict[str, tuple[int, int]] = {}
    for record in records:
        image_path = data_dir / record.image_path
        if not image_path.exists():
            errors.append(f"{record.sample_id}: falta la imagen {record.image_path}")
            continue
        if check_hashes:
            digest = sha256_file(image_path)
            if digest != record.document_id:
                errors.append(f"{record.sample_id}: hash de imagen no coincide con document_id")
        image_sizes[record.sample_id] = _image_size(image_path)

        for mask_path in record.mask_paths:
            mask_file = data_dir / mask_path
            if not mask_file.exists():
                errors.append(f"{record.sample_id}: falta la máscara {mask_path}")
                continue
            if mask_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                mask_size = _image_size(mask_file)
                if mask_size != image_sizes[record.sample_id]:
                    errors.append(
                        f"{record.sample_id}: máscara {mask_path} mide {mask_size}, "
                        f"la imagen {image_sizes[record.sample_id]}"
                    )

    for record in records:
        if record.is_authentic:
            if record.variant != "authentic":
                errors.append(f"{record.sample_id}: auténtico con variant={record.variant}")
            if record.primary_attack is not None or record.attack_types:
                errors.append(f"{record.sample_id}: auténtico con ataque declarado")
            if record.parent_id is not None:
                errors.append(f"{record.sample_id}: auténtico con parent_id")
            continue

        if record.variant != "tampered":
            errors.append(f"{record.sample_id}: manipulado con variant={record.variant}")
        if record.primary_attack not in ATTACK_TYPES:
            errors.append(f"{record.sample_id}: ataque inválido {record.primary_attack!r}")
        if not record.attack_types:
            errors.append(f"{record.sample_id}: manipulado sin attack_types")
        if record.parent_id is None:
            errors.append(f"{record.sample_id}: manipulado sin parent_id")
            continue
        parents = by_document_id.get(record.parent_id, [])
        if not parents:
            errors.append(f"{record.sample_id}: parent_id huérfano {record.parent_id}")
            continue
        parent = parents[0]
        if parent.source_id != record.source_id:
            errors.append(
                f"{record.sample_id}: el padre pertenece a {parent.source_id}, "
                f"la muestra a {record.source_id}"
            )
        if record.primary_attack == "T7" and not any(parent.is_authentic for parent in parents):
            errors.append(f"{record.sample_id}: T7 sin documento original reutilizado")

    for document_id, duplicates in by_document_id.items():
        if len(duplicates) < 2:
            continue
        non_t7 = [record for record in duplicates if record.primary_attack != "T7"]
        if len(non_t7) > 1:
            errors.append(f"document_id duplicado sin ser reutilización T7: {document_id}")
        groups = {record.source_id for record in duplicates}
        if len(groups) > 1:
            errors.append(f"document_id {document_id} aparece en varios grupos: {sorted(groups)}")

    authentic = [record for record in records if record.is_authentic]
    tampered = [record for record in records if not record.is_authentic]
    authentic_base = [record for record in authentic if not record.augmentations]
    tampered_base = [record for record in tampered if not record.augmentations]
    augmented = [record for record in records if record.augmentations]
    if expect_authentic is not None and len(authentic_base) != expect_authentic:
        errors.append(f"Auténticos base: esperados {expect_authentic}, hay {len(authentic_base)}")
    if expect_tampered is not None and len(tampered_base) != expect_tampered:
        errors.append(f"Manipulados base: esperados {expect_tampered}, hay {len(tampered_base)}")

    split_path = data_dir / "split_manifest.jsonl"
    if require_splits and not split_path.exists():
        errors.append("Falta split_manifest.jsonl")
    if split_path.exists():
        try:
            verify_no_group_leak(records)
        except ValueError as exc:
            errors.append(f"Fuga entre particiones: {exc}")
        if any(record.split is None for record in records):
            errors.append("Hay muestras sin partición asignada")

    build_path = data_dir / "build.json"
    if build_path.exists():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        counts = build.get("counts", {})
        if counts.get("total") != len(records):
            errors.append(
                f"build.json declara {counts.get('total')} muestras, el manifiesto tiene {len(records)}"
            )
        if counts.get("authentic") != len(authentic_base):
            errors.append("build.json no coincide con el número de auténticos base")
        if counts.get("attacks") != len(tampered_base):
            errors.append("build.json no coincide con el número de manipulados base")
        if counts.get("augmented") != len(augmented):
            errors.append("build.json no coincide con el número de aumentados")
        per_attack = build.get("per_attack")
        if per_attack:
            actual = Counter(record.primary_attack for record in tampered_base)
            for attack_type, expected in per_attack.items():
                if actual.get(attack_type, 0) != expected:
                    errors.append(
                        f"{attack_type}: build.json espera {expected}, hay {actual.get(attack_type, 0)}"
                    )
    else:
        warnings.append("No hay build.json")

    groups = {record.source_id for record in records}
    if len(groups) < 10:
        warnings.append(f"Solo hay {len(groups)} personas; el plan pide 10-12")

    counts = {
        "total": len(records),
        "authentic": len(authentic),
        "tampered": len(tampered),
        "authentic_base": len(authentic_base),
        "tampered_base": len(tampered_base),
        "augmented": len(augmented),
        "groups": len(groups),
        "by_attack": dict(
            sorted(Counter(str(record.primary_attack) for record in tampered_base).items())
        ),
        "by_split": dict(sorted(Counter(str(record.split) for record in records).items())),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "counts": counts}


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida un corpus sintético.")
    parser.add_argument("data", type=Path, help="Directorio del corpus.")
    parser.add_argument("--expect-authentic", type=int, default=None)
    parser.add_argument("--expect-tampered", type=int, default=None)
    parser.add_argument("--require-splits", action="store_true")
    parser.add_argument("--no-hashes", action="store_true", help="Omite la verificación de hashes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(
        args.data,
        expect_authentic=args.expect_authentic,
        expect_tampered=args.expect_tampered,
        require_splits=args.require_splits,
        check_hashes=not args.no_hashes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
