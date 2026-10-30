"""Construcción del espejo sintético del registro interno (F3, E3-5).

El registro real del CETPRO se cargará con ``load_registry``. Mientras no
exista, este módulo agrega una identidad por ``source_id`` desde el manifiesto
sintético y escribe ``data/registry/synthetic.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fraude.registry.cross import RegistryEntry, save_registry

DEFAULT_MANIFEST = Path("data/synthetic/v1/manifest.jsonl")
DEFAULT_OUT = Path("data/registry/synthetic.jsonl")

_NAME_PRIORITY = {
    "estudiante": 0,
    "estudiante_0": 1,
    "nombre": 2,
    "dni_name": 3,
}


def build_registry_from_manifest(path: Path = DEFAULT_MANIFEST) -> list[RegistryEntry]:
    """Agrega una entrada por persona usando solo documentos auténticos."""
    path = Path(path)
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    name_priority: dict[str, int] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("is_authentic", True):
            continue
        source_id = record.get("source_id") or record.get("sample_id")
        if not source_id:
            continue
        fields = record.get("fields", {})
        entry = grouped[source_id]

        _set(entry, "dni", _first(fields, "numero", "dni", "dni_0"))
        program = _first(fields, "programa")
        module = _first(fields, "modulo")
        period = _first(fields, "periodo")
        birth = _first(fields, "fecha_nacimiento")
        _set(entry, "programa", program)
        _set(entry, "modulo", module)
        _set(entry, "periodo", period)
        _set(entry, "fecha_nacimiento", birth)

        candidate_name, priority = _candidate_name(fields)
        if candidate_name and (
            source_id not in name_priority or priority < name_priority[source_id]
        ):
            entry["nombre"] = candidate_name
            name_priority[source_id] = priority

    entries = [
        RegistryEntry(
            source_id=source_id,
            nombre=str(values.get("nombre", "")).strip(),
            dni=values.get("dni"),
            programa=values.get("programa"),
            modulo=values.get("modulo"),
            periodo=values.get("periodo"),
            fecha_nacimiento=values.get("fecha_nacimiento"),
        )
        for source_id, values in sorted(grouped.items())
        if values.get("nombre")
    ]
    return entries


def _candidate_name(fields: dict[str, Any]) -> tuple[str | None, int]:
    for key in ("estudiante", "estudiante_0", "nombre"):
        value = fields.get(key)
        if value:
            return str(value).strip(), _NAME_PRIORITY[key]
    nombres = fields.get("nombres")
    apellidos = fields.get("apellidos")
    if nombres and apellidos:
        return f"{apellidos}, {nombres}", _NAME_PRIORITY["dni_name"]
    return None, 99


def _first(fields: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = fields.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _set(target: dict[str, Any], key: str, value: str | None) -> None:
    if value and not target.get(key):
        target[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construye el registro sintético espejo.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = build_registry_from_manifest(args.manifest)
    save_registry(entries, args.out)
    print(f"Registro sintético: {len(entries)} personas en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
