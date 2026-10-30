"""Cruce con el registro interno del CETPRO (F3, E3-5).

El registro puede ser un CSV institucional o un espejo sintético JSONL. La
coincidencia prioriza DNI exacto y usa similitud difusa para nombres,
programas y periodos. Nunca decide fraude: devuelve ``RegistryReport`` con
score y diferencias para que la fusión y la revisión humana lo expliquen.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fraude.core.schemas import RegistryReport

DEFAULT_THRESHOLD = 0.86
_DNI_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class RegistryEntry:
    """Persona o documento esperado en el registro interno."""

    nombre: str
    dni: str | None = None
    programa: str | None = None
    modulo: str | None = None
    periodo: str | None = None
    fecha_nacimiento: str | None = None
    source_id: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> RegistryEntry:
        return cls(
            nombre=str(_pick(mapping, "nombre", "name", "estudiante", "full_name") or "").strip(),
            dni=_optional(_pick(mapping, "dni", "documento", "numero")),
            programa=_optional(_pick(mapping, "programa", "program", "carrera")),
            modulo=_optional(_pick(mapping, "modulo", "module")),
            periodo=_optional(_pick(mapping, "periodo", "period")),
            fecha_nacimiento=_optional(
                _pick(mapping, "fecha_nacimiento", "birth_date", "nacimiento")
            ),
            source_id=_optional(_pick(mapping, "source_id", "id")),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "source_id": self.source_id,
            "nombre": self.nombre,
            "dni": self.dni,
            "programa": self.programa,
            "modulo": self.modulo,
            "periodo": self.periodo,
            "fecha_nacimiento": self.fecha_nacimiento,
        }


def load_registry(path: Path) -> list[RegistryEntry]:
    """Carga un registro CSV, JSON o JSONL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not isinstance(rows, list):
        raise ValueError("El registro debe ser una lista de objetos")
    return [entry for row in rows if (entry := RegistryEntry.from_mapping(row)).nombre]


def save_registry(entries: list[RegistryEntry], path: Path) -> Path:
    """Guarda un registro JSONL reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return path


def cross_registry(
    fields: dict[str, Any],
    registry: list[RegistryEntry],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> RegistryReport:
    """Compara campos extraídos contra el registro y devuelve diferencias."""
    if not registry:
        return RegistryReport(
            matched=False,
            score=None,
            diffs=["registro vacío o no disponible"],
        )

    values = {key: str(value).strip() for key, value in fields.items() if value is not None}
    name = _candidate_name(values)
    dni = _candidate_dni(values)
    programa = values.get("programa")
    modulo = values.get("modulo")
    periodo = values.get("periodo")
    birth = values.get("fecha_nacimiento")

    exact = [entry for entry in registry if dni and entry.dni == dni]
    if exact:
        entry = max(exact, key=lambda item: _name_similarity(name, item.nombre))
        score = _score(entry, name, programa, periodo, exact_dni=True)
        diffs = _diffs(entry, name, dni, programa, modulo, periodo, birth)
        return RegistryReport(matched=True, score=round(score, 4), diffs=diffs)

    best: tuple[float, RegistryEntry] | None = None
    for entry in registry:
        score = _score(entry, name, programa, periodo, exact_dni=False)
        if best is None or score > best[0]:
            best = (score, entry)
    if best is None:
        return RegistryReport(matched=False, score=None, diffs=["sin candidatos"])

    score, entry = best
    if score < threshold:
        return RegistryReport(
            matched=False,
            score=round(score, 4),
            diffs=[f"sin coincidencia suficiente (mejor={entry.nombre}, score={score:.2f})"],
        )
    diffs = _diffs(entry, name, dni, programa, modulo, periodo, birth)
    return RegistryReport(matched=True, score=round(score, 4), diffs=diffs)


def _score(
    entry: RegistryEntry,
    name: str | None,
    programa: str | None,
    periodo: str | None,
    *,
    exact_dni: bool,
) -> float:
    name_sim = _name_similarity(name, entry.nombre)
    program_sim = _text_similarity(programa, entry.programa)
    period_sim = _text_similarity(periodo, entry.periodo)
    if exact_dni:
        return min(1.0, 0.55 + 0.25 * name_sim + 0.15 * program_sim + 0.05 * period_sim)
    score = min(1.0, 0.55 * name_sim + 0.30 * program_sim + 0.15 * period_sim)
    if name_sim >= 0.98:
        # Partidas y otros documentos pueden no traer DNI, programa ni periodo.
        # Un nombre completo exacto es evidencia suficiente para localizar a la
        # persona; las diferencias de programa/periodo se reportan aparte.
        score = max(score, 0.90)
    return score


def _diffs(
    entry: RegistryEntry,
    name: str | None,
    dni: str | None,
    programa: str | None,
    modulo: str | None,
    periodo: str | None,
    birth: str | None,
) -> list[str]:
    diffs: list[str] = []
    if name and entry.nombre and _name_similarity(name, entry.nombre) < 0.95:
        diffs.append(f"nombre: extraido='{name}' registro='{entry.nombre}'")
    if dni and entry.dni and dni != entry.dni:
        diffs.append(f"dni: extraido='{dni}' registro='{entry.dni}'")
    for label, extracted, expected in (
        ("programa", programa, entry.programa),
        ("modulo", modulo, entry.modulo),
        ("periodo", periodo, entry.periodo),
        ("fecha_nacimiento", birth, entry.fecha_nacimiento),
    ):
        if extracted and expected and _normalize(extracted) != _normalize(expected):
            diffs.append(f"{label}: extraido='{extracted}' registro='{expected}'")
    return diffs


def _candidate_name(fields: dict[str, str]) -> str | None:
    for key in ("estudiante", "nombre", "estudiante_0"):
        if fields.get(key):
            return fields[key]
    nombres = fields.get("nombres")
    apellidos = fields.get("apellidos")
    if nombres and apellidos:
        return f"{apellidos}, {nombres}"
    return None


def _candidate_dni(fields: dict[str, str]) -> str | None:
    for key in ("dni", "numero", "dni_0"):
        value = fields.get(key)
        if value and _DNI_RE.match(value):
            return value
    return None


def _name_similarity(first: str | None, second: str | None) -> float:
    if not first or not second:
        return 0.0
    first_norm = _normalize(first)
    second_norm = _normalize(second)
    ordered = _ratio(first_norm, second_norm)
    first_tokens = " ".join(sorted(first_norm.split()))
    second_tokens = " ".join(sorted(second_norm.split()))
    return max(ordered, _ratio(first_tokens, second_tokens))


def _text_similarity(first: str | None, second: str | None) -> float:
    if not first or not second:
        return 0.0
    return _ratio(_normalize(first), _normalize(second))


def _ratio(first: str, second: str) -> float:
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return SequenceMatcher(None, first, second).ratio()
    return float(fuzz.ratio(first, second)) / 100.0


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", " ", without_accents.lower())
    return " ".join(cleaned.split())


def _pick(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _optional(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
