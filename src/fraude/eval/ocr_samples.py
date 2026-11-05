"""Medición de OCR real sobre el corpus local de muestras.

El arnés ejecuta docTR sobre cada PNG, guarda el texto reconocido y calcula:

- salud del OCR: palabras, confianza media, líneas y latencia;
- recall de valores conocidos en el texto (DNI, nombres, fechas, programa,
  módulo y periodo) usando ``_personas.csv`` como referencia;
- precisión del tipo documental y de los campos extraídos por ``extract_page``.

No calcula CER/WER porque no existe una transcripción completa de referencia.
Tampoco decide autenticidad ni sustituye la validación con documentos reales.

Uso:
    uv run python -m fraude.eval.ocr_samples data/muestras \
        --out artifacts/eval/ocr_samples.json \
        --text-dir artifacts/ocr/samples
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from fraude.extract.fields import extract_page
from fraude.ocr.engine import group_lines, run_ocr

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/muestras")
DEFAULT_OUT = Path("artifacts/eval/ocr_samples.json")
DEFAULT_TEXT_DIR = Path("artifacts/ocr/samples")
DEFAULT_INDEX = "_indice.csv"
DEFAULT_PEOPLE = "_personas.csv"

REQUIRED_COLUMNS = (
    "id",
    "tipo",
    "variante",
    "ataque",
    "persona",
    "documento_padre",
    "campo_original",
    "campo_alterado",
    "png",
)

# Campos que deberían aparecer literalmente en el texto del documento. La
# fecha de la partida se escribe en formato largo; el resto usa dd/mm/aaaa.
OCR_EXPECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "dni": ("dni", "apellidos", "nombres", "fecha_nacimiento"),
    "certificado_estudios": ("dni", "fecha_nacimiento"),
    "partida_nacimiento": ("fecha_nacimiento",),
    "ficha_matricula": ("dni", "fecha_nacimiento", "programa", "modulo", "periodo"),
    "acta_evaluacion": ("programa", "modulo", "periodo"),
    "certificado_modulo": ("dni", "programa", "modulo", "periodo"),
    "titulo_auxiliar_tecnico": ("programa",),
}

# Correspondencia entre el nombre esperado y el campo que devuelve el
# extractor. La partida usa fecha larga y el extractor solo acepta dd/mm/aaaa,
# así que no se evalúa extracción de fecha en ese tipo.
EXTRACTION_FIELDS: dict[str, dict[str, str]] = {
    "dni": {
        "dni": "numero",
        "apellidos": "apellidos",
        "nombres": "nombres",
        "fecha_nacimiento": "fecha_nacimiento",
    },
    "certificado_estudios": {
        "dni": "dni",
        "fecha_nacimiento": "fecha_nacimiento",
    },
    "partida_nacimiento": {},
    "ficha_matricula": {
        "dni": "dni",
        "fecha_nacimiento": "fecha_nacimiento",
        "programa": "programa",
        "modulo": "modulo",
        "periodo": "periodo",
    },
    "acta_evaluacion": {
        "programa": "programa",
        "modulo": "modulo",
        "periodo": "periodo",
    },
    "certificado_modulo": {
        "dni": "dni",
        "programa": "programa",
        "modulo": "modulo",
        "periodo": "periodo",
    },
    "titulo_auxiliar_tecnico": {"programa": "programa"},
}

_PERIOD_RE = re.compile(
    r"\b(20\d{2})\s*[-\u2013]\s*(i{1,2}|1{1,2}|2)\b",
    re.IGNORECASE,
)

# Campos de texto libre donde el OCR puede perder el espacio entre palabras
# ("BAUTISTAHUAMAN"). La comparación compacta ignora espacios y puntuación,
# pero no letras extra: "PATRICIAS" sigue siendo un error frente a "PATRICIA".
_TEXT_FIELDS = frozenset(
    {
        "apellidos",
        "direccion",
        "docente",
        "estudiante",
        "institucion",
        "lugar_nacimiento",
        "madre",
        "modulo",
        "nombre",
        "nombres",
        "padre",
        "programa",
    }
)

MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "setiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

LIMITATIONS = (
    "El corpus es sintético y pequeño (20 documentos, 12 personas); los "
    "resultados no estiman desempeño institucional.",
    "No hay transcripción completa de referencia: no se reporta CER/WER. El "
    "recall de campos mide presencia de valores conocidos, no calidad global.",
    "Los valores esperados provienen de _personas.csv y del índice; algunos "
    "campos del documento no están disponibles y no se evalúan.",
    "La partida de nacimiento usa fecha larga y el extractor actual solo "
    "reconoce fechas dd/mm/aaaa; su extracción de fecha no se puntúa.",
    "docTR corre en CPU en este host; la latencia no representa al laboratorio con GPU.",
)


@dataclass(frozen=True)
class OcrSampleResult:
    """Resultado OCR y de extracción de una muestra."""

    sample_id: str
    document_type: str
    variant: str
    attack: str | None
    image_path: str
    n_words: int | None
    mean_conf: float | None
    n_lines: int | None
    text_chars: int | None
    latency_ms: float | None
    expected_fields: dict[str, str]
    matched_fields: dict[str, bool]
    ocr_field_recall: float | None
    altered_token: str | None
    altered_token_present: bool | None
    extracted_doctype: str | None
    doctype_ok: bool
    extracted_fields: dict[str, Any]
    extraction_expected: dict[str, str]
    extraction_matched: dict[str, bool]
    extraction_field_accuracy: float | None
    text_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "document_type": self.document_type,
            "variant": self.variant,
            "attack": self.attack,
            "image_path": self.image_path,
            "n_words": self.n_words,
            "mean_conf": self.mean_conf,
            "n_lines": self.n_lines,
            "text_chars": self.text_chars,
            "latency_ms": self.latency_ms,
            "expected_fields": self.expected_fields,
            "matched_fields": self.matched_fields,
            "ocr_field_recall": self.ocr_field_recall,
            "altered_token": self.altered_token,
            "altered_token_present": self.altered_token_present,
            "extraction": {
                "document_type": self.extracted_doctype,
                "doctype_ok": self.doctype_ok,
                "fields": self.extracted_fields,
                "expected": self.extraction_expected,
                "matched": self.extraction_matched,
                "field_accuracy": self.extraction_field_accuracy,
            },
            "text_path": self.text_path,
            "error": self.error,
        }


def run_ocr_samples(
    root: Path = DEFAULT_ROOT,
    *,
    index_name: str = DEFAULT_INDEX,
    people_name: str = DEFAULT_PEOPLE,
    out_path: Path | None = DEFAULT_OUT,
    text_dir: Path | None = DEFAULT_TEXT_DIR,
    limit: int | None = None,
    warmup: bool = True,
) -> dict[str, Any]:
    """Ejecuta OCR real y devuelve el reporte serializable."""
    root = Path(root)
    rows = _load_rows(root / index_name, REQUIRED_COLUMNS)
    personas = _load_personas(root / people_name)
    if limit is not None:
        rows = rows[:limit]
    rows_by_id = {_text(row.get("id")): row for row in rows}

    warmup_ms = None
    if warmup and rows:
        warmup_path = root / _text(rows[0].get("png"))
        if warmup_path.is_file():
            started = perf_counter()
            try:
                run_ocr(warmup_path)
                warmup_ms = round((perf_counter() - started) * 1000, 2)
            except Exception as exc:  # docTR puede no estar instalado o fallar
                logger.warning("Warmup OCR falló: %s", exc)

    results: list[OcrSampleResult] = []
    for index, row in enumerate(rows, start=1):
        sample_id = _text(row.get("id"))
        logger.info("OCR %s (%d/%d)", sample_id, index, len(rows))
        results.append(
            _run_sample(
                root,
                row,
                personas=personas,
                rows_by_id=rows_by_id,
                text_dir=text_dir,
            )
        )

    summary = _summarize(results, warmup_ms=warmup_ms)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.eval.ocr_samples",
        "dataset": {
            "root": str(root),
            "index": index_name,
            "people": people_name,
            "n_samples": len(results),
        },
        "engine": _engine_info(),
        "config": {
            "warmup": warmup,
            "warmup_ms": warmup_ms,
            "limit": limit,
        },
        "summary": summary,
        "samples": [result.to_dict() for result in results],
        "limitations": list(LIMITATIONS),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def _run_sample(
    root: Path,
    row: dict[str, str],
    *,
    personas: dict[str, dict[str, str]],
    rows_by_id: dict[str, dict[str, str]],
    text_dir: Path | None,
) -> OcrSampleResult:
    sample_id = _text(row.get("id"))
    image_path = _text(row.get("png"))
    expected = _expected_values(row, personas=personas, rows_by_id=rows_by_id)
    altered_token = _altered_token(row)
    image_file = root / image_path
    if not image_file.is_file():
        return _error_result(
            row,
            image_path=image_path,
            expected=expected,
            altered_token=altered_token,
            error=f"No existe la imagen {image_path}",
        )

    started = perf_counter()
    try:
        page = run_ocr(image_file)
    except Exception as exc:  # una muestra rota no debe abortar el lote
        return _error_result(
            row,
            image_path=image_path,
            expected=expected,
            altered_token=altered_token,
            error=str(exc),
        )
    latency_ms = round((perf_counter() - started) * 1000, 2)

    text = page.text
    text_path = None
    if text_dir is not None:
        text_dir = Path(text_dir)
        text_dir.mkdir(parents=True, exist_ok=True)
        text_file = text_dir / f"{sample_id}.txt"
        text_file.write_text(text + "\n", encoding="utf-8")
        text_path = str(text_file)

    text_norm = _normalize(text)
    matched = {field: _value_present(value, text_norm, field) for field, value in expected.items()}
    ocr_field_recall = _ratio(sum(matched.values()), len(matched))

    extraction = extract_page(page)
    doctype_ok = extraction.document_type == _text(row.get("tipo"))
    extraction_expected = _extraction_expected(row, expected)
    extraction_matched = {
        field: _values_match(
            expected[field],
            str(extraction.fields.get(extraction_field, "")),
            field,
        )
        for field, extraction_field in extraction_expected.items()
    }
    extraction_accuracy = _ratio(
        sum(extraction_matched.values()),
        len(extraction_matched),
    )
    altered_present = None
    if altered_token is not None:
        altered_present = _normalize(altered_token) in text_norm

    return OcrSampleResult(
        sample_id=sample_id,
        document_type=_text(row.get("tipo")),
        variant=_text(row.get("variante")),
        attack=_text(row.get("ataque")) or None,
        image_path=image_path,
        n_words=page.n_words,
        mean_conf=page.mean_conf,
        n_lines=len(group_lines(page.words)),
        text_chars=len(text),
        latency_ms=latency_ms,
        expected_fields=expected,
        matched_fields=matched,
        ocr_field_recall=ocr_field_recall,
        altered_token=altered_token,
        altered_token_present=altered_present,
        extracted_doctype=extraction.document_type,
        doctype_ok=doctype_ok,
        extracted_fields=extraction.fields,
        extraction_expected=extraction_expected,
        extraction_matched=extraction_matched,
        extraction_field_accuracy=extraction_accuracy,
        text_path=text_path,
    )


def _error_result(
    row: dict[str, str],
    *,
    image_path: str,
    expected: dict[str, str],
    altered_token: str | None,
    error: str,
) -> OcrSampleResult:
    return OcrSampleResult(
        sample_id=_text(row.get("id")),
        document_type=_text(row.get("tipo")),
        variant=_text(row.get("variante")),
        attack=_text(row.get("ataque")) or None,
        image_path=image_path,
        n_words=None,
        mean_conf=None,
        n_lines=None,
        text_chars=None,
        latency_ms=None,
        expected_fields=expected,
        matched_fields={},
        ocr_field_recall=None,
        altered_token=altered_token,
        altered_token_present=None,
        extracted_doctype=None,
        doctype_ok=False,
        extracted_fields={},
        extraction_expected={},
        extraction_matched={},
        extraction_field_accuracy=None,
        text_path=None,
        error=error,
    )


def _expected_values(
    row: dict[str, str],
    *,
    personas: dict[str, dict[str, str]],
    rows_by_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    document_type = _text(row.get("tipo"))
    persona = personas.get(_text(row.get("persona")))
    if persona is None:
        return {}
    expected: dict[str, str] = {}
    for field in OCR_EXPECTED_FIELDS.get(document_type, ()):
        value = _persona_value(persona, field, document_type)
        if value:
            expected[field] = value

    if _text(row.get("ataque")) == "T7":
        parent = rows_by_id.get(_text(row.get("documento_padre")))
        parent_persona = personas.get(_text(parent.get("persona"))) if parent else None
        if parent_persona is not None:
            expected["dni"] = _text(parent_persona.get("dni"))
            expected["fecha_nacimiento"] = _text(parent_persona.get("fecha_nacimiento"))
            expected["apellidos"] = _text(persona.get("apellidos"))
            expected["nombres"] = _text(persona.get("nombres"))
    if _text(row.get("ataque")) == "T8":
        expected.pop("periodo", None)
    return {field: value for field, value in expected.items() if value}


def _persona_value(
    persona: dict[str, str],
    field: str,
    document_type: str,
) -> str:
    if field == "fecha_nacimiento" and document_type == "partida_nacimiento":
        return _long_date(_text(persona.get("fecha_nacimiento")))
    return _text(persona.get(field))


def _extraction_expected(
    row: dict[str, str],
    expected: dict[str, str],
) -> dict[str, str]:
    mapping = EXTRACTION_FIELDS.get(_text(row.get("tipo")), {})
    return {
        field: extraction_field for field, extraction_field in mapping.items() if field in expected
    }


def _altered_token(row: dict[str, str]) -> str | None:
    if _text(row.get("variante")) != "tampered":
        return None
    value = _text(row.get("campo_alterado"))
    return value if value.isdigit() else None


def _summarize(
    results: list[OcrSampleResult],
    *,
    warmup_ms: float | None,
) -> dict[str, Any]:
    ok = [result for result in results if result.error is None]
    latencies = [result.latency_ms for result in ok if result.latency_ms is not None]
    confidences = [result.mean_conf for result in ok if result.mean_conf is not None]
    words = [result.n_words for result in ok if result.n_words is not None]
    return {
        "n_samples": len(results),
        "n_ok": len(ok),
        "n_errors": len(results) - len(ok),
        "warmup_ms": warmup_ms,
        "latency_ms": _distribution(latencies),
        "mean_confidence": _distribution(confidences),
        "words": _distribution([float(value) for value in words]),
        "ocr_field_recall": _ratio(
            sum(sum(result.matched_fields.values()) for result in ok),
            sum(len(result.matched_fields) for result in ok),
        ),
        "doctype_accuracy": _ratio(
            sum(result.doctype_ok for result in ok),
            len(ok),
        ),
        "extraction_field_accuracy": _ratio(
            sum(sum(result.extraction_matched.values()) for result in ok),
            sum(len(result.extraction_matched) for result in ok),
        ),
        "altered_token_hits": sum(bool(result.altered_token_present) for result in ok),
        "altered_token_total": sum(result.altered_token is not None for result in ok),
        "by_document_type": _group_summary(results, "document_type"),
        "by_attack": _group_summary(results, "attack"),
    }


def _group_summary(
    results: list[OcrSampleResult],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[OcrSampleResult]] = {}
    for result in results:
        key = getattr(result, attribute) or "authentic"
        grouped.setdefault(key, []).append(result)
    return {
        key: {
            "n": len(group),
            "n_ok": sum(result.error is None for result in group),
            "ocr_field_recall": _ratio(
                sum(sum(result.matched_fields.values()) for result in group),
                sum(len(result.matched_fields) for result in group),
            ),
            "doctype_accuracy": _ratio(
                sum(result.doctype_ok for result in group if result.error is None),
                sum(result.error is None for result in group),
            ),
            "extraction_field_accuracy": _ratio(
                sum(sum(result.extraction_matched.values()) for result in group),
                sum(len(result.extraction_matched) for result in group),
            ),
            "mean_conf": _ratio(
                sum(result.mean_conf or 0.0 for result in group if result.error is None),
                sum(result.mean_conf is not None for result in group),
            ),
        }
        for key, group in sorted(grouped.items())
    }


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(float(np.mean(values)), 2),
        "p50": round(float(np.percentile(values, 50)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _load_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el índice: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise ValueError("Faltan columnas en el índice: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError(f"El índice está vacío: {path}")
    return rows


def _load_personas(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de personas: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "persona_id" not in (reader.fieldnames or []):
            raise ValueError("El archivo de personas no tiene persona_id")
        return {_text(row.get("persona_id")): row for row in reader}


def _engine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": "doctr",
        "doctr_version": _package_version("python-doctr"),
        "torch_version": _package_version("torch"),
        "cuda_available": False,
        "device": "unknown",
    }
    try:
        import torch
    except ImportError:
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device"] = "cuda" if info["cuda_available"] else "cpu"
    return info


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _long_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError:
        return value
    return f"{parsed.day} de {MONTHS[parsed.month - 1]} de {parsed.year}"


def _value_present(expected: str, text_norm: str, field: str) -> bool:
    """Comprueba si un valor esperado aparece en el texto OCR normalizado."""
    expected_norm = _normalize(expected)
    if expected_norm in text_norm:
        return True
    if field in _TEXT_FIELDS:
        return _compact_text(expected_norm) in _compact_text(text_norm)
    return False


def _values_match(expected: str, extracted: str, field: str) -> bool:
    """Compara el valor extraído con el esperado bajo las mismas reglas."""
    expected_norm = _normalize(expected)
    extracted_norm = _normalize(extracted)
    if expected_norm == extracted_norm:
        return True
    if field in _TEXT_FIELDS:
        return _compact_text(expected_norm) == _compact_text(extracted_norm)
    return False


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s/.-]", " ", lowered)
    lowered = re.sub(r"\s*([-/])\s*", r"\1", lowered)
    return _normalize_periods(" ".join(lowered.split()))


def _normalize_periods(value: str) -> str:
    """Unifica ``2024-1``, ``2024-I``, ``2024-11`` y ``2024-II``."""

    def replace(match: re.Match[str]) -> str:
        year, suffix = match.group(1), match.group(2).lower()
        if suffix in {"1", "i"}:
            return f"{year}-i"
        if suffix in {"2", "11", "ii"}:
            return f"{year}-ii"
        return match.group(0)

    return _PERIOD_RE.sub(replace, value)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mide OCR real con docTR sobre un corpus local.")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--people", default=DEFAULT_PEOPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = run_ocr_samples(
        args.root,
        index_name=args.index,
        people_name=args.people,
        out_path=args.out,
        text_dir=args.text_dir,
        limit=args.limit,
        warmup=not args.no_warmup,
    )
    summary = payload["summary"]
    print(f"{summary['n_samples']} muestras · {summary['n_ok']} OK · {summary['n_errors']} errores")
    print(
        "confianza media="
        f"{_fmt(summary['mean_confidence']['mean'])} · "
        f"palabras medias={_fmt(summary['words']['mean'])} · "
        f"latencia p50={_fmt(summary['latency_ms']['p50'])} ms"
    )
    print(
        "recall campos OCR="
        f"{_fmt(summary['ocr_field_recall'])} · "
        f"tipo documental={_fmt(summary['doctype_accuracy'])} · "
        f"extracción campos={_fmt(summary['extraction_field_accuracy'])}"
    )
    print(f"reporte -> {args.out}")
    return 0 if summary["n_errors"] == 0 else 1


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
