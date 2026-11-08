"""Evaluación de la rama semántica F6 sobre el corpus v2 base."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from fraude.core.schemas import Observation
from fraude.nlp.duplicates import DuplicateIndex
from fraude.nlp.explain import explain_observation
from fraude.nlp.normalize import field_matches
from fraude.nlp.semantic import analyze_semantics
from fraude.synth.attacks import load_manifest

DEFAULT_DATA = Path("data/synthetic/v2")
DEFAULT_OBSERVATIONS = Path("artifacts/eval/v2_pipeline.jsonl")
DEFAULT_OUT = Path("artifacts/eval/f6_semantic.json")
DEFAULT_MARKDOWN = Path("artifacts/eval/f6_semantic.md")


def _field_kind(name: str) -> str:
    if name == "dni" or name.startswith("dni_"):
        return "dni"
    if name.startswith("fecha_"):
        return "date"
    return "text"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_semantics(
    rows: list[dict[str, Any]],
    *,
    data_dir: Path = DEFAULT_DATA,
    use_llm: bool = False,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    authentic = [
        record
        for name in ("manifest.jsonl",)
        for record in load_manifest(data_dir / name)
        if record.is_authentic
    ]
    index = DuplicateIndex.from_records(authentic, data_dir)

    field_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "match": 0})
    doctype_ok = 0
    duplicate_tp = duplicate_fp = duplicate_fn = duplicate_tn = 0
    flags_by_attack: dict[str, Counter[str]] = defaultdict(Counter)
    explanation_ok = 0
    explanation_cites = 0

    for row in rows:
        observation = Observation.model_validate(row["observation"])
        image_path = data_dir / row["image_path"]
        with Image.open(image_path) as image:
            semantic = analyze_semantics(
                observation,
                image=image.convert("RGB"),
                duplicate_index=index,
                exclude_sample_id=row["sample_id"],
            )
        if observation.doctype.label == row["document_type"]:
            doctype_ok += 1
        for name, gold in (row.get("fields_gold") or {}).items():
            if name not in observation.fields:
                field_stats[name]["n"] += 1
                continue
            field_stats[name]["n"] += 1
            if field_matches(gold, observation.fields[name], kind=_field_kind(name)):
                field_stats[name]["match"] += 1

        attack = row.get("primary_attack") or "authentic"
        for flag in semantic.flags:
            flags_by_attack[attack][flag] += 1
        if attack == "T7":
            if semantic.duplicate is not None:
                duplicate_tp += 1
            else:
                duplicate_fn += 1
        elif attack == "authentic":
            if semantic.duplicate is not None:
                duplicate_fp += 1
            else:
                duplicate_tn += 1

        explanation = explain_observation(observation, semantic=semantic, use_llm=use_llm)
        if explanation.text.strip():
            explanation_ok += 1
        if "revisión humana" in explanation.text.lower() or "revisión" in explanation.text.lower():
            explanation_cites += 1

    field_accuracy = {
        name: {
            "n": values["n"],
            "match": values["match"],
            "accuracy": round(values["match"] / values["n"], 4) if values["n"] else None,
        }
        for name, values in sorted(field_stats.items())
    }
    return {
        "schema_version": "1.0",
        "generator": "fraude.nlp.evaluate",
        "n_documents": len(rows),
        "doctype_accuracy": round(doctype_ok / len(rows), 4) if rows else None,
        "field_accuracy": field_accuracy,
        "duplicate": {
            "t7_recall": round(duplicate_tp / (duplicate_tp + duplicate_fn), 4)
            if duplicate_tp + duplicate_fn
            else None,
            "authentic_fpr": round(duplicate_fp / (duplicate_fp + duplicate_tn), 4)
            if duplicate_fp + duplicate_tn
            else None,
            "tp": duplicate_tp,
            "fp": duplicate_fp,
            "fn": duplicate_fn,
            "tn": duplicate_tn,
        },
        "flags_by_attack": {
            key: dict(sorted(value.items())) for key, value in sorted(flags_by_attack.items())
        },
        "explanations": {
            "n": len(rows),
            "non_empty": explanation_ok,
            "mentions_review": explanation_cites,
            "source": "ollama" if use_llm else "template",
        },
        "limitations": [
            "El corpus es sintético y T7 es una copia exacta de la imagen auténtica; "
            "la detección de duplicados en documentos reales no está validada.",
            "La exactitud por campo se mide contra los campos del manifiesto, no "
            "contra anotación humana.",
            "El LLM está apagado por defecto; la evaluación usa la plantilla.",
            "No hay evaluación de explicaciones por evaluadores humanos.",
        ],
    }


def write_markdown(result: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F6 — rama semántica",
        "",
        f"- Documentos base: {result['n_documents']}",
        f"- Exactitud de tipo documental: {result['doctype_accuracy']:.4f}",
        "",
        "## Exactitud de extracción por campo",
        "",
        "| Campo | n | Aciertos | Exactitud |",
        "|---|---:|---:|---:|",
    ]
    for name, values in result["field_accuracy"].items():
        accuracy = "—" if values["accuracy"] is None else f"{values['accuracy']:.4f}"
        lines.append(f"| {name} | {values['n']} | {values['match']} | {accuracy} |")
    duplicate = result["duplicate"]
    lines.extend(
        [
            "",
            "## Duplicados (T7)",
            "",
            "| Métrica | Valor |",
            "|---|---:|",
            f"| Recall T7 | {_fmt(duplicate['t7_recall'])} |",
            f"| FPR auténticos | {_fmt(duplicate['authentic_fpr'])} |",
            f"| TP / FP / FN / TN | {duplicate['tp']} / {duplicate['fp']} / "
            f"{duplicate['fn']} / {duplicate['tn']} |",
            "",
            "## Banderas semánticas por ataque",
            "",
            "| Ataque | Banderas |",
            "|---|---|",
        ]
    )
    for attack, flags in result["flags_by_attack"].items():
        detail = ", ".join(f"{name} ({count})" for name, count in flags.items()) or "—"
        lines.append(f"| {attack} | {detail} |")
    explanations = result["explanations"]
    lines.extend(
        [
            "",
            "## Explicaciones",
            "",
            f"- No vacías: {explanations['non_empty']}/{explanations['n']}",
            f"- Mencionan revisión humana: {explanations['mentions_review']}/{explanations['n']}",
            f"- Fuente: {explanations['source']}",
            "",
            "> El corpus es sintético. T7 es una copia exacta, por lo que el",
            "> duplicado se detecta trivialmente; no es una validación real.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa F6 sobre el corpus v2 base.")
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--use-llm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.observations)
    result = evaluate_semantics(rows, data_dir=args.data, use_llm=args.use_llm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.markdown)
    print(
        json.dumps(
            {"doctype_accuracy": result["doctype_accuracy"], "duplicate": result["duplicate"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
