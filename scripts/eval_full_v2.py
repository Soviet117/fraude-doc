"""Sanity check del pipeline integral sobre los 50 documentos base (v2).

No es una validación: F7 se entrenó con estas filas y los checkpoints de F4/F5
son los de demo. Sirve para verificar el cableado y detectar errores de
integración antes de la sustentación.
"""

from __future__ import annotations

import json
from pathlib import Path

from fraude.pipeline.full import analyze_full
from fraude.registry.build import build_registry_from_manifest
from fraude.synth.attacks import load_manifest

DATA = Path("data/synthetic/v2")
OUT = Path("artifacts/eval/full_v2_serving.json")


def main() -> int:
    records = []
    for name in ("manifest.jsonl", "attacks_manifest.jsonl"):
        records += load_manifest(DATA / name)
    registry = build_registry_from_manifest(DATA / "manifest.jsonl")

    rows = []
    for index, record in enumerate(sorted(records, key=lambda item: item.sample_id), 1):
        analysis = analyze_full(
            DATA / record.image_path,
            registry=registry,
            exclude_sample_id=record.sample_id,
            artifacts_dir=None,
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "label": 0 if record.is_authentic else 1,
                "attack": record.primary_attack,
                "document_type": record.document_type,
                "p_fake": analysis.p_fake,
                "risk": analysis.risk.value,
                "doctype": analysis.observation.doctype.label,
                "quality": analysis.observation.quality.ok,
                "f4": analysis.f4_score,
                "f5": analysis.f5_summary,
                "timings": analysis.timings_ms,
                "evidences": [evidence.code for evidence in analysis.observation.fusion.evidences],
            }
        )
        print(index, record.sample_id, analysis.risk.value, analysis.p_fake, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Guardado {OUT} ({len(rows)} documentos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
