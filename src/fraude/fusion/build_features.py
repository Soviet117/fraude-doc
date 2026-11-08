"""Construye la tabla de características de F7 a partir de las señales F3-F6."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from fraude.core.schemas import Observation
from fraude.fusion.features import observation_features
from fraude.nlp.duplicates import DuplicateIndex
from fraude.nlp.semantic import analyze_semantics
from fraude.synth.attacks import load_manifest
from fraude.synth.folds import load_folds

LOGGER = logging.getLogger("fraude.fusion.build_features")

DEFAULT_OBSERVATIONS = Path("artifacts/eval/v2_pipeline.jsonl")
DEFAULT_DATA = Path("data/synthetic/v2")
DEFAULT_F4_OOF = Path("artifacts/models/f4/oof.jsonl")
DEFAULT_F4_CHECKPOINT = Path("artifacts/models/f4/fold_2.pt")
DEFAULT_F5_OOF = Path("artifacts/models/f5/oof.jsonl")
DEFAULT_F5_CHECKPOINT = Path("artifacts/models/f5/fold_0.pt")
DEFAULT_OUT = Path("artifacts/models/f7/features.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _base_scores(path: Path, *, score_key: str = "score") -> dict[str, float]:
    if not Path(path).is_file():
        return {}
    return {
        row["sample_id"]: float(row[score_key])
        for row in _load_jsonl(path)
        if not row.get("augmentations")
    }


def _f5_summaries(path: Path) -> dict[str, dict[str, float]]:
    if not Path(path).is_file():
        return {}
    summaries: dict[str, dict[str, float]] = {}
    for row in _load_jsonl(path):
        if row.get("augmentations"):
            continue
        map_path = Path(path).parent / row["map_path"]
        area_fraction = 0.0
        if map_path.is_file():
            with Image.open(map_path) as image:
                import numpy as np

                probability = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
                area_fraction = float((probability >= 0.5).mean())
        summaries[row["sample_id"]] = {
            "max_prob": float(row["max_prob"]),
            "mean_prob": float(row["mean_prob"]),
            "area_fraction": round(area_fraction, 6),
        }
    return summaries


def _f4_infer(checkpoint: Path, image_path: Path) -> float | None:
    if not Path(checkpoint).is_file():
        return None
    from fraude.models.infer import load_checkpoint, predict_image

    model, config = load_checkpoint(checkpoint)
    with Image.open(image_path) as image:
        return float(
            predict_image(
                model, image.convert("RGB"), image_size=int(config.get("image_size", 224))
            )
        )


def _f5_infer(checkpoint: Path, image_path: Path) -> dict[str, float] | None:
    if not Path(checkpoint).is_file():
        return None
    from fraude.models.seg_infer import load_checkpoint, localization_summary, predict_mask

    model, config = load_checkpoint(checkpoint)
    with Image.open(image_path) as image:
        probability = predict_mask(
            model,
            image.convert("RGB"),
            image_size=int(config.get("image_size", 256)),
        )
    return localization_summary(probability)


def build_features(
    observations_path: Path = DEFAULT_OBSERVATIONS,
    *,
    data_dir: Path = DEFAULT_DATA,
    f4_oof: Path = DEFAULT_F4_OOF,
    f4_checkpoint: Path = DEFAULT_F4_CHECKPOINT,
    f5_oof: Path = DEFAULT_F5_OOF,
    f5_checkpoint: Path = DEFAULT_F5_CHECKPOINT,
    out_path: Path = DEFAULT_OUT,
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    rows = _load_jsonl(observations_path)
    folds = load_folds(data_dir / "folds_manifest.jsonl")
    authentic = [
        record for record in load_manifest(data_dir / "manifest.jsonl") if record.is_authentic
    ]
    duplicate_index = DuplicateIndex.from_records(authentic, data_dir)
    f4_scores = _base_scores(f4_oof)
    f5_summaries = _f5_summaries(f5_oof)

    output: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        observation = Observation.model_validate(row["observation"])
        image_path = data_dir / row["image_path"]
        with Image.open(image_path) as image:
            semantic = analyze_semantics(
                observation,
                image=image.convert("RGB"),
                duplicate_index=duplicate_index,
                exclude_sample_id=sample_id,
            )
        f4_score = f4_scores.get(sample_id)
        if f4_score is None:
            f4_score = _f4_infer(f4_checkpoint, image_path)
        f5_summary = f5_summaries.get(sample_id)
        if f5_summary is None:
            f5_summary = _f5_infer(f5_checkpoint, image_path)
        features = observation_features(
            observation,
            f4_score=f4_score,
            f5_summary=f5_summary,
            semantic=semantic,
        )
        output.append(
            {
                "sample_id": sample_id,
                "source_id": row["source_id"],
                "fold": int(folds[sample_id]),
                "label": 0 if row["is_authentic"] else 1,
                "attack": row.get("primary_attack"),
                "document_type": row["document_type"],
                "f4_score": f4_score,
                "f5_summary": f5_summary,
                "semantic": semantic.to_dict(),
                "features": features,
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Características escritas en %s (%d filas)", out_path, len(output))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construye características de F7.")
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--f4-oof", type=Path, default=DEFAULT_F4_OOF)
    parser.add_argument("--f4-checkpoint", type=Path, default=DEFAULT_F4_CHECKPOINT)
    parser.add_argument("--f5-oof", type=Path, default=DEFAULT_F5_OOF)
    parser.add_argument("--f5-checkpoint", type=Path, default=DEFAULT_F5_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rows = build_features(
        args.observations,
        data_dir=args.data,
        f4_oof=args.f4_oof,
        f4_checkpoint=args.f4_checkpoint,
        f5_oof=args.f5_oof,
        f5_checkpoint=args.f5_checkpoint,
        out_path=args.out,
    )
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
