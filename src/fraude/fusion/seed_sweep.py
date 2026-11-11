"""Barrido de semillas de F7 (avance de F8).

Repite el entrenamiento/evaluación OOF con varias semillas y reporta media ±
desviación. No sustituye la evaluación integral de F8 (robustez, kappa,
McNemar), pero evita citar una sola corrida como si fuera estable.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from fraude.fusion.train import FusionConfig, train_fusion

LOGGER = logging.getLogger("fraude.fusion.seed_sweep")

DEFAULT_SEEDS = (42, 43, 44)
METRICS = ("f1", "precision", "recall", "fpr", "pr_auc", "roc_auc", "accuracy")


def sweep(
    base: FusionConfig,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Entrena F7 para cada semilla y agrega las métricas OOF."""
    out_dir = Path(out_dir or base.out_dir)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        config = replace(base, seed=seed, out_dir=out_dir / f"seed_{seed}")
        LOGGER.info("Semilla %d", seed)
        result = train_fusion(config)
        runs.append(
            {
                "seed": seed,
                "pooled": result["pooled"],
                "calibrated_pooled": result["calibrated_pooled"],
                "calibration": result["calibration"],
                "calibration_method": result["calibration_method"],
            }
        )
    summary: dict[str, Any] = {}
    for name in METRICS:
        values = [float(run["pooled"].get(name, 0.0)) for run in runs]
        summary[name] = {
            "mean": round(mean(values), 4),
            "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
            "values": [round(value, 4) for value in values],
        }
    payload = {
        "generator": "fraude.fusion.seed_sweep",
        "config": {
            **asdict(base),
            "features_path": str(base.features_path),
            "out_dir": str(base.out_dir),
        },
        "seeds": list(seeds),
        "runs": runs,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seed_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "seed_sweep.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# F7 — barrido de semillas",
        "",
        f"Semillas: {', '.join(str(seed) for seed in payload['seeds'])}",
        "",
        "| Métrica | Media | Desv. | Valores |",
        "|---|---:|---:|---|",
    ]
    for name, data in payload["summary"].items():
        lines.append(
            f"| {name} | {data['mean']:.4f} | {data['std']:.4f} | "
            f"{', '.join(f'{value:.4f}' for value in data['values'])} |"
        )
    lines.extend(
        [
            "",
            "> Barrido sobre el mismo corpus y las mismas particiones; la",
            "> variación refleja solo la semilla del modelo, no robustez ante",
            "> degradaciones ni documentos reales. F8 sigue incompleta.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Barrido de semillas de F7.")
    parser.add_argument("--features", type=Path, default=Path("artifacts/models/f7/features.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/f7/seeds"))
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n-boot", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    base = FusionConfig(
        features_path=args.features,
        out_dir=args.out,
        n_boot=args.n_boot,
    )
    payload = sweep(base, seeds=seeds, out_dir=args.out)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
